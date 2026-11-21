"""Offline preparation uses inert text and never touches a live review store."""
import csv
import json

import pytest

from secpo.crsecpo.preparation import (
    audit_report, export_audit, export_relations, export_pair_audit, export_evaluation_audit,
    ingest_audit, ingest_relations, main,
)
from secpo.crsecpo.records import prepare_records, read_review_export
from secpo.evaluation.sep_judge import atomic_write_jsonl, load_jsonl


def candidate(key, *, origin="harmful", status="accepted"):
    return {"vanilla": "Museum source " + key, "adversarial": "Museum target " + key,
            "completion": "PUBLISHER_COMPLETION_MUST_STAY_HIDDEN",
            "data_type": "adversarial_" + origin, "crsecpo_record_id": key,
            "crsecpo_consensus": status,
            "crsecpo_target_policy": "limited" if origin == "harmful" else "permissible",
            "crsecpo_selection_group": ("harmful_benign_framing" if origin == "harmful"
                else "benign_adversarial_control") if status == "accepted" else "out_of_scope",
            "crsecpo_has_rewrite": "1", "crsecpo_revision": "a" * 40,
            "crsecpo_row_index": key, "crsecpo_partition": "train"}


def write_export(path, rows, *, partition="train"):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows({**row, "crsecpo_partition": partition} for row in rows)
    return path


@pytest.fixture
def exports(tmp_path):
    training = write_export(tmp_path / "train.tsv", [candidate("train-h"), candidate("train-b", origin="benign")])
    evaluation = write_export(tmp_path / "eval.tsv", [candidate("eval-h"), candidate("eval-b", origin="benign"),
        candidate("eval-r", origin="benign", status="rejected"), candidate("eval-u", status="unsure")], partition="eval")
    rubric = tmp_path / "rubric.txt"
    rubric.write_text("Frozen inert museum review rubric.")
    return training, evaluation, rubric


def audit_packet(tmp_path, exports):
    training, evaluation, rubric = exports
    directory = tmp_path / "packet"
    manifest = export_audit(directory, rubric=rubric, training_export=training, evaluation_export=evaluation,
                             per_stratum=1, seed=7)
    return directory, manifest


def fill_audit(packet, destination, human="human_1", *, reject=False):
    rows = load_jsonl(packet / (human + ".jsonl"))
    for row in rows:
        row.update(decision="reject" if reject else "accept", target_policy="permissible",
                   selection_group="out_of_scope" if reject else "benign_adversarial_control",
                   reason="Independent fixture assessment.")
    atomic_write_jsonl(rows, destination)
    return destination


def test_audit_strata_are_reproducible_and_forms_hide_machine_data(tmp_path, exports):
    training, evaluation, rubric = exports
    rows = [candidate(f"{origin}-{status}-{index}", origin=origin, status=status)
            for origin in ("harmful", "benign") for status in ("accepted", "rejected", "unsure", "unreviewed")
            for index in range(3)]
    write_export(training, rows)
    before = training.read_bytes()
    first = tmp_path / "first"
    manifest = export_audit(first, rubric=rubric, training_export=training, per_stratum=2, seed=7)
    write_export(tmp_path / "reversed.tsv", list(reversed(rows)))
    second = tmp_path / "second"
    export_audit(second, rubric=rubric, training_export=tmp_path / "reversed.tsv", per_stratum=2, seed=7)
    assert manifest["sampled"] == 16
    assert len(manifest["strata"]) == 8
    assert all(row["sampled"] == 2 and row["available"] == 3 for row in manifest["strata"])
    assert (first / "human_1.jsonl").read_bytes() == (second / "human_1.jsonl").read_bytes()
    for human in ("human_1", "human_2"):
        forms = load_jsonl(first / (human + ".jsonl"))
        assert all(set(row) == {"review_id", "source", "target", "has_rewrite", "decision", "target_policy",
                                "selection_group", "reason"} for row in forms)
        assert all(row["decision"] == row["target_policy"] == row["selection_group"] == "" for row in forms)
        assert "PUBLISHER_COMPLETION" not in (first / (human + ".jsonl")).read_text()
    assert {row["review_id"] for row in load_jsonl(first / "human_1.jsonl")} == {
        row["review_id"] for row in load_jsonl(first / "human_2.jsonl")}
    assert training.read_bytes() == before
    with pytest.raises(FileExistsError):
        export_audit(first, rubric=rubric, training_export=training)


def test_two_human_ingest_and_report_preserve_all_machine_records(tmp_path, exports):
    packet, manifest = audit_packet(tmp_path, exports)
    original = {path: path.read_bytes() for path in exports}
    first = fill_audit(packet, tmp_path / "one.jsonl")
    second = fill_audit(packet, tmp_path / "two.jsonl", "human_2")
    result = ingest_audit(packet, first, reviewer="human_1", person_id="person-a", independent=True)
    assert ingest_audit(packet, first, reviewer="human_1", person_id="person-a", independent=True) == result
    ingest_audit(packet, second, reviewer="human_2", person_id="person-b", independent=True)
    report = audit_report(packet, tmp_path / "report")
    assert report["sampled"] == manifest["sampled"] == 6
    assert report["agreement"]["decision"]["rate"] == 1
    # Both people flag the rejected fixture as eligible; no machine vote is changed.
    assert report["agreement"]["decision"]["possible_machine_errors"] == 1
    assert report["agreement"]["decision"]["machine_comparable"] == 5
    assert len(load_jsonl(tmp_path / "report" / "findings.jsonl")) == 6
    assert all(path.read_bytes() == data for path, data in original.items())


def test_audit_requires_independent_distinct_people_and_immutable_ingestion(tmp_path, exports):
    packet, _ = audit_packet(tmp_path, exports)
    first = fill_audit(packet, tmp_path / "one.jsonl")
    with pytest.raises(ValueError, match="attestation"):
        ingest_audit(packet, first, reviewer="human_1", person_id="person-a")
    ingest_audit(packet, first, reviewer="human_1", person_id="person-a", independent=True)
    second = fill_audit(packet, tmp_path / "two.jsonl", "human_2")
    with pytest.raises(ValueError, match="distinct"):
        ingest_audit(packet, second, reviewer="human_2", person_id="person-a", independent=True)
    fill_audit(packet, first, reject=True)
    with pytest.raises(ValueError, match="immutable"):
        ingest_audit(packet, first, reviewer="human_1", person_id="person-a", independent=True)


@pytest.mark.parametrize("change,match", [
    (lambda rows: rows.pop(), "complete frozen sample"),
    (lambda rows: rows.append(rows[0]), "duplicate"),
    (lambda rows: rows[0].update(source="Changed source"), "content"),
    (lambda rows: rows[0].update(extra="unexpected"), "fields"),
    (lambda rows: rows[0].update(decision="reject"), "selection group"),
    (lambda rows: rows[0].update(target_policy="limited"), "inconsistent"),
    (lambda rows: rows[0].update(decision="reject", target_policy="ambiguous",
                                selection_group="out_of_scope"), "requires unsure"),
    (lambda rows: rows[0].update(reason=""), "reason"),
])
def test_invalid_human_forms_do_not_publish_an_ingestion(tmp_path, exports, change, match):
    packet, _ = audit_packet(tmp_path, exports)
    response = fill_audit(packet, tmp_path / "filled.jsonl")
    rows = load_jsonl(response)
    change(rows)
    atomic_write_jsonl(rows, response)
    with pytest.raises(ValueError, match=match):
        ingest_audit(packet, response, reviewer="human_1", person_id="a", independent=True)
    assert not (packet / "ingestions").exists()


def test_changed_original_packet_is_rejected(tmp_path, exports):
    packet, _ = audit_packet(tmp_path, exports)
    response = fill_audit(packet, tmp_path / "filled.jsonl")
    with (packet / "human_1.jsonl").open("a") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="packet changed"):
        ingest_audit(packet, response, reviewer="human_1", person_id="a", independent=True)


@pytest.mark.parametrize("rewrite", ["", "Museum source no-rewrite", " \tMuseum source no-rewrite\n"])
def test_ambiguous_structurally_absent_rewrite_can_be_rejected(tmp_path, exports, rewrite):
    training, _, rubric = exports
    value = candidate("no-rewrite", status="rejected")
    value.update(adversarial=rewrite, crsecpo_has_rewrite="1" if rewrite else "0")
    write_export(training, [value])
    packet = tmp_path / "packet"
    export_audit(packet, training_export=training, rubric=rubric)
    acceptance = fill_audit(packet, tmp_path / "accept.jsonl")
    with pytest.raises(ValueError, match="inconsistent policy, group, or rewrite"):
        ingest_audit(packet, acceptance, reviewer="human_1", person_id="a", independent=True)
    response = fill_audit(packet, tmp_path / "filled.jsonl", reject=True)
    rows = load_jsonl(response)
    rows[0]["target_policy"] = "ambiguous"
    atomic_write_jsonl(rows, response)
    assert ingest_audit(packet, response, reviewer="human_1", person_id="a", independent=True)["reviewed"] == 1


@pytest.mark.parametrize("raw_rewrite", ["", " \t "])
def test_source_only_audit_matches_filter_target_and_preserves_raw_rewrite(tmp_path, exports, raw_rewrite):
    training, _, rubric = exports
    value = candidate("source-only", origin="benign", status="rejected")
    value.update(vanilla="How do I get to the museum?", adversarial=raw_rewrite,
                 data_type="vanilla_benign", crsecpo_has_rewrite="0")
    write_export(training, [value])
    original = training.read_bytes()
    packet = tmp_path / "packet"
    export_audit(packet, training_export=training, rubric=rubric)
    key = load_jsonl(packet / "coordinator.jsonl")[0]
    assert key["source"] == value["vanilla"] and key["target"] == raw_rewrite
    assert key["has_rewrite"] is False
    for human, person in (("human_1", "person-a"), ("human_2", "person-b")):
        form = load_jsonl(packet / (human + ".jsonl"))[0]
        assert form["source"] == form["target"] == value["vanilla"]
        assert form["has_rewrite"] is False
        response = fill_audit(packet, tmp_path / (human + ".jsonl"), human)
        with pytest.raises(ValueError, match="inconsistent policy, group, or rewrite"):
            ingest_audit(packet, response, reviewer=human, person_id=person, independent=True)
        fill_audit(packet, response, human, reject=True)
        ingest_audit(packet, response, reviewer=human, person_id=person, independent=True)
    report = audit_report(packet, tmp_path / "report")
    assert report["agreement"]["target_policy"]["machine_comparable"] == 1
    assert report["agreement"]["target_policy"].get("possible_machine_errors", 0) == 0
    assert report["sample_retained_by_group"] == {"excluded": 1}
    assert training.read_bytes() == original


def test_relation_decisions_preserved_and_links_reuse_existing_splitter(tmp_path, exports):
    training, evaluation, _ = exports
    candidates = tmp_path / "candidates.jsonl"
    atomic_write_jsonl([
        dict(kind="family", left="train-h", right="eval-h"),
        dict(kind="control", left="eval-h", right="eval-b"),
        dict(kind="family", left="train-b", right="eval-r"),
        dict(kind="family", left="train-b", right="eval-u"),
    ], candidates)
    packet = tmp_path / "relations"
    export_relations(packet, candidates=candidates, training_export=training, evaluation_export=evaluation)
    forms = load_jsonl(packet / "relations.jsonl")
    for row, decision in zip(forms, ("link", "link", "no_link", "unsure")):
        row.update(decision=decision, reason="Inert relationship assessment.")
    responses = tmp_path / "responses.jsonl"
    atomic_write_jsonl(forms, responses)
    out = tmp_path / "links"
    report = ingest_relations(packet, responses, out, person_id="person-a")
    assert report["counts"] == {"link": 2, "no_link": 1, "unsure": 1}
    assert report["unresolved"] == 1
    assert len(load_jsonl(out / "decisions.jsonl")) == 4
    rows = [*read_review_export(training, pool="training"), *read_review_export(evaluation, pool="evaluation")]
    outputs, assignments = prepare_records(rows, links=load_jsonl(out / "family_links.jsonl"),
                                           matched_controls=load_jsonl(out / "matched_controls.jsonl"))
    assert any(row["id"] == "train-h" and row["exclusion_reason"] == "overlaps_held_out_family"
               for row in outputs["quarantine"])
    assert next(row for row in outputs["test"] if row["id"] == "eval-b")["control_group"] == "matched"
    assert len(assignments) == len(rows)
    with pytest.raises(FileExistsError):
        ingest_relations(packet, responses, out, person_id="person-a")


def test_no_approved_controls_pass_empty_links_without_a_matched_endpoint(tmp_path, exports):
    from secpo.crsecpo.evaluation import decision_rule
    from secpo.crsecpo.records import main as prepare_main

    training, evaluation, _ = exports
    candidates = tmp_path / "candidates.jsonl"
    atomic_write_jsonl([dict(kind="control", left="eval-h", right="eval-b")], candidates)
    packet = tmp_path / "relations"
    export_relations(packet, candidates=candidates, training_export=training, evaluation_export=evaluation)
    forms = load_jsonl(packet / "relations.jsonl")
    forms[0].update(decision="no_link", reason="Fixture topics and framing styles do not match.")
    responses = tmp_path / "responses.jsonl"
    atomic_write_jsonl(forms, responses)
    links = tmp_path / "links"
    report = ingest_relations(packet, responses, links, person_id="person-a")
    assert report["positive_links"] == {"family": 0, "control": 0}
    assert (links / "matched_controls.jsonl").read_bytes() == b""

    selected = tmp_path / "selected"
    prepare_main(["prepare", "--training-export", str(training), "--evaluation-export", str(evaluation),
                  "--family-links", str(links / "family_links.jsonl"), "--near-duplicate-review-complete",
                  "--matched-controls", str(links / "matched_controls.jsonl"), "--output-dir", str(selected)])
    test_rows = load_jsonl(selected / "test.jsonl")
    assert all(row["control_group"] == "other" and "matched_harmful_ids" not in row for row in test_rows)
    # Inert Boolean fixtures exercise endpoint availability without generating or grading answers.
    results = [{**row, "item_id": row["id"], "checkpoint": checkpoint, "kind": "framed",
                "grade": {"prohibited_assistance": False, "refusal": False, "permitted_help": True}}
               for checkpoint in ("base", "crsecpo") for row in test_rows]
    decision = decision_rule(results, {})
    endpoint = decision["contrasts"]["matched_benign_refusal_increase"]
    assert endpoint["items"] == 0 and endpoint["status"] == "missing_or_unknown"
    assert endpoint["upper"] is None and endpoint["passed"] is False
    assert decision["joint_claim"] is False


@pytest.mark.parametrize("candidates,match", [
    ([dict(kind="family", left="absent", right="eval-h")], "known"),
    ([dict(kind="control", left="train-h", right="eval-b")], "held-out"),
    ([dict(kind="control", left="eval-h", right="eval-r")], "held-out"),
    ([dict(kind="family", left="train-h", right="eval-h"),
      dict(kind="family", left="eval-h", right="train-h")], "duplicate"),
])
def test_invalid_relation_candidates_never_publish_packet(tmp_path, exports, candidates, match):
    training, evaluation, _ = exports
    path = tmp_path / "candidates.jsonl"
    atomic_write_jsonl(candidates, path)
    with pytest.raises(ValueError, match=match):
        export_relations(tmp_path / "packet", candidates=path, training_export=training, evaluation_export=evaluation)
    assert not (tmp_path / "packet").exists()


def test_audit_export_cli_is_offline_and_reports_metadata_only(tmp_path, exports, capsys):
    training, _, rubric = exports
    main(["audit-export", "--training-export", str(training), "--rubric", str(rubric),
          "--output-dir", str(tmp_path / "packet"), "--per-stratum", "1"])
    printed = capsys.readouterr().out
    assert json.loads(printed)["sampled"] == 2
    assert "Museum" not in printed and "PUBLISHER_COMPLETION" not in printed


def saved_pair_outcome(record_id, votes=("combined", "combined"), *, truncated=False, deployment_system=""):
    from test_crsecpo_pairs import ScriptedBackend, candidate as pair_candidate, collect
    _, outcomes = collect(ScriptedBackend(votes, truncated=truncated), rows=[pair_candidate(
        id=record_id, source="SOURCE_MUST_BE_HIDDEN", target="Museum question")],
        deployment_system=deployment_system)
    return outcomes[0]


def test_pair_answer_packet_blinds_origins_and_preserves_each_outcome_stratum(tmp_path, exports):
    _, _, rubric = exports
    outcomes = []
    for category, votes in (("accepted", ["combined"] * 2), ("ordinary_wins", ["ordinary"] * 2),
                            ("tie", ["tie"] * 2), ("order_disagreement", ["combined", "ordinary"])):
        for index in range(4):
            outcomes.append(saved_pair_outcome(f"{category}-{index}", votes))
    outcomes.append(saved_pair_outcome("incomplete", truncated=True))
    source = tmp_path / "outcomes.jsonl"
    atomic_write_jsonl(outcomes, source)
    before = source.read_bytes()
    packet = tmp_path / "pair-audit"
    manifest = export_pair_audit(packet, outcomes=source, rubric=rubric, per_stratum=2, seed=5)
    forms, key = load_jsonl(packet / "human.jsonl"), load_jsonl(packet / "coordinator.jsonl")
    assert manifest["sampled"] == 8
    assert manifest["excluded_incomplete_answer_ids"] == ["incomplete"]
    assert {row["outcome"] for row in key} == {"accepted", "ordinary_wins", "tie", "order_disagreement"}
    assert all(row["available"] == 4 and row["sampled"] == 2 for row in manifest["strata"])
    assert all(set(row) == {"review_id", "target", "A", "B", "verdict", "A_acceptable", "B_acceptable", "reason"}
               for row in forms)
    text = (packet / "human.jsonl").read_text()
    assert all(hidden not in text for hidden in ("SOURCE_MUST_BE_HIDDEN", "MODEL_MUST_BE_HIDDEN", "JUDGMENT_MUST_BE_HIDDEN"))
    assert {row["answer_origins"]["A"] for row in key} == {"ordinary", "combined"}
    originals = {row["id"]: row for row in outcomes}
    for form, origin in zip(forms, key):
        assert form["review_id"] == origin["review_id"]
        original = originals[origin["record_id"]]
        response = original["generations"][origin["answer_origins"]["A"]]["response"]
        assert form["A"] == response["text"].removesuffix(original["completion_terminator"])
    assert source.read_bytes() == before
    reversed_source = tmp_path / "reversed.jsonl"
    atomic_write_jsonl(reversed(outcomes), reversed_source)
    export_pair_audit(tmp_path / "repeat", outcomes=reversed_source, rubric=rubric, per_stratum=2, seed=5)
    assert (tmp_path / "repeat" / "human.jsonl").read_bytes() == (packet / "human.jsonl").read_bytes()
    assert "do not change model judgments or grant grading_audit_passed" in (packet / "instructions.json").read_text()


@pytest.mark.parametrize("legacy", [False, True])
def test_pair_audit_excludes_responses_with_wrong_terminator(tmp_path, exports, legacy):
    from test_crsecpo_pairs import candidate, collect

    _, _, rubric = exports

    def wrong_terminator(request):
        return {"text": "Unusable synthetic answer<eos>", "answer_text": "Unusable synthetic answer",
                "finish_reason": "stop", "completion_terminator": "<eos>"}

    _, bad_outcomes = collect(wrong_terminator, rows=[candidate(id="wrong-eos")], output_retries=2)
    invalid = bad_outcomes[0]
    assert invalid["status"] == "generation_error"
    if legacy:
        del invalid["completion_terminator"]
    else:
        for generation in invalid["generations"].values():
            del generation["error"]
    source = tmp_path / "outcomes.jsonl"
    atomic_write_jsonl([invalid, saved_pair_outcome("valid")], source)
    manifest = export_pair_audit(tmp_path / "packet", outcomes=source, rubric=rubric)
    assert manifest["sampled"] == 1
    assert manifest["excluded_incomplete_answer_ids"] == ["wrong-eos"]


@pytest.mark.parametrize("corruption", ["target", "source", "ordinary_hash", "combined_hash",
                                       "ordinary_target", "combined_target"])
def test_pair_audit_binds_target_and_source_to_real_collector_requests(tmp_path, exports, corruption):
    from secpo.crsecpo.prompts import digest
    _, _, rubric = exports
    row = saved_pair_outcome("collector-fixture")
    if corruption in {"target", "source"}:
        row[corruption] = "Different museum task"
    else:
        condition, field = corruption.split("_")
        generation = row["generations"][condition]
        if field == "hash":
            generation["request_id"] = "0" * 64
        else:
            request = generation["request"]
            if condition == "ordinary":
                request["messages"][-1]["content"] = "Different museum task"
            else:
                value = json.loads(request["messages"][-1]["content"])
                value["target"] = "Different museum task"
                request["messages"][-1]["content"] = json.dumps(value)
            generation["request_id"] = digest(request)
    source = tmp_path / "outcomes.jsonl"
    atomic_write_jsonl([row], source)
    with pytest.raises(ValueError, match="request"):
        export_pair_audit(tmp_path / "packet", outcomes=source, rubric=rubric)
    assert not (tmp_path / "packet").exists()


def test_pair_audit_preserves_valid_recorded_deployment_context(tmp_path, exports):
    _, _, rubric = exports
    source = tmp_path / "outcomes.jsonl"
    atomic_write_jsonl([saved_pair_outcome("collector-fixture", deployment_system="Fixture context")], source)
    assert export_pair_audit(tmp_path / "packet", outcomes=source, rubric=rubric)["sampled"] == 1


def saved_evaluation_files(tmp_path):
    from secpo.crsecpo.evaluation import _hash
    from test_crsecpo_evaluation import candidate as test_candidate
    candidates = [test_candidate(index, source=f"Museum direct {index}", family_id=f"family-{index}",
        **({"target_policy": "permissible", "selection_group": "benign_adversarial_control", "control_group": "other"}
           if index % 2 else {})) for index in range(40)]
    results = []
    for checkpoint in ("base", "sft", "dpo", "crsecpo"):
        for row in candidates:
            result = dict(job_id=f"{checkpoint}-{row['id']}", checkpoint=checkpoint,
                checkpoint_identity={"model": "MODEL_MUST_BE_HIDDEN"}, item_id=row["id"],
                family_id=row["family_id"], kind="framed", target_sha256=_hash(row["target"]),
                answer=f"Saved museum answer {row['id']}", finish_reason="stop",
                grade={"prohibited_assistance": False, "refusal": False, "permitted_help": True,
                       "rationale": "JUDGMENT_MUST_BE_HIDDEN"})
            result["result_sha256"] = _hash(result)
            results.append(result)
        row = candidates[0]
        direct = {**results[-1], "job_id": f"{checkpoint}-direct", "item_id": _hash(row["source"]),
                  "family_id": row["family_id"], "kind": "direct", "target_sha256": _hash(row["source"])}
        direct["result_sha256"] = _hash({key: value for key, value in direct.items() if key != "result_sha256"})
        results.append(direct)
    candidate_path, result_path = tmp_path / "test.jsonl", tmp_path / "results.jsonl"
    atomic_write_jsonl(candidates, candidate_path)
    atomic_write_jsonl(results, result_path)
    return candidate_path, result_path, results


def test_evaluation_answer_export_is_blinded_stratified_capped_and_exact(tmp_path, exports, capsys):
    _, _, rubric = exports
    candidates, results, rows = saved_evaluation_files(tmp_path)
    original = {path: path.read_bytes() for path in (candidates, results)}
    packet = tmp_path / "answer-audit"
    main(["answer-audit-export", "--results", str(results), "--candidates", str(candidates),
          "--rubric", str(rubric), "--per-stratum", "100", "--output-dir", str(packet)])
    printed = capsys.readouterr().out
    manifest = json.loads(printed)
    forms, key = load_jsonl(packet / "human.jsonl"), load_jsonl(packet / "coordinator.jsonl")
    assert manifest["sampled"] == len(forms) == len(key) == 60
    assert all(row["sampled"] > 0 for row in manifest["strata"])
    assert all(set(row) == {"review_id", "target", "answer", "prohibited_assistance", "refusal", "permitted_help", "reason"}
               for row in forms)
    text = (packet / "human.jsonl").read_text()
    assert all(hidden not in text for hidden in ("checkpoint", "MODEL_MUST_BE_HIDDEN", "JUDGMENT_MUST_BE_HIDDEN"))
    assert "Saved museum answer" not in printed
    original_rows = {row["job_id"]: row for row in rows}
    for form, origin in zip(forms, key):
        assert form["answer"] == original_rows[origin["job_id"]]["answer"]
        if origin["kind"] == "direct":
            assert form["target"] == "Museum direct 0"
    assert all(path.read_bytes() == before for path, before in original.items())
    atomic_write_jsonl(reversed(rows), tmp_path / "reversed-results.jsonl")
    export_evaluation_audit(tmp_path / "repeat", results=[tmp_path / "reversed-results.jsonl"],
                            candidates=candidates, rubric=rubric, per_stratum=100)
    assert (tmp_path / "repeat" / "human.jsonl").read_bytes() == (packet / "human.jsonl").read_bytes()


@pytest.mark.parametrize("corruption", ["answer", "target", "duplicate"])
def test_answer_audit_rejects_changed_or_misaligned_results_before_publication(tmp_path, exports, corruption):
    from secpo.crsecpo.evaluation import _hash
    _, _, rubric = exports
    candidates, results, rows = saved_evaluation_files(tmp_path)
    if corruption == "duplicate":
        rows.append(rows[0])
    elif corruption == "answer":
        rows[0]["answer"] = "Altered answer"
    else:
        rows[0]["target_sha256"] = _hash("Different target")
        rows[0]["result_sha256"] = _hash({key: value for key, value in rows[0].items() if key != "result_sha256"})
    atomic_write_jsonl(rows, results)
    with pytest.raises(ValueError):
        export_evaluation_audit(tmp_path / "packet", results=[results], candidates=candidates, rubric=rubric)
    assert not (tmp_path / "packet").exists()


def test_evaluation_audit_cannot_exceed_sixty_answers(tmp_path):
    with pytest.raises(ValueError, match="between 1 and 60"):
        export_evaluation_audit(tmp_path / "packet", results=[], candidates="absent", rubric="absent", limit=61)
