"""Offline comparison invariants; no hub calls or model downloads."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from secpo.constants import IGNORE_INDEX
from secpo.crsecpo.prompts import PROMPT_VERSION, answer_messages
from secpo.crsecpo.training import ComparisonCollator, METHODS, prepare
from secpo.data import PreferenceSchemaError, validate_preference_record
from secpo.loss import secpo_loss, sequence_logps


class Tokenizer:
    pad_token_id = 0
    eos_token_id = 1
    unk_token_id = -1

    def convert_tokens_to_ids(self, token):
        return 2 if token == "<|eot_id|>" else -1

    def decode(self, ids, **kwargs):
        return "<|eot_id|>" if ids == [2] else "<eos>"

    def __call__(self, text, add_special_tokens=False):
        chunks = text.split("<|eot_id|>")
        ids = []
        for i, chunk in enumerate(chunks):
            if i:
                ids.append(2)
            ids.extend(3 + ord(c) % 61 for c in chunk)
        return {"input_ids": ids}

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        text = "".join(f"<{m['role']}>{m['content']}" for m in messages) + "<assistant>"
        return self(text)["input_ids"] if tokenize else text


def row(**updates):
    from secpo.crsecpo.inference import ModelSpec
    from secpo.crsecpo.pairs import generate_pairs
    from test_crsecpo_pairs import ScriptedBackend
    record = dict(id="one", source="source", target="target", family_id="f", split="train",
                  target_policy="permissible", selection_group="benign_adversarial_control",
                  selection_status="accepted", provenance={"review_export_sha256": "f" * 64,
                    "publisher_partition": "train"})
    record.update(updates)
    return generate_pairs([record], backend=ScriptedBackend(), tokenizer=Tokenizer(),
        model_spec=ModelSpec("org/model", "a"*40, "org/model", "a"*40))[0][0]


def config(tmp_path, rows=None):
    path = tmp_path / "pairs.jsonl"
    if rows is not None:
        path.write_text("".join(json.dumps(item) + "\n" for item in rows))
    return SimpleNamespace(data_file=str(path), max_length=10000, max_prompt_length=None,
                           append_eos=True, online_exchange_dir=None, seed=42,
                           model_name_or_path="org/model", model_revision="a" * 40)


def test_all_methods_preserve_rows_and_policy_tensors(tmp_path):
    data = [row()]
    cfg = config(tmp_path, data)
    baseline = ComparisonCollator(Tokenizer(), cfg.max_length, method="dpo")(data)
    hashes = []
    for method in METHODS:
        rows, collator, manifest = prepare(cfg, method, tokenizer=Tokenizer())
        assert rows == data
        assert manifest["ordered_ids"] == ["one"]
        assert manifest["final_test_ingested"] is False
        hashes.append(manifest["input_sha256"])
        batch = collator(rows)
        for suffix in ("input_ids", "attention_mask", "labels"):
            key = suffix if method == "sft" else f"policy_chosen_{suffix}"
            assert torch.equal(batch[key], baseline[f"policy_chosen_{suffix}"])
        if method != "sft":
            for suffix in ("input_ids", "attention_mask", "labels"):
                assert torch.equal(batch[f"policy_rejected_{suffix}"], baseline[f"policy_rejected_{suffix}"])
    assert len(set(hashes)) == 1


def test_prompt_mask_response_identity_and_native_eot():
    batch = ComparisonCollator(Tokenizer(), 10000, method="crsecpo")([row()])
    for response in ("chosen", "rejected"):
        response_labels = []
        for context in ("policy", "reference"):
            name = f"{context}_{response}"
            labels = batch[name + "_labels"][0]
            ids = batch[name + "_input_ids"][0]
            assert labels[0].item() == IGNORE_INDEX
            assert torch.equal(ids[labels != IGNORE_INDEX], labels[labels != IGNORE_INDEX])
            response_labels.append(labels[labels != IGNORE_INDEX])
        assert torch.equal(*response_labels)
        assert response_labels[0][-1].item() == 2
        assert (response_labels[0] == 2).sum().item() == 1


def test_dpo_is_secpo_with_identical_contexts():
    data = row()
    dpo = ComparisonCollator(Tokenizer(), 10000, method="dpo")([data])
    data["clean_prompt"] = data["prompt"]
    secpo = ComparisonCollator(Tokenizer(), 10000, method="crsecpo")([data])
    assert all(torch.equal(dpo[key], secpo[key]) for key in dpo)
    # Analytic DPO loss equality, including summed response token logps.
    values = [torch.tensor([x], requires_grad=i < 2) for i, x in enumerate((-2., -3., -4., -6.))]
    actual = secpo_loss(*values, beta=0.1).losses
    expected = -torch.nn.functional.logsigmoid(0.1 * ((values[0]-values[1])-(values[2]-values[3])))
    torch.testing.assert_close(actual, expected)


def test_reference_is_frozen_and_prompt_logits_have_no_loss():
    labels = torch.tensor([[IGNORE_INDEX, IGNORE_INDEX, 3, 2]])
    policy = torch.randn(1, 4, 64, requires_grad=True)
    reference = torch.randn(1, 4, 64, requires_grad=True)
    lp = sequence_logps(policy, labels)
    with torch.no_grad():
        lr = sequence_logps(reference, labels)
    secpo_loss(lp, torch.zeros_like(lp), lr, lr - 2, beta=.1).losses.mean().backward()
    assert reference.grad is None
    assert policy.grad[0, 0].abs().sum() == 0
    assert policy.grad[0, 1:3].abs().sum() > 0


@pytest.mark.parametrize("method", METHODS)
def test_overlong_reference_rejected_even_for_sft(tmp_path, method):
    cfg = config(tmp_path, [row()])
    cfg.max_length = 50  # ordinary fits; both rubric references do not
    with pytest.raises(PreferenceSchemaError, match="one.*max_length"):
        prepare(cfg, method, tokenizer=Tokenizer())


@pytest.mark.parametrize("mutation,match", [
    (lambda r: r.update(split="final_test"), "final-test"),
    (lambda r: r["provenance"].update(publisher_partition="eval"), "evaluation source"),
    (lambda r: r["provenance"].pop("publisher_partition"), "evaluation source"),
    (lambda r: r.update(selection_status="unresolved"), "accepted"),
    (lambda r: r.update(target_policy="unknown"), "resolved"),
    (lambda r: r.update(selection_group="harmful_benign_framing"), "inconsistent"),
    (lambda r: r["pair_metadata"]["model"].update(tokenizer_revision="b"*40), "provenance"),
    (lambda r: r.update(clean_prompt="tampered"), "chat rendering"),
    (lambda r: r.update(chosen="unterminated"), "terminator"),
    (lambda r: r.update(chosen=None), "terminator"),
    (lambda r: r.update(chosen="good<|eot_id|><|eot_id|>"), "terminator"),
    (lambda r: r.update(chosen=r["rejected"], rejected=r["chosen"]), "recorded ordinary/combined"),
    (lambda r: r["pair_metadata"]["judgments"][0].update(outcome="ordinary_wins"), "recorded ordinary/combined"),
])
def test_leakage_and_provenance_fail_closed(tmp_path, mutation, match):
    data = row()
    mutation(data)
    with pytest.raises(PreferenceSchemaError, match=match):
        prepare(config(tmp_path, [data]), "sft", tokenizer=Tokenizer())


def test_duplicate_id_rejected(tmp_path):
    with pytest.raises(PreferenceSchemaError, match="duplicate"):
        prepare(config(tmp_path, [row(), row()]), "dpo", tokenizer=Tokenizer())


def test_arbitrary_reference_requires_opt_in():
    data = {key: row()[key] for key in ("prompt", "clean_prompt", "chosen", "rejected")}
    with pytest.raises(PreferenceSchemaError):
        validate_preference_record(data)
    assert validate_preference_record(data, allow_arbitrary_reference=True).clean_prompt == data["clean_prompt"]


def test_original_trainer_disables_adapter_and_reference_gradients(monkeypatch):
    """Run original trainer forwards on a tiny torch model, without downloads."""
    import contextlib
    import importlib.util
    from pathlib import Path
    import sys
    import types

    # Only Trainer's base class is needed: bypass its heavyweight constructor.
    # Load under an isolated name so this does not alter normal module imports.
    if importlib.util.find_spec("transformers") is None:
        stub = types.ModuleType("transformers")
        stub.Trainer = object
        monkeypatch.setitem(sys.modules, "transformers", stub)
    spec = importlib.util.spec_from_file_location(
        "secpo._crsecpo_fixture_trainer", Path(__file__).parents[1] / "src/secpo/trainer.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class TinyPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base = torch.nn.Embedding(64, 64)
            self.base.weight.requires_grad_(False)
            self.lora_weight = torch.nn.Parameter(torch.linspace(-.1, .1, 64))
            self.disabled = False
            self.calls = []

        @contextlib.contextmanager
        def disable_adapter(self):
            self.disabled = True
            try:
                yield
            finally:
                self.disabled = False

        def forward(self, input_ids, attention_mask, use_cache=False):
            self.calls.append((self.disabled, torch.is_grad_enabled()))
            logits = self.base(input_ids)
            if not self.disabled:
                logits = logits + self.lora_weight
            return SimpleNamespace(logits=logits)

    policy = TinyPolicy()
    frozen_before = policy.base.weight.detach().clone()
    trainer = object.__new__(module.SecPOTrainer)
    trainer.beta = .1
    batch = ComparisonCollator(Tokenizer(), 10000, method="crsecpo")([row()])
    before = {key: value.clone() for key, value in batch.items()}
    output = trainer.compute_secpo_output(policy, batch)
    output.losses.mean().backward()
    assert policy.calls == [(False, True), (False, True), (True, False), (True, False)]
    assert not policy.disabled
    assert policy.base.weight.grad is None
    assert policy.lora_weight.grad is not None
    assert policy.lora_weight.grad.abs().sum() > 0
    torch.testing.assert_close(policy.base.weight, frozen_before)
    assert all(torch.equal(value, before[key]) for key, value in batch.items())


@pytest.mark.parametrize("method", METHODS)
def test_real_hf_one_step_with_random_tiny_model(tmp_path, monkeypatch, method):
    """Optional real HF/PEFT integration: weights are random and local only."""
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    from dataclasses import replace
    from pathlib import Path
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import LlamaConfig, LlamaForCausalLM
    from secpo import training_cli
    from secpo.crsecpo.training import build_trainer

    class WordTokenizer(Tokenizer):
        def __call__(self, text, add_special_tokens=False):
            ids = []
            for index, chunk in enumerate(text.split("<|eot_id|>")):
                if index:
                    ids.append(2)
                ids.extend(3 + sum(map(ord, word)) % 61 for word in chunk.split())
            return {"input_ids": ids}

        def save_pretrained(self, path):
            # The checkpoint uses real Trainer/PEFT serialization; only the
            # injected fixture tokenizer has no pretrained files to save.
            pass

    data = row()
    cfg = config(tmp_path, [data])
    cfg = replace(training_cli.load_training_config(
        Path(__file__).parents[1] / "configs/train_crsecpo_llama31.json"),
        data_file=cfg.data_file, model_name_or_path=cfg.model_name_or_path,
        model_revision=cfg.model_revision, output_dir=str(tmp_path / method),
        max_length=4096, per_device_train_batch_size=1, gradient_accumulation_steps=1,
        max_steps=1, bf16=False, torch_dtype="float32", gradient_checkpointing=False,
        optim="adamw_torch", save_strategy="steps", save_steps=1,
        lora_dropout=0.0, report_to="none")
    initial_states = []

    def tiny_policy(config):
        base = LlamaForCausalLM(LlamaConfig(vocab_size=64, hidden_size=16,
            intermediate_size=24, num_hidden_layers=1, num_attention_heads=2,
            num_key_value_heads=2, max_position_embeddings=4096,
            bos_token_id=1, eos_token_id=2, pad_token_id=0))
        model = get_peft_model(base, LoraConfig(r=2, lora_alpha=4, lora_dropout=0.0,
            target_modules=["q_proj", "v_proj"], task_type=TaskType.CAUSAL_LM))
        initial_states.append({name: value.detach().clone() for name, value in model.named_parameters()})
        return model

    monkeypatch.setattr(training_cli, "_load_policy", tiny_policy)
    tokenizer = WordTokenizer()
    trainer, _, manifest = build_trainer(cfg, method, tokenizer=tokenizer)
    # A repeated build must re-seed before constructing both base and LoRA.
    build_trainer(cfg, method, tokenizer=tokenizer)
    assert all(torch.equal(value, initial_states[1][name]) for name, value in initial_states[0].items())
    base_before = {name: value.clone() for name, value in trainer.model.named_parameters() if not value.requires_grad}
    result = trainer.train()
    assert result.global_step == 1
    assert manifest["ordered_ids"] == ["one"]
    assert all(torch.equal(value, base_before[name]) for name, value in trainer.model.named_parameters() if not value.requires_grad)
    from secpo.crsecpo.training import guard_output, _write_manifest_once
    _write_manifest_once(Path(cfg.output_dir) / "training_manifest.json", manifest)
    guard_output(cfg, manifest, resume_from_checkpoint=str(Path(cfg.output_dir) / "checkpoint-1"))


@pytest.mark.parametrize("world_size,n_gpu", [(2, 1), (1, 2)])
def test_parallel_training_rejected_before_weights(tmp_path, monkeypatch, world_size, n_gpu):
    from secpo import training_cli
    from secpo.crsecpo.training import build_trainer
    monkeypatch.setattr(training_cli, "_build_training_arguments",
                        lambda _: SimpleNamespace(world_size=world_size, n_gpu=n_gpu))
    monkeypatch.setattr(training_cli, "_load_policy", lambda _: pytest.fail("weights loaded before runtime guard"))
    with pytest.raises(ValueError, match="one process and at most one GPU"):
        build_trainer(config(tmp_path, [row()]), "crsecpo", tokenizer=Tokenizer())


def run_manifest():
    return {"schema_version": "crsecpo-training-manifest-v1", "method": "crsecpo",
            "input_sha256": "a" * 64, "ordered_ids": ["one"],
            "training_config": {"seed": 42, "learning_rate": .001,
                                "model_revision": "b" * 40, "lora_r": 2,
                                "max_length": 4096, "output_dir": "old", "data_file": "old"}}


def resume_checkpoint(path, *, weights="adapter_model.safetensors"):
    path.mkdir(parents=True)
    for name in ("trainer_state.json", "optimizer.pt", "scheduler.pt",
                 "rng_state.pth", "adapter_config.json", weights):
        (path / name).write_bytes(b"fixture")
    return path


def test_output_guard_rejects_unrelated_and_requires_explicit_resume(tmp_path):
    from secpo.crsecpo.training import guard_output, _write_manifest_once
    cfg = SimpleNamespace(output_dir=str(tmp_path / "out"))
    output = tmp_path / "out"
    output.mkdir()
    (output / "adapter_model.safetensors").write_bytes(b"unrelated")
    with pytest.raises(ValueError, match="unrelated"):
        guard_output(cfg, run_manifest())
    (output / "adapter_model.safetensors").unlink()
    canonical = guard_output(cfg, run_manifest())
    _write_manifest_once(canonical, run_manifest())
    original = canonical.read_bytes()
    (output / "adapter_model.safetensors").write_bytes(b"trained")
    with pytest.raises(ValueError, match="explicitly resume"):
        guard_output(cfg, run_manifest())
    checkpoint = resume_checkpoint(output / "checkpoint-1")
    guard_output(cfg, run_manifest(), resume_from_checkpoint=str(checkpoint))
    _write_manifest_once(canonical, run_manifest())
    assert canonical.read_bytes() == original


@pytest.mark.parametrize("weights", ["adapter_model.safetensors", "adapter_model.bin"])
@pytest.mark.parametrize("missing", ["optimizer.pt", "scheduler.pt", "rng_state.pth",
                                     "adapter_config.json", "weights"])
def test_resume_requires_full_training_state_and_adapter(tmp_path, weights, missing):
    from secpo.crsecpo.training import guard_output, _write_manifest_once
    checkpoint = resume_checkpoint(tmp_path / "old" / "checkpoint-1", weights=weights)
    manifest = run_manifest()
    _write_manifest_once(checkpoint.parent / "training_manifest.json", manifest)
    cfg = SimpleNamespace(output_dir=str(tmp_path / "new"))
    guard_output(cfg, manifest, resume_from_checkpoint=str(checkpoint))
    artifact = checkpoint / (weights if missing == "weights" else missing)
    artifact.unlink()
    with pytest.raises(ValueError, match="complete single-process.*missing or empty"):
        guard_output(cfg, manifest, resume_from_checkpoint=str(checkpoint))
    artifact.touch()
    with pytest.raises(ValueError, match="complete single-process.*missing or empty"):
        guard_output(cfg, manifest, resume_from_checkpoint=str(checkpoint))
    assert not Path(cfg.output_dir).exists()


@pytest.mark.parametrize("change", [
    lambda m: m.update(method="dpo"),
    lambda m: m.update(input_sha256="c" * 64),
    lambda m: m.update(ordered_ids=["other"]),
    lambda m: m["training_config"].update(seed=43),
    lambda m: m["training_config"].update(learning_rate=.002),
    lambda m: m["training_config"].update(model_revision="c" * 40),
    lambda m: m["training_config"].update(lora_r=4),
    lambda m: m["training_config"].update(max_length=2048),
])
def test_output_guard_blocks_changed_full_run_identity(tmp_path, change):
    from secpo.crsecpo.training import guard_output, _write_manifest_once
    cfg = SimpleNamespace(output_dir=str(tmp_path))
    canonical = tmp_path / "training_manifest.json"
    _write_manifest_once(canonical, run_manifest())
    changed = run_manifest()
    change(changed)
    before = canonical.read_bytes()
    with pytest.raises(ValueError, match="differs"):
        guard_output(cfg, changed, prepare_only=True)
    with pytest.raises(ValueError, match="overwrite"):
        _write_manifest_once(canonical, changed)
    assert canonical.read_bytes() == before


def test_resume_rejects_foreign_dataset_before_new_output_created(tmp_path):
    from secpo.crsecpo.training import guard_output, _write_manifest_once
    checkpoint = tmp_path / "old" / "checkpoint-1"
    checkpoint.mkdir(parents=True)
    (checkpoint / "trainer_state.json").write_text("{}")
    _write_manifest_once(checkpoint.parent / "training_manifest.json", run_manifest())
    manifest = run_manifest()
    manifest["input_sha256"] = "d" * 64
    cfg = SimpleNamespace(output_dir=str(tmp_path / "new"))
    with pytest.raises(ValueError, match="differs"):
        guard_output(cfg, manifest, resume_from_checkpoint=str(checkpoint))
    assert not (tmp_path / "new").exists()


def test_cli_prepare_overrides_and_method_specific_default(tmp_path, monkeypatch):
    from dataclasses import make_dataclass
    import sys
    import types
    from secpo.crsecpo.training import main, freeze_pairs
    cfg = config(tmp_path, [row(), row(id="harmful", family_id="h", target_policy="limited",
        selection_group="harmful_benign_framing")])
    freeze_pairs(cfg, tokenizer=Tokenizer(), output_dir=tmp_path / "frozen", pair_limit=2)
    cfg.data_file = str(tmp_path / "frozen" / "train_pairs.jsonl")
    cfg.output_dir = str(tmp_path / "outputs")
    config_type = make_dataclass("FixtureConfig", [(name, object) for name in vars(cfg)], frozen=True)
    cfg = config_type(**vars(cfg))
    fake = types.ModuleType("secpo.training_cli")
    fake.load_training_config = lambda _: cfg
    fake._load_tokenizer = lambda _: Tokenizer()
    monkeypatch.setitem(sys.modules, "secpo.training_cli", fake)
    main(["--config", "unused.json", "--method", "sft", "--prepare"])
    default = tmp_path / "outputs" / "sft" / "training_manifest.json"
    assert json.loads(default.read_text())["method"] == "sft"
    alternate_data = tmp_path / "frozen" / "alternate.jsonl"
    alternate_data.write_bytes((tmp_path / "frozen" / "train_pairs.jsonl").read_bytes())
    explicit_output = tmp_path / "explicit"
    main(["--config", "unused.json", "--method", "dpo", "--prepare",
          "--output-dir", str(explicit_output), "--data-file", str(alternate_data)])
    manifest = json.loads((explicit_output / "training_manifest.json").read_text())
    assert manifest["method"] == "dpo"
    assert manifest["training_config"]["data_file"] == str(alternate_data)
    assert manifest["training_config"]["output_dir"] == str(explicit_output)


@pytest.mark.parametrize("method", METHODS)
def test_token_identical_responses_rejected_for_every_method(tmp_path, method):
    class AliasingTokenizer(Tokenizer):
        def __call__(self, text, add_special_tokens=False):
            ids = super().__call__(text, add_special_tokens=add_special_tokens)["input_ids"]
            return {"input_ids": [3 if value > 2 else value for value in ids]}

    data = row()
    # Ordinary/combined strings are different but equal-length under this tokenizer.
    with pytest.raises(PreferenceSchemaError, match="one.*identical response token IDs"):
        prepare(config(tmp_path, [data]), method, tokenizer=AliasingTokenizer())


def test_freeze_balanced_pairs_preserves_all_input_rows_and_common_length_checks(tmp_path):
    from secpo.crsecpo.training import freeze_pairs
    rows = [row(id="b1"), row(id="b2", family_id="b2"), row(id="b3", family_id="b3"),
            row(id="h1", family_id="h", target_policy="limited", selection_group="harmful_benign_framing"),
            row(id="h2", family_id="h2", target_policy="prohibited", selection_group="harmful_benign_framing"),
            row(id="overlong", source="long source " * 1000)]
    cfg = config(tmp_path, rows)
    original = Path(cfg.data_file).read_bytes()
    target = tmp_path / "frozen"
    manifest = freeze_pairs(cfg, tokenizer=Tokenizer(), output_dir=target, pair_limit=4)
    assert manifest["selected_counts"] == {"harmful": 2, "benign": 2}
    assert manifest["eligible_counts"] == {"harmful": 2, "benign": 3}
    audit = [json.loads(line) for line in (target / "selection_audit.jsonl").read_text().splitlines()]
    assert {r["id"] for r in audit} == {r["id"] for r in rows}
    assert next(r for r in audit if r["id"] == "overlong")["status"] == "excluded"
    assert Path(cfg.data_file).read_bytes() == original
    cfg.data_file = str(target / "train_pairs.jsonl")
    for method in METHODS:
        selected, _, run = prepare(cfg, method, tokenizer=Tokenizer(), require_selection=True)
        assert run["common_selection"] == manifest
        assert run["ordered_ids"] == manifest["ordered_ids"]
    with pytest.raises(FileExistsError):
        freeze_pairs(cfg, tokenizer=Tokenizer(), output_dir=target, pair_limit=4)
    (target / "train_pairs.jsonl").write_text(json.dumps(selected[0]) + "\n")
    with pytest.raises(PreferenceSchemaError, match="common selection"):
        prepare(cfg, "dpo", tokenizer=Tokenizer(), require_selection=True)


def test_scientific_training_requires_common_selection_and_both_classes(tmp_path):
    from secpo.crsecpo.training import freeze_pairs
    cfg = config(tmp_path, [row()])
    with pytest.raises(PreferenceSchemaError, match="freeze-pairs"):
        prepare(cfg, "dpo", tokenizer=Tokenizer(), require_selection=True)
    with pytest.raises(PreferenceSchemaError, match="both classes"):
        freeze_pairs(cfg, tokenizer=Tokenizer(), output_dir=tmp_path / "empty", pair_limit=2)
    assert not (tmp_path / "empty").exists()


def test_default_run_saves_a_resumable_checkpoint_before_completion(tmp_path):
    """The 2,048-pair budget must not finish before its first resume point."""
    from dataclasses import replace
    from transformers import TrainerControl, TrainerState
    from transformers.trainer_callback import DefaultFlowCallback
    from secpo.training_cli import _build_training_arguments, load_training_config

    cfg = load_training_config(Path(__file__).parents[1] / "configs/train_crsecpo_llama31.json")
    steps = 2048 // (cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps)
    args = _build_training_arguments(replace(cfg, output_dir=str(tmp_path),
        bf16=False, torch_dtype="float32", optim="adamw_torch"))
    state = TrainerState(global_step=cfg.save_steps, max_steps=steps, save_steps=cfg.save_steps)
    control = DefaultFlowCallback().on_step_end(args, state, TrainerControl())
    assert control.should_save and not control.should_training_stop
    assert not args.save_only_model  # Resume must retain optimizer/scheduler/RNG state.


def test_valid_custom_system_pair_rejected_by_user_only_training_protocol(tmp_path):
    from secpo.crsecpo.inference import ModelSpec
    from secpo.crsecpo.pairs import generate_pairs
    from secpo.crsecpo.prompts import validate_training_pair
    from test_crsecpo_pairs import ScriptedBackend
    pairs, _ = generate_pairs([row()], backend=ScriptedBackend(), tokenizer=Tokenizer(),
        model_spec=ModelSpec("org/model", "a"*40, "org/model", "a"*40),
        deployment_system="Respond using a museum-specific system instruction.")
    validate_training_pair(pairs[0], Tokenizer(), "org/model", "a"*40)
    with pytest.raises(PreferenceSchemaError, match="user-only deployment prompts"):
        prepare(config(tmp_path, pairs), "crsecpo", tokenizer=Tokenizer())
