#!/usr/bin/env bash
# Stage 0: install deps, authenticate to the Hugging Face hub, and pre-download
# the pinned Llama snapshot. Pair generation loads the model with
# local_files_only, so the weights MUST be cached before stage 1.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

echo "== Stage 0: setup =="

# Colab already ships a CUDA-matched torch; downgrading it to the repo's exact
# pins breaks the GPU build (and the latest vLLM needs the newer torch). So
# install the secpo CODE without forcing its pinned deps, then add its runtime
# libraries loosely so they match the torch already on the machine.
pip install -q --upgrade pip
pip install -q -e "${REPO_ROOT}" --no-deps
pip install -q "transformers>=5.5" "peft>=0.19" "datasets>=4" "accelerate>=1.1" "safetensors>=0.7"
pip install -q "openai>=1.40"   # optional OpenAI grading judge for stage 4
# PEFT eagerly version-checks torchao during LoRA injection and errors on the old
# torchao Colab preinstalls. We never use torchao (plain bf16 LoRA), so remove it.
pip uninstall -q -y torchao 2>/dev/null || true
if [ "${PAIRS_BACKEND}" = "vllm" ]; then
  pip install -q "vllm" || { echo "vLLM install failed; set PAIRS_BACKEND=hf in config.sh"; exit 1; }
fi

# Llama is a gated model: you must have accepted its license and provide a token.
# In Colab: from huggingface_hub import login; login()  -- or set HF_TOKEN.
if [ -n "${HF_TOKEN:-}" ]; then
  python -c "from huggingface_hub import login; login('${HF_TOKEN}')"
fi

echo "Downloading ${MODEL_ID}@${MODEL_REVISION} into the HF cache..."
python - <<PY
from huggingface_hub import snapshot_download
p = snapshot_download("${MODEL_ID}", revision="${MODEL_REVISION}")
print("cached at:", p)
PY

echo "Setup complete."
