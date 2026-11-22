#!/usr/bin/env bash
# Stage 0: install deps, authenticate to the Hugging Face hub, and pre-download
# the pinned Llama snapshot. Pair generation loads the model with
# local_files_only, so the weights MUST be cached before stage 1.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

echo "== Stage 0: setup =="

# Core deps. Add the 'vllm' extra only if you use the vLLM pair backend.
pip install -q -e "${REPO_ROOT}"
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
