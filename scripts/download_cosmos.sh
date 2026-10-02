#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_XET_HIGH_PERFORMANCE=1

uvx --from huggingface_hub hf download nvidia/Cosmos3-Nano \
  --revision e59a53c25979a090fa8706c9acc0c254a6e89b92 \
  --local-dir "${PROJECT_DIR}/models/Cosmos3-Nano" \
  --exclude 'assets/*' --exclude 'images/*' --exclude 'sound_tokenizer/*' \
  --exclude 'vae/*' --exclude 'scheduler/*'
