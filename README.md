# iseefood

Kitchen video evidence for Edesia. The planned demo combines YOLO ingredient
tracking, Cosmos handling actions, and recipes plus POS orders for expected
consumption. Footage supplies evidence; seeing an ingredient does not measure how
much was used.

## Local Cosmos setup

Uses [NVIDIA Cosmos3 Nano](https://huggingface.co/nvidia/Cosmos3-Nano), pinned to
revision `e59a53c25979a090fa8706c9acc0c254a6e89b92`, with its Reasoner tower loaded
through Transformers. The download includes about 31.5 GB of shared weights and
the vision encoder. Generator-only VAE and audio files are excluded.

Prerequisites: Linux, NVIDIA CUDA 13.0 GPU, adequate memory, and
[uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
bash scripts/download_cosmos.sh
uv venv .venv-cosmos --python 3.12 --seed
uv pip install --python .venv-cosmos/bin/python --torch-backend=cu130 -r requirements-cosmos.txt
.venv-cosmos/bin/python scripts/reason_video.py kitchen.mp4 \
  --ingredient chicken --ingredient rice --ingredient broccoli
```

Evidence is saved to `outputs/evidence.json`. Each event includes an ingredient,
action, timestamps, visible evidence, and uncertainty. The result carries a SHA256
clip identity and remains `needs_review`. It does not deduct inventory or infer
quantities from video. The raw model response is saved alongside the JSON;
invalid responses fail validation.

Verified locally on an NVIDIA GB10 with PyTorch `2.10.0+cu130` and Transformers
`5.11.0`. All eight weight files passed checksum verification, and a six-second
kitchen video produced valid timestamped action JSON on the GPU.

Model weights, environments, clips, generated outputs, and secrets are ignored by
Git. Setup and inference follow NVIDIA's
[Transformers Reasoner cookbook](https://github.com/NVIDIA/cosmos/blob/main/cookbooks/cosmos3/reasoner/run_with_transformers.ipynb).

## Integration plan

The local runner is the initial Cosmos setup. YOLO tracking, recipe/POS
reconciliation, the review UI, and inventory writes still need integration.

1. Track candidate ingredients in the clip with YOLO.
2. Ask Cosmos to label observed handling events and provide timestamps.
3. Match prepared dishes to recipes and POS orders to calculate expected usage.
4. Present discrepancies as possible waste, portion differences, or unmatched
   preparations, with a playable evidence clip.
5. Commit reviewed deductions once, using the clip identity and transaction-level
   idempotency to prevent repeated uploads from changing stock twice.

Picking up and returning an ingredient without visible use should propose no
deduction. Returning a partly used container does not undo earlier consumption.
Flag uncertain observations for review.
