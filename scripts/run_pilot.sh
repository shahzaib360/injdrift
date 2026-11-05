#!/usr/bin/env bash
# Reproduce every number in results/pilot on a CPU-only machine.
#
# Wall-clock on a 4-core Xeon (AMX bf16), one job at a time: roughly 7-8 hours.
# Every step is resumable: finished hardening runs and grid entries are skipped.
set -euo pipefail
cd "$(dirname "$0")/.."

harden() {  # <hub id> <tag>
  if [ ! -f "runs/$2/model.safetensors" ]; then
    injdrift harden --base "$1" --method secalign --n 1600 \
      --out "runs/$2" --results "results/pilot/harden-$2.json"
  fi
}

# 1. Hardened bases: mini-SecAlign (DPO on 1,600 injected Alpaca samples, LoRA merged).
harden Qwen/Qwen2.5-0.5B-Instruct qwen0.5b-secalign
harden HuggingFaceTB/SmolLM2-360M-Instruct smollm2-360m-secalign

# 2. Undefended vs hardened evaluations, then the fine-tuning grid (with tracking).
injdrift grid configs/grid/pilot.yaml

# 3. Adaptive attack: GCG against the hardened base and an eroded fine-tune, plus
#    transfer of the base's suffixes to the fine-tune.
python scripts/run_gcg.py --model runs/qwen0.5b-secalign --name qwen-secalign
python scripts/run_gcg.py --model runs/qwen0.5b-secalign --adapter runs/adapters/qwen-dolly-lora \
  --name qwen-dolly-lora
python scripts/run_gcg.py --model runs/qwen0.5b-secalign --adapter runs/adapters/qwen-dolly-lora \
  --name qwen-dolly-lora --transfer-from results/pilot/gcg/qwen-secalign.json

# 4. Tables and figures.
injdrift report results/pilot --out results/pilot
