#!/usr/bin/env bash
# Reproduce every number in results/pilot on a CPU-only machine.
#
# Wall-clock on a 4-core Xeon (AMX bf16), one job at a time: roughly 7-8 hours.
# Every step is resumable: finished hardening runs and grid entries are skipped.
set -euo pipefail
cd "$(dirname "$0")/.."

harden() {  # <hub id> <tag>
  if [ ! -f "runs/$2/model.safetensors" ]; then
    injdrift harden --base "$1" --method secalign --n 400 \
      --out "runs/$2" --results "results/pilot/harden-$2.json"
  fi
}

# 1. Hardened bases: mini-SecAlign (DPO on 400 injected Alpaca samples, LoRA merged).
#    (The 1,600-pair Qwen ablation in results/pilot was produced the same way with
#    --n 1600 --out runs/qwen0.5b-secalign-n1600.)
harden Qwen/Qwen2.5-0.5B-Instruct qwen0.5b-secalign
harden HuggingFaceTB/SmolLM2-360M-Instruct smollm2-360m-secalign

# 2. Undefended vs hardened evaluations, then the fine-tuning grid (with tracking).
injdrift grid configs/grid/pilot.yaml

# 3. Adaptive attack: GCG (8 cases, 30 steps, 32 candidates) against the hardened base and
#    against the most-eroded plain LoRA fine-tune, then transfer of the base's suffixes to
#    every Qwen fine-tune.
RUN=$(python - <<'PY'
import glob, json
best = None
for p in glob.glob("results/pilot/qwen-*.json"):
    d = json.load(open(p)); s = d["spec"]
    if s["peft"] != "lora" or s["preserve"] != "none" or s.get("reharden"):
        continue
    asr = d["eval"]["secure"]["asr"]["rate"]
    if best is None or asr > best[0]:
        best = (asr, s["run_id"])
print(best[1])
PY
)
GCG="--n 8 --steps 30 --batch 32"
python scripts/run_gcg.py --model runs/qwen0.5b-secalign --name qwen-secalign $GCG
python scripts/run_gcg.py --model runs/qwen0.5b-secalign --adapter "runs/adapters/$RUN" --name "$RUN" $GCG
for A in runs/adapters/qwen-*; do
  python scripts/run_gcg.py --model runs/qwen0.5b-secalign --adapter "$A" --name "$(basename "$A")" \
    --n 8 --transfer-from results/pilot/gcg/qwen-secalign.json
done

# 4. Tables and figures.
injdrift report results/pilot --out results/pilot
