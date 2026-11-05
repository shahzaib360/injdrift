"""Adaptive attack experiment: GCG against a model, and transfer of found suffixes.

Examples::

    # optimize suffixes against the hardened base
    python scripts/run_gcg.py --model runs/qwen0.5b-secalign --name qwen-secalign

    # same budget against a fine-tune (base + LoRA adapter saved by the grid)
    python scripts/run_gcg.py --model runs/qwen0.5b-secalign \\
        --adapter runs/adapters/qwen-dolly-lora --name qwen-dolly-lora

    # replay the suffixes found on the base against the fine-tune (no optimization)
    python scripts/run_gcg.py --model runs/qwen0.5b-secalign \\
        --adapter runs/adapters/qwen-dolly-lora --name qwen-dolly-lora \\
        --transfer-from results/pilot/gcg/qwen-secalign.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from injdrift.attacks.gcg import GCGConfig, _with_suffix, gcg_attack
from injdrift.data.suites import normalize
from injdrift.evaluate.cases import cases_for_items, suite_pool
from injdrift.evaluate.runner import encode_case
from injdrift.modeling import generate, load
from injdrift.stats import wilson

# Items 64.. of the AlpacaFarm pool: disjoint from the items of the standard profile.
FIRST_ITEM = 64


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--adapter")
    p.add_argument("--name", required=True)
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--batch", type=int, default=48)
    p.add_argument("--n-tokens", type=int, default=16)
    p.add_argument("--transfer-from", help="re-use suffixes from an earlier GCG result file")
    p.add_argument("--out", default="results/pilot/gcg")
    args = p.parse_args()

    loaded = load(args.model, adapter=args.adapter)
    items = suite_pool("alpacafarm")[FIRST_ITEM : FIRST_ITEM + args.n]
    cases = cases_for_items(items, ("naive",), ("end",), loaded.dialect, index_offset=FIRST_ITEM)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    if args.transfer_from:
        source = json.loads(Path(args.transfer_from).read_text())
        suffixes = {r["key"]: r.get("suffix", "") for r in source["results"]}
        encs, kept = [], []
        for c in cases:
            if c.key in suffixes:
                attacked = _with_suffix(c, suffixes[c.key])
                encs.append(encode_case(loaded, attacked, "secure").input_ids)
                kept.append(c)
        responses = generate(loaded, encs, max_new_tokens=16)
        results = [
            {
                "key": c.key,
                "suffix": suffixes[c.key],
                "response": r,
                "success": normalize(r).startswith(c.injected.canary),
            }
            for c, r in zip(kept, responses, strict=True)
        ]
        kind, target = "transfer", out_dir / f"{args.name}.transfer.json"
    else:
        cfg = GCGConfig(steps=args.steps, batch=args.batch, n_tokens=args.n_tokens)
        results = []
        for i, c in enumerate(cases, 1):
            res = gcg_attack(loaded, c, cfg=cfg)
            results.append(res)
            print(
                f"[gcg {args.name}] {i}/{len(cases)} {c.key} success={res['success']} "
                f"steps={res['steps']} loss={res['best_loss']:.2f} ({res['seconds']}s)",
                flush=True,
            )
        kind, target = "gcg", out_dir / f"{args.name}.json"

    k = sum(bool(r["success"]) for r in results)
    lo, hi = wilson(k, len(results))
    record = {
        "kind": kind,
        "model": args.model,
        "adapter": args.adapter,
        "transfer_from": args.transfer_from,
        "n": len(results),
        "asr": k / len(results) if results else None,
        "ci": [lo, hi],
        "seconds": round(time.perf_counter() - t0),
        "results": results,
    }
    target.write_text(json.dumps(record, indent=1))
    print(f"[gcg {args.name}] {kind} ASR {k}/{len(results)} -> {target}", flush=True)


if __name__ == "__main__":
    main()
