"""Run a grid of evaluations and fine-tunes from a YAML file, resumably.

Each entry runs in its own subprocess (memory is fully released between runs and a
crash only loses that run). An entry whose result JSON already exists is skipped, so
re-running the same command resumes a partially finished grid.

Example::

    results: results/pilot
    cache: runs/cache
    bases:
      qwen: runs/qwen0.5b-secalign
    evals:
      - {id: eval-qwen-secalign, model: qwen, fmt: [secure, naive]}
    defaults: {n: 768, epochs: 1.0}
    runs:
      - {id: qwen-medical-lora, base: qwen, recipe: medical}
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import yaml

RUN_KEYS = {
    "recipe": "--recipe",
    "fmt": "--fmt",
    "n": "--n",
    "peft": "--peft",
    "lr": "--lr",
    "epochs": "--epochs",
    "preserve": "--preserve",
    "reharden": "--reharden",
    "eval_profile": "--profile",
    "seed": "--seed",
}


def _commands(cfg: dict) -> list[tuple[str, Path, list[str]]]:
    results = Path(cfg["results"])
    cache = Path(cfg.get("cache", "runs/cache"))
    bases = cfg.get("bases", {})
    out: list[tuple[str, Path, list[str]]] = []
    for e in cfg.get("evals", []):
        model = bases.get(e["model"], e["model"])
        fmts = e.get("fmt", ["secure"])
        target = results / f"{e['id']}.json"
        cmd = [
            "eval",
            "--model",
            model,
            "--profile",
            e.get("profile", "standard"),
            "--out",
            str(target),
        ]
        cmd += ["--fmt", *fmts]
        out.append((e["id"], target, cmd))
    defaults = cfg.get("defaults", {})
    for r in cfg.get("runs", []):
        spec = {**defaults, **r}
        base_key = spec["base"]
        target = results / f"{spec['id']}.json"
        cmd = [
            "finetune",
            "--run-id",
            spec["id"],
            "--base",
            bases.get(base_key, base_key),
            "--results",
            str(results),
            "--cache",
            str(cache / base_key),
        ]
        for key, flag in RUN_KEYS.items():
            if key in spec and spec[key] is not None:
                cmd += [flag, str(spec[key])]
        out.append((spec["id"], target, cmd))
    return out


def run_grid(config: Path, *, only: list[str] | None = None, dry_run: bool = False) -> int:
    cfg = yaml.safe_load(config.read_text())
    failures = 0
    for run_id, target, cmd in _commands(cfg):
        if only and run_id not in only:
            continue
        if target.exists():
            print(f"[grid] skip {run_id} (done)", flush=True)
            continue
        full = [sys.executable, "-m", "injdrift.cli", *cmd]
        print(f"[grid] run {run_id}: {' '.join(cmd)}", flush=True)
        if dry_run:
            continue
        t0 = time.perf_counter()
        proc = subprocess.run(full, check=False)
        status = "ok" if proc.returncode == 0 else f"FAILED ({proc.returncode})"
        print(f"[grid] {run_id} {status} in {time.perf_counter() - t0:.0f}s", flush=True)
        failures += proc.returncode != 0
    return 1 if failures else 0
