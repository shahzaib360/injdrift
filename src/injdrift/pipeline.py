"""End-to-end steps: harden a base model, fine-tune it, evaluate it, record everything.

Result files are plain JSON (summary, config, training curve, tracked checkpoints,
probe metrics) plus a gzipped JSONL of every graded response, so every number in
the README can be traced back to model outputs.
"""

from __future__ import annotations

import gzip
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from injdrift.evaluate.cases import build_plan
from injdrift.evaluate.runner import evaluate, headline, summarize
from injdrift.modeling import Loaded, autocast, load, release_memory
from injdrift.probes.separation import RoleProbe, attention_share, build_probe_set, summarize_layers
from injdrift.train.examples import PrefExample, SFTExample, pref_example, sft_example
from injdrift.train.hardening import secalign_pairs, struq_conversations
from injdrift.train.loop import TrainConfig, pref_logps, sft_loss_sum, train
from injdrift.train.recipes import RECIPES
from injdrift.train.tracking import Tracker

MAX_LEN = 448
ADAPTERS = Path("runs/adapters")


class Logger:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
        self.t0 = time.perf_counter()

    def __call__(self, msg: str | dict) -> None:
        if isinstance(msg, dict):
            if "track" in msg:
                t = msg["track"]
                msg = (
                    f"[track] step {t['step']} ({100 * t['frac']:.0f}%): ASR {100 * t['asr']:.1f}% "
                    f"canary log-prob {t['canary_logprob']:.2f}"
                )
            elif msg.get("step", 0) % 8 == 0 or msg.get("step") == 1:
                extra = " ".join(
                    f"{k}={v:.3g}"
                    for k, v in msg.items()
                    if k not in ("step", "elapsed_s") and isinstance(v, float)
                )
                msg = f"step {msg['step']} {extra} ({msg['elapsed_s']}s)"
            else:
                return
        line = f"[{time.perf_counter() - self.t0:7.0f}s] {msg}"
        print(line, flush=True)
        if self.path:
            with self.path.open("a") as fh:
                fh.write(line + "\n")


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=_json_default))


def _json_default(o):
    if isinstance(o, float) and math.isnan(o):
        return None
    if hasattr(o, "tolist"):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o)}")


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def read_rows(path: Path) -> list[dict]:
    with gzip.open(path, "rt") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# -------------------------------------------------------------------------- examples


def sft_examples(loaded: Loaded, convs) -> list[SFTExample]:
    out = (sft_example(loaded.tokenizer, loaded.dialect, m, loaded.strip, MAX_LEN) for m in convs)
    return [e for e in out if e is not None]


def pref_examples(loaded: Loaded, records) -> list[PrefExample]:
    out = (
        pref_example(loaded.tokenizer, loaded.dialect, r, loaded.strip, MAX_LEN) for r in records
    )
    return [e for e in out if e is not None]


@torch.inference_mode()
def heldout_score(loaded: Loaded, exs: list, kind: str) -> float:
    """SFT: mean token NLL on held-out examples. DPO: mean (chosen - rejected) log-prob."""
    pad = loaded.tokenizer.pad_token_id
    if kind == "sft":
        total, count = 0.0, 0
        for k in range(0, len(exs), 8):
            with autocast(loaded.device):
                s, n = sft_loss_sum(loaded.model, exs[k : k + 8], pad)
            total, count = total + float(s), count + n
        return total / max(count, 1)
    diffs = []
    for k in range(0, len(exs), 4):
        with autocast(loaded.device):
            c, r = pref_logps(loaded.model, exs[k : k + 4], pad)
        diffs.extend((c - r).tolist())
    return sum(diffs) / max(len(diffs), 1)


# --------------------------------------------------------------------------- harden


def harden(
    base: str,
    out_dir: Path,
    *,
    method: str = "secalign",
    n: int = 1200,
    cfg: TrainConfig | None = None,
    seed: int = 0,
    results: Path | None = None,
) -> dict:
    """Train a mini-SecAlign (DPO) or mini-StruQ (SFT) model and save it merged."""
    log = Logger(out_dir / "train.log" if out_dir else None)
    out_dir.mkdir(parents=True, exist_ok=True)
    loaded = load(base)
    if method == "secalign":
        cfg = cfg or TrainConfig(kind="dpo", lr=2e-4, epochs=1.0, track_fracs=(0.25, 0.5, 0.75))
        exs = pref_examples(loaded, secalign_pairs(n, seed=seed))
    elif method == "struq":
        cfg = cfg or TrainConfig(kind="sft", lr=2e-4, epochs=1.0, track_fracs=(0.25, 0.5, 0.75))
        exs = sft_examples(loaded, struq_conversations(n, seed=seed))
    else:
        raise ValueError(f"unknown hardening method {method!r}")
    log(f"hardening {base} with {method}: {len(exs)} examples, {cfg.to_dict()}")
    tracker = Tracker(loaded)
    t0 = time.perf_counter()
    history = train(loaded, cfg, exs, tracker=tracker, log=log)
    train_s = time.perf_counter() - t0
    loaded.model = loaded.model.merge_and_unload().to(torch.bfloat16)
    loaded.model.save_pretrained(out_dir)
    loaded.tokenizer.save_pretrained(out_dir)
    record = {
        "base": base,
        "method": method,
        "n_examples": len(exs),
        "config": cfg.to_dict(),
        "train_seconds": round(train_s),
        "history": history,
        "saved_to": str(out_dir),
    }
    if results:
        write_json(results, record)
    log(f"saved hardened model to {out_dir}")
    return record


def evaluate_model(
    loaded: Loaded,
    *,
    profile: str = "standard",
    fmts: tuple[str, ...] = ("secure",),
    max_new_tokens: int = 24,
    log: Logger | None = None,
) -> tuple[dict, list[dict]]:
    cases = build_plan(profile, loaded.dialect)
    summaries, all_rows = {}, []
    for fmt in fmts:
        t0 = time.perf_counter()
        rows = evaluate(loaded, cases, fmt=fmt, max_new_tokens=max_new_tokens)
        summaries[fmt] = summarize(rows)
        all_rows.extend(rows)
        release_memory()
        if log:
            secs = time.perf_counter() - t0
            log(f"eval[{profile}/{fmt}] {len(cases)} cases in {secs:.0f}s")
            log(f"eval[{profile}/{fmt}] {headline(summaries[fmt])}")
    return summaries, all_rows


# ------------------------------------------------------------------------- baseline


def base_reference(loaded: Loaded, cache: Path, log: Logger) -> tuple:
    """Probe set, frozen role probe and step-0 tracking metrics for a base (cached)."""
    cache.mkdir(parents=True, exist_ok=True)
    ps = build_probe_set(loaded)
    probe_path, track_path = cache / "role_probe.npz", cache / "track0.json"
    if probe_path.exists() and track_path.exists():
        return ps, RoleProbe.from_npz(probe_path), json.loads(track_path.read_text())
    log("computing base reference (frozen role probe + step-0 tracking)")
    tracker = Tracker(loaded, probe_set=ps)
    track0 = tracker(loaded, 0)
    track0.update({"step": 0, "frac": 0.0})
    attn = attention_share(loaded, ps.drift)
    attn["summary"] = summarize_layers(attn["attn_share"])
    track0["attention"] = attn
    tracker.fixed_probe.to_npz(probe_path)
    write_json(track_path, track0)
    return ps, tracker.fixed_probe, json.loads(track_path.read_text())


# ------------------------------------------------------------------------- fine-tune


@dataclass
class RunSpec:
    run_id: str
    base: str
    recipe: str
    fmt: str = "naive"
    n: int = 768
    peft: str = "lora"
    lr: float | None = None
    epochs: float = 1.0
    preserve: str = "none"
    reharden: int = 0  # pairs for post-hoc re-hardening (0 = off)
    eval_profile: str = "standard"
    seed: int = 0
    extra: dict = field(default_factory=dict)  # free-form TrainConfig overrides


def finetune(spec: RunSpec, results_dir: Path, cache_dir: Path) -> dict:
    out_json = results_dir / f"{spec.run_id}.json"
    log = Logger(results_dir / "logs" / f"{spec.run_id}.log")
    log(f"=== run {spec.run_id}: {asdict(spec)}")
    loaded = load(spec.base)
    ps, fixed, track0 = base_reference(loaded, cache_dir, log)

    recipe = RECIPES[spec.recipe]
    records = recipe.build(spec.n + 64, spec.seed, spec.fmt)
    train_recs, held_recs = records[: spec.n], records[spec.n :]
    if recipe.kind == "sft":
        exs, held = sft_examples(loaded, train_recs), sft_examples(loaded, held_recs)
    else:
        exs, held = pref_examples(loaded, train_recs), pref_examples(loaded, held_recs)
    held_before = heldout_score(loaded, held, recipe.kind)

    anchors = replay = None
    if spec.preserve == "secanchor":
        anchors = pref_examples(loaded, secalign_pairs(256, seed=spec.seed, part="anchor"))
    elif spec.preserve == "replay":
        replay = sft_examples(
            loaded, struq_conversations(256, seed=spec.seed, part="anchor", clean_frac=0.0)
        )

    cfg = TrainConfig(
        kind=recipe.kind,
        peft=spec.peft,
        lr=spec.lr,
        epochs=spec.epochs,
        seed=spec.seed,
        preserve=spec.preserve,
        **spec.extra,
    )
    tracker = Tracker(loaded, probe_set=ps, fixed_probe=fixed)
    log(f"training: {len(exs)} {recipe.kind} examples ({recipe.description}), fmt={spec.fmt}")
    t0 = time.perf_counter()
    history = train(loaded, cfg, exs, anchors=anchors, replay=replay, tracker=tracker, log=log)
    train_s = time.perf_counter() - t0
    release_memory()

    if spec.peft == "lora":
        # Keep the adapter (tens of MB) so attacks such as GCG can be run on it later.
        loaded.model.save_pretrained(ADAPTERS / spec.run_id)
        loaded.model = loaded.model.merge_and_unload()
    loaded.model = loaded.model.to(torch.bfloat16)

    reharden_history = None
    if spec.reharden:
        log(f"post-hoc re-hardening on {spec.reharden} SecAlign pairs")
        pairs = pref_examples(loaded, secalign_pairs(spec.reharden, seed=spec.seed, part="anchor"))
        rcfg = TrainConfig(
            kind="dpo", peft="lora", lr=2e-4, epochs=1.0, seed=spec.seed, track_fracs=()
        )
        reharden_history = train(loaded, rcfg, pairs, log=log)
        loaded.model = loaded.model.merge_and_unload().to(torch.bfloat16)

    held_after = heldout_score(loaded, held, recipe.kind)
    total_steps = max(h["step"] for h in history if "step" in h)
    final_track = tracker(loaded, total_steps)
    final_track.update({"step": total_steps, "frac": 1.0})
    attn = attention_share(loaded, ps.drift)
    attn["summary"] = summarize_layers(attn["attn_share"])
    summaries, rows = evaluate_model(loaded, profile=spec.eval_profile, fmts=("secure",), log=log)

    tracks = [track0] + [h["track"] for h in history if "track" in h] + [final_track]
    record = {
        "spec": asdict(spec),
        "config": cfg.to_dict(),
        "recipe": {"name": recipe.name, "kind": recipe.kind, "description": recipe.description},
        "n_train": len(exs),
        "train_seconds": round(train_s),
        "heldout": {
            "before": held_before,
            "after": held_after,
            "metric": "nll" if recipe.kind == "sft" else "logp_margin",
        },
        "history": [h for h in history if "track" not in h],
        "reharden_history": reharden_history,
        "tracking": tracks,
        "final_attention": attn,
        "eval": summaries,
    }
    write_json(out_json, record)
    write_rows(results_dir / "rows" / f"{spec.run_id}.jsonl.gz", rows)
    log(f"done {spec.run_id}: {headline(summaries['secure'])}")
    return record
