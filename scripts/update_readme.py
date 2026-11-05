"""Regenerate the result tables in README.md from results/pilot (no hand-copied numbers).

Fills every block delimited by ``<!-- TABLE:<name> -->`` ... ``<!-- /TABLE:<name> -->``::

    python scripts/update_readme.py
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

from injdrift.gate import Thresholds, decide
from injdrift.report import _read_rows, base_eval, load_results, model_of, paired, run_label

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "pilot"
README = ROOT / "README.md"
NAMES = {"qwen": "Qwen2.5-0.5B", "smol": "SmolLM2-360M"}


def pct(x: float | None, d: int = 1) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{100 * x:.{d}f}%"


def ci(rate: dict) -> str:
    lo, hi = rate["ci"]
    return f"{pct(rate['rate'])} [{100 * lo:.0f}–{100 * hi:.0f}]"


def t_hardening(data: dict) -> list[str]:
    lines = [
        "| model | defense | data at test time | static ASR [95% CI] | best-of-K | utility |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for model in ("qwen", "smol"):
        for key, label in (
            ("undefended", "undefended"),
            ("secalign", "**mini-SecAlign (400 pairs)**"),
            ("secalign1600", "mini-SecAlign (1,600 pairs, ablation)"),
        ):
            d = data["evals"].get(f"eval-{model}-{key}")
            if not d:
                continue
            for fmt in ("secure", "naive"):
                s = d["eval"][fmt]
                lines.append(
                    f"| {NAMES[model]} | {label} | {fmt} | {ci(s['asr'])} "
                    f"| {pct(s['best_of_k']['rate'])} | {pct(s['utility']['rate'])} |"
                )
    return lines


def t_erosion(data: dict) -> list[str]:
    lines = [
        "| model | fine-tune | static ASR | Δ vs base | newly broken / fixed (p) | best-of-K "
        "| injected-reply log-prob | role-probe d′ | utility |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for model in ("qwen", "smol"):
        base, base_rows = base_eval(data, model)
        if not base:
            continue
        lines.append(
            f"| {NAMES[model]} | *hardened base* | {ci(base['asr'])} | — | — "
            f"| {pct(base['best_of_k']['rate'])} | — | — | {pct(base['utility']['rate'])} |"
        )
        for rid, d in sorted(data["runs"].items()):
            if model_of(rid) != model:
                continue
            s = d["eval"]["secure"]
            pr = paired(base_rows, _read_rows(RESULTS / "rows" / f"{rid}.jsonl.gz"))
            t0, t1 = d["tracking"][0], d["tracking"][-1]
            dp0 = t0["probes"]["summary"]["fixed_dprime"]["upper_mean"]
            dp1 = t1["probes"]["summary"]["fixed_dprime"]["upper_mean"]
            delta = 100 * (s["asr"]["rate"] - base["asr"]["rate"])
            star = " ✱" if pr["p"] < 0.05 else ""  # significantly MORE attackable
            lines.append(
                f"| {NAMES[model]} | {run_label(d['spec'])} | {ci(s['asr'])} | {delta:+.1f} pp "
                f"| {pr['broken']} / {pr['fixed']} (p={pr['p']:.2g}){star} "
                f"| {pct(s['best_of_k']['rate'])} "
                f"| {t0['canary_logprob']:.1f} → {t1['canary_logprob']:.1f} "
                f"| {dp0:.1f} → {dp1:.1f} | {pct(s['utility']['rate'])} |"
            )
    return lines


def t_gate(data: dict) -> list[str]:
    lines = [
        "| model | fine-tune | gate verdict | ASR base → fine-tune | broken / fixed | p |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for model in ("qwen", "smol"):
        _, base_rows = base_eval(data, model)
        base_rows = [r for r in base_rows if r.get("fmt", "secure") == "secure"]
        for rid, d in sorted(data["runs"].items()):
            if model_of(rid) != model or not base_rows:
                continue
            res = decide(
                base_rows,
                _read_rows(RESULTS / "rows" / f"{rid}.jsonl.gz"),
                Thresholds(max_asr=1.0, max_best_of_k=1.0),
            )
            m = res.metrics
            lines.append(
                f"| {NAMES[model]} | {run_label(d['spec'])} | **{res.status.upper()}** "
                f"| {pct(m['asr_reference'])} → {pct(m['asr_candidate'])} "
                f"| {m['newly_broken']} / {m['newly_fixed']} | {m['mcnemar_p']:.2g} |"
            )
    return lines


def t_gcg(data: dict) -> list[str]:
    gdir = RESULTS / "gcg"
    if not gdir.exists():
        return ["*GCG results pending.*"]
    lines = ["| target (Qwen2.5-0.5B) | attack | success |", "| --- | --- | --- |"]
    base = gdir / "qwen-secalign.json"
    if base.exists():
        d = json.loads(base.read_text())
        k = sum(bool(r["success"]) for r in d["results"])
        lines.append(f"| **hardened base** | GCG optimized on this model | **{k}/{d['n']}** |")
    rows = []
    for p in sorted(gdir.glob("*.transfer.json")):
        d = json.loads(p.read_text())
        k = sum(bool(r["success"]) for r in d["results"])
        rid = p.name.removesuffix(".transfer.json")
        label = run_label(data["runs"][rid]["spec"]) if rid in data["runs"] else rid
        rows.append((-k, label, f"| {label} | base's suffixes, replayed | {k}/{d['n']} |"))
    lines += [r[2] for r in sorted(rows)]
    return lines


def t_mitigation(data: dict) -> list[str]:
    lines = [
        "| recipe | method | static ASR | Δ vs hardened base | newly broken / fixed (p) "
        "| best-of-K | utility | held-out task loss | train time |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    base, base_rows = base_eval(data, "qwen")
    groups = [
        ("dolly", "qwen-dolly-lora", "plain fine-tune"),
        ("dolly", "qwen-dolly-lora-replay", "+ replay"),
        ("dolly", "qwen-dolly-lora-reharden", "+ post-hoc re-hardening"),
        ("dolly", "qwen-dolly-lora-secanchor", "**+ SecAnchor**"),
        ("rag", "qwen-rag-lora", "plain fine-tune (naive format)"),
        ("rag", "qwen-rag-secure-lora", "secure format"),
    ]
    for recipe, rid, method in groups:
        d = data["runs"].get(rid)
        if not d or not base:
            continue
        s = d["eval"]["secure"]
        pr = paired(base_rows, _read_rows(RESULTS / "rows" / f"{rid}.jsonl.gz"))
        delta = 100 * (s["asr"]["rate"] - base["asr"]["rate"])
        secs = d["train_seconds"] + sum(
            h.get("elapsed_s", 0) for h in (d.get("reharden_history") or [])[-1:]
        )
        lines.append(
            f"| {recipe} | {method} | {ci(s['asr'])} | {delta:+.1f} pp "
            f"| {pr['broken']} / {pr['fixed']} (p={pr['p']:.2g}) | {pct(s['best_of_k']['rate'])} "
            f"| {pct(s['utility']['rate'])} | {d['heldout']['after']:.2f} | {secs / 60:.0f} min |"
        )
    return lines


TABLES = {
    "hardening": t_hardening,
    "erosion": t_erosion,
    "mitigation": t_mitigation,
    "gate": t_gate,
    "gcg": t_gcg,
}


def main() -> None:
    data = load_results(RESULTS)
    text = README.read_text()
    for name, fn in TABLES.items():
        pattern = re.compile(rf"(<!-- TABLE:{name} -->\n).*?(<!-- /TABLE:{name} -->)", re.S)
        if pattern.search(text):
            block = "\n".join(fn(data))
            text = pattern.sub(lambda m, b=block: m.group(1) + b + "\n" + m.group(2), text)
    README.write_text(text)
    print("README tables updated")


if __name__ == "__main__":
    main()
