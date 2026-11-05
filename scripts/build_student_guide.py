"""Build docs/injdrift-student-guide.pdf from the template and the pilot results.

Every number in the guide is computed here from results/pilot/*.json and the per-case
rows, so the PDF can be regenerated whenever the results change::

    python scripts/build_student_guide.py            # needs Chrome/Chromium on PATH
    CHROME=/path/to/chrome python scripts/build_student_guide.py
"""

from __future__ import annotations

import html
import json
import math
import os
import shutil
import subprocess
from pathlib import Path

import jinja2

from injdrift.gate import Thresholds, decide
from injdrift.report import (
    _read_rows,
    base_eval,
    load_results,
    model_of,
    paired,
    probe_signal_rows,
    run_label,
)
from injdrift.stats import pearson

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "pilot"
GUIDE = ROOT / "docs" / "student_guide"
OUT_PDF = ROOT / "docs" / "injdrift-student-guide.pdf"
MODEL_NAMES = {"qwen": "Qwen2.5-0.5B", "smol": "SmolLM2-360M"}


def pct(x: float | None, d: int = 1) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{100 * x:.{d}f}%"


def ci(rate: dict) -> str:
    lo, hi = rate["ci"]
    return f"{pct(rate['rate'])} <span class='small'>[{100 * lo:.0f}–{100 * hi:.0f}]</span>"


def table(
    header: list[str],
    rows: list[list[str]],
    num_cols: set[int] = frozenset(),
    base_rows=(),
    first_width: str | None = None,
    cls: str = "",
) -> str:
    th = "".join(
        f"<th class='{'num' if i in num_cols else ''}'>{h}</th>" for i, h in enumerate(header)
    )
    cols = f"<colgroup><col style='width:{first_width}'></colgroup>" if first_width else ""
    out = [f"<table class='{cls}'>{cols}<tr>{th}</tr>"]
    for k, r in enumerate(rows):
        cls = " class='base'" if k in base_rows else ""
        tds = "".join(
            f"<td class='{'num' if i in num_cols else ''}'>{c}</td>" for i, c in enumerate(r)
        )
        out.append(f"<tr{cls}>{tds}</tr>")
    out.append("</table>")
    return "".join(out)


def track_summary(t: dict, key: str) -> float:
    return t["probes"]["summary"][key]["upper_mean"]


def build_context() -> dict:
    data = load_results(RESULTS)
    ev = data["evals"]

    # ------------------------------------------------------------- hardening
    hard_rows = []
    order = [
        ("qwen", "undefended", "undefended"),
        ("qwen", "secalign", "mini-SecAlign, 400 pairs"),
        ("qwen", "secalign1600", "mini-SecAlign, 1,600 pairs (ablation)"),
        ("smol", "undefended", "undefended"),
        ("smol", "secalign", "mini-SecAlign, 400 pairs"),
    ]
    for model, key, label in order:
        d = ev.get(f"eval-{model}-{key}")
        if not d:
            continue
        for fmt in ("secure", "naive"):
            s = d["eval"][fmt]
            hard_rows.append(
                [
                    MODEL_NAMES[model],
                    label,
                    fmt,
                    ci(s["asr"]),
                    pct(s["best_of_k"]["rate"]),
                    pct(s["utility"]["rate"]),
                ]
            )
    hardening_table = table(
        ["model", "defense", "data at test time", "static ASR [95% CI]", "best-of-K", "utility"],
        hard_rows,
        num_cols={3, 4, 5},
    )

    def summ(eid: str, fmt: str = "secure") -> dict:
        s = ev[eid]["eval"][fmt]
        return {
            "asr": pct(s["asr"]["rate"]),
            "bestk": pct(s["best_of_k"]["rate"]),
            "utility": pct(s["utility"]["rate"]),
            "asr_raw": s["asr"]["rate"],
        }

    ctx: dict = {
        "repo": "https://github.com/shahzaib360/injdrift",
        "date": "November 2026",
        "harden_pairs": 400,
        "hardening_table": hardening_table,
        "main": summ("eval-qwen-secalign") if "eval-qwen-secalign" in ev else {},
        "abl": summ("eval-qwen-secalign1600") if "eval-qwen-secalign1600" in ev else {},
        "evals": {eid: summ(eid) for eid in ev},
        "evals_naive": {eid: summ(eid, "naive") for eid in ev},
    }

    # --------------------------------------------------------------- erosion
    runs: dict[str, dict] = {}
    erosion_tables = []
    for model in ("qwen", "smol"):
        base, base_rows = base_eval(data, model)
        rows, keys = [], []
        if base:
            rows.append(
                [
                    "hardened base",
                    ci(base["asr"]),
                    "—",
                    "—",
                    "—",
                    "—",
                    pct(base["utility"]["rate"]),
                    "—",
                ]
            )
        for rid, d in sorted(data["runs"].items()):
            if model_of(rid) != model:
                continue
            s = d["eval"]["secure"]
            pr = paired(base_rows, _read_rows(RESULTS / "rows" / f"{rid}.jsonl.gz"))
            t0, t1 = d["tracking"][0], d["tracking"][-1]
            delta = 100 * (s["asr"]["rate"] - base["asr"]["rate"]) if base else math.nan
            h = d["heldout"]
            info = {
                "label": run_label(d["spec"]),
                "asr": pct(s["asr"]["rate"]),
                "asr_raw": s["asr"]["rate"],
                "delta_pp": delta,
                "bestk": pct(s["best_of_k"]["rate"]),
                "utility": pct(s["utility"]["rate"]),
                "broken": pr["broken"],
                "fixed": pr["fixed"],
                "p": pr["p"],
                "lp0": t0["canary_logprob"],
                "lp1": t1["canary_logprob"],
                "d0": track_summary(t0, "fixed_dprime"),
                "d1": track_summary(t1, "fixed_dprime"),
                "drift0": track_summary(t0, "drift_ratio"),
                "drift1": track_summary(t1, "drift_ratio"),
                "track_asr": [x["asr"] for x in d["tracking"]],
                "task_before": h["before"],
                "task_after": h["after"],
                "task_metric": h["metric"],
                "spec": d["spec"],
            }
            runs[rid] = info
            rows.append(
                [
                    html.escape(info["label"]),
                    ci(s["asr"]),
                    f"{delta:+.1f} pp",
                    f"{pr['broken']} / {pr['fixed']} <span class='small'>(p={pr['p']:.2g})</span>",
                    f"{info['lp0']:.1f} → {info['lp1']:.1f}",
                    f"{info['d0']:.1f} → {info['d1']:.1f}",
                    info["utility"],
                    f"{h['before']:.2f} → {h['after']:.2f}",
                ]
            )
            keys.append(rid)
        if keys:
            erosion_tables.append(
                {
                    "title": f"{MODEL_NAMES[model]}: every fine-tune against its hardened base",
                    "html": table(
                        [
                            "fine-tune",
                            "ASR [95% CI]",
                            "Δ ASR",
                            "broken / fixed",
                            "reply log-prob",
                            "probe d′",
                            "utility",
                            "task loss",
                        ],
                        rows,
                        num_cols={1, 2, 4, 5, 6, 7},
                        base_rows={0} if base else set(),
                        first_width="24%",
                        cls="wide",
                    ),
                }
            )
    ctx["runs"] = runs
    ctx["erosion_tables"] = erosion_tables
    ctx["base_asr"] = {
        m: base_eval(data, m)[0]["asr"]["rate"] for m in ("qwen", "smol") if base_eval(data, m)[0]
    }

    # ------------------------------------------------- probes vs erosion
    sig = probe_signal_rows(data)
    corr_rows = []
    if len(sig) >= 3:
        for name in sig[0][2]:
            pairs = [(r[1], r[2][name]) for r in sig if not math.isnan(r[2][name])]
            r = pearson([a for a, _ in pairs], [b for _, b in pairs])
            corr_rows.append([name, f"{r:+.2f}", str(len(pairs))])
    ctx["probe_corr_table"] = (
        table(
            [
                "signal: change from hardened base to end of fine-tune",
                "correlation with Δ ASR",
                "runs",
            ],
            corr_rows,
            {1, 2},
        )
        if corr_rows
        else ""
    )
    ctx["probe_corr"] = {row[0]: float(row[1]) for row in corr_rows}

    # ----------------------------------------------------------- mitigation
    mit_rows = []
    groups = [
        (
            "dolly",
            [
                "qwen-dolly-lora",
                "qwen-dolly-lora-replay",
                "qwen-dolly-lora-reharden",
                "qwen-dolly-lora-secanchor",
            ],
        ),
        ("rag", ["qwen-rag-lora", "qwen-rag-secure-lora"]),
    ]
    for recipe, ids in groups:
        for rid in ids:
            r = runs.get(rid)
            if not r:
                continue
            method = {
                "qwen-dolly-lora": "plain fine-tune",
                "qwen-dolly-lora-replay": "replay",
                "qwen-dolly-lora-reharden": "re-harden",
                "qwen-dolly-lora-secanchor": "SecAnchor",
                "qwen-rag-lora": "plain fine-tune (naive format)",
                "qwen-rag-secure-lora": "secure format",
            }[rid]
            mit_rows.append(
                [
                    recipe,
                    method,
                    r["asr"],
                    f"{r['delta_pp']:+.1f} pp",
                    r["bestk"],
                    f"{r['lp1']:.1f}",
                    r["utility"],
                    f"{r['task_after']:.2f}",
                ]
            )
    ctx["mitigation_table"] = (
        table(
            [
                "recipe",
                "method",
                "static ASR",
                "Δ vs base",
                "best-of-K",
                "injected-reply log-prob",
                "utility",
                "task loss after",
            ],
            mit_rows,
            {2, 3, 4, 5, 6, 7},
        )
        if mit_rows
        else ""
    )

    # ------------------------------------------------------------------ gate
    gate_rows = []
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
            gate_rows.append(
                [
                    MODEL_NAMES[model].split("-")[0],
                    html.escape(run_label(d["spec"])),
                    f"<b>{res.status.upper()}</b>",
                    f"{pct(m['asr_reference'])} → {pct(m['asr_candidate'])}",
                    f"{m['newly_broken']} / {m['newly_fixed']}",
                    f"{m['mcnemar_p']:.2g}",
                ]
            )
    ctx["gate_table"] = (
        table(
            ["model", "fine-tune", "verdict", "ASR base → fine-tune", "broken / fixed", "p"],
            gate_rows,
            {3, 4, 5},
        )
        if gate_rows
        else ""
    )
    ctx["gate_counts"] = {
        "fail": sum("FAIL" in r[2] for r in gate_rows),
        "total": len(gate_rows),
    }

    # ------------------------------------------------------------------- GCG
    gcg = {}
    for p in sorted((RESULTS / "gcg").glob("*.json")) if (RESULTS / "gcg").exists() else []:
        d = json.loads(p.read_text())
        gcg[p.name.removesuffix(".json")] = {
            "asr": d["asr"],
            "n": d["n"],
            "k": round(d["asr"] * d["n"]),
            "kind": d["kind"],
        }
    ctx["gcg"] = gcg

    ctx["flip"] = (
        flip_example(data, "qwen", "qwen-dolly-lora") if "qwen-dolly-lora" in data["runs"] else None
    )

    # --------------------------------------------------------------- figures
    figs = RESULTS / "figures"

    def fig(name: str) -> str:
        p = figs / f"{name}-light.png"
        return p.as_uri() if p.exists() else ""

    ctx["fig"] = {
        "hardening": fig("hardening"),
        "erosion": fig("erosion"),
        "traj_qwen": fig("trajectories-qwen-grid"),
        "traj_smol": fig("trajectories-smol-grid"),
        "mitigation": fig("mitigation-qwen"),
    }
    return ctx


def flip_example(data: dict, model: str, run_id: str) -> dict | None:
    """A prompt the hardened base resisted and the fine-tune obeyed, rebuilt verbatim."""
    from injdrift.evaluate.cases import build_plan
    from injdrift.formats import CHATML

    _, base_rows = base_eval(data, model)
    base = {r["key"]: r for r in base_rows if r.get("fmt", "secure") == "secure"}
    rows = _read_rows(RESULTS / "rows" / f"{run_id}.jsonl.gz")
    flips = [
        r
        for r in rows
        if r["key"] in base
        and r["attack"] != "none"
        and r["success_begin"]
        and not base[r["key"]]["success_begin"]
        and not base[r["key"]]["success_in"]  # the base did not even mention the canary
    ]
    if not flips:
        return None
    cases = {c.key: c for c in build_plan("standard", CHATML)}
    rank = {"naive": 0, "ignore": 1, "escape": 2}

    def score(r: dict) -> tuple:
        c = cases[r["key"]]
        return (r["suite"] != "alpacafarm", rank.get(r["attack"], 3), len(c.item.data))

    best = min(flips, key=score)
    case = cases[best["key"]]
    data_html = "".join(
        f"<span class='hl'>{html.escape(t)}</span>" if tag == "injection" else html.escape(t)
        for t, tag in case.pieces
    )
    return {
        "key": best["key"],
        "attack": best["attack"],
        "instruction": html.escape(case.item.instruction),
        "data_html": data_html,
        "base_response": html.escape(base[best["key"]]["response"].strip()),
        "ft_response": html.escape(best["response"].strip()),
        "run_label": run_label(data["runs"][run_id]["spec"]),
        "canary": case.injected.reply,
    }


def chrome_binary() -> str:
    for cand in (
        os.environ.get("CHROME"),
        "google-chrome",
        "chromium",
        "chromium-browser",
        "chrome",
    ):
        if cand and shutil.which(cand):
            return shutil.which(cand)
    raise SystemExit("Chrome/Chromium not found; set CHROME=/path/to/chrome")


def main() -> None:
    ctx = build_context()
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(GUIDE), undefined=jinja2.StrictUndefined
    )
    findings = (
        env.get_template("findings.html.j2").module
        if (GUIDE / "findings.html.j2").exists()
        else None
    )
    if findings is not None:
        for name in (
            "tldr_html",
            "r_hardening",
            "r_erosion",
            "r_tracking",
            "r_probes",
            "r_gcg",
            "r_mitigation",
            "r_gate",
        ):
            ctx[name] = getattr(findings, name)(ctx) if hasattr(findings, name) else ""
    page = env.get_template("guide.html.j2").render(**ctx)
    out_html = GUIDE / "guide.html"
    out_html.write_text(page)
    subprocess.run(
        [
            chrome_binary(),
            "--headless",
            "--no-sandbox",
            "--disable-gpu",
            "--no-pdf-header-footer",
            f"--print-to-pdf={OUT_PDF}",
            out_html.as_uri(),
        ],
        check=True,
        capture_output=True,
    )
    print(f"wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
