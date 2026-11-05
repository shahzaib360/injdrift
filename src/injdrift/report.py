"""Tables (Markdown) and figures (PNG, light + dark) from a results directory.

Everything is computed from the JSON records and the gzipped per-case rows written by
:mod:`injdrift.pipeline`, so the report can be regenerated at any time with
``injdrift report results/pilot --out results/pilot``.
"""

from __future__ import annotations

import gzip
import json
import math
from collections import defaultdict
from pathlib import Path

from injdrift.stats import mcnemar_one_sided
from injdrift.train.recipes import RECIPES

# Reference palette (validated: adjacent CVD dE >= 9.1 light / 8.4 dark).
THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "text": "#0b0b0b",
        "text2": "#52514e",
        "grid": "#e4e3df",
        "ref": "#8a8984",
        "series": [
            "#2a78d6",
            "#eb6834",
            "#1baf7a",
            "#eda100",
            "#e87ba4",
            "#008300",
            "#6250d6",
            "#e34948",
        ],
    },
    "dark": {
        "surface": "#1a1a19",
        "text": "#f0efec",
        "text2": "#c3c2b7",
        "grid": "#33332f",
        "ref": "#8f8e88",
        "series": [
            "#3987e5",
            "#d95926",
            "#199e70",
            "#c98500",
            "#d55181",
            "#008300",
            "#9085e9",
            "#e66767",
        ],
    },
}

PRESERVE_LABEL = {"none": "", "replay": " + replay", "secanchor": " + SecAnchor"}


# ----------------------------------------------------------------------------- loading


def _read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with gzip.open(path, "rt") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_results(results: Path) -> dict:
    runs, evals, hardens = {}, {}, {}
    for p in sorted(results.glob("*.json")):
        d = json.loads(p.read_text())
        if "spec" in d:
            runs[p.stem] = d
        elif "eval" in d and "profile" in d:
            evals[p.stem] = d
        elif "method" in d and "history" in d:
            hardens[p.stem] = d
    return {"runs": runs, "evals": evals, "hardens": hardens, "dir": results}


def model_of(run_id: str) -> str:
    return run_id.split("-")[0]


def run_label(spec: dict) -> str:
    fmt = " (secure fmt)" if spec["fmt"] == "secure" else ""
    peft = "LoRA" if spec["peft"] == "lora" else "full FT"
    extra = PRESERVE_LABEL.get(spec["preserve"], "")
    if spec.get("reharden"):
        extra += " + re-harden"
    return f"{spec['recipe']}{fmt} · {peft}{extra}"


def _pct(x: float | None, digits: int = 1) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{100 * x:.{digits}f}%"


def _rate_ci(d: dict) -> str:
    if not d or not d.get("n"):
        return "n/a"
    lo, hi = d["ci"]
    return f"{_pct(d['rate'])} [{_pct(lo, 0)}–{_pct(hi, 0)}]"


def paired(base_rows: list[dict], rows: list[dict]) -> dict:
    """Newly broken / newly fixed attacked cases relative to the base, McNemar p."""
    base = {
        r["key"]: r
        for r in base_rows
        if r.get("attack") != "none" and r.get("fmt", "secure") == "secure"
    }
    broken = fixed = 0
    for r in rows:
        b = base.get(r["key"])
        if b is None or r.get("attack") == "none" or r.get("position") != "end":
            continue
        if r["success_begin"] and not b["success_begin"]:
            broken += 1
        elif b["success_begin"] and not r["success_begin"]:
            fixed += 1
    return {"broken": broken, "fixed": fixed, "p": mcnemar_one_sided(broken, fixed)}


def _track_asr(rows: list[dict], keys: set[str]) -> float:
    sel = [r for r in rows if r["key"] in keys and r.get("fmt", "secure") == "secure"]
    return sum(bool(r["success_begin"]) for r in sel) / len(sel) if sel else math.nan


# ------------------------------------------------------------------------------ tables


def table_hardening(data: dict) -> list[str]:
    lines = [
        "| model | defense | data format at test time | ASR (static) [95% CI] "
        "| best-of-K | utility |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for eid, d in sorted(data["evals"].items()):
        parts = eid.split("-")  # eval-<model>-<defense>
        model, defense = parts[1], "-".join(parts[2:])
        for fmt, s in d["eval"].items():
            lines.append(
                f"| {model} | {defense} | {fmt} | {_rate_ci(s['asr'])} "
                f"| {_pct(s['best_of_k']['rate'])} | {_pct(s['utility']['rate'])} |"
            )
    return lines


def base_eval(data: dict, model: str) -> tuple[dict | None, list[dict]]:
    eid = f"eval-{model}-secalign"
    d = data["evals"].get(eid)
    if not d:
        return None, []
    return d["eval"].get("secure"), _read_rows(data["dir"] / "rows" / f"{eid}.jsonl.gz")


def table_erosion(data: dict, model: str) -> list[str]:
    base, base_rows = base_eval(data, model)
    lines = [
        "| fine-tune | ASR after [95% CI] | Δ vs hardened base | newly broken / fixed (McNemar p) "
        "| best-of-K | utility | task loss before → after |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    if base:
        lines.append(
            f"| *hardened base (no fine-tune)* | {_rate_ci(base['asr'])} | — | — "
            f"| {_pct(base['best_of_k']['rate'])} | {_pct(base['utility']['rate'])} | — |"
        )
    for rid, d in sorted(data["runs"].items()):
        if model_of(rid) != model:
            continue
        s = d["eval"]["secure"]
        rows = _read_rows(data["dir"] / "rows" / f"{rid}.jsonl.gz")
        pr = paired(base_rows, rows) if base_rows else None
        delta = f"{100 * (s['asr']['rate'] - base['asr']['rate']):+.1f} pp" if base else "n/a"
        pcell = f"{pr['broken']} / {pr['fixed']} (p={pr['p']:.2g})" if pr else "n/a"
        h = d["heldout"]
        task = f"{h['before']:.2f} → {h['after']:.2f}" + (
            " (NLL)" if h["metric"] == "nll" else " (pref. margin)"
        )
        lines.append(
            f"| {run_label(d['spec'])} | {_rate_ci(s['asr'])} | {delta} | {pcell} "
            f"| {_pct(s['best_of_k']['rate'])} | {_pct(s['utility']['rate'])} | {task} |"
        )
    return lines


def table_attacks(data: dict, model: str) -> list[str]:
    attacks = ["naive", "ignore", "escape", "completion", "completion_chat", "combined"]
    lines = ["| fine-tune | " + " | ".join(attacks) + " |", "|---" * (len(attacks) + 1) + "|"]
    base, _ = base_eval(data, model)
    if base:
        cells = [_pct(base["by_attack"].get(a, {}).get("rate"), 0) for a in attacks]
        lines.append("| *hardened base* | " + " | ".join(cells) + " |")
    for rid, d in sorted(data["runs"].items()):
        if model_of(rid) != model:
            continue
        by = d["eval"]["secure"]["by_attack"]
        cells = [_pct(by.get(a, {}).get("rate"), 0) for a in attacks]
        lines.append(f"| {run_label(d['spec'])} | " + " | ".join(cells) + " |")
    return lines


def _probe_cells(track: dict) -> list[str]:
    s = track.get("probes", {}).get("summary", {})
    get = lambda k: s.get(k, {}).get("upper_mean", math.nan)  # noqa: E731
    return [
        f"{get('fixed_auroc'):.3f}",
        f"{get('fixed_dprime'):.2f}",
        f"{get('refit_acc'):.3f}",
        f"{get('drift_ratio'):.2f}",
    ]


def table_probes(data: dict, model: str) -> list[str]:
    lines = [
        "| fine-tune | frozen-probe AUROC | frozen-probe d′ | re-fit probe acc. "
        "| drift ratio | attention share |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    base_done = False
    for rid, d in sorted(data["runs"].items()):
        if model_of(rid) != model:
            continue
        tracks = d["tracking"]
        if not base_done:
            att = tracks[0].get("attention", {}).get("summary", {}).get("upper_mean", math.nan)
            lines.append(
                "| *hardened base* | " + " | ".join(_probe_cells(tracks[0])) + f" | {att:.2f} |"
            )
            base_done = True
        att = d.get("final_attention", {}).get("summary", {}).get("upper_mean", math.nan)
        lines.append(
            f"| {run_label(d['spec'])} | "
            + " | ".join(_probe_cells(tracks[-1]))
            + f" | {att:.2f} |"
        )
    return lines


def table_tracking(data: dict, model: str) -> list[str]:
    lines = [
        "| fine-tune | ASR @0 | @⅓ | @⅔ | @1 | canary log-prob @0 → @1 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for rid, d in sorted(data["runs"].items()):
        if model_of(rid) != model:
            continue
        t = d["tracking"]
        asr = [_pct(x.get("asr"), 0) for x in t]
        while len(asr) < 4:
            asr.insert(-1, "—")
        lp = f"{t[0]['canary_logprob']:.2f} → {t[-1]['canary_logprob']:.2f}"
        lines.append(f"| {run_label(d['spec'])} | " + " | ".join(asr[:4]) + f" | {lp} |")
    return lines


# ----------------------------------------------------------------------------- figures


def _style(ax, th: dict) -> None:
    ax.set_facecolor(th["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(th["grid"])
        ax.spines[side].set_linewidth(1)
    ax.tick_params(colors=th["text2"], labelsize=9, length=0)
    ax.grid(True, color=th["grid"], linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(th["text2"])
    ax.yaxis.label.set_color(th["text2"])
    ax.title.set_color(th["text"])


def _figure(nrows: int, ncols: int, size: tuple[float, float], th: dict):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(nrows, ncols, figsize=size, squeeze=False)
    fig.patch.set_facecolor(th["surface"])
    for ax in axes.flat:
        _style(ax, th)
    return fig, axes


def _save(fig, out: Path, name: str, mode: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out / f"{name}-{mode}.png", dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor()
    )
    import matplotlib.pyplot as plt

    plt.close(fig)


def fig_erosion(data: dict, out: Path, mode: str) -> bool:
    models = sorted({model_of(r) for r in data["runs"]})
    if not models:
        return False
    th = THEMES[mode]
    groups = {
        "no data channel": 0,
        "data in user turn (naive)": 1,
        "data in input role (secure)": 2,
    }
    heights = [sum(model_of(r) == m for r in data["runs"]) for m in models]
    fig, axes = _figure(len(models), 1, (8.2, 1.2 + 0.42 * sum(heights) + 0.9 * len(models)), th)
    for ax, model in zip(axes[:, 0], models, strict=True):
        runs = [(rid, d) for rid, d in data["runs"].items() if model_of(rid) == model]
        runs.sort(key=lambda x: x[1]["eval"]["secure"]["asr"]["rate"])
        labels, vals, errs, colors = [], [], [[], []], []
        for _, d in runs:
            a = d["eval"]["secure"]["asr"]
            labels.append(run_label(d["spec"]))
            vals.append(100 * a["rate"])
            errs[0].append(100 * (a["rate"] - a["ci"][0]))
            errs[1].append(100 * (a["ci"][1] - a["rate"]))
            spec = d["spec"]
            if not RECIPES[spec["recipe"]].data_channel:
                g = "no data channel"
            else:
                g = (
                    "data in input role (secure)"
                    if spec["fmt"] == "secure"
                    else "data in user turn (naive)"
                )
            colors.append(th["series"][groups[g]])
        y = list(range(len(runs)))
        ax.barh(
            y,
            vals,
            height=0.62,
            color=colors,
            xerr=errs,
            ecolor=th["text2"],
            capsize=0,
            error_kw={"elinewidth": 1},
        )
        ax.set_yticks(y, labels, color=th["text"], fontsize=9)
        for yi, v, e in zip(y, vals, errs[1], strict=True):
            ax.text(v + e + 0.8, yi, f"{v:.1f}%", va="center", fontsize=8.5, color=th["text2"])
        refs = []
        base, _ = base_eval(data, model)
        if base:
            refs.append((100 * base["asr"]["rate"], "hardened base"))
        und = data["evals"].get(f"eval-{model}-undefended")
        if und:
            refs.append((100 * und["eval"]["secure"]["asr"]["rate"], "undefended"))
        for x, name in refs:
            ax.axvline(x, color=th["ref"], linewidth=1.2)
            ax.text(x + 0.6, len(runs) - 0.35, name, fontsize=8.5, color=th["text2"], va="bottom")
        ax.set_xlim(0, max([*vals, *(r[0] for r in refs), 10]) * 1.18 + 4)
        ax.set_ylim(-0.6, len(runs) + 0.4)
        ax.set_xlabel("static attack success rate (%), secure prompt format, 95% Wilson CI")
        ax.set_title(f"{model}: injection ASR after ordinary fine-tuning", loc="left", fontsize=11)
    from matplotlib.patches import Patch

    handles = [Patch(color=th["series"][i], label=g) for g, i in groups.items()]
    leg = fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.02),
    )
    for t in leg.get_texts():
        t.set_color(th["text"])
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _save(fig, out, "erosion", mode)
    return True


def _trajectory_runs(data: dict, model: str) -> list[tuple[str, dict]]:
    sel = [
        (rid, d)
        for rid, d in sorted(data["runs"].items())
        if model_of(rid) == model
        and d["spec"]["preserve"] == "none"
        and not d["spec"].get("reharden")
    ]
    return sel[:8]


def fig_trajectories(data: dict, out: Path, mode: str, model: str) -> bool:
    runs = _trajectory_runs(data, model)
    if not runs:
        return False
    th = THEMES[mode]
    panels = [
        ("asr", "ASR on tracking set (%)", lambda t: 100 * t["asr"]),
        ("canary", "log-prob of injected reply", lambda t: t["canary_logprob"]),
        (
            "drift",
            "injection drift ratio",
            lambda t: t["probes"]["summary"]["drift_ratio"]["upper_mean"],
        ),
        (
            "dprime",
            "frozen role-probe d′",
            lambda t: t["probes"]["summary"]["fixed_dprime"]["upper_mean"],
        ),
    ]
    fig, axes = _figure(1, 4, (15, 3.6), th)
    for ax, (_, ylabel, fn) in zip(axes[0], panels, strict=True):
        for i, (_, d) in enumerate(runs):
            xs = [t["frac"] for t in d["tracking"]]
            try:
                ys = [fn(t) for t in d["tracking"]]
            except (KeyError, TypeError):
                continue
            ax.plot(
                xs,
                ys,
                color=th["series"][i],
                linewidth=2,
                solid_capstyle="round",
                marker="o",
                markersize=4.5,
                markeredgecolor=th["surface"],
                markeredgewidth=1.5,
                label=run_label(d["spec"]),
            )
        ax.set_xlabel("fraction of fine-tuning")
        ax.set_title(ylabel, loc="left", fontsize=10)
        ax.set_xticks([0, 1 / 3, 2 / 3, 1], ["0", "⅓", "⅔", "1"])
    handles, labels = axes[0][0].get_legend_handles_labels()
    leg = fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(4, len(labels)),
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.12),
    )
    for t in leg.get_texts():
        t.set_color(th["text"])
    fig.suptitle(
        f"{model}: robustness and separation probes during fine-tuning",
        x=0.01,
        ha="left",
        color=th["text"],
        fontsize=11.5,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    _save(fig, out, f"trajectories-{model}", mode)
    return True


def fig_mitigation(data: dict, out: Path, mode: str, model: str) -> bool:
    by_recipe: dict[str, list[dict]] = defaultdict(list)
    for rid, d in data["runs"].items():
        if model_of(rid) == model and d["recipe"]["kind"] == "sft":
            by_recipe[(d["spec"]["recipe"], d["spec"]["peft"])].append(d)
    groups = {k: v for k, v in by_recipe.items() if len(v) >= 3}
    if not groups:
        return False
    th = THEMES[mode]
    fig, axes = _figure(1, len(groups), (5.2 * len(groups), 3.8), th)
    base, _ = base_eval(data, model)
    for ax, ((recipe, peft), runs) in zip(axes[0], sorted(groups.items()), strict=True):
        for d in runs:
            x, y = d["heldout"]["after"], 100 * d["eval"]["secure"]["asr"]["rate"]
            ax.plot(
                [x],
                [y],
                "o",
                color=th["series"][0],
                markersize=8,
                markeredgecolor=th["surface"],
                markeredgewidth=2,
            )
            name = (
                run_label(d["spec"])
                .split(" · ")[-1]
                .replace("LoRA", "plain")
                .replace("plain + ", "")
            )
            if d["spec"]["fmt"] == "secure":
                name += " (secure fmt)"
            ax.annotate(
                name,
                (x, y),
                xytext=(6, 4),
                textcoords="offset points",
                fontsize=8.5,
                color=th["text"],
            )
        if base:
            ax.axhline(100 * base["asr"]["rate"], color=th["ref"], linewidth=1.2)
            ax.text(
                ax.get_xlim()[0],
                100 * base["asr"]["rate"],
                " hardened base",
                fontsize=8.5,
                color=th["text2"],
                va="bottom",
            )
        ax.set_xlabel("held-out task loss after fine-tuning (NLL, lower = better task fit)")
        ax.set_ylabel("static ASR (%)")
        ax.set_title(f"{recipe} · {peft}: robustness vs task fit", loc="left", fontsize=10.5)
        ax.margins(x=0.25, y=0.2)
    fig.tight_layout()
    _save(fig, out, f"mitigation-{model}", mode)
    return True


def fig_hardening(data: dict, out: Path, mode: str) -> bool:
    if not data["hardens"]:
        return False
    th = THEMES[mode]
    fig, axes = _figure(1, 1, (6.2, 3.4), th)
    ax = axes[0][0]
    for i, (hid, h) in enumerate(sorted(data["hardens"].items())):
        tracks = [x["track"] for x in h["history"] if "track" in x]
        xs = [t["frac"] for t in tracks]
        ys = [100 * t["asr"] for t in tracks]
        und = data["evals"].get(f"eval-{_model_key(hid)}-undefended")
        hard = data["evals"].get(f"eval-{_model_key(hid)}-secalign")
        if und and hard:
            keys = _track_keys(data, _model_key(hid))
            und_rows = _read_rows(
                data["dir"] / "rows" / f"eval-{_model_key(hid)}-undefended.jsonl.gz"
            )
            hard_rows = _read_rows(
                data["dir"] / "rows" / f"eval-{_model_key(hid)}-secalign.jsonl.gz"
            )
            xs = [0.0, *xs, 1.0]
            ys = [100 * _track_asr(und_rows, keys), *ys, 100 * _track_asr(hard_rows, keys)]
        ax.plot(
            xs,
            ys,
            color=th["series"][i],
            linewidth=2,
            marker="o",
            markersize=4.5,
            markeredgecolor=th["surface"],
            markeredgewidth=1.5,
            label=f"{h['base']}",
        )
    ax.set_xlabel("fraction of mini-SecAlign training (DPO, LoRA)")
    ax.set_title("ASR during hardening (tracking set, secure format)", loc="left", fontsize=10.5)
    ax.set_ylabel("ASR (%)")
    leg = ax.legend(frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(th["text"])
    fig.tight_layout()
    _save(fig, out, "hardening", mode)
    return True


def _model_key(harden_id: str) -> str:
    return "qwen" if "qwen" in harden_id else "smol" if "smol" in harden_id else harden_id


def _track_keys(data: dict, model: str) -> set[str]:
    """Case keys of the 'track' profile, which is a subset of the 'standard' profile."""
    from injdrift.evaluate.cases import PROFILES, suite_pool

    rows = _read_rows(data["dir"] / "rows" / f"eval-{model}-secalign.jsonl.gz")
    block = PROFILES["track"][0]
    keep = {item.key for item in suite_pool(block.suite)[: block.n]}
    return {
        r["key"]
        for r in rows
        if r["item"] in keep and r["attack"] in block.attacks and r["position"] == "end"
    }


# ------------------------------------------------------------------------------ driver


def build_report(results: Path, out: Path) -> Path:
    data = load_results(results)
    models = sorted({model_of(r) for r in data["runs"]})
    md = ["# Pilot results (auto-generated by `injdrift report`)", ""]
    if data["evals"]:
        md += ["## Hardening", "", *table_hardening(data), ""]
    for model in models:
        md += [f"## {model}: erosion by fine-tuning recipe", "", *table_erosion(data, model), ""]
        md += [f"### {model}: ASR by attack after fine-tuning", "", *table_attacks(data, model), ""]
        md += [f"### {model}: tracked during fine-tuning", "", *table_tracking(data, model), ""]
        md += [
            f"### {model}: separation probes (upper-layer means)",
            "",
            *table_probes(data, model),
            "",
        ]
    out.mkdir(parents=True, exist_ok=True)
    path = out / "summary.md"
    path.write_text("\n".join(md) + "\n")
    figs = out / "figures"
    for mode in ("light", "dark"):
        fig_erosion(data, figs, mode)
        fig_hardening(data, figs, mode)
        for model in models:
            fig_trajectories(data, figs, mode, model)
            fig_mitigation(data, figs, mode, model)
    return path
