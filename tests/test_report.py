import json

import pytest

from injdrift import report
from injdrift.data.suites import TaskItem
from injdrift.pipeline import write_json, write_rows


def _summary(asr: float) -> dict:
    rate = {"rate": asr, "k": int(asr * 100), "n": 100, "ci": [max(0, asr - 0.05), asr + 0.05]}
    return {
        "asr": rate,
        "asr_in": rate,
        "best_of_k": rate,
        "utility": {"rate": 0.8, "k": 8, "n": 10, "ci": [0.5, 0.9]},
        "by_attack": {"naive": rate, "combined": rate},
        "by_suite": {"alpacafarm": rate},
    }


def _rows(success: list[bool]) -> list[dict]:
    return [
        {
            "key": f"alpacafarm/{i:03d}/naive/end/hacked",
            "item": f"alpacafarm/{i:03d}",
            "suite": "alpacafarm",
            "attack": "naive",
            "position": "end",
            "success_begin": s,
            "fmt": "secure",
        }
        for i, s in enumerate(success)
    ]


def _track(frac: float, asr: float) -> dict:
    s = {"upper_mean": 0.9, "max": 1.0}
    probes = {
        "summary": {k: s for k in ("fixed_auroc", "fixed_dprime", "refit_acc", "drift_ratio")}
    }
    return {"frac": frac, "asr": asr, "canary_logprob": -5 + asr, "probes": probes}


@pytest.fixture
def results(tmp_path, monkeypatch):
    d = tmp_path / "pilot"
    for name, asr, succ in [
        ("undefended", 0.6, [True] * 6 + [False] * 4),
        ("secalign", 0.02, [False] * 10),
    ]:
        write_json(
            d / f"eval-qwen-{name}.json",
            {"model": name, "profile": "standard", "eval": {"secure": _summary(asr)}},
        )
        write_rows(d / "rows" / f"eval-qwen-{name}.jsonl.gz", _rows(succ))
    for rid, recipe, fmt, asr in [
        ("qwen-medical-lora", "medical", "naive", 0.10),
        ("qwen-rag-lora", "rag", "naive", 0.30),
        ("qwen-rag-secure-lora", "rag", "secure", 0.05),
    ]:
        spec = {"recipe": recipe, "fmt": fmt, "peft": "lora", "preserve": "none", "reharden": 0}
        rec = {
            "spec": spec,
            "recipe": {"name": recipe, "kind": "sft"},
            "heldout": {"before": 2.0, "after": 1.5, "metric": "nll"},
            "tracking": [_track(0, 0.02), _track(1 / 3, asr / 2), _track(1, asr)],
            "final_attention": {"summary": {"upper_mean": 1.2}},
            "eval": {"secure": _summary(asr)},
        }
        write_json(d / f"{rid}.json", rec)
        write_rows(d / "rows" / f"{rid}.jsonl.gz", _rows([i < int(asr * 10) for i in range(10)]))
    hist = [{"step": 1, "loss": 0.7}, {"track": {"frac": 0.5, "asr": 0.1}}]
    write_json(
        d / "harden-qwen0.5b-secalign.json", {"method": "secalign", "base": "Qwen", "history": hist}
    )
    fake_pool = [TaskItem("alpacafarm", f"{i:03d}", "x", "y") for i in range(40)]
    monkeypatch.setattr("injdrift.evaluate.cases.suite_pool", lambda suite, seed=0: fake_pool)
    return d


def test_report_tables_and_figures(results, tmp_path):
    path = report.build_report(results, tmp_path / "out")
    md = path.read_text()
    assert "## Hardening" in md and "qwen: erosion by fine-tuning recipe" in md
    assert "rag (secure fmt) · LoRA" in md
    assert "+28.0 pp" in md  # 30% vs 2% hardened base
    figs = {p.name for p in (tmp_path / "out" / "figures").iterdir()}
    for name in ("erosion", "hardening", "trajectories-qwen"):
        assert f"{name}-light.png" in figs and f"{name}-dark.png" in figs


def test_paired_counts():
    base = _rows([False, True, False, False])
    cand = _rows([True, False, True, False])
    p = report.paired(base, cand)
    assert (p["broken"], p["fixed"]) == (2, 1)
    assert json.dumps(p)
