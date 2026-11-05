import argparse

import pytest

from injdrift import gate
from injdrift.cli import build_parser, main
from injdrift.gate import Thresholds, decide, to_markdown


def _rows(successes, n_clean=4, correct=4, prefix="it"):
    rows = []
    for i, s in enumerate(successes):
        rows.append(
            {
                "key": f"k{i}",
                "item": f"{prefix}{i % 10}",
                "attack": "naive",
                "position": "end",
                "success_begin": s,
            }
        )
    for j in range(n_clean):
        rows.append(
            {
                "key": f"c{j}",
                "item": f"{prefix}{j}",
                "attack": "none",
                "position": "-",
                "correct": j < correct,
            }
        )
    return rows


def test_gate_passes_identical_models():
    rows = _rows([False] * 40)
    res = decide(rows, rows, Thresholds(), ref_logprobs=[-5.0], cand_logprobs=[-5.0])
    assert res.passed and res.status == "pass"
    assert "PASS" in to_markdown(res, reference="r", candidate="c")


def test_gate_fails_on_significant_regression():
    ref = _rows([False] * 40)
    cand = _rows([i < 12 for i in range(40)])  # 30% ASR, 12 newly broken
    res = decide(ref, cand, Thresholds())
    by = {c.name: c.status for c in res.checks}
    assert by["asr_absolute"] == "fail" and by["asr_regression"] == "fail"
    assert res.metrics["newly_broken"] == 12 and res.metrics["mcnemar_p"] < 0.001


def test_gate_tolerates_noise_below_threshold():
    ref = _rows([i == 0 for i in range(40)])
    cand = _rows([i == 1 for i in range(40)])  # one broken, one fixed
    res = decide(ref, cand, Thresholds(max_asr=0.1))
    assert res.passed


def test_gate_warns_on_logprob_drift_and_utility():
    rows = _rows([False] * 40)
    cand = _rows([False] * 40, correct=1)
    res = decide(rows, cand, Thresholds(), ref_logprobs=[-6.0, -6.0], cand_logprobs=[-2.0, -2.0])
    by = {c.name: c.status for c in res.checks}
    assert by["canary_logprob"] == "warn" and by["utility"] == "warn"
    assert res.passed and res.status == "warn"


def test_gate_requires_shared_cases():
    with pytest.raises(ValueError):
        decide(
            _rows([False], prefix="a"),
            [{"key": "zz", "attack": "none", "position": "-", "item": "x"}],
            Thresholds(),
        )


def _gate_args(tmp_path, **kw):
    p = argparse.ArgumentParser()
    gate.add_arguments(p)
    argv = ["--reference", "ref", "--candidate", "cand", "--out-dir", str(tmp_path)]
    for k, v in kw.items():
        argv += [f"--{k}", str(v)]
    return p.parse_args(argv)


def test_gate_main_exit_codes(tmp_path, monkeypatch):
    good, bad = _rows([False] * 40), _rows([True] * 40)
    outputs = {"ref": (good, None), "cand": (bad, None)}
    monkeypatch.setattr(gate, "_evaluate", lambda m, a, p, f, lp: outputs[m])
    assert gate.main(_gate_args(tmp_path)) == 1
    assert (tmp_path / "gate.md").read_text().startswith("## injdrift gate: **FAIL**")
    outputs["cand"] = (good, None)
    assert gate.main(_gate_args(tmp_path)) == 0

    def boom(*_a):
        raise RuntimeError("no model")

    monkeypatch.setattr(gate, "_evaluate", boom)
    assert gate.main(_gate_args(tmp_path)) == 2


def test_cli_suites_and_grid_dry_run(tmp_path, capsys):
    assert main(["suites"]) == 0
    assert "hacked" in capsys.readouterr().out
    cfg = tmp_path / "grid.yaml"
    cfg.write_text(
        "results: " + str(tmp_path / "res") + "\n"
        "bases: {b: some/model}\n"
        "evals: [{id: e1, model: b}]\n"
        "defaults: {n: 16}\n"
        "runs: [{id: r1, base: b, recipe: medical, peft: full}]\n"
    )
    assert main(["grid", str(cfg), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "run e1: eval --model some/model" in out
    assert "--peft full" in out and "--n 16" in out
    assert build_parser().parse_args(["eval", "--model", "m"]).profile == "standard"
