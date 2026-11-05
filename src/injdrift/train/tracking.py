"""Mid-training measurements: is robustness eroding *while* the fine-tune runs?

The tracker is deliberately cheap (a few dozen prompts) so it can run several times
inside one fine-tune. It records greedy-decoding ASR, the teacher-forced log-prob of
the injected reply (a smooth signal that moves before greedy outputs flip), and the
separation probes.
"""

from __future__ import annotations

import math

from injdrift.evaluate.cases import build_plan
from injdrift.evaluate.runner import canary_logprobs, evaluate, summarize
from injdrift.modeling import Loaded
from injdrift.probes.separation import ProbeSet, RoleProbe, run_probes


class Tracker:
    def __init__(
        self,
        loaded: Loaded,
        *,
        fmt: str = "secure",
        profile: str = "track",
        probe_set: ProbeSet | None = None,
        fixed_probe: RoleProbe | None = None,
        max_new_tokens: int = 16,
    ) -> None:
        self.fmt = fmt
        self.cases = build_plan(profile, loaded.dialect)
        self.probe_set = probe_set
        self.fixed_probe = fixed_probe
        self.max_new_tokens = max_new_tokens

    def __call__(self, loaded: Loaded, step: int) -> dict:
        rows = evaluate(loaded, self.cases, fmt=self.fmt, max_new_tokens=self.max_new_tokens)
        s = summarize(rows)
        lps = [x for x in canary_logprobs(loaded, self.cases, fmt=self.fmt) if not math.isnan(x)]
        out = {
            "asr": s["asr"]["rate"],
            "asr_in": s["asr_in"]["rate"],
            "by_attack": {k: v["rate"] for k, v in s["by_attack"].items()},
            "canary_logprob": sum(lps) / len(lps) if lps else math.nan,
        }
        if self.probe_set is not None:
            probes, fitted = run_probes(loaded, self.probe_set, self.fixed_probe)
            if self.fixed_probe is None:
                self.fixed_probe = fitted
            out["probes"] = probes
        return out
