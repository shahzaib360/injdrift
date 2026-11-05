"""Training data for injection hardening, built the way StruQ / SecAlign build it.

From Alpaca-cleaned samples that carry an input, we inject the instruction of another
(self-contained) sample into the data and pair:

* **chosen**   -- the response to the *real* instruction (ignore the injection),
* **rejected** -- the response to the *injected* instruction (obey it).

SecAlign optimizes this preference with DPO; StruQ does supervised fine-tuning on the
chosen side (mixed with clean samples). Following Meta-SecAlign, the injection is
placed at a random end of the data. Injection styles are "naive" (append) and
"completion" (fake answer + fake instruction) using :data:`TRAIN_DELIMITERS`, which
never overlap with the evaluation delimiters.

Alpaca instructions are split by a stable hash into ``harden`` / ``anchor`` / ``probe``
partitions, so hardening data, robustness anchors used during downstream fine-tuning,
and probe strings never share an instruction. Anything that also appears in the
AlpacaFarm evaluation set is excluded.
"""

from __future__ import annotations

import random
from functools import lru_cache

from injdrift.attacks.static import TRAIN_DELIMITERS, fake_response, inject
from injdrift.data import sources
from injdrift.formats import Message, build_messages
from injdrift.probes.separation import partition
from injdrift.train.examples import PrefRecord


@lru_cache(maxsize=4)
def _pools(part: str) -> tuple[list[dict], list[dict]]:
    banned = {r["instruction"].strip() for r in sources.alpaca_farm_eval()}
    with_input, without_input = [], []
    for r in sources.alpaca_cleaned():
        ins = r["instruction"].strip()
        if ins in banned or partition(ins) != part:
            continue
        out = r["output"].strip()
        if not out or len(out) > 500 or len(ins) > 300:
            continue
        if r["input"].strip():
            if len(r["input"]) <= 500:
                with_input.append(r)
        elif len(ins) <= 160 and len(out) <= 400:
            without_input.append(r)
    return with_input, without_input


def injected_prompt(
    sample: dict, injected: dict, rng: random.Random, *, fmt: str = "secure"
) -> tuple[Message, ...]:
    style = rng.choice(("naive", "completion"))
    if style == "naive":
        payload = injected["instruction"]
    else:
        resp, inst = rng.choice(TRAIN_DELIMITERS)
        payload = (
            f"\n\n{resp}\n{fake_response(sample['output'])}\n\n{inst}\n{injected['instruction']}"
        )
    pieces = inject(sample["input"], payload, rng.choice(("end", "start")))
    return tuple(build_messages(sample["instruction"], pieces, fmt=fmt))


def secalign_pairs(
    n: int, *, seed: int = 0, part: str = "harden", fmt: str = "secure"
) -> list[PrefRecord]:
    """Preference pairs for SecAlign-style DPO."""
    with_input, without_input = _pools(part)
    rng = random.Random(f"secalign-{part}-{seed}")
    picks = rng.sample(with_input, min(n, len(with_input)))
    out = []
    for s in picks:
        inj = rng.choice(without_input)
        out.append(PrefRecord(injected_prompt(s, inj, rng, fmt=fmt), s["output"], inj["output"]))
    return out


def struq_conversations(
    n: int, *, seed: int = 0, part: str = "harden", fmt: str = "secure", clean_frac: float = 0.5
) -> list[list[Message]]:
    """StruQ-style SFT: injected prompts answered with the real task, plus clean ones."""
    with_input, without_input = _pools(part)
    rng = random.Random(f"struq-{part}-{seed}")
    picks = rng.sample(with_input, min(n, len(with_input)))
    convs = []
    for s in picks:
        if rng.random() < clean_frac:
            msgs = build_messages(s["instruction"], s["input"], fmt=fmt)
        else:
            msgs = list(injected_prompt(s, rng.choice(without_input), rng, fmt=fmt))
        convs.append(msgs + [Message.of("assistant", s["output"])])
    return convs
