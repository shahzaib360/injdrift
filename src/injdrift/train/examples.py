"""Turn chat records into token-level training examples.

* :class:`SFTExample` -- token ids plus a mask of positions that receive loss (the
  assistant's tokens and its end-of-turn marker).
* :class:`PrefExample` -- a shared prompt and two continuations (chosen / rejected)
  for DPO-style preference losses.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from injdrift.formats import LOSS_TAGS, Dialect, Message, encode, render


@dataclass
class SFTExample:
    input_ids: list[int]
    loss_mask: list[bool]

    @property
    def n_loss(self) -> int:
        return sum(self.loss_mask)


@dataclass
class PrefExample:
    prompt_ids: list[int]
    chosen_ids: list[int]
    rejected_ids: list[int]


@dataclass(frozen=True)
class PrefRecord:
    """Prompt messages (ending with a user / input turn) plus two candidate replies."""

    prompt: tuple[Message, ...]
    chosen: str
    rejected: str


def sft_example(
    tokenizer, dialect: Dialect, messages: Sequence[Message], strip: Sequence[str], max_len: int
) -> SFTExample | None:
    enc = encode(tokenizer, render(messages, dialect, add_generation_prompt=False, strip=strip))
    mask = enc.mask(LOSS_TAGS)
    if len(enc.input_ids) > max_len or not any(mask):
        return None
    return SFTExample(enc.input_ids, mask)


def pref_example(
    tokenizer,
    dialect: Dialect,
    record: PrefRecord,
    strip: Sequence[str],
    max_len: int,
    max_response: int = 160,
) -> PrefExample | None:
    """Encode both conversations and split them at the start of the assistant turn.

    Continuations longer than ``max_response`` tokens are cut (keeping the cut
    identical in spirit for both sides); examples whose prompt plus longest
    continuation exceed ``max_len`` are dropped.
    """
    sides = []
    for reply in (record.chosen, record.rejected):
        msgs = list(record.prompt) + [Message.of("assistant", reply)]
        enc = encode(tokenizer, render(msgs, dialect, add_generation_prompt=False, strip=strip))
        resp, eot = enc.token_spans.get("response"), enc.token_spans.get("eot")
        if not resp or not eot:
            return None
        start, end = resp[-1][0], eot[-1][1]
        sides.append((enc.input_ids[:start], enc.input_ids[start:end]))
    (p1, chosen), (p2, rejected) = sides
    if p1 != p2:
        return None
    chosen, rejected = chosen[:max_response], rejected[:max_response]
    if len(p1) + max(len(chosen), len(rejected)) > max_len:
        return None
    return PrefExample(p1, chosen, rejected)
