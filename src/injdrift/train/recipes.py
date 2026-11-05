"""Downstream fine-tuning recipes: what a company might do to a hardened model.

Each recipe is ordinary, benign data -- nothing in it mentions prompt injection.

| recipe          | kind | data channel | source                                   |
|-----------------|------|--------------|------------------------------------------|
| ``medical``     | SFT  | no           | medalpaca medical flashcards (domain QA) |
| ``dolly``       | SFT  | some         | databricks-dolly-15k (general tuning)    |
| ``rag``         | SFT  | yes          | neural-bridge rag-dataset-12000          |
| ``tool``        | SFT  | yes          | glaive function calling v2               |
| ``ultrafeedback`` | DPO | no          | UltraFeedback binarized (helpfulness)    |

Recipes with a data channel can be rendered in two formats. ``naive`` is what most
teams do (retrieved context / tool results inside the ordinary user turn, the way the
Qwen2.5 chat template renders tool results). ``secure`` keeps that content in the
``input`` role the hardened model was trained to distrust.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from dataclasses import dataclass

from injdrift.data import sources
from injdrift.formats import DEFAULT_SYSTEM, Message, build_messages
from injdrift.train.examples import PrefRecord


@dataclass(frozen=True)
class Recipe:
    name: str
    kind: str  # sft | dpo
    data_channel: bool
    description: str
    build: Callable[[int, int, str], list]


def _sample(rows: list, n: int, seed: int, salt: str) -> list:
    rng = random.Random(f"{salt}-{seed}")
    return rng.sample(rows, min(n, len(rows)))


def medical(n: int, seed: int, fmt: str) -> list[list[Message]]:
    rows = [r for r in sources.medical_flashcards() if 20 <= len(r["output"]) <= 600 and r["input"]]
    return [
        build_messages(r["input"], None, fmt=fmt, response=r["output"])
        for r in _sample(rows, n, seed, "medical")
    ]


def dolly(n: int, seed: int, fmt: str) -> list[list[Message]]:
    rows = [r for r in sources.dolly() if len(r["response"]) <= 800 and len(r["context"]) <= 1000]
    out = []
    for r in _sample(rows, n, seed, "dolly"):
        ctx = r["context"].strip() or None
        out.append(build_messages(r["instruction"], ctx, fmt=fmt, response=r["response"]))
    return out


RAG_TASK = "Answer the question using the provided context."


def rag(n: int, seed: int, fmt: str) -> list[list[Message]]:
    rows = [
        r
        for r in sources.rag12k()
        if isinstance(r["context"], str)
        and isinstance(r["answer"], str)
        and len(r["context"]) <= 1200
        and 0 < len(r["answer"]) <= 500
    ]
    out = []
    for r in _sample(rows, n, seed, "rag"):
        msgs = [Message.of("system", DEFAULT_SYSTEM)]
        if fmt == "secure":
            msgs.append(Message.of("user", f"{RAG_TASK}\n\nQuestion: {r['question']}"))
            msgs.append(Message.of("input", r["context"]))
        else:
            msgs.append(
                Message(
                    "user",
                    (
                        (f"{RAG_TASK}\n\nContext:\n", "instruction"),
                        (r["context"], "data"),
                        (f"\n\nQuestion: {r['question']}", "instruction"),
                    ),
                )
            )
        msgs.append(Message.of("assistant", r["answer"]))
        out.append(msgs)
    return out


def _parse_glaive(row: dict) -> dict | None:
    """First user -> function call -> function response -> answer exchange."""
    system = row["system"].removeprefix("SYSTEM:").strip()
    turns = [t.strip() for t in row["chat"].split("\n\n\n") if t.strip()]
    want = ("USER:", "ASSISTANT:", "FUNCTION RESPONSE:", "ASSISTANT:")
    if len(turns) < 4 or not all(t.startswith(w) for t, w in zip(turns, want, strict=False)):
        return None
    user, call, result, answer = (t.split(":", 1)[1].strip() for t in turns[:4])
    call = call.replace("<|endoftext|>", "").strip()
    answer = answer.replace("<|endoftext|>", "").strip()
    if "<functioncall>" not in call or not answer:
        return None
    try:
        functions = json.dumps(json.loads(system.split("-", 1)[1].strip()), separators=(",", ":"))
    except (ValueError, IndexError):
        return None
    return {"functions": functions, "user": user, "call": call, "result": result, "answer": answer}


def tool(n: int, seed: int, fmt: str) -> list[list[Message]]:
    parsed = []
    for row in sources.glaive_function_calling():
        if (
            "FUNCTION RESPONSE" not in row["chat"]
            or len(row["system"]) > 900
            or len(row["chat"]) > 1400
        ):
            continue
        p = _parse_glaive(row)
        if p:
            parsed.append(p)
    out = []
    for p in _sample(parsed, n, seed, "tool"):
        system = (
            "You are a helpful assistant with access to the following functions. "
            f"Use them if required.\n{p['functions']}"
        )
        msgs = [
            Message.of("system", system),
            Message.of("user", p["user"]),
            Message.of("assistant", p["call"]),
        ]
        if fmt == "secure":
            msgs.append(Message.of("input", p["result"]))
        else:
            msgs.append(
                Message(
                    "user",
                    (
                        ("<tool_response>\n", "sep"),
                        (p["result"], "tool_output"),
                        ("\n</tool_response>", "sep"),
                    ),
                )
            )
        msgs.append(Message.of("assistant", p["answer"]))
        out.append(msgs)
    return out


def ultrafeedback(n: int, seed: int, fmt: str) -> list[PrefRecord]:
    rows = []
    for r in sources.ultrafeedback_prefs():
        chosen, rejected = r["chosen"][-1]["content"], r["rejected"][-1]["content"]
        if (
            len(r["prompt"]) <= 400
            and len(chosen) <= 700
            and len(rejected) <= 700
            and r["score_chosen"] - r["score_rejected"] >= 1.0
        ):
            rows.append((r["prompt"], chosen, rejected))
    return [
        PrefRecord(tuple(build_messages(p, None, fmt=fmt)), c, rj)
        for p, c, rj in _sample(rows, n, seed, "ultrafeedback")
    ]


RECIPES: dict[str, Recipe] = {
    r.name: r
    for r in (
        Recipe("medical", "sft", False, "domain SFT on medical flashcard QA", medical),
        Recipe("dolly", "sft", True, "general instruction tuning (Dolly-15k)", dolly),
        Recipe("rag", "sft", True, "RAG question answering over retrieved context", rag),
        Recipe("tool", "sft", True, "tool-use SFT: function call, tool result, answer", tool),
        Recipe(
            "ultrafeedback", "dpo", False, "helpfulness DPO on UltraFeedback pairs", ultrafeedback
        ),
    )
}
