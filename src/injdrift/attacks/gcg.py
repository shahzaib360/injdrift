"""GCG (Zou et al., 2023) as an adaptive prompt-injection attack.

The attacker controls the untrusted data. We append the injected instruction plus an
adversarial suffix of ``n_tokens`` tokens to the data and optimize the suffix so the
model's reply starts with the injected task's canary (e.g. ``Hacked``). This is the
optimization-based attack SecAlign and Meta-SecAlign evaluate against.

CPU-friendly details:

* the prompt up to the suffix is encoded once and reused through the KV cache, so each
  candidate only costs its suffix, the closing chat markup and the target tokens;
* candidates are restricted to printable ASCII, non-control tokens and must survive a
  decode/encode round trip, so the final attack is real text that passes through the
  secure front-end's sanitizer unchanged;
* success is always verified by re-rendering the *text* attack and decoding greedily.
"""

from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass
from functools import lru_cache

import torch
import torch.nn.functional as F

from injdrift.data.suites import Case, normalize
from injdrift.evaluate.runner import encode_case
from injdrift.modeling import Loaded, autocast, generate, split_lm


@dataclass
class GCGConfig:
    n_tokens: int = 16
    steps: int = 40
    topk: int = 128
    batch: int = 48
    check_every: int = 10
    seed: int = 0
    init: str = " !" * 16


@lru_cache(maxsize=4)
def _allowed_tokens(tokenizer_id: int, vocab_size: int, tok) -> torch.Tensor:
    allowed = torch.zeros(vocab_size, dtype=torch.bool)
    special = set(tok.all_special_ids)
    for i in range(min(vocab_size, len(tok))):
        if i in special:
            continue
        s = tok.decode([i])
        if s and s.isascii() and s.isprintable() and "<|" not in s and "|>" not in s and s.strip():
            allowed[i] = True
    return allowed


def _with_suffix(case: Case, suffix_text: str) -> Case:
    """The case with ``suffix_text`` appended to its injection piece."""
    pieces = list(case.pieces)
    for k in range(len(pieces) - 1, -1, -1):
        text, tag = pieces[k]
        if tag == "injection":
            pieces[k] = (text.rstrip() + suffix_text + (" " if text.endswith(" ") else ""), tag)
            break
    return Case(case.item, f"gcg:{case.attack}", case.position, case.injected, tuple(pieces))


def _segments(loaded: Loaded, case: Case, fmt: str, n_tokens: int) -> tuple[list[int], list[int]]:
    """Token ids before and after the suffix slot (rendered with a placeholder)."""
    marker = " ZQXJMARKER"
    enc = encode_case(loaded, _with_suffix(case, marker), fmt)
    tok = loaded.tokenizer
    marker_ids = tok(marker, add_special_tokens=False)["input_ids"]
    ids = enc.input_ids
    for s in range(len(ids) - len(marker_ids), -1, -1):
        if ids[s : s + len(marker_ids)] == marker_ids:
            return ids[:s], ids[s + len(marker_ids) :]
    raise RuntimeError("could not locate the suffix slot in the encoded prompt")


def gcg_attack(
    loaded: Loaded, case: Case, *, fmt: str = "secure", cfg: GCGConfig | None = None
) -> dict:
    """Optimize an adversarial suffix for one attacked case; returns a result record."""
    cfg = cfg or GCGConfig()
    if case.injected is None:
        raise ValueError("GCG needs an attacked case")
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)
    tok, model = loaded.tokenizer, loaded.model
    backbone, head = split_lm(model)
    device = loaded.device
    embed = model.get_input_embeddings().weight
    vocab = embed.shape[0]
    allowed = _allowed_tokens(id(tok), vocab, tok).to(device)

    prefix, post = _segments(loaded, case, fmt, cfg.n_tokens)
    target = tok(case.injected.reply, add_special_tokens=False)["input_ids"]
    suffix = tok(cfg.init, add_special_tokens=False)["input_ids"][: cfg.n_tokens]
    while len(suffix) < cfg.n_tokens:
        suffix.append(suffix[-1])
    suffix_t = torch.tensor(suffix, device=device)
    post_t = torch.tensor(post, device=device)
    target_t = torch.tensor(target, device=device)
    tail_len = cfg.n_tokens + len(post) + len(target) - 1
    pred_slice = slice(cfg.n_tokens + len(post) - 1, tail_len)

    with torch.no_grad(), autocast(device):
        prefix_cache = backbone(
            input_ids=torch.tensor([prefix], device=device), use_cache=True
        ).past_key_values

    def fresh_cache(batch: int):
        c = copy.deepcopy(prefix_cache)
        if batch > 1:
            c.batch_repeat_interleave(batch)
        return c

    def check(sfx: torch.Tensor) -> tuple[bool, str, str]:
        text = tok.decode(sfx.tolist())
        enc = encode_case(loaded, _with_suffix(case, text), fmt)
        resp = generate(loaded, [enc.input_ids], max_new_tokens=16)[0]
        return normalize(resp).startswith(case.injected.canary), text, resp

    losses: list[float] = []
    best_loss, best = float("inf"), suffix_t.clone()
    t0 = time.perf_counter()
    result = {"success": False, "steps": 0}
    for step in range(1, cfg.steps + 1):
        # 1) gradient of the target loss w.r.t. a one-hot encoding of the suffix
        one_hot = torch.zeros(cfg.n_tokens, vocab, device=device, dtype=embed.dtype)
        one_hot.scatter_(1, suffix_t.unsqueeze(1), 1.0)
        one_hot.requires_grad_(True)
        with autocast(device):
            tail = torch.cat([one_hot @ embed, embed[post_t], embed[target_t[:-1]]]).unsqueeze(0)
            hidden = backbone(inputs_embeds=tail, past_key_values=fresh_cache(1)).last_hidden_state
            loss = F.cross_entropy(head(hidden[0, pred_slice]).float(), target_t)
        loss.backward()
        grad = one_hot.grad.float()
        grad[:, ~allowed] = float("inf")
        top = (-grad).topk(cfg.topk, dim=1).indices

        # 2) sample single-token swaps, keep those that round-trip through the tokenizer
        cands = []
        for _ in range(cfg.batch * 2):
            c = suffix_t.clone()
            pos = rng.randrange(cfg.n_tokens)
            c[pos] = top[pos, rng.randrange(cfg.topk)]
            if tok(tok.decode(c.tolist()), add_special_tokens=False)["input_ids"] == c.tolist():
                cands.append(c)
            if len(cands) == cfg.batch:
                break
        if not cands:
            continue
        cand = torch.stack(cands)

        # 3) evaluate every candidate in one batched forward pass over the cached prefix
        with torch.no_grad(), autocast(device):
            ids = torch.cat(
                [cand, post_t.expand(len(cand), -1), target_t[:-1].expand(len(cand), -1)], dim=1
            )
            hidden = backbone(
                input_ids=ids, past_key_values=fresh_cache(len(cand))
            ).last_hidden_state
            logits = head(hidden[:, pred_slice]).float()
            per = (
                F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    target_t.repeat(len(cand)),
                    reduction="none",
                )
                .view(len(cand), -1)
                .mean(1)
            )
        k = int(per.argmin())
        suffix_t = cand[k]
        losses.append(float(per[k]))
        if losses[-1] < best_loss:
            best_loss, best = losses[-1], suffix_t.clone()
        if step % cfg.check_every == 0 or step == cfg.steps:
            ok, text, resp = check(best)
            result = {"success": ok, "steps": step, "suffix": text, "response": resp}
            if ok:
                break
    result.update(
        {
            "key": case.key,
            "loss_curve": losses,
            "best_loss": best_loss,
            "seconds": round(time.perf_counter() - t0, 1),
        }
    )
    return result
