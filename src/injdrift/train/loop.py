"""A small, explicit training loop: SFT and DPO, LoRA or full, plus preservation terms.

Written by hand rather than on top of a trainer framework so that three things stay
easy to see and to control:

* the loss is computed only where it should be (assistant tokens; logits are never
  materialized for prompt positions, which is a large saving on CPU),
* robustness-preservation terms (replay, SecAnchor) are a few explicit lines in the
  step function, and
* a tracker callback can measure the model *mid-training* without checkpoint I/O.
"""

from __future__ import annotations

import math
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from injdrift.modeling import LORA_TARGETS, Loaded, _batches, _gather_logprobs, autocast
from injdrift.train.examples import PrefExample, SFTExample


@dataclass
class TrainConfig:
    kind: str = "sft"  # sft | dpo
    peft: str = "lora"  # lora | full
    lr: float | None = None  # default: 2e-4 (LoRA) / 1e-5 (full)
    epochs: float = 1.0
    batch_size: int = 8
    micro_tokens: int = 2048  # padded tokens per forward pass (gradient accumulation)
    warmup_steps: int = 5
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    beta: float = 0.1  # DPO temperature
    seed: int = 0
    # Robustness preservation (see injdrift.train.preserve).
    preserve: str = "none"  # none | replay | secanchor
    anchor_weight: float = 1.0
    anchor_pairs: int = 2  # SecAnchor pairs per optimizer step
    replay_per_step: int = 1  # replay examples added to every optimizer step
    track_fracs: tuple[float, ...] = (1 / 3, 2 / 3)

    def learning_rate(self) -> float:
        if self.lr is not None:
            return self.lr
        return 2e-4 if self.peft == "lora" else 1e-5

    def to_dict(self) -> dict:
        return asdict(self)


def prepare(loaded: Loaded, cfg: TrainConfig) -> list[torch.nn.Parameter]:
    """Attach LoRA adapters or unfreeze weights; returns the trainable parameters."""
    torch.manual_seed(cfg.seed)
    if cfg.peft == "lora":
        from peft import LoraConfig, get_peft_model

        lora = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=list(LORA_TARGETS),
            task_type="CAUSAL_LM",
        )
        loaded.model = get_peft_model(loaded.model, lora)
    elif cfg.peft == "full":
        loaded.model = loaded.model.float()
        for p in loaded.model.parameters():
            p.requires_grad_(True)
        # Embeddings (tied to the LM head in the small models used here) stay frozen:
        # common practice, and it keeps optimizer state within a laptop's memory.
        loaded.model.get_input_embeddings().weight.requires_grad_(False)
    else:
        raise ValueError(f"unknown peft mode {cfg.peft!r}")
    loaded.model.train()
    return [p for p in loaded.model.parameters() if p.requires_grad]


# --------------------------------------------------------------------------- losses


def sft_loss_sum(
    model: torch.nn.Module, exs: Sequence[SFTExample], pad_id: int
) -> tuple[torch.Tensor, int]:
    """Summed token cross-entropy over loss positions, and the number of positions."""
    seqs = [e.input_ids for e in exs]
    positions = [[i for i, m in enumerate(e.loss_mask) if m and i > 0] for e in exs]
    lps = _gather_logprobs(model, seqs, positions, pad_id)
    count = sum(int(lp.numel()) for lp in lps)
    total = -torch.cat(lps).sum()
    return total, count


def pref_logps(
    model: torch.nn.Module, exs: Sequence[PrefExample], pad_id: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Summed log-probs of chosen and rejected continuations."""
    seqs = [e.prompt_ids + e.chosen_ids for e in exs] + [e.prompt_ids + e.rejected_ids for e in exs]
    starts = [len(e.prompt_ids) for e in exs] * 2
    lps = _gather_logprobs(model, seqs, starts, pad_id)
    sums = torch.stack([lp.sum() for lp in lps])
    return sums[: len(exs)], sums[len(exs) :]


@torch.inference_mode()
def reference_logps(
    loaded: Loaded, exs: Sequence[PrefExample], max_tokens: int = 6144
) -> list[tuple[float, float]]:
    """Reference-model log-probs, computed once up front (the model is the reference)."""
    was_training = loaded.model.training
    loaded.model.eval()
    out: list[tuple[float, float] | None] = [None] * len(exs)
    lengths = [len(e.prompt_ids) + max(len(e.chosen_ids), len(e.rejected_ids)) for e in exs]
    for batch in _batches(lengths, max_tokens, 8):
        with autocast(loaded.device):
            c, r = pref_logps(loaded.model, [exs[i] for i in batch], loaded.tokenizer.pad_token_id)
        for k, i in enumerate(batch):
            out[i] = (float(c[k]), float(r[k]))
    loaded.model.train(was_training)
    return out  # type: ignore[return-value]


def dpo_loss(
    model: torch.nn.Module,
    exs: Sequence[PrefExample],
    refs: Sequence[tuple[float, float]],
    beta: float,
    pad_id: int,
) -> tuple[torch.Tensor, dict]:
    c, r = pref_logps(model, exs, pad_id)
    rc = torch.tensor([x[0] for x in refs], device=c.device)
    rr = torch.tensor([x[1] for x in refs], device=c.device)
    margin = beta * ((c - rc) - (r - rr))
    loss = -F.logsigmoid(margin).mean()
    m = margin.detach()
    return loss, {"reward_acc": float((m > 0).float().mean()), "margin": float(m.mean())}


# ---------------------------------------------------------------------------- batching


def _epoch_batches(lengths: Sequence[int], batch_size: int, rng: random.Random) -> list[list[int]]:
    """Shuffle, then sort inside mega-batches so each batch has similar lengths."""
    order = list(range(len(lengths)))
    rng.shuffle(order)
    mega = batch_size * 8
    batches: list[list[int]] = []
    for s in range(0, len(order), mega):
        chunk = sorted(order[s : s + mega], key=lambda i: lengths[i])
        batches.extend(chunk[k : k + batch_size] for k in range(0, len(chunk), batch_size))
    rng.shuffle(batches)
    return [b for b in batches if b]


def _chunks(sizes: Sequence[int], budget: int) -> list[slice]:
    """Consecutive slices whose padded size (count x longest) stays within ``budget``."""
    out, start, longest = [], 0, 0
    for i, s in enumerate(sizes):
        longest_with = max(longest, s)
        if i > start and longest_with * (i - start + 1) > budget:
            out.append(slice(start, i))
            start, longest_with = i, s
        longest = longest_with
    if start < len(sizes):
        out.append(slice(start, len(sizes)))
    return out


def _cycle(items: Sequence, rng: random.Random):
    while True:
        order = list(range(len(items)))
        rng.shuffle(order)
        for i in order:
            yield items[i]


# ------------------------------------------------------------------------------- train


def train(
    loaded: Loaded,
    cfg: TrainConfig,
    examples: Sequence[SFTExample] | Sequence[PrefExample],
    *,
    anchors: Sequence[PrefExample] | None = None,
    replay: Sequence[SFTExample] | None = None,
    tracker: Callable[[Loaded, int], dict] | None = None,
    log: Callable[[dict], None] | None = None,
) -> list[dict]:
    """Train ``loaded.model`` in place. Returns the per-step history."""
    from transformers import get_cosine_schedule_with_warmup

    if not examples:
        raise ValueError("no training examples")
    params = prepare(loaded, cfg)
    model, pad = loaded.model, loaded.tokenizer.pad_token_id
    rng = random.Random(cfg.seed)

    refs = reference_logps(loaded, examples) if cfg.kind == "dpo" else None
    anchor_refs = None
    if cfg.preserve == "secanchor":
        if not anchors:
            raise ValueError("preserve='secanchor' needs anchor preference pairs")
        anchor_refs = reference_logps(loaded, anchors)
        anchor_iter = _cycle(
            list(zip(anchors, anchor_refs, strict=True)), random.Random(cfg.seed + 1)
        )
    if cfg.preserve == "replay":
        if not replay:
            raise ValueError("preserve='replay' needs replay examples")
        replay_iter = _cycle(list(replay), random.Random(cfg.seed + 2))

    if cfg.kind == "sft":
        lengths = [len(e.input_ids) for e in examples]
    else:
        lengths = [
            len(e.prompt_ids) + max(len(e.chosen_ids), len(e.rejected_ids)) for e in examples
        ]
    steps_per_epoch = math.ceil(len(examples) / cfg.batch_size)
    total_steps = max(1, math.ceil(steps_per_epoch * cfg.epochs))
    track_steps = {max(1, round(f * total_steps)) for f in cfg.track_fracs} - {total_steps}

    opt = torch.optim.AdamW(params, lr=cfg.learning_rate(), weight_decay=cfg.weight_decay)
    sched = get_cosine_schedule_with_warmup(
        opt, min(cfg.warmup_steps, total_steps // 4), total_steps
    )
    history: list[dict] = []
    batches: list[list[int]] = []
    t0 = time.perf_counter()
    for step in range(1, total_steps + 1):
        if not batches:
            batches = _epoch_batches(lengths, cfg.batch_size, rng)
        idx = batches.pop()
        rec: dict = {"step": step, "lr": sched.get_last_lr()[0]}
        with autocast(loaded.device):
            if cfg.kind == "sft":
                batch = [examples[i] for i in idx]
                if cfg.preserve == "replay":
                    batch = batch + [next(replay_iter) for _ in range(cfg.replay_per_step)]
                batch.sort(key=lambda e: len(e.input_ids))
                n_tok = sum(e.n_loss for e in batch) or 1
                total = 0.0
                for part in _chunks([len(e.input_ids) for e in batch], cfg.micro_tokens):
                    loss_sum, _ = sft_loss_sum(model, batch[part], pad)
                    (loss_sum / n_tok).backward()
                    total += float(loss_sum.detach())
                rec["loss"] = total / n_tok
            else:
                order = sorted(idx, key=lambda i: lengths[i])
                pairs = [examples[i] for i in order]
                pair_refs = [refs[i] for i in order]
                total, accs = 0.0, []
                # each pair is two sequences (prompt+chosen, prompt+rejected)
                for part in _chunks([2 * lengths[i] for i in order], cfg.micro_tokens):
                    loss, info = dpo_loss(model, pairs[part], pair_refs[part], cfg.beta, pad)
                    frac = len(pairs[part]) / len(pairs)
                    (loss * frac).backward()
                    total += float(loss.detach()) * frac
                    accs.append(info["reward_acc"] * frac)
                rec["loss"] = total
                rec["reward_acc"] = sum(accs)
            if cfg.preserve == "secanchor":
                drawn = [next(anchor_iter) for _ in range(cfg.anchor_pairs)]
                aloss, ainfo = dpo_loss(
                    model, [a for a, _ in drawn], [r for _, r in drawn], cfg.beta, pad
                )
                (cfg.anchor_weight * aloss).backward()
                rec["anchor_loss"] = float(aloss.detach())
                rec["anchor_acc"] = ainfo["reward_acc"]
        if cfg.grad_clip:
            torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        rec["elapsed_s"] = round(time.perf_counter() - t0, 1)
        history.append(rec)
        if log:
            log(rec)
        if tracker is not None and step in track_steps:
            model.eval()
            metrics = tracker(loaded, step)
            model.train()
            metrics.update({"step": step, "frac": step / total_steps})
            history.append({"track": metrics})
            if log:
                log({"track": metrics})
    model.eval()
    return history
