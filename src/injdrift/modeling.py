"""Model loading, batched greedy generation and teacher-forced scoring.

Everything here is written for CPU first (bf16 autocast picks up AMX/AVX-512 bf16
kernels on recent Xeons) but runs unchanged on a GPU.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from injdrift.formats import Dialect, detect_dialect, special_strings

LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def autocast(device: torch.device | None = None):
    device = device or default_device()
    return torch.autocast(device.type, dtype=torch.bfloat16)


@dataclass
class Loaded:
    """A model with everything needed to prompt it securely."""

    model: torch.nn.Module
    tokenizer: object
    dialect: Dialect
    strip: tuple[str, ...]
    name: str

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def stop_ids(self) -> list[int]:
        ids = {self.tokenizer.convert_tokens_to_ids(self.dialect.eot)}
        if self.tokenizer.eos_token_id is not None:
            ids.add(self.tokenizer.eos_token_id)
        return sorted(i for i in ids if i is not None and i >= 0)


def load_tokenizer(name_or_path: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name_or_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load(
    name_or_path: str,
    *,
    adapter: str | None = None,
    merge: bool = True,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device | None = None,
    dialect: Dialect | None = None,
) -> Loaded:
    """Load a causal LM (optionally with a PEFT adapter, merged for fast inference)."""
    from transformers import AutoModelForCausalLM

    device = device or default_device()
    tok = load_tokenizer(name_or_path)
    model = AutoModelForCausalLM.from_pretrained(name_or_path, dtype=dtype)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
        if merge:
            model = model.merge_and_unload()
    model.to(device).eval()
    dialect = dialect or detect_dialect(tok)
    loaded = Loaded(model, tok, dialect, special_strings(tok, dialect), name_or_path)
    # Replace sampling defaults shipped with chat models (temperature, top-p,
    # repetition penalty): every evaluation in this project is greedy.
    from transformers import GenerationConfig

    model.generation_config = GenerationConfig(
        eos_token_id=loaded.stop_ids, pad_token_id=tok.pad_token_id
    )
    return loaded


def split_lm(model: torch.nn.Module) -> tuple[torch.nn.Module, torch.nn.Module]:
    """(transformer backbone, lm_head) for plain and PEFT-wrapped causal LMs."""
    inner = model.get_base_model() if hasattr(model, "get_base_model") else model
    return inner.model, inner.lm_head


def _batches(lengths: Sequence[int], max_tokens: int, max_batch: int) -> list[list[int]]:
    """Group indices (longest first) so padded batch size stays under ``max_tokens``."""
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    out: list[list[int]] = []
    cur: list[int] = []
    for i in order:
        longest = lengths[cur[0]] if cur else lengths[i]
        if cur and (len(cur) + 1 > max_batch or longest * (len(cur) + 1) > max_tokens):
            out.append(cur)
            cur = []
        cur.append(i)
    if cur:
        out.append(cur)
    return out


@torch.inference_mode()
def generate(
    loaded: Loaded,
    prompts: Sequence[Sequence[int]],
    *,
    max_new_tokens: int = 24,
    max_tokens: int = 6144,
    max_batch: int = 32,
    progress: Callable[[int, int], None] | None = None,
) -> list[str]:
    """Greedy decoding for many prompts at once (left-padded, length-bucketed)."""
    from transformers import GenerationConfig

    tok, model = loaded.tokenizer, loaded.model
    pad = tok.pad_token_id
    cfg = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=1.0,
        eos_token_id=loaded.stop_ids,
        pad_token_id=pad,
    )
    outputs: list[str] = [""] * len(prompts)
    done = 0
    for batch in _batches([len(p) for p in prompts], max_tokens, max_batch):
        width = max(len(prompts[i]) for i in batch)
        ids = torch.full((len(batch), width), pad, dtype=torch.long)
        mask = torch.zeros((len(batch), width), dtype=torch.long)
        for row, i in enumerate(batch):
            p = list(prompts[i])
            ids[row, width - len(p) :] = torch.tensor(p)
            mask[row, width - len(p) :] = 1
        with autocast(loaded.device):
            gen = model.generate(
                input_ids=ids.to(loaded.device),
                attention_mask=mask.to(loaded.device),
                generation_config=cfg,
            )
        for row, i in enumerate(batch):
            outputs[i] = tok.decode(gen[row, width:], skip_special_tokens=True)
        done += len(batch)
        if progress:
            progress(done, len(prompts))
    return outputs


def _gather_logprobs(
    model: torch.nn.Module,
    seqs: Sequence[Sequence[int]],
    targets: Sequence[Sequence[int]] | Sequence[int],
    pad_id: int,
) -> list[torch.Tensor]:
    """Log-probs of selected tokens given their prefixes, one tensor per sequence.

    ``targets`` holds, per sequence, either the index of the first scored token (score
    everything from there on) or an explicit list of token indices to score. Logits
    are only materialized at those positions, which matters on CPU where the
    vocabulary projection is a large share of the compute.
    """
    backbone, head = split_lm(model)
    device = next(model.parameters()).device
    width = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), width), dtype=torch.long)
    for r, s in enumerate(seqs):
        ids[r, : len(s)] = torch.tensor(list(s))
        mask[r, : len(s)] = 1
    ids, mask = ids.to(device), mask.to(device)
    hidden = backbone(input_ids=ids, attention_mask=mask).last_hidden_state
    rows, cols, tgts, owner = [], [], [], []
    for r, (s, tg) in enumerate(zip(seqs, targets, strict=True)):
        positions = range(max(tg, 1), len(s)) if isinstance(tg, int) else tg
        for pos in positions:
            if pos < 1:
                continue
            rows.append(r)
            cols.append(pos - 1)
            tgts.append(s[pos])
            owner.append(r)
    if not rows:
        return [torch.zeros(0, device=device) for _ in seqs]
    h = hidden[torch.tensor(rows, device=device), torch.tensor(cols, device=device)]
    logp = torch.log_softmax(head(h).float(), dim=-1)
    picked = logp.gather(1, torch.tensor(tgts, device=device).unsqueeze(1)).squeeze(1)
    out: list[list[torch.Tensor]] = [[] for _ in seqs]
    for k, r in enumerate(owner):
        out[r].append(picked[k])
    return [torch.stack(v) if v else torch.zeros(0, device=device) for v in out]


@torch.inference_mode()
def continuation_logprobs(
    loaded: Loaded,
    prompts: Sequence[Sequence[int]],
    continuations: Sequence[Sequence[int]],
    *,
    max_tokens: int = 8192,
    max_batch: int = 32,
) -> list[float]:
    """Mean per-token log-probability of each continuation given its prompt."""
    seqs = [list(p) + list(c) for p, c in zip(prompts, continuations, strict=True)]
    starts = [len(p) for p in prompts]
    out = [0.0] * len(seqs)
    for batch in _batches([len(s) for s in seqs], max_tokens, max_batch):
        with autocast(loaded.device):
            lps = _gather_logprobs(
                loaded.model,
                [seqs[i] for i in batch],
                [starts[i] for i in batch],
                loaded.tokenizer.pad_token_id,
            )
        for i, lp in zip(batch, lps, strict=True):
            out[i] = float(lp.mean()) if lp.numel() else float("nan")
    return out


def set_threads() -> None:
    """Respect INJDRIFT_THREADS (useful when several jobs share one machine)."""
    n = os.environ.get("INJDRIFT_THREADS")
    if n:
        torch.set_num_threads(int(n))
