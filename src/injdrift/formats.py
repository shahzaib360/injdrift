"""Prompt formats: where the trusted instruction and the untrusted data go.

Everything in this project hinges on one distinction: text the application *trusts*
(the user's instruction) versus text it merely *processes* (a retrieved passage, an
email, a tool result). Hardened models such as Meta-SecAlign are trained to keep the
two apart when untrusted text arrives in a dedicated ``input`` role.

Two deployment formats are modelled:

* ``secure`` -- instruction in the ``user`` turn, untrusted data in a separate
  ``input`` turn, special tokens stripped from the data. This is the front-end that
  StruQ / SecAlign / Meta-SecAlign assume.
* ``naive`` -- instruction and data concatenated into one ``user`` turn. This is what
  most applications, and most customer fine-tuning datasets, actually do.

Two chat dialects are supported: ``chatml`` (Qwen2.5, SmolLM2, ...) and ``llama3``
(Llama-3.x and Meta-SecAlign). Prompts are rendered by hand instead of through
``tokenizer.apply_chat_template`` because stock templates silently drop roles they
do not know, and ``input`` is exactly such a role.

Content is carried as *pieces*: ``(text, tag)`` pairs. Tags survive rendering as
character spans and survive tokenization as token spans, so probes can find the
exact tokens of an injected instruction and training can mask the loss to the
assistant's tokens.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

# Tags whose text is untrusted and therefore sanitized before rendering.
UNTRUSTED_TAGS = frozenset({"data", "injection", "tool_output"})
# Tags that receive language-modelling loss during supervised training.
LOSS_TAGS = frozenset({"response", "eot"})

DEFAULT_SYSTEM = "You are a helpful assistant."
FORMATS = ("secure", "naive")

Piece = tuple[str, str]


@dataclass(frozen=True)
class Message:
    """One chat turn whose content is a sequence of tagged pieces."""

    role: str
    pieces: tuple[Piece, ...]

    @classmethod
    def of(cls, role: str, content: str | Sequence[Piece], tag: str | None = None) -> Message:
        if isinstance(content, str):
            return cls(role, ((content, tag or _default_tag(role)),))
        return cls(role, tuple((str(t), str(g)) for t, g in content))

    @property
    def text(self) -> str:
        return "".join(t for t, _ in self.pieces)


def _default_tag(role: str) -> str:
    return {
        "system": "system",
        "user": "instruction",
        "input": "data",
        "tool": "tool_output",
        "assistant": "response",
    }.get(role, role)


@dataclass(frozen=True)
class Dialect:
    """Chat markup of a model family."""

    name: str
    bos: str
    header_open: str
    header_close: str
    eot: str
    after_eot: str
    markers: tuple[str, ...]

    def header(self, role: str) -> str:
        return f"{self.header_open}{role}{self.header_close}"


CHATML = Dialect(
    name="chatml",
    bos="",
    header_open="<|im_start|>",
    header_close="\n",
    eot="<|im_end|>",
    after_eot="\n",
    markers=("<|im_start|>", "<|im_end|>", "<|endoftext|>"),
)
LLAMA3 = Dialect(
    name="llama3",
    bos="<|begin_of_text|>",
    header_open="<|start_header_id|>",
    header_close="<|end_header_id|>\n\n",
    eot="<|eot_id|>",
    after_eot="",
    markers=(
        "<|begin_of_text|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|eot_id|>",
        "<|end_of_text|>",
    ),
)
DIALECTS = {d.name: d for d in (CHATML, LLAMA3)}


def detect_dialect(tokenizer) -> Dialect:
    """Pick the dialect from the special tokens a tokenizer knows about."""
    vocab = tokenizer.get_vocab()
    if "<|im_start|>" in vocab:
        return CHATML
    if "<|start_header_id|>" in vocab:
        return LLAMA3
    raise ValueError(
        "could not detect chat dialect (no <|im_start|> or <|start_header_id|> token); "
        "set `dialect` explicitly"
    )


def special_strings(tokenizer, dialect: Dialect) -> tuple[str, ...]:
    """Every string that would tokenize to a control token, longest first."""
    found = set(dialect.markers)
    for tok in getattr(tokenizer, "added_tokens_decoder", {}).values():
        content = getattr(tok, "content", str(tok))
        if len(content) > 1:
            found.add(content)
    for tok in getattr(tokenizer, "all_special_tokens", []):
        if len(tok) > 1:
            found.add(tok)
    return tuple(sorted(found, key=len, reverse=True))


def sanitize(text: str, strings: Iterable[str]) -> str:
    """Remove control-token strings from untrusted text.

    Runs to a fixpoint, so nested constructions such as ``<|im_<|im_end|>end|>`` cannot
    reassemble a delimiter after one pass.
    """
    strings = tuple(strings)
    previous = None
    while previous != text:
        previous = text
        for s in strings:
            text = text.replace(s, "")
    return text


@dataclass
class Rendered:
    """Rendered prompt text plus character spans for every tagged piece."""

    text: str
    spans: list[tuple[str, int, int]] = field(default_factory=list)

    def spans_for(self, tag: str) -> list[tuple[int, int]]:
        return [(s, e) for t, s, e in self.spans if t == tag]


def render(
    messages: Sequence[Message],
    dialect: Dialect,
    *,
    add_generation_prompt: bool,
    strip: Sequence[str] = (),
) -> Rendered:
    """Render messages to text, sanitizing untrusted pieces and recording spans.

    The end-of-turn marker of each assistant message is recorded under the ``eot`` tag
    so supervised training teaches the model where to stop.
    """
    strip = tuple(strip) or dialect.markers
    parts: list[str] = [dialect.bos]
    pos = len(dialect.bos)
    spans: list[tuple[str, int, int]] = []
    for msg in messages:
        header = dialect.header(msg.role)
        parts.append(header)
        pos += len(header)
        for text, tag in msg.pieces:
            if tag in UNTRUSTED_TAGS:
                text = sanitize(text, strip)
            spans.append((tag, pos, pos + len(text)))
            parts.append(text)
            pos += len(text)
        if msg.role == "assistant":
            spans.append(("eot", pos, pos + len(dialect.eot)))
        parts.append(dialect.eot + dialect.after_eot)
        pos += len(dialect.eot) + len(dialect.after_eot)
    if add_generation_prompt:
        parts.append(dialect.header("assistant"))
    return Rendered("".join(parts), spans)


def build_messages(
    instruction: str,
    data: str | Sequence[Piece] | None,
    *,
    fmt: str,
    system: str | None = DEFAULT_SYSTEM,
    response: str | None = None,
) -> list[Message]:
    """Messages for a single instruction (+ optional untrusted data) in a format."""
    if fmt not in FORMATS:
        raise ValueError(f"unknown format {fmt!r}; expected one of {FORMATS}")
    pieces: tuple[Piece, ...] = ()
    if isinstance(data, str):
        pieces = ((data, "data"),) if data else ()
    elif data:
        pieces = tuple(data)
    msgs: list[Message] = []
    if system:
        msgs.append(Message.of("system", system))
    if fmt == "secure":
        msgs.append(Message.of("user", instruction))
        if pieces:
            msgs.append(Message("input", pieces))
    else:
        user: list[Piece] = [(instruction, "instruction")]
        if pieces:
            user.append(("\n\n", "sep"))
            user.extend(pieces)
        msgs.append(Message("user", tuple(user)))
    if response is not None:
        msgs.append(Message.of("assistant", response))
    return msgs


@dataclass
class Encoded:
    """Token ids of a rendered prompt plus token-index spans per tag."""

    input_ids: list[int]
    token_spans: dict[str, list[tuple[int, int]]]

    def mask(self, tags: Iterable[str]) -> list[bool]:
        keep = [False] * len(self.input_ids)
        for tag in tags:
            for s, e in self.token_spans.get(tag, []):
                for i in range(s, e):
                    keep[i] = True
        return keep

    def indices(self, tag: str) -> list[int]:
        out: list[int] = []
        for s, e in self.token_spans.get(tag, []):
            out.extend(range(s, e))
        return out


def encode(tokenizer, rendered: Rendered) -> Encoded:
    """Tokenize rendered text and map character spans onto token spans.

    A token belongs to a span if their character ranges overlap, so a token that
    straddles a boundary is counted on both sides (rare with chat markup, where
    boundaries are newlines and control tokens).
    """
    enc = tokenizer(rendered.text, add_special_tokens=False, return_offsets_mapping=True)
    ids: list[int] = list(enc["input_ids"])
    offsets: list[tuple[int, int]] = [tuple(o) for o in enc["offset_mapping"]]
    token_spans: dict[str, list[tuple[int, int]]] = {}
    for tag, s, e in rendered.spans:
        if e <= s:
            continue
        idx = [i for i, (a, b) in enumerate(offsets) if a < e and b > s]
        if idx:
            token_spans.setdefault(tag, []).append((idx[0], idx[-1] + 1))
    return Encoded(ids, token_spans)
