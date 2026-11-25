"""Answer a question from retrieved extracts, with Claude citing the exact blocks it relied on.

Each retrieved chunk is sent as a `search_result` block whose content is one text block per subsection /
paragraph, each starting with its label ("[s 11(1)] ..."). Claude's citations point at (result, block), so every
citation maps back to a precise provision. Citations can't be combined with structured outputs, so the status comes
from a first line ("STATUS: answered") that this code parses.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .retrieve import Hit

ROOT = Path(__file__).resolve().parents[2]
MODEL = "claude-haiku-5-5"
PRICE = {"claude-haiku-5-5": (0.10, 0.50), "claude-sonnet-5-5": (2.0, 10.0)}  # $ per million tokens in / out
STATUSES = ("answered", "partly_answered", "not_in_sources", "out_of_scope")
DISCLAIMER = ("General information from POPIA, its Regulations and the Information Regulator's guidance notes, "
              "not legal advice. For a decision about your own situation, check with the Information Regulator or a lawyer.")


def system_prompt() -> str:
    return (ROOT / "prompt" / "answer.md").read_text()


def search_results(hits: list[Hit]) -> list[dict]:
    return [{
        "type": "search_result",
        "source": h.chunk["id"],
        "title": h.chunk["title"],
        "content": [{"type": "text", "text": f"[{b['label']}] {b['text']}"} for b in h.chunk["blocks"]],
        "citations": {"enabled": True},
    } for h in hits]


def request(question: str, hits: list[Hit], model: str = MODEL) -> dict:
    return {
        "model": model,
        "max_tokens": 2000,
        "system": system_prompt(),
        "messages": [{"role": "user", "content": [
            *search_results(hits),
            {"type": "text", "text": f"<question>\n{question}\n</question>"},
        ]}],
    }


@dataclass
class Citation:
    n: int
    source: str  # chunk id, e.g. "popia:s11"
    labels: list[str]  # block labels cited, e.g. ["s 11(1)"]
    cited_text: str


@dataclass
class Answer:
    status: str
    text: str  # answer with [n] markers after cited sentences
    citations: list[Citation] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = MODEL
    stop_reason: str = ""
    retried: bool = False

    @property
    def verified(self) -> bool:
        """An answer is only trusted when it cites the extracts; declines need no citations."""
        return self.status not in ("answered", "partly_answered") or bool(self.citations)

    @property
    def cost(self) -> float:
        p_in, p_out = PRICE.get(self.model, (0, 0))
        return (self.input_tokens * p_in + self.output_tokens * p_out) / 1e6

    def render(self) -> str:
        lines = [self.text.strip(), ""]
        if not self.verified:
            lines = ["WARNING: this answer has no citations, so it can't be checked against the documents.", ""] + lines
        lines += [f"[{c.n}] {', '.join(c.labels)}" for c in self.citations]
        return "\n".join(lines + ["", DISCLAIMER])


def parse(response, hits: list[Hit], model: str = MODEL) -> Answer:
    """Join the text blocks, attach [n] markers for citations, and pull out the status line."""
    parts: list[str] = []
    cites: list[Citation] = []
    seen: dict[tuple, int] = {}
    for block in response.content:
        if getattr(block, "type", "") != "text":
            continue  # thinking blocks
        parts.append(block.text)
        marks = []
        for c in getattr(block, "citations", None) or []:
            if getattr(c, "type", "") != "search_result_location" or not 0 <= c.search_result_index < len(hits):
                continue
            chunk = hits[c.search_result_index].chunk
            labels = [b["label"] for b in chunk["blocks"][c.start_block_index:c.end_block_index]]
            k = (chunk["id"], tuple(labels))
            if k not in seen:
                seen[k] = len(cites) + 1
                cites.append(Citation(seen[k], chunk["id"], labels, c.cited_text))
            if seen[k] not in marks:
                marks.append(seen[k])
        if marks:
            parts.append("".join(f"[{m}]" for m in marks))
    text = "".join(parts).strip()
    m = re.match(r"\s*STATUS:\s*([a-z_]+)\s*", text)
    status = m.group(1) if m and m.group(1) in STATUSES else "unparsed"
    if m:
        text = text[m.end():]
    if getattr(response, "stop_reason", "") == "refusal":
        status = "refused"
    u = response.usage
    return Answer(status, text.strip(), cites, u.input_tokens, u.output_tokens, model, getattr(response, "stop_reason", ""))


RETRY_NOTE = ("Your previous reply to this question cited none of the search results. Answer again, citing the "
              "search results that support each statement, or use STATUS: not_in_sources if they don't.")


def answer(question: str, hits: list[Hit], client=None, model: str = MODEL) -> Answer:
    """One Claude call; if it answers without citing anything, one retry with a reminder (then flagged unverified)."""
    import anthropic

    client = client or anthropic.Anthropic()
    first = parse(client.messages.create(**request(question, hits, model)), hits, model)
    if first.verified:
        return first
    req = request(question, hits, model)
    req["messages"][0]["content"].append({"type": "text", "text": RETRY_NOTE})
    second = parse(client.messages.create(**req), hits, model)
    second.input_tokens += first.input_tokens
    second.output_tokens += first.output_tokens
    second.retried = True
    return second
