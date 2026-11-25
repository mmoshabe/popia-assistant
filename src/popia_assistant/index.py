"""Embed every chunk with Voyage and cache the vectors (cache/embeddings.json), so re-indexing costs nothing.

    uv run python -m popia_assistant.index

Voyage's free allowance (200M tokens) covers this many times over, but an account without a payment method is
limited to a few requests a minute, so this saves after every request, resumes where it stopped, and halves the
batch when it hits the rate limit.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from .chunk import ROOT

EMBED_MODEL = "voyage-4"
CACHE = ROOT / "cache"


def embed_text(c: dict) -> str:
    """What gets embedded: the title (document, section, heading, context) and the text, so 'consent' finds s 11."""
    return f"{c['title']}\n{chr(10).join(b['text'] for b in c['blocks'])}"


def key(text: str) -> str:
    return hashlib.sha256(f"{EMBED_MODEL}\n{text}".encode()).hexdigest()[:24]


def load_chunks() -> list[dict]:
    return json.loads((CACHE / "chunks.json").read_text())


def load_vectors(chunks: list[dict]) -> np.ndarray:
    """Unit-normalised matrix, one row per chunk (fails if any chunk isn't embedded yet)."""
    cache = json.loads((CACHE / "embeddings.json").read_text())
    missing = [c["id"] for c in chunks if key(embed_text(c)) not in cache]
    if missing:
        raise SystemExit(f"{len(missing)} chunks not embedded yet: run `uv run python -m popia_assistant.index`")
    m = np.array([cache[key(embed_text(c))] for c in chunks], dtype=np.float32)
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def build(client=None, batch: int = 16, wait_s: float = 21.0) -> None:
    import voyageai  # imported here so tests and retrieval-only code don't need the client

    client = client or voyageai.Client()
    path = CACHE / "embeddings.json"
    cache: dict[str, list[float]] = json.loads(path.read_text()) if path.exists() else {}
    todo = [embed_text(c) for c in load_chunks()]
    todo = [t for t in todo if key(t) not in cache]
    tokens = 0
    print(f"{len(todo)} chunks to embed ({len(cache)} cached)")
    while todo:
        part = todo[:batch]
        try:
            res = client.embed(part, model=EMBED_MODEL, input_type="document")
        except voyageai.error.RateLimitError:
            batch = max(1, batch // 2)
            print(f"rate limited: waiting 60 s, batch now {batch}")
            time.sleep(60)
            continue
        for t, v in zip(part, res.embeddings):
            cache[key(t)] = v
        tokens += res.total_tokens
        todo = todo[len(part):]
        path.write_text(json.dumps(cache))
        print(f"embedded {len(part)} ({res.total_tokens} tokens), {len(todo)} left")
        if todo:
            time.sleep(wait_s)  # stay under ~3 requests a minute
    print(f"done: {len(cache)} vectors, {tokens} tokens used this run (free allowance: 200M)")


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    build()
