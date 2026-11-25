"""Hybrid retrieval: vector search + BM25 keyword search + exact section lookup, merged, then reranked.

Why all three: vectors find meaning ("can I email my customers about a sale" -> s 69), BM25 finds rare exact words
("biometric", "Form 4"), and a question naming "section 22" or "regulation 4" should always see that text.
"""

from __future__ import annotations

import math
import re
import time
from collections import Counter
from dataclasses import dataclass

import numpy as np

from .index import EMBED_MODEL, embed_text, load_chunks, load_vectors

RERANK_MODEL = "rerank-2.5"  # rerank-3 has no free allowance; rerank-2.5 does
STOP = set("a an and are as at be by can do does for from has have how i if in is it its may me must my no not of on or "
           "our should so that the their them then there these they this to under was we what when where which who will "
           "with you your".split())


def tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOP]


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.4, b: float = 0.75):
        self.toks = [tokens(d) for d in docs]
        self.k1, self.b = k1, b
        self.avg = sum(map(len, self.toks)) / len(self.toks)
        df = Counter(t for d in self.toks for t in set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.tf = [Counter(d) for d in self.toks]

    def scores(self, query: str) -> np.ndarray:
        q = tokens(query)
        out = np.zeros(len(self.toks))
        for i, (tf, d) in enumerate(zip(self.tf, self.toks)):
            norm = self.k1 * (1 - self.b + self.b * len(d) / self.avg)
            out[i] = sum(self.idf.get(t, 0) * tf[t] * (self.k1 + 1) / (tf[t] + norm) for t in q if t in tf)
        return out


SECTION_REF = re.compile(r"\b(?:section|sec|s)\.?\s*(\d{1,3})\b|\b(?:regulation|reg)\.?\s*(\d{1,2})\b", re.I)


DEFINED = re.compile(r'definition of "([^"]+)"')


def asks_for_definition(query: str, term: str) -> bool:
    """True when the question quotes a defined term or asks what it means ("what is a responsible party")."""
    t = re.escape(term)
    return bool(re.search(rf"['\"\u2018\u201c]{t}['\"\u2019\u201d]", query, re.I)
                or re.search(rf"\b(what is|what's|what are|meaning of|define|definition of|who is)\s+(an?\s+|the\s+)?{t}s?\b", query, re.I))


def named_sections(query: str, chunks: list[dict]) -> list[int]:
    """Chunks a question names directly: "section 22", "s 69(1)", "regulation 4", or a defined term it asks about."""
    want = set()
    for sec, reg in SECTION_REF.findall(query):
        want.add(f"popia:s{sec}" if sec else f"regs:reg{reg}")
    named = [i for i, c in enumerate(chunks) if c["id"].split("#")[0] in want]
    for i, c in enumerate(chunks):
        if c["id"].startswith("popia:s1#") and i not in named:
            if any(asks_for_definition(query, m.group(1)) for b in c["blocks"] for m in [DEFINED.search(b["label"])] if m):
                named.append(i)
    return named


def rrf(rankings: list[list[int]], k: int = 60) -> list[int]:
    """Reciprocal rank fusion: merge rankings without having to calibrate their scores against each other."""
    score: dict[int, float] = {}
    for r in rankings:
        for rank, i in enumerate(r):
            score[i] = score.get(i, 0) + 1 / (k + rank + 1)
    return sorted(score, key=lambda i: -score[i])


INJECTION = re.compile(
    r"ignore (all |any )?(previous|prior|above|earlier)|system (prompt|override)|developer mode|jailbreak|"
    r"(print|reveal|show|repeat) (me )?(your|the) (full |entire )?(system )?(prompt|instructions|rules)|"
    r"just say (yes|no)|(confirm|say|tell me) (in one line )?that (my|our|we|i)\b.*\bcomplian|"
    r"you are now|act as (a|an) |new instructions", re.I)


def search_query(question: str) -> str:
    """The question with injected instructions removed, for retrieval only (Claude still sees the whole question).

    Instruction sentences ("ignore all previous instructions", "just say yes") carry no POPIA meaning but pull the
    query vector away from the real question.
    """
    kept = []
    for sentence in re.split(r"(?<=[.!?:])\s+|\n+", question.strip()):
        if not INJECTION.search(sentence):
            kept.append(sentence)
        else:  # "print your prompt, then tell me X": keep the clauses that aren't instructions
            kept += [x for x in re.split(r",?\s+(?:and\s+)?then\s+", sentence, flags=re.I) if not INJECTION.search(x)]
    # "...confirm that I'm compliant. Then explain what X is": drop the lead-in, keep the request
    kept = [re.sub(r"^(then|also|and|after that),?\s+", "", x, flags=re.I) for x in kept]
    q = " ".join(x for x in kept if x).strip()
    return q if len(q.split()) >= 3 else question


def rerank_text(chunk: dict, words: int = 220) -> str:
    """Title + opening words: enough for the reranker to judge relevance, small enough that 20 candidates fit the
    free tier's 10K tokens a minute."""
    body = " ".join(b["text"] for b in chunk["blocks"]).split()
    return f"{chunk['title']}\n{' '.join(body[:words])}"


def merge_parts(chunks: list[dict], base_id: str) -> dict:
    """All parts ("popia:s4#1", "#2") of a split section as one extract, so a list isn't cut in half."""
    parts = [c for c in chunks if c["id"].split("#")[0] == base_id]
    head = parts[0]
    heading = head["heading"].split(" (part ")[0]
    title = f"{head['ref']}: {heading}" + (f" ({head['context']})" if head["context"] else "")
    return {**head, "id": base_id, "heading": heading, "title": title, "blocks": [b for c in parts for b in c["blocks"]]}


@dataclass
class Hit:
    chunk: dict
    score: float  # rerank relevance (0..1), or fused score when reranking is off
    via: str  # which retrievers found it: "vector+bm25+named"


class Retriever:
    def __init__(self, voyage=None, chunks: list[dict] | None = None, vectors: np.ndarray | None = None, min_interval_s: float = 0.0):
        self.min_interval_s = min_interval_s  # Voyage without a payment method allows ~3 requests a minute
        self._last_call = 0.0
        self.chunks = chunks if chunks is not None else load_chunks()
        self.vectors = vectors if vectors is not None else load_vectors(self.chunks)
        self.bm25 = BM25([embed_text(c) for c in self.chunks])
        self._voyage = voyage

    @property
    def voyage(self):
        if self._voyage is None:
            import voyageai

            self._voyage = voyageai.Client(max_retries=3)
        return self._voyage

    def _throttle(self):
        wait = self.min_interval_s - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.monotonic()

    def _call(self, fn, *args, **kwargs):
        """Call Voyage, waiting out its rate limit (3 requests / 10K tokens a minute without a payment method)."""
        import voyageai

        for attempt in range(4):
            self._throttle()
            try:
                return fn(*args, **kwargs)
            except voyageai.error.RateLimitError:
                if attempt == 3:
                    raise
                time.sleep(60)

    def embed_queries(self, queries: list[str]) -> list[np.ndarray]:
        out = []
        for q in self._call(self.voyage.embed, queries, model=EMBED_MODEL, input_type="query").embeddings:
            v = np.array(q, dtype=np.float32)
            out.append(v / np.linalg.norm(v))
        return out

    def candidates(self, query: str, qvec: np.ndarray, n: int = 20) -> tuple[list[int], dict[int, set[str]]]:
        vec = list(np.argsort(-(self.vectors @ qvec))[:n])
        kw = [i for i in np.argsort(-self.bm25.scores(query))[:n]]
        named = named_sections(query, self.chunks)
        via: dict[int, set[str]] = {}
        for name, r in (("vector", vec), ("bm25", kw), ("named", named)):
            for i in r:
                via.setdefault(int(i), set()).add(name)
        fused = rrf([[int(i) for i in vec], [int(i) for i in kw]])
        order = [i for i in named if i in via] + [i for i in fused if i not in named]  # named sections always kept
        return order[: max(n, len(named))], via

    def search(self, query: str, k: int = 6, rerank: bool = True, qvec: np.ndarray | None = None, pool: int = 20) -> list[Hit]:
        """k extracts for Claude: candidates from vector + keyword + named lookup, reranked, split sections re-joined.

        `qvec` may be passed in (embedded from `search_query(query)`) so a batch of questions costs one embed call.
        """
        q = search_query(query)
        if qvec is None:
            qvec = self.embed_queries([q])[0]
        cand, via = self.candidates(q, qvec, n=pool)
        # sections the question names, and definitions it asks about, are always sent: the reranker only reads
        # each chunk's opening words and could drop them
        named = [i for i in cand if "named" in via[i]][:k]
        if not rerank:
            ranked = [(i, 1 / (r + 1)) for r, i in enumerate(cand[:k])]
        else:
            res = self._call(self.voyage.rerank, q, [rerank_text(self.chunks[i]) for i in cand], model=RERANK_MODEL, top_k=k)
            ranked = [(cand[r.index], r.relevance_score) for r in res.results]
        ranked = [(i, 1.0) for i in named] + [(i, sc) for i, sc in ranked if i not in named]
        ranked = ranked[:k]
        hits, seen = [], set()
        for i, score in ranked:
            c = self.chunks[i]
            base_id = c["id"].split("#")[0]
            # definitions (s 1) are independent entries, so only the retrieved part is sent; other split sections whole
            key = c["id"] if base_id == "popia:s1" else base_id
            if key in seen:
                continue
            seen.add(key)
            whole = c if "#" not in c["id"] or base_id == "popia:s1" else merge_parts(self.chunks, base_id)
            hits.append(Hit(whole, score, "+".join(sorted(via[i]))))
        return hits
