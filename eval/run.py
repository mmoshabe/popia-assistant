"""Run the evaluation (see eval/LABELING.md).

    uv run python eval/run.py --split dev --retrieval-only     # $0: Voyage only (free allowance), no Claude calls
    uv run python eval/run.py --split dev                      # retrieval + Claude answers (ask before running: costs money)
    uv run python eval/run.py --rescore eval/results/<file>.json   # re-score saved answers for free

Saved results hold every question's retrieved chunk ids, the answer text, status and citations, so scoring changes
never need new API calls.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from popia_assistant.answer import MODEL, PRICE  # noqa: E402

ANSWERABLE = {"act", "regs", "guidance", "multi", "adversarial"}


def base(cid: str) -> str:
    return cid.split("#")[0]


def score(q: dict, r: dict) -> dict:
    """One question's checks. r: {"retrieved": [ids], "status", "text", "citations": [{"source", "labels"}]}."""
    exp = set(q["expect_sources"])
    got = [base(c) for c in r["retrieved"]]
    s = {"id": q["id"], "split": q["split"], "category": q["category"]}
    if exp:
        s["retrieval_hit"] = bool(exp & set(got))
        s["first_relevant_rank"] = next((i + 1 for i, c in enumerate(got) if c in exp), None)
    if "status" not in r:
        return s
    text = r["text"]
    cited = {base(c["source"]) for c in r["citations"]}
    s["status_ok"] = r["status"] in q["expect_status"]
    s["citations_from_sent_extracts"] = cited <= set(got)
    if q["category"] in ANSWERABLE and exp:
        s["cites_expected_source"] = bool(exp & cited)
        s["has_citations"] = bool(cited)
    s["facts_ok"] = all(re.search(p, text, re.I | re.M) for p in q["must_include"])
    s["forbidden_absent"] = not any(re.search(p, text, re.I | re.M) for p in q["must_not"])
    if q["category"] in ("abstain", "out_of_scope"):
        s["abstained"] = r["status"] in ("not_in_sources", "out_of_scope", "partly_answered") and s["forbidden_absent"]
    s["correct"] = s["status_ok"] and s["facts_ok"] and s["forbidden_absent"] and s.get("cites_expected_source", True)
    return s


def pct(xs: list[bool]) -> str:
    return f"{100 * sum(xs) / len(xs):.0f}% ({sum(xs)}/{len(xs)})" if xs else "n/a"


def summarise(scores: list[dict], runs: list[dict], k: int) -> dict:
    get = lambda key: [s[key] for s in scores if key in s]  # noqa: E731
    m = {"questions": len(scores), f"retrieval_recall@{k}": pct(get("retrieval_hit"))}
    ranks = [s["first_relevant_rank"] for s in scores if s.get("first_relevant_rank")]
    if ranks:
        m["mrr"] = round(sum(1 / r for r in ranks) / len(get("retrieval_hit")), 3)
    if any("status" in r for r in runs):
        m |= {
            "fully_correct": pct(get("correct")),
            "status_accuracy": pct(get("status_ok")),
            "cites_expected_source": pct(get("cites_expected_source")),
            "key_facts_present": pct([s["facts_ok"] for s in scores if s["category"] in ANSWERABLE and "facts_ok" in s]),
            "abstained_when_it_should": pct(get("abstained")),
            "no_forbidden_output": pct(get("forbidden_absent")),
            "citations_only_from_sent_extracts": pct(get("citations_from_sent_extracts")),
            "answers_retried_for_missing_citations": sum(1 for r in runs if r.get("retried")),
            "unverified_answers_shown": sum(1 for r in runs if r.get("verified") is False),
        }
        tin = sum(r.get("input_tokens", 0) for r in runs)
        tout = sum(r.get("output_tokens", 0) for r in runs)
        model = runs[0].get("model", MODEL)
        p_in, p_out = PRICE.get(model, (0, 0))
        m["model"] = model
        m["claude_cost_usd_total"] = round((tin * p_in + tout * p_out) / 1e6, 5)
        m["claude_cost_usd_per_question"] = round((tin * p_in + tout * p_out) / 1e6 / max(1, len(runs)), 5)
        lat = sorted(r["seconds"] for r in runs if "seconds" in r)
        if lat:
            m["latency_s_p50"] = round(lat[len(lat) // 2], 1)
    return m


def report(path: Path, qs: list[dict], runs: list[dict], k: int, label: str) -> None:
    by = {q["id"]: q for q in qs}
    scores = [score(by[r["id"]], r) for r in runs]
    m = summarise(scores, runs, k)
    cats = sorted({s["category"] for s in scores})
    lines = [f"# {label}", "", "| Metric | Value |", "|---|---|", *[f"| {a} | {b} |" for a, b in m.items()], ""]
    if any("correct" in s for s in scores):
        lines += ["| Category | Fully correct |", "|---|---|",
                  *[f"| {c} | {pct([s['correct'] for s in scores if s['category'] == c])} |" for c in cats], ""]
    misses = [s for s in scores if not s.get("correct", s.get("retrieval_hit", True)) or s.get("retrieval_hit") is False]
    lines += [f"## Misses ({len(misses)})", ""]
    for s in misses:
        r = next(x for x in runs if x["id"] == s["id"])
        q = by[s["id"]]
        why = [k2 for k2 in ("retrieval_hit", "status_ok", "cites_expected_source", "facts_ok", "forbidden_absent") if s.get(k2) is False]
        lines.append(f"- **{s['id']}** ({q['category']}) {q['question']}  \n  failed: {', '.join(why)}; "
                     f"status `{r.get('status', '-')}`; retrieved {', '.join(base(c) for c in r['retrieved'][:6])}")
        if r.get("text"):
            lines.append(f"  > {r['text'][:400].replace(chr(10), ' ')}")
    path.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(json.dumps(m, indent=2))
    print(f"report: {path.with_suffix('.md').relative_to(ROOT)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test", "test2", "all"], default="dev")
    ap.add_argument("--ids", default="")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--retrieval-only", action="store_true")
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--voyage-interval", type=float, default=30.0, help="seconds between Voyage calls (free tier: 3 requests and 10K tokens a minute)")
    ap.add_argument("--rescore", type=Path)
    a = ap.parse_args()

    qs = [json.loads(line) for line in (ROOT / "eval" / "questions.jsonl").read_text().splitlines() if line.strip()]
    if a.rescore:
        saved = json.loads(a.rescore.read_text())
        report(a.rescore, qs, saved["runs"], saved["k"], f"{saved['label']} (rescored)")
        return 0
    qs = [q for q in qs if (a.split == "all" or q["split"] == a.split) and (not a.ids or q["id"] in a.ids.split(","))]

    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    from popia_assistant.answer import answer
    from popia_assistant.retrieve import Retriever, search_query

    retriever = Retriever(min_interval_s=a.voyage_interval)
    qvecs = retriever.embed_queries([search_query(q["question"]) for q in qs])  # one request for all questions
    runs = []
    for q, qv in zip(qs, qvecs):
        t0 = time.time()
        hits = retriever.search(q["question"], k=a.k, rerank=not a.no_rerank, qvec=qv)
        run = {"id": q["id"], "retrieved": [h.chunk["id"] for h in hits], "scores": [round(h.score, 3) for h in hits],
               "via": [h.via for h in hits]}
        if not a.retrieval_only:
            t1 = time.time()
            ans = answer(q["question"], hits, model=a.model)
            run |= {"status": ans.status, "text": ans.text, "model": ans.model, "input_tokens": ans.input_tokens,
                    "output_tokens": ans.output_tokens, "seconds": round(time.time() - t1, 2), "stop_reason": ans.stop_reason,
                    "verified": ans.verified, "retried": ans.retried,
                    "citations": [{"n": c.n, "source": c.source, "labels": c.labels} for c in ans.citations]}
        run["retrieval_seconds"] = round((t1 if not a.retrieval_only else time.time()) - t0, 2)
        runs.append(run)
        print(f"{q['id']} {run.get('status', '')} {', '.join(base(c) for c in run['retrieved'][:3])}", file=sys.stderr)

    mode = "retrieval" if a.retrieval_only else a.model
    label = f"{a.split} · {mode} · k={a.k}{' · no rerank' if a.no_rerank else ''}"
    stamp = time.strftime("%Y%m%d-%H%M")
    out = ROOT / "eval" / "results" / f"{a.split}-{'retrieval' if a.retrieval_only else 'answers'}-{stamp}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"label": label, "k": a.k, "runs": runs}, indent=1))
    report(out, qs, runs, a.k, label)
    return 0


if __name__ == "__main__":
    sys.exit(main())
