"""The eval's scoring rules, checked on hand-made runs (no API calls)."""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("evalrun", ROOT / "eval" / "run.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)

Q = {"id": "q07", "split": "dev", "category": "act", "question": "biggest fine?", "expect_status": ["answered"],
     "expect_sources": ["popia:s109"], "must_include": ["R ?10 ?million|R ?10 000 000"], "must_not": []}


def run(**kw):
    base = {"id": "q07", "retrieved": ["popia:s107", "popia:s109#1"], "status": "answered",
            "text": "Up to R10 million.[1]", "citations": [{"source": "popia:s109#1", "labels": ["s 109(2)"]}]}
    return base | kw


def test_correct_answer_scores_correct():
    s = ev.score(Q, run())
    assert s["retrieval_hit"] and s["first_relevant_rank"] == 2 and s["correct"]


def test_right_status_but_wrong_fact_is_not_correct():
    s = ev.score(Q, run(text="Up to R5 million.[1]"))
    assert s["status_ok"] and not s["facts_ok"] and not s["correct"]


def test_uncited_answer_is_not_correct():
    s = ev.score(Q, run(citations=[]))
    assert s["cites_expected_source"] is False and not s["correct"]


def test_citing_an_extract_that_was_not_sent_is_flagged():
    s = ev.score(Q, run(citations=[{"source": "popia:s1#2", "labels": ["s 1"]}]))
    assert s["citations_from_sent_extracts"] is False


def test_abstain_question_needs_a_decline_status():
    q = {**Q, "category": "abstain", "expect_status": ["not_in_sources", "partly_answered"], "expect_sources": [],
         "must_include": [], "must_not": ["R ?\\d"]}
    good = ev.score(q, run(status="not_in_sources", text="The documents don't say.", citations=[]))
    bad = ev.score(q, run(status="answered", text="It costs R100.", citations=[]))
    assert good["correct"] and good["abstained"]
    assert not bad["correct"] and not bad["abstained"]


def test_labels_are_well_formed():
    qs = [json.loads(line) for line in (ROOT / "eval" / "questions.jsonl").read_text().splitlines() if line.strip()]
    assert len({q["id"] for q in qs}) == len(qs)
    for q in qs:
        assert q["split"] in ("dev", "test", "test2") and q["expect_status"]
        assert bool(q["expect_sources"]) == (q["category"] not in ("abstain", "out_of_scope")), q["id"]
