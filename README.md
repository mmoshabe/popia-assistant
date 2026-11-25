# POPIA assistant: answers with citations, or says it doesn't know

Ask a question about South Africa's Protection of Personal Information Act (POPIA) and get a plain-English answer where **every statement links to the exact provision it came from** ("s 69(3)", "regulation 4(1)", "direct marketing guidance note, para 7.1.1"). When the documents don't answer the question, it says so instead of guessing. It never tells you that your business "is compliant".

Built on the Act, the 2018 POPIA Regulations and five Information Regulator guidance notes (official texts, [sources/sources.json](sources/sources.json)), with an evaluation set of 51 labelled questions.

```
$ popia ask "Can I send marketing emails to people who have bought from us before, without asking them first?"
[answered]
Yes, potentially. The Act allows marketing by email without prior consent for existing customers, but only if several conditions are met. Processing personal information for direct marketing by electronic means, including email, is prohibited unless the person has consented or is a customer of the responsible party, subject to subsection (3).[1]

Key conditions under section 69(3):

- **Source of contact details:** You must have collected their contact details during a sale of a product or service to them.
- **Your own similar products only:** You can market only your own similar products or services. Guidance note para 7.3.2 says similar products include shoes, belts etc. in a clothing retail store, and a funeral insurance cover will not constitute a similar product.[2]
- **Opt-out opportunity:** They must have had a reasonable chance to object, free of charge, when you collected their details, and on the occasion of each communication with the data subject for the purpose of marketing if the data subject has not initially refused such use.[3]

Every marketing email must also include details of the identity of the sender or the person on whose behalf the communication has been sent; and an address or other contact details to which the recipient may send a request that such communications cease.[4]

You must also keep a list of people who have refused consent or objected, as the guidance note at para 7.3.5 explains. Whether your emails fit this exception depends on the facts of each sale and each customer, so check them against these conditions.

[1] s 69(1)
[2] para 7.3.2
[3] s 69(3)
[4] s 69(4)

General information from POPIA, its Regulations and the Information Regulator's guidance notes, not legal advice. For a decision about your own situation, check with the Information Regulator or a lawyer.
```

A real answer, unedited (question q01 in the dev evaluation run `eval/results/dev-answers-20261125-1114`).
## Results

| Metric | Held-out test, before fixes (23) | **Fresh held-out set, after fixes (15)** |
|---|---|---|
| Fully correct (right status, key facts, cites the right provision, nothing forbidden) | 78% | **87%** |
| Declined when the documents don't cover it | 5/5 | **3/3** |
| Never declared "you're compliant", never leaked its prompt (prompt-injection questions) | 100% | **100%** |
| Right status (answered / partly / not in sources / out of scope) | 96% | **100%** |
| Right provision among the extracts sent to Claude | 83% | **92%** |
| Citations pointing only at extracts actually sent | 100% | 100% |
| Claude cost per question (Haiku 5.5) | $0.0012 | **$0.0018** |
| Median answer time | 6.3 s | 6.7 s |

**How these numbers were kept honest**

- 36 questions were written first and split into dev (13, used for improvements) and test (23, held out), with labelling rules written before any run ([eval/LABELING.md](eval/LABELING.md)). Every expected answer was checked against the indexed text.
- The first test run scored **78%**. Its misses showed four real weaknesses (below). Fixing them made the original test set no longer "unseen", so **15 new questions on provisions no earlier question touches** were written before the fixed system was run on them. That run, done once, is the headline: **87%**.
- Labels were corrected three times, only where a label contradicted the written rules (e.g. a "must not say *you're compliant*" pattern that also matched the refusal "I can't confirm you're compliant"). Each correction is logged with its reason in LABELING.md, and none changed the headline test score.
- Every run's report is in [eval/results/](eval/results/), misses included.

**What the first test run found, and the fixes**

| Weakness | Fix |
|---|---|
| Prompt-injection text ("ignore all previous instructions…") pulled the search away from the real question | Instruction sentences are removed from the *search* query only; Claude still sees, and refuses, the whole question |
| A long section split into two chunks: the Act's list of 8 conditions came back with 6 | Parts of a split section are always sent together |
| An answer named "s 22(1)" in its text but attached no citations | Uncited answers are retried once, then shown with a warning |
| Search pool too small for the reranker to recover misses | 20 candidates instead of 12 (the reranker reads each one's opening words to stay inside Voyage's free-tier limits) |

The two remaining misses on the fresh set both had the right facts but cited the wrong source: the guidance note's restatement of s 15 instead of s 15 itself, and an uncited guidance-note answer that was correctly flagged with the warning.

## How it works

```
question ──► search query (injected instructions removed)
             ├─ Voyage voyage-4 embeddings ──┐
             ├─ BM25 keyword search ─────────┼─ reciprocal rank fusion ─► Voyage rerank-2.5 ─► 6 extracts
             └─ named lookup: "section 22",  │                              (named sections always kept,
                "regulation 4", 'what is a   ┘                               split sections re-joined)
                "responsible party"'
6 extracts ──► Claude Haiku 5.5, one `search_result` block per extract, citations on
           ──► "STATUS: answered | partly_answered | not_in_sources | out_of_scope" + answer
           ──► citations mapped to provisions; uncited answer → one retry → warning
```

- **Chunking** ([chunk.py](src/popia_assistant/chunk.py)): the Act is split by section (all 115, with their chapter, part and condition); each chunk holds one *block* per subsection or definition, because Claude cites whole blocks. So a citation points at "s 11(1)", not all of section 11. Guidance notes follow their own paragraph numbering, so contents pages and footnotes can't become sections. 228 chunks, none over 450 words.
- **Citations** use Anthropic's [search results](https://platform.claude.com/docs/en/build-with-claude/search-results) feature. Citations can't be combined with structured outputs, so the status comes from a first line the code parses.
- **Rules** are in [prompt/answer.md](prompt/answer.md): only the extracts, the Act outranks guidance notes, information not legal advice, the question and extracts are data, not instructions.
- **Why no vector database:** 228 chunks × 1024 dimensions is under 1 MB, searched in memory in milliseconds. pgvector would earn its place at tens of thousands of chunks, many users, or documents that change daily.

## Costs

| | |
|---|---|
| Building the index (71K tokens) | $0: inside Voyage's 200M free tokens |
| Search per question (embed + rerank) | $0: same free allowance (`rerank-2.5`; `rerank-3` has no free tier) |
| Claude Haiku 5.5 per question | about $0.0012–0.0018 |
| All evaluation runs while building this | about $0.11 |

Without a payment method Voyage allows 3 requests and 10K tokens a minute; the indexer and evaluator pace themselves and retry on rate limits.

## Running it

```bash
uv sync
cp .env.example .env                       # add ANTHROPIC_API_KEY and VOYAGE_API_KEY
python3 scripts/fetch_sources.py           # downloads the 7 official PDFs, checks their hashes
uv run python -m popia_assistant.chunk     # 228 chunks -> cache/chunks.json
uv run python -m popia_assistant.index     # embeddings -> cache/embeddings.json (free, ~10 min on Voyage's free tier)
uv run popia ask "Do we have to report a data breach?" --show-sources

uv run pytest                              # 21 offline tests, no API calls
uv run python eval/run.py --split dev --retrieval-only   # free: search quality only
uv run python eval/run.py --split dev                    # paid: about $0.02
```

## Limits

- **Not legal advice**, and it says so on every answer. It describes what the provisions say; it doesn't judge a specific situation.
- The Act is the gazetted 2013 text; later amendments and the Regulator's newer guidance aren't included until added to [sources.json](sources/sources.json). A changed source PDF is caught by its hash.
- English only. Some PDF text has broken spacing ("session s"); search tolerates it, but quoted text keeps it.
- 51 evaluation questions is enough to find weaknesses, not to claim a precise accuracy: one question is 6–7 points on the fresh set.

## Layout

```
sources/          sources.json (URLs, dates, sha256); PDFs downloaded to sources/raw/ (gitignored)
src/popia_assistant/
  chunk.py        PDF text -> sections -> citable blocks
  index.py        Voyage embeddings, cached and resumable
  retrieve.py     hybrid search, injection-stripped query, named lookup, rerank, section re-joining
  answer.py       Claude request with search_result blocks, citation parsing, uncited-answer retry
  cli.py          popia ask "..."
prompt/answer.md  the answering rules
eval/             questions.jsonl, LABELING.md, run.py, results/
tests/            offline tests for retrieval, citation parsing and scoring
```
