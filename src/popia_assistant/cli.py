"""popia ask "Can I send marketing SMSes to people who bought from me?"   (add --show-sources to see retrieval)"""

from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv

from .chunk import ROOT


def main() -> int:
    load_dotenv(ROOT / ".env")
    ap = argparse.ArgumentParser(prog="popia")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ask = sub.add_parser("ask", help="answer a question with citations")
    ask.add_argument("question")
    ask.add_argument("--k", type=int, default=6, help="extracts sent to Claude")
    ask.add_argument("--model", default=None)
    ask.add_argument("--show-sources", action="store_true")
    a = ap.parse_args()

    from .answer import MODEL, answer
    from .retrieve import Retriever

    hits = Retriever().search(a.question, k=a.k)
    if a.show_sources:
        for h in hits:
            print(f"  {h.score:.2f}  {h.via:<18} {h.chunk['title'][:90]}", file=sys.stderr)
    ans = answer(a.question, hits, model=a.model or MODEL)
    print(f"[{ans.status}]\n{ans.render()}")
    print(f"\n({ans.input_tokens} in / {ans.output_tokens} out tokens, ${ans.cost:.4f})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
