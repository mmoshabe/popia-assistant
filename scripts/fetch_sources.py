"""Download the official source PDFs listed in sources/sources.json into sources/raw/ (gitignored).

    python3 scripts/fetch_sources.py          # download missing files, check hashes
    python3 scripts/fetch_sources.py --pin    # record sha256 of the files on disk into sources.json

A changed hash means the publisher replaced the document: re-check citations before re-indexing.
"""

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "sources" / "sources.json"
RAW = ROOT / "sources" / "raw"


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    sources = json.loads(MANIFEST.read_text())
    RAW.mkdir(parents=True, exist_ok=True)
    bad = 0
    for s in sources:
        dest = RAW / f"{s['id']}.pdf"
        if not dest.exists():
            data = b""
            for attempt in range(3):
                try:
                    req = urllib.request.Request(s["url"], headers={"User-Agent": "Mozilla/5.0 (popia-assistant source fetch)"})
                    with urllib.request.urlopen(req, timeout=60) as r:
                        data = r.read()
                    break
                except OSError as e:  # timeouts and TLS hiccups on government servers are common: retry
                    print(f"retry {s['id']} ({attempt + 1}/3): {e}")
            if not data.startswith(b"%PDF"):
                print(f"FAIL {s['id']}: not a PDF")
                bad += 1
                continue
            dest.write_bytes(data)
        digest = sha256(dest)
        if "--pin" in sys.argv:
            s["sha256"] = digest
        elif s.get("sha256") and s["sha256"] != digest:
            print(f"CHANGED {s['id']}: hash differs from the pinned one")
            bad += 1
        print(f"ok   {s['id']:<12} {dest.stat().st_size // 1024:>5} KB")
    if "--pin" in sys.argv:
        MANIFEST.write_text(json.dumps(sources, indent=1, ensure_ascii=False) + "\n")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
