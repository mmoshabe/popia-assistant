"""Turn the source PDFs into citable chunks.

A chunk is one retrievable unit (a section of the Act, a regulation, a heading of a guidance note). It holds
*blocks* (a subsection, a definition, a numbered paragraph): Claude cites whole blocks, so the block is the
smallest thing an answer can point at, e.g. "s 11(1)" rather than all of section 11.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[2]
SOURCES = ROOT / "sources"
MAX_WORDS = 450  # long sections are split into several chunks of whole blocks

logging.getLogger("pypdf").setLevel(logging.ERROR)


@dataclass
class Block:
    label: str  # how an answer refers to it: "s 11(1)", "reg 4(1)", "para 7.3"
    text: str


@dataclass
class Chunk:
    id: str  # stable source id passed to Claude, e.g. "popia:s11" or "popia:s1#2"
    doc: str
    doc_title: str
    ref: str  # "POPIA s 11"
    heading: str  # "Consent, justification and objection"
    context: str  # "Chapter 3 > Condition 2: Processing limitation"
    pages: list[int]
    blocks: list[Block] = field(default_factory=list)

    @property
    def title(self) -> str:
        return f"{self.ref}: {self.heading}" + (f" ({self.context})" if self.context else "")

    def text(self) -> str:
        return "\n".join(b.text for b in self.blocks)

    def words(self) -> int:
        return len(self.text().split())


# ---------------------------------------------------------------- extraction

NOISE = [
    re.compile(r"^\d+\s+No\.\s*\d+\s+GOVERNMENT GAZETTE", re.I),
    re.compile(r"GOVERNMENT GAZETTE,\s*\d+", re.I),
    re.compile(r"^Act No\. 4 of 2013 Protection of Personal Information Act, 2013$"),
    re.compile(r"^This gazette is also available free online", re.I),
    re.compile(r"^Page \d+ of \d+$", re.I),
    re.compile(r"^\d{1,3}$"),  # page numbers and the gazette's margin line numbers
]


def clean(line: str) -> str:
    line = line.replace("ﬁ", "fi").replace("ﬂ", "fl").replace("‘‘", "“").replace("’’", "”")
    return re.sub(r"\s+", " ", line).strip()


def pdf_lines(path: Path) -> list[tuple[int, str]]:
    """(page number, cleaned line) for every non-empty, non-boilerplate line."""
    out = []
    for i, page in enumerate(PdfReader(path).pages, start=1):
        for raw in (page.extract_text() or "").splitlines():
            line = clean(raw)
            if line and not any(p.search(line) for p in NOISE):
                out.append((i, line))
    return out


# ---------------------------------------------------------------- Act and Regulations (numbered sections)

SUB = re.compile(r"^\((\d+[A-Z]?)\)\s")
DEFINITION = re.compile(r"^“([^”]{2,80})”")


def section_blocks(num: str, lines: list[str], prefix: str) -> list[Block]:
    """Split a section's lines into blocks: one per subsection, or one per definition in a definitions section."""
    text = re.sub(rf"^{num}\.\s*", "", " ".join(lines[:1])) if lines else ""
    rows = [text] + lines[1:] if lines else []
    blocks: list[Block] = []
    for row in rows:
        m_sub, m_def = SUB.match(row), DEFINITION.match(row)
        if m_sub:
            blocks.append(Block(f"{prefix} {num}({m_sub.group(1)})", row))
        elif m_def and num == "1":
            blocks.append(Block(f'{prefix} 1, definition of "{m_def.group(1)}"', row))
        elif blocks:
            blocks[-1].text += " " + row
        else:
            blocks.append(Block(f"{prefix} {num}", row))
    return [b for b in blocks if b.text.strip()]


def numbered_sections(doc: dict, lines: list[tuple[int, str]], start: int, stop: int, prefix: str, ref: str) -> list[Chunk]:
    """Walk lines[start:stop] and cut at "N. " where N is the next expected section number.

    The line just before a section start is its heading; CHAPTER / Part / Condition lines set the context.
    """
    chunks: list[Chunk] = []
    expected, cur, body, pages = 1, None, [], []
    chapter = part = condition = ""
    pending: list[str] = []  # lines seen since the last section text that might be headings

    def flush():
        if cur is None:
            return
        n, heading, ctx = cur
        chunks.append(Chunk(f"{doc['id']}:{'s' if prefix == 's' else 'reg'}{n}", doc["id"], doc["title"], f"{ref} {n}",
                            heading, ctx, sorted(set(pages)), section_blocks(n, body, prefix)))

    i = start
    while i < stop:
        page, line = lines[i]
        nxt = lines[i + 1][1] if i + 1 < stop else ""
        if re.fullmatch(r"CHAPTER \d+", line):
            chapter, part, condition = f"Chapter {line.split()[1]}: {nxt.title()}", "", ""
            i += 2
            continue
        if re.fullmatch(r"Part [A-Z]", line):
            part, condition = f"{line}: {nxt}", ""
            i += 2
            continue
        if re.fullmatch(r"Condition \d", line):
            condition = f"{line}: {nxt}"
            i += 2
            continue
        m = re.match(rf"^({expected})\.(?=\s|[A-Z(])", line)
        if m:
            heading = body.pop() if body and not body[-1].rstrip().endswith((";", ",", "—", "-", ":", ".")) and len(body[-1]) < 120 else ""
            if heading and heading[0].islower() and body and len(body[-1]) < 120 and not body[-1].rstrip().endswith((";", ",", ".", ":")):
                heading = body.pop() + " " + heading  # heading wrapped onto two lines
            flush()
            cur = (str(expected), heading, " > ".join(x for x in (chapter, part, condition) if x))
            body, pages = [line], [page]
            expected += 1
        else:
            body.append(line)  # before the first section this only collects its heading
            pages.append(page)
        i += 1
    flush()
    return chunks


# ---------------------------------------------------------------- guidance notes (numbered paragraphs)

TOP = re.compile(r"^(\d{1,2})\.?\s+([A-Z][^.;:]{2,100})$")
PARA = re.compile(r"^(\d{1,2}(?:\.\d{1,2})+)\.?\s")


def guidance_chunks(doc: dict, lines: list[tuple[int, str]]) -> list[Chunk]:
    """Sections follow the paragraph numbering: paragraph "7.3" belongs to section 7.

    A section starts only when a numbered paragraph moves to the next number ("8.1" after "7.x"). Heading-shaped
    lines (contents pages, numbered lists, footnotes) never start one; a section without numbered paragraphs stays
    part of the previous one. The heading is the "N. Title" line just before the first paragraph.
    """
    sections: list[dict] = [{"num": 0, "head": "Introduction", "raw": []}]
    in_body = False

    def heading_from(prev: dict, num: int) -> str:
        raw = prev["raw"]
        for k in range(len(raw) - 1, max(len(raw) - 9, -1), -1):
            m = re.match(rf"^{num}\.?\s+([A-Z].{{2,100}})$", raw[k][1])
            if m:
                head = m.group(1).strip()
                nxt = raw[k + 1][1] if k + 1 < len(raw) else ""
                if nxt and len(nxt) < 90 and not nxt.endswith(".") and (nxt.isupper() or nxt[:1].islower()):
                    head += " " + nxt  # heading wrapped onto a second line
                del raw[k:]  # the heading (and anything after it) leaves the previous section
                return head
        return ""

    for page, line in lines:
        cur = sections[-1]
        m_para = PARA.match(line)
        top = int(m_para.group(1).split(".")[0]) if m_para else None  # heading-shaped lines (lists, footnotes) never start one
        if m_para and not in_body and top == 1:
            in_body = True
        if in_body and top in (cur["num"] + 1, cur["num"] + 2):  # +2: a section with no numbered paragraphs
            head = heading_from(cur, top)
            sections.append({"num": top, "head": head or f"Section {top}", "raw": [(page, line)]})
            continue
        cur["raw"].append((page, line))

    chunks = []
    for sec in sections:
        num, raw = sec["num"], sec["raw"]
        blocks: list[Block] = []
        for _, line in raw:
            m = PARA.match(line)
            if m:
                blocks.append(Block(f"para {m.group(1)}", line))
            elif blocks:
                blocks[-1].text += " " + line
            else:
                blocks.append(Block(f"section {num}" if num else "introduction", line))
        if sum(len(b.text.split()) for b in blocks) < 15:
            continue
        ref = f"{short(doc)} {'section ' + str(num) if num else 'introduction'}"
        chunks.append(Chunk(f"{doc['id']}:{num}", doc["id"], doc["title"], ref, sec["head"].capitalize() if sec["head"].isupper() else sec["head"],
                            "", sorted({p for p, _ in raw}), blocks))
    return chunks


def short(doc: dict) -> str:
    return {"gn-io": "Information Officers guidance note", "gn-dm": "Direct marketing guidance note",
            "gn-special": "Special personal information guidance note", "gn-children": "Children's information guidance note",
            "gn-prior": "Prior authorisation guidance note"}.get(doc["id"], doc["title"])


# ---------------------------------------------------------------- splitting long chunks

def split_block(b: Block) -> list[Block]:
    """A single block over MAX_WORDS (e.g. s 40(1), a long list) is cut at its lettered items: "(a) ... (h) ..."."""
    if len(b.text.split()) <= MAX_WORDS:
        return [b]
    pieces = re.split(r"(?=\s\([a-z]{1,3}\)\s)", b.text)
    if max(len(x.split()) for x in pieces) > MAX_WORDS:  # no (a)/(b) items: fall back to sentences
        pieces = re.split(r"(?<=[.;])\s+(?=[A-Z(\u2022\u25cf\uf0b7-]|[ivx]+\.)", b.text)
        pieces = [pieces[0]] + [" " + x for x in pieces[1:]]
    groups: list[list[str]] = [[]]
    for piece in pieces:
        if groups[-1] and len(" ".join(groups[-1] + [piece]).split()) > MAX_WORDS:
            groups.append([])
        groups[-1].append(piece)
    out = []
    for i, g in enumerate(groups):
        first = re.match(r"\s*\(([a-z]{1,3})\)", g[0])
        out.append(Block(b.label if i == 0 or not first else f"{b.label}({first.group(1)})", "".join(g).strip()))
    return out


def split_long(chunk: Chunk) -> list[Chunk]:
    chunk.blocks = [x for b in chunk.blocks for x in split_block(b)]
    if chunk.words() <= MAX_WORDS:
        return [chunk]
    parts, cur = [], []
    for b in chunk.blocks:
        if cur and sum(len(x.text.split()) for x in cur) + len(b.text.split()) > MAX_WORDS:
            parts.append(cur)
            cur = []
        cur.append(b)
    parts.append(cur)
    return [Chunk(f"{chunk.id}#{i + 1}", chunk.doc, chunk.doc_title, chunk.ref,
                  chunk.heading + (f" (part {i + 1} of {len(parts)})" if len(parts) > 1 else ""), chunk.context, chunk.pages, p)
            for i, p in enumerate(parts)]


# ---------------------------------------------------------------- entry point

def build(raw_dir: Path = SOURCES / "raw") -> list[Chunk]:
    docs = json.loads((SOURCES / "sources.json").read_text())
    out: list[Chunk] = []
    for doc in docs:
        lines = pdf_lines(raw_dir / f"{doc['id']}.pdf")
        texts = [t for _, t in lines]
        if doc["id"] == "popia":
            # body starts at the second "CHAPTER 1" (the first is the arrangement of sections); the schedule after
            # s 115 only amends other Acts, so it is left out
            start = [i for i, t in enumerate(texts) if t == "CHAPTER 1"][1]
            stop = next(i for i, t in enumerate(texts) if t.startswith("SCHEDULE") and i > start)
            got = numbered_sections(doc, lines, start, stop, "s", "POPIA s")
        elif doc["id"] == "regs":
            stop = next((i for i, t in enumerate(texts) if re.fullmatch(r"FORM 1", t)), len(texts))
            start = next(i for i, t in enumerate(texts) if re.match(r"^1\.\s", t))
            got = numbered_sections(doc, lines, start - 1, stop, "reg", "Regulation")
        else:
            got = guidance_chunks(doc, lines)
        for c in got:
            out.extend(split_long(c))
    return out


def main() -> None:
    chunks = build()
    out = ROOT / "cache" / "chunks.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps([asdict(c) | {"title": c.title} for c in chunks], indent=1, ensure_ascii=False))
    by_doc: dict[str, list[Chunk]] = {}
    for c in chunks:
        by_doc.setdefault(c.doc, []).append(c)
    for d, cs in by_doc.items():
        w = [c.words() for c in cs]
        print(f"{d:<12} {len(cs):>4} chunks {sum(len(c.blocks) for c in cs):>5} blocks  words/chunk median {sorted(w)[len(w) // 2]:>4} max {max(w):>4}")
    print(f"total        {len(chunks):>4} chunks -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
