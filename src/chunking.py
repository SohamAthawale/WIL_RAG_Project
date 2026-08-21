"""Chunking strategies behind one interface.

Walert sidestepped chunking - their knowledge base was an FAQ, already segmented
one-entry-per-passage. Our sources are unstructured government pages, so chunking
is a live design variable and therefore an evaluation dimension they did not have.

Strategies: fixed_size, paragraph (the original behaviour), structure_aware, sentence_window.

D9: prepend_metadata=True prefixes each chunk with its subclass identifier BEFORE
embedding, so the identifier is part of the embedded content rather than a stored
field. Run as a controlled before/after - the delta is the result.
"""

import re
from pathlib import Path

# Derived from each document's filename and its "Document:" header line, checked
# against the source text. Explicit rather than parsed: five documents, and a wrong
# label would silently corrupt every chunk in a run.
SUBCLASS_LABELS = {
    "subclass_500_student": "Subclass 500 — Student visa",
    "subclass_485_temporary_graduate": "Subclass 485 — Temporary Graduate visa",
    "subclass_482_skills_in_demand": "Subclass 482 — Skills in Demand visa",
    "subclass_417_working_holiday": "Subclass 417 — Working Holiday visa",
    "whm_6_month_work_limitation": "Condition 8547 — WHM 6 month work limitation (Subclass 417 and 462)",
}

HEADER_KEYS = ("Source:", "Retrieved:", "Document:")


def parse_document(text: str) -> tuple[dict, str]:
    """Split the provenance header from the body."""
    meta: dict[str, str] = {}
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if any(line.startswith(k) for k in HEADER_KEYS):
            key, _, val = line.partition(":")
            meta[key.strip().lower()] = val.strip()
            i += 1
        elif not line and meta:
            i += 1
            break
        elif not line:
            i += 1
        else:
            break
    return meta, "\n".join(lines[i:]).strip()


def label_for(stem: str, meta: dict) -> str:
    if stem in SUBCLASS_LABELS:
        return SUBCLASS_LABELS[stem]
    doc = meta.get("document", stem)
    return doc.split("—")[0].strip() or stem


def is_heading(line: str) -> bool:
    """A section heading, as these Home Affairs pages use them.

    'Stay' is a heading; 'Stay: Up to 4 years' is content and must stay attached
    to the stream above it - splitting those apart is what produced the q02 failure.
    """
    t = line.strip()
    if not t or len(t) > 70:
        return False
    if t.startswith(("-", "•", "*")):
        return False
    if t.startswith("Note:"):
        return True
    if ":" in t and t.split(":", 1)[1].strip():
        return False
    if not t[0].isupper():
        return False
    return not t.endswith((".", ",", ";"))


def _merge_short(chunks: list[str], min_chars: int) -> list[str]:
    out: list[str] = []
    for c in chunks:
        if out and len(out[-1]) < min_chars:
            out[-1] = f"{out[-1]}\n\n{c}".strip()
        else:
            out.append(c)
    if len(out) > 1 and len(out[-1]) < min_chars:
        out[-2] = f"{out[-2]}\n\n{out[-1]}".strip()
        out.pop()
    return out


def _has_bullets(text: str) -> bool:
    return any(line.lstrip().startswith(("-", "•", "*")) for line in text.splitlines())


def _bind_units(section: str) -> list[str]:
    """Group paragraphs into atomic units that must not be split across chunks.

    Two binding rules, both aimed at the same failure: a chunk that states a rule
    without the exemptions that qualify it. That is not merely incomplete - acting
    on it could cause a visa breach.

    Forward:  a paragraph ending in ':' binds to the block it introduces.
    Backward: a block containing bullets binds to the paragraph that follows it,
              because that paragraph usually states the rule the list qualifies.

    The backward rule is what keeps the condition 8547 exemption list attached to
    "If work does not fall within an exemption, you can only work ... 6 months".
    Without it, that sentence is orphaned as a 134-character chunk - measured
    across 30 of 34 sweep configurations before this rule existed.

    A unit is never split, even if it exceeds max_chars. Correctness beats size.
    """
    paras = [p.strip() for p in re.split(r"\n\s*\n", section) if p.strip()]
    units: list[str] = []
    i = 0
    while i < len(paras):
        buf = [paras[i]]
        while buf[-1].rstrip().endswith(":") and i + 1 < len(paras):
            i += 1
            buf.append(paras[i])
        if _has_bullets(buf[-1]) and i + 1 < len(paras):
            i += 1
            buf.append(paras[i])
        units.append("\n\n".join(buf))
        i += 1
    return units


def paragraph(text: str, min_chars: int = 200, **_) -> list[str]:
    """Split on blank lines, merging short paragraphs forward.

    Accumulates over bound *units* rather than raw paragraphs, so a rule never
    gets separated from the exemption list that qualifies it. See _bind_units.
    """
    parts = _bind_units(text)
    chunks: list[str] = []
    buf = ""
    for part in parts:
        buf = f"{buf}\n\n{part}".strip() if buf else part
        if len(buf) >= min_chars:
            chunks.append(buf)
            buf = ""
    if buf:
        if chunks:
            chunks[-1] = f"{chunks[-1]}\n\n{buf}"
        else:
            chunks.append(buf)
    return chunks


def fixed_size(text: str, chunk_size: int = 600, overlap: float = 0.1, **_) -> list[str]:
    """N characters with proportional overlap, snapped to whitespace."""
    step = max(1, int(chunk_size * (1 - overlap)))
    chunks, i = [], 0
    while i < len(text):
        piece = text[i : i + chunk_size]
        if i + chunk_size < len(text):
            cut = piece.rfind(" ")
            if cut > chunk_size * 0.6:
                piece = piece[:cut]
        piece = piece.strip()
        if piece:
            chunks.append(piece)
        i += step
    return chunks


def structure_aware(text: str, min_chars: int = 120, max_chars: int = 900, **_) -> list[str]:
    """Split at heading lines; cap section size at paragraph boundaries."""
    sections: list[list[str]] = []
    current: list[str] = []
    for line in text.splitlines():
        if is_heading(line) and current and any(x.strip() for x in current):
            sections.append(current)
            current = [line]
        else:
            current.append(line)
    if current:
        sections.append(current)

    chunks: list[str] = []
    for sec in sections:
        body = "\n".join(sec).strip()
        if not body:
            continue
        if len(body) <= max_chars:
            chunks.append(body)
            continue
        packed = ""
        for unit in _bind_units(body):
            if packed and len(packed) + len(unit) + 2 > max_chars:
                chunks.append(packed)
                packed = unit
            else:
                packed = f"{packed}\n\n{unit}".strip() if packed else unit
        if packed:
            chunks.append(packed)
    return _merge_short([c for c in chunks if c], min_chars)


def sentence_window(text: str, window: int = 3, **_) -> list[str]:
    """Sliding window of N sentences."""
    flat = re.sub(r"\s+", " ", text).strip()
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", flat) if s.strip()]
    if len(sents) <= window:
        return [" ".join(sents)] if sents else []
    return [" ".join(sents[i : i + window]) for i in range(0, len(sents) - window + 1)]


STRATEGIES = {
    "fixed_size": fixed_size,
    "paragraph": paragraph,
    "structure_aware": structure_aware,
    "sentence_window": sentence_window,
}


def chunk_document(
    path: Path, strategy: str = "structure_aware", prepend_metadata: bool = False, **params
) -> list[dict]:
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy {strategy!r}. Options: {sorted(STRATEGIES)}")
    raw = path.read_text()
    meta, body = parse_document(raw)
    label = label_for(path.stem, meta)
    texts = STRATEGIES[strategy](body, **params)
    out = []
    for i, t in enumerate(texts):
        out.append({
            "id": f"{path.stem}::{i}",
            "source": path.name,
            "subclass_label": label,
            "source_url": meta.get("source", ""),
            "retrieved_at": meta.get("retrieved", ""),
            "text": f"[{label}] {t}" if prepend_metadata else t,
        })
    return out


if __name__ == "__main__":
    docs = sorted((Path(__file__).resolve().parent.parent / "data" / "documents").glob("*.txt"))
    print(f"{'strategy':<18} {'doc':<40} {'chunks':>7} {'median chars':>13}")
    print("-" * 82)
    for strat in STRATEGIES:
        total = 0
        for d in docs:
            cs = chunk_document(d, strat)
            lens = sorted(len(c["text"]) for c in cs)
            total += len(cs)
            print(f"{strat:<18} {d.name:<40} {len(cs):>7} {lens[len(lens)//2]:>13}")
        print(f"{'':<18} {'TOTAL':<40} {total:>7}")
        print("-" * 82)
