"""Pin the corpus: record exactly which version of each source document is in use.

A content hash is a fingerprint of a file's bytes. Change one character and the
hash changes completely, so comparing hashes tells you whether a document is
byte-for-byte what it was when someone last worked against it.

This matters because chunk-level relevance judgements are tied to a specific
version of the corpus. If a Home Affairs page is re-downloaded and one sentence
has changed, every judgement made against the old text is quietly wrong. Without
a manifest nobody finds out; with one, --verify says so immediately.

Usage:
    python manifest.py            # build data/documents/MANIFEST.json
    python manifest.py --verify   # re-hash and report any file that changed
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "data" / "documents"
MANIFEST = DOCS / "MANIFEST.json"

HEADER_KEYS = ("Source:", "Retrieved:", "Document:")


def sha256_of(path: Path) -> str:
    """Hash the raw bytes, not decoded text.

    Bytes, because a difference in encoding or line endings is a real difference
    in the file even when the decoded text looks identical.
    """
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def read_headers(path: Path) -> dict:
    """Pull Source / Retrieved / Document out of the first lines.

    Read, never invented. A document without provenance headers is flagged rather
    than given a plausible-looking default.
    """
    meta = {}
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped:
            if meta:
                break
            continue
        matched = next((k for k in HEADER_KEYS if stripped.startswith(k)), None)
        if matched is None:
            break
        key, _, value = stripped.partition(":")
        meta[key.strip().lower()] = value.strip()
    return meta


def describe(path: Path) -> dict:
    raw = path.read_bytes()
    meta = read_headers(path)
    missing = [k.rstrip(":").lower() for k in HEADER_KEYS if k.rstrip(":").lower() not in meta]
    return {
        "filename": path.name,
        "sha256": sha256_of(path),
        "source_url": meta.get("source", ""),
        "retrieved_at": meta.get("retrieved", ""),
        "document_title": meta.get("document", ""),
        "bytes": len(raw),
        "lines": len(raw.decode().splitlines()),
        "missing_headers": missing,
    }


def build() -> dict:
    paths = sorted(DOCS.glob("*.txt"))
    if not paths:
        raise SystemExit(f"No .txt documents found in {DOCS}")
    entries = [describe(p) for p in paths]

    # Deliberately the same computation as corpus_snapshot_id() in ingest.py:
    # filename then raw bytes, in sorted order. Matching it means the short id
    # below is exactly the directory name under data/snapshots/, so you can tell
    # at a glance which snapshots were built from this corpus version and which
    # are stale. Hashing the per-file digests instead would give a different
    # value and lose that link.
    corpus_hash = hashlib.sha256()
    for path in paths:
        corpus_hash.update(path.name.encode())
        corpus_hash.update(path.read_bytes())
    digest = corpus_hash.hexdigest()

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_documents": len(entries),
        "corpus_sha256": digest,
        "snapshot_id": digest[:12],
        "documents": entries,
    }


def verify() -> int:
    """Re-hash every document and report drift. Returns a process exit code."""
    if not MANIFEST.exists():
        print(f"No manifest at {MANIFEST}. Run without --verify first.")
        return 1

    recorded = json.loads(MANIFEST.read_text())
    by_name = {d["filename"]: d for d in recorded["documents"]}
    on_disk = {p.name: p for p in sorted(DOCS.glob("*.txt"))}

    problems = []
    for name, entry in by_name.items():
        if name not in on_disk:
            problems.append(f"MISSING   {name} — recorded in the manifest, not on disk")
            continue
        current = sha256_of(on_disk[name])
        if current != entry["sha256"]:
            problems.append(
                f"CHANGED   {name}\n"
                f"            recorded {entry['sha256'][:16]}…\n"
                f"            current  {current[:16]}…"
            )
        else:
            print(f"  ok        {name}")
    for name in on_disk:
        if name not in by_name:
            problems.append(f"UNTRACKED {name} — on disk, not in the manifest")

    if problems:
        print("\nDRIFT DETECTED:\n")
        for p in problems:
            print(f"  {p}")
        print(
            "\nAny relevance judgement made against a changed document is no longer valid.\n"
            "Re-judge those questions, or restore the recorded version."
        )
        return 1

    print(f"\nNo drift. All {len(by_name)} documents match the manifest.")
    print(f"snapshot_id: {recorded.get('snapshot_id', recorded['corpus_sha256'][:12])}")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verify", action="store_true",
                    help="re-hash and report any document that changed")
    args = ap.parse_args()

    if args.verify:
        raise SystemExit(verify())

    data = build()
    MANIFEST.write_text(json.dumps(data, indent=2) + "\n")
    print(f"Wrote {MANIFEST.relative_to(ROOT)}")
    print(f"  {data['n_documents']} documents · snapshot_id {data['snapshot_id']}\n")
    for d in data["documents"]:
        flag = f"  ⚠ missing headers: {', '.join(d['missing_headers'])}" if d["missing_headers"] else ""
        print(f"  {d['filename']:<46} {d['bytes']:>6} bytes  {d['sha256'][:12]}…{flag}")


if __name__ == "__main__":
    main()
