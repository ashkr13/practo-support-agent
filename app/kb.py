"""Load the knowledge-base documents from kb/*.md (Task 2)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

KB_DIR = Path(__file__).resolve().parent.parent / "kb"

# Split after ., ! or ? followed by whitespace. KB text avoids abbreviations, so this is safe.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Document:
    doc_id: str  # filename stem, e.g. "02_cancellation_rescheduling"
    title: str
    text: str  # body without the heading


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_END.split(text.strip()) if s.strip()]


def load_documents(kb_dir: Path = KB_DIR) -> list[Document]:
    docs = []
    for path in sorted(kb_dir.glob("*.md")):
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        title = lines[0].lstrip("# ").strip()
        body = " ".join(line.strip() for line in lines[1:] if line.strip())
        docs.append(Document(doc_id=path.stem, title=title, text=body))
    return docs
