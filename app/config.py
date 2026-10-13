"""Central configuration, read from environment variables (.env) with safe defaults."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# Zero-network env switches are set in app/__init__.py (runs before any app submodule import).

MOCK_LLM: bool = os.getenv("MOCK_LLM", "true").strip().lower() in {"1", "true", "yes"}
CHROMA_PATH: Path = ROOT / os.getenv("CHROMA_PATH", "chroma_db")
LOG_PATH: Path = ROOT / os.getenv("LOG_PATH", "logs/requests.jsonl")
MAX_TOKENS_PER_REQUEST: int = int(os.getenv("MAX_TOKENS_PER_REQUEST", "10000"))

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Cross-encoder used by the output groundedness check (v2) to score "does this passage answer this question?"
ANSWERABILITY_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# Calibrated by scripts/task10_calibrate_answerability.py on already-seen questions (dev set + eval set A),
# red-flag emergency questions excluded (handled by the emergency guardrail):
#   in-scope min -0.606 vs near/out-of-domain max -1.481 -> midpoint -1.04
ANSWERABILITY_THRESHOLD = -1.04

# Chunking parameters (Task 3).
FIXED_CHUNK_WORDS = 30
FIXED_CHUNK_OVERLAP_WORDS = 10

TOP_K = 3

# Chunking strategy used by the deployed agent, chosen in Task 5
# (transcripts/task05_chunking_evaluation.txt): equal recall, complete-sentence answers.
DEPLOY_STRATEGY = "sentence"

# "I don't know" thresholds on top-1 cosine similarity (Task 4), calibrated empirically by
# scripts/task04_calibrate_threshold.py (transcripts/task04_threshold_calibration.txt):
#   fixed:    in-scope min 0.5910 vs out-of-scope max 0.1617 -> midpoint 0.3763 -> 0.38
#   sentence: in-scope min 0.5857 vs out-of-scope max 0.1587 -> midpoint 0.3722 -> 0.37
SIMILARITY_THRESHOLD: dict[str, float] = {"fixed": 0.38, "sentence": 0.37}

if not MOCK_LLM:
    raise NotImplementedError(
        "Only MOCK_LLM=true is implemented. A real LLM can be added behind this flag "
        "(see README 'Extending with a real LLM')."
    )
