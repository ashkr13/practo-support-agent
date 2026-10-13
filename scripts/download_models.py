"""ONE-TIME setup: download the two free local models from Hugging Face (~90 MB each).

This is the only step that needs internet. Afterwards everything runs offline
(app/__init__.py sets HF_HUB_OFFLINE=1). Run:  .venv/bin/python scripts/download_models.py
"""
import os

os.environ["HF_HUB_OFFLINE"] = "0"          # explicitly allow network for this setup step only
os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

from sentence_transformers import CrossEncoder, SentenceTransformer  # noqa: E402

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
ANSWERABILITY_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

if __name__ == "__main__":
    print("embedding dim:", SentenceTransformer(EMBEDDING_MODEL).encode(["ok"]).shape[1])
    print("cross-encoder score:", float(CrossEncoder(ANSWERABILITY_MODEL).predict([("q", "p")])[0]))
    print("Models cached. The project now runs fully offline.")
