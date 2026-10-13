"""Practo Domain Support Agent.

Zero-network switches are set HERE, at package import, so they take effect before any submodule
imports crewai, chromadb or huggingface libraries -- regardless of which `app.*` module is imported
first. (Libraries read these variables at import time; setting them later is too late.)
"""

import os

os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")  # ChromaDB
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
