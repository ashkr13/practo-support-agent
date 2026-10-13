"""Milestone 0 smoke tests: the environment is reproducible and offline-ready."""
import sys


def test_python_version_supported_by_crewai():
    assert (3, 10) <= sys.version_info[:2] < (3, 14)


def test_core_libraries_import():
    from crewai.llms.base_llm import BaseLLM  # noqa: F401
    from autogen_agentchat.teams import RoundRobinGroupChat  # noqa: F401
    from langchain_core.chat_history import InMemoryChatMessageHistory  # noqa: F401
    import chromadb, fastapi  # noqa: F401,E401


def test_embedding_model_loads_offline(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    assert model.encode(["test"]).shape == (1, 384)
