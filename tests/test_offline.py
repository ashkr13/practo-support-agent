"""NFR1: the full crew runs with ALL outbound network connections blocked."""
import socket

import pytest

from app.crew import run_crew
from app.mock_llm import MockCrewLLM
from app.rag import build_index


class NetworkBlocked(RuntimeError):
    pass


@pytest.fixture
def no_network(monkeypatch):
    attempts = []

    def refuse(self, address, *args, **kwargs):
        attempts.append(address)
        raise NetworkBlocked(f"network call attempted to {address}")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    yield attempts


def test_crew_kickoff_makes_no_network_calls(no_network):
    build_index()
    run = run_crew("Can I reschedule APT-1003, and what is the cancellation window?",
                   MockCrewLLM(model="mock-react-llm"))
    assert run.response.grounded
    assert no_network == [], f"outbound connections attempted: {no_network}"


def test_rag_import_first_in_fresh_process_makes_no_network_calls():
    """Regression: importing app.rag FIRST (before app.config) once let huggingface_hub go online."""
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    probe = (
        "import socket\n"
        "attempts = []\n"
        "def refuse(self, address, *a, **k):\n"
        "    attempts.append(address); raise OSError('network blocked')\n"
        "socket.socket.connect = refuse\n"
        "socket.socket.connect_ex = refuse\n"
        "from app.rag import build_index, grounded_answer\n"
        "build_index()\n"
        "grounded_answer('How long will my thyroid test results take?')\n"
        "print('ATTEMPTS', attempts)\n"
    )
    env = {k: v for k, v in __import__("os").environ.items()
           if k not in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "CREWAI_DISABLE_TELEMETRY",
                        "OTEL_SDK_DISABLED", "HF_HUB_DISABLE_TELEMETRY", "ANONYMIZED_TELEMETRY")}
    out = subprocess.run([sys.executable, "-c", probe], cwd=root, env=env, capture_output=True, text=True,
                         timeout=300)
    assert "ATTEMPTS []" in out.stdout, out.stdout[-500:] + out.stderr[-1500:]
    assert "unauthenticated requests to the HF Hub" not in out.stderr
