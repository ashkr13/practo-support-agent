"""Tasks 11-12 live demo: start a REAL uvicorn server on 127.0.0.1, exercise every endpoint over real
HTTP and WebSocket connections (including a client that disconnects mid-answer), then show the
JSON-Lines log entries those requests produced.

Only loopback traffic (127.0.0.1); no external network access.
"""
import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402
import uvicorn  # noqa: E402
from websockets.sync.client import connect  # noqa: E402

from app import logging_utils, rag  # noqa: E402

PORT = 8765
BASE = f"http://127.0.0.1:{PORT}"
LOG = Path(__file__).resolve().parent.parent / "logs" / "demo_requests.jsonl"


def start_server() -> uvicorn.Server:
    server = uvicorn.Server(uvicorn.Config("app.api:app", host="127.0.0.1", port=PORT, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(300):
        try:
            if httpx.get(f"{BASE}/health", timeout=1).status_code == 200:
                return server
        except httpx.HTTPError:
            time.sleep(0.2)
    raise RuntimeError("server did not start")


def show(label: str, r: httpx.Response) -> None:
    body = r.json()
    brief = {k: body[k] for k in ("trace_id", "blocked", "guardrail_events") if k in body}
    if "response" in body:
        brief["answer"] = body["response"]["answer"][:110]
    print(f"  {label}\n    HTTP {r.status_code}  {json.dumps(brief if brief else body)[:330]}")


if __name__ == "__main__":
    LOG.parent.mkdir(exist_ok=True)
    LOG.unlink(missing_ok=True)
    logging_utils.set_log_path(LOG)
    server = start_server()
    print(f"uvicorn running at {BASE}\n\n--- HTTP endpoints")
    with httpx.Client(base_url=BASE, timeout=60) as http:
        show("GET  /health", http.get("/health"))
        show("POST /ask  (policy)", http.post("/ask", json={"question": "How do I file an insurance claim?",
                                                             "session_id": "live-1"}))
        show("POST /ask  (lookup + phone number to be masked)", http.post("/ask", json={
            "question": "My number is +91 98765 43210, what is the status of APT-1038?", "session_id": "live-1"}))
        show("POST /ask  (validation error: empty question)", http.post("/ask", json={"question": ""}))
        show("POST /add-document", http.post("/add-document", json={
            "doc_id": "13_parking_policy", "title": "Clinic Parking Policy",
            "text": "Parking at partner clinics is free for the first two hours for patients with an "
                    "appointment. After two hours, a parking fee of 40 rupees per hour applies."}))
        show("POST /ask  (now answerable from the new document)", http.post("/ask", json={
            "question": "What is the parking fee at the clinic?", "session_id": "live-2"}))

        print("\n--- WebSocket /ws/chat (multi-turn)")
        with connect(f"ws://127.0.0.1:{PORT}/ws/chat?session_id=live-ws") as ws:
            for q in ["What is the status of APT-1003?", "Can I reschedule that appointment?"]:
                ws.send(json.dumps({"question": q}))
                reply = json.loads(ws.recv())
                print(f"  > {q}\n  < remembered={reply['remembered_record_id']} "
                      f"answer={reply['response']['answer'][:100]}...")
            ws.send("not json")
            print(f"  > not json\n  < {json.loads(ws.recv())['error']}")

        print("\n--- WebSocket client disconnects MID-ANSWER")
        drop = connect(f"ws://127.0.0.1:{PORT}/ws/chat?session_id=live-drop")
        drop.send(json.dumps({"question": "How long will my thyroid test results take?"}))
        drop.close()  # leave before the answer arrives
        print("  client closed the socket without waiting for the answer")
        time.sleep(3)

        print("\n--- Server still healthy afterwards")
        show("GET  /health", http.get("/health"))
        with connect(f"ws://127.0.0.1:{PORT}/ws/chat") as ws:
            ws.send(json.dumps({"question": "How do I get a second opinion?"}))
            print(f"  new WebSocket client answered: {json.loads(ws.recv())['response']['answer'][:90]}...")

    server.should_exit = True
    time.sleep(1)
    rag.build_index(prune=True)  # remove the demo document from the persistent index

    lines = LOG.read_text().splitlines()
    print(f"\n--- {LOG.relative_to(LOG.parent.parent)}: {len(lines)} JSON-Lines entries")
    for line in lines:
        e = json.loads(line)
        keep = {k: e[k] for k in ("event", "trace_id", "path", "status_code", "latency_ms", "session_id",
                                  "request_text", "detail", "turns_completed", "close_code") if k in e}
        print("  " + json.dumps(keep)[:260])
    raw = LOG.read_text()
    print(f"\nRaw phone '98765 43210' present in log file: {'98765 43210' in raw or '9876543210' in raw}")
    print(f"'[PHONE_REDACTED]' present in log file: {'[PHONE_REDACTED]' in raw}")
