"""Task 13: run a 15-query evaluation set through the FULL pipeline and score it with the
LLM-as-judge (MOCK_LLM). Prints per-query scores and the four averages; writes eval/results_<set>.json.

Usage: python scripts/task13_evaluate.py B   (held-out set B, the official Task 13 result)
       python scripts/task13_evaluate.py A   (set A: held-out for v1, used to design v2)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.memory import reset_session  # noqa: E402
from app.pipeline import ask  # noqa: E402
from app.rag import build_index  # noqa: E402
from app.review import build_review_task  # noqa: E402
from app.tools import TOOL_CALLS  # noqa: E402
from eval.eval_set import EVAL_SET  # noqa: E402
from eval.eval_set_b import EVAL_SET_B  # noqa: E402
from eval.judge import MockJudgeLLM, build_judge_prompt, observed_behaviour  # noqa: E402

METRICS = ("accuracy", "grounding", "completeness", "safety")
ROOT = Path(__file__).resolve().parent.parent


def context_lines(tool_calls: list[dict]) -> list[str]:
    """Retrieved context + appointment record, formatted exactly as the review stage sees them."""
    policy = next((c["result"] for c in tool_calls if c["tool"] == "rag_lookup"), None)
    appt = next((c["result"] for c in tool_calls if c["tool"] == "check_appointment_status"), None)
    task = build_review_task("", "", policy["context"] if policy else [], appt)
    return task.split("RETRIEVED CONTEXT:\n", 1)[1].splitlines()


if __name__ == "__main__":
    which = (sys.argv[1] if len(sys.argv) > 1 else "B").upper()
    dataset = {"A": EVAL_SET, "B": EVAL_SET_B}[which]
    build_index()
    judge = MockJudgeLLM()
    rows = []
    for item in dataset:
        reset_session(f"eval-{item['id']}")
        start = len(TOOL_CALLS)
        r = ask(item["query"], f"eval-{item['id']}")
        observed = observed_behaviour(r.response.answer, r.blocked)
        prompt = build_judge_prompt(r.question_masked, item["expect"], item["facts"], observed,
                                    r.response.answer, context_lines(TOOL_CALLS[start:]))
        scores = json.loads(judge.complete(prompt))
        rows.append({**item, "observed": observed, "answer": r.response.answer,
                     "events": r.guardrail_events, **scores})

    print(f"Task 13 evaluation - set {which}, 15 queries, full pipeline, LLM-as-judge under MOCK_LLM (scores 1-5)\n")
    print(f"{'ID':<4} {'topic':<42} {'expect':<9} {'observed':<9} ACC GRD CMP SAF")
    for x in rows:
        print(f"{x['id']:<4} {x['topic'][:42]:<42} {x['expect']:<9} {x['observed']:<9} "
              f"{x['accuracy']:>3} {x['grounding']:>3} {x['completeness']:>3} {x['safety']:>3}")
    avgs = {m: sum(x[m] for x in rows) / len(rows) for m in METRICS}
    print(f"{'AVERAGE (n=15)':<66} " + " ".join(f"{avgs[m]:.2f}" for m in METRICS))

    print("\nPer-query detail:")
    for x in rows:
        print(f"\n{x['id']} [{x['topic']}] {x['query']}")
        print(f"  answer   : {x['answer']}")
        print(f"  events   : {x['events']}")
        print(f"  judge    : {x['rationale']}")

    behaviour_ok = sum(x["expect"] == x["observed"] for x in rows)
    print(f"\nCorrect behaviour: {behaviour_ok}/15 "
          f"(held-out check of the groundedness/refusal rules from Task 10)")
    print(f"Judge calls: {judge.calls}")
    out = sys.argv[2] if len(sys.argv) > 2 else f"results_{which}.json"
    (ROOT / "eval" / out).write_text(json.dumps({"rows": rows, "averages": avgs}, indent=1))
