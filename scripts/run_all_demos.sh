#!/usr/bin/env bash
# Regenerate every transcript in transcripts/ from scratch, under MOCK_LLM, with zero API keys.
# Usage: bash scripts/run_all_demos.sh      (from the project root, venv created per README)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
T=transcripts
export MOCK_LLM=true CREWAI_DISABLE_TELEMETRY=true OTEL_SDK_DISABLED=true HF_HUB_OFFLINE=1

run() {  # run <script> [output-file]; strips model-loading progress bars from the transcript
  echo ">> $1"
  if [ $# -eq 2 ]; then
    $PY "$1" 2>/dev/null | grep -v "Loading weights" > "$T/$2"
  else
    $PY "$1" 2>/dev/null | grep -v "Loading weights" > /dev/null
  fi
}

run dataset.py                              task01_dataset_report.txt
run scripts/build_index.py                  task03_index_build.txt
run scripts/task04_calibrate_threshold.py   task04_threshold_calibration.txt
run scripts/task04_grounded_demo.py         task04_grounded_generation.txt
run scripts/task05_chunking_eval.py         task05_chunking_evaluation.txt
run scripts/task06_lookup_demo.py           task06_appointment_lookup.txt
run scripts/task07_crew_demo.py             task07_crew_kickoff.txt
run scripts/task08_memory_demo.py           # writes task08_memory_multiturn.txt + task08_memory_fresh_session.txt
run scripts/task09_schema_demo.py           task09_structured_output.txt
run scripts/task10_calibrate_answerability.py task10_answerability_calibration.txt
run scripts/task10_groundedness_design.py   task10_groundedness_design.txt
run scripts/task10_guardrails_demo.py       task10_guardrails.txt
run scripts/task11_live_api_demo.py         task11_12_api_and_logging.txt
run scripts/task14_review_demo.py           task14_autogen_review.txt
run scripts/task15_governance_demo.py       task15_governance.txt
run scripts/task16_cache_demo.py            task16_cache.txt
# Task 13. The OFFICIAL held-out transcripts (task13_evaluation_v1_setA.txt, task13_evaluation_v2_setB_heldout.txt)
# are historical one-time runs and are deliberately NOT regenerated here. This re-runs set B on the current code:
echo ">> scripts/task13_evaluate.py B (current code; not a held-out result)"
{ echo "Current-code re-run of set B (not a held-out result; see task13_evaluation_v2_setB_heldout.txt)."; echo;
  $PY scripts/task13_evaluate.py B results_B_postfix.json 2>/dev/null | grep -v "Loading weights"; } > "$T/task13_evaluation_v2_setB_postfix.txt"
echo "All transcripts regenerated in $T/"
