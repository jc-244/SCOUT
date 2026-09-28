#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."

export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://localhost:8000/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
# Without this, a serve running --tool-call-parser yields no actions and traj.jsonl is never written.
export QWEN35_TOOLS_API=1
# Keep the driver off the GPUs (easyocr grabs GPU 0 otherwise).
export CUDA_VISIBLE_DEVICES=""
export VERIFIER_MODULE="${VERIFIER_MODULE:-scout_verifier.GPTBedrockVerifier}"

# test_205.json = test_all.json (300) minus categories 08/09 (defanged URLs) and 12 (popup attack).
python scripts/python/run_multienv_qwen35.py \
  --headless \
  --observation_type screenshot \
  --action_space pyautogui \
  --result_dir data/results/osblind_run1 \
  --test_config_base_dir data/OSBlind/evaluation_example_safety \
  --test_all_meta_path data/OSBlind/evaluation_example_safety/test_205.json \
  --max_steps 50 \
  --num_envs 8 \
  --provider_name docker \
  --path_to_vm data/docker_vm_data/Ubuntu.qcow2 \
  --client_password password \
  --password password \
  --domain all \
  --model Qwen/Qwen3.6-35B-A3B \
  --api_backend openai \
  --temperature 1 \
  --top_p 0.9 \
  --max_tokens 32768 \
  --coordinate_type relative
