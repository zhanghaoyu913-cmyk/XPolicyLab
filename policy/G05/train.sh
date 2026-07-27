#!/usr/bin/env bash
set -euo pipefail

bench_name=${1:?bench_name required}
ckpt_name=${2:?ckpt_name required}
env_cfg_type=${3:?env_cfg_type required}
action_type=${4:?action_type required}
seed=${5:?seed required}
gpu_id=${6:?gpu_id required}
shift 6 || true

if [[ "${action_type}" != "joint" ]]; then
  echo "G05 train.sh currently supports action_type=joint, got ${action_type}" >&2
  exit 2
fi
if [[ -z "${G05_ROOT:-}" ]]; then
  echo "Set G05_ROOT to the G0.5/GalaxeaVLA source checkout before training." >&2
  exit 2
fi
if [[ ! -d "${G05_ROOT}" ]]; then
  echo "G05_ROOT does not exist: ${G05_ROOT}" >&2
  exit 2
fi
if [[ -z "${ROBODOJO_LEROBOT_V30_ROOT:-}" ]]; then
  echo "Set ROBODOJO_LEROBOT_V30_ROOT to the RoboDojo LeRobot v3.0 dataset root." >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${G05_PYTHON:-$(command -v python3)}"
TASK_CONFIG="${G05_TASK_CONFIG:-robodojo_arx_x5_joint}"
RUN_ID="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}-${seed}"
OUTPUT_ROOT="${G05_OUTPUT_ROOT:-${SCRIPT_DIR}/checkpoints}"

if [[ "${gpu_id}" == *","* ]]; then
  IFS=',' read -r -a _gpus <<< "${gpu_id}"
  num_gpus="${#_gpus[@]}"
else
  num_gpus=1
fi

export CUDA_VISIBLE_DEVICES="${gpu_id}"
export PYTHONPATH="${G05_ROOT}:${PYTHONPATH:-}"
export G05_OUTPUT_DIR="${G05_OUTPUT_DIR:-${OUTPUT_ROOT}}"
export EXP_NAME="${EXP_NAME:-${RUN_ID}}"
export PYTHON_BIN

cd "${G05_ROOT}"
exec bash scripts/run/finetune.sh \
  "${num_gpus}" \
  "${TASK_CONFIG}" \
  "seed=${seed}" \
  "$@"
