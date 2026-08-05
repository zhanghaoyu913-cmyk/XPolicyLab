#!/usr/bin/env bash
set -euo pipefail

bench_name=${1:?bench_name required}
ckpt_name=${2:?ckpt_name required}
env_cfg_type=${3:?env_cfg_type required}
action_type=${4:?action_type required}
seed=${5:?seed required}
gpu_id=${6:?gpu_id required}
shift 6 || true

if [[ "${bench_name}" != "RoboDojo" ]]; then
  echo "G05 train.sh expects bench_name=RoboDojo, got ${bench_name}" >&2
  exit 2
fi
if [[ "${env_cfg_type}" != "arx_x5" ]]; then
  echo "G05 train.sh currently supports env_cfg_type=arx_x5, got ${env_cfg_type}" >&2
  exit 2
fi
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
MODE_CONFIG="${G05_TRAIN_MODE:-fm_only}"
OUTPUT_ROOT="${G05_OUTPUT_ROOT:-${SCRIPT_DIR}/checkpoints}"
RUN_ID="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}-${seed}"
# The submitted RoboDojo G05 FM-only training run used seed=7. XPolicyLab
# passes a seed argument as part of its standard 5-tuple; keep that value in
# RUN_ID, but default the actual G05 training seed to the reproduced run seed.
TRAIN_SEED="${G05_TRAIN_SEED:-7}"
if [[ "${TRAIN_SEED}" == "0" ]]; then
  echo "G05_TRAIN_SEED must be non-zero; use 7 to reproduce the submitted run." >&2
  exit 2
fi

case "${MODE_CONFIG}" in
  fm_only|ar_fm) ;;
  fm) MODE_CONFIG="fm_only" ;;
  both|joint_fm) MODE_CONFIG="ar_fm" ;;
  *)
    echo "Unsupported G05_TRAIN_MODE=${MODE_CONFIG}; expected fm_only or ar_fm." >&2
    exit 2
    ;;
esac

if [[ "${gpu_id}" == *,* ]]; then
  IFS=',' read -r -a _gpus <<< "${gpu_id}"
  num_gpus="${#_gpus[@]}"
else
  num_gpus=1
fi
if [[ -n "${NPROC_PER_NODE:-}" ]]; then
  num_gpus="${NPROC_PER_NODE}"
fi

per_gpu_batch="${G05_BATCH_SIZE_PER_GPU:-8}"
global_batch="${G05_GLOBAL_BATCH:-256}"
if [[ -n "${G05_GRAD_ACCUMULATION_STEPS:-}" ]]; then
  grad_accum="${G05_GRAD_ACCUMULATION_STEPS}"
else
  denom=$((num_gpus * per_gpu_batch))
  if (( denom <= 0 || global_batch % denom != 0 )); then
    echo "Cannot derive grad accumulation for global_batch=${global_batch}, num_gpus=${num_gpus}, per_gpu_batch=${per_gpu_batch}." >&2
    echo "Set G05_GRAD_ACCUMULATION_STEPS explicitly." >&2
    exit 2
  fi
  grad_accum=$((global_batch / denom))
fi

export CUDA_VISIBLE_DEVICES="${gpu_id}"
export PYTHONPATH="${G05_ROOT}:${G05_ROOT}/src:${PYTHONPATH:-}"
export G05_OUTPUT_DIR="${G05_OUTPUT_DIR:-${OUTPUT_ROOT}}"
export EXP_NAME="${EXP_NAME:-${RUN_ID}}"
export RUN_EXP_NAME="${RUN_ID}"
export PYTHON_BIN

hydra_args=(
  "+g05_benchmark/robodojo=${MODE_CONFIG}"
  "seed=${TRAIN_SEED}"
  "model.batch_size=${per_gpu_batch}"
  "model.grad_accumulation_steps=${grad_accum}"
  "model.max_epochs=null"
)
if [[ -n "${G05_MAX_STEPS:-}" ]]; then
  hydra_args+=("model.max_steps=${G05_MAX_STEPS}")
fi
if [[ -n "${G05_RESUME_CKPT:-}" ]]; then
  hydra_args+=("resume_ckpt=${G05_RESUME_CKPT}")
fi

cd "${G05_ROOT}"
if [[ -f scripts/validate_g05_benchmark_config.py ]]; then
  "${PYTHON_BIN}" scripts/validate_g05_benchmark_config.py \
    --task "${TASK_CONFIG}" --benchmark robodojo --mode "${MODE_CONFIG}"
fi

printf '[G05 train] mode=%s task=%s run_id=%s train_seed=%s num_gpus=%s per_gpu_batch=%s grad_accum=%s global_batch=%s\n' \
  "${MODE_CONFIG}" "${TASK_CONFIG}" "${RUN_ID}" "${TRAIN_SEED}" "${num_gpus}" "${per_gpu_batch}" "${grad_accum}" "$((num_gpus * per_gpu_batch * grad_accum))"

exec bash scripts/run/finetune.sh \
  "${num_gpus}" \
  "${TASK_CONFIG}" \
  "${hydra_args[@]}" \
  "$@"
