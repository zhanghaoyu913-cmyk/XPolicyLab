#!/usr/bin/env bash
set -euo pipefail

# Fast train-entry validation for cluster machines. This checks that the G05
# RoboDojo training wrapper resolves the GalaxeaVLA checkout, RoboDojo dataset,
# benchmark mode, batch/accumulation settings, and Hydra config without running
# a full training job or writing checkpoints.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${G05_ROOT:-}" ]]; then
  echo "Set G05_ROOT to the G0.5/GalaxeaVLA source checkout." >&2
  exit 2
fi
if [[ -z "${ROBODOJO_LEROBOT_V30_ROOT:-}" ]]; then
  echo "Set ROBODOJO_LEROBOT_V30_ROOT to the RoboDojo LeRobot v3.0 dataset root." >&2
  exit 2
fi

GPU_LIST="${G05_VALIDATE_GPUS:-0}"
MODE="${G05_TRAIN_MODE:-fm_only}"
MAX_STEPS="${G05_VALIDATE_MAX_STEPS:-1}"
OUTPUT_ROOT="${G05_VALIDATE_OUTPUT_ROOT:-/tmp/g05_xpolicylab_train_validate}"

export G05_TRAIN_MODE="${MODE}"
export G05_MAX_STEPS="${MAX_STEPS}"
export G05_OUTPUT_ROOT="${OUTPUT_ROOT}"

exec bash "${SCRIPT_DIR}/train.sh" \
  RoboDojo validate arx_x5 joint 0 "${GPU_LIST}" \
  --dry-run --max_datasets 1 \
  logger.mode=offline \
  "$@"
