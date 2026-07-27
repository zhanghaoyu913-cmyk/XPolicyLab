#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ -z "${G05_ROOT:-}" ]]; then
  echo "[install] Set G05_ROOT to the G0.5/GalaxeaVLA source checkout." >&2
  echo "          Example: export G05_ROOT=/path/to/GalaxeaVLA_github_port" >&2
  exit 2
fi
if [[ ! -d "${G05_ROOT}" ]]; then
  echo "[install] G05_ROOT does not exist: ${G05_ROOT}" >&2
  exit 2
fi

PYTHON_BIN="${G05_PYTHON:-$(command -v python3)}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[install] Python executable not found: ${PYTHON_BIN}" >&2
  exit 2
fi

echo "[install] python=${PYTHON_BIN}"
echo "[install] G05_ROOT=${G05_ROOT}"

echo "[install] install XPolicyLab websocket/server dependencies"
"${PYTHON_BIN}" -m pip install -e "${XPL_ROOT}"

if [[ -f "${G05_ROOT}/pyproject.toml" || -f "${G05_ROOT}/setup.py" ]]; then
  echo "[install] install G0.5 source checkout in editable mode"
  "${PYTHON_BIN}" -m pip install -e "${G05_ROOT}"
else
  echo "[install] warning: no pyproject.toml/setup.py found under G05_ROOT; skip editable G0.5 install"
fi

if [[ -n "${G05_CKPT_PATH:-}" ]]; then
  echo "[install] validating G05_CKPT_PATH=${G05_CKPT_PATH}"
  if [[ ! -e "${G05_CKPT_PATH}" ]]; then
    echo "[install] G05_CKPT_PATH does not exist: ${G05_CKPT_PATH}" >&2
    exit 2
  fi
  ckpt_dir="${G05_CKPT_PATH}"
  [[ -f "${ckpt_dir}" ]] && ckpt_dir="$(dirname "${ckpt_dir}")"
  run_dir="${ckpt_dir}"
  if [[ "$(basename "${run_dir}")" == "checkpoints" ]]; then
    run_dir="$(dirname "${run_dir}")"
  fi
  for required in ".hydra/config.yaml" "dataset_stats.json" "action_tokenizer.pt"; do
    if [[ ! -s "${run_dir}/${required}" ]]; then
      echo "[install] warning: checkpoint sidecar not found: ${run_dir}/${required}" >&2
    fi
  done
fi

"${PYTHON_BIN}" - <<'PY'
import importlib
for name in ("websockets", "msgpack", "msgpack_numpy", "pydantic", "yaml"):
    importlib.import_module(name)
print("[install] XPolicyLab websocket dependencies import OK")
PY

echo "[install] done"
