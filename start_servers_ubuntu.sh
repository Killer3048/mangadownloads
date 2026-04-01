#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ -z "${TIPER_PYTHON:-}" ]]; then
  if [[ -x "${ROOT_DIR}/tiper/.venv/bin/python" ]]; then
    TIPER_PYTHON="${ROOT_DIR}/tiper/.venv/bin/python"
  else
    TIPER_PYTHON="python3"
  fi
fi
export TIPER_HOST="${TIPER_HOST:-127.0.0.1}"
export TIPER_PORT="${TIPER_PORT:-8765}"

START_OCR_WORKER="${START_OCR_WORKER:-1}"
if [[ -z "${OCR_WORKER_PYTHON:-}" ]]; then
  if [[ -x "${ROOT_DIR}/tiper/ocr_worker/.venv/bin/python" ]]; then
    OCR_WORKER_PYTHON="${ROOT_DIR}/tiper/ocr_worker/.venv/bin/python"
  else
    OCR_WORKER_PYTHON="python3"
  fi
fi
OCR_WORKER_HOST="${OCR_WORKER_HOST:-127.0.0.1}"
OCR_WORKER_PORT="${OCR_WORKER_PORT:-8777}"

if [[ "$START_OCR_WORKER" == "1" ]]; then
  export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK="${PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK:-1}"
fi

if [[ "$START_OCR_WORKER" == "1" && -z "${TIPER_OCR_WORKER_URL:-}" ]]; then
  export TIPER_OCR_WORKER_URL="http://${OCR_WORKER_HOST}:${OCR_WORKER_PORT}"
fi

echo "[+] Root: ${ROOT_DIR}"
echo "[+] TIPER_PYTHON: ${TIPER_PYTHON}"
echo "[+] OCR_WORKER_PYTHON: ${OCR_WORKER_PYTHON}"

if ! "${TIPER_PYTHON}" -c "import fastapi, uvicorn, pandas" >/dev/null 2>&1; then
  echo "[!] Python deps missing for tiper server."
  echo "    Run: ${TIPER_PYTHON} -m pip install -r tiper/requirements.txt"
  exit 1
fi

pids=()
cleanup() {
  for pid in "${pids[@]:-}"; do
    kill "${pid}" >/dev/null 2>&1 || true
  done
}
trap cleanup EXIT INT TERM

if [[ "$START_OCR_WORKER" == "1" ]]; then
  if ! "${OCR_WORKER_PYTHON}" -c "import fastapi, uvicorn, paddleocr, paddle" >/dev/null 2>&1; then
    echo "[!] Python deps missing for OCR worker."
    echo "    Create a separate env and install: tiper/ocr_worker/requirements.txt (+ paddlepaddle/paddlepaddle-gpu)"
    exit 1
  fi
  echo "[+] Starting OCR worker: http://${OCR_WORKER_HOST}:${OCR_WORKER_PORT}"
  (
    exec "${OCR_WORKER_PYTHON}" -m uvicorn app:app --app-dir "${ROOT_DIR}/tiper/ocr_worker" --host "${OCR_WORKER_HOST}" --port "${OCR_WORKER_PORT}"
  ) &
  pids+=("$!")
fi

echo "[+] Starting tiper server: http://${TIPER_HOST}:${TIPER_PORT}"
(
  cd "${ROOT_DIR}"
  exec "${TIPER_PYTHON}" -m tiper.server
) &
pids+=("$!")

echo "[+] Servers are running. Press Ctrl+C to stop."
wait
