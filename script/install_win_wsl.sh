#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCANR_DIR="${ROOT_DIR}/scanr"
INSTALL_CMD="${SCANR_DIR}/install_win.cmd"

if [[ ! -f "${INSTALL_CMD}" ]]; then
  echo "[ERROR] Not found: ${INSTALL_CMD}" >&2
  exit 1
fi

if ! command -v cmd.exe >/dev/null 2>&1; then
  if [[ -x /mnt/c/Windows/System32/cmd.exe ]]; then
    CMD_EXE="/mnt/c/Windows/System32/cmd.exe"
  else
    echo "[ERROR] cmd.exe not found. This script must run inside WSL with Windows interop enabled." >&2
    exit 1
  fi
else
  CMD_EXE="cmd.exe"
fi

if command -v wslpath >/dev/null 2>&1; then
  SCANR_DIR_WIN="$(wslpath -w "${SCANR_DIR}")"
else
  if [[ -z "${WSL_DISTRO_NAME:-}" ]]; then
    echo "[ERROR] wslpath is missing and WSL_DISTRO_NAME is not set; cannot build a Windows path." >&2
    exit 1
  fi
  # Fallback UNC path: \\wsl$\<distro>\home\...
  POSIX="${SCANR_DIR#/}"
  SCANR_DIR_WIN="\\\\wsl$\\${WSL_DISTRO_NAME}\\${POSIX//\//\\}"
fi

echo "[+] Installing TypeR into Windows CEP folder..."
echo "    Scanr dir: ${SCANR_DIR}"
echo "    Windows path: ${SCANR_DIR_WIN}"

exec "${CMD_EXE}" /C "pushd \"${SCANR_DIR_WIN}\" && install_win.cmd"
