from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional


_WIN_DRIVE_RE = re.compile(r"^([a-zA-Z]):[\\\\/](.*)$")


def is_wsl() -> bool:
  return bool(os.environ.get("WSL_DISTRO_NAME") or os.environ.get("WSL_INTEROP"))


def win_path_to_wsl(path: str) -> str:
  m = _WIN_DRIVE_RE.match(path or "")
  if not m:
    return path
  drive = m.group(1).lower()
  rest = (m.group(2) or "").replace("\\", "/")
  return f"/mnt/{drive}/{rest}"


def unc_wsl_to_posix(path: str) -> str:
  p = path or ""
  if p.startswith("\\\\wsl$\\") or p.startswith("//wsl$/"):
    p2 = p.replace("/", "\\")
    parts = p2.split("\\")
    # ["", "", "wsl$", "Distro", "home", ...]
    if len(parts) >= 5:
      rel = parts[4:]
      return "/" + "/".join([x for x in rel if x])
  return path


def normalize_source_path(raw_path: str) -> Path:
  p = (raw_path or "").strip().strip('"')
  if os.name != "nt":
    p = unc_wsl_to_posix(p)
    p = win_path_to_wsl(p)
    return Path(p)
  # Windows: accept WSL /mnt/<drive>/ paths too.
  if p.startswith("/mnt/") and len(p) > 6:
    drive = p[5]
    rest = p[7:].replace("/", "\\")
    return Path(f"{drive.upper()}:\\{rest}")
  return Path(p)


def wsl_path_to_unc(path: Path) -> Optional[str]:
  if os.name == "nt":
    return None
  if not is_wsl():
    return None
  distro = os.environ.get("WSL_DISTRO_NAME")
  if not distro:
    return None
  # \\wsl$\<distro>\home\...
  posix = str(path).replace("/", "\\").lstrip("\\")
  return f"\\\\wsl$\\{distro}\\{posix}"

