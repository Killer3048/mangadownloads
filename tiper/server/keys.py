from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional


def _load_yaml(path: Path) -> Dict[str, Any]:
  try:
    import yaml  # type: ignore
  except Exception:
    return {}

  if not path.exists():
    return {}
  try:
    with path.open("r", encoding="utf-8") as f:
      data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}
  except Exception:
    return {}


def load_keys(tiper_root: Path) -> Dict[str, Any]:
  """
  Load secrets from keys.yaml.

  Resolution order:
  1) TIPER_KEYS_PATH env var (explicit)
  2) <repo_root>/keys.yaml (tiper_root.parent)
  3) <tiper_root>/keys.yaml
  """
  candidates: list[Path] = []
  env_path = (os.environ.get("TIPER_KEYS_PATH") or "").strip()
  if env_path:
    candidates.append(Path(env_path))
  candidates.append((tiper_root.parent / "keys.yaml").resolve())
  candidates.append((tiper_root / "keys.yaml").resolve())

  for p in candidates:
    data = _load_yaml(p)
    if data:
      return data
  return {}


def get_xai_api_key(tiper_root: Path) -> Optional[str]:
  # Prefer env var for convenience in CI / scripts.
  env_key = (os.environ.get("XAI_API_KEY") or "").strip()
  if env_key:
    return env_key

  keys = load_keys(tiper_root)
  xai = keys.get("xai") if isinstance(keys, dict) else None
  if isinstance(xai, dict):
    key = (xai.get("api_key") or "").strip()
    return key or None
  return None


def get_xai_config(tiper_root: Path) -> Dict[str, Any]:
  keys = load_keys(tiper_root)
  xai = keys.get("xai") if isinstance(keys, dict) else None
  return xai if isinstance(xai, dict) else {}

