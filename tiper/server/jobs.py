from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from PIL import Image

from tiper.server.paths import wsl_path_to_unc


Image.MAX_IMAGE_PIXELS = None


def _utc_now_iso() -> str:
  return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
  try:
    with os.fdopen(fd, "w", encoding="utf-8") as f:
      json.dump(data, f, ensure_ascii=False, indent=2)
      f.write("\n")
    Path(tmp).replace(path)
  finally:
    try:
      os.unlink(tmp)
    except Exception:
      pass


def read_json(path: Path) -> Dict[str, Any]:
  if not path.exists():
    return {}
  with path.open("r", encoding="utf-8") as f:
    return json.load(f)


def write_json(path: Path, data: Dict[str, Any]) -> None:
  _atomic_write_json(path, data)


def job_paths(job_dir: Path) -> Dict[str, str]:
  return {
    "job_dir": str(job_dir),
    "job_json": str(job_dir / "job.json"),
    "original_png": str(job_dir / "original.png"),
    "bubbles_auto": str(job_dir / "bubbles_auto.json"),
    "bubbles_final": str(job_dir / "bubbles_final.json"),
    "original_txt": str(job_dir / "original.txt"),
    "translate_txt": str(job_dir / "translate.txt"),
    "cleaned_png": str(job_dir / "cleaned.png"),
    "text_styles": str(job_dir / "text_styles.json"),
    "bubble_classes": str(job_dir / "bubble_classes.json"),
    "bubble_geometry": str(job_dir / "bubble_geometry.json"),
    "ocr_mask_stats": str(job_dir / "ocr_mask_stats.json"),
    "translation_fit": str(job_dir / "translation_fit.json"),
    "frames_dir": str(job_dir / "frames"),
  }


def new_job_id() -> str:
  from uuid import uuid4

  ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
  return f"{ts}__{uuid4().hex[:8]}"


def create_job(jobs_root: Path, source_png: Path, config: Dict[str, Any]) -> Dict[str, Any]:
  job_id = new_job_id()
  job_dir = jobs_root / job_id
  job_dir.mkdir(parents=True, exist_ok=False)

  original_dst = job_dir / "original.png"
  shutil.copy2(str(source_png), str(original_dst))

  with Image.open(original_dst) as img:
    width, height = img.size

  data: Dict[str, Any] = {
    "schema_version": 1,
    "job_id": job_id,
    "created_at": _utc_now_iso(),
    "status": "created",
    "progress": {"phase": None, "done": 0, "total": 0, "pct": 0},
    "config": config or {},
    "image": {"width": int(width), "height": int(height), "file": "original.png"},
    "paths": job_paths(job_dir),
    "paths_open": {
      "job_dir": wsl_path_to_unc(job_dir) or str(job_dir),
      "original_txt": wsl_path_to_unc(job_dir / "original.txt") or str(job_dir / "original.txt"),
      "translate_txt": wsl_path_to_unc(job_dir / "translate.txt") or str(job_dir / "translate.txt"),
      "cleaned_png": wsl_path_to_unc(job_dir / "cleaned.png") or str(job_dir / "cleaned.png"),
    },
    "last_error": None,
  }

  _atomic_write_json(job_dir / "job.json", data)
  return data


def update_job(job_dir: Path, patch: Dict[str, Any]) -> Dict[str, Any]:
  job_json = job_dir / "job.json"
  current = read_json(job_json)
  # shallow merge
  current.update(patch or {})
  _atomic_write_json(job_json, current)
  return current


def set_job_progress(job_dir: Path, phase: str, done: int, total: int) -> None:
  pct = 0
  if total > 0:
    pct = int(max(0, min(100, round((done / float(total)) * 100))))
  update_job(job_dir, {"progress": {"phase": phase, "done": int(done), "total": int(total), "pct": pct}})


def set_job_status(job_dir: Path, status: str) -> None:
  update_job(job_dir, {"status": status})


def set_job_error(job_dir: Path, message: str) -> None:
  update_job(job_dir, {"status": "error", "last_error": message})
