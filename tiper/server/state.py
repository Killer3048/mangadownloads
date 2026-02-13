from __future__ import annotations

import os
import threading
from dataclasses import field
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from tiper.server.paths import is_wsl


def _load_yaml_config(path: Path) -> Dict[str, Any]:
  try:
    import yaml  # type: ignore
  except Exception:
    return {}

  if not path.exists():
    return {}
  with path.open("r", encoding="utf-8") as f:
    return yaml.safe_load(f) or {}


def default_job_config(tiper_root: Path) -> Dict[str, Any]:
  cfg = _load_yaml_config(tiper_root / "config.yaml")

  bl = cfg.get("blaster", {}) if isinstance(cfg, dict) else {}
  cut = cfg.get("cutting", {}) if isinstance(cfg, dict) else {}
  fr = cfg.get("frames", {}) if isinstance(cfg, dict) else {}
  bubble_cls_cfg = cfg.get("bubble_class", {}) if isinstance(cfg, dict) else {}

  # Primary OCR languages (EasyOCR / Paddle).
  langs_raw = bl.get("languages", ["ko", "en"])
  langs: list[str] = []
  for l in (langs_raw or []):
    s = str(l or "").strip()
    if s and s not in langs:
      langs.append(s)
  if not langs:
    langs = ["ko", "en"]

  base_conf = float(bl.get("confidence", 0.14))
  # Match the original tiper logic: lower the threshold by 0.10 to increase recall.
  primary_conf = max(0.0, base_conf - 0.10)

  sec = bl.get("secondary_pass", {}) if isinstance(bl, dict) else {}
  sec_enabled = bool(sec.get("enabled", False)) if isinstance(sec, dict) else False
  sec_lang = str(sec.get("language") or "th").strip() if isinstance(sec, dict) else "th"
  sec_base_conf = float(sec.get("confidence", 0.25)) if isinstance(sec, dict) else 0.25
  sec_conf = max(0.0, sec_base_conf - 0.10)

  bubble_model_path_raw = bl.get("bubble_model_path")
  if bubble_model_path_raw:
    bubble_model_path = Path(str(bubble_model_path_raw))
    if not bubble_model_path.is_absolute():
      bubble_model_path = (tiper_root / bubble_model_path).resolve()
  else:
    bubble_model_path = (tiper_root / "models" / "yolov8m_seg-speech-bubble" / "model.pt").resolve()

  slice_model_path_raw = cut.get("model", "yolo11x.pt")
  if slice_model_path_raw:
    slice_model_path = Path(str(slice_model_path_raw))
    if not slice_model_path.is_absolute():
      # Try tiper_root first, then fallback to cutting/ subdir if needed.
      candidate = (tiper_root / slice_model_path).resolve()
      if candidate.exists():
        slice_model_path = candidate
      else:
        slice_model_path = (tiper_root / "cutting" / slice_model_path).resolve()
  else:
    slice_model_path = (tiper_root / "yolo11x.pt").resolve()

  bubble_cls_model_raw = bubble_cls_cfg.get("model_path")
  if bubble_cls_model_raw:
    bubble_cls_model_path = Path(str(bubble_cls_model_raw))
    if not bubble_cls_model_path.is_absolute():
      bubble_cls_model_path = (tiper_root / bubble_cls_model_path).resolve()
  else:
    bubble_cls_model_path = (tiper_root / "models" / "yolo11m-bubbles-cls" / "best.pt").resolve()

  bubble_cls_imgsz = bubble_cls_cfg.get("imgsz", None)
  if bubble_cls_imgsz is not None:
    try:
      bubble_cls_imgsz = int(bubble_cls_imgsz)
    except Exception:
      bubble_cls_imgsz = None

  return {
    "ocr": {
      "languages": langs or ["ko", "en"],
      "conf": base_conf,
      # Inpaint mask OCR confidence (EasyOCR/Paddle). Keep close to original tiper behaviour.
      "mask_conf": primary_conf,
      "lang_hint": None,
      "worker_url": os.environ.get("TIPER_OCR_WORKER_URL") or None,
      "secondary_pass": {
        "enabled": sec_enabled,
        "language": sec_lang,
        "confidence": sec_base_conf,
        "mask_conf": sec_conf,
      },
    },
    "bubbles": {
      "model_path": str(bubble_model_path),
      "conf": float(bl.get("bubble_confidence", 0.7)),
      "imgsz": int(cut.get("imgsz", 1280)),
      "chunk_h": int(cut.get("det_chunk_height", 4096)),
      "overlap": int(cut.get("det_overlap", 256)),
      "nms_iou": 0.5,
      "device": cut.get("device", "auto"),
    },
    "slicing": {
      "min_h": int(cut.get("min_height", 1000)),
      "max_h": int(cut.get("max_height", 4000)),
      "margin": int(cut.get("margin", 20)),
      "weights": cut.get("weights", {}),
      "boxes": {
        "enabled": True,
        "model_path": str(slice_model_path),
        "conf": float(cut.get("conf", 0.25)),
        "imgsz": int(cut.get("imgsz", 1280)),
        "chunk_h": int(cut.get("det_chunk_height", 4096)),
        "overlap": int(cut.get("det_overlap", 256)),
        "device": cut.get("device", "auto"),
        "min_w": int(cut.get("min_width", 20)),
        "min_h": int(cut.get("min_height_box", 20)),
        "text": {
          "enabled": bool(cut.get("use_text_detection", True)),
          "langs": cut.get("text_langs", ["ko", "en"]),
          "conf": float(cut.get("text_conf", 0.40)),
          "chunk_h": int(cut.get("text_chunk_height")) if cut.get("text_chunk_height") is not None else int(cut.get("det_chunk_height", 4096)),
          "overlap": int(cut.get("text_overlap")) if cut.get("text_overlap") is not None else int(cut.get("det_overlap", 256)),
          "min_w": int(cut.get("text_min_box_width", 20)),
          "min_h": int(cut.get("text_min_box_height", 20)),
        },
      },
    },
    "inpaint": {
      "device": bl.get("device", "cuda"),
    },
    "frames": {
      "enabled": bool(fr.get("enabled", True)),
      "min_height": int(fr.get("min_height", 4000)),
      "max_height": int(fr.get("max_height", 7800)),
      "output_subdir": fr.get("output_subdir", "frames"),
      "format": fr.get("format", "png"),
      "quality": int(fr.get("quality", 95)),
    },
    "bubble_class": {
      "enabled": bool(bubble_cls_cfg.get("enabled", True)),
      "model_path": str(bubble_cls_model_path),
      "conf": float(bubble_cls_cfg.get("conf", 0.6)),
      "imgsz": bubble_cls_imgsz,
      "device": bubble_cls_cfg.get("device", "auto"),
    },
    "style": {
      "enabled": True,
      "mask_conf": primary_conf,
      "crop_pad": 2,
      "min_mask_pixels": 30,
      "min_glyph_pixels": 20,
      "min_bg_pixels": 50,
      "min_stroke_delta": 8.0,
      "min_stroke_px": 1.0,
    },
    "autofit": {
      "enabled": True,
      "min_pt": 16,
      "max_pt": 34,
      # Font size search step (pt). For comics typography a 2pt step is usually enough and avoids jitter.
      "pt_step": 2,
      # Photoshop document resolution (DPI). UI sends it at job creation; fallback to 72 if missing.
      "dpi": None,
      # Bubble contour extraction from full-page segmentation (IoU-matched).
      "seg_conf": 0.25,
      "seg_imgsz": int(cut.get("imgsz", 1280)),
      "seg_iou_thr": 0.30,
      # Inner safe region inside the bubble mask.
      "core_frac": 0.55,
      # Font-fit heuristics (PIL approximation).
      "leading_factor": 0.25,
      "cap_factor": 1.25,
      "font_path": None,
      # Adaptive shrink of the usable region to avoid text touching the bubble contour.
      "comfort": {
        "enabled": True,
        "base_frac": 0.12,
        "min_frac": 0.06,
        "max_frac": 0.22,
        "min_px": 4,
        "max_px": 60,
        "fallback_bonus": 0.08,
        # Desired minimal leftover space (ratio) inside the fit box.
        "target_slack": 0.06,
        # Utility term weights for selecting a readable layout among fit candidates.
        "layout_badness_weight": 0.7,
        "hyphen_util_penalty": 2.4,
        # Limit how much we can reduce the chosen font size for comfort.
        "max_reduce_pt": 4,
      },
      # Auto line-breaking rules ("komponovka.md"). When enabled, server may return
      # translation text with explicit '\n' line breaks to improve readability.
      "komponovka": {
        "enabled": True,
        "mode": "auto",  # auto|force|off
        "lang": "auto",  # auto|ru|en
        "apply_to_translation": True,
        # Generic profile fallback. Per-bubble profile is inferred from bubble
        # class + geometry and blended with these values.
        "profile": {
          "alpha": 0.78,
          "min_factor": 0.64,
          "line_break_penalty": 0.42,
          "shape_penalty": 1.8,
          "center_peak_penalty": 1.6,
          "hyphen_break_penalty": 4.0,
        },
      },
    },
  }


@dataclass
class JobRuntime:
  detect_task: Optional[Any] = None
  ocr_task: Optional[Any] = None
  imagine_task: Optional[Any] = None
  lock: threading.Lock = field(default_factory=threading.Lock)


class SharedResources:
  def __init__(self, tiper_root: Path):
    self.tiper_root = tiper_root
    self.jobs_root = Path(os.environ.get("TIPER_JOBS_DIR") or (tiper_root / "jobs")).resolve()
    self.jobs_root.mkdir(parents=True, exist_ok=True)
    self.default_config = default_job_config(tiper_root)

    self._jobs: Dict[str, JobRuntime] = {}
    self._jobs_lock = threading.Lock()

    self._bubble_ai = None
    self._ocr_mask = None
    self._ocr_mask_secondary = None
    self._ocr_mask_inpaint = None
    self._ocr_mask_inpaint_secondary = None
    self._ocr_translation = None
    self._inpainter = None
    self._slice_yolo = None
    self._slice_yolo_path = None
    self._slice_text_readers = None
    self._slice_text_langs = None
    self._slice_text_device = None
    self._bubble_cls = None
    self._bubble_cls_path = None

    # Single lock for GPU-heavy sections (YOLO / LaMa / OCR GPU).
    self.gpu_lock = threading.Lock()

  def job_runtime(self, job_id: str) -> JobRuntime:
    with self._jobs_lock:
      if job_id not in self._jobs:
        self._jobs[job_id] = JobRuntime()
      return self._jobs[job_id]

  def bubble_ai(self):
    if self._bubble_ai is None:
      from tiper.server.ai_bubble import BubbleDetector

      self._bubble_ai = BubbleDetector()
    return self._bubble_ai

  def ocr_mask(self):
    if self._ocr_mask is None:
      from tiper.ocr.backend import EasyOcrBackend, EasyOcrConfig

      langs = list(self.default_config.get("ocr", {}).get("languages") or ["ko", "en"])
      # Legacy/style OCR masks still run inside torch env → EasyOCR is OK.
      self._ocr_mask = EasyOcrBackend(EasyOcrConfig(languages=langs, gpu=True))
    return self._ocr_mask

  def ocr_mask_inpaint(self):
    if self._ocr_mask_inpaint is None:
      from tiper.ocr.backend import MultiLangOcrBackend, MultiLangOcrConfig, RemoteOcrBackend, RemoteOcrConfig

      ocr_cfg = self.default_config.get("ocr", {}) or {}
      worker_url = ocr_cfg.get("worker_url")
      if worker_url:
        # Inpaint mask OCR runs via Paddle worker (separate venv).
        remote = RemoteOcrBackend(RemoteOcrConfig(url=str(worker_url), endpoint="/ocr", timeout_s=300))
        # Explicit passes as requested: Korean + English (Thai is in secondary pass).
        self._ocr_mask_inpaint = MultiLangOcrBackend(remote, MultiLangOcrConfig(languages=["korean", "en"]))
      else:
        # Fallback to legacy EasyOCR when worker is not configured.
        self._ocr_mask_inpaint = self.ocr_mask()

    return self._ocr_mask_inpaint

  def ocr_mask_inpaint_secondary(self):
    if self._ocr_mask_inpaint_secondary is None:
      from tiper.ocr.backend import MultiLangOcrBackend, MultiLangOcrConfig, RemoteOcrBackend, RemoteOcrConfig

      sec = (self.default_config.get("ocr", {}) or {}).get("secondary_pass") or {}
      if not (isinstance(sec, dict) and sec.get("enabled")):
        self._ocr_mask_inpaint_secondary = False
        return None

      worker_url = (self.default_config.get("ocr", {}) or {}).get("worker_url")
      if not worker_url:
        # Secondary is only used for mask; keep behaviour consistent with ocr_mask_secondary.
        self._ocr_mask_inpaint_secondary = self.ocr_mask_secondary() or False
        return self._ocr_mask_inpaint_secondary if self._ocr_mask_inpaint_secondary is not False else None

      remote = RemoteOcrBackend(RemoteOcrConfig(url=str(worker_url), endpoint="/ocr", timeout_s=300))
      # Explicit pass as requested: Thai.
      self._ocr_mask_inpaint_secondary = MultiLangOcrBackend(remote, MultiLangOcrConfig(languages=["th"]))

    if self._ocr_mask_inpaint_secondary is False:
      return None
    return self._ocr_mask_inpaint_secondary

  def ocr_mask_secondary(self):
    if self._ocr_mask_secondary is None:
      sec = (self.default_config.get("ocr", {}) or {}).get("secondary_pass") or {}
      if not (isinstance(sec, dict) and sec.get("enabled")):
        self._ocr_mask_secondary = False
        return None

      sec_lang = str(sec.get("language") or "th").strip().lower()
      if not sec_lang:
        self._ocr_mask_secondary = False
        return None

      # EasyOCR constraint: Thai requires English ("th" is only compatible with "en").
      langs = [sec_lang]
      if sec_lang == "th" and "en" not in langs:
        langs.append("en")

      from tiper.ocr.backend import EasyOcrBackend, EasyOcrConfig

      self._ocr_mask_secondary = EasyOcrBackend(EasyOcrConfig(languages=langs, gpu=True))

    # When disabled we store False sentinel to avoid re-checking on every call.
    if self._ocr_mask_secondary is False:
      return None
    return self._ocr_mask_secondary

  def ocr_translation(self):
    if self._ocr_translation is None:
      from tiper.ocr.backend import EasyOcrBackend, EasyOcrConfig, RemoteOcrBackend, RemoteOcrConfig

      worker_url = self.default_config.get("ocr", {}).get("worker_url")
      if worker_url:
        # Translation OCR runs on bubble crops; use the worker's hybrid VL+fallback endpoint.
        self._ocr_translation = RemoteOcrBackend(RemoteOcrConfig(url=str(worker_url), endpoint="/ocr_bubble"))
      else:
        langs = list(self.default_config.get("ocr", {}).get("languages") or ["ko", "en"])
        self._ocr_translation = EasyOcrBackend(EasyOcrConfig(languages=langs, gpu=True))
    return self._ocr_translation

  def inpainter(self):
    if self._inpainter is None:
      from tiper.blaster.blaster_module import BigLamaInpainter

      model_dir = self.tiper_root / "models" / "big-lama"
      device = str(self.default_config.get("inpaint", {}).get("device") or "cuda")
      self._inpainter = BigLamaInpainter(str(model_dir), device=device)
    return self._inpainter

  def slice_yolo(self, model_path: str):
    if self._slice_yolo is None or self._slice_yolo_path != model_path:
      from tiper.cutting.boxes import load_yolo11_model

      self._slice_yolo = load_yolo11_model(model_path)
      self._slice_yolo_path = model_path
    return self._slice_yolo

  def slice_text_readers(self, langs: list[str], device: str = "auto"):
    key = (tuple([str(l or "").strip() for l in (langs or []) if str(l or "").strip()]), str(device or "auto"))
    if self._slice_text_readers is None or self._slice_text_langs != key:
      from tiper.cutting.boxes import load_text_readers

      self._slice_text_readers = load_text_readers(list(key[0]), device=key[1])
      self._slice_text_langs = key
      self._slice_text_device = key[1]
    return self._slice_text_readers or []

  def bubble_cls(self, model_path: str):
    if self._bubble_cls is None or self._bubble_cls_path != model_path:
      from tiper.cutting.boxes import load_yolo11_model

      self._bubble_cls = load_yolo11_model(model_path)
      self._bubble_cls_path = model_path
    return self._bubble_cls

  def unload_slice_models(self):
    self._slice_yolo = None
    self._slice_yolo_path = None
    self._slice_text_readers = None
    self._slice_text_langs = None
    self._slice_text_device = None
    try:
      import torch  # type: ignore

      if torch.cuda.is_available():
        torch.cuda.empty_cache()
    except Exception:
      pass

  def unload_ocr_models(self):
    self._ocr_mask = None
    self._ocr_mask_secondary = None
    self._ocr_mask_inpaint = None
    self._ocr_mask_inpaint_secondary = None
    self._ocr_translation = None
    try:
      import torch  # type: ignore

      if torch.cuda.is_available():
        torch.cuda.empty_cache()
    except Exception:
      pass

  def unload_inpainter(self):
    self._inpainter = None
    try:
      import torch  # type: ignore

      if torch.cuda.is_available():
        torch.cuda.empty_cache()
    except Exception:
      pass

  def unload_bubble_cls(self):
    self._bubble_cls = None
    self._bubble_cls_path = None
    try:
      import torch  # type: ignore

      if torch.cuda.is_available():
        torch.cuda.empty_cache()
    except Exception:
      pass
