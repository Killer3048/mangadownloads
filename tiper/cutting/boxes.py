from __future__ import annotations

from typing import Any, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image

try:  # pragma: no cover - optional at runtime
  import torch
except Exception:  # pragma: no cover
  torch = None

try:  # pragma: no cover - optional at runtime
  from ultralytics import YOLO
except Exception:  # pragma: no cover
  YOLO = None

try:  # pragma: no cover - optional at runtime
  import easyocr
except Exception:  # pragma: no cover
  easyocr = None


Box = Tuple[float, float, float, float, float, int]


def _unique_langs(langs: Iterable[str]) -> List[str]:
  out: List[str] = []
  for l in langs or []:
    s = str(l or "").strip()
    if s and s not in out:
      out.append(s)
  return out


def build_text_language_groups(lang_list: Sequence[str]) -> List[List[str]]:
  langs = _unique_langs(lang_list)
  groups: List[List[str]] = []

  if "ko" in langs:
    ko_group = ["ko"]
    if "en" not in langs:
      ko_group.append("en")
    else:
      ko_group.append("en")
      langs.remove("en")
    langs.remove("ko")
    groups.append(ko_group)

  if langs:
    groups.append(langs)

  return groups


def load_yolo11_model(model_path: str) -> Any:
  if YOLO is None:  # pragma: no cover
    raise RuntimeError("Missing dependency: ultralytics")
  return YOLO(model_path)


def load_text_readers(lang_list: Sequence[str], device: str = "auto") -> List[Any]:
  if easyocr is None:  # pragma: no cover
    raise RuntimeError("Missing dependency: easyocr")

  use_gpu = False
  if str(device or "auto") != "cpu" and torch is not None and torch.cuda.is_available():
    use_gpu = True

  groups = build_text_language_groups(lang_list)
  if not groups:
    return []

  readers: List[Any] = []
  for g in groups:
    readers.append(easyocr.Reader(g, gpu=use_gpu))
  return readers


def detect_yolo_boxes(
  image: Image.Image,
  model: Any,
  chunk_height: int = 4096,
  overlap: int = 256,
  conf: float = 0.25,
  imgsz: int = 1280,
  device: str = "auto",
  min_w: int = 20,
  min_h: int = 20,
) -> List[Box]:
  width, height = image.size
  boxes: List[Box] = []

  y = 0
  while y < height:
    chunk_top = max(0, int(y - overlap))
    chunk_bottom = min(height, int(y + chunk_height + overlap))

    crop = image.crop((0, chunk_top, width, chunk_bottom))

    predict_kwargs = {
      "conf": float(conf),
      "imgsz": int(imgsz),
      "verbose": False,
    }
    if device and device != "auto":
      predict_kwargs["device"] = device

    results = model.predict(crop, **predict_kwargs)
    for r in results:
      bboxes = getattr(r, "boxes", None)
      if bboxes is None or len(bboxes) == 0:
        continue

      xyxy = bboxes.xyxy.cpu().numpy()
      confs = bboxes.conf.cpu().numpy()
      clses = bboxes.cls.cpu().numpy()

      for (x1, y1, x2, y2), c, cls_id in zip(xyxy, confs, clses):
        if (y2 - y1) < float(min_h) or (x2 - x1) < float(min_w):
          continue

        g_y1 = float(chunk_top + y1)
        g_y2 = float(chunk_top + y2)
        g_x1 = float(x1)
        g_x2 = float(x2)

        g_y1 = max(0.0, min(g_y1, height - 1.0))
        g_y2 = max(0.0, min(g_y2, height - 1.0))
        g_x1 = max(0.0, min(g_x1, width - 1.0))
        g_x2 = max(0.0, min(g_x2, width - 1.0))

        if g_y2 <= g_y1 or g_x2 <= g_x1:
          continue

        boxes.append((g_x1, g_y1, g_x2, g_y2, float(c), int(cls_id)))

    y += chunk_height

  return boxes


def detect_text_boxes(
  image: Image.Image,
  reader: Any,
  chunk_height: int = 4096,
  overlap: int = 256,
  conf: float = 0.4,
  min_w: int = 20,
  min_h: int = 20,
) -> List[Box]:
  width, height = image.size
  boxes: List[Box] = []

  y = 0
  while y < height:
    chunk_top = max(0, int(y - overlap))
    chunk_bottom = min(height, int(y + chunk_height + overlap))

    crop = image.crop((0, chunk_top, width, chunk_bottom)).convert("RGB")
    crop_np = np.array(crop)

    results = reader.readtext(
      crop_np,
      detail=1,
      paragraph=False,
    )

    for (bbox, _text, prob) in results:
      if float(prob) < float(conf):
        continue

      xs = [p[0] for p in bbox]
      ys = [p[1] for p in bbox]
      x1 = min(xs)
      x2 = max(xs)
      y1 = min(ys)
      y2 = max(ys)

      w = x2 - x1
      h = y2 - y1
      if w < float(min_w) or h < float(min_h):
        continue

      g_y1 = float(chunk_top + y1)
      g_y2 = float(chunk_top + y2)
      g_x1 = float(x1)
      g_x2 = float(x2)

      g_y1 = max(0.0, min(g_y1, height - 1.0))
      g_y2 = max(0.0, min(g_y2, height - 1.0))
      g_x1 = max(0.0, min(g_x1, width - 1.0))
      g_x2 = max(0.0, min(g_x2, width - 1.0))

      if g_y2 <= g_y1 or g_x2 <= g_x1:
        continue

      boxes.append((g_x1, g_y1, g_x2, g_y2, float(prob), -1))

    y += chunk_height

  return boxes
