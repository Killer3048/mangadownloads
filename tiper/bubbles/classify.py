from __future__ import annotations

import math
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

from PIL import Image

try:  # pragma: no cover - optional at runtime
  import numpy as np
except Exception:  # pragma: no cover
  np = None


DEFAULT_CLASS_LABELS = [
  "elipse",
  "cloude",
  "other",
  "rectangle",
  "sea_uchirin",
  "thorn",
]


def _clip_box(left: float, top: float, right: float, bottom: float, width: int, height: int) -> Tuple[int, int, int, int]:
  l = max(0, min(int(width), int(math.floor(left))))
  t = max(0, min(int(height), int(math.floor(top))))
  r = max(0, min(int(width), int(math.ceil(right))))
  b = max(0, min(int(height), int(math.ceil(bottom))))
  if r <= l:
    r = min(int(width), l + 1)
  if b <= t:
    b = min(int(height), t + 1)
  return l, t, r, b


def _resolve_labels(result: Optional[Any], model: Any) -> Dict[int, str]:
  names = None
  if result is not None:
    names = getattr(result, "names", None)
  if not names:
    names = getattr(model, "names", None)

  label_map: Dict[int, str] = {}
  if isinstance(names, dict):
    for k, v in names.items():
      try:
        label_map[int(k)] = str(v)
      except Exception:
        continue
  elif isinstance(names, (list, tuple)):
    for i, v in enumerate(names):
      label_map[int(i)] = str(v)

  if not label_map:
    for i, v in enumerate(DEFAULT_CLASS_LABELS):
      label_map[int(i)] = v

  return label_map


def _extract_top1(result: Any) -> Tuple[Optional[int], Optional[float]]:
  probs = getattr(result, "probs", None)
  if probs is None:
    return None, None

  top1 = getattr(probs, "top1", None)
  if top1 is not None:
    try:
      class_id = int(top1)
    except Exception:
      class_id = None
    conf_val = getattr(probs, "top1conf", None)
    try:
      conf = float(conf_val) if conf_val is not None else None
    except Exception:
      conf = None
    return class_id, conf

  data = getattr(probs, "data", None)
  if data is None:
    data = probs

  try:
    if hasattr(data, "detach"):
      data = data.detach().cpu().numpy()
    elif hasattr(data, "cpu"):
      data = data.cpu().numpy()
  except Exception:
    pass

  try:
    if np is not None:
      arr = np.asarray(data).reshape(-1)
      if arr.size == 0:
        return None, None
      class_id = int(arr.argmax())
      conf = float(arr[class_id])
      return class_id, conf
  except Exception:
    pass

  try:
    arr_list = list(data)
    if not arr_list:
      return None, None
    class_id = int(max(range(len(arr_list)), key=lambda i: arr_list[i]))
    conf = float(arr_list[class_id])
    return class_id, conf
  except Exception:
    return None, None


def classify_bubbles(
  image: Image.Image,
  bubbles: Sequence[dict],
  model: Any,
  min_conf: Optional[float] = 0.6,
  imgsz: Optional[int] = None,
  device: str = "auto",
  on_progress: Optional[Callable[[int, int], None]] = None,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[int, str]]:
  width, height = image.size
  crops = []
  ids = []

  for b in bubbles:
    bid = b.get("id") or ""
    if not bid:
      continue
    bb = b.get("bbox") or {}
    try:
      left_f = float(bb.get("left", 0.0))
      top_f = float(bb.get("top", 0.0))
      right_f = float(bb.get("right", left_f))
      bottom_f = float(bb.get("bottom", top_f))
    except Exception:
      continue

    left, top, right, bottom = _clip_box(left_f, top_f, right_f, bottom_f, width, height)
    if right <= left or bottom <= top:
      continue

    # Convert to grayscale first to remove color information, then back to RGB for model
    crop = image.crop((left, top, right, bottom)).convert("L").convert("RGB")
    crops.append(crop)
    ids.append(str(bid))

  if not crops:
    return {}, {}

  predict_kwargs: Dict[str, Any] = {"verbose": False}
  if imgsz:
    predict_kwargs["imgsz"] = int(imgsz)
  if device and device != "auto":
    predict_kwargs["device"] = str(device)

  try:
    results = model.predict(crops, **predict_kwargs)
  except Exception:
    results = []
    for crop in crops:
      try:
        res = model.predict(crop, **predict_kwargs)
        if isinstance(res, list) and res:
          results.append(res[0])
        elif res is not None:
          results.append(res)
        else:
          results.append(None)
      except Exception:
        results.append(None)

  first_result = None
  if results:
    for r in results:
      if r is not None:
        first_result = r
        break

  labels_map = _resolve_labels(first_result, model)
  classes: Dict[str, Dict[str, Any]] = {}
  total = len(ids)

  limit = min(len(ids), len(results)) if results else 0
  for idx in range(limit):
    r = results[idx]
    if r is None:
      if on_progress:
        on_progress(idx + 1, total)
      continue
    class_id, conf = _extract_top1(r)
    if class_id is None:
      if on_progress:
        on_progress(idx + 1, total)
      continue

    label = labels_map.get(int(class_id))
    if not label and 0 <= int(class_id) < len(DEFAULT_CLASS_LABELS):
      label = DEFAULT_CLASS_LABELS[int(class_id)]
    if not label:
      label = str(class_id)

    entry: Dict[str, Any] = {
      "class": str(label),
      "class_id": int(class_id),
    }
    if conf is not None:
      entry["confidence"] = float(conf)
      if min_conf is not None:
        entry["definite"] = bool(float(conf) >= float(min_conf))

    classes[ids[idx]] = entry
    if on_progress:
      on_progress(idx + 1, total)

  if on_progress and limit < total:
    for idx in range(limit, total):
      on_progress(idx + 1, total)

  return classes, labels_map
