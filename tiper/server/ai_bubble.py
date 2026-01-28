from __future__ import annotations

import base64
import io
import os
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image

try:
  import torch
except Exception:  # pragma: no cover
  torch = None

try:
  import cv2
except Exception:  # pragma: no cover
  cv2 = None

try:
  from ultralytics import YOLO
except Exception:  # pragma: no cover
  YOLO = None


Image.MAX_IMAGE_PIXELS = None


def _decode_base64_image(image_base64: str) -> Image.Image:
  if "," in image_base64 and image_base64.strip().lower().startswith("data:"):
    image_base64 = image_base64.split(",", 1)[1]
  data = base64.b64decode(image_base64, validate=False)
  return Image.open(io.BytesIO(data)).convert("RGB")


def _torch_device(default: Optional[str] = None) -> str:
  if default:
    return default
  if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
    return "cuda"
  return "cpu"


def _distance_center_and_core_bbox(
  mask_u8: np.ndarray,
  core_frac: float = 0.55,
  center_pow: float = 2.0,
) -> Tuple[Tuple[float, float], Tuple[float, float, float, float]]:
  ys, xs = np.where(mask_u8 > 0)
  if len(xs) == 0:
    return (0.0, 0.0), (0.0, 0.0, 0.0, 0.0)

  x1b, x2b = float(xs.min()), float(xs.max())
  y1b, y2b = float(ys.min()), float(ys.max())

  if cv2 is None:
    cx = float(xs.mean())
    cy = float(ys.mean())
    return (cx, cy), (x1b, y1b, x2b, y2b)

  dist = cv2.distanceTransform((mask_u8 > 0).astype(np.uint8), cv2.DIST_L2, 5)
  _min_val, max_val, _min_loc, max_loc = cv2.minMaxLoc(dist)

  if not (max_val > 0):
    cx = float(xs.mean())
    cy = float(ys.mean())
    return (cx, cy), (x1b, y1b, x2b, y2b)

  # Weighted center biases toward the safest interior region and ignores tails/spikes.
  weights = (dist ** float(center_pow)).astype(np.float32, copy=False)
  total = float(weights.sum())
  if total > 0:
    sum_x = float((weights.sum(axis=0) * np.arange(mask_u8.shape[1], dtype=np.float32)).sum())
    sum_y = float((weights.sum(axis=1) * np.arange(mask_u8.shape[0], dtype=np.float32)).sum())
    cx = sum_x / total
    cy = sum_y / total
  else:
    cx = float(max_loc[0])
    cy = float(max_loc[1])

  # If the weighted center lands outside the mask (can happen on concave shapes), snap to dist max.
  cx_i = int(round(cx))
  cy_i = int(round(cy))
  if cy_i < 0 or cy_i >= mask_u8.shape[0] or cx_i < 0 or cx_i >= mask_u8.shape[1] or mask_u8[cy_i, cx_i] == 0:
    cx, cy = float(max_loc[0]), float(max_loc[1])

  core = dist >= (max_val * float(core_frac))
  cy_idxs, cx_idxs = np.where(core)
  if len(cx_idxs) == 0:
    return (cx, cy), (x1b, y1b, x2b, y2b)

  x1c, x2c = float(cx_idxs.min()), float(cx_idxs.max())
  y1c, y2c = float(cy_idxs.min()), float(cy_idxs.max())
  return (cx, cy), (x1c, y1c, x2c, y2c)


class BubbleDetector:
  def __init__(self):
    self._lock = None
    self._model = None
    self.model_path = os.environ.get(
      "BUBBLE_MODEL_PATH",
      str(Path(__file__).resolve().parent.parent / "models" / "yolov8m_seg-speech-bubble" / "model.pt"),
    )
    self.device = _torch_device(os.environ.get("BUBBLE_DEVICE", None))
    self.conf = float(os.environ.get("BUBBLE_CONF", "0.2"))
    self.imgsz = int(os.environ.get("BUBBLE_IMGSZ", "1280"))
    self.core_frac = float(os.environ.get("BUBBLE_CORE_FRAC", "0.55"))
    self.center_pow = float(os.environ.get("BUBBLE_CENTER_POW", "2.0"))
    import threading

    self._lock = threading.Lock()

  def load(self, model_path: Optional[str] = None, device: Optional[str] = None) -> None:
    if YOLO is None:  # pragma: no cover
      raise RuntimeError("Missing dependency: ultralytics")
    with self._lock:
      if model_path:
        self.model_path = model_path
      if device:
        self.device = device
      if self._model is not None:
        return
      if not os.path.exists(self.model_path):
        raise FileNotFoundError(self.model_path)
      self._model = YOLO(self.model_path)

  def unload(self) -> None:
    with self._lock:
      self._model = None
    if torch is not None and hasattr(torch, "cuda"):
      try:
        torch.cuda.empty_cache()
      except Exception:
        pass

  def health(self) -> dict:
    return {
      "status": "ok",
      "model_loaded": self._model is not None,
      "model_path": self.model_path,
      "device": self.device,
      "conf": self.conf,
      "imgsz": self.imgsz,
      "core_frac": self.core_frac,
      "center_pow": self.center_pow,
    }

  def detect(self, img: Image.Image, roi_offset: Tuple[float, float], point_abs: Tuple[float, float], conf: Optional[float], imgsz: Optional[int]) -> dict:
    if self._model is None:
      self.load()

    conf_val = float(conf if conf is not None else self.conf)
    imgsz_val = int(imgsz if imgsz is not None else self.imgsz)

    roi_x, roi_y = roi_offset
    px_abs, py_abs = point_abs
    px = float(px_abs - roi_x)
    py = float(py_abs - roi_y)

    with self._lock:
      kwargs = {
        "conf": conf_val,
        "imgsz": imgsz_val,
        "device": self.device,
        "verbose": False,
        "retina_masks": True,
      }
      try:
        results = self._model.predict(img, **kwargs)
      except TypeError:
        kwargs.pop("retina_masks", None)
        results = self._model.predict(img, **kwargs)

    if not results:
      return {"selected": None, "bubbles": []}

    r0 = results[0]
    boxes = getattr(r0, "boxes", None)
    if boxes is None or len(boxes) == 0:
      return {"selected": None, "bubbles": []}

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    clses = boxes.cls.cpu().numpy()

    masks = getattr(r0, "masks", None)
    mask_data = None
    if masks is not None and getattr(masks, "data", None) is not None:
      try:
        mask_data = masks.data.cpu().numpy()
      except Exception:
        mask_data = None

    bubbles = []
    for i, ((x1, y1, x2, y2), prob, cls_id) in enumerate(zip(xyxy, confs, clses)):
      bubbles.append(
        {
          "index": int(i),
          "confidence": float(prob),
          "cls": int(cls_id),
          "bbox": {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)},
        }
      )

    def contains_point(b: dict) -> bool:
      bb = b["bbox"]
      return bb["x1"] <= px <= bb["x2"] and bb["y1"] <= py <= bb["y2"]

    inside = [b for b in bubbles if contains_point(b)]
    if inside:
      inside.sort(key=lambda b: ((b["bbox"]["x2"] - b["bbox"]["x1"]) * (b["bbox"]["y2"] - b["bbox"]["y1"]), -b["confidence"]))
      chosen = inside[0]
    else:
      def dist2_to_center(b: dict) -> float:
        bb = b["bbox"]
        cx = (bb["x1"] + bb["x2"]) / 2.0
        cy = (bb["y1"] + bb["y2"]) / 2.0
        dx = cx - px
        dy = cy - py
        return dx * dx + dy * dy

      bubbles.sort(key=lambda b: (dist2_to_center(b), -b["confidence"]))
      chosen = bubbles[0]

    bb = chosen["bbox"]
    center_x = (bb["x1"] + bb["x2"]) / 2.0
    center_y = (bb["y1"] + bb["y2"]) / 2.0
    core = (bb["x1"], bb["y1"], bb["x2"], bb["y2"])

    if mask_data is not None:
      try:
        m = mask_data[chosen["index"]]
        mask_u8 = (m > 0.5).astype(np.uint8) * 255
        (cx, cy), (x1c, y1c, x2c, y2c) = _distance_center_and_core_bbox(
          mask_u8,
          core_frac=self.core_frac,
          center_pow=self.center_pow,
        )
        center_x, center_y = cx, cy
        core = (x1c, y1c, x2c, y2c)
      except Exception:
        pass

    selected = {
      "confidence": chosen["confidence"],
      "cls": chosen["cls"],
      "center": {"x": float(center_x + roi_x), "y": float(center_y + roi_y)},
      "bbox": {
        "x1": float(bb["x1"] + roi_x),
        "y1": float(bb["y1"] + roi_y),
        "x2": float(bb["x2"] + roi_x),
        "y2": float(bb["y2"] + roi_y),
      },
      "core_bbox": {
        "x1": float(core[0] + roi_x),
        "y1": float(core[1] + roi_y),
        "x2": float(core[2] + roi_x),
        "y2": float(core[3] + roi_y),
      },
    }
    return {"selected": selected, "bubbles": bubbles}


def detect_request(detector: BubbleDetector, payload: dict) -> dict:
  image_base64 = payload.get("image_base64")
  roi_offset = payload.get("roi_offset") or {}
  point = payload.get("point") or {}
  conf = payload.get("conf")
  imgsz = payload.get("imgsz")

  if not image_base64:
    raise ValueError("missing_image_base64")

  img = _decode_base64_image(image_base64)

  roi_x = float(roi_offset.get("x", 0))
  roi_y = float(roi_offset.get("y", 0))
  px = float(point.get("x"))
  py = float(point.get("y"))

  t0 = time.time()
  result = detector.detect(img, roi_offset=(roi_x, roi_y), point_abs=(px, py), conf=conf, imgsz=imgsz)
  result["timings_ms"] = {"total": int((time.time() - t0) * 1000)}
  return result
