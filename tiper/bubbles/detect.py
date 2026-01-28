from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from tiper.cutting.smart_slicer import SmartSlicer


BoxXYXY = Tuple[float, float, float, float]


def _iou_xyxy(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
  x1 = np.maximum(box[0], boxes[:, 0])
  y1 = np.maximum(box[1], boxes[:, 1])
  x2 = np.minimum(box[2], boxes[:, 2])
  y2 = np.minimum(box[3], boxes[:, 3])

  inter_w = np.maximum(0.0, x2 - x1)
  inter_h = np.maximum(0.0, y2 - y1)
  inter = inter_w * inter_h

  area1 = np.maximum(0.0, box[2] - box[0]) * np.maximum(0.0, box[3] - box[1])
  area2 = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
  union = np.maximum(1e-9, area1 + area2 - inter)
  return inter / union


def nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_thr: float = 0.5) -> List[int]:
  if boxes.size == 0:
    return []
  order = scores.argsort()[::-1]
  keep: List[int] = []
  while order.size > 0:
    i = int(order[0])
    keep.append(i)
    if order.size == 1:
      break
    ious = _iou_xyxy(boxes[i], boxes[order[1:]])
    order = order[1:][ious < iou_thr]
  return keep


@dataclass(frozen=True)
class BubbleDetConfig:
  model_conf: float = 0.7
  imgsz: int = 1280
  device: str = "auto"
  chunk_h: int = 4096
  overlap: int = 256
  nms_iou: float = 0.5
  # Smart chunking (YOLO slices chosen by SmartSlicer heuristics).
  smart_chunking: bool = True
  slice_min_h: int = 4000
  slice_max_h: int = 8000
  slice_margin: int = 20
  edge_weight: float = 1.0
  variance_weight: float = 0.5
  gradient_weight: float = 0.5
  white_space_weight: float = 2.0
  distance_penalty: float = 0.00001
  min_w: int = 20
  min_h: int = 20


def _compute_smart_cuts(height: int, slicer: SmartSlicer, cost_map: np.ndarray, min_h: int, max_h: int) -> List[int]:
  preferred_h = int((min_h + max_h) // 2)
  cuts = [0]
  current = 0
  while True:
    remaining = height - current
    if remaining <= max_h:
      cuts.append(height)
      break

    cut_y = slicer.find_best_cut(cost_map, current, min_h=min_h, max_h=max_h, preferred_h=preferred_h)
    if cut_y <= current:
      cut_y = current + min_h

    if height - cut_y < min_h:
      cut_y = height

    cut_y = min(cut_y, height)
    cuts.append(int(cut_y))
    current = int(cut_y)

    if current >= height:
      break

  dedup: List[int] = []
  for c in cuts:
    if not dedup or c != dedup[-1]:
      dedup.append(int(c))
  return dedup


def _smart_slice_boundaries(image: Image.Image, cfg: BubbleDetConfig, boxes: Optional[Sequence[Tuple[float, float, float, float, float, int]]] = None) -> List[int]:
  _w, h = image.size
  if h <= 0:
    return [0]

  min_h = int(cfg.slice_min_h)
  max_h = int(cfg.slice_max_h)

  slicer = SmartSlicer(
    edge_weight=float(cfg.edge_weight),
    variance_weight=float(cfg.variance_weight),
    gradient_weight=float(cfg.gradient_weight),
    white_space_weight=float(cfg.white_space_weight),
    distance_penalty=float(cfg.distance_penalty),
    forbidden_cost=1e9,
  )
  cost_map = slicer.compute_cost_map(image, boxes=list(boxes or []), margin=int(cfg.slice_margin))
  return _compute_smart_cuts(h, slicer, cost_map, min_h=min_h, max_h=max_h)


def compute_smart_slice_boundaries(
  image: Image.Image,
  cfg: BubbleDetConfig,
  boxes: Optional[Sequence[Tuple[float, float, float, float, float, int]]] = None,
) -> List[int]:
  return _smart_slice_boundaries(image, cfg, boxes=boxes)


def detect_bubbles_chunked(
  image: Image.Image,
  model,
  cfg: BubbleDetConfig,
  cuts: Optional[Sequence[int]] = None,
  on_progress: Optional[Callable[[int, int], None]] = None,
) -> List[dict]:
  width, height = image.size
  boxes: List[BoxXYXY] = []
  scores: List[float] = []
  clses: List[int] = []

  chunk_h = max(64, int(cfg.chunk_h))
  overlap = max(0, int(cfg.overlap))
  if cfg.smart_chunking:
    cuts_list = [int(c) for c in cuts] if cuts else _smart_slice_boundaries(image, cfg)
    if len(cuts_list) < 2:
      cuts_list = [0, height]
    total_chunks = max(1, len(cuts_list) - 1)

    idx = 0
    for i in range(len(cuts_list) - 1):
      top = int(cuts_list[i])
      bottom = int(cuts_list[i + 1])
      if bottom <= top:
        idx += 1
        if on_progress:
          on_progress(idx, total_chunks)
        continue

      chunk_top = max(0, int(top))
      chunk_bottom = min(height, int(bottom))
      crop = image.crop((0, chunk_top, width, chunk_bottom)).convert("RGB")

      predict_kwargs = {
        "conf": float(cfg.model_conf),
        "imgsz": int(cfg.imgsz),
        "verbose": False,
      }
      if cfg.device and cfg.device != "auto":
        predict_kwargs["device"] = cfg.device

      results = model.predict(crop, **predict_kwargs)
      for r in results:
        bboxes = getattr(r, "boxes", None)
        if bboxes is None or len(bboxes) == 0:
          continue

        xyxy = bboxes.xyxy.cpu().numpy()
        confs = bboxes.conf.cpu().numpy()
        clz = bboxes.cls.cpu().numpy()

        for (x1, y1, x2, y2), c, cls_id in zip(xyxy, confs, clz):
          if (y2 - y1) < cfg.min_h or (x2 - x1) < cfg.min_w:
            continue
          g_x1 = float(max(0.0, min(float(x1), width - 1.0)))
          g_x2 = float(max(0.0, min(float(x2), width - 1.0)))
          g_y1 = float(max(0.0, min(float(chunk_top + y1), height - 1.0)))
          g_y2 = float(max(0.0, min(float(chunk_top + y2), height - 1.0)))
          if g_x2 <= g_x1 or g_y2 <= g_y1:
            continue
          boxes.append((g_x1, g_y1, g_x2, g_y2))
          scores.append(float(c))
          clses.append(int(cls_id))

      idx += 1
      if on_progress:
        on_progress(idx, total_chunks)
  else:
    total_chunks = int(np.ceil(height / float(chunk_h))) if height > 0 else 0
    y = 0
    idx = 0
    while y < height:
      chunk_top = max(0, int(y - overlap))
      chunk_bottom = min(height, int(y + chunk_h + overlap))

      crop = image.crop((0, chunk_top, width, chunk_bottom)).convert("RGB")

      predict_kwargs = {
        "conf": float(cfg.model_conf),
        "imgsz": int(cfg.imgsz),
        "verbose": False,
      }
      if cfg.device and cfg.device != "auto":
        predict_kwargs["device"] = cfg.device

      results = model.predict(crop, **predict_kwargs)
      for r in results:
        bboxes = getattr(r, "boxes", None)
        if bboxes is None or len(bboxes) == 0:
          continue

        xyxy = bboxes.xyxy.cpu().numpy()
        confs = bboxes.conf.cpu().numpy()
        clz = bboxes.cls.cpu().numpy()

        for (x1, y1, x2, y2), c, cls_id in zip(xyxy, confs, clz):
          if (y2 - y1) < cfg.min_h or (x2 - x1) < cfg.min_w:
            continue
          g_x1 = float(max(0.0, min(float(x1), width - 1.0)))
          g_x2 = float(max(0.0, min(float(x2), width - 1.0)))
          g_y1 = float(max(0.0, min(float(chunk_top + y1), height - 1.0)))
          g_y2 = float(max(0.0, min(float(chunk_top + y2), height - 1.0)))
          if g_x2 <= g_x1 or g_y2 <= g_y1:
            continue
          boxes.append((g_x1, g_y1, g_x2, g_y2))
          scores.append(float(c))
          clses.append(int(cls_id))

      idx += 1
      if on_progress:
        on_progress(idx, total_chunks)
      y += chunk_h

  if not boxes:
    return []

  arr_boxes = np.asarray(boxes, dtype=np.float32)
  arr_scores = np.asarray(scores, dtype=np.float32)
  keep = nms_xyxy(arr_boxes, arr_scores, float(cfg.nms_iou))

  out: List[dict] = []
  for i in keep:
    x1, y1, x2, y2 = arr_boxes[i].tolist()
    out.append(
      {
        "bbox": {"left": float(x1), "top": float(y1), "right": float(x2), "bottom": float(y2)},
        "confidence": float(arr_scores[i]),
        "cls": int(clses[i]),
        "source": "auto",
      }
    )
  return out
