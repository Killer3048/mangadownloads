from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


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


def _rect_poly(x1: float, y1: float, x2: float, y2: float) -> List[List[float]]:
  return [[float(x1), float(y1)], [float(x2), float(y1)], [float(x2), float(y2)], [float(x1), float(y2)]]


def _curvature_poly_from_bbox(x1: float, y1: float, x2: float, y2: float, steps: int = 12) -> List[List[float]]:
  """
  Generate an approximate bubble shape (smooth oval/super-ellipse) using cubic Bezier curves
  connecting the midpoints of the bounding box sides.
  """
  l, t, r, b = float(x1), float(y1), float(x2), float(y2)
  cx = (l + r) / 2.0
  cy = (t + b) / 2.0
  rx = (r - l) / 2.0
  ry = (b - t) / 2.0

  # Kappa for circle approximation approx 0.55228 for circle.
  # We use a slightly higher value to make it a bit more "squarish" but still smooth,
  # or stick to circle-like. User mentioned "curvature pen" which usually implies smooth tangents.
  # Let's use standard circle approx.
  k = 0.55228475
  
  # Control point offsets
  dx = rx * k
  dy = ry * k

  # Segments: (Start, C1, C2, End)
  # 1. Top midpoint -> Right midpoint
  # Start: (cx, t)
  # C1:    (cx + dx, t)
  # C2:    (r, cy - dy)
  # End:   (r, cy)
  s1 = ((cx, t), (cx + dx, t), (r, cy - dy), (r, cy))

  # 2. Right -> Bottom
  # Start: (r, cy)
  # C1:    (r, cy + dy)
  # C2:    (cx + dx, b)
  # End:   (cx, b)
  s2 = ((r, cy), (r, cy + dy), (cx + dx, b), (cx, b))

  # 3. Bottom -> Left
  # Start: (cx, b)
  # C1:    (cx - dx, b)
  # C2:    (l, cy + dy)
  # End:   (l, cy)
  s3 = ((cx, b), (cx - dx, b), (l, cy + dy), (l, cy))

  # 4. Left -> Top
  # Start: (l, cy)
  # C1:    (l, cy - dy)
  # C2:    (cx - dx, t)
  # End:   (cx, t)
  s4 = ((l, cy), (l, cy - dy), (cx - dx, t), (cx, t))

  points: List[List[float]] = []

  def _bezier(p0, p1, p2, p3, t):
    mt = 1.0 - t
    c0 = mt**3
    c1 = 3 * (mt**2) * t
    c2 = 3 * mt * (t**2)
    c3 = t**3
    x = c0*p0[0] + c1*p1[0] + c2*p2[0] + c3*p3[0]
    y = c0*p0[1] + c1*p1[1] + c2*p2[1] + c3*p3[1]
    return [x, y]

  for seg in (s1, s2, s3, s4):
    p0, p1, p2, p3 = seg
    # Add points (skip last to avoid duplicate, except very last overall?)
    # Usually easier to just add steps points.
    for i in range(steps):
      t = float(i) / float(steps)
      points.append(_bezier(p0, p1, p2, p3, t))
  
  # Close the loop
  # points.append([points[0][0], points[0][1]]) 
  return points


def _iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
  ax1, ay1, ax2, ay2 = [float(v) for v in a[:4]]
  bx1, by1, bx2, by2 = [float(v) for v in b[:4]]
  ix1 = max(ax1, bx1)
  iy1 = max(ay1, by1)
  ix2 = min(ax2, bx2)
  iy2 = min(ay2, by2)
  iw = max(0.0, ix2 - ix1)
  ih = max(0.0, iy2 - iy1)
  inter = iw * ih
  area_a = max(1e-6, (ax2 - ax1) * (ay2 - ay1))
  area_b = max(1e-6, (bx2 - bx1) * (by2 - by1))
  union = area_a + area_b - inter
  return float(inter / union) if union > 0 else 0.0


def _poly_rel(poly_abs: Sequence[Sequence[float]], left: int, top: int, w: int, h: int) -> List[List[float]]:
  out: List[List[float]] = []
  for p in poly_abs:
    try:
      x, y = p
      xr = float(x) - float(left)
      yr = float(y) - float(top)
      xr = max(0.0, min(float(w), xr))
      yr = max(0.0, min(float(h), yr))
      out.append([xr, yr])
    except Exception:
      continue
  return out


def _bbox_from_mask(mask_u8: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
  ys, xs = np.where(mask_u8 > 0)
  if xs.size == 0:
    return None
  x1 = int(xs.min())
  x2 = int(xs.max()) + 1
  y1 = int(ys.min())
  y2 = int(ys.max()) + 1
  return x1, y1, x2, y2


def _core_bbox_from_mask(mask_u8: np.ndarray, core_frac: float = 0.55) -> Tuple[int, int, int, int]:
  """
  Compute an inner "core bbox" using distance transform. Avoids tails/spikes of the bubble.
  Falls back to the full non-zero bbox if distance transform degenerates.
  """
  h, w = mask_u8.shape[:2]
  bbox = _bbox_from_mask(mask_u8)
  if bbox is None:
    return 0, 0, int(w), int(h)

  try:
    import cv2
  except Exception:
    return bbox

  dist = cv2.distanceTransform((mask_u8 > 0).astype(np.uint8), cv2.DIST_L2, 5)
  max_val = float(dist.max()) if dist.size else 0.0
  if not (max_val > 0):
    return bbox

  core = dist >= (max_val * float(core_frac))
  ys, xs = np.where(core)
  if xs.size == 0:
    return bbox
  x1 = int(xs.min())
  x2 = int(xs.max()) + 1
  y1 = int(ys.min())
  y2 = int(ys.max()) + 1
  return x1, y1, x2, y2


def compute_bubble_geometry_from_page(
  image: Image.Image,
  bubbles: Sequence[dict],
  yolo_model,
  conf: float = 0.25,
  imgsz: int = 1280,
  iou_thr: float = 0.30,
  core_frac: float = 0.55,
) -> Dict[str, Any]:
  """
  Run YOLO-seg once on the full page, then IoU-match each bubble bbox to a seg instance.
  Returns per-bubble core_bbox in absolute (page) coordinates.
  """
  page = image.convert("RGB")
  page_w, page_h = page.size

  kwargs = {"conf": float(conf), "imgsz": int(imgsz), "verbose": False, "retina_masks": True}
  try:
    res = yolo_model.predict(page, **kwargs)
  except TypeError:
    kwargs.pop("retina_masks", None)
    res = yolo_model.predict(page, **kwargs)

  r0 = res[0] if res else None
  boxes_xyxy: List[List[float]] = []
  polys_abs: List[List[List[float]]] = []
  if r0 is not None and getattr(r0, "boxes", None) is not None and len(r0.boxes) > 0:
    boxes_xyxy = r0.boxes.xyxy.cpu().numpy().astype(float).tolist()
    masks = getattr(r0, "masks", None)
    if masks is not None and hasattr(masks, "xy") and masks.xy:
      for poly in masks.xy:
        try:
          polys_abs.append([[float(x), float(y)] for x, y in np.asarray(poly).tolist()])
        except Exception:
          polys_abs.append([])
    else:
      polys_abs = [[] for _ in boxes_xyxy]

  try:
    import cv2
  except Exception:
    cv2 = None  # type: ignore

  items: Dict[str, Any] = {}
  for b in bubbles:
    bid = str(b.get("id") or "")
    if not bid:
      continue
    bb = b.get("bbox") or {}
    left, top, right, bottom = _clip_box(
      float(bb.get("left", 0.0)),
      float(bb.get("top", 0.0)),
      float(bb.get("right", 0.0)),
      float(bb.get("bottom", 0.0)),
      width=page_w,
      height=page_h,
    )
    cw = max(1, right - left)
    ch = max(1, bottom - top)

    bubble_xyxy = [float(left), float(top), float(right), float(bottom)]
    best_iou = 0.0
    best_idx: Optional[int] = None
    for i, det_xyxy in enumerate(boxes_xyxy):
      iou = _iou_xyxy(bubble_xyxy, det_xyxy)
      if iou > best_iou:
        best_iou = float(iou)
        best_idx = int(i)

    source: str
    poly_rel: List[List[float]]
    if best_idx is not None and best_iou >= float(iou_thr) and best_idx < len(polys_abs) and polys_abs[best_idx]:
      source = "seg_contour"
      poly_rel = _poly_rel(polys_abs[best_idx], left=left, top=top, w=cw, h=ch)
      if len(poly_rel) < 3:
        source = "bbox_fallback_bad_poly"
        poly_rel = _curvature_poly_from_bbox(0.0, 0.0, float(cw), float(ch))
    else:
      source = "bbox_fallback"
      poly_rel = _curvature_poly_from_bbox(0.0, 0.0, float(cw), float(ch))

    core_abs = {"left": float(left), "top": float(top), "right": float(right), "bottom": float(bottom)}
    if cv2 is not None:
      mask = np.zeros((ch, cw), dtype=np.uint8)
      pts = np.asarray(poly_rel, dtype=np.int32)
      if pts.size:
        cv2.fillPoly(mask, [pts], 255)
      x1c, y1c, x2c, y2c = _core_bbox_from_mask(mask, core_frac=float(core_frac))
      core_abs = {
        "left": float(x1c + left),
        "top": float(y1c + top),
        "right": float(x2c + left),
        "bottom": float(y2c + top),
      }

    items[bid] = {
      "source": source,
      "best_iou": float(best_iou),
      "seg_det_index": int(best_idx) if best_idx is not None else None,
      "core_bbox": core_abs,
      "poly_rel": poly_rel,
    }

  return {
    "schema_version": 1,
    "conf": float(conf),
    "imgsz": int(imgsz),
    "iou_thr": float(iou_thr),
    "core_frac": float(core_frac),
    "detections": int(len(boxes_xyxy)),
    "items": items,
  }

