from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import math
import numpy as np
from PIL import Image

from tiper.cutting.smart_slicer import SmartSlicer
from tiper.ocr.grouping import OcrToken, tokens_to_text


@dataclass(frozen=True)
class InpaintConfig:
  min_h: int = 4000
  max_h: int = 8000
  margin: int = 20
  # SmartSlicer weights
  edge_weight: float = 1.0
  variance_weight: float = 0.5
  gradient_weight: float = 0.5
  white_space_weight: float = 2.0
  distance_penalty: float = 0.00001


def compute_cuts(height: int, slicer: SmartSlicer, cost_map: np.ndarray, min_h: int, max_h: int) -> List[int]:
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


def _bubble_for_point(bubbles: Sequence[dict], x: float, y: float) -> Optional[dict]:
  best = None
  best_area = None
  for b in bubbles:
    bb = b.get("bbox") or {}
    left = float(bb.get("left"))
    top = float(bb.get("top"))
    right = float(bb.get("right"))
    bottom = float(bb.get("bottom"))
    if left <= x <= right and top <= y <= bottom:
      area = max(1.0, (right - left) * (bottom - top))
      if best is None or area < float(best_area):
        best = b
        best_area = area
  return best


def _bubble_for_rect(bubbles: Sequence[dict], left: float, top: float, right: float, bottom: float) -> Optional[dict]:
  best = None
  best_inter = 0.0
  best_area = None
  for b in bubbles:
    bb = b.get("bbox") or {}
    b_left = float(bb.get("left"))
    b_top = float(bb.get("top"))
    b_right = float(bb.get("right"))
    b_bottom = float(bb.get("bottom"))

    inter_w = max(0.0, min(right, b_right) - max(left, b_left))
    inter_h = max(0.0, min(bottom, b_bottom) - max(top, b_top))
    inter = inter_w * inter_h
    if inter <= 0.0:
      continue

    area = max(1.0, (b_right - b_left) * (b_bottom - b_top))
    if best is None or inter > best_inter or (inter == best_inter and area < float(best_area)):
      best = b
      best_inter = inter
      best_area = area
  return best


def _rect_from_poly(poly: Sequence[Sequence[float]]) -> Tuple[float, float, float, float]:
  xs = [float(p[0]) for p in poly]
  ys = [float(p[1]) for p in poly]
  return min(xs), min(ys), max(xs), max(ys)


def _estimate_line_count_from_tokens(tokens: Sequence[Tuple[float, float]]) -> int:
  """
  Rough line count estimate based on OCR token centers and heights.
  Uses simple 1D clustering of y-mid values.
  """
  if not tokens:
    return 1

  ys = np.asarray([float(t[0]) for t in tokens], dtype=np.float32)
  hs = np.asarray([max(1.0, float(t[1])) for t in tokens], dtype=np.float32)
  if ys.size == 0:
    return 1

  try:
    med_h = float(np.median(hs)) if hs.size else 12.0
  except Exception:
    med_h = 12.0
  tol = max(1.0, med_h * 0.60)

  order = np.argsort(ys)
  ys_sorted = ys[order]

  lines: List[float] = []
  counts: List[int] = []
  for y in ys_sorted.tolist():
    placed = False
    for i, ly in enumerate(lines):
      if abs(float(y) - float(ly)) <= tol:
        n = int(counts[i]) + 1
        lines[i] = (float(ly) * (n - 1) + float(y)) / float(n)
        counts[i] = n
        placed = True
        break
    if not placed:
      lines.append(float(y))
      counts.append(1)

  return max(1, min(12, int(len(lines))))


def compute_cost_map_and_cuts(
  image: Image.Image,
  bubbles: Sequence[dict],
  cfg: InpaintConfig,
) -> Tuple[np.ndarray, List[int]]:
  width, height = image.size
  if width <= 0 or height <= 0:
    return np.zeros((0,), dtype=np.float32), [0]

  slicer = SmartSlicer(
    edge_weight=float(cfg.edge_weight),
    variance_weight=float(cfg.variance_weight),
    gradient_weight=float(cfg.gradient_weight),
    white_space_weight=float(cfg.white_space_weight),
    distance_penalty=float(cfg.distance_penalty),
    forbidden_cost=1e9,
  )

  boxes_for_cost = []
  for b in bubbles:
    bb = b.get("bbox") or {}
    boxes_for_cost.append(
      (
        float(bb.get("left")),
        float(bb.get("top")),
        float(bb.get("right")),
        float(bb.get("bottom")),
        1.0,
        0,
      )
    )

  cost_map = slicer.compute_cost_map(image, boxes_for_cost, margin=int(cfg.margin))
  cuts = compute_cuts(height, slicer, cost_map, min_h=int(cfg.min_h), max_h=int(cfg.max_h))
  return cost_map, cuts


def ocr_texts_sliced(
  image: Image.Image,
  bubbles: Sequence[dict],
  ocr_backend,
  conf: float,
  lang: Optional[str],
  cuts: Sequence[int],
  on_progress: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, str]:
  width, height = image.size
  if width <= 0 or height <= 0:
    return {}

  tokens_by_id: Dict[str, List[OcrToken]] = {}
  for b in bubbles:
    bid = b.get("id")
    if bid:
      tokens_by_id[str(bid)] = []

  cuts_list = [int(c) for c in cuts] if cuts else [0, height]
  if len(cuts_list) < 2:
    cuts_list = [0, height]

  total_slices = max(1, len(cuts_list) - 1)
  for i in range(len(cuts_list) - 1):
    top = int(cuts_list[i])
    bottom = int(cuts_list[i + 1])
    if bottom <= top:
      if on_progress:
        on_progress(i + 1, total_slices)
      continue

    slice_img = image.crop((0, top, width, bottom)).convert("RGB")
    results = ocr_backend.readtext(slice_img, lang=lang)
    for poly, text, prob in results:
      if float(prob) < float(conf):
        continue
      x1, y1, x2, y2 = _rect_from_poly(poly)
      cx = (x1 + x2) / 2.0
      cy = (y1 + y2) / 2.0
      global_y = cy + float(top)

      bubble = _bubble_for_point(bubbles, cx, global_y)
      if not bubble:
        bubble = _bubble_for_rect(bubbles, float(x1), float(y1) + float(top), float(x2), float(y2) + float(top))
      if not bubble:
        continue

      bubble_id = str(bubble.get("id") or "")
      if not bubble_id:
        continue
      if bubble_id not in tokens_by_id:
        tokens_by_id[bubble_id] = []

      tokens_by_id[bubble_id].append(
        OcrToken(text=str(text or ""), x_mid=float(cx), y_mid=float(global_y), h=max(1.0, float(y2 - y1)))
      )

    if on_progress:
      on_progress(i + 1, total_slices)

  texts: Dict[str, str] = {}
  for b in bubbles:
    bubble_id = str(b.get("id") or "")
    if not bubble_id:
      continue
    texts[bubble_id] = tokens_to_text(tokens_by_id.get(bubble_id) or [])
  return texts


def ocr_texts_per_bubble(
  image: Image.Image,
  bubbles: Sequence[dict],
  ocr_backend,
  conf: float,
  lang: Optional[str],
  on_progress: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, str]:
  texts: Dict[str, str] = {}
  width, height = image.size
  total = len(bubbles)
  for idx, b in enumerate(bubbles):
    bubble_id = b.get("id") or ""
    bb = b.get("bbox") or {}
    try:
      left_f = float(bb.get("left", 0.0))
      top_f = float(bb.get("top", 0.0))
      right_f = float(bb.get("right", left_f))
      bottom_f = float(bb.get("bottom", top_f))
    except Exception:
      left_f, top_f, right_f, bottom_f = 0.0, 0.0, 0.0, 0.0

    left = max(0, min(int(width), int(math.floor(left_f))))
    top = max(0, min(int(height), int(math.floor(top_f))))
    right = max(0, min(int(width), int(math.ceil(right_f))))
    bottom = max(0, min(int(height), int(math.ceil(bottom_f))))

    if right <= left:
      right = min(int(width), left + 1)
    if bottom <= top:
      bottom = min(int(height), top + 1)

    crop = image.crop((left, top, right, bottom)).convert("RGB")
    results = ocr_backend.readtext(crop, lang=lang)

    tokens: List[OcrToken] = []
    for poly, text, prob in results:
      if float(prob) < float(conf):
        continue
      x1, y1, x2, y2 = _rect_from_poly(poly)
      x_mid = (x1 + x2) / 2.0
      y_mid = (y1 + y2) / 2.0
      tokens.append(OcrToken(text=text, x_mid=x_mid, y_mid=y_mid, h=max(1.0, y2 - y1)))

    texts[bubble_id] = tokens_to_text(tokens)
    if on_progress:
      on_progress(idx + 1, total)

  return texts


def inpaint_with_bubbles_sliced(
  image: Image.Image,
  bubbles: Sequence[dict],
  ocr_backend,
  ocr_conf: float,
  inpainter,
  cfg: InpaintConfig,
  cost_map: Optional[np.ndarray] = None,
  cuts: Optional[Sequence[int]] = None,
  out_ocr_stats: Optional[Dict[str, dict]] = None,
  on_progress: Optional[Callable[[int, int], None]] = None,
  secondary_ocr_backend=None,
  secondary_ocr_conf: Optional[float] = None,
) -> Tuple[Image.Image, np.ndarray]:
  width, height = image.size
  slicer = SmartSlicer(
    edge_weight=float(cfg.edge_weight),
    variance_weight=float(cfg.variance_weight),
    gradient_weight=float(cfg.gradient_weight),
    white_space_weight=float(cfg.white_space_weight),
    distance_penalty=float(cfg.distance_penalty),
    forbidden_cost=1e9,
  )

  boxes_for_cost = []
  for b in bubbles:
    bb = b.get("bbox") or {}
    boxes_for_cost.append(
      (
        float(bb.get("left")),
        float(bb.get("top")),
        float(bb.get("right")),
        float(bb.get("bottom")),
        1.0,
        0,
      )
    )

  if cost_map is None or len(cost_map) != height:
    cost_map = slicer.compute_cost_map(image, boxes_for_cost, margin=int(cfg.margin))

  if cuts is None:
    cuts = compute_cuts(height, slicer, cost_map, min_h=int(cfg.min_h), max_h=int(cfg.max_h))
  cuts_list = [int(c) for c in cuts] if cuts else [0, height]
  if len(cuts_list) < 2:
    cuts_list = [0, height]

  token_yh: Dict[str, List[Tuple[float, float]]] = {}
  if out_ocr_stats is not None:
    # Initialize with bubble areas so missing OCR stays explicit.
    for b in bubbles:
      bid = str(b.get("id") or "")
      if not bid:
        continue
      bb = b.get("bbox") or {}
      left = float(bb.get("left", 0.0))
      top = float(bb.get("top", 0.0))
      right = float(bb.get("right", left))
      bottom = float(bb.get("bottom", top))
      area = max(1.0, (right - left) * (bottom - top))
      out_ocr_stats.setdefault(
        bid,
        {
          "mask_area": 0,
          "bubble_area": float(area),
          "ocr_bbox": None,
          "n_polys": 0,
        },
      )

  processed_slices: List[Image.Image] = []

  total_slices = max(1, len(cuts_list) - 1)
  for i in range(len(cuts_list) - 1):
    top = cuts_list[i]
    bottom = cuts_list[i + 1]
    slice_img = image.crop((0, top, width, bottom)).convert("RGB")
    slice_h = bottom - top

    mask = np.zeros((slice_h, width), dtype=np.uint8)
    bubble_region = np.zeros((slice_h, width), dtype=np.uint8)
    bubble_spans: List[Tuple[str, int, int, int, int]] = []
    for b in bubbles:
      bb = b.get("bbox") or {}
      b_left = int(float(bb.get("left")))
      b_right = int(float(bb.get("right")))
      b_top = int(float(bb.get("top")))
      b_bottom = int(float(bb.get("bottom")))

      if b_bottom <= int(top) or b_top >= int(bottom):
        continue
      rel_top = max(0, b_top - int(top))
      rel_bottom = min(slice_h, b_bottom - int(top))
      if rel_bottom <= rel_top:
        continue
      left = max(0, b_left)
      right = min(width, b_right)
      if right <= left:
        continue
      bubble_region[rel_top:rel_bottom, left:right] = 255
      if out_ocr_stats is not None:
        bid = str(b.get("id") or "")
        if bid:
          bubble_spans.append((bid, left, rel_top, right, rel_bottom))

    results = ocr_backend.readtext(slice_img, lang=None)
    for poly, _text, prob in results:
      if float(prob) < float(ocr_conf):
        continue
      x1, y1, x2, y2 = _rect_from_poly(poly)
      cx = (x1 + x2) / 2.0
      cy = (y1 + y2) / 2.0
      global_y = cy + float(top)
      bubble = _bubble_for_point(bubbles, cx, global_y)
      if not bubble:
        bubble = _bubble_for_rect(bubbles, float(x1), float(y1) + float(top), float(x2), float(y2) + float(top))
      if not bubble:
        continue
      if out_ocr_stats is not None:
        bid = str(bubble.get("id") or "")
        if bid:
          entry = out_ocr_stats.get(bid)
          if entry is None:
            bb = bubble.get("bbox") or {}
            b_left = float(bb.get("left", 0.0))
            b_top = float(bb.get("top", 0.0))
            b_right = float(bb.get("right", b_left))
            b_bottom = float(bb.get("bottom", b_top))
            area = max(1.0, (b_right - b_left) * (b_bottom - b_top))
            entry = {"mask_area": 0, "bubble_area": float(area), "ocr_bbox": None, "n_polys": 0}
            out_ocr_stats[bid] = entry
          entry["n_polys"] = int(entry.get("n_polys", 0)) + 1
          token_yh.setdefault(bid, []).append((float(global_y), max(1.0, float(y2 - y1))))
      pts = np.asarray(poly, dtype=np.int32)
      cv2.fillPoly(mask, [pts], 255)

    if secondary_ocr_backend is not None and secondary_ocr_conf is not None:
      results2 = secondary_ocr_backend.readtext(slice_img, lang=None)
      for poly, _text, prob in results2:
        if float(prob) < float(secondary_ocr_conf):
          continue
        x1, y1, x2, y2 = _rect_from_poly(poly)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        global_y = cy + float(top)
        bubble = _bubble_for_point(bubbles, cx, global_y)
        if not bubble:
          bubble = _bubble_for_rect(bubbles, float(x1), float(y1) + float(top), float(x2), float(y2) + float(top))
        if not bubble:
          continue
        if out_ocr_stats is not None:
          bid = str(bubble.get("id") or "")
          if bid:
            entry = out_ocr_stats.get(bid)
            if entry is None:
              bb = bubble.get("bbox") or {}
              b_left = float(bb.get("left", 0.0))
              b_top = float(bb.get("top", 0.0))
              b_right = float(bb.get("right", b_left))
              b_bottom = float(bb.get("bottom", b_top))
              area = max(1.0, (b_right - b_left) * (b_bottom - b_top))
              entry = {"mask_area": 0, "bubble_area": float(area), "ocr_bbox": None, "n_polys": 0}
              out_ocr_stats[bid] = entry
            entry["n_polys"] = int(entry.get("n_polys", 0)) + 1
            token_yh.setdefault(bid, []).append((float(global_y), max(1.0, float(y2 - y1))))
        pts = np.asarray(poly, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)

    # Keep the mask strictly within bubble rectangles (avoid touching outside regions).
    if np.any(bubble_region):
      mask = cv2.bitwise_and(mask, bubble_region)

    if out_ocr_stats is not None and bubble_spans:
      for bid, left, rel_top, right, rel_bottom in bubble_spans:
        sub = mask[rel_top:rel_bottom, left:right]
        if sub.size == 0 or not np.any(sub):
          continue
        area = int(np.count_nonzero(sub))
        entry = out_ocr_stats.get(bid)
        if entry is None:
          entry = {"mask_area": 0, "bubble_area": 1.0, "ocr_bbox": None, "n_polys": 0}
          out_ocr_stats[bid] = entry
        entry["mask_area"] = int(entry.get("mask_area", 0)) + area

        ys, xs = np.where(sub > 0)
        if xs.size == 0:
          continue
        x1b = int(left + int(xs.min()))
        x2b = int(left + int(xs.max()) + 1)
        y1b = int(int(top) + int(rel_top) + int(ys.min()))
        y2b = int(int(top) + int(rel_top) + int(ys.max()) + 1)

        bb = entry.get("ocr_bbox")
        if not isinstance(bb, dict):
          entry["ocr_bbox"] = {"left": float(x1b), "top": float(y1b), "right": float(x2b), "bottom": float(y2b)}
        else:
          bb["left"] = float(min(float(bb.get("left", x1b)), float(x1b)))
          bb["top"] = float(min(float(bb.get("top", y1b)), float(y1b)))
          bb["right"] = float(max(float(bb.get("right", x2b)), float(x2b)))
          bb["bottom"] = float(max(float(bb.get("bottom", y2b)), float(y2b)))

    mask_pil = Image.fromarray(mask).convert("L")
    inpainted = inpainter(slice_img, mask_pil)
    processed_slices.append(inpainted)

    if on_progress:
      on_progress(i + 1, total_slices)

  merged = Image.new("RGB", (width, height))
  y = 0
  for s in processed_slices:
    merged.paste(s, (0, y))
    y += s.height

  if out_ocr_stats is not None:
    for bid, entry in out_ocr_stats.items():
      try:
        bubble_area = float(entry.get("bubble_area", 1.0))
      except Exception:
        bubble_area = 1.0
      try:
        mask_area = float(entry.get("mask_area", 0.0))
      except Exception:
        mask_area = 0.0
      cov = float(mask_area / bubble_area) if bubble_area > 0 else 0.0
      entry["cov"] = float(max(0.0, min(1.0, cov)))

      bb = entry.get("ocr_bbox")
      if isinstance(bb, dict):
        try:
          bbox_h = float(bb.get("bottom", 0.0) - bb.get("top", 0.0))
        except Exception:
          bbox_h = 0.0
        lines = _estimate_line_count_from_tokens(token_yh.get(str(bid)) or [])
        entry["line_count"] = int(lines)
        if bbox_h > 0.0 and lines > 0:
          est_line_h = bbox_h / float(lines)
          entry["font_est_px"] = float(est_line_h / 1.2)
        else:
          entry["font_est_px"] = None
      else:
        entry["line_count"] = 0
        entry["font_est_px"] = None

  return merged, cost_map
