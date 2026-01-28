from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class StyleExtractConfig:
  conf: float = 0.4
  crop_pad: int = 2
  min_mask_pixels: int = 30
  min_glyph_pixels: int = 20
  min_bg_pixels: int = 50
  min_stroke_delta: float = 8.0
  min_stroke_px: float = 1.0


def _rect_from_poly(poly: Sequence[Sequence[float]]) -> Tuple[float, float, float, float]:
  xs = [float(p[0]) for p in poly]
  ys = [float(p[1]) for p in poly]
  return min(xs), min(ys), max(xs), max(ys)


def _clip_box(left: float, top: float, right: float, bottom: float, width: int, height: int) -> Tuple[int, int, int, int]:
  l = max(0, min(int(width), int(left)))
  t = max(0, min(int(height), int(top)))
  r = max(0, min(int(width), int(right)))
  b = max(0, min(int(height), int(bottom)))
  if r <= l:
    r = min(int(width), l + 1)
  if b <= t:
    b = min(int(height), t + 1)
  return l, t, r, b


def _median_color(pixels: np.ndarray) -> np.ndarray:
  if pixels.size == 0:
    return np.array([0, 0, 0], dtype=np.float32)
  return np.median(pixels.reshape(-1, 3), axis=0)


def _lab_color(rgb: np.ndarray) -> np.ndarray:
  arr = np.asarray([[rgb]], dtype=np.uint8)
  lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB).astype(np.float32)
  return lab[0, 0]


def _color_distance_lab(lab_img: np.ndarray, lab_ref: np.ndarray) -> np.ndarray:
  diff = lab_img - lab_ref.reshape(1, 1, 3)
  return np.sqrt(np.sum(diff * diff, axis=2))


def _otsu_threshold(values: np.ndarray) -> Optional[float]:
  if values.size == 0:
    return None
  v = values.astype(np.float32)
  v_max = float(np.percentile(v, 98))
  if v_max <= 1e-3:
    return None
  v_norm = np.clip((v / v_max) * 255.0, 0, 255).astype(np.uint8)
  thr, _ = cv2.threshold(v_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
  return float(thr) / 255.0 * v_max


def _kmeans_1d_two(values: np.ndarray, max_iter: int = 12) -> Optional[Tuple[float, float, float]]:
  if values.size < 20:
    return None
  v = values.astype(np.float32)
  c1 = float(np.percentile(v, 30))
  c2 = float(np.percentile(v, 70))
  if abs(c2 - c1) < 1e-3:
    return None
  for _ in range(max_iter):
    d1 = np.abs(v - c1)
    d2 = np.abs(v - c2)
    mask = d1 <= d2
    if mask.all() or (~mask).all():
      break
    c1_new = float(v[mask].mean())
    c2_new = float(v[~mask].mean())
    if abs(c1_new - c1) < 0.01 and abs(c2_new - c2) < 0.01:
      break
    c1, c2 = c1_new, c2_new
  low, high = (c1, c2) if c1 < c2 else (c2, c1)
  if (high - low) < 0.75:
    return None
  return (low + high) / 2.0, low, high




def _extract_colors_from_mask(
  crop_rgb: np.ndarray,
  mask: np.ndarray,
  cfg: StyleExtractConfig,
) -> Dict[str, Dict[str, float]]:
  mask_bool = mask > 0
  if mask_bool.sum() < cfg.min_mask_pixels:
    return {}

  bg_pixels = crop_rgb[~mask_bool]
  if bg_pixels.size < cfg.min_bg_pixels * 3:
    bg_pixels = crop_rgb.reshape(-1, 3)
  bg_rgb = _median_color(bg_pixels)
  bg_lab = _lab_color(bg_rgb)

  crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
  lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
  dist_bg = _color_distance_lab(lab, bg_lab)

  dist_vals = dist_bg[mask_bool]
  thr = _otsu_threshold(dist_vals)
  if thr is None or thr < 3.0:
    glyph_mask = mask_bool
  else:
    glyph_mask = (dist_bg > thr) & mask_bool
    if glyph_mask.sum() < cfg.min_glyph_pixels or glyph_mask.sum() > mask_bool.sum() * 0.95:
      glyph_mask = mask_bool

  glyph_pixels = crop_rgb[glyph_mask]
  fill_rgb = _median_color(glyph_pixels)
  fill_lab = _lab_color(fill_rgb)

  glyph_u8 = (glyph_mask.astype(np.uint8) * 255)
  dist_tx = cv2.distanceTransform(glyph_u8, cv2.DIST_L2, 5)
  dist_tx_vals = dist_tx[glyph_mask]
  if dist_tx_vals.size >= cfg.min_glyph_pixels and float(dist_tx_vals.max()) >= cfg.min_stroke_px:
    kmeans = _kmeans_1d_two(dist_tx_vals)
    if kmeans is not None:
      thr_dist, low, high = kmeans
      core = dist_tx >= thr_dist
      ring = glyph_mask & (~core)
      if ring.sum() >= cfg.min_mask_pixels and core.sum() >= cfg.min_mask_pixels:
        core_rgb = _median_color(crop_rgb[core])
        ring_rgb = _median_color(crop_rgb[ring])
        delta = np.linalg.norm(_lab_color(core_rgb) - _lab_color(ring_rgb))
        if delta >= cfg.min_stroke_delta:
          fill_rgb = core_rgb
          fill_lab = _lab_color(fill_rgb)

  style = {
    "fill": {"r": float(fill_rgb[0]), "g": float(fill_rgb[1]), "b": float(fill_rgb[2])},
  }
  return style


def extract_bubble_styles(
  image: Image.Image,
  bubbles: Sequence[dict],
  ocr_backend,
  conf: float,
  lang: Optional[str],
  cfg: Optional[StyleExtractConfig] = None,
) -> Dict[str, Dict[str, Dict[str, float]]]:
  if cfg is None:
    cfg = StyleExtractConfig(conf=float(conf))

  width, height = image.size
  styles: Dict[str, Dict[str, Dict[str, float]]] = {}

  for b in bubbles:
    bubble_id = b.get("id") or ""
    if not bubble_id:
      continue
    bb = b.get("bbox") or {}
    left_f = float(bb.get("left", 0.0))
    top_f = float(bb.get("top", 0.0))
    right_f = float(bb.get("right", left_f))
    bottom_f = float(bb.get("bottom", top_f))

    pad = int(cfg.crop_pad)
    left, top, right, bottom = _clip_box(left_f - pad, top_f - pad, right_f + pad, bottom_f + pad, width, height)
    if right <= left or bottom <= top:
      continue

    crop = image.crop((left, top, right, bottom)).convert("RGB")
    crop_rgb = np.array(crop)

    results = ocr_backend.readtext(crop, lang=lang)
    if not results:
      continue

    mask = np.zeros((crop_rgb.shape[0], crop_rgb.shape[1]), dtype=np.uint8)
    for poly, _text, prob in results:
      if float(prob) < float(cfg.conf):
        continue
      pts = np.asarray(poly, dtype=np.int32)
      if pts.size == 0:
        continue
      cv2.fillPoly(mask, [pts], 255)

    if mask.sum() < cfg.min_mask_pixels:
      continue

    style = _extract_colors_from_mask(crop_rgb, mask, cfg)
    if style:
      styles[str(bubble_id)] = style

  return styles
