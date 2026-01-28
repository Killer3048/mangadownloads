from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
  repo_root = Path(__file__).resolve().parents[1]
  sys.path.insert(0, str(repo_root))

  from tiper.autofit.fontfit import _hyphen_points_pyphen, compute_fit_map

  # Hyphenation points (RU) – should support common words.
  pts_poluchit = _hyphen_points_pyphen("получить", lang="ru")
  if 4 not in pts_poluchit:
    print(f"[FAIL] expected hyphen point 4 in 'получить', got: {pts_poluchit}", file=sys.stderr)
    return 1

  pts_massa = _hyphen_points_pyphen("масса", lang="ru")
  if 3 not in pts_massa:
    print(f"[FAIL] expected hyphen point 3 in 'масса', got: {pts_massa}", file=sys.stderr)
    return 1

  if _hyphen_points_pyphen("из-за", lang="ru"):
    print("[FAIL] expected no hyphenation points for 'из-за' (already hyphenated)", file=sys.stderr)
    return 1

  # Smoke-check full fit pipeline doesn't crash and produces layout_text.
  bubbles = [{"id": "B1", "bbox": {"left": 0, "top": 0, "right": 300, "bottom": 200}}]
  translations = {"B1": "Я хотела дать тебе шанс получить опыт."}
  bubble_geometry = {"items": {"B1": {"source": "bbox_fallback", "core_bbox": {"left": 0, "top": 0, "right": 300, "bottom": 200}}}}
  ocr_stats = {"items": {"B1": {"ocr_bbox": {"left": 50, "top": 50, "right": 250, "bottom": 150}, "cov": 0.2}}}
  config = {
    "min_pt": 16,
    "max_pt": 34,
    "pt_step": 2,
    "dpi": 72,
    "leading_factor": 0.25,
    "cap_factor": 1.25,
    "komponovka": {"enabled": True, "mode": "force", "lang": "ru", "apply_to_translation": False},
  }

  out = compute_fit_map(
    translations=translations,
    bubbles=bubbles,
    bubble_geometry=bubble_geometry,
    ocr_mask_stats=ocr_stats,
    config=config,
    bubble_classes=None,
    image_dpi=72,
  )
  lt = (((out.get("B1") or {}) if isinstance(out, dict) else {}) or {}).get("layout_text")
  if not isinstance(lt, str) or not lt.strip():
    print(f"[FAIL] expected non-empty layout_text, got: {lt!r}", file=sys.stderr)
    return 1

  print("[OK] komponovka_smoke")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
