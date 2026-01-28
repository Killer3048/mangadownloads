#!/usr/bin/env python3
"""
Pre-download PaddleOCR models into PaddleOCR's *default* cache directories.

Why:
- PaddleOCR can lazily download extra assets/models on first run.
- This script forces all models needed by our pipeline to be downloaded upfront.

Run this inside the OCR worker environment (the one where you installed paddleocr + paddlepaddle).

Examples:
  python script/download_models.py --langs korean,en
  python script/download_models.py --langs ko,en,ja,zh,zh_tw
  python script/download_models.py --langs korean,en --use-gpu 0
  python script/download_models.py --langs korean,en --device cpu
"""

from __future__ import annotations

import argparse
import inspect
import os
import sys
from pathlib import Path
from typing import Dict


_LANG_ALIASES: Dict[str, str] = {
  "ko": "korean",
  "kr": "korean",
  "korean": "korean",
  "en": "en",
  "eng": "en",
  "ja": "japan",
  "jp": "japan",
  "japanese": "japan",
  "japan": "japan",
  "zh": "ch",
  "cn": "ch",
  "zh_cn": "ch",
  "zh-hans": "ch",
  "ch": "ch",
  "zh_tw": "chinese_cht",
  "zh-hant": "chinese_cht",
  "chinese_cht": "chinese_cht",
}


def _normalize_lang(raw: str) -> str:
  key = (raw or "").strip().lower().replace("-", "_")
  return _LANG_ALIASES.get(key, raw.strip())


def _print_cache_hints() -> None:
  home = Path.home()
  candidates = [
    home / ".paddleocr",
    home / ".cache" / "paddleocr",
    home / ".cache" / ".paddleocr",
  ]
  existing = [p for p in candidates if p.exists()]
  if existing:
    print("\n[+] Detected PaddleOCR cache dirs:")
    for p in existing:
      print(f"    - {p}")
  else:
    print("\n[i] Cache dir location depends on PaddleOCR version.")
    print("    Common paths:")
    for p in candidates:
      print(f"    - {p}")


def _force_download(lang: str, use_textline_orientation: bool, device: str) -> None:
  try:
    from paddleocr import PaddleOCR  # type: ignore
  except Exception as e:
    raise RuntimeError("paddleocr is not installed in this environment") from e

  try:
    import paddle  # type: ignore  # noqa: F401
  except Exception as e:
    raise RuntimeError(
      "paddlepaddle is not installed. Install `paddlepaddle` (CPU) or `paddlepaddle-gpu` (CUDA) in this environment."
    ) from e

  print(f"[+] Initializing PaddleOCR(lang={lang}, textline_orientation={use_textline_orientation}, device={device})")

  # PaddleOCR API differences:
  # - 3.x: device is controlled via `device=...` (in **kwargs), and `use_textline_orientation`.
  # - 2.x: device is controlled via `use_gpu=...`, and `use_angle_cls`.
  use_gpu = str(device).lower().startswith("gpu")
  sig = inspect.signature(PaddleOCR.__init__)
  supports_textline_orientation = "use_textline_orientation" in sig.parameters

  ocr = None
  if supports_textline_orientation:
    kwargs = {"lang": lang, "use_textline_orientation": bool(use_textline_orientation), "device": device}
    if "use_doc_orientation_classify" in sig.parameters:
      kwargs["use_doc_orientation_classify"] = False
    if "use_doc_unwarping" in sig.parameters:
      kwargs["use_doc_unwarping"] = False
    try:
      ocr = PaddleOCR(**kwargs)
    except ValueError as e:
      # PaddleOCR 3.x rejects unknown common kwargs (e.g. in very old/odd builds).
      if str(e).startswith("Unknown argument:"):
        kwargs.pop("device", None)
        ocr = PaddleOCR(**kwargs)
      else:
        raise
  else:
    kwargs = {"lang": lang, "use_angle_cls": bool(use_textline_orientation), "use_gpu": bool(use_gpu)}
    ocr = PaddleOCR(**kwargs)

  # Trigger any lazy downloads by running a tiny inference.
  try:
    import numpy as np

    dummy = np.full((256, 256, 3), 255, dtype=np.uint8)
    if hasattr(ocr, "predict"):
      try:
        _ = ocr.predict(dummy, use_textline_orientation=bool(use_textline_orientation))
      except TypeError:
        _ = ocr.ocr(dummy, cls=bool(use_textline_orientation))
    else:
      _ = ocr.ocr(dummy, cls=bool(use_textline_orientation))
  except Exception as e:
    # Some environments (GPU builds without CUDA, etc.) may fail to run inference,
    # but the model downloads usually already happened during initialization.
    print(f"[!] Inference check failed for lang={lang}: {e}")


def main(argv: list[str]) -> int:
  parser = argparse.ArgumentParser(description="Download PaddleOCR models to default cache dirs.")
  parser.add_argument(
    "--langs",
    default="korean,en",
    help="Comma-separated languages. Supports aliases: ko,en,ja,zh,zh_tw (default: korean,en).",
  )
  parser.add_argument(
    "--disable-source-check",
    action="store_true",
    help="Skip PaddleOCR connectivity check to model hosters (sets DISABLE_MODEL_SOURCE_CHECK=True).",
  )
  parser.add_argument(
    "--use-textline-orientation",
    type=int,
    default=1,
    help="Download textline orientation model too (PaddleOCR 3.x; default: 1).",
  )
  parser.add_argument(
    "--use-angle-cls",
    type=int,
    default=None,
    help="Deprecated alias for --use-textline-orientation (kept for older PaddleOCR).",
  )
  parser.add_argument(
    "--device",
    default=None,
    help="Device hint for PaddleOCR 3.x (e.g. cpu, gpu, gpu:0). If omitted, --use-gpu is used.",
  )
  parser.add_argument(
    "--use-gpu",
    type=int,
    default=0,
    help="Legacy flag for older PaddleOCR (0=CPU, 1=GPU). For PaddleOCR 3.x this maps to --device.",
  )
  args = parser.parse_args(argv)

  if args.disable_source_check:
    os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"

  langs = [_normalize_lang(x) for x in args.langs.split(",") if x.strip()]
  if not langs:
    print("[ERROR] No languages provided.", file=sys.stderr)
    return 2

  # Deduplicate while keeping order.
  seen = set()
  langs_unique = []
  for l in langs:
    if l not in seen:
      langs_unique.append(l)
      seen.add(l)

  print("[+] Downloading PaddleOCR models (default cache paths).")
  device = (args.device or ("gpu" if bool(args.use_gpu) else "cpu")).strip()
  for lang in langs_unique:
    use_orientation = bool(args.use_textline_orientation)
    if args.use_angle_cls is not None:
      use_orientation = bool(args.use_angle_cls)
    _force_download(lang, use_textline_orientation=use_orientation, device=device)

  _print_cache_hints()
  print("\n[+] Done.")
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv[1:]))
