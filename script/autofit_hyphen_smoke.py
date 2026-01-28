from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
  repo_root = Path(__file__).resolve().parents[1]
  sys.path.insert(0, str(repo_root))

  try:
    import pyphen  # noqa: F401
  except Exception:
    print("[SKIP] pyphen is not installed; hyphen-preservation regression is not applicable.")
    return 0

  from PIL import ImageFont

  from tiper.autofit.fontfit import _compose_text_output

  font = ImageFont.load_default()
  cfg = {"komponovka": {"enabled": True, "mode": "force", "lang": "ru"}}

  samples = [
    "бел-о-кан",
    "из-за",
    "супер-длинный",
  ]

  for text in samples:
    out, applied = _compose_text_output(
      text,
      font,
      max_width=40,
      max_height=2000,
      font_pt=20,
      leading_factor=0.25,
      config=cfg,
    )
    if not applied:
      print(f"[FAIL] expected komponovka to be applied for sample {text!r}", file=sys.stderr)
      return 1
    if out.replace("\n", "") != text:
      print(f"[FAIL] hyphens lost: in={text!r} out={out!r}", file=sys.stderr)
      return 1

  print("[OK] autofit_hyphen_smoke")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())

