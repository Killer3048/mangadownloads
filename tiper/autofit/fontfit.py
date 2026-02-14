from __future__ import annotations

import math
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

from tiper.server.paths import win_path_to_wsl
class AtomicToken:
  def __init__(self, text: str, can_break: bool, penalty: float, join_char: str, join_break: str, join_cont: str):
    self.text = text
    self.can_break = can_break
    self.penalty = penalty
    self.join_break = join_break # string to append if we break AFTER this token (e.g. "-")
    self.join_cont = join_cont   # string to append if we continue (e.g. " ")

_EXPLICIT_HYPHEN_RUN_RE = re.compile(r"([-\u2010\u2011\u2012\u2013\u2212]+)")


def _split_word_preserving_hyphens(word: str) -> List[Tuple[str, Optional[str]]]:
  """
  Split a whitespace-delimited token into (segment, join_after) pairs while preserving
  explicit hyphen-like characters as literal characters (not hyphenation points).

  Example: "бел-о-кан" -> [("бел", "-"), ("о", "-"), ("кан", None)]
  """
  if not word:
    return []

  parts = re.split(_EXPLICIT_HYPHEN_RUN_RE, word)
  if len(parts) == 1:
    return [(word, None)]

  segments: List[Tuple[str, Optional[str]]] = []
  prefix = ""

  for idx, part in enumerate(parts):
    if part == "":
      continue

    if _EXPLICIT_HYPHEN_RUN_RE.fullmatch(part):
      next_seg = parts[idx + 1] if idx + 1 < len(parts) else ""
      if segments:
        if next_seg:
          text, join = segments[-1]
          segments[-1] = (text, (join or "") + part)
        else:
          # Trailing hyphens must be part of the segment text (no next segment to join to).
          text, join = segments[-1]
          segments[-1] = (text + part, join)
      else:
        if next_seg:
          # Leading hyphens: attach to the first segment.
          prefix += part
        else:
          # Word is only hyphens.
          segments.append((part, None))
      continue

    seg_text = prefix + part
    prefix = ""
    segments.append((seg_text, None))

  if prefix:
    if segments:
      text, join = segments[-1]
      segments[-1] = (text + prefix, join)
    else:
      segments.append((prefix, None))

  return segments

def _tokenize_atomic(text: str, lang: str, *, allow_auto_hyphen: bool = True) -> List[AtomicToken]:
  """
  Splits text into atomic units (syllables or words).
  """
  raw_words = _tokenize_ws(text)
  out: List[AtomicToken] = []
  
  dic = None
  if allow_auto_hyphen:
    try:
      import pyphen

      py_lang = "ru_RU" if lang == "ru" else ("en_US" if lang == "en" else None)
      if py_lang:
        dic = pyphen.Pyphen(lang=py_lang)
    except Exception:
      dic = None

  for i, word in enumerate(raw_words):
      is_last_word = (i == len(raw_words) - 1)
      
      # Naive cleanup for syllables (strip punct) could be better,
      # but pyphen handles punct okay usually. 
      # Better: strip punct, split core, reattach punct to last syllable?
      # For now, let's just try feeding word.
      
      segments = _split_word_preserving_hyphens(word)
      if not segments:
        continue

      # If a token already has explicit hyphen(s), avoid additional pyphen
      # splitting inside the same token: this prevents awkward forms like
      # "С-се-\nстра".
      has_explicit_hyphen = len(segments) > 1
      explicit_tail_split = False
      if has_explicit_hyphen and len(segments) == 2:
        head_norm = _tok_norm(segments[0][0])
        tail_norm = _tok_norm(segments[1][0])
        # Allow split in the tail for stutter-like forms ("С-сестра"),
        # otherwise we lose too much font size in tiny bubbles.
        if len(head_norm) <= 2 and len(tail_norm) >= 6:
          explicit_tail_split = True

      for seg_idx, (seg_text, join_after) in enumerate(segments):
        allow_seg_pyphen = (not has_explicit_hyphen) or (explicit_tail_split and seg_idx == 1 and not join_after)
        # Split punctuation from the core token before pyphenation.
        lead = ""
        core = seg_text
        trail = ""
        m = re.match(r"^([\\(\\[«“\"']*)(.*?)([\\)\\]»”\"'.,:;!?…]*)$", seg_text)
        if m:
          lead = m.group(1) or ""
          core = m.group(2) or ""
          trail = m.group(3) or ""

        syllables: List[str] = []
        if dic and allow_seg_pyphen and len(core) > 6:
          # pyphen inserter returns "syl-la-ble"
          inserted = dic.inserted(core)
          if "-" in inserted:
            syllables = [s for s in inserted.split("-") if s]

        if not syllables:
          base = core if core else seg_text
          syllables = [base]

        if lead:
          syllables[0] = lead + syllables[0]
        if trail:
          syllables[-1] = syllables[-1] + trail

        for j, syl in enumerate(syllables):
          is_last_syl = (j == len(syllables) - 1)

          if is_last_syl:
            if join_after:
              # Explicit hyphen in the original text: preserve it both on continue and on break.
              out.append(AtomicToken(syl, True, 6.0, join_char="", join_break=join_after, join_cont=join_after))
            else:
              # End of word.
              # If break: implies newline (no extra char).
              # If continue: need SPACE (unless last word of para).
              join_cont = " " if not is_last_word else ""
              out.append(AtomicToken(syl, True, 0.0, join_char="", join_break="", join_cont=join_cont))
          else:
            # Mid-word syllable.
            # If break: add HYPHEN. If continue: add nothing (join parts).
            # Penalize very short fragments much harder to avoid unnatural hyphenation.
            rem = "".join(syllables[j + 1 :])
            pen = 2.0  # Base penalty: hyphens are acceptable for shape
            if len(syl) <= 2 or len(rem) <= 2:
              pen += 1.5   # Even short fragments are OK for shape filling
            elif len(syl) <= 3 or len(rem) <= 3:
              pen += 0.8
            out.append(AtomicToken(syl, True, pen, join_char="", join_break="-", join_cont=""))
              
  return out


def _safe_float(v: Any, default: float = 0.0) -> float:
  try:
    return float(v)
  except Exception:
    return float(default)


def _bbox_tuple(bb: Optional[dict]) -> Optional[Tuple[float, float, float, float]]:
  if not isinstance(bb, dict):
    return None
  left = _safe_float(bb.get("left", 0.0))
  top = _safe_float(bb.get("top", 0.0))
  right = _safe_float(bb.get("right", left))
  bottom = _safe_float(bb.get("bottom", top))
  if right <= left or bottom <= top:
    return None
  return left, top, right, bottom


def _bbox_dict(t: Tuple[float, float, float, float]) -> Dict[str, float]:
  l, t0, r, b = t
  return {"left": float(l), "top": float(t0), "right": float(r), "bottom": float(b)}


def _expand_bbox_about_center(bbox: Tuple[float, float, float, float], scale: float) -> Tuple[float, float, float, float]:
  x1, y1, x2, y2 = bbox
  cx = (x1 + x2) / 2.0
  cy = (y1 + y2) / 2.0
  bw = max(1.0, float(x2 - x1))
  bh = max(1.0, float(y2 - y1))
  nw = bw * float(scale)
  nh = bh * float(scale)
  return (cx - nw / 2.0, cy - nh / 2.0, cx + nw / 2.0, cy + nh / 2.0)


def _clamp_bbox(bbox: Tuple[float, float, float, float], clamp: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
  x1, y1, x2, y2 = bbox
  cx1, cy1, cx2, cy2 = clamp
  x1 = max(x1, cx1)
  y1 = max(y1, cy1)
  x2 = min(x2, cx2)
  y2 = min(y2, cy2)
  if x2 <= x1 or y2 <= y1:
    return clamp
  return (x1, y1, x2, y2)


def _blend_bbox(
  primary: Tuple[float, float, float, float],
  secondary: Tuple[float, float, float, float],
  weight_secondary: float,
) -> Tuple[float, float, float, float]:
  w = max(0.0, min(1.0, float(weight_secondary)))
  if w <= 0.0:
    return primary
  if w >= 1.0:
    return secondary
  x1 = (float(primary[0]) * (1.0 - w)) + (float(secondary[0]) * w)
  y1 = (float(primary[1]) * (1.0 - w)) + (float(secondary[1]) * w)
  x2 = (float(primary[2]) * (1.0 - w)) + (float(secondary[2]) * w)
  y2 = (float(primary[3]) * (1.0 - w)) + (float(secondary[3]) * w)
  if x2 <= x1 or y2 <= y1:
    return primary
  return (x1, y1, x2, y2)


def _inset_bbox(bbox: Tuple[float, float, float, float], inset_x: float, inset_y: float) -> Tuple[float, float, float, float]:
  x1, y1, x2, y2 = bbox
  w = float(x2 - x1)
  h = float(y2 - y1)
  if w <= 1.0 or h <= 1.0:
    return bbox

  dx = max(0.0, float(inset_x))
  dy = max(0.0, float(inset_y))
  # Keep at least a minimal usable area.
  max_dx = max(0.0, (w - 2.0) / 2.0)
  max_dy = max(0.0, (h - 2.0) / 2.0)
  dx = min(dx, max_dx)
  dy = min(dy, max_dy)

  nx1 = x1 + dx
  ny1 = y1 + dy
  nx2 = x2 - dx
  ny2 = y2 - dy
  if nx2 <= nx1 or ny2 <= ny1:
    return bbox
  return (nx1, ny1, nx2, ny2)


def _relax_fallback_clamp_for_short_text(
  clamp_bbox: Tuple[float, float, float, float],
  *,
  core_bbox: Tuple[float, float, float, float],
  geom_item: Dict[str, Any],
  ocr_item: Dict[str, Any],
  text: str,
) -> Tuple[float, float, float, float]:
  src = str((geom_item or {}).get("source") or "").lower()
  fallback = (not src) or src.startswith("bbox_fallback")
  if not fallback:
    return clamp_bbox

  words = len(_tokenize_ws(text))
  try:
    lc = int((ocr_item or {}).get("line_count") or 0)
  except Exception:
    lc = 0

  # For short lines in bbox-fallback mode, comfort clamp is often too narrow.
  # Slightly expand it back toward core bbox.
  relax = 1.0
  if lc <= 1:
    relax = 1.14
  elif words <= 4:
    relax = 1.10
  elif words <= 7:
    relax = 1.06
  if relax <= 1.0:
    return clamp_bbox

  return _clamp_bbox(_expand_bbox_about_center(clamp_bbox, scale=float(relax)), core_bbox)


def _comfort_clamp_bbox(
  core_bbox: Tuple[float, float, float, float],
  *,
  cov: Optional[float],
  ocr_item: Dict[str, Any],
  geom_item: Dict[str, Any],
  config: Dict[str, Any],
) -> Tuple[float, float, float, float]:
  """
  "Comfort" shrink of the usable region for fitting.

  Problem: current autofit often produces text that visually touches the bubble contour.
  Fix: shrink the allowed region inside the bubble (adaptive) so the chosen font size
  ends up ~a few pt smaller and leaves a pleasant margin.
  """
  ccfg = config.get("comfort") if isinstance(config, dict) else None
  if ccfg is True:
    ccfg = {"enabled": True}
  if not isinstance(ccfg, dict):
    ccfg = {}

  enabled = bool(ccfg.get("enabled", True))
  if not enabled:
    return core_bbox

  base_frac = _safe_float(ccfg.get("base_frac", 0.07), 0.07)
  min_frac = _safe_float(ccfg.get("min_frac", 0.03), 0.03)
  max_frac = _safe_float(ccfg.get("max_frac", 0.14), 0.14)
  # Extra shrink when we don't have a true contour-based core bbox (bbox fallback).
  # Keep this adaptive: fixed heavy bonus makes fonts too small on many real pages.
  fallback_bonus = _safe_float(ccfg.get("fallback_bonus", 0.06), 0.06)
  min_px = int(ccfg.get("min_px", 3))
  max_px = int(ccfg.get("max_px", 40))

  x1, y1, x2, y2 = core_bbox
  w = max(1.0, float(x2 - x1))
  h = max(1.0, float(y2 - y1))
  min_side = min(w, h)

  frac = float(base_frac)

  # If OCR text occupies a large fraction of the bubble, don't shrink much.
  cov_f = float(cov) if cov is not None else None
  if cov_f is not None:
    if cov_f < 0.08:
      frac += 0.06
    elif cov_f < 0.18:
      frac += 0.04
    elif cov_f < 0.30:
      frac += 0.02

  # More margin when we have no segmentation core (bbox fallback tends to include tails),
  # but scale it down when OCR coverage is decent.
  src = str(geom_item.get("source") or "").lower()
  if (not src) or src.startswith("bbox_fallback"):
    fb = float(fallback_bonus)
    if cov_f is not None:
      if cov_f >= 0.30:
        fb *= 0.20
      elif cov_f >= 0.20:
        fb *= 0.35
      elif cov_f >= 0.12:
        fb *= 0.55
      else:
        fb *= 0.80
    frac += fb

  # When OCR takes significant area, keep more usable box to avoid unnecessary downscaling.
  if cov_f is not None:
    if cov_f >= 0.18:
      frac -= 0.02
    if cov_f >= 0.28:
      frac -= 0.01

  # If OCR already produced many lines, shrinking too much tends to over-reduce font size.
  try:
    line_count = int(ocr_item.get("line_count") or 0)
  except Exception:
    line_count = 0
  if line_count >= 4:
    frac -= 0.02

  # Small bubbles: keep the shrink milder.
  if min_side < 120:
    frac *= 0.55
  elif min_side < 200:
    frac *= 0.75

  frac = max(float(min_frac), min(float(max_frac), float(frac)))

  dx = (w * frac) / 2.0
  dy = (h * frac) / 2.0
  dx = max(float(min_px), min(float(max_px), float(dx)))
  dy = max(float(min_px), min(float(max_px), float(dy)))

  inner = _inset_bbox(core_bbox, inset_x=dx, inset_y=dy)
  return inner if inner != core_bbox else core_bbox


def _text_width(text: str, font: ImageFont.FreeTypeFont) -> float:
  if hasattr(font, "getlength"):
    try:
      return float(font.getlength(text))
    except Exception:
      pass
  try:
    x1, y1, x2, y2 = font.getbbox(text)
    return float(x2 - x1)
  except Exception:
    return float(len(text)) * float(getattr(font, "size", 16))


def _wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> str:
  max_width = int(max(1, max_width))
  paras = (text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
  out_lines: List[str] = []
  lang = _detect_lang(text)

  def _split_long_token(token: str) -> List[str]:
    tok = str(token or "")
    if not tok:
      return [""]

    # Preserve surrounding quotes/brackets and trailing punctuation.
    m = re.match(r"^([\\(\\[«“]*)(.*?)([\\)\\]»”\\.,:;!?…]*)$", tok)
    lead = m.group(1) if m else ""
    core = m.group(2) if m else tok
    trail = m.group(3) if m else ""

    # If the core already fits, keep as-is.
    if _text_width(tok, font) <= max_width:
      return [tok]

    parts: List[str] = []
    remainder = core

    # Try syllable hyphenation when available (pyphen), otherwise a simple heuristic for RU.
    def _hyphen_points(word: str) -> List[int]:
      w = word or ""
      if len(w) < 6:
        return []
      # 1) pyphen (if installed)
      try:
        import pyphen  # type: ignore

        py_lang = "ru_RU" if lang == "ru" else ("en_US" if lang == "en" else None)
        if py_lang:
          dic = pyphen.Pyphen(lang=py_lang)
          inserted = dic.inserted(w)
          if "-" in inserted:
            # Convert parts to indices.
            idxs: List[int] = []
            pos = 0
            for chunk in inserted.split("-")[:-1]:
              pos += len(chunk)
              if 3 <= pos <= len(w) - 3:
                idxs.append(pos)
            return idxs
      except Exception:
        pass

      # 2) RU fallback: split after vowels (very rough).
      if lang == "ru":
        vowels = set("аеёиоуыэюяАЕЁИОУЫЭЮЯ")
        idxs = []
        for i in range(3, len(w) - 3):
          if w[i - 1] in vowels and w[i] not in vowels:
            idxs.append(i)
        return idxs
      return []

    while remainder and _text_width(remainder, font) > max_width:
      points = _hyphen_points(remainder)
      best_cut = None
      for cut in points:
        prefix = remainder[:cut]
        if _text_width(prefix + "-", font) <= max_width:
          best_cut = cut
      if best_cut is None:
        # Fallback: character split.
        cut = max(3, min(len(remainder) - 3, int(len(remainder) * 0.5)))
        while cut > 3 and _text_width(remainder[:cut] + "-", font) > max_width:
          cut -= 1
        best_cut = max(3, cut)

      parts.append(remainder[:best_cut] + "-")
      remainder = remainder[best_cut:]

    parts.append(remainder)

    # Re-attach punctuation only on the first/last chunks.
    if parts:
      parts[0] = lead + parts[0]
      parts[-1] = parts[-1] + trail
    return parts

  for para in paras:
    words = [w for w in (para or "").split(" ") if w != ""]
    if not words:
      out_lines.append("")
      continue
    line = ""
    prev_word = ""
    for word in words:
      cand = word if not line else (line + " " + word)
      fits = _text_width(cand, font) <= max_width

      if fits:
        prev_word = word
        line = cand
        continue

      # Check if we can break after prev_word (komponovka rules)
      can_break = True
      if prev_word and word:
        p = _tok_norm(prev_word)
        n = _tok_norm(word)
        # Check _RU_NO_BREAK_AFTER (hard ban)
        if p in _RU_NO_BREAK_AFTER:
          can_break = False
        # Check unbreakable pairs like "с ума", "в общем"
        _UNBREAKABLE_PAIRS_LOCAL = {
          # Existing pairs
          ("так", "же"), ("то", "есть"), ("то", "же"),
          ("потому", "что"), ("для", "того"), ("перед", "тем"),
          ("после", "того"), ("из-за", "того"), ("с", "ума"),
          ("как", "бы"), ("как", "будто"), ("что", "бы"),
          ("может", "быть"), ("должен", "быть"),
          ("в", "общем"), ("в", "целом"), ("в", "итоге"),
          ("на", "самом"), ("самом", "деле"),
          ("друг", "друга"), ("друг", "другу"), ("друг", "другом"),
          # NEW: "к тому же" pattern
          ("к", "тому"), ("тому", "же"),
          ("прежде", "чем"), ("вместо", "того"), ("ради", "того"),
          # NEW: Interjections and fixed expressions
          ("ну", "что"), ("ну", "вот"), ("ну", "да"), ("ну", "ж"),
          ("вот", "так"), ("вот", "и"), ("вот", "же"),
          ("как", "же"), ("что", "ж"), ("ну", "же"),
          # NEW: Numerals with nouns
          ("один", "раз"), ("два", "раза"), ("три", "раза"),
          # NEW: Modal constructions
          ("надо", "бы"), ("нужно", "бы"), ("можно", "бы"),
          ("хотел", "бы"), ("хотела", "бы"), ("хотелось", "бы"),
          # NEW: Fixed verb phrases
          ("будет", "ли"), ("есть", "ли"), ("было", "бы"),
          ("могу", "ли"), ("можно", "ли"),
        }
        if (p, n) in _UNBREAKABLE_PAIRS_LOCAL:
          can_break = False

      if line and can_break:
        out_lines.append(line)
        prev_word = word
        line = word
      elif line and not can_break:
        # Force join: don't break, keep word on same line even if too wide
        prev_word = word
        line = cand
      else:
        # Single too-long token: split by syllables (pyphen) with fallback.
        chunks = _split_long_token(word)
        for ch in chunks[:-1]:
          if ch:
            out_lines.append(ch)
        prev_word = chunks[-1] if chunks else word
        line = prev_word
    if line:
      out_lines.append(line)

  while out_lines and out_lines[0] == "":
    out_lines.pop(0)
  while out_lines and out_lines[-1] == "":
    out_lines.pop()
  return "\n".join(out_lines)

_RU_BAD_END_STRONG = {
  "а",
  "и",
  "но",
  "да",
  "ли",
  "же",
  "бы",
  "б",
  "в",
  "во",
  "к",
  "ко",
  "с",
  "со",
  "у",
  "о",
  "об",
  "обо",
  "на",
  "над",
  "под",
  "по",
  "при",
  "для",
  "от",
  "до",
  "за",
  "из",
  "изо",
  "без",
  "безо",
  "про",
  "через",
  "перед",
  "передо",
  "между",
  "что",
  "как",
  "чтобы",
  "когда",
  "если",
  "то",
  "вот",
  "не",
  "ни",
}

_RU_BAD_END_WEAK = {
  "я",
  "ты",
  "он",
  "она",
  "оно",
  "мы",
  "вы",
  "они",
  "мне",
  "тебе",
  "ему",
  "ей",
  "нам",
  "вам",
  "им",
  "меня",
  "тебя",
  "его",
  "ее",
  "её",
  "нас",
  "вас",
  "их",
  "мой",
  "твой",
  "наш",
  "ваш",
  "свой",
  "этот",
  "эта",
  "это",
  "эти",
  "тот",
  "та",
  "те",
}

_RU_NO_BREAK_AFTER = {
  # Particles (absolute ban)
  "не",
  "ни",
  "бы",
  "б",
  "ли",
  "же",
  # Single-letter prepositions
  "в",
  "к",
  "с",
  "у",
  "о",
  # Two-letter prepositions (critical)
  "на",
  "за",
  "до",
  "по",
  "из",
  "об",
  # Single-letter conjunctions
  "и",
  "а",
}

_RU_MONTHS_GEN = {
  "января",
  "февраля",
  "марта",
  "апреля",
  "мая",
  "июня",
  "июля",
  "августа",
  "сентября",
  "октября",
  "ноября",
  "декабря",
}

_RU_KEEP_NUM_WITH = _RU_MONTHS_GEN | {"года", "году", "год", "лет", "раз", "часа", "часов", "минут", "минуты", "секунд", "секунды"}

_STRIP_CHARS = "()[]«»“”.,:;!?…\"'"
_PUNCT_CHARS = set(".,:;!?…")
_CLOSING_CHARS = set("”»)]\"'")
_NUM_RE = re.compile(r"^\d+[\d.,:–\-]*$")


def _detect_lang(text: str, fallback: str = "auto") -> str:
  desired = str(fallback or "auto").strip().lower()
  if desired in ("ru", "rus", "russian"):
    return "ru"
  if desired in ("en", "eng", "english"):
    return "en"
  if desired not in ("auto", ""):
    return desired

  cyr = 0
  lat = 0
  for ch in text or "":
    o = ord(ch)
    if 0x0400 <= o <= 0x04FF:
      cyr += 1
    elif (0x0041 <= o <= 0x007A) or (0x00C0 <= o <= 0x00FF):
      lat += 1
  if cyr >= max(3, lat * 2):
    return "ru"
  if lat >= max(3, cyr * 2):
    return "en"
  return "auto"


def _tok_norm(token: str) -> str:
  return (token or "").strip(_STRIP_CHARS).strip().lower()


def _is_number_token(token: str) -> bool:
  t = _tok_norm(token)
  return bool(t) and bool(_NUM_RE.match(t))


def _tokenize_ws(text: str) -> List[str]:
  text = " ".join((text or "").split())
  return text.split(" ") if text else []


def _break_allowed(prev_tok: str, next_tok: str, *, lang: str) -> bool:
  if not prev_tok or not next_tok:
    return True

  if lang == "ru":
    p = _tok_norm(prev_tok)
    n = _tok_norm(next_tok)

    if p in _RU_NO_BREAK_AFTER:
      return False

    if _is_number_token(prev_tok) and n in _RU_KEEP_NUM_WITH:
      return False

    _UNBREAKABLE_PAIRS = {
      # Existing pairs
      ("так", "же"), ("то", "есть"), ("то", "же"),
      ("потому", "что"), ("для", "того"), ("перед", "тем"),
      ("после", "того"), ("из-за", "того"), ("с", "ума"),
      ("как", "бы"), ("как", "будто"),
      ("что", "бы"),
      ("может", "быть"), ("должен", "быть"),
      ("в", "общем"), ("в", "целом"), ("в", "итоге"),
      ("на", "самом"), ("самом", "деле"),
      ("друг", "друга"), ("друг", "другу"), ("друг", "другом"),
      # NEW: "к тому же" pattern
      ("к", "тому"), ("тому", "же"),
      ("прежде", "чем"), ("вместо", "того"), ("ради", "того"),
      # NEW: Interjections and fixed expressions
      ("ну", "что"), ("ну", "вот"), ("ну", "да"), ("ну", "ж"),
      ("вот", "так"), ("вот", "и"), ("вот", "же"),
      ("как", "же"), ("что", "ж"), ("ну", "же"),
      # NEW: Numerals with nouns
      ("один", "раз"), ("два", "раза"), ("три", "раза"),
      # NEW: Modal constructions
      ("надо", "бы"), ("нужно", "бы"), ("можно", "бы"),
      ("хотел", "бы"), ("хотела", "бы"), ("хотелось", "бы"),
      # NEW: Fixed verb phrases
      ("будет", "ли"), ("есть", "ли"), ("было", "бы"),
      ("могу", "ли"), ("можно", "ли"),
    }
    if (p, n) in _UNBREAKABLE_PAIRS:
      return False

  return True


def _line_end_penalty(last_tok: str, *, lang: str) -> float:
  if not last_tok:
    return 0.0
  if lang != "ru":
    return 0.0
  base = _tok_norm(last_tok)
  if not base:
    return 0.0
  n = len(base)
  if base in ("не", "ни"):
    return 100.0
  if base in _RU_BAD_END_STRONG:
    if n <= 2:
      return 80.0
    if n <= 3:
      return 36.0
    return 18.0
  if base in _RU_BAD_END_WEAK:
    if n <= 2:
      return 10.0
    if n <= 3:
      return 6.0
    return 3.0
  return 0.0


def _punct_break_bonus(last_tok: str) -> float:
  if not last_tok:
    return 0.0
  s = str(last_tok).strip()
  while s and s[-1] in _CLOSING_CHARS:
    s = s[:-1]
  if s and s[-1] in _PUNCT_CHARS:
    return -4.0
  return 0.0


def _polygon_widths_at_lines(
  poly_rel: Optional[List[List[float]]],
  box_h: float,
  line_count: int,
  *,
  inset_frac: float = 0.10,
) -> Optional[List[float]]:
  """
  Compute horizontal chord widths of a convex polygon at each line's vertical center.
  Returns absolute pixel widths, or None if polygon is unavailable/degenerate.
  """
  if not poly_rel or len(poly_rel) < 3 or line_count <= 0 or box_h <= 0:
    return None

  # Inset to add a small margin inside the polygon
  # Collect all y values to find range
  ys = [float(p[1]) for p in poly_rel]
  xs = [float(p[0]) for p in poly_rel]
  y_min, y_max = min(ys), max(ys)
  poly_h = y_max - y_min
  if poly_h <= 0:
    return None

  # Apply inset fraction (avoid touching edges)
  inset_y = poly_h * inset_frac
  usable_top = y_min + inset_y
  usable_bot = y_max - inset_y
  usable_h = usable_bot - usable_top
  if usable_h <= 0:
    return None

  widths: List[float] = []
  n = len(poly_rel)
  for line_idx in range(line_count):
    # Vertical center of this line
    y_center = usable_top + (float(line_idx) + 0.5) / float(line_count) * usable_h

    # Find all x-intersections of horizontal line y=y_center with polygon edges
    x_hits: List[float] = []
    for i in range(n):
      x1, y1 = float(poly_rel[i][0]), float(poly_rel[i][1])
      x2, y2 = float(poly_rel[(i + 1) % n][0]), float(poly_rel[(i + 1) % n][1])
      if (y1 <= y_center <= y2) or (y2 <= y_center <= y1):
        dy = y2 - y1
        if abs(dy) < 1e-9:
          x_hits.extend([x1, x2])
        else:
          t = (y_center - y1) / dy
          x_hits.append(x1 + t * (x2 - x1))

    if len(x_hits) < 2:
      # Fallback: use bbox width
      widths.append(max(xs) - min(xs))
      continue

    x_hits.sort()
    # Width is max_x - min_x with inset
    chord = x_hits[-1] - x_hits[0]
    inset_x_total = chord * inset_frac * 2
    widths.append(max(1.0, chord - inset_x_total))

  return widths if all(w > 0 for w in widths) else None


def _coerce_poly_rel(raw_poly: Any) -> Optional[List[List[float]]]:
  if not isinstance(raw_poly, list) or len(raw_poly) < 3:
    return None
  out: List[List[float]] = []
  for p in raw_poly:
    if not isinstance(p, (list, tuple)) or len(p) < 2:
      continue
    try:
      out.append([float(p[0]), float(p[1])])
    except Exception:
      continue
  return out if len(out) >= 3 else None


def _isotonic_increasing(values: Sequence[float]) -> List[float]:
  vals = [float(v) for v in values]
  if not vals:
    return []

  means: List[float] = []
  counts: List[int] = []
  for v in vals:
    means.append(float(v))
    counts.append(1)
    while len(means) >= 2 and means[-2] > means[-1]:
      total = (means[-2] * float(counts[-2])) + (means[-1] * float(counts[-1]))
      cnt = counts[-2] + counts[-1]
      means[-2] = total / float(cnt)
      counts[-2] = cnt
      means.pop()
      counts.pop()

  out: List[float] = []
  for m, c in zip(means, counts):
    out.extend([float(m)] * int(c))
  return out


def _enforce_narrow_wide_narrow(
  widths: Sequence[float],
  *,
  min_value: float = 0.0,
) -> List[float]:
  vals = [max(0.0, float(v)) for v in widths]
  n = len(vals)
  if n <= 2:
    return [max(float(min_value), v) for v in vals]

  if n % 2 == 1:
    c = n // 2
    left = _isotonic_increasing(vals[: c + 1])
    right = list(reversed(_isotonic_increasing(list(reversed(vals[c:])))))
    peak = max(left[-1], right[0])
    left[-1] = peak
    right[0] = peak
    merged = left[:-1] + right
  else:
    c1 = (n // 2) - 1
    c2 = n // 2
    left = _isotonic_increasing(vals[: c1 + 1])
    right = list(reversed(_isotonic_increasing(list(reversed(vals[c2:])))))
    peak = max(left[-1], right[0])
    left[-1] = peak
    right[0] = peak
    merged = left + right

  return [max(float(min_value), float(v)) for v in merged]


def _shape_monotonicity_stats(line_widths: Sequence[float], *, tol_ratio: float = 0.04) -> Dict[str, float]:
  widths = [max(0.0, float(w)) for w in line_widths]
  n = len(widths)
  if n <= 2:
    return {"hard": 0.0, "soft": 0.0, "edge_ratio": 1.0, "peak_shift": 0.0}

  peak = (float(n) - 1.0) / 2.0
  hard = 0
  soft = 0.0

  for i in range(n - 1):
    cur = widths[i]
    nxt = widths[i + 1]
    should_grow = (float(i) + 0.5) < peak
    delta = float(nxt - cur)
    tol = max(2.0, max(cur, nxt) * float(tol_ratio))
    if should_grow:
      if delta < -tol:
        hard += 1
      if delta < 0.0:
        soft += (-delta) / max(10.0, cur)
    else:
      if delta > tol:
        hard += 1
      if delta > 0.0:
        soft += delta / max(10.0, nxt)

  peak_idx = max(range(n), key=lambda idx: widths[idx])
  peak_shift = abs(float(peak_idx) - peak)
  peak_w = max(widths)
  edge_w = max(widths[0], widths[-1])
  edge_ratio = edge_w / max(1.0, peak_w)

  if edge_ratio > 0.92:
    hard += 1
  soft += max(0.0, edge_ratio - 0.82) * 4.0
  soft += float(peak_shift) * 0.60

  return {
    "hard": float(hard),
    "soft": float(soft),
    "edge_ratio": float(edge_ratio),
    "peak_shift": float(peak_shift),
  }


def _oval_widths_for_lines(
  max_width: float,
  line_count: int,
  *,
  alpha: float = 0.78,
  min_factor: float = 0.62,
  aspect_ratio: float = 1.0,
) -> List[float]:
  """
  Compute per-line target widths using inscribed ellipse equation.
  For a line at normalized position y in [-1, 1], width = rx * sqrt(1 - (y/ry)^2).
  This produces the narrow-wide-narrow shape naturally.
  
  aspect_ratio: bubble_w / bubble_h — affects ellipse eccentricity.
  """
  if line_count <= 0:
    return []
  if line_count == 1:
    return [float(max_width)]
  if line_count == 2:
    # For 2 lines, both should be close to max width (nearly flat)
    return [float(max_width)] * 2

  a = max(0.0, min(0.95, float(alpha)))
  mn = max(0.45, min(1.0, float(min_factor)))

  widths: List[float] = []
  for i in range(line_count):
    # Normalized position: -1 at top, +1 at bottom
    y_norm = (2.0 * (float(i) + 0.5) / float(line_count)) - 1.0
    # Ellipse equation: factor = sqrt(1 - (alpha * y)^2)
    val = 1.0 - (a * y_norm) ** 2
    f = math.sqrt(max(0.0, val))
    f = max(mn, min(1.0, f))
    widths.append(float(max_width) * f)

  return widths


def _line_width_targets(
  max_width: float,
  line_count: int,
  *,
  alpha: float,
  min_factor: float,
  poly_rel: Optional[List[List[float]]] = None,
  poly_inset_frac: float = 0.11,
  poly_blend: float = 0.74,
) -> List[float]:
  if line_count <= 0:
    return []

  base = _oval_widths_for_lines(float(max_width), int(line_count), alpha=float(alpha), min_factor=float(min_factor))
  if line_count <= 2:
    return base

  poly = _coerce_poly_rel(poly_rel)
  if not poly:
    return _enforce_narrow_wide_narrow(base, min_value=float(max_width) * float(min_factor))

  ys = [float(p[1]) for p in poly]
  poly_h = max(1.0, (max(ys) - min(ys)))
  inset = max(0.02, min(0.24, float(poly_inset_frac)))
  poly_widths = _polygon_widths_at_lines(poly, box_h=poly_h, line_count=int(line_count), inset_frac=inset)
  if not poly_widths:
    return _enforce_narrow_wide_narrow(base, min_value=float(max_width) * float(min_factor))

  poly_widths = _enforce_narrow_wide_narrow(poly_widths, min_value=1.0)
  poly_max = max(poly_widths)
  if poly_max <= 0:
    return _enforce_narrow_wide_narrow(base, min_value=float(max_width) * float(min_factor))

  base_factors = [max(0.0, min(1.0, float(w) / max(1.0, float(max_width)))) for w in base]
  poly_factors = [max(float(min_factor), min(1.0, float(w) / float(poly_max))) for w in poly_widths]

  blend = max(0.0, min(1.0, float(poly_blend)))
  factors = [
    (float(poly_factors[i]) * blend) + (float(base_factors[i]) * (1.0 - blend))
    for i in range(int(line_count))
  ]
  factors = _enforce_narrow_wide_narrow(factors, min_value=float(min_factor))
  fmax = max(factors) if factors else 1.0
  if fmax <= 0:
    fmax = 1.0
  factors = [max(float(min_factor), min(1.0, float(f) / float(fmax))) for f in factors]
  return [float(max_width) * float(f) for f in factors]


def _get_shape_factor(
  line_idx: int,
  total_lines: int,
  *,
  alpha: float = 0.78,
  min_factor: float = 0.62,
) -> float:
  """
  Returns a width factor (0..1) for a given line index.
  Uses inscribed ellipse equation for smooth narrow-wide-narrow profile.
  """
  if total_lines <= 1:
    return 1.0
  if total_lines == 2:
    return 1.0

  a = max(0.0, min(0.95, float(alpha)))
  mn = max(0.45, min(1.0, float(min_factor)))

  y_norm = (2.0 * (float(line_idx) + 0.5) / float(total_lines)) - 1.0
  val = 1.0 - (a * y_norm) ** 2
  f = math.sqrt(max(0.0, float(val)))
  return max(mn, min(1.0, float(f)))


def _estimate_max_lines(font: ImageFont.FreeTypeFont, font_pt: int, box_h: int, spacing: int) -> int:
  try:
    ascent, descent = font.getmetrics()
    line_h = int(ascent + descent)
  except Exception:
    line_h = int(font_pt)
  line_h = max(1, int(line_h))
  spacing = max(0, int(spacing))
  # total height for L lines: L*line_h + (L-1)*spacing <= box_h
  if box_h <= 0:
    return 1
  return max(1, int((int(box_h) + spacing) // (line_h + spacing if (line_h + spacing) > 0 else line_h)))


def _solve_dp_fixed_lines(
  tokens: List[AtomicToken],
  widths: List[float],
  font: ImageFont.FreeTypeFont,
  line_count: int,
  lang: str,
) -> Tuple[float, List[str]]:
  """DP solver for optimal line breaking with per-line width targets."""
  n = len(tokens)
  num_lines = len(widths)
  if num_lines == 0 or n == 0:
    return 1e18, []

  INF = 1e18
  dp = [[INF] * (n + 1) for _ in range(num_lines + 1)]
  prev = [[None] * (n + 1) for _ in range(num_lines + 1)]
  dp[0][0] = 0.0

  for k in range(num_lines):
    target_w = widths[k]
    center_dist = abs(float(k) - (float(num_lines) - 1.0) / 2.0)
    max_center_dist = (float(num_lines) - 1.0) / 2.0 if num_lines > 1 else 1.0
    center_factor = 1.0 - (center_dist / max(1.0, max_center_dist)) * 0.35
    fill_weight = 22.0 * center_factor

    for i in range(n):
      cost_so_far = dp[k][i]
      if cost_so_far >= INF:
        continue

      remaining_tokens = n - i
      remaining_lines = num_lines - k
      if remaining_tokens < remaining_lines:
        continue

      current_line_text = ""

      for j in range(i, n):
        remain_after = n - (j + 1)
        lines_after = num_lines - (k + 1)
        if remain_after < lines_after:
          continue

        chunk = tokens[j].text
        sep = ""
        if j > i:
          sep = tokens[j - 1].join_cont

        seg_text = (current_line_text + sep + chunk) if j > i else chunk
        width_if_break = _text_width(seg_text + tokens[j].join_break, font)
        width_content = _text_width(seg_text, font)

        if width_content > target_w * 1.04:
          if width_content > target_w * 1.11:
            break

        current_line_text = seg_text

        if not tokens[j].can_break:
          continue

        if j < n - 1 and (" " in (tokens[j].join_cont or "")):
          if not _break_allowed(tokens[j].text, tokens[j + 1].text, lang=lang):
            continue

        if width_if_break > target_w * 1.06:
          continue

        is_last_line = (k == num_lines - 1)
        is_hyphen_break = (tokens[j].join_break == "-") and ((tokens[j].join_cont or "") == "")

        fill_ratio = width_if_break / max(1.0, target_w)
        if fill_ratio > 1.0:
          deviation = (fill_ratio - 1.0) * 90.0
        else:
          deviation = ((1.0 - fill_ratio) ** 1.65) * fill_weight
        if is_last_line:
          deviation *= 0.55

        pen = tokens[j].penalty
        pen += _line_end_penalty(tokens[j].text, lang=lang)
        pen += _punct_break_bonus(tokens[j].text)

        if is_hyphen_break:
          pen += 1.8
          if tokens[j].text and str(tokens[j].text)[0].isupper():
            pen += 3.2

        pen += 0.08

        line_cost = deviation + pen

        if cost_so_far + line_cost < dp[k + 1][j + 1]:
          dp[k + 1][j + 1] = cost_so_far + line_cost
          prev[k + 1][j + 1] = i

  final_cost = dp[num_lines][n]
  if final_cost >= INF:
    return INF, []

  lines_rev = []
  cur = n
  for k in range(num_lines, 0, -1):
    start = prev[k][cur]
    if start is None:
      return INF, []

    seg_str = ""
    for idx in range(start, cur):
      if idx > start:
        seg_str += tokens[idx - 1].join_cont
      seg_str += tokens[idx].text

    # Add break char if not last line overall
    if cur < n:
      seg_str += tokens[cur - 1].join_break

    lines_rev.append(seg_str)
    cur = start

  return final_cost, list(reversed(lines_rev))


def _compose_paragraph_dp(
  para: str,
  font: ImageFont.FreeTypeFont,
  *,
  max_width: int,
  max_lines: int,
  lang: str,
  profile_alpha: float = 0.78,
  profile_min_factor: float = 0.62,
  line_break_penalty: float = 0.4,
  shape_penalty: float = 2.5,
  center_peak_penalty: float = 2.0,
  hyphen_break_penalty: float = 5.0,
  profile_poly_rel: Optional[List[List[float]]] = None,
  profile_poly_inset_frac: float = 0.11,
  profile_poly_blend: float = 0.74,
) -> str:
  """Compose a paragraph into lines that follow a narrow-wide-narrow shape profile.

  Explores all possible line counts and multiple alpha variants to find the
  optimal layout. The DP solver handles per-line target widths and consecutive
  hyphen penalties internally.
  """
  tokens_plain = _tokenize_atomic(para, lang=lang, allow_auto_hyphen=False)
  tokens_auto = _tokenize_atomic(para, lang=lang, allow_auto_hyphen=True)

  plans: List[Tuple[bool, List[AtomicToken]]] = []
  if tokens_plain:
    plans.append((False, tokens_plain))
  if tokens_auto:
    plans.append((True, tokens_auto))
  if not plans:
    return ""

  best_cost = 1e18
  best_lines: List[str] = []
  best_strict_cost = 1e18
  best_strict_lines: List[str] = []
  word_count = len(_tokenize_ws(para))
  hyphen_penalty_eff = float(hyphen_break_penalty)
  if word_count <= 2:
    hyphen_penalty_eff *= 0.30
  elif word_count <= 4:
    hyphen_penalty_eff *= 0.55

  one_line_possible = (_text_width(para, font) <= float(max_width))

  # Alpha variants to explore for better shape fitting
  alpha_variants = [
    max(0.0, float(profile_alpha) - 0.06),
    float(profile_alpha),
    min(0.95, float(profile_alpha) + 0.08),
  ]

  for auto_hyphen, tokens in plans:
    start_L = 1 if one_line_possible else 2
    start_L = max(1, min(int(max_lines), start_L))

    for L in range(start_L, max_lines + 1):
      # For 1-2 lines, use flat profile; for 3+, try alpha variants
      if L <= 2:
        variants = [(profile_alpha, profile_min_factor)]
      else:
        variants = [
          (a, float(profile_min_factor))
          for a in alpha_variants
        ]

      for alpha_v, mf_v in variants:
        widths = _line_width_targets(
          float(max_width),
          L,
          alpha=float(alpha_v),
          min_factor=float(mf_v),
          poly_rel=profile_poly_rel,
          poly_inset_frac=float(profile_poly_inset_frac),
          poly_blend=float(profile_poly_blend),
        )
        if not widths:
          continue

        cost, lines = _solve_dp_fixed_lines(tokens, widths, font, L, lang)
        if cost >= 1e18 or not lines:
          continue

        # Post-hoc shape quality evaluation
        line_widths = [max(0.0, _text_width((ln or "").rstrip(), font)) for ln in lines]
        shape_stats = _shape_monotonicity_stats(line_widths)
        shape_hard = int(shape_stats.get("hard", 0.0))
        shape_soft = float(shape_stats.get("soft", 0.0))

        if len(line_widths) >= 4 and shape_hard > 0:
          cost += float(shape_hard) * 120.0
        elif len(line_widths) == 3 and shape_hard > 0:
          cost += float(shape_hard) * 70.0

        cost += shape_soft * 9.0

        if len(line_widths) >= 3:
          cost += _shape_sequence_penalty(line_widths) * float(shape_penalty)
          cost += _center_peak_penalty(line_widths) * float(center_peak_penalty)
        elif len(line_widths) == 2:
          cost += _shape_sequence_penalty(line_widths) * float(shape_penalty) * 0.2

        # Post-hoc: small penalty per hyphen (shape is more important)
        hyphen_breaks = 0
        for idx, ln in enumerate(lines[:-1]):
          s = (ln or "").strip()
          if s.endswith("-") and s.count("-") < 2:
            hyphen_breaks += 1
        cost += float(hyphen_breaks) * float(hyphen_penalty_eff) * 0.22

        # Slight preference for non-auto-hyphen layouts when quality is very close.
        if auto_hyphen:
          cost += 0.5

        # Vertical fill: prefer line counts that fill the bubble's height.
        # Layouts using fewer lines than max_lines leave vertical space empty.
        # This is the key incentive for using hyphens to achieve more lines.
        if max_lines > 1:
          fill_ratio_v = float(L) / float(max_lines)
          if fill_ratio_v < 0.5:
            # Very under-filled: very strong penalty
            cost += (1.0 - fill_ratio_v) * 25.0
          elif fill_ratio_v < 0.75:
            # Under-filled: strong penalty
            cost += (1.0 - fill_ratio_v) * 12.0
          elif fill_ratio_v < 0.90:
            # Slightly under-filled: moderate penalty
            cost += (1.0 - fill_ratio_v) * 5.0
          # else: well-filled or fully filled, no penalty

        strict_shape_ok = (len(line_widths) <= 2) or (shape_hard <= 0)

        if strict_shape_ok and cost < best_strict_cost:
          best_strict_cost = cost
          best_strict_lines = lines

        if cost < best_cost:
          best_cost = cost
          best_lines = lines

  if best_strict_lines:
    return "\n".join(best_strict_lines)

  if not best_lines:
    return _wrap_text(para, font, max_width=max_width)

  return "\n".join(best_lines)


def _compose_text_output(
  text: str,
  font: ImageFont.FreeTypeFont,
  *,
  max_width: int,
  max_height: int,
  font_pt: int,
  leading_factor: float,
  config: Dict[str, Any],
) -> Tuple[str, bool]:
  komp = config.get("komponovka") if isinstance(config, dict) else None
  if komp is True:
    komp = {"enabled": True}
  if not isinstance(komp, dict):
    komp = {"enabled": bool(config.get("komponovka_enabled", False))}

  enabled = bool(komp.get("enabled", False))
  mode = str(komp.get("mode", "auto") or "auto").strip().lower()
  lang = _detect_lang(text, fallback=str(komp.get("lang", "auto")))
  profile_cfg = komp.get("profile") if isinstance(komp, dict) else None
  if not isinstance(profile_cfg, dict):
    profile_cfg = {}
  bubble_profile = config.get("_layout_profile") if isinstance(config, dict) else None
  if not isinstance(bubble_profile, dict):
    bubble_profile = {}

  alpha = _safe_float(
    bubble_profile.get("alpha", profile_cfg.get("alpha", 0.78)),
    0.78,
  )
  min_factor = _safe_float(
    bubble_profile.get("min_factor", profile_cfg.get("min_factor", 0.62)),
    0.62,
  )
  line_break_penalty = _safe_float(
    bubble_profile.get("line_break_penalty", profile_cfg.get("line_break_penalty", 0.42)),
    0.42,
  )
  shape_penalty = _safe_float(
    bubble_profile.get("shape_penalty", profile_cfg.get("shape_penalty", 2.5)),
    2.5,
  )
  center_peak_penalty = _safe_float(
    bubble_profile.get("center_peak_penalty", profile_cfg.get("center_peak_penalty", 2.0)),
    2.0,
  )
  hyphen_break_penalty = _safe_float(
    bubble_profile.get("hyphen_break_penalty", profile_cfg.get("hyphen_break_penalty", 5.0)),
    5.0,
  )
  poly_rel = _coerce_poly_rel(bubble_profile.get("poly_rel"))
  poly_inset_frac = _safe_float(
    bubble_profile.get("poly_inset_frac", profile_cfg.get("poly_inset_frac", 0.11)),
    0.11,
  )
  poly_blend = _safe_float(
    bubble_profile.get("poly_blend", profile_cfg.get("poly_blend", 0.74)),
    0.74,
  )
  alpha = max(0.0, min(0.95, float(alpha)))
  min_factor = max(0.45, min(1.0, float(min_factor)))
  line_break_penalty = max(0.0, min(4.0, float(line_break_penalty)))
  shape_penalty = max(0.0, min(8.0, float(shape_penalty)))
  center_peak_penalty = max(0.0, min(8.0, float(center_peak_penalty)))
  hyphen_break_penalty = max(0.0, min(12.0, float(hyphen_break_penalty)))
  poly_inset_frac = max(0.02, min(0.24, float(poly_inset_frac)))
  poly_blend = max(0.0, min(1.0, float(poly_blend)))

  raw = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
  if not raw:
    return "", False

  if not enabled or mode in ("off", "false", "0", "disabled"):
    return raw, False

  if mode == "auto" and "\n" in raw:
    # Respect manual line breaks by default.
    return raw, False

  spacing = int(round(float(font_pt) * float(leading_factor)))
  max_lines = _estimate_max_lines(font, font_pt, int(max_height), int(spacing))

  # Split into paragraphs (should be a single paragraph for mode=auto).
  paras = raw.split("\n")
  out_paras: List[str] = []
  for para in paras:
    para = " ".join((para or "").split())
    if not para:
      out_paras.append("")
      continue
    out_paras.append(
      _compose_paragraph_dp(
        para,
        font,
        max_width=int(max_width),
        max_lines=int(max_lines),
        lang=lang,
        profile_alpha=alpha,
        profile_min_factor=min_factor,
        line_break_penalty=line_break_penalty,
        shape_penalty=shape_penalty,
        center_peak_penalty=center_peak_penalty,
        hyphen_break_penalty=hyphen_break_penalty,
        profile_poly_rel=poly_rel,
        profile_poly_inset_frac=poly_inset_frac,
        profile_poly_blend=poly_blend,
      )
    )

  return "\n".join(out_paras).strip("\n"), True


def _fit_at_size(
  text: str,
  font: ImageFont.FreeTypeFont,
  font_pt: int,
  box_w: int,
  box_h: int,
  draw: ImageDraw.ImageDraw,
  leading_factor: float,
  *,
  config: Dict[str, Any],
) -> Tuple[bool, str, str]:
  spacing = int(round(float(font_pt) * float(leading_factor)))

  # If komponovka is enabled, we want to output the composed text (with explicit line breaks)
  # AND measure fit against the same text. Otherwise, keep output text unchanged and only
  # use greedy wrapping for measurement (to approximate Photoshop text-box wrapping).
  output_text = _compose_text_output(
    text,
    font,
    max_width=int(box_w),
    max_height=int(box_h),
    font_pt=int(font_pt),
    leading_factor=float(leading_factor),
    config=config,
  )
  output_text, applied = output_text
  measure_text = output_text if applied else _wrap_text(output_text, font, max_width=int(box_w))

  try:
    x1, y1, x2, y2 = draw.multiline_textbbox((0, 0), measure_text, font=font, spacing=spacing, align="center")
    tw = int(math.ceil(x2 - x1))
    th = int(math.ceil(y2 - y1))
  except Exception:
    lines = measure_text.split("\n") if measure_text else [""]
    tw = int(max((_text_width(line, font) for line in lines), default=0))
    try:
      ascent, descent = font.getmetrics()
      lh = int(ascent + descent + spacing)
    except Exception:
      lh = int(font_pt + spacing)
    th = int(len(lines) * lh)
  return (tw <= int(box_w) and th <= int(box_h)), measure_text, output_text


def _measure_multiline_text(
  text: str,
  font: ImageFont.FreeTypeFont,
  *,
  spacing: int,
  draw: ImageDraw.ImageDraw,
) -> Tuple[int, int]:
  try:
    x1, y1, x2, y2 = draw.multiline_textbbox((0, 0), text, font=font, spacing=int(spacing), align="center")
    tw = int(math.ceil(x2 - x1))
    th = int(math.ceil(y2 - y1))
    return max(0, tw), max(0, th)
  except Exception:
    lines = text.split("\n") if text else [""]
    tw = int(max((_text_width(line, font) for line in lines), default=0))
    try:
      ascent, descent = font.getmetrics()
      lh = int(ascent + descent + int(spacing))
    except Exception:
      lh = int(getattr(font, "size", 16) + int(spacing))
    th = int(len(lines) * lh)
    return max(0, tw), max(0, th)


def _layout_shape_stats(
  layout_text: str,
  font: ImageFont.FreeTypeFont,
  *,
  tol_ratio: float = 0.04,
) -> Dict[str, float]:
  lines = [ln for ln in (layout_text or "").split("\n") if (ln or "").strip()]
  line_count = len(lines)
  if line_count <= 2:
    return {"line_count": float(line_count), "hard": 0.0, "soft": 0.0}
  widths = [max(0.0, _text_width((ln or "").rstrip(), font)) for ln in lines]
  st = _shape_monotonicity_stats(widths, tol_ratio=float(tol_ratio))
  return {
    "line_count": float(line_count),
    "hard": float(st.get("hard", 0.0)),
    "soft": float(st.get("soft", 0.0)),
  }


def _layout_badness_score(layout_text: str) -> float:
  lines = (layout_text or "").split("\n") if layout_text else []
  if not lines:
    return 0.0
  word_count = len(_tokenize_ws(" ".join(lines)))
  if word_count <= 2:
    hyphen_w = 1.0
    short_hyphen_w = 8.0
  elif word_count <= 4:
    hyphen_w = 2.0
    short_hyphen_w = 3.8
  else:
    hyphen_w = 3.0
    short_hyphen_w = 5.0

  stutter_like = False
  if len(lines) >= 2:
    l0 = (lines[0] or "").strip()
    l1 = (lines[1] or "").strip()
    l0n = _tok_norm(l0.rstrip("-"))
    l1n = _tok_norm(l1)
    if l0.endswith("-") and len(l0n) <= 2 and l1n:
      stutter_like = True

  hyphen_breaks = 0
  short_hyphen_breaks = 0
  tiny_lines = 0
  for i, ln in enumerate(lines):
    s = (ln or "").strip()
    if len(_tok_norm(s)) <= 2 and i < len(lines) - 1 and not (stutter_like and i == 0):
      tiny_lines += 1
    if s.endswith("-"):
      hyphen_breaks += 1
      frag = _tok_norm(s[:-1].split(" ")[-1] if " " in s[:-1] else s[:-1])
      if len(frag) <= 3 and not (stutter_like and i == 0):
        short_hyphen_breaks += 1

  # Approximate visual line widths to penalize non "narrow-wide-narrow" sequences.
  width_chars: List[float] = []
  for ln in lines:
    compact = re.sub(r"\s+", "", (ln or "").strip())
    width_chars.append(float(len(compact)))
  strict_stats = _shape_monotonicity_stats(width_chars, tol_ratio=0.07)
  shape_bad = _shape_sequence_penalty(width_chars)
  peak_bad = _center_peak_penalty(width_chars)

  return (
    (hyphen_breaks * hyphen_w)
    + (short_hyphen_breaks * short_hyphen_w)
    + (tiny_lines * 2.0)
    + (float(strict_stats.get("hard", 0.0)) * 4.5)
    + (float(strict_stats.get("soft", 0.0)) * 1.8)
    + (shape_bad * 0.55)
    + (peak_bad * 0.45)
  )


def _shape_sequence_penalty(line_widths: List[float]) -> float:
  n = len(line_widths)
  if n <= 2:
    return 0.0

  pen = 0.0
  peak = (float(n) - 1.0) / 2.0
  for i in range(n - 1):
    cur = float(line_widths[i])
    nxt = float(line_widths[i + 1])
    # Before peak width should grow; after peak it should shrink.
    should_grow = (float(i) + 0.5) < peak
    if should_grow and nxt < cur:
      pen += ((cur - nxt) / max(12.0, cur)) * 8.0
    if (not should_grow) and nxt > cur:
      pen += ((nxt - cur) / max(12.0, nxt)) * 8.0

  peak_w = max(float(w) for w in line_widths)
  edge_w = max(float(line_widths[0]), float(line_widths[-1]))
  if peak_w > 0.0 and edge_w > (peak_w * 0.95):
    pen += ((edge_w / peak_w) - 0.95) * 12.0
  return pen


def _center_peak_penalty(line_widths: List[float]) -> float:
  n = len(line_widths)
  if n <= 2:
    return 0.0

  widths = [max(0.0, float(w)) for w in line_widths]
  peak_idx = max(range(n), key=lambda i: widths[i])
  center = (float(n) - 1.0) / 2.0

  pen = abs(float(peak_idx) - center) * 2.4

  edge_w = max(widths[0], widths[-1])
  if n % 2 == 1:
    center_w = widths[n // 2]
    if center_w > 0.0 and edge_w >= center_w:
      pen += ((edge_w - center_w) / max(12.0, center_w)) * 18.0
  else:
    c1 = (n // 2) - 1
    c2 = n // 2
    center_w = max(widths[c1], widths[c2])
    if center_w > 0.0 and edge_w >= center_w:
      pen += ((edge_w - center_w) / max(12.0, center_w)) * 14.0

  # Very asymmetric top/bottom edges look visually noisy inside rounded bubbles.
  edge_max = max(12.0, edge_w)
  pen += (abs(widths[0] - widths[-1]) / edge_max) * 3.0
  return pen


def _max_fitting_size(
  text: str,
  font_cache: Dict[int, ImageFont.FreeTypeFont],
  box_w: int,
  box_h: int,
  draw: ImageDraw.ImageDraw,
  leading_factor: float,
  pts: Sequence[int],
  *,
  config: Dict[str, Any],
) -> Tuple[Optional[int], Optional[str]]:
  pts_list = [int(p) for p in pts if p is not None]
  pts_list = [p for p in pts_list if p > 0]
  pts_list.sort()
  if not pts_list:
    return None, None

  lo = 0
  hi = len(pts_list) - 1
  best: Optional[int] = None
  best_text: Optional[str] = None
  while lo <= hi:
    mid_idx = (lo + hi) // 2
    mid_pt = int(pts_list[mid_idx])
    font = font_cache.get(mid_pt)
    if font is None:
      return None, None
    ok, _measure_text, out_text = _fit_at_size(
      text,
      font,
      mid_pt,
      box_w,
      box_h,
      draw=draw,
      leading_factor=leading_factor,
      config=config,
    )
    if ok:
      best = int(mid_pt)
      best_text = out_text
      lo = mid_idx + 1
    else:
      hi = mid_idx - 1
  return best, best_text


def _pick_font_path(config: Dict[str, Any]) -> Optional[str]:
  cfg_path = str(config.get("font_path") or "").strip()
  candidates = [
    cfg_path or None,
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
  ]
  for p in candidates:
    if not p:
      continue
    try:
      if Path(p).exists():
        return str(p)
    except Exception:
      continue
  return None


def _font_hint_key(font: Dict[str, Any]) -> str:
  ps = str(font.get("fontPostScriptName") or font.get("postScriptName") or "").strip().lower()
  fam = str(font.get("fontFamily") or font.get("fontName") or font.get("family") or "").strip().lower()
  style = str(font.get("fontStyleName") or font.get("style") or "").strip().lower()
  return f"{ps}|{fam}|{style}"


def _fc_match_file(pattern: str) -> Optional[str]:
  try:
    res = subprocess.run(
      ["fc-match", "-f", "%{file}\n", str(pattern)],
      capture_output=True,
      text=True,
      check=False,
    )
  except Exception:
    return None
  out = (res.stdout or "").strip().splitlines()
  if not out:
    return None
  path = out[0].strip()
  if not path:
    return None
  try:
    if Path(path).exists():
      return path
  except Exception:
    return None
  return None


_WIN_FONT_EXTS = {".ttf", ".otf", ".ttc", ".otc"}
_WIN_FONTS_DIR: Optional[Path] = None
_WIN_FONT_INDEX: Optional[List[Tuple[str, Path]]] = None
_WIN_PATH_RE = re.compile(r"^[a-zA-Z]:[\\\\/]")


def _normalize_win_path(path: str) -> Optional[str]:
  p = (path or "").strip().strip('"')
  if not p:
    return None
  if _WIN_PATH_RE.match(p):
    return win_path_to_wsl(p)
  return p


def _get_windows_fonts_dir() -> Optional[Path]:
  global _WIN_FONTS_DIR
  if _WIN_FONTS_DIR is not None:
    return _WIN_FONTS_DIR
  env_dir = os.environ.get("TIPER_WIN_FONTS_DIR") or os.environ.get("WINDOWS_FONTS_DIR")
  candidates = [env_dir, "/mnt/c/Windows/Fonts", "/mnt/c/Windows/fonts"]
  for cand in candidates:
    if not cand:
      continue
    try:
      path = Path(str(cand))
      if path.exists():
        _WIN_FONTS_DIR = path
        return _WIN_FONTS_DIR
    except Exception:
      continue
  _WIN_FONTS_DIR = None
  return None


def _normalize_token(text: str) -> str:
  return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def _build_windows_font_index() -> List[Tuple[str, Path]]:
  global _WIN_FONT_INDEX
  if _WIN_FONT_INDEX is not None:
    return _WIN_FONT_INDEX
  base = _get_windows_fonts_dir()
  items: List[Tuple[str, Path]] = []
  if base is not None:
    try:
      for p in base.iterdir():
        if p.is_file() and p.suffix.lower() in _WIN_FONT_EXTS:
          norm = _normalize_token(p.stem)
          if norm:
            items.append((norm, p))
    except Exception:
      items = []
  _WIN_FONT_INDEX = items
  return _WIN_FONT_INDEX


def _find_windows_font_file(font_hint: Dict[str, Any]) -> Optional[str]:
  index = _build_windows_font_index()
  if not index:
    return None

  ps = str(font_hint.get("fontPostScriptName") or font_hint.get("postScriptName") or "").strip()
  fam = str(font_hint.get("fontFamily") or "").strip()
  name = str(font_hint.get("fontName") or "").strip()
  style = str(font_hint.get("fontStyleName") or "").strip()

  base_raw = [ps, fam, name]
  style_l = style.lower()
  style_tokens: List[str] = []
  if style:
    style_tokens.append(style)
  if "bold" in style_l:
    style_tokens.extend(["bold", "bd"])
  if "italic" in style_l or "oblique" in style_l:
    style_tokens.extend(["italic", "it", "i"])
  if ("bold" in style_l) and ("italic" in style_l or "oblique" in style_l):
    style_tokens.append("bi")

  style_raw: List[str] = []
  for st in style_tokens:
    style_raw.extend([f"{fam}{st}", f"{name}{st}", f"{ps}{st}", st])

  def _weighted_tokens(raw: List[str]) -> List[Tuple[str, int]]:
    tokens: List[Tuple[str, int]] = []
    for t in raw:
      if not t:
        continue
      norm = _normalize_token(t)
      if not norm:
        continue
      weight = 0
      if "bolditalic" in norm or norm.endswith("bi"):
        weight = 5
      elif "bold" in norm or norm.endswith("bd"):
        weight = 3
      elif "italic" in norm or norm.endswith("it") or norm.endswith("i"):
        weight = 2
      tokens.append((norm, weight))
    return tokens

  style_tokens = _weighted_tokens(style_raw)
  base_tokens = _weighted_tokens(base_raw)

  def _pick(tokens_w: List[Tuple[str, int]]) -> Optional[Path]:
    if not tokens_w:
      return None
    best_score = -1
    best = None
    for norm, path in index:
      score = 0
      for t, w in tokens_w:
        if norm == t:
          score = max(score, 100 + w)
        elif t in norm:
          score = max(score, 70 + w)
        elif norm in t:
          score = max(score, 60 + w)
      if score > best_score:
        best_score = score
        best = path
    if best is not None and best_score >= 60:
      return best
    return None

  best_path = _pick(style_tokens) or _pick(base_tokens)
  if best_path is not None:
    return str(best_path)
  return None


def _resolve_font_path(
  font_hint: Optional[Dict[str, Any]],
  config: Dict[str, Any],
  *,
  fallback_path: Optional[str],
  cache: Dict[str, Optional[str]],
) -> Optional[str]:
  # Explicit path always wins (useful for headless servers).
  cfg_path = str(config.get("font_path") or "").strip()
  cfg_path_win = str(config.get("font_path_win") or config.get("font_path_windows") or "").strip()
  for raw in (cfg_path, cfg_path_win):
    if not raw:
      continue
    norm = _normalize_win_path(raw) or raw
    try:
      if Path(norm).exists():
        return norm
    except Exception:
      pass

  if not (font_hint and isinstance(font_hint, dict)):
    return fallback_path

  hint_path = font_hint.get("fontPath") or font_hint.get("path") or font_hint.get("file")
  if hint_path:
    norm = _normalize_win_path(str(hint_path))
    if norm:
      try:
        if Path(norm).exists():
          return norm
      except Exception:
        pass

  key = _font_hint_key(font_hint)
  if key in cache:
    return cache[key] or fallback_path

  ps = str(font_hint.get("fontPostScriptName") or font_hint.get("postScriptName") or "").strip()
  fam = str(font_hint.get("fontFamily") or "").strip()
  name = str(font_hint.get("fontName") or "").strip()
  style = str(font_hint.get("fontStyleName") or "").strip()

  patterns: List[str] = []
  for base in [ps, fam, name]:
    if not base:
      continue
    if style:
      patterns.append(f"{base}:style={style}")
    patterns.append(base)

  resolved: Optional[str] = None
  for pat in patterns:
    resolved = _fc_match_file(pat)
    if resolved:
      break

  if not resolved:
    resolved = _find_windows_font_file(font_hint)

  cache[key] = resolved or fallback_path
  return resolved or fallback_path


def _select_font_hint_for_bubble(
  bid: str,
  *,
  config: Dict[str, Any],
  bubble_classes: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
  by_id = config.get("font_by_id")
  if isinstance(by_id, dict):
    item = by_id.get(bid)
    if isinstance(item, dict) and item:
      return item

  base = config.get("font_base")
  base_hint = base if isinstance(base, dict) and base else None

  cls_map = config.get("bubble_class_map") or config.get("class_map")
  if not isinstance(cls_map, dict) or not bubble_classes or not isinstance(bubble_classes, dict):
    return base_hint

  info = bubble_classes.get(bid) or {}
  if not isinstance(info, dict):
    return base_hint

  label = info.get("class") or info.get("label")
  if not label:
    return base_hint

  entry = cls_map.get(str(label)) or None
  if not isinstance(entry, dict) or not entry:
    return base_hint

  # Match the frontend behaviour: avoid applying low-confidence class overrides.
  if info.get("definite") is False:
    return base_hint
  conf_val = info.get("confidence")
  definite = info.get("definite")
  try:
    conf = float(conf_val) if conf_val is not None else None
  except Exception:
    conf = None
  try:
    thr = float(config.get("bubble_class_conf", 0.6))
  except Exception:
    thr = 0.6
  if definite is None and conf is not None and conf < thr:
    return base_hint

  return entry


def _layout_profile_for_bubble(
  *,
  bubble_bbox: Tuple[float, float, float, float],
  core_bbox: Tuple[float, float, float, float],
  geom_item: Dict[str, Any],
  class_info: Dict[str, Any],
  config: Dict[str, Any],
) -> Dict[str, Any]:
  komp = config.get("komponovka") if isinstance(config, dict) else None
  if not isinstance(komp, dict):
    komp = {}
  profile_cfg = komp.get("profile")
  if not isinstance(profile_cfg, dict):
    profile_cfg = {}

  bubble_w = max(1.0, float(bubble_bbox[2] - bubble_bbox[0]))
  bubble_h = max(1.0, float(bubble_bbox[3] - bubble_bbox[1]))
  core_w = max(1.0, float(core_bbox[2] - core_bbox[0]))
  core_h = max(1.0, float(core_bbox[3] - core_bbox[1]))
  bubble_area = bubble_w * bubble_h
  core_area = core_w * core_h
  shape_factor = core_area / bubble_area if bubble_area > 0 else 0.5
  aspect = bubble_w / bubble_h if bubble_h > 0 else 1.0

  src = str((geom_item or {}).get("source") or "").lower()
  fallback = (not src) or src.startswith("bbox_fallback")
  label = str((class_info or {}).get("class") or (class_info or {}).get("label") or "").strip().lower()

  preset = "mild"
  if label == "rectangle":
    preset = "rect"
  elif label in {"thorn", "cloude", "sea_uchirin"}:
    preset = "tapered"
  elif not fallback:
    if shape_factor >= 0.62 or aspect >= 1.40:
      preset = "rect"
    elif shape_factor <= 0.44:
      preset = "tapered"
    else:
      preset = "ellipse"
  elif label == "elipse":
    preset = "ellipse"

  table: Dict[str, Dict[str, float]] = {
    "rect": {
      "alpha": 0.55,
      "min_factor": 0.80,
      "line_break_penalty": 0.78,
      "shape_penalty": 1.8,
      "center_peak_penalty": 1.5,
      "hyphen_break_penalty": 5.5,
    },
    "mild": {
      "alpha": 0.75,
      "min_factor": 0.62,
      "line_break_penalty": 0.55,
      "shape_penalty": 2.5,
      "center_peak_penalty": 2.0,
      "hyphen_break_penalty": 5.0,
    },
    "ellipse": {
      "alpha": 0.85,
      "min_factor": 0.55,
      "line_break_penalty": 0.45,
      "shape_penalty": 2.8,
      "center_peak_penalty": 2.2,
      "hyphen_break_penalty": 5.0,
    },
    "tapered": {
      "alpha": 0.90,
      "min_factor": 0.50,
      "line_break_penalty": 0.40,
      "shape_penalty": 3.0,
      "center_peak_penalty": 2.5,
      "hyphen_break_penalty": 4.8,
    },
  }
  out = dict(table.get(preset, table["mild"]))

  # Very tall balloons benefit from slightly stronger tapering.
  if (bubble_h / bubble_w) >= 1.35 and preset in {"ellipse", "tapered"}:
    out["alpha"] += 0.05
    out["min_factor"] -= 0.03

  # Fallback: keep profile close to "narrow-wide-narrow", only slightly milder.
  if fallback:
    out["alpha"] *= 0.96
    out["min_factor"] += 0.02
    out["line_break_penalty"] += 0.06
    out["shape_penalty"] = float(out.get("shape_penalty", 2.2)) * 0.92
    out["center_peak_penalty"] = float(out.get("center_peak_penalty", 1.8)) * 0.92
    out["hyphen_break_penalty"] = float(out.get("hyphen_break_penalty", 4.0)) + 0.30

  for key in (
    "alpha",
    "min_factor",
    "line_break_penalty",
    "shape_penalty",
    "center_peak_penalty",
    "hyphen_break_penalty",
  ):
    if key in profile_cfg:
      out[key] = _safe_float(profile_cfg.get(key), out[key])

  out["alpha"] = max(0.0, min(0.95, float(out.get("alpha", 0.72))))
  out["min_factor"] = max(0.45, min(1.0, float(out.get("min_factor", 0.66))))
  out["line_break_penalty"] = max(0.0, min(4.0, float(out.get("line_break_penalty", 0.45))))
  out["shape_penalty"] = max(0.0, min(8.0, float(out.get("shape_penalty", 2.2))))
  out["center_peak_penalty"] = max(0.0, min(8.0, float(out.get("center_peak_penalty", 1.8))))
  out["hyphen_break_penalty"] = max(0.0, min(12.0, float(out.get("hyphen_break_penalty", 4.0))))

  poly_rel = _coerce_poly_rel((geom_item or {}).get("poly_rel"))
  out["poly_rel"] = poly_rel
  out["poly_inset_frac"] = max(0.02, min(0.24, _safe_float(profile_cfg.get("poly_inset_frac", 0.11), 0.11)))
  out["poly_blend"] = max(0.0, min(1.0, _safe_float(profile_cfg.get("poly_blend", 0.74), 0.74)))
  return out


def _compute_adaptive_reduction(
  best_pt: int,
  layout_bbox: Tuple[float, float, float, float],
  core_bbox: Tuple[float, float, float, float],
  bubble_bbox: Tuple[float, float, float, float],
  text: str,
  font_cache: Dict[int, "ImageFont.FreeTypeFont"],
  draw: "ImageDraw.ImageDraw",
  leading_factor: float,
  config: Dict[str, Any],
  *,
  min_pt: int = 14,
  max_reduction: int = 3,
) -> int:
  """Compute adaptive font size reduction — deliberately conservative.

  The DP solver already produces well-shaped layouts; the comfort reduction only
  adds a small margin so text doesn't touch bubble edges.  Keep reduction to
  ≤3pt so font sizes remain as large as possible.
  """
  # Get adaptive slack config
  slack_cfg = config.get("adaptive_slack", {})
  if not isinstance(slack_cfg, dict):
    slack_cfg = {}

  # If adaptive fit is disabled, use legacy -2pt (reduced from -4pt)
  if not config.get("use_adaptive_fit", True):
    return max(min_pt, int(best_pt) - 2)

  # Short phrases need minimal reduction
  word_count = len((text or "").split())
  if word_count <= 3:
    return max(min_pt, int(best_pt) - 1)

  # 1. Calculate shape_factor
  core_w = max(1.0, float(core_bbox[2] - core_bbox[0]))
  core_h = max(1.0, float(core_bbox[3] - core_bbox[1]))
  bubble_w = max(1.0, float(bubble_bbox[2] - bubble_bbox[0]))
  bubble_h = max(1.0, float(bubble_bbox[3] - bubble_bbox[1]))
  core_area = core_w * core_h
  bubble_area = bubble_w * bubble_h
  shape_factor = core_area / bubble_area if bubble_area > 0 else 0.5

  # 2. Determine target safety_slack — much lower than before
  elliptical_thr = float(slack_cfg.get("elliptical_threshold", 0.4))
  moderate_thr = float(slack_cfg.get("moderate_threshold", 0.6))

  if shape_factor < elliptical_thr:
    target_slack = float(slack_cfg.get("elliptical_slack", 0.08))
  elif shape_factor < moderate_thr:
    target_slack = float(slack_cfg.get("moderate_slack", 0.06))
  else:
    target_slack = float(slack_cfg.get("rectangular_slack", 0.04))

  # 3. Get iteration parameters — use 1pt steps for fine control
  max_reduce = min(int(slack_cfg.get("max_reduction_pt", max_reduction)), 3)
  pt_step = int(slack_cfg.get("pt_step", 1))
  if pt_step <= 0:
    pt_step = 1

  layout_w = float(layout_bbox[2] - layout_bbox[0])
  layout_h = float(layout_bbox[3] - layout_bbox[1])

  # 4. Iterate to find pt with sufficient slack
  final_pt = best_pt
  for pt in range(int(best_pt), max(int(min_pt), int(best_pt) - max_reduce - 1), -pt_step):
    font = font_cache.get(int(pt))
    if font is None:
      continue

    try:
      spacing = int(round(float(pt) * float(leading_factor)))
      composed, _applied = _compose_text_output(
        text,
        font,
        max_width=int(layout_w),
        max_height=int(layout_h),
        font_pt=int(pt),
        leading_factor=float(leading_factor),
        config=config,
      )
      wrapped = composed or _wrap_text(text, font, int(layout_w))
      tw, th = _measure_multiline_text(wrapped, font, spacing=spacing, draw=draw)
    except Exception:
      continue

    slack_w = (layout_w - float(tw)) / layout_w if layout_w > 0 else 0
    slack_h = (layout_h - float(th)) / layout_h if layout_h > 0 else 0
    actual_slack = min(float(slack_w), float(slack_h))

    if actual_slack >= target_slack:
      final_pt = int(pt)
      break

  return max(int(min_pt), int(final_pt))


def compute_fit_map(
  *,
  translations: Dict[str, str],
  bubbles: Sequence[dict],
  bubble_geometry: Optional[Dict[str, Any]],
  ocr_mask_stats: Optional[Dict[str, Any]],
  config: Dict[str, Any],
  bubble_classes: Optional[Dict[str, Any]] = None,
  image_dpi: Optional[float] = None,
) -> Dict[str, Dict[str, Any]]:
  """
  Compute per-bubble {font_pt, layout_bbox} using:
  - core_bbox from bubble segmentation (fallback to bubble bbox)
  - ocr_bbox + coverage from OCR mask stats (fallback to core_bbox)
  """
  min_pt = int(config.get("min_pt", 16))
  max_pt = int(config.get("max_pt", 34))
  min_pt = max(1, min_pt)
  max_pt = max(min_pt, max_pt)

  pt_step = int(config.get("pt_step", 2))
  if pt_step <= 0:
    pt_step = 2

  pts_all = list(range(int(min_pt), int(max_pt) + 1, int(pt_step)))
  if not pts_all:
    pts_all = [int(min_pt)]

  dpi = image_dpi if image_dpi is not None else config.get("dpi")
  dpi_f = _safe_float(dpi, 72.0)
  if not (dpi_f > 0):
    dpi_f = 72.0
  px_to_pt = 72.0 / float(dpi_f)

  leading_factor = float(config.get("leading_factor", 0.25))
  cap_factor = float(config.get("cap_factor", 1.25))

  geom_items = (bubble_geometry or {}).get("items") if isinstance(bubble_geometry, dict) else None
  if not isinstance(geom_items, dict):
    geom_items = {}
  ocr_items = (ocr_mask_stats or {}).get("items") if isinstance(ocr_mask_stats, dict) else None
  if not isinstance(ocr_items, dict):
    ocr_items = {}

  fallback_path = _pick_font_path(config)
  resolve_cache: Dict[str, Optional[str]] = {}
  font_cache_by_path: Dict[str, Dict[int, ImageFont.FreeTypeFont]] = {}

  def get_font_cache(font_hint: Optional[Dict[str, Any]]) -> Dict[int, ImageFont.FreeTypeFont]:
    path = _resolve_font_path(font_hint, config, fallback_path=fallback_path, cache=resolve_cache) or ""
    if path in font_cache_by_path:
      return font_cache_by_path[path]

    cache: Dict[int, ImageFont.FreeTypeFont] = {}
    if path:
      for pt in pts_all:
        try:
          cache[int(pt)] = ImageFont.truetype(path, int(pt))
        except Exception:
          cache = {}
          break

    if not cache and fallback_path and path != fallback_path:
      for pt in pts_all:
        try:
          cache[int(pt)] = ImageFont.truetype(str(fallback_path), int(pt))
        except Exception:
          cache = {}
          break

    if not cache:
      # Absolute fallback; avoids crashing the pipeline even if no fonts are available.
      default_font = ImageFont.load_default()
      for pt in pts_all:
        cache[int(pt)] = default_font

    font_cache_by_path[path] = cache
    return cache

  dummy = Image.new("RGB", (8, 8))
  draw = ImageDraw.Draw(dummy)

  out: Dict[str, Dict[str, Any]] = {}
  for b in bubbles:
    bid = str(b.get("id") or "")
    if not bid:
      continue
    text = (translations.get(bid) or "").strip()
    if not text:
      continue

    font_hint = _select_font_hint_for_bubble(bid, config=config, bubble_classes=bubble_classes)
    font_cache = get_font_cache(font_hint)

    bubble_bb = _bbox_tuple(b.get("bbox") or {}) or (0.0, 0.0, 1.0, 1.0)
    g = geom_items.get(bid) or {}
    if not isinstance(g, dict):
      g = {}
    core_bb = _bbox_tuple(g.get("core_bbox") or {}) or bubble_bb
    src = str(g.get("source") or "").lower()
    fallback_src = (not src) or src.startswith("bbox_fallback")
    class_info = {}
    if isinstance(bubble_classes, dict):
      maybe_info = bubble_classes.get(bid) or {}
      if isinstance(maybe_info, dict):
        class_info = maybe_info

    bubble_cfg = dict(config)
    bubble_cfg["_layout_profile"] = _layout_profile_for_bubble(
      bubble_bbox=bubble_bb,
      core_bbox=core_bb,
      geom_item=g,
      class_info=class_info,
      config=config,
    )

    o = ocr_items.get(bid) or {}
    cov = _safe_float(o.get("cov", 0.0), 0.0)
    text_word_count = len(_tokenize_ws(text))
    try:
      ocr_line_count = int(o.get("line_count") or 0) if isinstance(o, dict) else 0
    except Exception:
      ocr_line_count = 0

    # In bbox-fallback mode, core bbox can be overly conservative for short lines.
    # Slightly expand toward the bubble bbox to restore usable width.
    core_fit_bb = core_bb
    if fallback_src and (ocr_line_count <= 1 or text_word_count <= 3):
      core_fit_bb = _clamp_bbox(_expand_bbox_about_center(core_bb, scale=1.14), bubble_bb)

    clamp_bb = _comfort_clamp_bbox(core_fit_bb, cov=cov, ocr_item=o if isinstance(o, dict) else {}, geom_item=g if isinstance(g, dict) else {}, config=config)
    clamp_bb = _relax_fallback_clamp_for_short_text(
      clamp_bb,
      core_bbox=core_fit_bb,
      geom_item=g if isinstance(g, dict) else {},
      ocr_item=o if isinstance(o, dict) else {},
      text=text,
    )

    # IMPORTANT: usable area is anchored to the bubble/core geometry, not OCR text bbox.
    # OCR bbox is only a soft hint.
    ocr_bb_raw = _bbox_tuple(o.get("ocr_bbox") or {})
    has_ocr_bbox = ocr_bb_raw is not None
    ocr_bb = ocr_bb_raw or core_fit_bb

    # Base layout from bubble core.
    core_seed_scale = _safe_float(config.get("core_seed_scale", 1.03), 1.03)
    core_seed_scale = max(1.0, min(1.30, float(core_seed_scale)))
    core_seed = _clamp_bbox(_expand_bbox_about_center(core_fit_bb, scale=core_seed_scale), clamp_bb)

    # OCR hint layout (expanded), blended lightly into core seed.
    if cov < 0.08:
      ocr_scale = 1.85
    elif cov < 0.18:
      ocr_scale = 1.55
    else:
      ocr_scale = 1.32
    ocr_seed = _clamp_bbox(_expand_bbox_about_center(ocr_bb, scale=float(ocr_scale)), clamp_bb)

    ocr_influence = _safe_float(config.get("ocr_bbox_influence", 0.30), 0.30)
    if fallback_src:
      ocr_influence = _safe_float(config.get("ocr_bbox_influence_fallback", 0.16), 0.16)
    if text_word_count <= 3:
      ocr_influence *= 0.55
    if not has_ocr_bbox:
      ocr_influence = 0.0
    ocr_influence = max(0.0, min(0.50, float(ocr_influence)))

    layout = _clamp_bbox(_blend_bbox(core_seed, ocr_seed, ocr_influence), clamp_bb)
    cur = layout
    chosen_layout = layout

    best_pt: Optional[int] = None
    best_text: Optional[str] = None
    for _attempt in range(5):
      bw_pt = int(max(1.0, (cur[2] - cur[0]) * px_to_pt))
      bh_pt = int(max(1.0, (cur[3] - cur[1]) * px_to_pt))
      cand_pt, cand_text = _max_fitting_size(
        text,
        font_cache,
        bw_pt,
        bh_pt,
        draw=draw,
        leading_factor=leading_factor,
        pts=pts_all,
        config=bubble_cfg,
      )
      if cand_pt is not None:
        best_pt = int(cand_pt)
        best_text = cand_text
        chosen_layout = cur
        break
      cur = _clamp_bbox(_expand_bbox_about_center(cur, scale=1.15), clamp_bb)

    if best_pt is None:
      best_pt = int(min_pt)
      chosen_layout = clamp_bb
      best_text = None

    # Extra "comfort" step: even when text fits, avoid hugging the inner box border.
    # This tends to reduce font size by ~4–6pt in many bubbles where the max-fitting size looks too tight.
    ccfg = config.get("comfort") if isinstance(config, dict) else None
    if ccfg is True:
      ccfg = {"enabled": True}
    if not isinstance(ccfg, dict):
      ccfg = {}

    if bool(ccfg.get("enabled", True)):
      target = _safe_float(ccfg.get("target_slack", 0.06), 0.06)
      shape_tol_ratio = _safe_float(ccfg.get("shape_tolerance_ratio", 0.04), 0.04)
      try:
        shape_relaxed_hard = int(ccfg.get("shape_relaxed_hard", 1))
      except Exception:
        shape_relaxed_hard = 1
      try:
        max_reduce = int(ccfg.get("max_reduce_pt", 4))
      except Exception:
        max_reduce = 4
      if max_reduce < 0:
        max_reduce = 0
      shape_tol_ratio = max(0.01, min(0.12, float(shape_tol_ratio)))
      shape_relaxed_hard = max(0, min(3, int(shape_relaxed_hard)))
      # Snap to step (2pt by default).
      max_reduce = int((max_reduce // pt_step) * pt_step) if pt_step > 0 else int(max_reduce)

      if cov < 0.08:
        target += 0.02
      elif cov < 0.18:
        target += 0.01
      elif cov < 0.30:
        target += 0.005

      lc = int(ocr_line_count)
      if lc >= 4:
        target -= 0.01
      elif lc <= 1:
        target -= 0.015

      if fallback_src:
        target -= 0.015

      # Short phrases can look much better with a larger optical size.
      if text_word_count <= 3:
        target -= 0.01
      elif text_word_count <= 6:
        target -= 0.005

      target = max(0.03, min(0.14, float(target)))

      bw_pt_fit = int(max(1.0, (chosen_layout[2] - chosen_layout[0]) * px_to_pt))
      bh_pt_fit = int(max(1.0, (chosen_layout[3] - chosen_layout[1]) * px_to_pt))

      # Only reduce within a bounded range to avoid making tight bubbles too small.
      min_allowed = int(max(min_pt, best_pt - max_reduce))
      pts_desc = [p for p in pts_all if int(min_allowed) <= int(p) <= int(best_pt)]
      pts_desc.sort(reverse=True)

      if max_reduce > 0 and int(best_pt) > int(min_pt):
        strict_choice: Optional[Tuple[int, str]] = None
        relaxed_choice: Optional[Tuple[int, str]] = None
        fallback_choice: Optional[Tuple[int, str]] = None
        fallback_rank: Optional[Tuple[int, float, int]] = None

        for pt in pts_desc:
          font = font_cache.get(int(pt))
          if font is None:
            continue
          ok, measure_text, out_text = _fit_at_size(
            text,
            font,
            int(pt),
            bw_pt_fit,
            bh_pt_fit,
            draw=draw,
            leading_factor=leading_factor,
            config=bubble_cfg,
          )
          if not ok:
            continue
          spacing = int(round(float(pt) * float(leading_factor)))
          tw, th = _measure_multiline_text(measure_text, font, spacing=spacing, draw=draw)
          slack_w = (float(bw_pt_fit) - float(tw)) / float(bw_pt_fit) if bw_pt_fit > 0 else 0.0
          slack_h = (float(bh_pt_fit) - float(th)) / float(bh_pt_fit) if bh_pt_fit > 0 else 0.0
          slack = min(float(slack_w), float(slack_h))
          layout_lines = [ln for ln in (out_text or "").split("\n") if (ln or "").strip()]
          line_widths = [max(0.0, _text_width((ln or "").rstrip(), font)) for ln in layout_lines]
          shape_stats = _shape_monotonicity_stats(line_widths, tol_ratio=float(shape_tol_ratio))
          hard = int(shape_stats.get("hard", 0.0))

          badness = _layout_badness_score(out_text)
          hyphen_breaks = 0
          for ln in (out_text or "").split("\n")[:-1]:
            s = (ln or "").strip()
            if s.endswith("-") and s.count("-") < 2:
              hyphen_breaks += 1
          soft = float(shape_stats.get("soft", 0.0))
          quality = (badness * 0.25) + (soft * 2.8) + (float(hyphen_breaks) * 0.35)

          cand = (int(pt), out_text)
          cand_rank = (int(hard), float(quality), -int(pt))
          if fallback_rank is None or cand_rank < fallback_rank:
            fallback_rank = cand_rank
            fallback_choice = cand

          if slack < float(target):
            continue

          if hard <= 0:
            strict_choice = cand
            break

          if relaxed_choice is None and hard <= int(shape_relaxed_hard):
            relaxed_choice = cand

        picked = strict_choice or relaxed_choice or fallback_choice
        if picked is not None:
          best_pt = int(picked[0])
          best_text = picked[1]

    # Naturalness cap based on original OCR estimate (pixel→pt).
    est_px = o.get("font_est_px")
    est_pt = None
    if est_px is not None:
      try:
        est_pt = float(est_px) * px_to_pt
      except Exception:
        est_pt = None
    if est_pt is not None and est_pt > 0:
      cap = int(round(est_pt * cap_factor))
      cap = int(max(min_pt, min(max_pt, cap)))
      # Keep the configured step: snap down to the closest allowed pt.
      cap_pts = [p for p in pts_all if p <= cap]
      cap_pt = max(cap_pts) if cap_pts else int(min_pt)
      best_pt = int(min(best_pt, cap_pt))

    komp = config.get("komponovka") if isinstance(config, dict) else None
    if komp is True:
      komp = {"enabled": True}
    if not isinstance(komp, dict):
      komp = {"enabled": bool(config.get("komponovka_enabled", False))}

    shape_tol_ratio_final = max(0.01, min(0.12, _safe_float(ccfg.get("shape_tolerance_ratio", 0.04), 0.04)))
    try:
      shape_target_hard = int(ccfg.get("shape_target_hard", 0))
    except Exception:
      shape_target_hard = 0
    shape_target_hard = max(0, min(3, int(shape_target_hard)))
    try:
      shape_guard_max_drop_pt = int(ccfg.get("shape_guard_max_drop_pt", 10))
    except Exception:
      shape_guard_max_drop_pt = 10
    shape_guard_max_drop_pt = max(0, int(shape_guard_max_drop_pt))
    shape_guard_enabled = bool(ccfg.get("shape_guard_enabled", True))

    if shape_guard_enabled and bool(komp.get("enabled", False)):
      bw_pt_final = int(max(1.0, (chosen_layout[2] - chosen_layout[0]) * px_to_pt))
      bh_pt_final = int(max(1.0, (chosen_layout[3] - chosen_layout[1]) * px_to_pt))

      strict_pick: Optional[Tuple[int, str]] = None
      relaxed_pick: Optional[Tuple[int, str]] = None
      relaxed_rank: Optional[Tuple[int, float, int]] = None

      guard_floor_pt = int(max(int(min_pt), int(best_pt) - int(shape_guard_max_drop_pt)))
      pts_guard = [p for p in pts_all if guard_floor_pt <= int(p) <= int(best_pt)]
      pts_guard.sort(reverse=True)

      for pt in pts_guard:
        font_guard = font_cache.get(int(pt))
        if font_guard is None:
          continue
        ok_guard, _measure_text_guard, out_text_guard = _fit_at_size(
          text,
          font_guard,
          int(pt),
          bw_pt_final,
          bh_pt_final,
          draw=draw,
          leading_factor=leading_factor,
          config=bubble_cfg,
        )
        if not ok_guard:
          continue

        shape_guard = _layout_shape_stats(out_text_guard, font_guard, tol_ratio=float(shape_tol_ratio_final))
        hard_guard = int(shape_guard.get("hard", 0.0))
        soft_guard = float(shape_guard.get("soft", 0.0))
        line_count_guard = int(shape_guard.get("line_count", 0.0))

        if line_count_guard <= 2:
          strict_pick = (int(pt), out_text_guard)
          break

        rank = (int(hard_guard), float(soft_guard), -int(pt))
        if relaxed_rank is None or rank < relaxed_rank:
          relaxed_rank = rank
          relaxed_pick = (int(pt), out_text_guard)

        if hard_guard <= int(shape_target_hard):
          strict_pick = (int(pt), out_text_guard)
          break

      picked_guard = strict_pick or relaxed_pick
      if picked_guard is not None:
        best_pt = int(picked_guard[0])
        best_text = picked_guard[1]

    final_pt = int(best_pt)
    entry: Dict[str, Any] = {"font_pt": int(final_pt), "layout_bbox": _bbox_dict(chosen_layout)}

    if bool(komp.get("enabled", False)):
      try:
        bw_pt_final = int(max(1.0, (chosen_layout[2] - chosen_layout[0]) * px_to_pt))
        bh_pt_final = int(max(1.0, (chosen_layout[3] - chosen_layout[1]) * px_to_pt))
        font_final = font_cache.get(int(final_pt))
        if font_final is not None:
          composed, _applied = _compose_text_output(
            text,
            font_final,
            max_width=bw_pt_final,
            max_height=bh_pt_final,
            font_pt=int(final_pt),
            leading_factor=float(leading_factor),
            config=bubble_cfg,
          )
          entry["layout_text"] = composed or (best_text or text)
        else:
          entry["layout_text"] = best_text or text
      except Exception:
        entry["layout_text"] = best_text or text

    out[bid] = entry

  return out
