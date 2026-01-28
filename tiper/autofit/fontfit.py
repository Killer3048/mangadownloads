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
  def __init__(self, text: str, can_break: bool, penalty: float, join_char: str):
    self.text = text
    self.can_break = can_break
    self.penalty = penalty
    self.join_char = join_char # Character to effectively add if we break here (e.g. "-")

def _tokenize_atomic(text: str, lang: str) -> List[AtomicToken]:
  """
  Splits text into atomic units (syllables or words).
  Returns sequence of tokens.
  Logic:
   1. Split by spaces -> words.
   2. For each word:
       - If break allowed inside (syllables): split into syllables.
       - create tokens: 
          syl1 (can_break=True, pen=HYPHEN_PEN, join_char="-")
          syl2 ...
          last_syl (can_break=True, pen=SPACE_PEN, join_char="") IF followed by space
  """
  raw_words = _tokenize_ws(text)
  out: List[AtomicToken] = []
  
  import pyphen
  dic = None
  try:
      py_lang = "ru_RU" if lang == "ru" else ("en_US" if lang == "en" else None)
      if py_lang: dic = pyphen.Pyphen(lang=py_lang)
  except: pass

  for i, word in enumerate(raw_words):
      # Is this the last word?
      is_last_word = (i == len(raw_words) - 1)
      
      # Clean punct
      # (Simplified for brevity, usually we separate punct as own token if we want to break before it? 
      #  But usually we break AFTER punct attached to word.)
      
      syllables = []
      if dic and len(word) > 4:
           inserted = dic.inserted(word)
           if "-" in inserted:
               syllables = inserted.split("-")
      
      if not syllables:
           syllables = [word]
           
      for j, syl in enumerate(syllables):
          is_last_syl = (j == len(syllables) - 1)
          
          # Default: join with next syllable in same word -> empty join, NO break (unless we mark it)
          # Wait, if we split strict syllables, we WANT to allow break.
          
          if is_last_syl:
              # End of word.
              # Break allowed? Yes, usually.
              # Join char? Empty (newline) if break, else SPACE if not break (to join with next word)
              # WAIT. DP reconstructs lines. 
              # If we put "word1" on line 1, and "word2" on line 1 -> "word1 word2".
              # So if we don't break between word1 and word2, we need a space.
              # If we DO break, we don't need a space (newline).
              
              # Token property: visual_text = self.text
              # If valid break here: line ends with visual_text + self.join_char_if_broken (usually implicit?)
              # If NO break here: line continues with visual_text + self.join_char_if_connected
              
              # Let's simplify: AtomicToken just holds text. Separation is handled by DP state?
              # No, "hyphen" is content added ONLY on break. "Space" is content added ONLY on continue.
              
              # Model:
              # Token | Break Action | Continue Action
              # "po"  | add "-"      | add ""
              # "et"  | add "-"      | add ""
              # "mu"  | add ""       | add " " (space before next word)
              
              join_char_break = "" 
              join_char_cont = " " if not is_last_word else ""
              
              penalty = 0.0
              can_break = True
              
              # Check unbreakable with next word logic (orphan particles etc) later?
              # For now assume mostly yes.
              
              out.append(AtomicToken(syl, True, penalty, join_break="", join_cont=" " if not is_last_word else ""))
          else:
              # Mid-word syllable
              out.append(AtomicToken(syl, True, 10.0, join_break="-", join_cont=""))
              
  return out


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

def _tokenize_atomic(text: str, lang: str) -> List[AtomicToken]:
  """
  Splits text into atomic units (syllables or words).
  """
  raw_words = _tokenize_ws(text)
  out: List[AtomicToken] = []
  
  import pyphen
  dic = None
  try:
      py_lang = "ru_RU" if lang == "ru" else ("en_US" if lang == "en" else None)
      if py_lang: dic = pyphen.Pyphen(lang=py_lang)
  except: pass

  for i, word in enumerate(raw_words):
      is_last_word = (i == len(raw_words) - 1)
      
      # Naive cleanup for syllables (strip punct) could be better,
      # but pyphen handles punct okay usually. 
      # Better: strip punct, split core, reattach punct to last syllable?
      # For now, let's just try feeding word.
      
      segments = _split_word_preserving_hyphens(word)
      if not segments:
        continue

      for seg_text, join_after in segments:
        syllables = []
        if dic and len(seg_text) > 4:
          # pyphen inserter returns "syl-la-ble"
          inserted = dic.inserted(seg_text)
          if "-" in inserted:
            syllables = inserted.split("-")

        if not syllables:
          syllables = [seg_text]

        for j, syl in enumerate(syllables):
          is_last_syl = (j == len(syllables) - 1)

          if is_last_syl:
            if join_after:
              # Explicit hyphen in the original text: preserve it both on continue and on break.
              out.append(AtomicToken(syl, True, 10.0, join_char="", join_break=join_after, join_cont=join_after))
            else:
              # End of word.
              # If break: implies newline (no extra char).
              # If continue: need SPACE (unless last word of para).
              join_cont = " " if not is_last_word else ""
              out.append(AtomicToken(syl, True, 0.0, join_char="", join_break="", join_cont=join_cont))
          else:
            # Mid-word syllable.
            # If break: add HYPHEN. If continue: add nothing (join parts).
            out.append(AtomicToken(syl, True, 10.0, join_char="", join_break="-", join_cont=""))
              
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

  base_frac = _safe_float(ccfg.get("base_frac", 0.12), 0.12)
  min_frac = _safe_float(ccfg.get("min_frac", 0.06), 0.06)
  max_frac = _safe_float(ccfg.get("max_frac", 0.22), 0.22)
  # Extra shrink when we don't have a true contour-based core bbox (bbox fallback).
  # Empirically many bubbles then produce text that visually hugs the contour, so we
  # shrink a bit more by default.
  fallback_bonus = _safe_float(ccfg.get("fallback_bonus", 0.10), 0.10)
  min_px = int(ccfg.get("min_px", 4))
  max_px = int(ccfg.get("max_px", 60))

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

  # More margin when we have no segmentation core (bbox fallback tends to include tails).
  src = str(geom_item.get("source") or "").lower()
  if (not src) or src.startswith("bbox_fallback"):
    frac += float(fallback_bonus)

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
              if 2 <= pos <= len(w) - 2:
                idxs.append(pos)
            return idxs
      except Exception:
        pass

      # 2) RU fallback: split after vowels (very rough).
      if lang == "ru":
        vowels = set("аеёиоуыэюяАЕЁИОУЫЭЮЯ")
        idxs = []
        for i in range(2, len(w) - 2):
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
        cut = max(2, min(len(remainder) - 2, int(len(remainder) * 0.5)))
        while cut > 2 and _text_width(remainder[:cut] + "-", font) > max_width:
          cut -= 1
        best_cut = max(2, cut)

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


def _get_shape_factor(line_idx: int, total_lines: int) -> float:
  """
  Returns a width factor (0.0 to 1.0) for a given line index in an oval/diamond profile.
  """
  if total_lines <= 1:
    return 1.0
  
  # Normalized Y center of the line.
  # 0 to 1 range.
  y_rel = (float(line_idx) + 0.5) / float(total_lines)
  
  # Map to -1 to 1
  y_norm = (y_rel - 0.5) * 2.0
  
  # Ellipse equation: x^2 + y^2 = 1 => x = sqrt(1 - y^2)
  # We clamp the curve so it doesn't go to 0 width.
  # Let's say min width is 65% at the tips.
  # effective y range: if we use full -1..1, we get 0 width.
  # let's assume the text box is inscribed in approx 85% of the height of the ellipse?
  # or simply: width = max_width * sqrt(1 - (alpha * y)^2)
  alpha = 0.75 # at y=1, alpha*y = 0.75, width = sqrt(1 - 0.5625) = 0.66
  
  val = 1.0 - (alpha * y_norm) ** 2
  if val < 0: return 0.5 # Should not happen with alpha < 1
  return math.sqrt(val)


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
  max_lines: int,
  lang: str,
) -> Tuple[float, List[str]]:
  n = len(tokens)
  num_lines = len(widths)
  if num_lines == 0:
    return 1e18, []

  INF = 1e18
  dp = [[INF] * (n + 1) for _ in range(num_lines + 1)]
  prev = [[None] * (n + 1) for _ in range(num_lines + 1)]
  dp[0][0] = 0.0
  
  for k in range(num_lines):
    max_w = widths[k]
    for i in range(n):
      cost_so_far = dp[k][i]
      if cost_so_far >= INF:
        continue
      
      # Fill line k+1 starting with token i
      current_line_text = ""
      
      for j in range(i, n):
        # Add token j to line
        chunk = tokens[j].text
        sep = ""
        if j > i:
             sep = tokens[j-1].join_cont
        
        # Tentative line text if we include j
        seg_text = (current_line_text + sep + chunk) if j > i else chunk
        
        # Case A: Break after j
        width_if_break = _text_width(seg_text + tokens[j].join_break, font)
        
        # Case B: Continue (no break needed yet, just check content)
        width_content = _text_width(seg_text, font)
        
        if width_content > max_w:
             # Stop expanding, this token makes it too big even without hyphen
             # Exception: if it's the ONLY token on line, we must accept it (overflow)
             if j == i:
                 pass 
             else:
                 break
                 
        current_line_text = seg_text
        
        # Evaluate break after j
        if not tokens[j].can_break:
            continue
            
        # Check fit with hyphen
        if width_if_break > max_w and j > i:
            # Can't break here because hyphen makes it overflow
            continue
            
        slack = max(0.0, float(max_w) - width_if_break)
        
        # Cost
        is_last_overall = (k == num_lines - 1) and (j == n - 1)
        ragged_weight = 10.0
        ragged = (slack / max(1.0, float(max_w))) ** 2 * ragged_weight
        
        # Penalties
        pen = tokens[j].penalty
        
        line_cost = ragged + pen
        
        if cost_so_far + line_cost < dp[k+1][j+1]:
            dp[k+1][j+1] = cost_so_far + line_cost
            prev[k+1][j+1] = i
            
  final_cost = dp[num_lines][n]
  if final_cost >= INF:
      return INF, []
      
  lines_rev = []
  cur = n
  for k in range(num_lines, 0, -1):
      start = prev[k][cur]
      if start is None: return INF, []
      
      seg_str = ""
      for idx in range(start, cur):
          if idx > start:
              seg_str += tokens[idx-1].join_cont
          seg_str += tokens[idx].text
          
      # Add break char if not last line overall (approx logic)
      if cur < n:
          seg_str += tokens[cur-1].join_break
          
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
) -> str:
  tokens = _tokenize_atomic(para, lang=lang)
  if not tokens:
    return ""
    
  n = len(tokens)
  
  # Heuristics for search
  total_text_len = sum(len(t.text) for t in tokens) # approx char length
  avg_w = _text_width("x", font)
  total_px = total_text_len * avg_w
  min_lines = max(1, int(total_px / float(max_width)))
  
  best_cost = 1e18
  best_lines = []
  
  start_L = max(1, min_lines)
  
  for L in range(start_L, max_lines + 1):
      widths = []
      for i in range(L):
          f = _get_shape_factor(i, L)
          widths.append(float(max_width) * f)
          
      cost, lines = _solve_dp_fixed_lines(tokens, widths, font, max_lines, lang)
      if cost < best_cost:
          best_cost = cost
          best_lines = lines
          
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
  max_reduction: int = 8,
) -> int:
  """
  Compute adaptive font size reduction based on:
  1. shape_factor - ratio of core_bbox area to bubble_bbox area
  2. Actual slack at each candidate font size
  
  Returns:
    final_pt: Final font size after adaptive reduction
  """
  # Get adaptive slack config
  slack_cfg = config.get("adaptive_slack", {})
  if not isinstance(slack_cfg, dict):
    slack_cfg = {}
  
  # If adaptive fit is disabled, use legacy -4pt
  if not config.get("use_adaptive_fit", True):
    return max(min_pt, int(best_pt) - 4)
  
  # 1. Calculate shape_factor
  core_w = max(1.0, float(core_bbox[2] - core_bbox[0]))
  core_h = max(1.0, float(core_bbox[3] - core_bbox[1]))
  bubble_w = max(1.0, float(bubble_bbox[2] - bubble_bbox[0]))
  bubble_h = max(1.0, float(bubble_bbox[3] - bubble_bbox[1]))
  core_area = core_w * core_h
  bubble_area = bubble_w * bubble_h
  shape_factor = core_area / bubble_area if bubble_area > 0 else 0.5
  
  # 2. Determine target safety_slack based on shape
  elliptical_thr = float(slack_cfg.get("elliptical_threshold", 0.4))
  moderate_thr = float(slack_cfg.get("moderate_threshold", 0.6))
  
  if shape_factor < elliptical_thr:
    # Elliptical/complex shapes - need more margin
    target_slack = float(slack_cfg.get("elliptical_slack", 0.15))
  elif shape_factor < moderate_thr:
    # Moderate shapes
    target_slack = float(slack_cfg.get("moderate_slack", 0.12))
  else:
    # Rectangular shapes - less margin needed
    target_slack = float(slack_cfg.get("rectangular_slack", 0.10))
  
  # 3. Get iteration parameters
  max_reduce = int(slack_cfg.get("max_reduction_pt", max_reduction))
  pt_step = int(slack_cfg.get("pt_step", 2))
  if pt_step <= 0:
    pt_step = 2
  
  layout_w = float(layout_bbox[2] - layout_bbox[0])
  layout_h = float(layout_bbox[3] - layout_bbox[1])
  
  # 4. Iterate to find pt with sufficient slack
  final_pt = best_pt
  for pt in range(int(best_pt), max(int(min_pt), int(best_pt) - max_reduce - 1), -pt_step):
    font = font_cache.get(int(pt))
    if font is None:
      continue
    
    # Wrap and measure text at this pt
    try:
      spacing = int(round(float(pt) * float(leading_factor)))
      wrapped = _wrap_text_komponovka(text, font, int(layout_w), draw, config)
      tw, th = _measure_multiline_text(wrapped, font, spacing=spacing, draw=draw)
    except Exception:
      continue
    
    # Calculate slack
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
    core_bb = _bbox_tuple(g.get("core_bbox") or {}) or bubble_bb

    o = ocr_items.get(bid) or {}
    ocr_bb = _bbox_tuple(o.get("ocr_bbox") or {}) or core_bb
    cov = _safe_float(o.get("cov", 0.0), 0.0)
    if cov < 0.08:
      scale = 1.60
    elif cov < 0.18:
      scale = 1.35
    else:
      scale = 1.20

    clamp_bb = _comfort_clamp_bbox(core_bb, cov=cov, ocr_item=o if isinstance(o, dict) else {}, geom_item=g if isinstance(g, dict) else {}, config=config)
    layout = _clamp_bbox(_expand_bbox_about_center(ocr_bb, scale=scale), clamp_bb)
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
        config=config,
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
      target = _safe_float(ccfg.get("target_slack", 0.08), 0.08)
      try:
        max_reduce = int(ccfg.get("max_reduce_pt", 6))
      except Exception:
        max_reduce = 6
      if max_reduce < 0:
        max_reduce = 0
      # Snap to step (2pt by default).
      max_reduce = int((max_reduce // pt_step) * pt_step) if pt_step > 0 else int(max_reduce)

      if cov < 0.08:
        target += 0.03
      elif cov < 0.18:
        target += 0.02
      elif cov < 0.30:
        target += 0.01

      try:
        lc = int(o.get("line_count") or 0) if isinstance(o, dict) else 0
      except Exception:
        lc = 0
      if lc >= 4:
        target -= 0.01

      target = max(0.04, min(0.15, float(target)))

      bw_pt_fit = int(max(1.0, (chosen_layout[2] - chosen_layout[0]) * px_to_pt))
      bh_pt_fit = int(max(1.0, (chosen_layout[3] - chosen_layout[1]) * px_to_pt))

      # Only reduce within a bounded range to avoid making tight bubbles too small.
      min_allowed = int(max(min_pt, best_pt - max_reduce))
      pts_desc = [p for p in pts_all if int(min_allowed) <= int(p) <= int(best_pt)]
      pts_desc.sort(reverse=True)

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
          config=config,
        )
        if not ok:
          continue
        spacing = int(round(float(pt) * float(leading_factor)))
        tw, th = _measure_multiline_text(measure_text, font, spacing=spacing, draw=draw)
        slack_w = (float(bw_pt_fit) - float(tw)) / float(bw_pt_fit) if bw_pt_fit > 0 else 0.0
        slack_h = (float(bh_pt_fit) - float(th)) / float(bh_pt_fit) if bh_pt_fit > 0 else 0.0
        slack = min(float(slack_w), float(slack_h))
        if slack >= float(target):
          best_pt = int(pt)
          best_text = out_text
          break

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

    # Apply adaptive font size reduction (replaces fixed -4pt)
    final_pt = _compute_adaptive_reduction(
      best_pt=int(best_pt),
      layout_bbox=chosen_layout,
      core_bbox=core_bb,
      bubble_bbox=bubble_bb,
      text=text,
      font_cache=font_cache,
      draw=draw,
      leading_factor=leading_factor,
      config=config,
      min_pt=int(min_pt),
      max_reduction=8,
    )
    entry: Dict[str, Any] = {"font_pt": int(final_pt), "layout_bbox": _bbox_dict(chosen_layout)}

    komp = config.get("komponovka") if isinstance(config, dict) else None
    if komp is True:
      komp = {"enabled": True}
    if not isinstance(komp, dict):
      komp = {"enabled": bool(config.get("komponovka_enabled", False))}

    if bool(komp.get("enabled", False)):
      try:
        bw_pt_final = int(max(1.0, (chosen_layout[2] - chosen_layout[0]) * px_to_pt))
        bh_pt_final = int(max(1.0, (chosen_layout[3] - chosen_layout[1]) * px_to_pt))
        font_final = font_cache.get(int(best_pt))
        if font_final is not None:
          composed, _applied = _compose_text_output(
            text,
            font_final,
            max_width=bw_pt_final,
            max_height=bh_pt_final,
            font_pt=int(best_pt),
            leading_factor=float(leading_factor),
            config=config,
          )
          entry["layout_text"] = composed or (best_text or text)
        else:
          entry["layout_text"] = best_text or text
      except Exception:
        entry["layout_text"] = best_text or text

    out[bid] = entry

  return out
