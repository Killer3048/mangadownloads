from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Iterable, List, Sequence


@dataclass(frozen=True)
class OcrToken:
  text: str
  x_mid: float
  y_mid: float
  h: float


def tokens_to_text(
  tokens: Sequence[OcrToken],
  line_tol_factor: float = 0.6,
  join_with_space: bool = True,
) -> str:
  if not tokens:
    return ""

  hs = [max(1.0, float(t.h)) for t in tokens]
  med_h = float(median(hs)) if hs else 10.0
  line_tol = med_h * float(line_tol_factor)

  sorted_tokens = sorted(tokens, key=lambda t: (t.y_mid, t.x_mid))
  lines: List[List[OcrToken]] = []
  line_y: List[float] = []

  for token in sorted_tokens:
    placed = False
    for i in range(len(lines)):
      if abs(token.y_mid - line_y[i]) <= line_tol:
        lines[i].append(token)
        # update representative y as mean-ish
        line_y[i] = (line_y[i] * (len(lines[i]) - 1) + token.y_mid) / float(len(lines[i]))
        placed = True
        break
    if not placed:
      lines.append([token])
      line_y.append(token.y_mid)

  out_lines: List[str] = []
  for line in lines:
    line_sorted = sorted(line, key=lambda t: t.x_mid)
    if join_with_space:
      out_lines.append(" ".join([t.text for t in line_sorted if t.text]))
    else:
      out_lines.append("".join([t.text for t in line_sorted if t.text]))

  return "\n".join([l for l in out_lines if l is not None]).strip()

