from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

try:
  import requests
except Exception:  # pragma: no cover
  requests = None  # type: ignore


OcrResult = Tuple[List[List[float]], str, float]


class OcrBackend:
  def readtext(self, image: Image.Image, lang: Optional[str] = None) -> List[OcrResult]:
    raise NotImplementedError


@dataclass
class MultiLangOcrConfig:
  languages: List[str]
  ignore_errors: bool = True


class MultiLangOcrBackend(OcrBackend):
  """
  Wrapper that runs the same backend multiple times with different lang values
  and concatenates the results.

  Intended for PaddleOCR where "lang" selects a model (korean/en/th/...).
  """

  def __init__(self, backend: OcrBackend, cfg: MultiLangOcrConfig):
    self._backend = backend
    self._cfg = cfg

  def readtext(self, image: Image.Image, lang: Optional[str] = None) -> List[OcrResult]:
    if lang is not None:
      return self._backend.readtext(image, lang=lang)

    langs: List[str] = []
    for l in (self._cfg.languages or []):
      s = str(l or "").strip()
      if s and s not in langs:
        langs.append(s)

    if not langs:
      return self._backend.readtext(image, lang=None)

    out: List[OcrResult] = []
    for l in langs:
      try:
        out.extend(self._backend.readtext(image, lang=l))
      except Exception:
        if not bool(self._cfg.ignore_errors):
          raise
        continue
    return out


@dataclass
class EasyOcrConfig:
  languages: List[str]
  gpu: bool = True
  decoder: str = "beamsearch"
  beam_width: int = 10


class EasyOcrBackend(OcrBackend):
  def __init__(self, cfg: EasyOcrConfig):
    try:
      import easyocr  # type: ignore
    except Exception as e:  # pragma: no cover
      raise RuntimeError("easyocr is not installed") from e

    self._cfg = cfg
    self._reader = easyocr.Reader(cfg.languages, gpu=bool(cfg.gpu))

  def readtext(self, image: Image.Image, lang: Optional[str] = None) -> List[OcrResult]:
    img_np = np.array(image.convert("RGB"))
    results = self._reader.readtext(
      img_np,
      decoder=self._cfg.decoder,
      beamWidth=int(self._cfg.beam_width),
      detail=1,
      paragraph=False,
    )
    out: List[OcrResult] = []
    for bbox, text, prob in results:
      out.append((bbox, text, float(prob)))
    return out


@dataclass
class RemoteOcrConfig:
  url: str
  endpoint: str = "/ocr"
  timeout_s: int = 120


class RemoteOcrBackend(OcrBackend):
  def __init__(self, cfg: RemoteOcrConfig):
    if requests is None:  # pragma: no cover
      raise RuntimeError("requests is not installed")
    self._cfg = cfg

  def readtext(self, image: Image.Image, lang: Optional[str] = None) -> List[OcrResult]:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    payload: Dict[str, Any] = {
      "image_base64": img_b64,
      "lang": lang,
      "det": True,
      "rec": True,
      "cls": True,
    }
    endpoint = str(self._cfg.endpoint or "/ocr")
    if not endpoint.startswith("/"):
      endpoint = "/" + endpoint
    resp = requests.post(self._cfg.url.rstrip("/") + endpoint, json=payload, timeout=int(self._cfg.timeout_s))
    resp.raise_for_status()
    data = resp.json()
    results = data.get("results") or []

    out: List[OcrResult] = []
    for item in results:
      poly = item.get("poly") or item.get("bbox")
      text = item.get("text") or ""
      conf = float(item.get("conf") or item.get("score") or 0.0)
      if not poly:
        continue
      out.append((poly, text, conf))
    return out
