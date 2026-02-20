from __future__ import annotations

import base64
import io
import inspect
import os
import threading
from functools import lru_cache
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image, ImageEnhance, ImageOps


os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "1")
os.environ.setdefault("DISABLE_MODEL_SOURCE_CHECK", "True")

app = FastAPI(title="tiper OCR worker", version="1.0")
app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)


def _decode_image(image_base64: str) -> Image.Image:
  if "," in image_base64 and image_base64.strip().lower().startswith("data:"):
    image_base64 = image_base64.split(",", 1)[1]
  data = base64.b64decode(image_base64, validate=False)
  return Image.open(io.BytesIO(data)).convert("RGB")


def _to_py_list(value):
  if value is None:
    return []
  if isinstance(value, list):
    return value
  try:
    return list(value)
  except Exception:
    return [value]


def _to_py_points(poly) -> List[List[float]]:
  if poly is None:
    return []
  try:
    poly = poly.tolist()
  except Exception:
    pass
  out: List[List[float]] = []
  for p in _to_py_list(poly):
    try:
      x, y = p
      out.append([float(x), float(y)])
    except Exception:
      continue
  return out


def _normalize_paddle_output(out) -> List[Dict[str, Any]]:
  """
  Normalize PaddleOCR outputs into [{"poly": [[x,y]...], "text": str, "conf": float}, ...].

  Supports:
  - PaddleOCR 2.x output: List[List[(poly, (text, conf))]]
  - PaddleOCR 3.x (PaddleX pipeline) output: List[OCRResult] where OCRResult has rec_polys/rec_texts/rec_scores
  """
  results: List[Dict[str, Any]] = []
  if out is None:
    return results

  out_list = _to_py_list(out)
  if not out_list:
    return results

  # PaddleOCR 2.x: list of lines; each line is list of (poly, (text, conf))
  if isinstance(out_list[0], list):
    for line in out_list:
      for item in line or []:
        try:
          poly, (text, conf) = item
        except Exception:
          continue
        results.append({"poly": _to_py_points(poly), "text": str(text or ""), "conf": float(conf or 0.0)})
    return results

  # PaddleOCR 3.x: list of OCRResult(dict-like) objects.
  for page in out_list:
    if not isinstance(page, dict):
      continue
    polys = page.get("rec_polys") or page.get("dt_polys") or []
    texts = page.get("rec_texts") or []
    scores = page.get("rec_scores") or []

    polys_list = _to_py_list(polys)
    texts_list = _to_py_list(texts)
    scores_list = _to_py_list(scores)
    n = min(len(polys_list), len(texts_list)) if polys_list and texts_list else 0
    for i in range(n):
      text = texts_list[i]
      if isinstance(text, (list, tuple)) and text:
        text = text[0]
      conf = 0.0
      if i < len(scores_list):
        try:
          conf = float(scores_list[i])
        except Exception:
          conf = 0.0
      results.append({"poly": _to_py_points(polys_list[i]), "text": str(text or ""), "conf": conf})

  return results


_PADDLE_DEVICE: Optional[str] = None
_VL_LOCK = threading.Lock()


def _ensure_paddle_device() -> str:
  """
  Ensure Paddle is set to the right device.

  Env: TIPER_OCR_DEVICE = auto | cpu | gpu | gpu:0 | cuda:0 ...
  """
  global _PADDLE_DEVICE  # noqa: PLW0603 - process-wide Paddle device

  if _PADDLE_DEVICE:
    return _PADDLE_DEVICE

  try:
    import paddle  # type: ignore
  except Exception:
    _PADDLE_DEVICE = "unavailable"
    return _PADDLE_DEVICE

  desired_raw = (os.environ.get("TIPER_OCR_DEVICE") or "auto").strip().lower()
  desired = desired_raw or "auto"

  def _try_set(dev: str) -> bool:
    try:
      paddle.set_device(dev)
      return True
    except Exception:
      return False

  # Normalize common CUDA-style values.
  if desired.startswith("cuda"):
    desired = "gpu" + desired[4:]

  if desired in ("auto",):
    if paddle.is_compiled_with_cuda():
      try:
        if paddle.device.cuda.device_count() > 0 and _try_set("gpu"):
          pass
        else:
          _try_set("cpu")
      except Exception:
        _try_set("cpu")
    else:
      _try_set("cpu")
  elif desired in ("cpu",):
    _try_set("cpu")
  elif desired.startswith("gpu"):
    if not _try_set(desired):
      _try_set("cpu")
  else:
    if not _try_set(desired):
      _try_set("cpu")

  try:
    _PADDLE_DEVICE = str(paddle.device.get_device())
  except Exception:
    _PADDLE_DEVICE = desired_raw
  return _PADDLE_DEVICE


def _paddle_status() -> Dict[str, Any]:
  try:
    import paddle  # type: ignore

    device = _ensure_paddle_device()
    status: Dict[str, Any] = {
      "version": getattr(paddle, "__version__", None),
      "compiled_with_cuda": bool(paddle.is_compiled_with_cuda()),
      "device": device,
    }
    try:
      status["cuda_device_count"] = int(paddle.device.cuda.device_count())
    except Exception:
      status["cuda_device_count"] = None
    return status
  except Exception as e:
    return {"error": str(e)}


def _norm_ws(s: str) -> str:
  return " ".join((s or "").split())


def _otsu_threshold(gray_u8) -> int:
  import numpy as np

  gray_u8 = np.asarray(gray_u8, dtype=np.uint8)
  hist = np.bincount(gray_u8.reshape(-1), minlength=256).astype(np.float64)
  total = int(gray_u8.size)
  if total <= 0:
    return 128

  sum_total = float(np.dot(np.arange(256, dtype=np.float64), hist))
  sum_b = 0.0
  w_b = 0.0
  var_max = -1.0
  thr = 128
  for t in range(256):
    w_b += float(hist[t])
    if w_b <= 0:
      continue
    w_f = float(total) - w_b
    if w_f <= 0:
      break
    sum_b += float(t) * float(hist[t])
    m_b = sum_b / w_b
    m_f = (sum_total - sum_b) / w_f
    var_between = w_b * w_f * (m_b - m_f) ** 2
    if var_between > var_max:
      var_max = var_between
      thr = t
  return int(thr)


def _vl_preprocess_variants(img_rgb: Image.Image) -> List[tuple[str, Image.Image]]:
  """
  OCR-style preprocessing variants used as retries ONLY when VL returns empty text.
  """
  variants: List[tuple[str, Image.Image]] = []

  gray = img_rgb.convert("L")

  # 1) Contrast + sharpness (helps low-contrast gray bubbles)
  c2 = ImageEnhance.Contrast(gray).enhance(2.0)
  c2s2 = ImageEnhance.Sharpness(c2).enhance(2.0)
  variants.append(("gray_contrast2_sharp2", c2s2.convert("RGB")))

  # 2) Otsu binarize + invert (helps white text/outline on gray bubble)
  import numpy as np

  arr = np.array(gray, dtype=np.uint8)
  thr = _otsu_threshold(arr)
  bw = (arr > thr).astype(np.uint8) * 255
  bw_img = Image.fromarray(bw, mode="L")
  variants.append(("otsu_bw_invert", ImageOps.invert(bw_img).convert("RGB")))

  # 3) Autocontrast (helps punctuation bubbles like !!!)
  variants.append(("gray_autocontrast", ImageOps.autocontrast(gray).convert("RGB")))

  # 4) Autocontrast + padding (helps some layout edge cases)
  variants.append(
    (
      "pad20_gray_autocontrast",
      ImageOps.expand(ImageOps.autocontrast(gray).convert("RGB"), border=20, fill=(255, 255, 255)),
    )
  )

  return variants


def _vl_blocks_to_results(blocks, width: int, height: int) -> List[Dict[str, Any]]:
  results: List[Dict[str, Any]] = []
  for blk in _to_py_list(blocks):
    bbox = None
    content = None
    label = None

    if isinstance(blk, dict):
      bbox = blk.get("block_bbox") or blk.get("bbox")
      content = blk.get("block_content") or blk.get("content")
      label = blk.get("block_label") or blk.get("label")
    else:
      bbox = getattr(blk, "bbox", None)
      content = getattr(blk, "content", None)
      label = getattr(blk, "label", None)

    text = str(content or "").strip()
    if not text:
      continue

    x1, y1, x2, y2 = 0.0, 0.0, float(width), float(height)
    if bbox is not None:
      try:
        x1, y1, x2, y2 = [float(v) for v in _to_py_list(bbox)[:4]]
      except Exception:
        x1, y1, x2, y2 = 0.0, 0.0, float(width), float(height)

    # Clamp
    x1 = max(0.0, min(float(width), x1))
    y1 = max(0.0, min(float(height), y1))
    x2 = max(0.0, min(float(width), x2))
    y2 = max(0.0, min(float(height), y2))
    if x2 <= x1:
      x2 = min(float(width), x1 + 1.0)
    if y2 <= y1:
      y2 = min(float(height), y1 + 1.0)

    poly = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    results.append({"poly": poly, "text": text, "conf": 1.0, "label": str(label or "")})

  return results


@lru_cache(maxsize=1)
def _get_vl():
  try:
    from paddleocr import PaddleOCRVL  # type: ignore
  except Exception as e:
    raise RuntimeError("paddleocr is not installed") from e

  _ensure_paddle_device()
  return PaddleOCRVL(pipeline_version="v1", use_doc_orientation_classify=False, use_doc_unwarping=False)


def _run_vl(img_rgb: Image.Image) -> List[Dict[str, Any]]:
  import numpy as np

  pipe = _get_vl()
  # PaddleX ReadImage(format="BGR") treats numpy arrays as already-BGR.
  # PIL gives RGB, so convert to BGR to avoid color-channel confusion.
  img_np = np.array(img_rgb.convert("RGB"))[:, :, ::-1].copy()
  with _VL_LOCK:
    out = pipe.predict(img_np)
  if not out:
    return []
  res0 = out[0]
  try:
    blocks = res0.get("parsing_res_list")  # type: ignore[union-attr]
  except Exception:
    blocks = None
  return _vl_blocks_to_results(blocks or [], width=img_rgb.size[0], height=img_rgb.size[1])


@lru_cache(maxsize=8)
def _get_ocr(lang: str):
  try:
    from paddleocr import PaddleOCR  # type: ignore
  except Exception as e:
    raise RuntimeError("paddleocr is not installed") from e

  _ensure_paddle_device()

  kwargs: Dict[str, Any] = {"lang": lang}

  # Our pipeline does OCR on *bubble crops* (not full-page document OCR).
  # Doc-preprocessor modules (orientation/unwarping) can hurt small crops and add latency,
  # so we disable them when the installed PaddleOCR supports these flags.
  try:
    sig = inspect.signature(PaddleOCR.__init__)
    if "use_doc_orientation_classify" in sig.parameters:
      kwargs["use_doc_orientation_classify"] = False
    if "use_doc_unwarping" in sig.parameters:
      kwargs["use_doc_unwarping"] = False

    ocr_version = (os.environ.get("TIPER_PADDLE_OCR_VERSION") or "").strip()
    if ocr_version and "ocr_version" in sig.parameters:
      kwargs["ocr_version"] = ocr_version
  except Exception:
    pass

  try:
    return PaddleOCR(use_textline_orientation=True, **kwargs)
  except Exception:
    # Older PaddleOCR versions.
    kwargs.pop("ocr_version", None)
    kwargs.pop("use_doc_orientation_classify", None)
    kwargs.pop("use_doc_unwarping", None)
    return PaddleOCR(use_angle_cls=True, **kwargs)


class OcrRequest(BaseModel):
  image_base64: str
  lang: Optional[str] = None
  det: bool = True
  rec: bool = True
  cls: bool = True


@app.get("/health")
def health() -> Dict[str, Any]:
  cache = None
  try:
    ci = _get_ocr.cache_info()  # type: ignore[attr-defined]
    cache = {"hits": ci.hits, "misses": ci.misses, "maxsize": ci.maxsize, "currsize": ci.currsize}
  except Exception:
    cache = None
  return {
    "status": "ok",
    "paddle": _paddle_status(),
    "tiper_ocr_device": os.environ.get("TIPER_OCR_DEVICE") or "auto",
    "ocr_cache": cache,
  }


@app.post("/ocr_bubble")
def ocr_bubble(req: OcrRequest) -> Dict[str, Any]:
  """
  Bubble-crop OCR pipeline:
  - PaddleOCRVL first
  - If VL returns empty → retry a few OCR-style preprocessing variants
  - If still empty → fallback to PaddleOCR(lang=korean by default)

  Output is the same shape as /ocr: {"results": [{"poly":..., "text":..., "conf":...}, ...]}
  """
  try:
    img = _decode_image(req.image_base64)
  except Exception as e:
    raise HTTPException(status_code=400, detail={"error": "invalid_image", "detail": str(e)}) from e

  # 1) VL
  try:
    results = _run_vl(img)
  except Exception:
    results = []

  # 2) Retry on preprocessed variants if empty
  if not any(_norm_ws(r.get("text") or "") for r in results):
    for _name, vimg in _vl_preprocess_variants(img):
      try:
        results = _run_vl(vimg)
      except Exception:
        results = []
      if any(_norm_ws(r.get("text") or "") for r in results):
        break

  # 3) Fallback PaddleOCR
  if not any(_norm_ws(r.get("text") or "") for r in results):
    lang = (req.lang or "korean").strip().lower()
    try:
      ocr_engine = _get_ocr(lang)
    except Exception as e:
      raise HTTPException(status_code=500, detail={"error": "ocr_init_failed", "detail": str(e)}) from e

    try:
      import numpy as np

      img_np = np.array(img)
      if hasattr(ocr_engine, "predict"):
        out = ocr_engine.predict(img_np, use_textline_orientation=bool(req.cls))
      else:
        out = ocr_engine.ocr(img_np, cls=bool(req.cls))
    except Exception as e:
      raise HTTPException(status_code=500, detail={"error": "ocr_failed", "detail": str(e)}) from e

    results = _normalize_paddle_output(out)

  return {"results": results}


@app.post("/ocr")
def ocr(req: OcrRequest) -> Dict[str, Any]:
  try:
    img = _decode_image(req.image_base64)
  except Exception as e:
    raise HTTPException(status_code=400, detail={"error": "invalid_image", "detail": str(e)}) from e

  lang = (req.lang or "korean").strip().lower()
  try:
    ocr_engine = _get_ocr(lang)
  except Exception as e:
    raise HTTPException(status_code=500, detail={"error": "ocr_init_failed", "detail": str(e)}) from e

  try:
    # PaddleOCR expects numpy BGR or path; PIL RGB is ok via numpy conversion.
    import numpy as np

    img_np = np.array(img)
    if hasattr(ocr_engine, "predict"):
      out = ocr_engine.predict(img_np, use_textline_orientation=bool(req.cls))
    else:
      out = ocr_engine.ocr(img_np, cls=bool(req.cls))
  except Exception as e:
    raise HTTPException(status_code=500, detail={"error": "ocr_failed", "detail": str(e)}) from e

  return {"results": _normalize_paddle_output(out)}
