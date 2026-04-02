from __future__ import annotations

import os
import re
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, List

from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from tiper.bubbles.detect import BubbleDetConfig, compute_smart_slice_boundaries, detect_bubbles_chunked
from tiper.cutting.boxes import detect_text_boxes, detect_yolo_boxes
from tiper.inpaint.pipeline import InpaintConfig, compute_cost_map_and_cuts, inpaint_with_bubbles_sliced, ocr_texts_per_bubble
from tiper.server.ai_bubble import detect_request
from tiper.server.keys import get_xai_api_key, get_xai_config
from tiper.server.jobs import create_job, read_json, set_job_error, set_job_progress, set_job_status, update_job, write_json
from tiper.server.paths import normalize_source_path, wsl_path_to_unc
from tiper.server.schemas import CreateJobRequest, SubmitBubblesRequest, ImagineRequest
from tiper.server.state import SharedResources
from tiper.server.xai_imagine import XaiImagineError, imagine_edit_image, load_xai_imagine_config

app = FastAPI(title="tiper server", version="1.0")
app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)


_TIPER_ROOT = Path(__file__).resolve().parent.parent
_RES = SharedResources(_TIPER_ROOT)


def _job_dir(job_id: str) -> Path:
  job_dir = (_RES.jobs_root / job_id).resolve()
  if not job_dir.exists():
    raise HTTPException(status_code=404, detail="job_not_found")
  return job_dir


def _strip_style_stroke(styles: Any) -> Dict[str, Any]:
  if not isinstance(styles, dict):
    return {}
  cleaned: Dict[str, Any] = {}
  for key, style in styles.items():
    if isinstance(style, dict) and "stroke" in style:
      style = {k: v for k, v in style.items() if k != "stroke"}
    cleaned[key] = style
  return cleaned


def _normalize_quotes(text: str) -> str:
  """
  Convert Russian guillemets («») to straight double quotes ("").
  If there are nested quotes, convert inner „" (lapki) to single quotes ('').
  
  Rules:
  - «text» → "text"
  - «outer „inner" text» → "outer 'inner' text"
  - „text" → 'text' (if used as inner quotes)
  """
  if not text:
    return text
  
  # First pass: convert «» to ""
  text = text.replace("«", "\"").replace("»", "\"")
  
  # Second pass: convert „" (German/Russian low-high quotes) to ''
  # These are typically used for nested quotes
  text = text.replace("„", "'").replace(""", "'")
  
  # Also handle other quote variants that might appear
  text = text.replace(""", "\"")  # Right double quotation mark
  text = text.replace("‹", "'").replace("›", "'")  # Single guillemets
  
  return text


def _safe_filename(name: str) -> str:
  s = (name or "").strip()
  if not s:
    return "item"
  s = re.sub(r"[^a-zA-Z0-9_.-]+", "_", s)
  s = s.strip("._-")
  return s or "item"


def _imagine_dir(job_dir: Path) -> Path:
  return job_dir / "imagine"


def _read_imagine_status(job_dir: Path) -> Dict[str, Any]:
  return read_json(_imagine_dir(job_dir) / "status.json")


def _write_imagine_status(job_dir: Path, payload: Dict[str, Any]) -> None:
  write_json(_imagine_dir(job_dir) / "status.json", payload)


@app.get("/health")
def health() -> Dict[str, Any]:
  with _RES.gpu_lock:
    return _RES.bubble_ai().health()


@app.post("/load")
def load(payload: Dict[str, Any]) -> Dict[str, Any]:
  with _RES.gpu_lock:
    try:
      _RES.bubble_ai().load(model_path=payload.get("model_path"), device=payload.get("device"))
    except FileNotFoundError as e:
      raise HTTPException(status_code=400, detail={"error": "model_not_found", "model_path": str(e)}) from e
    return _RES.bubble_ai().health()


@app.post("/unload")
def unload(payload: Dict[str, Any]) -> Dict[str, Any]:
  with _RES.gpu_lock:
    _RES.bubble_ai().unload()
    return _RES.bubble_ai().health()


@app.post("/detect")
def detect(payload: Dict[str, Any]) -> Dict[str, Any]:
  with _RES.gpu_lock:
    try:
      return detect_request(_RES.bubble_ai(), payload)
    except ValueError as e:
      raise HTTPException(status_code=400, detail={"error": str(e)}) from e
    except FileNotFoundError as e:
      raise HTTPException(status_code=400, detail={"error": "model_not_found", "model_path": str(e)}) from e
    except Exception as e:
      raise HTTPException(status_code=500, detail={"error": "detect_failed", "detail": str(e)}) from e


@app.post("/jobs")
def jobs_create(req: CreateJobRequest) -> Dict[str, Any]:
  raw_path = req.source_png_path
  src = normalize_source_path(raw_path)
  if not src.exists():
    raise HTTPException(status_code=400, detail={"error": "source_not_found", "path": str(src)})

  cfg = dict(_RES.default_config)
  if req.config:
    for k, v in req.config.items():
      if isinstance(cfg.get(k), dict) and isinstance(v, dict):
        merged = dict(cfg[k])  # type: ignore[index]
        merged.update(v)
        cfg[k] = merged
      else:
        cfg[k] = v

  try:
    job = create_job(_RES.jobs_root, src, cfg)
  except Exception as e:
    raise HTTPException(status_code=500, detail={"error": "job_create_failed", "detail": str(e)}) from e
  return {"job": job}


@app.get("/jobs/{job_id}")
def jobs_get(job_id: str) -> Dict[str, Any]:
  job_dir = _job_dir(job_id)
  return {"job": read_json(job_dir / "job.json")}


@app.post("/jobs/{job_id}/detect_bubbles")
def jobs_detect_bubbles(job_id: str) -> Dict[str, Any]:
  job_dir = _job_dir(job_id)
  runtime = _RES.job_runtime(job_id)

  with runtime.lock:
    if runtime.detect_task and runtime.detect_task.is_alive():
      return {"status": "detect_bubbles_running"}

    job = read_json(job_dir / "job.json")
    if job.get("status") in ("await_edit", "await_ocr_clean", "ocr_clean_running", "await_translation", "done") and Path(job["paths"]["bubbles_auto"]).exists():
      return {"status": job.get("status"), "job": job}

    def _work():
      try:
        set_job_status(job_dir, "detect_bubbles_running")

        bcfg_raw = (job.get("config") or {}).get("bubbles") or {}
        slicing_cfg = (job.get("config") or {}).get("slicing") or {}
        boxes_cfg = (slicing_cfg.get("boxes") or {}) if isinstance(slicing_cfg, dict) else {}
        model_path = str(bcfg_raw.get("model_path") or _RES.default_config["bubbles"]["model_path"])
        with _RES.gpu_lock:
          from PIL import Image

          img = Image.open(job_dir / "original.png").convert("RGB")

          boxes = []
          if boxes_cfg.get("enabled", True):
            try:
              slice_model_path = str(boxes_cfg.get("model_path") or "")
              if slice_model_path:
                yolo_model = _RES.slice_yolo(slice_model_path)
                boxes.extend(
                  detect_yolo_boxes(
                    img,
                    yolo_model,
                    chunk_height=int(boxes_cfg.get("chunk_h", 4096)),
                    overlap=int(boxes_cfg.get("overlap", 256)),
                    conf=float(boxes_cfg.get("conf", 0.25)),
                    imgsz=int(boxes_cfg.get("imgsz", 1280)),
                    device=str(boxes_cfg.get("device", "auto")),
                    min_w=int(boxes_cfg.get("min_w", 20)),
                    min_h=int(boxes_cfg.get("min_h", 20)),
                  )
                )
            except Exception:
              boxes = []

            text_cfg = boxes_cfg.get("text") or {}
            if text_cfg.get("enabled", False):
              try:
                langs = list(text_cfg.get("langs") or [])
                readers = _RES.slice_text_readers(langs, device=str(boxes_cfg.get("device", "auto")))
                for reader in readers:
                  boxes.extend(
                    detect_text_boxes(
                      img,
                      reader,
                      chunk_height=int(text_cfg.get("chunk_h", boxes_cfg.get("chunk_h", 4096))),
                      overlap=int(text_cfg.get("overlap", boxes_cfg.get("overlap", 256))),
                      conf=float(text_cfg.get("conf", 0.40)),
                      min_w=int(text_cfg.get("min_w", 20)),
                      min_h=int(text_cfg.get("min_h", 20)),
                    )
                  )
              except Exception:
                pass

            # Release slicing models before loading the bubble detector to save VRAM.
            _RES.unload_slice_models()

          _RES.bubble_ai().load(model_path=model_path, device=bcfg_raw.get("device"))
          model = _RES.bubble_ai()._model  # noqa: SLF001 - shared internal model

          weights = slicing_cfg.get("weights") or {}
          det_cfg = BubbleDetConfig(
            model_conf=float(bcfg_raw.get("conf", _RES.default_config["bubbles"]["conf"])),
            imgsz=int(bcfg_raw.get("imgsz", _RES.default_config["bubbles"]["imgsz"])),
            device=str(bcfg_raw.get("device", _RES.default_config["bubbles"]["device"])),
            chunk_h=int(bcfg_raw.get("chunk_h", _RES.default_config["bubbles"]["chunk_h"])),
            overlap=int(bcfg_raw.get("overlap", _RES.default_config["bubbles"]["overlap"])),
            nms_iou=float(bcfg_raw.get("nms_iou", _RES.default_config["bubbles"]["nms_iou"])),
            smart_chunking=True,
            slice_min_h=int(slicing_cfg.get("min_h", _RES.default_config["slicing"]["min_h"])),
            slice_max_h=int(slicing_cfg.get("max_h", _RES.default_config["slicing"]["max_h"])),
            slice_margin=int(slicing_cfg.get("margin", _RES.default_config["slicing"]["margin"])),
            edge_weight=float(weights.get("edge_score", 1.0)),
            variance_weight=float(weights.get("variance_score", 0.5)),
            gradient_weight=float(weights.get("gradient_score", 0.5)),
            white_space_weight=float(weights.get("white_space_score", 2.0)),
            distance_penalty=float(weights.get("distance_penalty", 0.00001)),
          )

          set_job_progress(job_dir, "detect_chunking", 0, 1)
          cuts = compute_smart_slice_boundaries(img, det_cfg, boxes=boxes)
          set_job_progress(job_dir, "detect_chunking", 1, 1)

          bubbles = detect_bubbles_chunked(
            img,
            model,
            det_cfg,
            cuts=cuts,
            on_progress=lambda d, t: set_job_progress(job_dir, "detect_bubbles", d, t),
          )

        # Sort bubbles in a stable reading order (top→bottom, then left→right).
        # Ultralytics+NMS returns boxes in score order; for UI editing we want spatial order.
        if bubbles:
          heights = []
          items = []
          for b in bubbles:
            bb = b.get("bbox") or {}
            left = float(bb.get("left", 0.0))
            top = float(bb.get("top", 0.0))
            right = float(bb.get("right", left))
            bottom = float(bb.get("bottom", top))
            w = max(0.0, right - left)
            h = max(0.0, bottom - top)
            x_mid = left + (w / 2.0)
            y_mid = top + (h / 2.0)
            heights.append(h)
            items.append((b, x_mid, y_mid))

          heights_sorted = sorted([h for h in heights if h is not None])
          if heights_sorted:
            mid = len(heights_sorted) // 2
            med_h = float(heights_sorted[mid]) if (len(heights_sorted) % 2) else float(heights_sorted[mid - 1] + heights_sorted[mid]) / 2.0
          else:
            med_h = 0.0
          row_tol = max(1.0, med_h * 0.5)

          items.sort(key=lambda t: (t[2], t[1]))
          rows: List[Dict[str, Any]] = []
          for b, x_mid, y_mid in items:
            placed = False
            for row in rows:
              if abs(float(y_mid) - float(row["y"])) <= row_tol:
                row["items"].append((b, x_mid, y_mid))
                n = len(row["items"])
                row["y"] = (float(row["y"]) * (n - 1) + float(y_mid)) / n
                placed = True
                break
            if not placed:
              rows.append({"y": float(y_mid), "items": [(b, x_mid, y_mid)]})

          rows.sort(key=lambda r: float(r["y"]))
          ordered = []
          for row in rows:
            row_items = row.get("items") or []
            row_items.sort(key=lambda t: t[1])
            for b, _x, _y in row_items:
              ordered.append(b)
          bubbles = ordered

        # Save bubbles_auto.json
        bubble_json = {
          "schema_version": 1,
          "image": job.get("image") or {},
          "bubbles": [
            {
              "id": f"AUTO_{i+1:04d}",
              "bbox": b["bbox"],
              "source": "auto",
              "confidence": b.get("confidence"),
            }
            for i, b in enumerate(bubbles)
          ],
        }
        update_job(job_dir, {"status": "await_edit"})
        write_json(job_dir / "bubbles_auto.json", bubble_json)
        set_job_progress(job_dir, "detect_bubbles", 1, 1)
      except Exception as e:
        set_job_error(job_dir, f"detect_bubbles_failed: {e}")
      finally:
        # Release VRAM for slicing + bubble detection models after this phase.
        try:
          _RES.unload_slice_models()
        except Exception:
          pass
        try:
          _RES.bubble_ai().unload()
        except Exception:
          pass

    runtime.detect_task = threading.Thread(target=_work, daemon=True)
    runtime.detect_task.start()

  return {"status": "detect_bubbles_running"}


@app.get("/jobs/{job_id}/bubbles_auto")
def jobs_get_bubbles_auto(job_id: str) -> Dict[str, Any]:
  job_dir = _job_dir(job_id)
  path = job_dir / "bubbles_auto.json"
  if not path.exists():
    raise HTTPException(status_code=404, detail="bubbles_auto_missing")
  return read_json(path)


@app.post("/jobs/{job_id}/bubbles")
def jobs_submit_bubbles(job_id: str, req: SubmitBubblesRequest) -> Dict[str, Any]:
  job_dir = _job_dir(job_id)
  bubbles = []
  for b in req.bubbles:
    if hasattr(b, "model_dump"):  # pydantic v2
      bubbles.append(b.model_dump())
    else:  # pragma: no cover - pydantic v1
      bubbles.append(b.dict())

  data = {
    "schema_version": 1,
    "image": read_json(job_dir / "job.json").get("image") or {},
    "bubbles": bubbles,
  }

  write_json(job_dir / "bubbles_final.json", data)
  update_job(job_dir, {"status": "await_ocr_clean"})
  return {"status": "ok", "count": len(bubbles)}


@app.get("/jobs/{job_id}/bubbles_final")
def jobs_get_bubbles_final(job_id: str) -> Dict[str, Any]:
  job_dir = _job_dir(job_id)
  path = job_dir / "bubbles_final.json"
  if not path.exists():
    raise HTTPException(status_code=404, detail="bubbles_final_missing")
  return read_json(path)


@app.post("/jobs/{job_id}/imagine")
def jobs_imagine_start(job_id: str, req: ImagineRequest) -> Dict[str, Any]:
  job_dir = _job_dir(job_id)
  runtime = _RES.job_runtime(job_id)

  squares: List[Dict[str, Any]] = []
  for s in req.squares or []:
    if hasattr(s, "model_dump"):  # pydantic v2
      squares.append(s.model_dump())
    else:  # pragma: no cover - pydantic v1
      squares.append(s.dict())

  if not squares:
    raise HTTPException(status_code=400, detail="imagine_squares_empty")

  xai_key = get_xai_api_key(_TIPER_ROOT)
  if not xai_key:
    raise HTTPException(
      status_code=400,
      detail={"error": "xai_api_key_missing", "hint": "Fill keys.yaml (xai.api_key) or set XAI_API_KEY env var."},
    )

  xai_cfg_raw = get_xai_config(_TIPER_ROOT)
  default_prompt = str(xai_cfg_raw.get("default_prompt") or "").strip()
  prompt = (
    (req.prompt or "").strip()
    or default_prompt
    or "Remove ONLY text (dialogue inside bubbles, narration/captions, SFX text) and any watermark/logo/page-number/overlay text. Preserve everything else exactly: speech bubbles (outline + fill), panel borders/frames, line art, shading, textures, and background structures. If text overlaps existing art, reconstruct the original art underneath. Do NOT remove or alter bubbles or frames. Do NOT add new elements."
  )

  imagine_dir = _imagine_dir(job_dir)
  if req.reset:
    try:
      shutil.rmtree(imagine_dir)
    except Exception:
      pass

  imagine_dir.mkdir(parents=True, exist_ok=True)
  (imagine_dir / "input").mkdir(parents=True, exist_ok=True)
  (imagine_dir / "output_raw").mkdir(parents=True, exist_ok=True)
  (imagine_dir / "output_fit").mkdir(parents=True, exist_ok=True)

  with runtime.lock:
    if runtime.imagine_task and runtime.imagine_task.is_alive():
      return {"status": "imagine_running", "imagine": _read_imagine_status(job_dir)}

    _write_imagine_status(
      job_dir,
      {
        "schema_version": 1,
        "status": "running",
        "prompt": prompt,
        "progress": {"done": 0, "total": len(squares), "pct": 0},
        "items": [],
        "last_error": None,
      },
    )

    def _work():
      try:
        import time
        from concurrent.futures import ThreadPoolExecutor, as_completed

        from PIL import Image, ImageOps

        cfg = load_xai_imagine_config(xai_cfg_raw, api_key=xai_key)

        max_workers = 1
        try:
          max_workers = int(xai_cfg_raw.get("max_concurrency") or 1)
        except Exception:
          max_workers = 1
        if max_workers < 1:
          max_workers = 1
        if max_workers > 8:
          max_workers = 8

        max_attempts = 1
        try:
          max_attempts = int(xai_cfg_raw.get("max_attempts") or xai_cfg_raw.get("max_tries") or 1)
        except Exception:
          max_attempts = 1
        if max_attempts < 1:
          max_attempts = 1
        if max_attempts > 6:
          max_attempts = 6

        max_429_wait_sec = 300
        try:
          max_429_wait_sec = int(xai_cfg_raw.get("max_429_wait_sec") or 300)
        except Exception:
          max_429_wait_sec = 300
        if max_429_wait_sec < 0:
          max_429_wait_sec = 0
        if max_429_wait_sec > 3600:
          max_429_wait_sec = 3600

        img = Image.open(job_dir / "original.png").convert("RGB")
        doc_w, doc_h = img.size

        items_out: List[Dict[str, Any]] = []
        tasks: List[Dict[str, Any]] = []
        total = len(squares)

        # 1) Pre-crop inputs sequentially (avoids sharing a giant PIL image across threads).
        for idx, s in enumerate(squares):
          bb = (s.get("bbox") or {}) if isinstance(s, dict) else {}
          left = int(float(bb.get("left", 0)))
          top = int(float(bb.get("top", 0)))
          right = int(float(bb.get("right", left)))
          bottom = int(float(bb.get("bottom", top)))

          left = max(0, min(left, doc_w - 1))
          top = max(0, min(top, doc_h - 1))
          right = max(left + 1, min(right, doc_w))
          bottom = max(top + 1, min(bottom, doc_h))

          roi_w = max(1, right - left)
          roi_h = max(1, bottom - top)

          raw_id = str(s.get("id") or f"R{idx+1:04d}")
          safe_id = _safe_filename(raw_id)
          if any(it.get("safe_id") == safe_id for it in items_out):
            safe_id = f"{safe_id}_{idx+1:04d}"

          in_path = imagine_dir / "input" / f"{safe_id}.png"
          crop = img.crop((left, top, right, bottom))
          crop.save(in_path)

          item: Dict[str, Any] = {
            "id": raw_id,
            "safe_id": safe_id,
            "bbox": {"left": left, "top": top, "right": right, "bottom": bottom},
            "status": "pending",
            "error": None,
            "paths": {"input_png": str(in_path)},
            "paths_open": {},
          }
          items_out.append(item)
          tasks.append({"index": idx, "safe_id": safe_id, "in_path": in_path, "roi_w": roi_w, "roi_h": roi_h})

        # Persist initial state (so UI can show items immediately).
        _write_imagine_status(
          job_dir,
          {
            "schema_version": 1,
            "status": "running",
            "prompt": prompt,
            "progress": {"done": 0, "total": total, "pct": 0},
            "items": items_out,
            "last_error": None,
          },
        )
        try:
          write_json(imagine_dir / "patches.json", {"schema_version": 1, "items": items_out})
        except Exception:
          pass

        def _process_one(task: Dict[str, Any]) -> Dict[str, Any]:
          from PIL import Image as PILImage

          idx = int(task["index"])
          safe_id = str(task["safe_id"])
          in_path = Path(task["in_path"])
          roi_w = int(task["roi_w"])
          roi_h = int(task["roi_h"])

          with PILImage.open(in_path) as _im:
            crop_img = _im.convert("RGB")

          attempt = 0
          backoff = 1.0
          backoff_429 = 5.0
          waited_429 = 0.0
          while True:
            attempt += 1
            try:
              out_img = imagine_edit_image(crop_img, prompt=prompt, cfg=cfg)
              break
            except XaiImagineError as e:
              code = getattr(e, "status_code", None)
              if code == 429 and max_429_wait_sec > 0:
                ra = getattr(e, "retry_after_sec", None)
                wait_s = float(ra) if ra is not None else backoff_429
                if wait_s < 1.0:
                  wait_s = 1.0
                if wait_s > 120.0:
                  wait_s = 120.0
                if waited_429 + wait_s > float(max_429_wait_sec):
                  raise
                time.sleep(wait_s)
                waited_429 += wait_s
                backoff_429 = min(120.0, backoff_429 * 2.0)
                continue

              retriable = code in (500, 502, 503, 504)
              if retriable and attempt < max_attempts:
                time.sleep(backoff)
                backoff = min(20.0, backoff * 2.0)
                continue
              raise
            except Exception:
              if attempt < max_attempts:
                time.sleep(backoff)
                backoff = min(20.0, backoff * 2.0)
                continue
              raise

          raw_path = imagine_dir / "output_raw" / f"{safe_id}.png"
          try:
            out_img.convert("RGB").save(raw_path)
          except Exception:
            out_img.save(raw_path)

          fit = ImageOps.fit(out_img.convert("RGB"), (roi_w, roi_h), method=Image.LANCZOS, centering=(0.5, 0.5))
          fit_path = imagine_dir / "output_fit" / f"{safe_id}.png"
          fit.save(fit_path)

          return {
            "index": idx,
            "status": "ok",
            "error": None,
            "paths": {"output_raw_png": str(raw_path), "output_fit_png": str(fit_path)},
            "paths_open": {"output_fit_png": wsl_path_to_unc(fit_path) or str(fit_path), "output_raw_png": wsl_path_to_unc(raw_path) or str(raw_path)},
          }

        # 2) Call xAI (optionally in parallel).
        done = 0
        if max_workers <= 1 or len(tasks) <= 1:
          for task in tasks:
            idx = int(task["index"])
            items_out[idx]["status"] = "running"
            try:
              result = _process_one(task)
              items_out[idx]["status"] = "ok"
              items_out[idx]["paths"].update(result.get("paths") or {})
              items_out[idx]["paths_open"] = result.get("paths_open") or {}
            except XaiImagineError as e:
              items_out[idx]["status"] = "error"
              items_out[idx]["error"] = str(e)
            except Exception as e:
              items_out[idx]["status"] = "error"
              items_out[idx]["error"] = f"imagine_failed: {e}"

            done += 1
            pct = int(max(0, min(100, round((done / float(total)) * 100)))) if total > 0 else 0
            _write_imagine_status(job_dir, {"schema_version": 1, "status": "running", "prompt": prompt, "progress": {"done": done, "total": total, "pct": pct}, "items": items_out, "last_error": None})
            try:
              write_json(imagine_dir / "patches.json", {"schema_version": 1, "items": items_out})
            except Exception:
              pass
        else:
          with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(_process_one, task): int(task["index"]) for task in tasks}
            for fut in as_completed(futures):
              idx = futures[fut]
              try:
                result = fut.result()
                items_out[idx]["status"] = "ok"
                items_out[idx]["paths"].update(result.get("paths") or {})
                items_out[idx]["paths_open"] = result.get("paths_open") or {}
              except XaiImagineError as e:
                items_out[idx]["status"] = "error"
                items_out[idx]["error"] = str(e)
              except Exception as e:
                items_out[idx]["status"] = "error"
                items_out[idx]["error"] = f"imagine_failed: {e}"

              done += 1
              pct = int(max(0, min(100, round((done / float(total)) * 100)))) if total > 0 else 0
              _write_imagine_status(job_dir, {"schema_version": 1, "status": "running", "prompt": prompt, "progress": {"done": done, "total": total, "pct": pct}, "items": items_out, "last_error": None})
              try:
                write_json(imagine_dir / "patches.json", {"schema_version": 1, "items": items_out})
              except Exception:
                pass

        _write_imagine_status(job_dir, {"schema_version": 1, "status": "done", "prompt": prompt, "progress": {"done": done, "total": total, "pct": 100}, "items": items_out, "last_error": None})
        try:
          write_json(imagine_dir / "patches.json", {"schema_version": 1, "items": items_out})
        except Exception:
          pass
      except Exception as e:
        _write_imagine_status(
          job_dir,
          {
            "schema_version": 1,
            "status": "error",
            "prompt": prompt,
            "progress": {"done": 0, "total": len(squares), "pct": 0},
            "items": [],
            "last_error": str(e),
          },
        )

    runtime.imagine_task = threading.Thread(target=_work, daemon=True)
    runtime.imagine_task.start()

  return {"status": "imagine_running"}


@app.get("/jobs/{job_id}/imagine")
def jobs_imagine_status(job_id: str) -> Dict[str, Any]:
  job_dir = _job_dir(job_id)
  status = _read_imagine_status(job_dir)
  if not status:
    return {"schema_version": 1, "status": "not_started"}
  return status


@app.post("/jobs/{job_id}/run_ocr_clean")
def jobs_run_ocr_clean(job_id: str) -> Dict[str, Any]:
  job_dir = _job_dir(job_id)
  runtime = _RES.job_runtime(job_id)

  with runtime.lock:
    if runtime.ocr_task and runtime.ocr_task.is_alive():
      return {"status": "ocr_clean_running"}

    job = read_json(job_dir / "job.json")
    if job.get("status") in ("await_translation", "done") and Path(job["paths"]["cleaned_png"]).exists():
      return {"status": job.get("status"), "job": job}

    bubbles_data = read_json(job_dir / "bubbles_final.json")
    bubbles = bubbles_data.get("bubbles") or []
    if not bubbles:
      raise HTTPException(status_code=400, detail="bubbles_final_empty")

    def _work():
      try:
        set_job_status(job_dir, "ocr_clean_running")

        ocr_cfg = (job.get("config") or {}).get("ocr") or {}
        bubble_cfg = (job.get("config") or {}).get("bubbles") or {}
        slicing_cfg = (job.get("config") or {}).get("slicing") or {}
        inpaint_cfg = (job.get("config") or {}).get("inpaint") or {}
        autofit_cfg = (job.get("config") or {}).get("autofit") or {}
        if not isinstance(autofit_cfg, dict):
          autofit_cfg = {}

        from PIL import Image

        img = Image.open(job_dir / "original.png").convert("RGB")

        # SmartSlicer cuts are shared across OCR + inpaint.
        weights = slicing_cfg.get("weights") or {}
        cfg_inpaint = InpaintConfig(
          min_h=int(slicing_cfg.get("min_h", _RES.default_config["slicing"]["min_h"])),
          max_h=int(slicing_cfg.get("max_h", _RES.default_config["slicing"]["max_h"])),
          margin=int(slicing_cfg.get("margin", _RES.default_config["slicing"]["margin"])),
          edge_weight=float(weights.get("edge_score", 1.0)),
          variance_weight=float(weights.get("variance_score", 0.5)),
          gradient_weight=float(weights.get("gradient_score", 0.5)),
          white_space_weight=float(weights.get("white_space_score", 2.0)),
          distance_penalty=float(weights.get("distance_penalty", 0.00001)),
        )

        cost_map, cuts = compute_cost_map_and_cuts(img, bubbles, cfg_inpaint)

        # 0) Bubble geometry (core bbox) from full-page segmentation (IoU-matched).
        if autofit_cfg.get("enabled", True):
          try:
            from tiper.autofit.bubble_geometry import compute_bubble_geometry_from_page

            seg_conf = float(autofit_cfg.get("seg_conf", 0.25))
            seg_imgsz = int(autofit_cfg.get("seg_imgsz", bubble_cfg.get("imgsz", _RES.default_config["bubbles"]["imgsz"])))
            seg_iou_thr = float(autofit_cfg.get("seg_iou_thr", 0.30))
            core_frac = float(autofit_cfg.get("core_frac", 0.55))

            model_path = str(bubble_cfg.get("model_path") or _RES.default_config["bubbles"]["model_path"])
            if model_path and not Path(model_path).is_absolute():
              model_path = str((_TIPER_ROOT / model_path).resolve())
            device = str(bubble_cfg.get("device", _RES.default_config["bubbles"]["device"]))

            with _RES.gpu_lock:
              set_job_progress(job_dir, "bubble_geometry", 0, 1)
              _RES.bubble_ai().load(model_path=model_path, device=device)
              model = _RES.bubble_ai()._model  # noqa: SLF001 - shared internal model
              geom = compute_bubble_geometry_from_page(
                img,
                bubbles,
                model,
                conf=seg_conf,
                imgsz=seg_imgsz,
                iou_thr=seg_iou_thr,
                core_frac=core_frac,
              )
              set_job_progress(job_dir, "bubble_geometry", 1, 1)

            write_json(job_dir / "bubble_geometry.json", geom)
          except Exception:
            # Geometry is optional; don't fail the job.
            pass

        # 1) Text extraction per bubble (crop from the full original image, not from slices).
        with _RES.gpu_lock:
          texts = ocr_texts_per_bubble(
            img,
            bubbles,
            _RES.ocr_translation(),
            conf=float(ocr_cfg.get("conf", _RES.default_config["ocr"]["conf"])),
            lang=ocr_cfg.get("lang_hint"),
            on_progress=lambda d, t: set_job_progress(job_dir, "ocr_text", d, t),
          )

        # Write original.txt
        ordered = sorted(bubbles, key=lambda b: b.get("id") or "")
        out_lines: List[str] = []
        for b in ordered:
          bid = b.get("id")
          out_lines.append(f"[{bid}]")
          text_line = (texts.get(bid, "") or "").strip()
          if not text_line:
            text_line = "....."
          out_lines.append(text_line)
          out_lines.append("")
        (job_dir / "original.txt").write_text("\n".join(out_lines).strip() + "\n", encoding="utf-8")

        # 1b) Bubble class classification (based on cropped bubbles).
        bubble_cls_cfg = (job.get("config") or {}).get("bubble_class") or {}
        if not isinstance(bubble_cls_cfg, dict):
          bubble_cls_cfg = {}
        if bubble_cls_cfg.get("enabled", True):
          try:
            from tiper.bubbles.classify import classify_bubbles

            default_cls_cfg = _RES.default_config.get("bubble_class") or {}
            model_path = str(bubble_cls_cfg.get("model_path") or default_cls_cfg.get("model_path") or "")
            if model_path and not Path(model_path).is_absolute():
              model_path = str((_TIPER_ROOT / model_path).resolve())
            cls_conf = float(bubble_cls_cfg.get("conf", default_cls_cfg.get("conf", 0.6)))
            cls_imgsz = bubble_cls_cfg.get("imgsz", default_cls_cfg.get("imgsz"))
            cls_device = str(bubble_cls_cfg.get("device", default_cls_cfg.get("device", "auto")))

            with _RES.gpu_lock:
              cls_model = _RES.bubble_cls(model_path)
              classes, labels = classify_bubbles(
                img,
                bubbles,
                cls_model,
                min_conf=cls_conf,
                imgsz=cls_imgsz,
                device=cls_device,
                on_progress=lambda d, t: set_job_progress(job_dir, "bubble_class", d, t),
              )

            bubble_class_json = {
              "schema_version": 1,
              "image": job.get("image") or {},
              "classes": classes,
            }
            if labels:
              bubble_class_json["labels"] = {str(k): v for k, v in labels.items()}
            write_json(job_dir / "bubble_classes.json", bubble_class_json)
          except Exception:
            # Bubble classification is optional; don't fail the job.
            pass

        # 2) Inpaint using slice OCR + LaMa
        ocr_mask_stats: Dict[str, Any] = {}
        with _RES.gpu_lock:
          sec_cfg = ocr_cfg.get("secondary_pass") if isinstance(ocr_cfg, dict) else None
          sec_enabled = bool(sec_cfg.get("enabled", False)) if isinstance(sec_cfg, dict) else False
          sec_conf = None
          if sec_enabled:
            sec_conf = float(
              sec_cfg.get("mask_conf", (_RES.default_config.get("ocr", {}) or {}).get("secondary_pass", {}).get("mask_conf", 0.0))
            )
          cleaned, cost_map = inpaint_with_bubbles_sliced(
            img,
            bubbles,
            _RES.ocr_mask_inpaint(),
            ocr_conf=float(ocr_cfg.get("mask_conf", _RES.default_config["ocr"].get("mask_conf", 0.0))),
            secondary_ocr_backend=_RES.ocr_mask_inpaint_secondary() if sec_enabled else None,
            secondary_ocr_conf=sec_conf,
            inpainter=_RES.inpainter(),
            cfg=cfg_inpaint,
            cost_map=cost_map,
            cuts=cuts,
            out_ocr_stats=ocr_mask_stats,
            on_progress=lambda d, t: set_job_progress(job_dir, "inpaint", d, t),
          )

        if autofit_cfg.get("enabled", True):
          try:
            write_json(
              job_dir / "ocr_mask_stats.json",
              {
                "schema_version": 1,
                "conf": float(ocr_cfg.get("mask_conf", _RES.default_config["ocr"].get("mask_conf", 0.0))),
                "items": ocr_mask_stats,
              },
            )
          except Exception:
            pass

        tmp_path = None
        try:
          fd, tmp_path = tempfile.mkstemp(prefix="cleaned.", suffix=".png", dir=str(job_dir))
          os.close(fd)
          cleaned.save(tmp_path)
          Path(tmp_path).replace(job_dir / "cleaned.png")
        finally:
          if tmp_path:
            try:
              Path(tmp_path).unlink()
            except Exception:
              pass

        # 3) Extract per-bubble text styles (fill only) using EasyOCR masks.
        style_cfg = (job.get("config") or {}).get("style") or {}
        if style_cfg.get("enabled", True):
          try:
            from tiper.ocr.style import StyleExtractConfig, extract_bubble_styles

            style_conf = float(style_cfg.get("mask_conf", ocr_cfg.get("mask_conf", _RES.default_config["ocr"].get("mask_conf", 0.0))))
            style_opts = StyleExtractConfig(
              conf=style_conf,
              crop_pad=int(style_cfg.get("crop_pad", 2)),
              min_mask_pixels=int(style_cfg.get("min_mask_pixels", 30)),
              min_glyph_pixels=int(style_cfg.get("min_glyph_pixels", 20)),
              min_bg_pixels=int(style_cfg.get("min_bg_pixels", 50)),
              min_stroke_delta=float(style_cfg.get("min_stroke_delta", 8.0)),
              min_stroke_px=float(style_cfg.get("min_stroke_px", 1.0)),
            )

            with _RES.gpu_lock:
              styles = extract_bubble_styles(
                img,
                bubbles,
                _RES.ocr_mask(),
                conf=style_conf,
                lang=ocr_cfg.get("lang_hint"),
                cfg=style_opts,
              )
            styles = _strip_style_stroke(styles)
            write_json(job_dir / "text_styles.json", {"styles": styles})
          except Exception:
            # Style extraction is optional; don't fail the job.
            pass

        update_job(job_dir, {"status": "await_translation"})
        set_job_progress(job_dir, "done", 1, 1)
      except Exception as e:
        set_job_error(job_dir, f"ocr_clean_failed: {e}")
      finally:
        # Release OCR + inpaint models after this phase.
        try:
          _RES.unload_ocr_models()
        except Exception:
          pass
        try:
          _RES.unload_inpainter()
        except Exception:
          pass
        try:
          _RES.unload_bubble_cls()
        except Exception:
          pass

    runtime.ocr_task = threading.Thread(target=_work, daemon=True)
    runtime.ocr_task.start()

  return {"status": "ocr_clean_running"}


_HEADER_RE = re.compile(r"^\s*\[(B\d+)\]\s*$")


@app.post("/jobs/{job_id}/get_translation")
def jobs_get_translation(job_id: str, payload: Dict[str, Any] = Body(default_factory=dict)) -> Dict[str, Any]:
  job_dir = _job_dir(job_id)
  tpath = job_dir / "translate.txt"
  if not tpath.exists():
    raise HTTPException(status_code=404, detail="translate_txt_missing")

  job = read_json(job_dir / "job.json")

  bubbles_data = read_json(job_dir / "bubbles_final.json")
  bubbles = bubbles_data.get("bubbles") or []
  expected_ids = [b.get("id") for b in bubbles if b.get("id")]

  raw_text = tpath.read_text(encoding="utf-8-sig")
  raw_text = re.sub(r"[\u00a0\u200b\u200c\u200d\u2060\ufeff\u202f\u3000]", " ", raw_text)
  raw = raw_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
  translations: Dict[str, str] = {}
  current = None
  buf: List[str] = []
  for line in raw:
    m = _HEADER_RE.match(line)
    if m:
      if current is not None:
        translations[current] = "\n".join(buf).strip()
      current = m.group(1)
      buf = []
      continue
    if current is not None:
      buf.append(line)
  if current is not None:
    translations[current] = "\n".join(buf).strip()

  # Normalize quotes: «» → "", „" → ''
  translations = {bid: _normalize_quotes(text) for bid, text in translations.items()}

  if not translations:
    raise HTTPException(status_code=400, detail="translate_txt_empty")

  missing = [bid for bid in expected_ids if bid not in translations]
  extra = [bid for bid in translations.keys() if bid not in set(expected_ids)]
  if missing or extra:
    raise HTTPException(
      status_code=400,
      detail={
        "error": "translate_ids_mismatch",
        "missing": missing,
        "extra": extra,
      },
    )

  styles_path = job_dir / "text_styles.json"
  styles = {}
  if styles_path.exists():
    try:
      styles = (read_json(styles_path) or {}).get("styles") or {}
    except Exception:
      styles = {}
  styles = _strip_style_stroke(styles)

  bubble_classes_path = job_dir / "bubble_classes.json"
  bubble_classes = {}
  if bubble_classes_path.exists():
    try:
      bubble_classes = (read_json(bubble_classes_path) or {}).get("classes") or {}
    except Exception:
      bubble_classes = {}

  fit_map: Dict[str, Any] = {}
  try:
    autofit_cfg = (job.get("config") or {}).get("autofit") or {}
    if not isinstance(autofit_cfg, dict):
      autofit_cfg = {}
    if autofit_cfg.get("enabled", True):
      from tiper.autofit.fontfit import compute_fit_map

      if isinstance(payload, dict):
        override = payload.get("autofit") or {}
        if isinstance(override, dict) and override:
          merged = dict(autofit_cfg)
          merged.update(override)
          autofit_cfg = merged

      geom_path = job_dir / "bubble_geometry.json"
      geom = read_json(geom_path) if geom_path.exists() else {}
      ocr_stats_path = job_dir / "ocr_mask_stats.json"
      ocr_stats = read_json(ocr_stats_path) if ocr_stats_path.exists() else {}

      fit_map = compute_fit_map(
        translations=translations,
        bubbles=bubbles,
        bubble_geometry=geom,
        ocr_mask_stats=ocr_stats,
        config=autofit_cfg,
        bubble_classes=bubble_classes,
      )

      # If autofit produced explicit line breaks (komponovka), apply them to the response
      # so the Photoshop extension receives already-composed text.
      try:
        komp = autofit_cfg.get("komponovka") if isinstance(autofit_cfg, dict) else None
        apply_komp = True
        if isinstance(komp, dict) and komp:
          apply_komp = bool(komp.get("apply_to_translation", True))
        if apply_komp and isinstance(fit_map, dict):
          for bid, item in fit_map.items():
            if not isinstance(item, dict):
              continue
            lt = item.get("layout_text")
            if isinstance(lt, str) and lt.strip():
              translations[str(bid)] = lt
      except Exception:
        pass
      try:
        write_json(job_dir / "translation_fit.json", {"schema_version": 1, "fit": fit_map})
      except Exception:
        pass
  except Exception:
    fit_map = {}

  return {
    "translations": [{"id": bid, "text": translations.get(bid, "")} for bid in expected_ids],
    "styles": styles,
    "bubble_classes": bubble_classes,
    "fit": fit_map,
  }
