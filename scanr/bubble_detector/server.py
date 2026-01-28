#!/usr/bin/env python3
import argparse
import base64
import io
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from PIL import Image
from typing import Optional, Tuple

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

try:
    from ultralytics import YOLO
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: ultralytics. Install with `pip install ultralytics`."
    ) from e


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    handler.send_header("Access-Control-Allow-Headers", "Content-Type")
    handler.end_headers()
    handler.wfile.write(body)


def _read_json(handler: BaseHTTPRequestHandler) -> dict:
    length = int(handler.headers.get("Content-Length", "0") or "0")
    if length <= 0:
        return {}
    raw = handler.rfile.read(length)
    return json.loads(raw.decode("utf-8"))


def _decode_base64_image(image_base64: str) -> Image.Image:
    # Support both raw base64 and data-urls.
    if "," in image_base64 and image_base64.strip().lower().startswith("data:"):
        image_base64 = image_base64.split(",", 1)[1]
    data = base64.b64decode(image_base64, validate=False)
    img = Image.open(io.BytesIO(data)).convert("RGB")
    return img


def _torch_device(default: Optional[str] = None) -> str:
    if default:
        return default
    if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _distance_center_and_core_bbox(
    mask_u8: np.ndarray,
) -> Tuple[Tuple[float, float], Tuple[float, float, float, float]]:
    # mask_u8: 0/255 (H,W)
    ys, xs = np.where(mask_u8 > 0)
    if len(xs) == 0:
        return (0.0, 0.0), (0.0, 0.0, 0.0, 0.0)

    if cv2 is None:
        cx = float(xs.mean())
        cy = float(ys.mean())
        x1, x2 = float(xs.min()), float(xs.max())
        y1, y2 = float(ys.min()), float(ys.max())
        return (cx, cy), (x1, y1, x2, y2)

    dist = cv2.distanceTransform((mask_u8 > 0).astype(np.uint8), cv2.DIST_L2, 5)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(dist)
    cx, cy = float(max_loc[0]), float(max_loc[1])

    if not (max_val > 0):
        x1, x2 = float(xs.min()), float(xs.max())
        y1, y2 = float(ys.min()), float(ys.max())
        return (cx, cy), (x1, y1, x2, y2)

    core = dist >= (max_val * 0.5)
    cy_idxs, cx_idxs = np.where(core)
    if len(cx_idxs) == 0:
        x1, x2 = float(xs.min()), float(xs.max())
        y1, y2 = float(ys.min()), float(ys.max())
        return (cx, cy), (x1, y1, x2, y2)

    x1, x2 = float(cx_idxs.min()), float(cx_idxs.max())
    y1, y2 = float(cy_idxs.min()), float(cy_idxs.max())
    return (cx, cy), (x1, y1, x2, y2)


class BubbleDetectorServer:
    def __init__(self, model_path: str, device: str, conf: float, imgsz: int):
        self._lock = threading.Lock()
        self._model = None
        self.model_path = model_path
        self.device = device
        self.conf = conf
        self.imgsz = imgsz

    def load(self, model_path: Optional[str] = None, device: Optional[str] = None) -> None:
        with self._lock:
            if model_path:
                self.model_path = model_path
            if device:
                self.device = device
            if self._model is not None:
                return
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(self.model_path)
            self._model = YOLO(self.model_path)

    def unload(self) -> None:
        with self._lock:
            self._model = None
        if torch is not None and hasattr(torch, "cuda"):
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    def health(self) -> dict:
        return {
            "status": "ok",
            "model_loaded": self._model is not None,
            "model_path": self.model_path,
            "device": self.device,
            "conf": self.conf,
            "imgsz": self.imgsz,
        }

    def detect(
        self,
        img: Image.Image,
        roi_offset: Tuple[float, float],
        point_abs: Tuple[float, float],
        conf: Optional[float],
        imgsz: Optional[int],
    ) -> dict:
        if self._model is None:
            self.load()

        conf_val = float(conf if conf is not None else self.conf)
        imgsz_val = int(imgsz if imgsz is not None else self.imgsz)

        roi_x, roi_y = roi_offset
        px_abs, py_abs = point_abs
        px = float(px_abs - roi_x)
        py = float(py_abs - roi_y)

        with self._lock:
            kwargs = {
                "conf": conf_val,
                "imgsz": imgsz_val,
                "device": self.device,
                "verbose": False,
                "retina_masks": True,
            }
            try:
                results = self._model.predict(img, **kwargs)
            except TypeError:
                # Older ultralytics versions may not support `retina_masks`.
                kwargs.pop("retina_masks", None)
                results = self._model.predict(img, **kwargs)

        if not results:
            return {"selected": None, "bubbles": []}

        r0 = results[0]
        boxes = getattr(r0, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return {"selected": None, "bubbles": []}

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clses = boxes.cls.cpu().numpy()

        masks = getattr(r0, "masks", None)
        mask_data = None
        if masks is not None and getattr(masks, "data", None) is not None:
            try:
                mask_data = masks.data.cpu().numpy()
            except Exception:
                mask_data = None

        bubbles = []
        for i, ((x1, y1, x2, y2), prob, cls_id) in enumerate(zip(xyxy, confs, clses)):
            x1f, y1f, x2f, y2f = float(x1), float(y1), float(x2), float(y2)
            bubbles.append(
                {
                    "index": int(i),
                    "confidence": float(prob),
                    "cls": int(cls_id),
                    "bbox": {"x1": x1f, "y1": y1f, "x2": x2f, "y2": y2f},
                }
            )

        def contains_point(b: dict) -> bool:
            bb = b["bbox"]
            return bb["x1"] <= px <= bb["x2"] and bb["y1"] <= py <= bb["y2"]

        inside = [b for b in bubbles if contains_point(b)]
        if inside:
            inside.sort(
                key=lambda b: (
                    (b["bbox"]["x2"] - b["bbox"]["x1"]) * (b["bbox"]["y2"] - b["bbox"]["y1"]),
                    -b["confidence"],
                )
            )
            chosen = inside[0]
        else:
            def dist2_to_center(b: dict) -> float:
                bb = b["bbox"]
                cx = (bb["x1"] + bb["x2"]) / 2.0
                cy = (bb["y1"] + bb["y2"]) / 2.0
                dx = cx - px
                dy = cy - py
                return dx * dx + dy * dy

            bubbles.sort(key=lambda b: (dist2_to_center(b), -b["confidence"]))
            chosen = bubbles[0]

        bb = chosen["bbox"]
        center_x = (bb["x1"] + bb["x2"]) / 2.0
        center_y = (bb["y1"] + bb["y2"]) / 2.0
        core = (bb["x1"], bb["y1"], bb["x2"], bb["y2"])

        if mask_data is not None:
            try:
                m = mask_data[chosen["index"]]
                mask_u8 = (m > 0.5).astype(np.uint8) * 255
                (cx, cy), (x1c, y1c, x2c, y2c) = _distance_center_and_core_bbox(mask_u8)
                center_x, center_y = cx, cy
                core = (x1c, y1c, x2c, y2c)
            except Exception:
                pass

        selected = {
            "confidence": chosen["confidence"],
            "cls": chosen["cls"],
            "center": {"x": float(center_x + roi_x), "y": float(center_y + roi_y)},
            "bbox": {
                "x1": float(bb["x1"] + roi_x),
                "y1": float(bb["y1"] + roi_y),
                "x2": float(bb["x2"] + roi_x),
                "y2": float(bb["y2"] + roi_y),
            },
            "core_bbox": {
                "x1": float(core[0] + roi_x),
                "y1": float(core[1] + roi_y),
                "x2": float(core[2] + roi_x),
                "y2": float(core[3] + roi_y),
            },
        }

        return {"selected": selected, "bubbles": bubbles}


class Handler(BaseHTTPRequestHandler):
    server_version = "TypeRBubbleDetector/1.0"

    def do_OPTIONS(self):  # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):  # noqa: N802
        if self.path == "/" or self.path == "":
            return _json_response(self, 200, {"name": "TypeR Bubble Detector", "health": "/health"})
        if self.path == "/health":
            return _json_response(self, 200, self.server.detector.health())
        return _json_response(self, 404, {"error": "not_found"})

    def do_POST(self):  # noqa: N802
        if self.path not in ("/load", "/unload", "/detect"):
            return _json_response(self, 404, {"error": "not_found"})

        try:
            data = _read_json(self)
        except Exception as e:
            return _json_response(self, 400, {"error": "invalid_json", "detail": str(e)})

        if self.path == "/load":
            model_path = data.get("model_path")
            device = data.get("device")
            try:
                self.server.detector.load(model_path=model_path, device=device)
            except FileNotFoundError:
                return _json_response(self, 400, {"error": "model_not_found", "model_path": model_path or self.server.detector.model_path})
            except Exception as e:
                return _json_response(self, 500, {"error": "load_failed", "detail": str(e)})
            return _json_response(self, 200, self.server.detector.health())

        if self.path == "/unload":
            try:
                self.server.detector.unload()
            except Exception as e:
                return _json_response(self, 500, {"error": "unload_failed", "detail": str(e)})
            return _json_response(self, 200, self.server.detector.health())

        # /detect
        image_base64 = data.get("image_base64")
        roi_offset = data.get("roi_offset") or {}
        point = data.get("point") or {}
        conf = data.get("conf")
        imgsz = data.get("imgsz")

        if not image_base64:
            return _json_response(self, 400, {"error": "missing_image_base64"})

        try:
            img = _decode_base64_image(image_base64)
        except Exception as e:
            return _json_response(self, 400, {"error": "invalid_image", "detail": str(e)})

        try:
            roi_x = float(roi_offset.get("x", 0))
            roi_y = float(roi_offset.get("y", 0))
            px = float(point.get("x"))
            py = float(point.get("y"))
        except Exception:
            return _json_response(self, 400, {"error": "missing_point_or_roi_offset"})

        t0 = time.time()
        try:
            result = self.server.detector.detect(
                img,
                roi_offset=(roi_x, roi_y),
                point_abs=(px, py),
                conf=conf,
                imgsz=imgsz,
            )
        except FileNotFoundError:
            return _json_response(self, 400, {"error": "model_not_found", "model_path": self.server.detector.model_path})
        except Exception as e:
            return _json_response(self, 500, {"error": "detect_failed", "detail": str(e)})

        result["timings_ms"] = {"total": int((time.time() - t0) * 1000)}
        return _json_response(self, 200, result)

    def log_message(self, fmt, *args):  # noqa: D401
        # Keep server quiet by default (CEP calls can be frequent).
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="TypeR bubble detector HTTP server (YOLOv8).")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--model", default=os.environ.get("BUBBLE_MODEL_PATH", os.path.join(os.path.dirname(__file__), "models", "bubble.pt")))
    parser.add_argument("--device", default=os.environ.get("BUBBLE_DEVICE", None))
    parser.add_argument("--conf", type=float, default=float(os.environ.get("BUBBLE_CONF", "0.2")))
    parser.add_argument("--imgsz", type=int, default=int(os.environ.get("BUBBLE_IMGSZ", "1280")))
    args = parser.parse_args()

    device = _torch_device(args.device)
    detector = BubbleDetectorServer(model_path=args.model, device=device, conf=args.conf, imgsz=args.imgsz)

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    httpd.detector = detector  # type: ignore[attr-defined]
    print(f"[TypeR Bubble Detector] Listening on http://{args.host}:{args.port}")
    print(f"[TypeR Bubble Detector] Model: {detector.model_path} (device={detector.device}, conf={detector.conf}, imgsz={detector.imgsz})")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
