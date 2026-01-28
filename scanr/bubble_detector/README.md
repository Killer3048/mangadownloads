# TypeR Bubble Detector (YOLOv8 via Python server)

TypeR can optionally use a local HTTP server to detect the speech bubble around the active text layer and align text to the bubble center.

## Run

1. Put your YOLO bubble model weights at `bubble_detector/models/bubble.pt` (or pass a custom path).
2. Install deps (example):
   - `pip install -r bubble_detector/requirements.txt`
3. Start the server:
   - `python bubble_detector/server.py --port 8765 --model bubble_detector/models/bubble.pt`

TypeR will connect to `http://localhost:8765`.

## Endpoints

- `GET /health`
- `POST /load`
- `POST /unload`
- `POST /detect`

`/detect` expects JSON:

```json
{
  "image_base64": "<base64 PNG/JPEG bytes>",
  "roi_offset": {"x": 0, "y": 0},
  "point": {"x": 123, "y": 456},
  "conf": 0.5,
  "imgsz": 1280
}
```

It returns the selected bubble (center + core bbox) in *document coordinates*.

