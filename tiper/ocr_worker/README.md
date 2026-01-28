# PaddleOCR worker (optional)

This is an **optional** companion service for higher-quality OCR (especially CJK) without mixing `paddlepaddle` and `torch` in the same environment.

## Run

1) Create a separate venv (recommended) and install Paddle + PaddleOCR.
2) Start the worker:

Optional (recommended): pre-download models into PaddleOCR default cache:

```bash
python ../../script/download_models.py --langs korean,en,japan,ch,chinese_cht
```

```bash
python -m uvicorn app:app --host 127.0.0.1 --port 8777
```

Optional: force Paddle device:
- `TIPER_OCR_DEVICE=cpu`
- `TIPER_OCR_DEVICE=gpu` / `TIPER_OCR_DEVICE=gpu:0`

3) Point the main `tiper` server to it:

```bash
export TIPER_OCR_WORKER_URL="http://127.0.0.1:8777"
```

Endpoints:
- `GET /health` (includes Paddle/CUDA info)
- `POST /ocr` → `{ image_base64, lang }` → `{ results: [{ poly, text, conf }, ...] }` (baseline PaddleOCR)
- `POST /ocr_bubble` → `{ image_base64, lang }` → `{ results: [{ poly, text, conf }, ...] }` (PaddleOCRVL + retries + fallback)
