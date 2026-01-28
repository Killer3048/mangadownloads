#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
from typing import List, Tuple, Any, Dict, Optional

import numpy as np
from PIL import Image, ImageFile

# For huge long images (like 690 x 180000)
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    from PIL import ImageCms
except ImportError:
    ImageCms = None

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    import torch
except ImportError:
    torch = None

try:
    import yaml
except ImportError:
    yaml = None

try:
    import easyocr
except ImportError:
    easyocr = None

# Import the new SmartSlicer
from smart_slicer import SmartSlicer


def load_config(script_dir: Path) -> Dict[str, Any]:
    if yaml is None:
        raise SystemExit(
            "Module 'yaml' (pyyaml) not found. Install:\n"
            "    pip install pyyaml"
        )

    cfg_path = script_dir / "config.yaml"
    if not cfg_path.is_file():
        raise SystemExit(
            f"Configuration file not found: {cfg_path}\n"
            f"Create it (see script description)."
        )

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if not isinstance(cfg, dict):
        raise SystemExit("config.yaml must contain a YAML object (key->value map).")

    return cfg


def cfg_get(cfg: Dict[str, Any], key: str, default: Any) -> Any:
    return cfg.get(key, default)


def find_merged_png(script_dir: Path, filename: str = "merged.png") -> Path:
    for dirpath, _, filenames in os.walk(script_dir):
        if filename in filenames:
            return Path(dirpath) / filename
    raise FileNotFoundError(
        f"File {filename} not found in any subdirectory of {script_dir}.\n"
        f"Or specify the path in config.yaml (image_path: \"ep162/merged.png\")."
    )


def load_yolo_model(model_name: str = "yolo11x.pt") -> Any:
    if YOLO is None:
        raise SystemExit(
            "Module 'ultralytics' not found. Install dependencies:\n"
            "    pip install ultralytics pillow numpy pyyaml easyocr opencv-python\n"
            "and PyTorch with CUDA support if GPU is needed."
        )

    print(f"[INFO] Loading Ultralytics YOLO11 model: {model_name}")
    model = YOLO(model_name)

    if torch is not None:
        print(f"[INFO] torch.cuda.is_available() = {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            try:
                dev_name = torch.cuda.get_device_name(0)
            except Exception:
                dev_name = "CUDA device 0"
            print(f"[INFO] GPU detected: {dev_name}")
        else:
            print("[WARN] CUDA not available. YOLO11 will run on CPU.")
    else:
        print("[WARN] Module 'torch' not imported. Check PyTorch installation.")

    return model


def build_text_language_groups(lang_list: List[str]) -> List[List[str]]:
    langs: List[str] = []
    for l in lang_list:
        if not l:
            continue
        l = str(l).strip()
        if not l:
            continue
        if l not in langs:
            langs.append(l)

    groups: List[List[str]] = []

    if "ko" in langs:
        ko_group = ["ko"]
        if "en" not in langs:
            ko_group.append("en")
        else:
            ko_group.append("en")
            langs.remove("en")

        langs.remove("ko")
        groups.append(ko_group)

    if langs:
        groups.append(langs)

    return groups


def load_text_detectors(lang_list: List[str], device: str = "auto") -> List[Any]:
    if easyocr is None:
        raise SystemExit(
            "Module 'easyocr' not found. Install:\n"
            "    pip install easyocr"
        )

    use_gpu = False
    if device != "cpu" and torch is not None and torch.cuda.is_available():
        use_gpu = True

    groups = build_text_language_groups(lang_list)
    if not groups:
        raise SystemExit("Language list for text detector is empty after processing.")

    readers: List[Any] = []
    for g in groups:
        print(f"[INFO] Loading EasyOCR text detector (langs={g}, gpu={use_gpu})")
        reader = easyocr.Reader(g, gpu=use_gpu)
        readers.append(reader)

    return readers


def run_object_detection(
    image: Image.Image,
    model: Any,
    chunk_height: int = 4096,
    overlap: int = 256,
    conf: float = 0.25,
    imgsz: int = 1280,
    device: str = "auto",
) -> List[Tuple[float, float, float, float, float, int]]:
    width, height = image.size
    boxes: List[Tuple[float, float, float, float, float, int]] = []

    print(f"[INFO] Image size: {width} x {height}")
    print(
        f"[INFO] YOLO: block detection: chunk_height={chunk_height}, "
        f"overlap={overlap}, imgsz={imgsz}, conf={conf}, device={device}"
    )

    y = 0
    idx = 0

    while y < height:
        chunk_top = max(0, int(y - overlap))
        chunk_bottom = min(height, int(y + chunk_height + overlap))

        crop = image.crop((0, chunk_top, width, chunk_bottom))

        print(f"[INFO] YOLO block #{idx}: y=[{chunk_top}:{chunk_bottom}]")

        predict_kwargs = dict(
            conf=conf,
            imgsz=imgsz,
            verbose=False,
        )
        if device != "auto":
            predict_kwargs["device"] = device

        results = model.predict(crop, **predict_kwargs)

        for r in results:
            bboxes = getattr(r, "boxes", None)
            if bboxes is None or len(bboxes) == 0:
                continue

            xyxy = bboxes.xyxy.cpu().numpy()
            confs = bboxes.conf.cpu().numpy()
            clses = bboxes.cls.cpu().numpy()

            for (x1, y1, x2, y2), c, cls_id in zip(xyxy, confs, clses):
                if (y2 - y1) < 20 or (x2 - x1) < 20:
                    continue

                g_y1 = float(chunk_top + y1)
                g_y2 = float(chunk_top + y2)
                g_x1 = float(x1)
                g_x2 = float(x2)

                g_y1 = max(0.0, min(g_y1, height - 1.0))
                g_y2 = max(0.0, min(g_y2, height - 1.0))
                g_x1 = max(0.0, min(g_x1, width - 1.0))
                g_x2 = max(0.0, min(g_x2, width - 1.0))

                if g_y2 <= g_y1 or g_x2 <= g_x1:
                    continue

                boxes.append((g_x1, g_y1, g_x2, g_y2, float(c), int(cls_id)))

        y += chunk_height
        idx += 1

    print(f"[INFO] YOLO: objects found: {len(boxes)}")
    return boxes


def run_text_detection(
    image: Image.Image,
    reader: Any,
    chunk_height: int,
    overlap: int,
    conf: float,
    min_box_w: int,
    min_box_h: int,
) -> List[Tuple[float, float, float, float, float, int]]:
    width, height = image.size
    boxes: List[Tuple[float, float, float, float, float, int]] = []

    print(
        f"[INFO] TEXT: text detection by blocks: chunk_height={chunk_height}, "
        f"overlap={overlap}, conf={conf}"
    )

    y = 0
    idx = 0

    while y < height:
        chunk_top = max(0, int(y - overlap))
        chunk_bottom = min(height, int(y + chunk_height + overlap))

        crop = image.crop((0, chunk_top, width, chunk_bottom)).convert("RGB")
        crop_np = np.array(crop)

        print(f"[INFO] TEXT block #{idx}: y=[{chunk_top}:{chunk_bottom}]")

        results = reader.readtext(
            crop_np,
            detail=1,
            paragraph=False,
        )

        for (bbox, _text, prob) in results:
            if prob < conf:
                continue

            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]

            x1 = min(xs)
            x2 = max(xs)
            y1 = min(ys)
            y2 = max(ys)

            w = x2 - x1
            h = y2 - y1

            if w < min_box_w or h < min_box_h:
                continue

            g_y1 = float(chunk_top + y1)
            g_y2 = float(chunk_top + y2)
            g_x1 = float(x1)
            g_x2 = float(x2)

            g_y1 = max(0.0, min(g_y1, height - 1.0))
            g_y2 = max(0.0, min(g_y2, height - 1.0))
            g_x1 = max(0.0, min(g_x1, width - 1.0))
            g_x2 = max(0.0, min(g_x2, width - 1.0))

            if g_y2 <= g_y1 or g_x2 <= g_x1:
                continue

            boxes.append((g_x1, g_y1, g_x2, g_y2, float(prob), -1))

        y += chunk_height
        idx += 1

    print(f"[INFO] TEXT: text boxes found: {len(boxes)}")
    return boxes


def get_srgb_profile_bytes() -> Optional[bytes]:
    if ImageCms is None:
        return None
    try:
        prof = ImageCms.createProfile("sRGB")
        # Pillow versions differ slightly in API surfaces; try both.
        try:
            return ImageCms.ImageCmsProfile(prof).tobytes()
        except Exception:
            return prof.tobytes()
    except Exception:
        return None


def resolve_output_icc_profile(
    img: Image.Image,
    policy: str = "preserve",
) -> Optional[bytes]:
    """
    policy:
      - preserve: keep input profile if present, otherwise None
      - srgb_if_missing: keep input if present, else embed sRGB
      - strip: always None
    """
    icc_in = img.info.get("icc_profile")

    policy = (policy or "preserve").lower().strip()

    if policy == "strip":
        return None

    if icc_in:
        return icc_in

    if policy == "srgb_if_missing":
        return get_srgb_profile_bytes()

    return None


def compute_cut_positions(
    img_height: int,
    slicer: SmartSlicer,
    cost_map: np.ndarray,
    min_h: int,
    max_h: int,
) -> List[int]:
    preferred_h = (min_h + max_h) // 2
    cuts: List[int] = [0]
    current = 0

    print(f"[INFO] Calculating cut lines: min={min_h}, max={max_h}, preferred={preferred_h}")

    while True:
        remaining = img_height - current
        if remaining <= max_h:
            cuts.append(img_height)
            break

        cut_y = slicer.find_best_cut(
            cost_map=cost_map,
            current_y=current,
            min_h=min_h,
            max_h=max_h,
            preferred_h=preferred_h
        )

        if cut_y <= current:
            print(f"[WARN] SmartSlicer returned {cut_y} <= current {current}. Forcing min_h advance.")
            cut_y = current + min_h

        # Avoid micro-last-pages
        if img_height - cut_y < min_h:
            cut_y = img_height

        cut_y = min(cut_y, img_height)
        cuts.append(cut_y)
        current = cut_y

        if current >= img_height:
            break

    dedup_cuts: List[int] = []
    for c in cuts:
        if not dedup_cuts or c != dedup_cuts[-1]:
            dedup_cuts.append(c)

    print(f"[INFO] Number of cut lines (including 0 and H): {len(dedup_cuts)}")
    for i in range(len(dedup_cuts) - 1):
        h = dedup_cuts[i + 1] - dedup_cuts[i]
        print(f"  Page {i + 1}: height {h}")

    return dedup_cuts


def save_pages(
    image: Image.Image,
    cuts: List[int],
    output_dir: Path,
    output_ext: str = "jpg",
    jpeg_quality: int = 100,
    png_compress_level: int = 6,
    icc_profile: Optional[bytes] = None,
) -> None:
    width, height = image.size

    output_ext = (output_ext or "jpg").lower().strip(".")
    is_jpeg = output_ext in ("jpg", "jpeg")
    is_png = output_ext == "png"

    print(f"[INFO] Saving pages to {output_dir} as .{output_ext}")

    for i in range(len(cuts) - 1):
        top = cuts[i]
        bottom = cuts[i + 1]
        if bottom <= top:
            continue

        crop = image.crop((0, top, width, bottom))

        out_path = output_dir / f"{i + 1}.{output_ext}"

        if is_jpeg:
            crop = crop.convert("RGB")
            save_kwargs = dict(
                format="JPEG",
                quality=int(jpeg_quality),
                subsampling=0,
                optimize=True,
            )
            if icc_profile:
                save_kwargs["icc_profile"] = icc_profile

            crop.save(out_path, **save_kwargs)

        elif is_png:
            # Keep alpha if exists
            save_kwargs = dict(
                format="PNG",
                optimize=True,
                compress_level=int(png_compress_level),
            )
            if icc_profile:
                save_kwargs["icc_profile"] = icc_profile

            crop.save(out_path, **save_kwargs)

        else:
            # Fallback: try PIL's default save for other formats
            crop = crop.convert("RGB")
            save_kwargs = {}
            if icc_profile:
                save_kwargs["icc_profile"] = icc_profile
            crop.save(out_path, **save_kwargs)

        print(f"[INFO] Page {i + 1}: {out_path.name} (height {bottom - top})")


def main() -> None:
    script_dir = Path(__file__).resolve().parent

    cfg = load_config(script_dir)

    image_path_cfg = cfg_get(cfg, "image_path", None)
    min_height = int(cfg_get(cfg, "min_height", 5000))
    max_height = int(cfg_get(cfg, "max_height", 7500))
    det_chunk_height = int(cfg_get(cfg, "det_chunk_height", 4096))
    det_overlap = int(cfg_get(cfg, "det_overlap", 256))
    model_name = str(cfg_get(cfg, "model", "yolo11x.pt"))
    imgsz = int(cfg_get(cfg, "imgsz", 1280))
    conf = float(cfg_get(cfg, "conf", 0.25))
    margin = int(cfg_get(cfg, "margin", 20))
    device = str(cfg_get(cfg, "device", "auto"))

    use_text_detection = bool(cfg_get(cfg, "use_text_detection", True))
    text_langs_cfg = cfg_get(cfg, "text_langs", ["ru"])
    text_conf = float(cfg_get(cfg, "text_conf", 0.40))
    text_min_box_w = int(cfg_get(cfg, "text_min_box_width", 20))
    text_min_box_h = int(cfg_get(cfg, "text_min_box_height", 20))

    text_chunk_height = cfg_get(cfg, "text_chunk_height", None)
    if text_chunk_height is None:
        text_chunk_height = det_chunk_height
    else:
        text_chunk_height = int(text_chunk_height)

    text_overlap = cfg_get(cfg, "text_overlap", None)
    if text_overlap is None:
        text_overlap = det_overlap
    else:
        text_overlap = int(text_overlap)

    if isinstance(text_langs_cfg, str):
        text_langs = [text_langs_cfg]
    else:
        text_langs = list(text_langs_cfg)

    # Output config
    output_ext = str(cfg_get(cfg, "output_ext", "jpg"))
    jpeg_quality = int(cfg_get(cfg, "jpeg_quality", 100))
    png_compress_level = int(cfg_get(cfg, "png_compress_level", 6))

    # ICC policy for output
    # preserve | srgb_if_missing | strip
    output_icc_policy = str(cfg_get(cfg, "output_icc_policy", "srgb_if_missing"))

    # Smart Slicer Config
    weights = cfg_get(cfg, "weights", {})
    edge_weight = float(weights.get("edge_score", 1.0))
    variance_weight = float(weights.get("variance_score", 0.5))
    gradient_weight = float(weights.get("gradient_score", 0.5))
    white_space_weight = float(weights.get("white_space_score", 2.0))
    distance_penalty = float(weights.get("distance_penalty", 0.0001))

    if image_path_cfg is not None and str(image_path_cfg).strip() != "":
        image_path = Path(image_path_cfg)
        if not image_path.is_absolute():
            image_path = (script_dir / image_path).resolve()
        if not image_path.is_file():
            raise SystemExit(f"File not found by path from config.yaml: {image_path}")
    else:
        image_path = find_merged_png(script_dir, filename="merged.png")

    print(f"[INFO] Using image: {image_path}")

    output_dir = image_path.parent

    img = Image.open(image_path)
    width, height = img.size
    print(f"[INFO] Opened image: {width} x {height}")

    # Resolve ICC profile for output (preserve/attach sRGB)
    icc_profile_out = resolve_output_icc_profile(img, policy=output_icc_policy)
    if img.info.get("icc_profile"):
        print("[INFO] Input ICC profile detected. Will preserve for outputs.")
    else:
        if output_icc_policy.lower() == "srgb_if_missing" and icc_profile_out:
            print("[INFO] No input ICC profile. Will embed sRGB in outputs.")
        elif output_icc_policy.lower() == "strip":
            print("[INFO] ICC profiles will be stripped from outputs.")
        else:
            print("[INFO] No input ICC profile. Outputs will be saved without ICC.")

    model = load_yolo_model(model_name)

    yolo_boxes = run_object_detection(
        img,
        model=model,
        chunk_height=det_chunk_height,
        overlap=det_overlap,
        conf=conf,
        imgsz=imgsz,
        device=device,
    )

    all_boxes = list(yolo_boxes)

    if use_text_detection:
        text_readers = load_text_detectors(text_langs, device=device)
        for reader in text_readers:
            text_boxes = run_text_detection(
                img,
                reader=reader,
                chunk_height=text_chunk_height,
                overlap=text_overlap,
                conf=text_conf,
                min_box_w=text_min_box_w,
                min_box_h=text_min_box_h,
            )
            all_boxes.extend(text_boxes)

    print(f"[INFO] Total boxes (objects + text): {len(all_boxes)}")

    slicer = SmartSlicer(
        edge_weight=edge_weight,
        variance_weight=variance_weight,
        gradient_weight=gradient_weight,
        white_space_weight=white_space_weight,
        distance_penalty=distance_penalty,
        forbidden_cost=1e9
    )

    print("[INFO] Computing Cost Map...")
    cost_map = slicer.compute_cost_map(img, all_boxes, margin=margin)
    print("[INFO] Cost Map Computed.")

    cuts = compute_cut_positions(
        img_height=height,
        slicer=slicer,
        cost_map=cost_map,
        min_h=min_height,
        max_h=max_height,
    )

    save_pages(
        img,
        cuts,
        output_dir,
        output_ext=output_ext,
        jpeg_quality=jpeg_quality,
        png_compress_level=png_compress_level,
        icc_profile=icc_profile_out,
    )

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
