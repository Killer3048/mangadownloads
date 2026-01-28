import sys
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import yaml
from PIL import Image
import torch
import shutil
import warnings
import gc

# Allow processing very large images without Pillow's decompression bomb guard.
# The pipeline works on intentionally large merged pages, so we raise the limit
# and silence the warning to avoid hard failures while keeping behavior unchanged.
Image.MAX_IMAGE_PIXELS = None
warnings.simplefilter("ignore", Image.DecompressionBombWarning)

# Add subdirectories to path to allow imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'cutting'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'blaster'))

try:
    from cutting.main import (
        load_yolo_model,
        run_object_detection,
        run_text_detection,
        compute_cut_positions,
        load_text_detectors,
    )
    from cutting.smart_slicer import SmartSlicer
    from blaster.blaster_module import Blaster
    from batch_processor import BatchProcessor
    from frames_slicer import FramesSlicer
except ImportError as e:
    print(f"[ERROR] Import failed: {e}")
    print("Make sure 'cutting' and 'blaster' directories exist and contain the necessary files.")
    sys.exit(1)


@dataclass
class SharedResources:
    """Hold heavy GPU resources to reuse across chapters and avoid VRAM layering."""
    yolo_model: object
    text_readers: list
    blaster: Blaster


def build_shared_resources(config: dict) -> SharedResources:
    """Load once-per-run GPU models to prevent repeated allocations."""
    cut_cfg = config.get("cutting", {})
    blast_cfg = config.get("blaster", {})

    # Cutting models (YOLO + EasyOCR)
    yolo_model = load_yolo_model(cut_cfg.get("model", "yolo11x.pt"))
    text_readers = []
    if cut_cfg.get("use_text_detection", True):
        text_langs = cut_cfg.get("text_langs", ["ko", "en"])
        text_readers = load_text_detectors(text_langs, device=cut_cfg.get("device", "auto"))

    # Blaster stack (bubble detector + Big-LaMa + EasyOCR)
    blaster = Blaster(
        languages=blast_cfg.get("languages", ["ko", "en"]),
        confidence=blast_cfg.get("confidence", 0.4),
        device=blast_cfg.get("device", "cuda"),
        dilate_size=blast_cfg.get("dilate_size", 5),
        debug=blast_cfg.get("debug", False),
        debug_dir=None,
        secondary_pass=blast_cfg.get("secondary_pass", None),
        bubble_model_path=blast_cfg.get("bubble_model_path", None),
        bubble_confidence=blast_cfg.get("bubble_confidence", 0.4),
    )

    return SharedResources(
        yolo_model=yolo_model,
        text_readers=text_readers,
        blaster=blaster,
    )


def cleanup_cuda_cache():
    """Free cached VRAM without unloading the resident models."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def process_single_chapter(config, image_path, chapter_prefix, chapter_folder, resources: Optional[SharedResources] = None):
    """
    Process a single chapter through the entire pipeline.
    
    Args:
        config: Configuration dictionary
        image_path: Path to the original image
        chapter_prefix: Chapter prefix for naming (e.g., "ch166")
        chapter_folder: Path to the chapter folder
        resources: Optional shared GPU resources to reuse between chapters
    
    Returns:
        bool: True if processing succeeded, False otherwise
    """
    print(f"\n[INFO] Processing chapter: {chapter_prefix}")
    print(f"[INFO] Image path: {image_path}")
    
    # GPU Check
    print("=== GPU CHECK ===")
    if torch.cuda.is_available():
        print(f"[INFO] Torch CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("[WARN] Torch CUDA NOT available. Running on CPU.")
    
    # Load Image
    img = Image.open(image_path)
    width, height = img.size
    print(f"[INFO] Image Size: {width}x{height}")
    
    # Create output directory inside chapter folder
    output_dir = chapter_folder / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # === CUTTING PHASE ===
    print("=== STARTING CUTTING PHASE ===")
    cut_cfg = config.get("cutting", {})
    
    # Load Models for Cutting (reuse if provided to avoid VRAM bloat)
    model_name = cut_cfg.get("model", "yolo11x.pt")
    yolo_model = resources.yolo_model if resources else load_yolo_model(model_name)
    
    # Run Detections
    det_chunk_height = cut_cfg.get("det_chunk_height", 4096)
    det_overlap = cut_cfg.get("det_overlap", 256)
    conf = cut_cfg.get("conf", 0.25)
    imgsz = cut_cfg.get("imgsz", 1280)
    device = cut_cfg.get("device", "auto")
    
    yolo_boxes = run_object_detection(
        img, yolo_model, det_chunk_height, det_overlap, conf, imgsz, device
    )
    all_boxes = list(yolo_boxes)
    
    if cut_cfg.get("use_text_detection", True):
        text_readers = resources.text_readers if resources else load_text_detectors(
            cut_cfg.get("text_langs", ["ko", "en"]),
            device=device
        )
        for reader in text_readers:
            text_boxes = run_text_detection(
                img, reader,
                cut_cfg.get("det_chunk_height", 4096),
                cut_cfg.get("det_overlap", 256),
                cut_cfg.get("text_conf", 0.40),
                20, 20  # min w/h
            )
            all_boxes.extend(text_boxes)
    
    # Smart Slicing
    weights = cut_cfg.get("weights", {})
    slicer = SmartSlicer(
        edge_weight=weights.get("edge_score", 1.0),
        variance_weight=weights.get("variance_score", 0.5),
        gradient_weight=weights.get("gradient_score", 0.5),
        white_space_weight=weights.get("white_space_score", 2.0),
        distance_penalty=weights.get("distance_penalty", 0.0001),
        forbidden_cost=1e9
    )
    
    print("[INFO] Computing Cost Map...")
    margin = cut_cfg.get("margin", 20)
    cost_map = slicer.compute_cost_map(img, all_boxes, margin=margin)
    
    cuts = compute_cut_positions(
        height, slicer, cost_map, 
        cut_cfg.get("min_height", 1000), 
        cut_cfg.get("max_height", 4000)
    )
    
    # Save Slices
    slices_dir = output_dir / "slices"
    slices_dir.mkdir(exist_ok=True)
    
    slice_paths = []
    
    print(f"[INFO] Saving {len(cuts)-1} slices...")
    for i in range(len(cuts) - 1):
        top = cuts[i]
        bottom = cuts[i + 1]
        if bottom <= top:
            continue
        
        crop = img.crop((0, top, width, bottom))
        slice_filename = f"slice_{i:03d}.png"
        slice_path = slices_dir / slice_filename
        crop.save(slice_path)
        slice_paths.append(slice_path)
    
    print("=== CUTTING PHASE COMPLETE ===")
    
    # === FRAMES SLICING (SECONDARY PASS) ===
    frames_cfg = config.get("frames", {})
    if frames_cfg.get("enabled", False):
        print("=== STARTING FRAMES SLICING ===")
        frames_slicer = FramesSlicer(config)
        frames_slicer.slice_frames(
            image=img,
            chapter_folder=chapter_folder,
            cost_map=cost_map,
            boxes=all_boxes,
            margin=margin
        )
        print("=== FRAMES SLICING COMPLETE ===")
    
    # === BLASTER PHASE ===
    print("=== STARTING BLASTER PHASE ===")
    blast_cfg = config.get("blaster", {})
    
    # Create debug directory for blaster
    debug_dir = output_dir / "debug" if blast_cfg.get("debug", False) else None
    
    blaster = resources.blaster if resources else Blaster(
        languages=blast_cfg.get("languages", ["ko", "en"]),
        confidence=blast_cfg.get("confidence", 0.4),
        device=blast_cfg.get("device", "cuda"),
        dilate_size=blast_cfg.get("dilate_size", 5),
        debug=blast_cfg.get("debug", False),
        debug_dir=str(debug_dir) if debug_dir else None,
        secondary_pass=blast_cfg.get("secondary_pass", None),
        bubble_model_path=blast_cfg.get("bubble_model_path", None),
        bubble_confidence=blast_cfg.get("bubble_confidence", 0.4)
    )
    
    processed_slices = []
    
    for p in slice_paths:
        print(f"[INFO] Blasting slice: {p.name}")
        
        # Define output path for debug file generation
        out_name = p.stem + "_blasted.png"
        out_path = slices_dir / out_name
        
        # Process with output_path to enable debug image saving
        result_img = blaster.process_image(str(p), output_path=str(out_path))
        
        if result_img:
            processed_slices.append(result_img)
        else:
            print(f"[WARN] Failed to process {p.name}, using original.")
            processed_slices.append(Image.open(p))
    
    # Print processing summary
    blaster.print_summary()
    
    print("=== BLASTER PHASE COMPLETE ===")
    
    # === MERGE PHASE ===
    print("=== STARTING MERGE PHASE ===")
    
    # Create blank canvas
    merged_img = Image.new("RGB", (width, height))
    
    current_y = 0
    for i, s_img in enumerate(processed_slices):
        expected_h = cuts[i+1] - cuts[i]
        if s_img.height != expected_h:
            print(f"[WARN] Slice {i} height mismatch. Expected {expected_h}, got {s_img.height}. Resizing.")
            s_img = s_img.resize((width, expected_h))
        
        merged_img.paste(s_img, (0, current_y))
        current_y += expected_h
    
    # Save final cleaned image to chapter folder
    final_output = chapter_folder / f"{chapter_prefix}_cleaned.png"
    merged_img.save(final_output)
    print(f"[SUCCESS] Final image saved to: {final_output}")
    
    print("=== MERGE PHASE COMPLETE ===")

    # Release any transient GPU/CPU allocations while keeping shared models resident
    cleanup_cuda_cache()
    gc.collect()

    return True


def main():
    # 1. Load Config
    config = load_config()
    
    batch_cfg = config.get("batch", {})
    
    if batch_cfg.get("enabled", False):
        # === BATCH MODE ===
        print("=" * 60)
        print("BATCH PROCESSING MODE")
        print("=" * 60)
        
        # Load heavy models once to prevent VRAM layering across chapters
        print("[INFO] Pre-loading shared GPU resources for batch run...")
        shared_resources = build_shared_resources(config)
        
        batch = BatchProcessor(config)
        
        # Get range limits
        start_from = batch_cfg.get("start_from")
        stop_at = batch_cfg.get("stop_at")
        
        def pipeline_wrapper(image_path, chapter_prefix, chapter_folder):
            return process_single_chapter(
                config,
                image_path,
                chapter_prefix,
                chapter_folder,
                resources=shared_resources
            )
        
        stats = batch.process_all(
            pipeline_func=pipeline_wrapper,
            start_from=start_from,
            stop_at=stop_at
        )
        
        print(f"\n[COMPLETE] Batch processing finished.")
        print(f"  Processed: {stats['processed']}/{stats['total']}")
        if stats['failed'] > 0:
            print(f"  Failed chapters: {stats['failed_chapters']}")
    
    else:
        # === SINGLE IMAGE MODE ===
        print("=" * 60)
        print("SINGLE IMAGE MODE")
        print("=" * 60)
        
        image_path_str = config.get("image_path", "chapters/ch166/ch166_original.png")
        output_dir_str = config.get("output_dir", "output")
        
        image_path = Path(image_path_str)
        output_dir = Path(output_dir_str)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if not image_path.exists():
            print(f"[ERROR] Image not found: {image_path}")
            return
        
        # Extract chapter info from path
        chapter_folder = image_path.parent
        chapter_prefix = chapter_folder.name
        
        # Reuse models even in single run to keep behavior consistent
        shared_resources = build_shared_resources(config)
        success = process_single_chapter(
            config,
            image_path,
            chapter_prefix,
            chapter_folder,
            resources=shared_resources
        )
        
        if success:
            print("\n[COMPLETE] Single image processing finished.")
        else:
            print("\n[ERROR] Processing failed.")


if __name__ == "__main__":
    main()
