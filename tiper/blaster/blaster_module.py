import sys
import cv2
import numpy as np
from PIL import Image
import easyocr
import torch
import time
import logging
from pathlib import Path
from datetime import datetime
import yaml
from omegaconf import OmegaConf
from ultralytics import YOLO

# Add lama to path
LAMA_PATH = Path(__file__).parent.parent / "lama"
sys.path.insert(0, str(LAMA_PATH))

from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.evaluation.utils import move_to_device


def setup_logger(debug_dir=None):
    """Setup logging with optional file output."""
    logger = logging.getLogger('blaster')
    logger.setLevel(logging.DEBUG)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('[%(levelname)s] %(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler (if debug_dir provided)
    if debug_dir:
        debug_dir = Path(debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
        log_file = debug_dir / f"blaster_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger


class BigLamaInpainter:
    """Wrapper for official Big-LaMa model."""
    
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load config
        config_path = Path(model_path) / 'config.yaml'
        with open(config_path, 'r') as f:
            train_config = OmegaConf.create(yaml.safe_load(f))
        
        train_config.training_model.predict_only = True
        train_config.visualizer.kind = 'noop'
        
        # Load checkpoint
        checkpoint_path = Path(model_path) / 'models' / 'best.ckpt'
        self.model = load_checkpoint(train_config, str(checkpoint_path), strict=False, map_location='cpu')
        self.model.freeze()
        self.model.to(self.device)
        
    def __call__(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        """Inpaint image using mask."""
        # Convert to numpy
        img_np = np.array(image).astype(np.float32) / 255.0
        mask_np = np.array(mask.convert('L')).astype(np.float32) / 255.0
        
        # Pad to multiple of 8
        h, w = img_np.shape[:2]
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        
        if pad_h > 0 or pad_w > 0:
            img_np = np.pad(img_np, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
            mask_np = np.pad(mask_np, ((0, pad_h), (0, pad_w)), mode='reflect')
        
        # Prepare batch
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0)
        
        batch = {
            'image': img_tensor,
            'mask': mask_tensor
        }
        
        # Move to device and inpaint
        with torch.no_grad():
            batch = move_to_device(batch, self.device)
            batch['mask'] = (batch['mask'] > 0) * 1
            result = self.model(batch)
            
            # Get result
            inpainted = result['inpainted'][0].permute(1, 2, 0).cpu().numpy()
        
        # Remove padding
        if pad_h > 0 or pad_w > 0:
            inpainted = inpainted[:h, :w]
        
        # Convert back to PIL
        inpainted = np.clip(inpainted * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(inpainted)


class Blaster:
    def __init__(self, languages=['ko', 'en'], confidence=0.4, device='cuda', dilate_size=5, 
                 debug=False, debug_dir=None, model_path=None, secondary_pass=None,
                 bubble_model_path=None, bubble_confidence=0.4):
        self.languages = languages
        self.base_confidence = confidence
        self.confidence = max(0.0, self.base_confidence - 0.1)  # lower language threshold by 0.1
        self.dilate_size = dilate_size
        self.debug = debug
        self.debug_dir = Path(debug_dir) if debug_dir else None
        self.device = device if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        self.bubble_confidence = bubble_confidence
        
        # Secondary pass config
        self.secondary_pass = secondary_pass or {}
        self.secondary_enabled = self.secondary_pass.get('enabled', False)
        self.secondary_language = self.secondary_pass.get('language', 'th')
        self.secondary_base_confidence = self.secondary_pass.get('confidence', 0.25)
        self.secondary_confidence = max(0.0, self.secondary_base_confidence - 0.1)

        # Speech-bubble detector config
        if bubble_model_path is None:
            bubble_model_path = Path(__file__).parent.parent / "models" / "yolov8m_seg-speech-bubble" / "model.pt"
        self.bubble_model_path = Path(bubble_model_path)
        
        # Setup logger
        self.logger = setup_logger(self.debug_dir if debug else None)
        
        self.logger.info(f"Initializing Blaster:")
        self.logger.info(f"  - Languages: {languages}")
        self.logger.info(f"  - Device: {self.device}")
        self.logger.info(f"  - Primary confidence: {self.confidence:.2f} (input {self.base_confidence:.2f} minus 0.10)")
        self.logger.info(f"  - Dilate size: {dilate_size}")
        self.logger.info(f"  - Debug mode: {debug}")
        self.logger.info(f"  - Speech-bubble model: {self.bubble_model_path} (conf={self.bubble_confidence})")
        if self.secondary_enabled:
            self.logger.info(
                f"  - Secondary pass: {self.secondary_language} (conf={self.secondary_confidence:.2f}, input {self.secondary_base_confidence:.2f} minus 0.10)"
            )
        if self.debug_dir:
            self.logger.info(f"  - Debug directory: {self.debug_dir}")
        
        # Load EasyOCR readers
        self.logger.info("Loading EasyOCR readers...")
        init_start = time.time()

        # Primary reader for all languages
        self.reader = easyocr.Reader(self.languages, gpu=(self.device == 'cuda'))
        
        # Secondary reader (Thai for stylized text)
        if self.secondary_enabled:
            self.reader_secondary = easyocr.Reader([self.secondary_language], gpu=(self.device == 'cuda'))
            self.logger.info(f"  - Loaded {self.secondary_language} reader for second pass")
        else:
            self.reader_secondary = None
            
        self.logger.info(f"EasyOCR loaded in {time.time() - init_start:.2f}s")

        # Load speech-bubble detector (Ultralytics YOLO)
        self.logger.info("Loading speech-bubble detector model...")
        bubble_start = time.time()
        if not self.bubble_model_path.exists():
            raise FileNotFoundError(f"Speech-bubble model not found at {self.bubble_model_path}")
        try:
            self.bubble_model = YOLO(str(self.bubble_model_path))
            # Move to target device if supported
            try:
                self.bubble_model.to(self.device)
            except Exception:
                # Some YOLO versions load device per predict; ignore if .to not available
                pass
        except Exception as e:
            self.logger.error(f"Failed to load speech-bubble model: {e}")
            raise
        self.logger.info(f"Speech-bubble detector loaded in {time.time() - bubble_start:.2f}s")
        
        # Load Big-LaMa model
        self.logger.info("Loading Big-LaMa inpainting model...")
        init_start = time.time()
        
        if model_path is None:
            model_path = Path(__file__).parent.parent / "models" / "big-lama"
        
        self.lama = BigLamaInpainter(model_path, device=self.device)
        self.logger.info(f"Big-LaMa loaded in {time.time() - init_start:.2f}s (device: {self.device})")
        
        # Statistics
        self.stats = {
            'total_images': 0,
            'total_texts_found': 0,
            'total_texts_removed': 0,
            'failed_images': 0,
            'processing_times': []
        }
        
    def _run_ocr(self, img_cv, reader, pass_name=""):
        """Run OCR and return results."""
        try:
            results = reader.readtext(
                img_cv,
                decoder="beamsearch",
                beamWidth=10
            )
            self.logger.debug(f"OCR {pass_name}: Found {len(results)} text regions")
            return results
        except Exception as e:
            self.logger.error(f"OCR {pass_name} failed: {e}")
            return []
    
    def _merge_results(self, results1, results2, iou_threshold=0.5):
        """Merge OCR results - upgrade confidence for overlapping regions only, don't add new."""
        if not results2:
            return results1, set()
        
        # Convert to mutable list
        merged = list(results1)
        upgraded_indices = set()  # Track which indices were upgraded
        
        for bbox2, text2, prob2 in results2:
            pts2 = np.array(bbox2)
            x2_min, y2_min = pts2.min(axis=0)
            x2_max, y2_max = pts2.max(axis=0)
            
            best_match_idx = None
            best_iou = 0
            
            for idx, (bbox1, text1, prob1) in enumerate(merged):
                pts1 = np.array(bbox1)
                x1_min, y1_min = pts1.min(axis=0)
                x1_max, y1_max = pts1.max(axis=0)
                
                # Calculate IoU
                xi_min = max(x1_min, x2_min)
                yi_min = max(y1_min, y2_min)
                xi_max = min(x1_max, x2_max)
                yi_max = min(y1_max, y2_max)
                
                if xi_max > xi_min and yi_max > yi_min:
                    inter_area = (xi_max - xi_min) * (yi_max - yi_min)
                    area1 = (x1_max - x1_min) * (y1_max - y1_min)
                    area2 = (x2_max - x2_min) * (y2_max - y2_min)
                    union_area = area1 + area2 - inter_area
                    iou = inter_area / union_area if union_area > 0 else 0
                    
                    if iou > iou_threshold and iou > best_iou:
                        best_match_idx = idx
                        best_iou = iou
            
            if best_match_idx is not None:
                # Found overlapping detection - replace if secondary has higher confidence
                _, _, prob1 = merged[best_match_idx]
                if prob2 > prob1:
                    self.logger.debug(f"  Upgraded conf: {prob1:.3f} -> {prob2:.3f} (secondary pass)")
                    merged[best_match_idx] = (bbox2, text2, prob2)
                    upgraded_indices.add(best_match_idx)
            # Don't add new regions from secondary pass
        
        return merged, upgraded_indices

    def _detect_bubbles(self, img_cv):
        """Run speech-bubble detection and return list of axis-aligned boxes."""
        bubbles = []
        try:
            results = self.bubble_model.predict(img_cv, conf=self.bubble_confidence, verbose=False, device=self.device)
        except Exception as e:
            self.logger.error(f"Speech-bubble detection failed: {e}")
            return bubbles

        for r in results:
            bboxes = getattr(r, "boxes", None)
            if bboxes is None or len(bboxes) == 0:
                continue

            xyxy = bboxes.xyxy.cpu().numpy()
            confs = bboxes.conf.cpu().numpy()
            clses = bboxes.cls.cpu().numpy()

            for (x1, y1, x2, y2), prob, cls_id in zip(xyxy, confs, clses):
                bubbles.append({
                    'bbox': (float(x1), float(y1), float(x2), float(y2)),
                    'conf': float(prob),
                    'cls': int(cls_id)
                })

        self.logger.debug(f"Speech-bubble detector: found {len(bubbles)} bubbles (conf>={self.bubble_confidence})")
        return bubbles

    @staticmethod
    def _point_in_box(point, box):
        x, y = point
        x1, y1, x2, y2 = box
        return x1 <= x <= x2 and y1 <= y <= y2

    def _text_inside_bubble(self, bbox_points, bubble_boxes):
        """Return True if text polygon center lies inside any bubble box."""
        if not bubble_boxes:
            return False

        xs = [p[0] for p in bbox_points]
        ys = [p[1] for p in bbox_points]
        center = (sum(xs) / len(xs), sum(ys) / len(ys))

        for b in bubble_boxes:
            if self._point_in_box(center, b['bbox']):
                return True
        return False
        
    def process_image(self, image_path, output_path=None):
        """Process image with comprehensive debug logging."""
        process_start = time.time()
        self.stats['total_images'] += 1
        
        image_name = Path(image_path).name if isinstance(image_path, str) else "unknown"
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Processing image: {image_name}")
        self.logger.debug(f"Full path: {image_path}")
        
        # Load image
        load_start = time.time()
        if isinstance(image_path, str):
            img_cv = cv2.imread(image_path)
            img_pil = Image.open(image_path).convert("RGB")
        else:
            raise ValueError("image_path must be a path string")

        if img_cv is None:
            self.logger.error(f"Could not read image: {image_path}")
            self.stats['failed_images'] += 1
            return None

        original_size = img_pil.size
        h, w = img_cv.shape[:2]
        self.logger.debug(f"Image loaded in {time.time() - load_start:.3f}s - Size: {w}x{h}")

        # Speech-bubble detection
        bubble_detect_start = time.time()
        bubbles = self._detect_bubbles(img_cv)
        self.logger.info(
            f"Speech bubbles detected: {len(bubbles)} (conf>={self.bubble_confidence}) in {time.time() - bubble_detect_start:.2f}s"
        )

        # OCR Detection - Primary pass (Korean + English)
        ocr_start = time.time()
        results = self._run_ocr(img_cv, self.reader, "primary (ko+en)")
        upgraded_indices = set()
        
        # OCR Detection - Secondary pass (Thai for stylized text)
        if self.reader_secondary is not None:
            results_secondary = self._run_ocr(img_cv, self.reader_secondary, f"secondary ({self.secondary_language})")
            # Filter secondary results by secondary confidence and merge
            results_secondary_filtered = [
                (bbox, text, prob) for bbox, text, prob in results_secondary 
                if prob >= self.secondary_confidence
            ]
            results, upgraded_indices = self._merge_results(results, results_secondary_filtered)
        
        ocr_time = time.time() - ocr_start
        self.logger.debug(f"Total OCR completed in {ocr_time:.3f}s - Found {len(results)} text regions")
        
        # Analyze and log all detections
        self.logger.debug("-" * 40)
        self.logger.debug("DETECTION DETAILS:")
        
        # Create mask and debug visualization
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Create comprehensive debug image
        if self.debug:
            debug_img = img_cv.copy()
            debug_overlay = img_cv.copy()
            # Draw detected speech bubbles in blue for reference
            for b in bubbles:
                x1, y1, x2, y2 = map(int, b['bbox'])
                cv2.rectangle(debug_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(
                    debug_img,
                    f"B:{b['conf']:.2f}",
                    (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 0, 0),
                    1
                )

        accepted_texts = []
        rejected_texts = []
        
        for i, (bbox, text, prob) in enumerate(results):
            pts = np.array(bbox, dtype=np.int32)
            
            # Calculate bounding box area
            x_coords = [p[0] for p in bbox]
            y_coords = [p[1] for p in bbox]
            bbox_w = max(x_coords) - min(x_coords)
            bbox_h = max(y_coords) - min(y_coords)
            area = bbox_w * bbox_h
            
            # Truncate long text for logging
            text_preview = text[:30] + "..." if len(text) > 30 else text
            
            # Use different thresholds: secondary_confidence for upgraded regions, primary for others
            if i in upgraded_indices:
                threshold = self.secondary_confidence
            else:
                threshold = self.confidence

            inside_bubble = self._text_inside_bubble(bbox, bubbles)

            if prob >= threshold and inside_bubble:
                accepted_texts.append({
                    'text': text, 
                    'prob': prob, 
                    'area': area,
                    'bbox': bbox
                })
                cv2.fillPoly(mask, [pts], 255)
                self.stats['total_texts_found'] += 1
                
                status = "✓ ACCEPTED"
                
                if self.debug:
                    # Green for accepted
                    cv2.polylines(debug_img, [pts], True, (0, 255, 0), 2)
                    cv2.fillPoly(debug_overlay, [pts], (0, 255, 0))
                    label = f"{prob:.2f}: {text_preview}"
                    cv2.putText(debug_img, label, (int(min(x_coords)), int(min(y_coords) - 5)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            else:
                reason = "below threshold" if prob < threshold else "outside bubble"
                rejected_texts.append({
                    'text': text, 
                    'prob': prob, 
                    'area': area,
                    'bbox': bbox,
                    'reason': reason
                })
                status = f"✗ REJECTED ({reason})"
                
                if self.debug:
                    # Red for rejected (still visible for debugging)
                    cv2.polylines(debug_img, [pts], True, (0, 0, 255), 1)
                    label = f"{prob:.2f}"
                    cv2.putText(debug_img, label, (int(min(x_coords)), int(min(y_coords) - 5)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
            
            self.logger.debug(f"  [{i:03d}] {status} | conf={prob:.3f} | area={area:>6}px | text=\"{text_preview}\"")
        
        self.logger.debug("-" * 40)
        self.logger.info(
            f"Detection summary: {len(accepted_texts)} accepted, {len(rejected_texts)} rejected "
            f"(text_conf={self.confidence}, bubble_conf={self.bubble_confidence})"
        )
        
        # Calculate debug output path
        debug_base = None
        if self.debug:
            if output_path:
                debug_base = Path(output_path).parent / "debug"
            elif self.debug_dir:
                debug_base = self.debug_dir
            else:
                debug_base = Path(image_path).parent / "debug"
            
            debug_base.mkdir(parents=True, exist_ok=True)
            stem = Path(image_path).stem
        
        # Save debug detection image
        if self.debug and debug_base:
            # Blend overlay
            alpha = 0.3
            debug_blended = cv2.addWeighted(debug_overlay, alpha, debug_img, 1 - alpha, 0)
            
            debug_boxes_path = debug_base / f"{stem}_01_detections.png"
            cv2.imwrite(str(debug_boxes_path), debug_blended)
            self.logger.debug(f"Saved detection debug: {debug_boxes_path}")
            
        if not accepted_texts:
            self.logger.info("No text found above confidence threshold - returning original image")
            if output_path:
                img_pil.save(output_path)
            self.stats['processing_times'].append(time.time() - process_start)
            return img_pil

        # Mask processing
        self.logger.debug(f"Creating mask with dilate_size={self.dilate_size}")
        
        mask_pixels_before = np.sum(mask > 0)
        
        if self.dilate_size > 0:
            kernel = np.ones((self.dilate_size, self.dilate_size), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)
        
        mask_pixels_after = np.sum(mask > 0)
        mask_coverage = (mask_pixels_after / (h * w)) * 100
        
        self.logger.debug(f"Mask stats: {mask_pixels_before} -> {mask_pixels_after} pixels ({mask_coverage:.2f}% of image)")

        # Save debug mask
        if self.debug and debug_base:
            # Create colored mask visualization
            mask_viz = np.zeros((h, w, 3), dtype=np.uint8)
            mask_viz[mask > 0] = [0, 255, 255]  # Yellow for mask areas
            
            debug_mask_path = debug_base / f"{stem}_02_mask.png"
            cv2.imwrite(str(debug_mask_path), mask)
            self.logger.debug(f"Saved mask debug: {debug_mask_path}")
            
            # Mask overlay on original
            mask_overlay = img_cv.copy()
            mask_overlay[mask > 0] = [0, 255, 255]
            mask_blended = cv2.addWeighted(mask_overlay, 0.4, img_cv, 0.6, 0)
            debug_mask_overlay_path = debug_base / f"{stem}_03_mask_overlay.png"
            cv2.imwrite(str(debug_mask_overlay_path), mask_blended)
            self.logger.debug(f"Saved mask overlay debug: {debug_mask_overlay_path}")

        mask_pil = Image.fromarray(mask).convert("L")
        
        # Big-LaMa Inpainting
        self.logger.debug("Starting Big-LaMa inpainting...")
        lama_start = time.time()
        
        result_pil = None
        max_retries = 3
        for attempt in range(max_retries):
            try:
                result_pil = self.lama(img_pil, mask_pil)
                self.logger.debug(f"Big-LaMa completed in {time.time() - lama_start:.3f}s")
                break
            except Exception as e:
                self.logger.warning(f"Big-LaMa inpainting failed (attempt {attempt+1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    import traceback
                    self.logger.error(traceback.format_exc())
        
        if result_pil is None:
            self.logger.error("Big-LaMa inpainting failed after all retries")
            self.stats['failed_images'] += 1
            return None
        
        self.stats['total_texts_removed'] += len(accepted_texts)

        # Size correction
        if result_pil.size != original_size:
            w_orig, h_orig = original_size
            w_res, h_res = result_pil.size
            self.logger.debug(f"Size mismatch: result={w_res}x{h_res}, original={w_orig}x{h_orig}")
            
            if w_res >= w_orig and h_res >= h_orig:
                self.logger.info(f"Cropping result from {result_pil.size} to {original_size}")
                result_pil = result_pil.crop((0, 0, w_orig, h_orig))
            else:
                self.logger.warning(f"Result smaller than original. Resizing with LANCZOS.")
                result_pil = result_pil.resize(original_size, Image.LANCZOS)

        # Save result
        if output_path:
            result_pil.save(output_path)
            self.logger.info(f"Saved result: {output_path}")
        
        # Save debug comparison
        if self.debug and debug_base:
            # Side-by-side comparison
            result_cv = cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR)
            comparison = np.hstack([img_cv, result_cv])
            debug_compare_path = debug_base / f"{stem}_04_comparison.png"
            cv2.imwrite(str(debug_compare_path), comparison)
            self.logger.debug(f"Saved comparison debug: {debug_compare_path}")
            
            # Difference image
            diff = cv2.absdiff(img_cv, result_cv)
            debug_diff_path = debug_base / f"{stem}_05_diff.png"
            cv2.imwrite(str(debug_diff_path), diff)
            self.logger.debug(f"Saved diff debug: {debug_diff_path}")
            
        total_time = time.time() - process_start
        self.stats['processing_times'].append(total_time)
        self.logger.info(f"Completed in {total_time:.2f}s | Removed {len(accepted_texts)} text regions")
        
        return result_pil
    
    def get_stats(self):
        """Return processing statistics."""
        avg_time = sum(self.stats['processing_times']) / len(self.stats['processing_times']) if self.stats['processing_times'] else 0
        return {
            **self.stats,
            'avg_processing_time': avg_time
        }
    
    def print_summary(self):
        """Print summary of all processing."""
        stats = self.get_stats()
        self.logger.info("=" * 60)
        self.logger.info("BLASTER SUMMARY")
        self.logger.info(f"  Total images processed: {stats['total_images']}")
        self.logger.info(f"  Total texts found: {stats['total_texts_found']}")
        self.logger.info(f"  Total texts removed: {stats['total_texts_removed']}")
        self.logger.info(f"  Failed images: {stats['failed_images']}")
        self.logger.info(f"  Average processing time: {stats['avg_processing_time']:.2f}s")
        self.logger.info("=" * 60)

if __name__ == "__main__":
    blaster = Blaster(debug=True, debug_dir="./debug_output")
    print("Blaster initialized with Big-LaMa in debug mode")
