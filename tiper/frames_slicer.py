#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Frames Slicer Module

Secondary slicing pass for creating frames with different height settings.
Used for creating reader-friendly frame outputs separate from the cleaning pipeline.
"""

import sys
from pathlib import Path
from typing import List, Optional
import logging

import numpy as np
from PIL import Image

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'cutting'))

try:
    from cutting.smart_slicer import SmartSlicer
except ImportError:
    from smart_slicer import SmartSlicer


def setup_frames_logger():
    """Setup logging for frames slicer."""
    logger = logging.getLogger('frames_slicer')
    logger.setLevel(logging.DEBUG)
    
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter('[FRAMES] %(message)s')
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)
    
    return logger


class FramesSlicer:
    """Secondary slicing pass for creating frames with different height settings."""
    
    def __init__(self, config: dict):
        frames_cfg = config.get('frames', {})
        self.enabled = frames_cfg.get('enabled', False)
        self.min_height = frames_cfg.get('min_height', 4000)
        self.max_height = frames_cfg.get('max_height', 7800)
        self.output_subdir = frames_cfg.get('output_subdir', 'frames')
        self.format = frames_cfg.get('format', 'png')  # png or jpg
        self.quality = frames_cfg.get('quality', 95)  # for jpg
        
        self.logger = setup_frames_logger()
        
        if self.enabled:
            self.logger.info(f"FramesSlicer initialized:")
            self.logger.info(f"  - Min height: {self.min_height}")
            self.logger.info(f"  - Max height: {self.max_height}")
            self.logger.info(f"  - Output subdir: {self.output_subdir}")
            self.logger.info(f"  - Format: {self.format}")
    
    def compute_frame_cuts(
        self,
        img_height: int,
        cost_map: np.ndarray,
        slicer: SmartSlicer
    ) -> List[int]:
        """
        Compute cut positions for frames using the SmartSlicer.
        Uses different min/max height than the main slicing pass.
        """
        preferred_h = (self.min_height + self.max_height) // 2
        cuts = [0]
        current = 0
        
        self.logger.debug(f"Computing frame cuts: min={self.min_height}, max={self.max_height}, preferred={preferred_h}")
        
        while True:
            remaining = img_height - current
            if remaining <= self.max_height:
                cuts.append(img_height)
                break
            
            cut_y = slicer.find_best_cut(
                cost_map=cost_map,
                current_y=current,
                min_h=self.min_height,
                max_h=self.max_height,
                preferred_h=preferred_h
            )
            
            # Safety check
            if cut_y <= current:
                self.logger.warning(f"SmartSlicer returned {cut_y} <= current {current}. Forcing min_h advance.")
                cut_y = current + self.min_height
            
            cut_y = min(cut_y, img_height - 1)
            cuts.append(cut_y)
            current = cut_y
        
        # Deduplicate
        dedup_cuts = []
        for c in cuts:
            if not dedup_cuts or c != dedup_cuts[-1]:
                dedup_cuts.append(c)
        
        self.logger.info(f"Frame cuts: {len(dedup_cuts) - 1} frames")
        
        return dedup_cuts
    
    def slice_frames(
        self,
        image: Image.Image,
        chapter_folder: Path,
        cost_map: Optional[np.ndarray] = None,
        boxes: Optional[List] = None,
        margin: int = 20
    ) -> List[Path]:
        """
        Slice image into frames and save to chapter_folder/frames/
        
        Args:
            image: PIL Image to slice
            chapter_folder: Path to chapter folder
            cost_map: Pre-computed cost map (optional, will compute if not provided)
            boxes: Detection boxes for cost map computation (needed if cost_map is None)
            margin: Margin for forbidden zones
        
        Returns:
            List of saved frame paths
        """
        if not self.enabled:
            return []
        
        width, height = image.size
        self.logger.info(f"Slicing frames from image {width}x{height}")
        
        # Create output directory
        output_dir = chapter_folder / self.output_subdir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize SmartSlicer for frames
        slicer = SmartSlicer(
            edge_weight=1.0,
            variance_weight=0.5,
            gradient_weight=0.5,
            white_space_weight=2.0,
            distance_penalty=0.0001,
            forbidden_cost=1e9
        )
        
        # Compute or reuse cost map
        if cost_map is None:
            self.logger.info("Computing cost map for frames...")
            if boxes is None:
                boxes = []
            cost_map = slicer.compute_cost_map(image, boxes, margin=margin)
        else:
            self.logger.info("Reusing existing cost map for frames")
        
        # Compute frame cuts with frame-specific heights
        cuts = self.compute_frame_cuts(height, cost_map, slicer)
        
        # Save frames
        saved_paths = []
        image_rgb = image.convert("RGB")
        
        for i in range(len(cuts) - 1):
            top = cuts[i]
            bottom = cuts[i + 1]
            
            if bottom <= top:
                continue
            
            crop = image_rgb.crop((0, top, width, bottom))
            
            if self.format.lower() == 'jpg' or self.format.lower() == 'jpeg':
                frame_path = output_dir / f"frame_{i+1:03d}.jpg"
                crop.save(frame_path, format="JPEG", quality=self.quality, optimize=True)
            else:
                frame_path = output_dir / f"frame_{i+1:03d}.png"
                crop.save(frame_path, format="PNG", optimize=True)
            
            saved_paths.append(frame_path)
            self.logger.debug(f"Saved frame {i+1}: {frame_path.name} (height {bottom - top})")
        
        self.logger.info(f"Saved {len(saved_paths)} frames to {output_dir}")
        
        return saved_paths


if __name__ == "__main__":
    # Test
    test_config = {
        'frames': {
            'enabled': True,
            'min_height': 4000,
            'max_height': 7800,
            'output_subdir': 'frames',
            'format': 'png'
        }
    }
    
    slicer = FramesSlicer(test_config)
    print(f"FramesSlicer initialized: enabled={slicer.enabled}")
