#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch Processor Module

Handles batch processing of multiple chapter folders for manhwa text removal pipeline.
Supports:
- Finding chapter folders by pattern (ch*, chapter*)
- Renaming original images to chXXX_original.png
- Cleaning up artifacts (slices/, debug/) after processing
"""

import re
import shutil
from pathlib import Path
from typing import List, Tuple, Optional, Callable
import logging


def setup_batch_logger():
    """Setup logging for batch processor."""
    logger = logging.getLogger('batch_processor')
    logger.setLevel(logging.DEBUG)
    
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter('[BATCH] %(message)s')
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)
    
    return logger


class BatchProcessor:
    """Handles batch processing of multiple chapter folders."""
    
    def __init__(self, config: dict):
        batch_cfg = config.get('batch', {})
        self.chapters_dir = Path(batch_cfg.get('chapters_dir', 'chapters'))
        self.patterns = [p.strip() for p in batch_cfg.get('chapter_pattern', 'ch*,chapter*').split(',')]
        self.cleanup_enabled = batch_cfg.get('cleanup_artifacts', True)
        self.logger = setup_batch_logger()
        
        self.logger.info(f"BatchProcessor initialized:")
        self.logger.info(f"  - Chapters dir: {self.chapters_dir}")
        self.logger.info(f"  - Patterns: {self.patterns}")
        self.logger.info(f"  - Cleanup artifacts: {self.cleanup_enabled}")
    
    def find_chapters(self) -> List[Path]:
        """
        Find all chapter folders matching patterns.
        Returns sorted list of folder paths.
        """
        if not self.chapters_dir.exists():
            self.logger.error(f"Chapters directory not found: {self.chapters_dir}")
            return []
        
        chapters = set()
        for pattern in self.patterns:
            matched = list(self.chapters_dir.glob(pattern))
            for m in matched:
                if m.is_dir():
                    chapters.add(m)
        
        # Sort by chapter number
        def sort_key(path: Path) -> int:
            num = self.get_chapter_number(path)
            try:
                return int(num)
            except ValueError:
                return 0
        
        sorted_chapters = sorted(chapters, key=sort_key)
        self.logger.info(f"Found {len(sorted_chapters)} chapter folders")
        
        return sorted_chapters
    
    def get_chapter_number(self, folder: Path) -> str:
        """
        Extract chapter number from folder name.
        Examples:
            ch166 → 166
            chapter165 → 165
            ch_170 → 170
        """
        name = folder.name.lower()
        # Remove common prefixes
        for prefix in ['chapter', 'ch_', 'ch']:
            if name.startswith(prefix):
                name = name[len(prefix):]
                break
        
        # Extract digits
        match = re.search(r'(\d+)', name)
        if match:
            return match.group(1)
        
        # Fallback to folder name
        return folder.name
    
    def get_chapter_prefix(self, folder: Path) -> str:
        """
        Get chapter prefix based on folder name format.
        Examples:
            ch166 → ch166
            chapter165 → chapter165
        """
        return folder.name
    
    def prepare_chapter(self, folder: Path) -> Tuple[Optional[Path], str]:
        """
        Prepare chapter for processing:
        1. Find the single .png file in the folder
        2. Rename to chXXX_original.png if needed
        3. Return (image_path, chapter_prefix)
        
        Returns (None, "") if preparation fails.
        """
        chapter_prefix = self.get_chapter_prefix(folder)
        
        # Find all PNG files (excluding _cleaned.png and frames)
        png_files = []
        for f in folder.iterdir():
            if f.is_file() and f.suffix.lower() == '.png':
                # Skip cleaned output and frame files
                if '_cleaned' not in f.stem and 'frame_' not in f.stem:
                    png_files.append(f)
        
        if len(png_files) == 0:
            self.logger.error(f"No PNG files found in {folder}")
            return None, ""
        
        if len(png_files) > 1:
            # Check if one is already the _original file
            original_files = [f for f in png_files if '_original' in f.stem]
            if len(original_files) == 1:
                self.logger.info(f"Found existing original: {original_files[0].name}")
                return original_files[0], chapter_prefix
            
            self.logger.error(f"Multiple PNG files found in {folder}: {[f.name for f in png_files]}")
            return None, ""
        
        # Single PNG file found
        png_file = png_files[0]
        expected_name = f"{chapter_prefix}_original.png"
        
        if png_file.name == expected_name:
            self.logger.info(f"File already named correctly: {png_file.name}")
            return png_file, chapter_prefix
        
        # Rename to _original.png
        new_path = folder / expected_name
        self.logger.info(f"Renaming: {png_file.name} → {expected_name}")
        png_file.rename(new_path)
        
        return new_path, chapter_prefix
    
    def cleanup_chapter(self, folder: Path) -> None:
        """
        Remove artifact directories after processing:
        - slices/
        - debug/
        - output/ (if exists inside chapter folder)
        
        Does NOT remove frames/ folder.
        """
        if not self.cleanup_enabled:
            self.logger.debug(f"Cleanup disabled, skipping {folder}")
            return
        
        artifact_dirs = ['slices', 'debug', 'output']
        
        for dirname in artifact_dirs:
            artifact_path = folder / dirname
            if artifact_path.exists() and artifact_path.is_dir():
                self.logger.info(f"Removing artifact directory: {artifact_path}")
                try:
                    shutil.rmtree(artifact_path)
                except Exception as e:
                    self.logger.error(f"Failed to remove {artifact_path}: {e}")
    
    def process_all(
        self,
        pipeline_func: Callable[[Path, str, Path], bool],
        start_from: Optional[int] = None,
        stop_at: Optional[int] = None
    ) -> dict:
        """
        Main batch processing loop.
        
        Args:
            pipeline_func: Function(image_path, chapter_prefix, chapter_folder) -> success
            start_from: Optional chapter number to start from (inclusive)
            stop_at: Optional chapter number to stop at (inclusive)
        
        Returns:
            dict with processing statistics
        """
        chapters = self.find_chapters()
        
        if not chapters:
            self.logger.error("No chapters to process")
            return {'total': 0, 'processed': 0, 'failed': 0, 'skipped': 0}
        
        stats = {
            'total': len(chapters),
            'processed': 0,
            'failed': 0,
            'skipped': 0,
            'failed_chapters': []
        }
        
        for i, chapter_folder in enumerate(chapters):
            chapter_num = self.get_chapter_number(chapter_folder)
            
            # Check range filters
            if start_from is not None:
                try:
                    if int(chapter_num) < start_from:
                        self.logger.info(f"Skipping {chapter_folder.name} (before start_from={start_from})")
                        stats['skipped'] += 1
                        continue
                except ValueError:
                    pass
            
            if stop_at is not None:
                try:
                    if int(chapter_num) > stop_at:
                        self.logger.info(f"Skipping {chapter_folder.name} (after stop_at={stop_at})")
                        stats['skipped'] += 1
                        continue
                except ValueError:
                    pass
            
            self.logger.info("=" * 60)
            self.logger.info(f"Processing chapter {i+1}/{len(chapters)}: {chapter_folder.name}")
            self.logger.info("=" * 60)
            
            # Prepare chapter
            image_path, chapter_prefix = self.prepare_chapter(chapter_folder)
            
            if image_path is None:
                self.logger.error(f"Failed to prepare chapter {chapter_folder.name}")
                stats['failed'] += 1
                stats['failed_chapters'].append(chapter_folder.name)
                continue
            
            # Run pipeline
            try:
                success = pipeline_func(image_path, chapter_prefix, chapter_folder)
                
                if success:
                    stats['processed'] += 1
                    # Cleanup artifacts
                    self.cleanup_chapter(chapter_folder)
                else:
                    stats['failed'] += 1
                    stats['failed_chapters'].append(chapter_folder.name)
                    
            except Exception as e:
                self.logger.error(f"Pipeline failed for {chapter_folder.name}: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                stats['failed'] += 1
                stats['failed_chapters'].append(chapter_folder.name)
        
        # Print summary
        self.logger.info("=" * 60)
        self.logger.info("BATCH PROCESSING COMPLETE")
        self.logger.info(f"  Total chapters: {stats['total']}")
        self.logger.info(f"  Processed: {stats['processed']}")
        self.logger.info(f"  Failed: {stats['failed']}")
        self.logger.info(f"  Skipped: {stats['skipped']}")
        if stats['failed_chapters']:
            self.logger.info(f"  Failed chapters: {stats['failed_chapters']}")
        self.logger.info("=" * 60)
        
        return stats


if __name__ == "__main__":
    # Test run
    test_config = {
        'batch': {
            'chapters_dir': 'chapters',
            'chapter_pattern': 'ch*,chapter*',
            'cleanup_artifacts': True
        }
    }
    
    processor = BatchProcessor(test_config)
    chapters = processor.find_chapters()
    
    for ch in chapters:
        print(f"  {ch.name} → number: {processor.get_chapter_number(ch)}")
