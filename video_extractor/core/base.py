#!/usr/bin/env python3
"""
Base Video Highlights Extractor
Common functionality shared across all extraction modes.
"""

from datetime import datetime
import os
import subprocess
import json
from pathlib import Path
from typing import List, Tuple, Dict
import cv2
import numpy as np
from moviepy import VideoFileClip
import logging

# Fix tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class BaseHighlightExtractor:
    """Base class for video highlight extraction"""

    def __init__(
        self,
        video_path: str,
        min_duration: int = 30,
        max_highlights: int = None,
        mode: str = "basic",
        content_tags: list = None,
    ):
        self.video_path = video_path
        self.min_duration = min_duration
        self.max_highlights = max_highlights
        self.mode = mode
        self.content_tags = content_tags  # Can be None for generic behavior
        self.output_dir = (
            Path(video_path).parent / f"{Path(video_path).stem}_highlights"
        )
        self.output_dir.mkdir(exist_ok=True)

        # Get video info
        self.cap = cv2.VideoCapture(self.video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps

        logger.info(
            f"Video: {self.duration:.1f}s, {self.fps:.1f}fps, {self.total_frames} frames"
        )

    def __del__(self):
        """Clean up video capture"""
        if hasattr(self, "cap") and self.cap:
            self.cap.release()

    def analyze_video_content(self) -> List[Tuple[float, float, float]]:
        """
        Analyze video to find highlights.
        Must be implemented by subclasses.
        Returns list of (start_time, end_time, score) tuples
        """
        raise NotImplementedError("Subclasses must implement analyze_video_content")

    def _calculate_max_highlights(self) -> int:
        """Calculate maximum number of highlights based on video duration"""
        if self.max_highlights:
            return self.max_highlights

        # Auto-calculate: one highlight per 2-3 minutes of video
        auto_max = max(1, int(self.duration / 150))  # 150 seconds = 2.5 minutes
        return min(auto_max, 10)  # Cap at 10 highlights

    def _merge_overlapping_segments(
        self, segments: List[Tuple[float, float, float]]
    ) -> List[Tuple[float, float, float]]:
        """Merge only truly overlapping segments, not close ones"""
        if not segments:
            return []

        # Sort by start time
        segments = sorted(segments, key=lambda x: x[0])
        merged = [segments[0]]

        for current in segments[1:]:
            last = merged[-1]

            # Only merge if actually overlapping (not just close)
            if current[0] < last[1]:
                # Merge segments, keeping the higher score
                new_end = max(last[1], current[1])
                new_score = max(last[2], current[2])
                merged[-1] = (last[0], new_end, new_score)
            else:
                merged.append(current)

        return merged

    def _ensure_minimum_duration(
        self, segments: List[Tuple[float, float, float]]
    ) -> List[Tuple[float, float, float]]:
        """Ensure all segments meet minimum duration requirement"""
        adjusted = []

        for start, end, score in segments:
            duration = end - start

            if duration < self.min_duration:
                # Extend the segment to meet minimum duration
                extension_needed = self.min_duration - duration

                # Try to extend equally in both directions
                new_start = max(0, start - extension_needed / 2)
                new_end = min(self.duration, end + extension_needed / 2)

                # If we hit a boundary, extend more in the other direction
                if new_start == 0:
                    new_end = min(self.duration, start + self.min_duration)
                elif new_end == self.duration:
                    new_start = max(0, end - self.min_duration)

                adjusted.append((new_start, new_end, score))
            else:
                adjusted.append((start, end, score))

        return adjusted

    def _select_best_highlights(
        self, segments: List[Tuple[float, float, float]]
    ) -> List[Tuple[float, float, float]]:
        """Select the best highlights avoiding overlaps"""
        if not segments:
            return []

        # Sort by score (descending)
        segments = sorted(segments, key=lambda x: x[2], reverse=True)

        selected = []
        max_highlights = self._calculate_max_highlights()

        for segment in segments:
            if len(selected) >= max_highlights:
                break

            # Check if this segment overlaps with any already selected
            overlaps = False
            for selected_segment in selected:
                if (
                    segment[0] < selected_segment[1]
                    and segment[1] > selected_segment[0]
                ):
                    overlaps = True
                    break

            if not overlaps:
                selected.append(segment)

        # Sort selected highlights by start time
        return sorted(selected, key=lambda x: x[0])

    def create_highlight_videos(
        self, highlights: List[Tuple[float, float, float]]
    ) -> List[str]:
        """Create highlight videos from segments"""
        output_files = []

        for i, (start_time, end_time, score) in enumerate(highlights, 1):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = (
                self.output_dir / f"{self.mode}_highlight_{i:02d}_{timestamp}.mp4"
            )

            logger.info(
                f"Creating highlight {i}/{len(highlights)}: {start_time:.1f}s-{end_time:.1f}s (score: {score:.3f})"
            )

            # Use FFmpeg to extract and crop the segment
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(self.video_path),
                "-ss",
                str(start_time),
                "-t",
                str(end_time - start_time),
                "-vf",
                "crop=ih*9/16:ih:(iw-ih*9/16)/2:0,scale=1080:1920",
                "-c:v",
                "libx264",
                "-c:a",
                "aac",
                "-preset",
                "medium",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",  # Force 8-bit for QuickTime compatibility
                "-profile:v",
                "high",  # H.264 High profile
                "-level:v",
                "4.0",  # H.264 level 4.0
                str(output_file),
            ]

            try:
                subprocess.run(cmd, check=True, capture_output=True)
                output_files.append(str(output_file))
                logger.info(f"✓ Created: {output_file.name}")
            except subprocess.CalledProcessError as e:
                logger.error(f"✗ Failed to create {output_file.name}: {e}")

        return output_files

    def extract_highlights(self) -> List[str]:
        """Main method to extract highlights from video"""
        logger.info(f"Starting highlight extraction from: {self.video_path}")
        logger.info(
            f"Min duration: {self.min_duration}s, Max highlights: {self._calculate_max_highlights()}"
        )

        # Analyze video content (implemented by subclasses)
        segments = self.analyze_video_content()

        if not segments:
            logger.warning("No highlights found in video")
            return []

        logger.info(f"Found {len(segments)} potential highlight segments")

        # Process segments
        segments = self._merge_overlapping_segments(segments)
        segments = self._ensure_minimum_duration(segments)
        highlights = self._select_best_highlights(segments)

        logger.info(f"Selected {len(highlights)} final highlights")

        # Create highlight videos
        output_files = self.create_highlight_videos(highlights)

        logger.info(
            f"✓ Extraction complete! Created {len(output_files)} highlights in: {self.output_dir}"
        )
        return output_files
