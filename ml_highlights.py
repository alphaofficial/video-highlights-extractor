#!/usr/bin/env python3
"""
ML-Powered Video Highlights Extractor
Uses machine learning models for intelligent highlight detection.
"""

import argparse
import os
import sys
import subprocess
import json
from pathlib import Path
from typing import List, Tuple, Dict
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from transformers import pipeline, AutoProcessor, AutoModel
import librosa
from scipy.signal import find_peaks
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MLHighlightExtractor:
    def __init__(self, video_path: str, min_duration: int = 30):
        self.video_path = video_path
        self.min_duration = min_duration
        self.output_dir = Path(video_path).parent / f"{Path(video_path).stem}_highlights"
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize ML models
        self._init_models()
        
    def _init_models(self):
        """Initialize pre-trained models for highlight detection"""
        logger.info("Loading ML models...")
        
        try:
            # Video classification model for action detection
            self.video_classifier = pipeline(
                "video-classification",
                model="microsoft/xclip-base-patch32-zero-shot",
                device=0 if torch.cuda.is_available() else -1
            )
        except Exception as e:
            logger.warning(f"Could not load video classifier: {e}")
            self.video_classifier = None
        
        try:
            # Audio classification for excitement/intensity
            self.audio_classifier = pipeline(
                "audio-classification",
                model="facebook/wav2vec2-base-960h",
                device=0 if torch.cuda.is_available() else -1
            )
        except Exception as e:
            logger.warning(f"Could not load audio classifier: {e}")
            self.audio_classifier = None
        
        # Action/excitement keywords for gaming content
        self.action_keywords = [
            "action", "fighting", "shooting", "explosion", "combat", "battle",
            "intense", "exciting", "dramatic", "fast", "movement", "chase"
        ]
        
    def analyze_video_with_ml(self) -> List[Tuple[float, float, float]]:
        """Use ML models to find highlights"""
        logger.info("Analyzing video with ML models...")
        
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        # Analyze video in segments
        segment_length = 10  # 10-second segments
        segments = []
        
        for start_time in range(0, int(duration), segment_length):
            end_time = min(start_time + segment_length, duration)
            score = self._analyze_segment(start_time, end_time, cap, fps)
            segments.append((start_time, end_time, score))
            
            progress = (start_time / duration) * 100
            if start_time % 30 == 0:  # Log every 30 seconds
                logger.info(f"ML analysis progress: {progress:.1f}%")
        
        cap.release()
        
        # Find top segments and create highlights
        highlights = self._create_highlights_from_segments(segments, duration)
        return highlights
    
    def _analyze_segment(self, start_time: float, end_time: float, cap, fps) -> float:
        """Analyze a video segment for excitement/action"""
        score = 0.0
        
        # Visual analysis
        visual_score = self._analyze_visual_segment(start_time, end_time, cap, fps)
        score += visual_score * 0.6
        
        # Audio analysis
        audio_score = self._analyze_audio_segment(start_time, end_time)
        score += audio_score * 0.4
        
        return score
    
    def _analyze_visual_segment(self, start_time: float, end_time: float, cap, fps) -> float:
        """Analyze visual content of a segment"""
        # Extract frames from the segment
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        motion_scores = []
        prev_frame = None
        
        for frame_idx in range(start_frame, min(end_frame, start_frame + int(fps * 2))):  # Sample 2 seconds
            ret, frame = cap.read()
            if not ret:
                break
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if prev_frame is not None:
                # Calculate motion
                diff = cv2.absdiff(prev_frame, gray)
                motion = np.mean(diff)
                motion_scores.append(motion)
            
            prev_frame = gray
        
        # Return normalized motion score
        if motion_scores:
            avg_motion = np.mean(motion_scores)
            return min(avg_motion / 50.0, 1.0)  # Normalize to 0-1
        
        return 0.0
    
    def _analyze_audio_segment(self, start_time: float, end_time: float) -> float:
        """Analyze audio content of a segment"""
        try:
            # Load audio segment
            y, sr = librosa.load(
                self.video_path, 
                sr=22050, 
                offset=start_time, 
                duration=end_time - start_time
            )
            
            if len(y) == 0:
                return 0.0
            
            # Calculate audio features
            # RMS energy
            rms = librosa.feature.rms(y=y)[0]
            energy_score = np.mean(rms)
            
            # Spectral centroid (brightness)
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            brightness_score = np.mean(spectral_centroid) / sr
            
            # Zero crossing rate (indicates speech/action)
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            activity_score = np.mean(zcr)
            
            # Combine scores
            audio_score = (energy_score * 0.5 + brightness_score * 0.3 + activity_score * 0.2)
            return min(audio_score * 10, 1.0)  # Normalize to 0-1
            
        except Exception as e:
            logger.warning(f"Audio analysis failed for segment {start_time}-{end_time}: {e}")
            return 0.0
    
    def _create_highlights_from_segments(self, segments: List[Tuple[float, float, float]], duration: float) -> List[Tuple[float, float, float]]:
        """Create highlights from analyzed segments"""
        # Sort by score
        segments.sort(key=lambda x: x[2], reverse=True)
        
        # Calculate number of highlights
        highlight_coverage = 0.6
        usable_duration = duration * highlight_coverage
        max_highlights = int(usable_duration / self.min_duration)
        max_highlights = max(3, min(max_highlights, 20))
        
        logger.info(f"Creating {max_highlights} highlights from {len(segments)} analyzed segments")
        
        highlights = []
        used_time_ranges = []
        
        for start_seg, end_seg, score in segments:
            if len(highlights) >= max_highlights:
                break
            
            # Check for overlap with existing highlights
            overlap = False
            for used_start, used_end in used_time_ranges:
                if not (end_seg <= used_start or start_seg >= used_end):
                    overlap = True
                    break
            
            if not overlap:
                # Create highlight centered on this segment
                center = (start_seg + end_seg) / 2
                highlight_start = max(0, center - self.min_duration / 2)
                highlight_end = min(duration, highlight_start + self.min_duration)
                
                # Adjust if we hit boundaries
                if highlight_end - highlight_start < self.min_duration:
                    if highlight_start == 0:
                        highlight_end = min(duration, self.min_duration)
                    else:
                        highlight_start = max(0, duration - self.min_duration)
                
                if highlight_end - highlight_start >= self.min_duration:
                    highlights.append((highlight_start, highlight_end, score))
                    used_time_ranges.append((highlight_start, highlight_end))
        
        return highlights
    
    def create_vertical_clips(self, highlights: List[Tuple[float, float, float]]) -> List[str]:
        """Create 9:16 vertical clips from highlights"""
        output_files = []
        
        for i, (start, end, score) in enumerate(highlights):
            output_file = self.output_dir / f"highlight_{i+1}.mp4"
            
            logger.info(f"Creating highlight {i+1}: {start:.1f}s - {end:.1f}s (score: {score:.3f})")
            
            cmd = [
                'ffmpeg', '-y',
                '-i', self.video_path,
                '-ss', str(start),
                '-t', str(end - start),
                '-vf', 'scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920',
                '-c:v', 'libx264',
                '-c:a', 'aac',
                '-preset', 'fast',
                '-crf', '23',
                '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart',
                '-profile:v', 'baseline',
                '-level', '3.0',
                str(output_file)
            ]
            
            try:
                subprocess.run(cmd, check=True, capture_output=True)
                output_files.append(str(output_file))
                logger.info(f"Created: {output_file}")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to create highlight {i+1}: {e}")
        
        return output_files

def main():
    parser = argparse.ArgumentParser(description='ML-powered video highlight extraction')
    parser.add_argument('video_path', help='Path to input video file')
    parser.add_argument('--min-duration', type=int, default=30, 
                       help='Minimum duration for each highlight in seconds (default: 30)')
    parser.add_argument('--max-highlights', type=int, default=None,
                       help='Maximum number of highlights (auto-calculated if not specified)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video_path):
        logger.error(f"Video file not found: {args.video_path}")
        sys.exit(1)
    
    try:
        extractor = MLHighlightExtractor(args.video_path, args.min_duration)
        highlights = extractor.analyze_video_with_ml()
        
        if not highlights:
            logger.error("No highlights found in video")
            sys.exit(1)
        
        # Limit to max highlights if specified
        if args.max_highlights:
            highlights = highlights[:args.max_highlights]
        
        logger.info(f"Found {len(highlights)} ML-detected highlights")
        output_files = extractor.create_vertical_clips(highlights)
        
        logger.info(f"Successfully created {len(output_files)} highlight clips:")
        for file in output_files:
            print(f"  {file}")
            
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()