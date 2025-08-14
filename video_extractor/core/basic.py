#!/usr/bin/env python3
"""
Basic Video Highlights Extractor
Uses audio energy analysis and visual motion detection.
"""

import os
from typing import List, Tuple
import cv2
import numpy as np
import librosa
from scipy.signal import find_peaks
import logging

from .base import BaseHighlightExtractor

# Fix tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)

class BasicHighlightExtractor(BaseHighlightExtractor):
    """Basic highlight extractor using audio and visual analysis"""
    
    def __init__(self, video_path: str, min_duration: int = 30, max_highlights: int = None, mode: str = "basic"):
        super().__init__(video_path, min_duration, max_highlights, mode)
    
    def analyze_video_content(self) -> List[Tuple[float, float, float]]:
        """
        Analyze video to find highlights based on:
        - Audio energy/volume spikes
        - Visual motion/activity
        - Scene changes
        Returns list of (start_time, end_time, score) tuples
        """
        logger.info("Analyzing video content for highlights...")
        
        # Audio analysis
        audio_highlights = self._analyze_audio()
        
        # Visual analysis
        visual_highlights = self._analyze_visual_activity()
        
        # Combine and score highlights
        highlights = self._combine_highlights(audio_highlights, visual_highlights)
        
        return highlights
    
    def _analyze_audio(self) -> List[Tuple[float, float, float]]:
        """Analyze audio for energy spikes and interesting moments"""
        try:
            logger.info("Loading audio for analysis...")
            # Extract audio using librosa with lower sample rate for faster processing
            y, sr = librosa.load(self.video_path, sr=22050)  # Downsample to 22kHz
            logger.info(f"Audio loaded: {len(y)/sr:.1f}s duration at {sr}Hz")
            
            # Calculate RMS energy in windows
            hop_length = sr // 4  # 0.25 second windows
            rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
            
            # Find peaks in audio energy with better parameters
            # Use lower threshold to find more potential highlights
            threshold = np.percentile(rms, 70)  # Lower threshold for more highlights
            min_distance = sr // hop_length  # 1 second minimum distance
            peaks, properties = find_peaks(rms, height=threshold, distance=min_distance)
            logger.info(f"Found {len(peaks)} audio peaks")
            
            # Convert peaks to non-overlapping segments
            highlights = []
            segment_duration = 15  # 15-second segments
            
            for peak in peaks:
                peak_time = peak * hop_length / sr
                start_time = max(0, peak_time - segment_duration/2)
                end_time = min(len(y)/sr, peak_time + segment_duration/2)
                score = rms[peak]
                highlights.append((start_time, end_time, score))
            
            return highlights
            
        except Exception as e:
            logger.warning(f"Audio analysis failed: {e}")
            return []
    
    def _analyze_visual_activity(self) -> List[Tuple[float, float, float]]:
        """Analyze visual content for motion and scene changes"""
        highlights = []
        frame_count = 0
        prev_frame = None
        motion_scores = []
        
        # Sample every 60 frames for better performance on large files
        sample_rate = 60
        
        logger.info("Analyzing visual motion...")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Skip frames for performance
            if frame_count % sample_rate != 0:
                continue
            
            # Convert to grayscale and resize for faster processing
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (320, 180))  # Much smaller for speed
            
            if prev_frame is not None:
                # Calculate frame difference
                diff = cv2.absdiff(prev_frame, gray)
                motion_score = np.mean(diff)
                motion_scores.append((frame_count / self.fps, motion_score))
            
            prev_frame = gray
        
        if motion_scores:
            # Find motion peaks
            scores = [score for _, score in motion_scores]
            threshold = np.percentile(scores, 80)  # Top 20% of motion
            
            for timestamp, score in motion_scores:
                if score > threshold:
                    start_time = max(0, timestamp - 10)  # 10s before motion peak
                    end_time = min(self.duration, timestamp + 10)  # 10s after motion peak
                    highlights.append((start_time, end_time, score / 100))  # Normalize score
        
        logger.info(f"Found {len(highlights)} visual motion highlights")
        return highlights
    
    def _combine_highlights(self, audio_highlights: List[Tuple[float, float, float]], 
                          visual_highlights: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
        """Combine audio and visual highlights with weighted scoring"""
        
        all_highlights = []
        
        # Add audio highlights with weight
        for start, end, score in audio_highlights:
            all_highlights.append((start, end, score * 0.7))  # Audio weight: 70%
        
        # Add visual highlights with weight
        for start, end, score in visual_highlights:
            all_highlights.append((start, end, score * 0.3))  # Visual weight: 30%
        
        # Sort by score
        all_highlights.sort(key=lambda x: x[2], reverse=True)
        
        logger.info(f"Combined {len(audio_highlights)} audio + {len(visual_highlights)} visual = {len(all_highlights)} total highlights")
        
        return all_highlights