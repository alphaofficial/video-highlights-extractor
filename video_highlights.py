#!/usr/bin/env python3
"""
Video Highlights Extractor CLI
Extracts 9:16 vertical highlights from videos using AI-powered scene detection.
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
from moviepy.editor import VideoFileClip
import librosa
from sklearn.cluster import KMeans
from scipy.signal import find_peaks
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VideoHighlightExtractor:
    def __init__(self, video_path: str, min_duration: int = 90):
        self.video_path = video_path
        self.min_duration = min_duration
        self.output_dir = Path(video_path).parent / f"{Path(video_path).stem}_highlights"
        self.output_dir.mkdir(exist_ok=True)
        
    def analyze_video_content(self) -> List[Tuple[float, float, float]]:
        """
        Analyze video to find highlights based on:
        - Audio energy/volume spikes
        - Visual motion/activity
        - Scene changes
        Returns list of (start_time, end_time, score) tuples
        """
        logger.info("Analyzing video content for highlights...")
        
        # Load video
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        # Audio analysis
        audio_highlights = self._analyze_audio()
        
        # Visual analysis
        visual_highlights = self._analyze_visual_activity(cap, fps)
        
        # Combine and score highlights
        highlights = self._combine_highlights(audio_highlights, visual_highlights, duration)
        
        cap.release()
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
            
            # Convert to time segments
            highlights = []
            for peak in peaks:
                start_time = max(0, (peak * hop_length / sr) - 5)  # 5s before peak
                end_time = min(len(y)/sr, (peak * hop_length / sr) + 5)  # 5s after peak
                score = rms[peak]
                highlights.append((start_time, end_time, score))
            
            return highlights
            
        except Exception as e:
            logger.warning(f"Audio analysis failed: {e}")
            return []
    
    def _analyze_visual_activity(self, cap, fps) -> List[Tuple[float, float, float]]:
        """Analyze visual content for motion and scene changes"""
        highlights = []
        frame_count = 0
        prev_frame = None
        motion_scores = []
        
        # Sample every 60 frames for better performance on large files
        sample_rate = 60
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"Analyzing visual activity: {total_frames} frames at {fps:.1f}fps")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % sample_rate == 0:
                # Progress indicator
                if frame_count % (sample_rate * 100) == 0:
                    progress = (frame_count / total_frames) * 100
                    logger.info(f"Visual analysis progress: {progress:.1f}%")
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                if prev_frame is not None:
                    # Calculate frame difference as motion indicator
                    diff = cv2.absdiff(prev_frame, gray)
                    motion = np.mean(diff)
                    motion_scores.append((frame_count / fps, motion))
                
                prev_frame = gray
            
            frame_count += 1
        
        logger.info(f"Visual analysis complete: {len(motion_scores)} motion samples")
        
        # Find high motion periods
        if motion_scores:
            times, scores = zip(*motion_scores)
            scores = np.array(scores)
            threshold = np.percentile(scores, 80)
            
            for i, (time, score) in enumerate(motion_scores):
                if score > threshold:
                    start_time = max(0, time - 10)
                    end_time = min(times[-1], time + 10)
                    highlights.append((start_time, end_time, score))
        
        return highlights
    
    def _combine_highlights(self, audio_highlights: List, visual_highlights: List, duration: float) -> List[Tuple[float, float, float]]:
        """Combine and merge overlapping highlights"""
        all_highlights = audio_highlights + visual_highlights
        
        if not all_highlights:
            # Fallback: create evenly distributed highlights across the video
            highlight_coverage = 0.7
            usable_duration = duration * highlight_coverage
            num_highlights = int(usable_duration / self.min_duration)
            num_highlights = max(3, min(num_highlights, 15))
            
            fallback_highlights = []
            spacing = duration / (num_highlights + 1)
            
            for i in range(num_highlights):
                start_time = spacing * (i + 1) - self.min_duration / 2
                start_time = max(0, start_time)
                end_time = min(duration, start_time + self.min_duration)
                
                if end_time - start_time >= self.min_duration:
                    fallback_highlights.append((start_time, end_time, 1.0))
            
            return fallback_highlights
        
        # Sort by start time
        all_highlights.sort(key=lambda x: x[0])
        
        # Merge overlapping segments
        merged = []
        for start, end, score in all_highlights:
            if merged and start <= merged[-1][1]:
                # Overlapping - merge
                prev_start, prev_end, prev_score = merged[-1]
                merged[-1] = (prev_start, max(end, prev_end), max(score, prev_score))
            else:
                merged.append((start, end, score))
        
        # Process segments to create proper highlight clips
        final_highlights = []
        for start, end, score in merged:
            # Create multiple clips from long segments
            segment_duration = end - start
            if segment_duration > self.min_duration * 2:
                # Split long segments into multiple highlights
                num_clips = int(segment_duration / self.min_duration)
                for i in range(num_clips):
                    clip_start = start + (i * self.min_duration)
                    clip_end = min(clip_start + self.min_duration, end)
                    if clip_end - clip_start >= self.min_duration:
                        final_highlights.append((clip_start, clip_end, score))
            else:
                # Extend or adjust short segments
                if segment_duration < self.min_duration:
                    center = (start + end) / 2
                    new_start = max(0, center - self.min_duration / 2)
                    new_end = min(duration, center + self.min_duration / 2)
                    
                    # Adjust if we hit boundaries
                    if new_end - new_start < self.min_duration:
                        if new_start == 0:
                            new_end = min(duration, self.min_duration)
                        else:
                            new_start = max(0, duration - self.min_duration)
                    
                    final_highlights.append((new_start, new_end, score))
                else:
                    final_highlights.append((start, end, score))
        
        # Sort by score and determine number of highlights based on video length and highlight duration
        final_highlights.sort(key=lambda x: x[2], reverse=True)
        
        # Auto-determine number of highlights based on video duration and minimum highlight length
        # Calculate how many highlights we can reasonably extract
        highlight_coverage = 0.6  # Use 60% of video for highlights to avoid overlap
        usable_duration = duration * highlight_coverage
        max_highlights = int(usable_duration / self.min_duration)
        
        # Set reasonable bounds
        max_highlights = max(2, min(max_highlights, 15))  # Between 2-15 highlights
        
        logger.info(f"Creating {max_highlights} highlights from {duration:.1f}s video (min duration: {self.min_duration}s each)")
        return final_highlights[:max_highlights]
    
    def create_vertical_clips(self, highlights: List[Tuple[float, float, float]]) -> List[str]:
        """Create 9:16 vertical clips from highlights"""
        output_files = []
        
        for i, (start, end, score) in enumerate(highlights):
            output_file = self.output_dir / f"highlight_{i+1}.mp4"
            
            logger.info(f"Creating highlight {i+1}: {start:.1f}s - {end:.1f}s (score: {score:.2f})")
            
            # Use FFmpeg for efficient processing with better compatibility
            cmd = [
                'ffmpeg', '-y',
                '-i', self.video_path,
                '-ss', str(start),
                '-t', str(end - start),
                '-vf', 'scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920',
                '-c:v', 'libx264',
                '-c:a', 'aac',
                '-preset', 'medium',
                '-crf', '23',
                '-pix_fmt', 'yuv420p',  # Ensure compatibility
                '-movflags', '+faststart',  # Web optimization
                '-profile:v', 'baseline',  # Better compatibility
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
    parser = argparse.ArgumentParser(description='Extract 9:16 vertical highlights from videos')
    parser.add_argument('video_path', help='Path to input video file')
    parser.add_argument('--min-duration', type=int, default=90, 
                       help='Minimum duration for each highlight in seconds (default: 90)')
    parser.add_argument('--max-highlights', type=int, default=5,
                       help='Maximum number of highlights to extract (default: 5)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video_path):
        logger.error(f"Video file not found: {args.video_path}")
        sys.exit(1)
    
    try:
        extractor = VideoHighlightExtractor(args.video_path, args.min_duration)
        highlights = extractor.analyze_video_content()
        
        if not highlights:
            logger.error("No highlights found in video")
            sys.exit(1)
        
        # Limit to max highlights
        highlights = highlights[:args.max_highlights]
        
        logger.info(f"Found {len(highlights)} highlights")
        output_files = extractor.create_vertical_clips(highlights)
        
        logger.info(f"Successfully created {len(output_files)} highlight clips:")
        for file in output_files:
            print(f"  {file}")
            
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()