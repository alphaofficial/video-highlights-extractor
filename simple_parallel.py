#!/usr/bin/env python3
"""
Simple Parallel Video Highlights Extractor
Focuses on parallel FFmpeg processing for speed.
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path
from typing import List, Tuple
import cv2
import numpy as np
import librosa
from scipy.signal import find_peaks
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_single_highlight(args):
    """Create a single highlight clip - for parallel processing"""
    i, start, end, video_path, output_dir = args
    
    output_file = output_dir / f"highlight_{i+1:02d}.mp4"
    
    logger.info(f"Creating highlight {i+1}: {start:.1f}s - {end:.1f}s")
    
    cmd = [
        'ffmpeg', '-y', '-loglevel', 'error',
        '-i', str(video_path),
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
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return str(output_file)
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to create highlight {i+1}: {e.stderr}")
        return None

class SimpleParallelExtractor:
    def __init__(self, video_path: str, min_duration: int = 30):
        self.video_path = video_path
        self.min_duration = min_duration
        self.output_dir = Path(video_path).parent / f"{Path(video_path).stem}_highlights"
        self.output_dir.mkdir(exist_ok=True)
        
    def analyze_video_fast(self) -> List[Tuple[float, float, float]]:
        """Fast analysis using audio energy and basic motion detection"""
        logger.info("Analyzing video for highlights...")
        
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        cap.release()
        
        logger.info(f"Video: {duration:.1f}s, {fps:.1f}fps")
        
        # Audio analysis
        audio_highlights = self._analyze_audio_fast()
        
        # Visual analysis (simplified)
        visual_highlights = self._analyze_visual_fast()
        
        # Combine highlights
        all_highlights = audio_highlights + visual_highlights
        
        if not all_highlights:
            # Fallback: create evenly spaced highlights
            return self._create_fallback_highlights(duration)
        
        # Merge and score highlights
        highlights = self._merge_highlights(all_highlights, duration)
        
        return highlights
    
    def _analyze_audio_fast(self) -> List[Tuple[float, float, float]]:
        """Fast audio analysis"""
        try:
            logger.info("Analyzing audio...")
            y, sr = librosa.load(self.video_path, sr=22050)
            
            # RMS energy in 1-second windows
            hop_length = sr
            rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
            
            # Find peaks
            threshold = np.percentile(rms, 75)
            peaks, _ = find_peaks(rms, height=threshold, distance=2)
            
            highlights = []
            for peak in peaks:
                time = peak * hop_length / sr
                start_time = max(0, time - 2)
                end_time = min(len(y)/sr, time + 2)
                score = rms[peak]
                highlights.append((start_time, end_time, score))
            
            logger.info(f"Found {len(highlights)} audio highlights")
            return highlights
            
        except Exception as e:
            logger.warning(f"Audio analysis failed: {e}")
            return []
    
    def _analyze_visual_fast(self) -> List[Tuple[float, float, float]]:
        """Fast visual analysis"""
        try:
            logger.info("Analyzing visual content...")
            cap = cv2.VideoCapture(self.video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            motion_scores = []
            frame_count = 0
            prev_frame = None
            
            # Sample every 60 frames
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % 60 == 0:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    gray = cv2.resize(gray, (320, 180))  # Resize for speed
                    
                    if prev_frame is not None:
                        diff = cv2.absdiff(prev_frame, gray)
                        motion = np.mean(diff)
                        time = frame_count / fps
                        motion_scores.append((time, motion))
                    
                    prev_frame = gray
                
                frame_count += 1
            
            cap.release()
            
            if not motion_scores:
                return []
            
            # Find high motion periods
            times, scores = zip(*motion_scores)
            scores = np.array(scores)
            threshold = np.percentile(scores, 80)
            
            highlights = []
            for time, score in motion_scores:
                if score > threshold:
                    highlights.append((time - 2, time + 2, score))
            
            logger.info(f"Found {len(highlights)} visual highlights")
            return highlights
            
        except Exception as e:
            logger.warning(f"Visual analysis failed: {e}")
            return []
    
    def _create_fallback_highlights(self, duration: float) -> List[Tuple[float, float, float]]:
        """Create evenly spaced highlights as fallback"""
        logger.info("Creating fallback highlights")
        
        num_highlights = max(3, min(int(duration * 0.6 / self.min_duration), 10))
        highlights = []
        
        spacing = duration / (num_highlights + 1)
        for i in range(num_highlights):
            start_time = spacing * (i + 1) - self.min_duration / 2
            start_time = max(0, start_time)
            end_time = min(duration, start_time + self.min_duration)
            
            if end_time - start_time >= self.min_duration:
                highlights.append((start_time, end_time, 1.0))
        
        return highlights
    
    def _merge_highlights(self, highlights: List[Tuple[float, float, float]], duration: float) -> List[Tuple[float, float, float]]:
        """Merge overlapping highlights and create final list"""
        if not highlights:
            return []
        
        # Sort by start time
        highlights.sort(key=lambda x: x[0])
        
        # Merge overlapping
        merged = []
        for start, end, score in highlights:
            if merged and start <= merged[-1][1]:
                # Overlapping - merge
                prev_start, prev_end, prev_score = merged[-1]
                merged[-1] = (prev_start, max(end, prev_end), max(score, prev_score))
            else:
                merged.append((start, end, score))
        
        # Create final highlights with proper duration
        final_highlights = []
        for start, end, score in merged:
            # Ensure minimum duration
            center = (start + end) / 2
            new_start = max(0, center - self.min_duration / 2)
            new_end = min(duration, new_start + self.min_duration)
            
            # Adjust if needed
            if new_end - new_start < self.min_duration:
                if new_start == 0:
                    new_end = min(duration, self.min_duration)
                else:
                    new_start = max(0, duration - self.min_duration)
            
            if new_end - new_start >= self.min_duration:
                final_highlights.append((new_start, new_end, score))
        
        # Remove overlaps and sort by score
        final_highlights = self._remove_overlaps(final_highlights)
        final_highlights.sort(key=lambda x: x[2], reverse=True)
        
        # Limit number of highlights
        max_highlights = max(3, min(len(final_highlights), 10))
        return final_highlights[:max_highlights]
    
    def _remove_overlaps(self, highlights: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
        """Remove overlapping highlights, keeping higher scored ones"""
        if not highlights:
            return []
        
        # Sort by score (descending)
        highlights.sort(key=lambda x: x[2], reverse=True)
        
        final = []
        for start, end, score in highlights:
            # Check if this overlaps with any already selected
            overlap = False
            for existing_start, existing_end, _ in final:
                if not (end <= existing_start or start >= existing_end):
                    overlap = True
                    break
            
            if not overlap:
                final.append((start, end, score))
        
        return final
    
    def create_highlights_parallel(self, highlights: List[Tuple[float, float, float]], max_workers: int = None) -> List[str]:
        """Create highlight clips in parallel"""
        if not highlights:
            return []
        
        max_workers = max_workers or min(len(highlights), mp.cpu_count())
        logger.info(f"Creating {len(highlights)} highlights in parallel with {max_workers} workers")
        
        # Prepare arguments for parallel processing
        args_list = []
        for i, (start, end, score) in enumerate(highlights):
            args_list.append((i, start, end, self.video_path, self.output_dir))
        
        output_files = []
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_args = {executor.submit(create_single_highlight, args): args for args in args_list}
            
            completed = 0
            for future in as_completed(future_to_args):
                args = future_to_args[future]
                try:
                    result = future.result()
                    if result:
                        output_files.append(result)
                    completed += 1
                    
                    progress = (completed / len(highlights)) * 100
                    logger.info(f"Progress: {progress:.1f}% ({completed}/{len(highlights)})")
                    
                except Exception as e:
                    logger.error(f"Task failed for {args}: {e}")
        
        output_files.sort()
        return output_files

def main():
    parser = argparse.ArgumentParser(description='Simple parallel video highlight extraction')
    parser.add_argument('video_path', help='Path to input video file')
    parser.add_argument('--min-duration', type=int, default=30, 
                       help='Duration for each highlight in seconds (default: 30)')
    parser.add_argument('--max-workers', type=int, default=None,
                       help='Maximum number of parallel workers (default: auto)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video_path):
        logger.error(f"Video file not found: {args.video_path}")
        sys.exit(1)
    
    try:
        extractor = SimpleParallelExtractor(args.video_path, args.min_duration)
        highlights = extractor.analyze_video_fast()
        
        if not highlights:
            logger.error("No highlights found")
            sys.exit(1)
        
        logger.info(f"Found {len(highlights)} highlights")
        for i, (start, end, score) in enumerate(highlights):
            logger.info(f"  Highlight {i+1}: {start:.1f}s - {end:.1f}s (score: {score:.3f})")
        
        output_files = extractor.create_highlights_parallel(highlights, args.max_workers)
        
        logger.info(f"Successfully created {len(output_files)} highlight clips:")
        for file in output_files:
            print(f"  {file}")
            
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()