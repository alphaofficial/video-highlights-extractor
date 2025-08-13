#!/usr/bin/env python3
"""
Parallel SmolVLM-Powered Video Highlights Extractor
Uses SmolVLM with parallel processing for faster highlight creation.
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
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import librosa
from scipy.signal import find_peaks
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp
from functools import partial

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ParallelSmolVLMExtractor:
    def __init__(self, video_path: str, min_duration: int = 30, max_workers: int = None):
        self.video_path = video_path
        self.min_duration = min_duration
        self.max_workers = max_workers or min(8, mp.cpu_count())  # Limit to 8 or CPU count
        self.output_dir = Path(video_path).parent / f"{Path(video_path).stem}_highlights"
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize SmolVLM
        self._init_smolvlm()
        
    def _init_smolvlm(self):
        """Initialize SmolVLM model"""
        logger.info("Loading SmolVLM model...")
        
        try:
            model_name = "HuggingFaceTB/SmolVLM-Instruct"
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            logger.info("SmolVLM loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load SmolVLM: {e}")
            logger.info("Falling back to basic analysis...")
            self.model = None
            self.processor = None
    
    def analyze_video_parallel(self) -> List[Tuple[float, float, float]]:
        """Analyze video using parallel processing"""
        logger.info("Analyzing video with parallel SmolVLM processing...")
        
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        cap.release()
        
        # Create segments for parallel analysis
        segment_length = 5  # 5-second segments
        segments = []
        for start_time in range(0, int(duration), segment_length):
            end_time = min(start_time + segment_length, duration)
            segments.append((start_time, end_time))
        
        logger.info(f"Processing {len(segments)} segments in parallel with {self.max_workers} workers")
        
        # Process segments in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all segment analysis tasks
            future_to_segment = {
                executor.submit(self._analyze_segment_wrapper, seg): seg 
                for seg in segments
            }
            
            analyzed_segments = []
            completed = 0
            
            for future in as_completed(future_to_segment):
                segment = future_to_segment[future]
                try:
                    start_time, end_time = segment
                    score = future.result()
                    analyzed_segments.append((start_time, end_time, score))
                    completed += 1
                    
                    if completed % 10 == 0:
                        progress = (completed / len(segments)) * 100
                        logger.info(f"Parallel analysis progress: {progress:.1f}% ({completed}/{len(segments)})")
                        
                except Exception as e:
                    logger.warning(f"Segment analysis failed for {segment}: {e}")
                    start_time, end_time = segment
                    analyzed_segments.append((start_time, end_time, 0.0))
        
        # Create highlights from analyzed segments
        highlights = self._create_highlights_from_segments(analyzed_segments, duration)
        return highlights
    
    def _analyze_segment_wrapper(self, segment: Tuple[float, float]) -> float:
        """Wrapper for segment analysis to work with ThreadPoolExecutor"""
        start_time, end_time = segment
        return self._analyze_segment_with_vlm(start_time, end_time)
    
    def _analyze_segment_with_vlm(self, start_time: float, end_time: float) -> float:
        """Analyze a video segment using SmolVLM"""
        if self.model is None:
            return self._basic_segment_analysis(start_time, end_time)
        
        # Extract key frames from the segment
        frames = self._extract_key_frames(start_time, end_time, num_frames=2)  # Reduced frames for speed
        
        if not frames:
            return 0.0
        
        # Analyze frames with SmolVLM
        vlm_scores = []
        for frame in frames:
            score = self._analyze_frame_with_vlm(frame)
            vlm_scores.append(score)
        
        # Combine with audio analysis
        audio_score = self._analyze_audio_segment(start_time, end_time)
        
        # Weight: 70% VLM, 30% audio
        final_score = np.mean(vlm_scores) * 0.7 + audio_score * 0.3
        return final_score
    
    def _extract_key_frames(self, start_time: float, end_time: float, num_frames: int = 2) -> List[np.ndarray]:
        """Extract key frames from a video segment"""
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = []
        segment_duration = end_time - start_time
        
        for i in range(num_frames):
            frame_time = start_time + (i + 0.5) * (segment_duration / num_frames)
            frame_number = int(frame_time * fps)
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            
            if ret:
                frames.append(frame)
        
        cap.release()
        return frames
    
    def _analyze_frame_with_vlm(self, frame: np.ndarray) -> float:
        """Analyze a single frame using SmolVLM - optimized version"""
        try:
            # Convert frame to PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            
            # Single optimized prompt for speed
            prompt = "Rate this gaming moment's excitement level from 0-10. Consider action, combat, explosions, and intensity."
            
            # Process with SmolVLM
            inputs = self.processor(images=image, text=prompt, return_tensors="pt")
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=20, do_sample=False)  # Reduced tokens for speed
            
            # Decode response
            response = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            # Extract numeric score from response
            score = self._extract_score_from_response(response)
            return score / 10.0  # Normalize to 0-1
            
        except Exception as e:
            logger.warning(f"Frame analysis failed: {e}")
            return 0.5  # Default score
    
    def _extract_score_from_response(self, response: str) -> float:
        """Extract numeric score from VLM response"""
        import re
        
        # Look for numbers in the response
        numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', response.lower())
        
        if numbers:
            try:
                score = float(numbers[0])
                return max(0, min(10, score))
            except ValueError:
                pass
        
        # Fallback: keyword analysis
        response_lower = response.lower()
        if any(word in response_lower for word in ['high', 'intense', 'exciting', '8', '9', '10']):
            return 8.0
        elif any(word in response_lower for word in ['medium', 'moderate', '5', '6', '7']):
            return 5.0
        elif any(word in response_lower for word in ['low', 'calm', '1', '2', '3']):
            return 2.0
        
        return 5.0
    
    def _basic_segment_analysis(self, start_time: float, end_time: float) -> float:
        """Fallback basic analysis when VLM is not available"""
        # Quick motion analysis
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        start_frame = int(start_time * fps)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        motion_scores = []
        prev_frame = None
        
        # Sample fewer frames for speed
        for _ in range(10):
            ret, frame = cap.read()
            if not ret:
                break
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if prev_frame is not None:
                diff = cv2.absdiff(prev_frame, gray)
                motion = np.mean(diff)
                motion_scores.append(motion)
            
            prev_frame = gray
        
        cap.release()
        
        motion_score = np.mean(motion_scores) / 50.0 if motion_scores else 0.0
        audio_score = self._analyze_audio_segment(start_time, end_time)
        
        return (motion_score * 0.6 + audio_score * 0.4)
    
    def _analyze_audio_segment(self, start_time: float, end_time: float) -> float:
        """Analyze audio content of a segment"""
        try:
            y, sr = librosa.load(
                self.video_path, 
                sr=22050, 
                offset=start_time, 
                duration=end_time - start_time
            )
            
            if len(y) == 0:
                return 0.0
            
            # Quick audio features
            rms = librosa.feature.rms(y=y)[0]
            energy_score = np.mean(rms)
            
            return min(energy_score * 8, 1.0)
            
        except Exception as e:
            return 0.0
    
    def _create_highlights_from_segments(self, segments: List[Tuple[float, float, float]], duration: float) -> List[Tuple[float, float, float]]:
        """Create highlights from analyzed segments"""
        segments.sort(key=lambda x: x[2], reverse=True)
        
        # Calculate number of highlights
        highlight_coverage = 0.6
        usable_duration = duration * highlight_coverage
        max_highlights = int(usable_duration / self.min_duration)
        max_highlights = max(3, min(max_highlights, 15))
        
        logger.info(f"Creating {max_highlights} highlights from {len(segments)} analyzed segments")
        
        highlights = []
        used_time_ranges = []
        
        for start_seg, end_seg, score in segments:
            if len(highlights) >= max_highlights:
                break
            
            # Check for overlap
            overlap = False
            for used_start, used_end in used_time_ranges:
                if not (end_seg <= used_start or start_seg >= used_end):
                    overlap = True
                    break
            
            if not overlap and score > 0.3:
                center = (start_seg + end_seg) / 2
                highlight_start = max(0, center - self.min_duration / 2)
                highlight_end = min(duration, highlight_start + self.min_duration)
                
                if highlight_end - highlight_start < self.min_duration:
                    if highlight_start == 0:
                        highlight_end = min(duration, self.min_duration)
                    else:
                        highlight_start = max(0, duration - self.min_duration)
                
                if highlight_end - highlight_start >= self.min_duration:
                    highlights.append((highlight_start, highlight_end, score))
                    used_time_ranges.append((highlight_start, highlight_end))
        
        return highlights
    
    def create_vertical_clips_parallel(self, highlights: List[Tuple[float, float, float]]) -> List[str]:
        """Create 9:16 vertical clips from highlights using parallel processing"""
        logger.info(f"Creating {len(highlights)} highlights in parallel with {self.max_workers} workers")
        
        # Create partial function with fixed parameters
        create_clip_func = partial(self._create_single_clip, video_path=self.video_path, output_dir=self.output_dir)
        
        output_files = []
        
        # Process clips in parallel
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all clip creation tasks
            future_to_highlight = {
                executor.submit(create_clip_func, i, highlight): (i, highlight)
                for i, highlight in enumerate(highlights)
            }
            
            completed = 0
            for future in as_completed(future_to_highlight):
                i, highlight = future_to_highlight[future]
                try:
                    output_file = future.result()
                    if output_file:
                        output_files.append(output_file)
                    completed += 1
                    
                    progress = (completed / len(highlights)) * 100
                    logger.info(f"Clip creation progress: {progress:.1f}% ({completed}/{len(highlights)})")
                    
                except Exception as e:
                    logger.error(f"Failed to create highlight {i+1}: {e}")
        
        # Sort output files by number
        output_files.sort()
        return output_files

def _create_single_clip(i: int, highlight: Tuple[float, float, float], video_path: str, output_dir: Path) -> str:
    """Create a single clip - designed to work with ProcessPoolExecutor"""
    start, end, score = highlight
    output_file = output_dir / f"highlight_{i+1}.mp4"
    
    cmd = [
        'ffmpeg', '-y',
        '-i', video_path,
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
        return str(output_file)
    except subprocess.CalledProcessError as e:
        return None

def main():
    parser = argparse.ArgumentParser(description='Parallel SmolVLM-powered video highlight extraction')
    parser.add_argument('video_path', help='Path to input video file')
    parser.add_argument('--min-duration', type=int, default=30, 
                       help='Minimum duration for each highlight in seconds (default: 30)')
    parser.add_argument('--max-highlights', type=int, default=None,
                       help='Maximum number of highlights (auto-calculated if not specified)')
    parser.add_argument('--max-workers', type=int, default=None,
                       help='Maximum number of parallel workers (default: auto)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video_path):
        logger.error(f"Video file not found: {args.video_path}")
        sys.exit(1)
    
    try:
        extractor = ParallelSmolVLMExtractor(args.video_path, args.min_duration, args.max_workers)
        highlights = extractor.analyze_video_parallel()
        
        if not highlights:
            logger.error("No highlights found in video")
            sys.exit(1)
        
        # Limit to max highlights if specified
        if args.max_highlights:
            highlights = highlights[:args.max_highlights]
        
        logger.info(f"Found {len(highlights)} SmolVLM-detected highlights")
        output_files = extractor.create_vertical_clips_parallel(highlights)
        
        logger.info(f"Successfully created {len(output_files)} highlight clips:")
        for file in output_files:
            print(f"  {file}")
            
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()