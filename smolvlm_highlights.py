#!/usr/bin/env python3
"""
SmolVLM-Powered Video Highlights Extractor
Uses SmolVLM vision-language model for intelligent highlight detection.
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SmolVLMHighlightExtractor:
    def __init__(self, video_path: str, min_duration: int = 30):
        self.video_path = video_path
        self.min_duration = min_duration
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
    
    def analyze_video_with_smolvlm(self) -> List[Tuple[float, float, float]]:
        """Use SmolVLM to analyze video content for highlights with parallel processing"""
        logger.info("Analyzing video with SmolVLM...")
        
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        cap.release()
        
        logger.info(f"Video: {duration:.1f}s, {fps:.1f}fps, {total_frames} frames")
        
        # Analyze video in larger segments for speed
        segment_length = 10  # 10-second segments
        segments = []
        for start_time in range(0, int(duration), segment_length):
            end_time = min(start_time + segment_length, duration)
            segments.append((start_time, end_time))
        
        logger.info(f"Processing {len(segments)} segments for analysis")
        
        # Analyze segments (can be made parallel later if needed)
        analyzed_segments = []
        for i, (start_time, end_time) in enumerate(segments):
            score = self._analyze_segment_with_vlm(start_time, end_time)
            analyzed_segments.append((start_time, end_time, score))
            
            if i % 10 == 0:
                progress = (i / len(segments)) * 100
                logger.info(f"Analysis progress: {progress:.1f}% ({i}/{len(segments)})")
        
        # Create highlights from analyzed segments
        highlights = self._create_highlights_from_segments(analyzed_segments, duration)
        return highlights
    
    def _analyze_segment_with_vlm(self, start_time: float, end_time: float) -> float:
        """Analyze a video segment using SmolVLM"""
        if self.model is None:
            # Fallback to basic analysis
            return self._basic_segment_analysis(start_time, end_time)
        
        # Extract key frames from the segment
        frames = self._extract_key_frames(start_time, end_time, num_frames=2)  # Reduced for speed
        
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
            # Extract frames at evenly spaced intervals
            frame_time = start_time + (i + 0.5) * (segment_duration / num_frames)
            frame_number = int(frame_time * fps)
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            
            if ret:
                frames.append(frame)
        
        cap.release()
        return frames
    
    def _analyze_frame_with_vlm(self, frame: np.ndarray) -> float:
        """Analyze a single frame using SmolVLM"""
        try:
            # Convert frame to PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            
            # Gaming-specific prompts for highlight detection
            prompts = [
                "Is this an action-packed gaming moment with combat, shooting, or intense gameplay? Rate the excitement level from 0-10.",
                "Does this frame show explosions, gunfire, fast movement, or other exciting gaming action? Score from 0-10.",
                "How intense and highlight-worthy is this gaming moment? Rate from 0-10."
            ]
            
            scores = []
            for prompt in prompts:
                try:
                    # Process with SmolVLM
                    inputs = self.processor(images=image, text=prompt, return_tensors="pt")
                    
                    if torch.cuda.is_available():
                        inputs = {k: v.cuda() for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = self.model.generate(**inputs, max_new_tokens=50, do_sample=False)
                    
                    # Decode response
                    response = self.processor.decode(outputs[0], skip_special_tokens=True)
                    
                    # Extract numeric score from response
                    score = self._extract_score_from_response(response)
                    scores.append(score)
                    
                except Exception as e:
                    logger.warning(f"VLM analysis failed for prompt: {e}")
                    scores.append(0.5)  # Default score
            
            return np.mean(scores) / 10.0  # Normalize to 0-1
            
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
                # Clamp to 0-10 range
                return max(0, min(10, score))
            except ValueError:
                pass
        
        # Fallback: look for keywords
        response_lower = response.lower()
        if any(word in response_lower for word in ['high', 'intense', 'exciting', 'action', 'combat']):
            return 8.0
        elif any(word in response_lower for word in ['medium', 'moderate', 'some']):
            return 5.0
        elif any(word in response_lower for word in ['low', 'calm', 'quiet', 'static']):
            return 2.0
        
        return 5.0  # Default middle score
    
    def _basic_segment_analysis(self, start_time: float, end_time: float) -> float:
        """Fallback basic analysis when VLM is not available"""
        # Motion analysis
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        motion_scores = []
        prev_frame = None
        
        for _ in range(min(30, end_frame - start_frame)):  # Sample up to 30 frames
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
        
        # Audio analysis
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
            
            # Audio features for gaming highlights
            rms = librosa.feature.rms(y=y)[0]
            energy_score = np.mean(rms)
            
            # Spectral features
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            brightness_score = np.mean(spectral_centroid) / sr
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            activity_score = np.mean(zcr)
            
            # Combine scores
            audio_score = (energy_score * 0.5 + brightness_score * 0.3 + activity_score * 0.2)
            return min(audio_score * 8, 1.0)
            
        except Exception as e:
            logger.warning(f"Audio analysis failed: {e}")
            return 0.0
    
    def _create_highlights_from_segments(self, segments: List[Tuple[float, float, float]], duration: float) -> List[Tuple[float, float, float]]:
        """Create highlights from analyzed segments"""
        # Sort by score
        segments.sort(key=lambda x: x[2], reverse=True)
        
        # Calculate number of highlights
        highlight_coverage = 0.6
        usable_duration = duration * highlight_coverage
        max_highlights = int(usable_duration / self.min_duration)
        max_highlights = max(3, min(max_highlights, 15))
        
        logger.info(f"Creating {max_highlights} highlights from {len(segments)} SmolVLM-analyzed segments")
        
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
            
            if not overlap and score > 0.3:  # Only use segments with decent scores
                # Create highlight centered on this segment
                center = (start_seg + end_seg) / 2
                highlight_start = max(0, center - self.min_duration / 2)
                highlight_end = min(duration, highlight_start + self.min_duration)
                
                # Adjust boundaries
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
        """Create 9:16 vertical clips from highlights using parallel processing"""
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import multiprocessing as mp
        
        if not highlights:
            return []
        
        max_workers = min(len(highlights), mp.cpu_count())
        logger.info(f"Creating {len(highlights)} highlights in parallel using {max_workers} workers")
        
        # Prepare clip data for parallel processing
        clip_data_list = []
        for i, (start, end, score) in enumerate(highlights):
            clip_data = (i + 1, start, end, score, self.video_path, str(self.output_dir))
            clip_data_list.append(clip_data)
            logger.info(f"  Highlight {i+1}: {start:.1f}s - {end:.1f}s (SmolVLM score: {score:.3f})")
        
        output_files = []
        
        # Execute in parallel
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_index = {executor.submit(self._create_single_clip, clip_data): clip_data[0] 
                             for clip_data in clip_data_list}
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result()
                    if result:
                        output_files.append(result)
                        logger.info(f"✓ Highlight {index} completed")
                    else:
                        logger.error(f"✗ Highlight {index} failed")
                    
                    completed += 1
                    progress = (completed / len(highlights)) * 100
                    logger.info(f"Progress: {progress:.1f}% ({completed}/{len(highlights)})")
                        
                except Exception as e:
                    logger.error(f"✗ Highlight {index} crashed: {e}")
        
        output_files.sort()
        return output_files
    
    @staticmethod
    def _create_single_clip(clip_data):
        """Create a single clip - static method for ProcessPoolExecutor"""
        import subprocess
        from pathlib import Path
        
        index, start, end, score, video_path, output_dir = clip_data
        output_file = Path(output_dir) / f"highlight_{index:02d}.mp4"
        
        cmd = [
            'ffmpeg', '-y', '-loglevel', 'error',
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
        except subprocess.CalledProcessError:
            return None

def main():
    parser = argparse.ArgumentParser(description='SmolVLM-powered video highlight extraction')
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
        extractor = SmolVLMHighlightExtractor(args.video_path, args.min_duration)
        highlights = extractor.analyze_video_with_smolvlm()
        
        if not highlights:
            logger.error("No highlights found in video")
            sys.exit(1)
        
        # Limit to max highlights if specified
        if args.max_highlights:
            highlights = highlights[:args.max_highlights]
        
        logger.info(f"Found {len(highlights)} SmolVLM-detected highlights")
        output_files = extractor.create_vertical_clips(highlights)
        
        logger.info(f"Successfully created {len(output_files)} highlight clips:")
        for file in output_files:
            print(f"  {file}")
            
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()