#!/usr/bin/env python3
"""
ML-Powered Video Highlights Extractor
Uses machine learning models for intelligent highlight detection.
"""

import os
from typing import List, Tuple
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from transformers import pipeline
import librosa
from scipy.signal import find_peaks
import logging

from .base import BaseHighlightExtractor

# Fix tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)

class MLHighlightExtractor(BaseHighlightExtractor):
    """ML-powered highlight extractor using pre-trained models"""
    
    def __init__(self, video_path: str, min_duration: int = 30, max_highlights: int = None, mode: str = "ml", content_tags: list = None):
        super().__init__(video_path, min_duration, max_highlights, mode, content_tags)
        
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
            # Image classification for frame analysis
            self.image_classifier = pipeline(
                "image-classification",
                model="microsoft/resnet-50",
                device=0 if torch.cuda.is_available() else -1
            )
        except Exception as e:
            logger.warning(f"Could not load image classifier: {e}")
            self.image_classifier = None
        
        if self.video_classifier is None and self.image_classifier is None:
            logger.warning("No ML models loaded, falling back to basic analysis")
    
    def analyze_video_content(self) -> List[Tuple[float, float, float]]:
        """Use ML models to analyze video content for highlights"""
        logger.info("Analyzing video with ML models...")
        
        # Analyze video in segments
        segment_length = 15  # 15-second segments for ML analysis
        segments = []
        for start_time in range(0, int(self.duration), segment_length):
            end_time = min(start_time + segment_length, self.duration)
            segments.append((start_time, end_time))
        
        logger.info(f"Processing {len(segments)} segments for ML analysis")
        
        # Analyze segments
        analyzed_segments = []
        for i, (start_time, end_time) in enumerate(segments):
            score = self._analyze_segment_with_ml(start_time, end_time)
            analyzed_segments.append((start_time, end_time, score))
            
            if i % 5 == 0:
                progress = (i / len(segments)) * 100
                logger.info(f"ML analysis progress: {progress:.1f}% ({i}/{len(segments)})")
        
        # Create highlights from analyzed segments
        highlights = self._create_highlights_from_segments(analyzed_segments)
        return highlights
    
    def _analyze_segment_with_ml(self, start_time: float, end_time: float) -> float:
        """Analyze a video segment using ML models"""
        if self.video_classifier is None and self.image_classifier is None:
            # Fallback to basic analysis
            return self._basic_segment_analysis(start_time, end_time)
        
        # Extract frames for analysis
        frames = self._extract_frames_for_ml(start_time, end_time)
        
        if not frames:
            return 0.0
        
        # Analyze with available models
        visual_score = 0.0
        
        if self.image_classifier:
            visual_score = self._analyze_frames_with_classifier(frames)
        
        # Combine with audio analysis
        audio_score = self._analyze_audio_segment(start_time, end_time)
        
        # Weighted combination: 60% visual, 40% audio
        final_score = visual_score * 0.6 + audio_score * 0.4
        
        return final_score
    
    def _extract_frames_for_ml(self, start_time: float, end_time: float, num_frames: int = 3) -> List[np.ndarray]:
        """Extract frames from a video segment for ML analysis"""
        frames = []
        
        cap = cv2.VideoCapture(self.video_path)
        
        # Calculate frame positions
        start_frame = int(start_time * self.fps)
        end_frame = int(end_time * self.fps)
        frame_step = max(1, (end_frame - start_frame) // num_frames)
        
        for i in range(num_frames):
            frame_pos = start_frame + i * frame_step
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        
        cap.release()
        return frames
    
    def _analyze_frames_with_classifier(self, frames: List[np.ndarray]) -> float:
        """Analyze frames using image classification model"""
        try:
            action_keywords = [
                'action', 'sport', 'game', 'competition', 'fight', 'race',
                'explosion', 'fire', 'motion', 'speed', 'intense', 'dramatic'
            ]
            
            total_score = 0.0
            
            for frame in frames:
                # Convert frame to PIL Image
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                from PIL import Image
                image = Image.fromarray(frame_rgb)
                
                # Classify the frame
                results = self.image_classifier(image)
                
                # Score based on action-related classifications
                frame_score = 0.0
                for result in results[:5]:  # Top 5 predictions
                    label = result['label'].lower()
                    confidence = result['score']
                    
                    # Check if label contains action keywords
                    for keyword in action_keywords:
                        if keyword in label:
                            frame_score += confidence * 0.5
                            break
                
                total_score += frame_score
            
            return min(total_score / len(frames), 1.0)  # Normalize to 0-1
            
        except Exception as e:
            logger.warning(f"Frame classification failed: {e}")
            return 0.0
    
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
            
            # Calculate multiple audio features
            # RMS energy
            rms = librosa.feature.rms(y=y)[0]
            avg_energy = np.mean(rms)
            
            # Spectral centroid (brightness)
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            avg_brightness = np.mean(spectral_centroids)
            
            # Zero crossing rate (indicates speech/music vs noise)
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            avg_zcr = np.mean(zcr)
            
            # Combine features for a composite score
            energy_score = min(avg_energy * 10, 1.0)
            brightness_score = min(avg_brightness / 3000, 1.0)  # Normalize
            activity_score = min(avg_zcr * 20, 1.0)
            
            # Weighted combination
            audio_score = energy_score * 0.5 + brightness_score * 0.3 + activity_score * 0.2
            
            return min(audio_score, 1.0)
            
        except Exception as e:
            logger.warning(f"Audio segment analysis failed: {e}")
            return 0.0
    
    def _basic_segment_analysis(self, start_time: float, end_time: float) -> float:
        """Fallback basic analysis when ML models are not available"""
        # Simple motion detection
        cap = cv2.VideoCapture(self.video_path)
        
        start_frame = int(start_time * self.fps)
        end_frame = int(end_time * self.fps)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        motion_scores = []
        prev_frame = None
        
        for frame_num in range(start_frame, min(end_frame, start_frame + 45)):  # Sample max 45 frames
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (160, 90))  # Small for speed
            
            if prev_frame is not None:
                diff = cv2.absdiff(prev_frame, gray)
                motion_score = np.mean(diff)
                motion_scores.append(motion_score)
            
            prev_frame = gray
        
        cap.release()
        
        if motion_scores:
            return min(np.mean(motion_scores) / 40.0, 1.0)  # Normalize
        return 0.0
    
    def _create_highlights_from_segments(self, analyzed_segments: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
        """Create highlight segments from analyzed video segments"""
        # Filter segments with score above threshold
        threshold = 0.25  # Minimum score for consideration
        good_segments = [(start, end, score) for start, end, score in analyzed_segments if score > threshold]
        
        if not good_segments:
            # If no segments meet threshold, take the best ones
            analyzed_segments.sort(key=lambda x: x[2], reverse=True)
            good_segments = analyzed_segments[:min(6, len(analyzed_segments))]
        
        logger.info(f"Selected {len(good_segments)} segments for highlight creation")
        
        return good_segments