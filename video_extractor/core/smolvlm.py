#!/usr/bin/env python3
"""
Gaming-Aware SmolVLM Video Highlights Extractor
Uses SmolVLM for intelligent gaming context understanding
"""

import os
from typing import List, Tuple, Optional, Dict, Any
import cv2
import numpy as np
import logging
from PIL import Image
import random

# ==========================
# Constants (configuration and prompts)
# ==========================
# Additional scalar constants
EDGE_DENSITY_MULTIPLIER: float = 2.0
PROGRESS_LOG_EVERY: int = 2
MOTION_SAMPLE_MAX_SPAN_FRAMES: int = 300
DEFAULT_SVLM_PARSE_SCORE: float = 0.7
AUDIO_EXCEPTION_DEFAULT_SCORE: float = 0.3

# Model configuration
MODEL_CONFIG: Dict[str, Any] = {
    "SMOLVLM_MODEL": "HuggingFaceTB/SmolVLM-Instruct",
}

# Analysis configuration and thresholds
ANALYSIS_CONFIG: Dict[str, Any] = {
    # Segmenting
    "SEGMENT_LENGTH": 30,                 # seconds
    "KEY_FRAMES_PER_SEGMENT": 3,
    "SAMPLE_CONTENT_TIMES": [0.2, 0.5, 0.8],  # relative positions for identification frames

    # Motion analysis
    "MOTION_SAMPLE_STEP": 30,             # frames between samples
    "MOTION_RESIZE": (320, 180),          # grayscale resize for speed
    "MOTION_NORM_DIVISOR": 25.0,          # normalization for motion score
    "EDGE_CANNY_LO": 50,
    "EDGE_CANNY_HI": 150,
    "EDGE_DENSITY_WEIGHT": 0.3,           # weight for edge density in enhanced motion
    "MOTION_WEIGHT": 0.7,                 # weight for motion in enhanced motion

    # Fallback motion scoring
    "FALLBACK_DEFAULT_SCORE": 0.4,
    "FALLBACK_RANDOM_RANGE": (0.05, 0.15), # randomness added in fallback

    # Candidate selection
    "CANDIDATE_MAX": 10,                  # limit number of SmolVLM calls
    "CANDIDATE_MOTION_MIN": 0.4,          # min motion to be a candidate

    # Combined scoring (motion + SmolVLM)
    "SMOLVLM_WEIGHT": 0.7,
    "MOTION_WEIGHT_FOR_SVLM_COMBO": 0.3,
    "SMOLVLM_FINAL_THRESHOLD": 0.4,

    # Enhanced (no-SmolVLM) scoring combination
    "ENHANCED_COMBINED_MOTION_WEIGHT": 0.6,
    "ENHANCED_COMBINED_AUDIO_WEIGHT": 0.4,
    "ENHANCED_KEEP_THRESHOLD": 0.5,

    # Audio analysis
    "AUDIO_SR": 22050,
    "AUDIO_ENERGY_SCALE": 15.0,
    "AUDIO_ZCR_SCALE": 25.0,
    "AUDIO_ROLLOFF_DIV": 4000.0,
    "AUDIO_ENERGY_WEIGHT": 0.5,
    "AUDIO_SPEECH_WEIGHT": 0.3,
    "AUDIO_EXCITEMENT_WEIGHT": 0.2,

    # Generation parameters
    "GEN_MAX_NEW_TOKENS": 50,
}

# Prompt formatting
PROMPT_FORMAT: Dict[str, str] = {
    "IMAGE_PREFIX": "<image>\n",
}

# Content detection prompts (identify content type)
CONTENT_DETECTION_PROMPTS: Dict[str, str] = {
    "generic": "What type of video content is this? One word answer.",
    "gaming": "What gaming genre is this? FPS, MOBA, Racing, Fighting, or Other?",
    "cooking": "What type of cooking content is this? Recipe, Baking, Grilling, or Other?",
    "sports": "What sport is this? Football, Basketball, Soccer, Tennis, or Other?",
    "music": "What type of music content is this? Concert, Music Video, DJ Set, or Other?",
    "tutorial": "What type of tutorial is this? Tech, DIY, Educational, or Other?",
}
CONTENT_DETECTION_TEMPLATE_OTHER: str = "What type of {tag} content is this? One word answer."

# Content-scoring prompts per tag
CONTENT_PROMPTS: Dict[str, List[str]] = {
    "generic": [
        "Rate this moment's excitement level from 1-10. Consider action, drama, and interest. Respond with just the number.",
        "Is this a highlight-worthy moment that viewers would want to see? Rate from 1-10.",
        "Does this show interesting or engaging content? Rate excitement 1-10.",
    ],
    "gaming": [
        "Rate this gaming moment's excitement level from 1-10. Consider action, skill demonstration, and dramatic tension. Respond with just the number.",
        "Is this a highlight-worthy gaming moment that players would want to clip and share? Rate from 1-10.",
        "Does this show eliminations, clutch plays, or intense action? Rate excitement 1-10.",
    ],
    "cooking": [
        "Rate this cooking moment's interest level from 1-10. Consider technique demonstration, final reveals, and key steps. Respond with just the number.",
        "Is this a highlight-worthy cooking moment that viewers would want to see? Rate from 1-10.",
        "Does this show important cooking techniques, ingredient reveals, or finished dishes? Rate interest 1-10.",
    ],
    "sports": [
        "Rate this sports moment's excitement level from 1-10. Consider goals, amazing plays, and dramatic moments. Respond with just the number.",
        "Is this a highlight-worthy sports moment that fans would want to replay? Rate from 1-10.",
        "Does this show scoring, incredible athleticism, or game-changing moments? Rate excitement 1-10.",
    ],
    "music": [
        "Rate this music moment's energy level from 1-10. Consider performance peaks, crowd reactions, and musical highlights. Respond with just the number.",
        "Is this a highlight-worthy music moment that fans would want to share? Rate from 1-10.",
        "Does this show powerful vocals, instrumental solos, or crowd engagement? Rate energy 1-10.",
    ],
    "tutorial": [
        "Rate this tutorial moment's importance from 1-10. Consider key explanations, demonstrations, and results. Respond with just the number.",
        "Is this a highlight-worthy tutorial moment that learners would want to reference? Rate from 1-10.",
        "Does this show important steps, before/after comparisons, or key insights? Rate importance 1-10.",
    ],
}

# Generic prompt templates for other tags
CONTENT_PROMPT_TEMPLATES: List[str] = [
    "Rate this {tag} moment's interest level from 1-10. Consider engagement, importance, and shareability. Respond with just the number.",
    "Is this a highlight-worthy {tag} moment that viewers would want to see? Rate from 1-10.",
    "Does this show important or exciting {tag} content? Rate interest 1-10.",
]

# Simple rating prompts for segment scoring
RATING_PROMPTS: Dict[str, str] = {
    "generic": "Rate this moment 1-10:",
    "templated": "Rate this {tag} moment 1-10:",
}

from .base import BaseHighlightExtractor

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logger = logging.getLogger(__name__)

class SmolVLMHighlightExtractor(BaseHighlightExtractor):
    """Gaming-aware SmolVLM highlight extractor"""
    
    def __init__(self, video_path: str, min_duration: int = 30, max_highlights: int = None, mode: str = "smolvlm", content_tags: list = None):
        super().__init__(video_path, min_duration, max_highlights, mode, content_tags)
        
        # Initialize SmolVLM model
        self.model = None
        self.processor = None
        self.content_type = None
        self._init_smolvlm()
        
    def _init_smolvlm(self):
        """Initialize SmolVLM model for gaming analysis"""
        try:
            if self.content_tags:
                logger.info(f"Loading SmolVLM model for {', '.join(self.content_tags)} content analysis...")
            else:
                logger.info("Loading SmolVLM model for generic content analysis...")
            from transformers import AutoProcessor, AutoModelForVision2Seq
            import torch
            
            model_name = MODEL_CONFIG["SMOLVLM_MODEL"]
            
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            logger.info("SmolVLM model loaded successfully")
            
        except Exception as e:
            logger.warning(f"Could not load SmolVLM model: {e}")
            logger.info("Falling back to enhanced motion detection")
            self.model = None
            self.processor = None
    
    def analyze_video_content(self) -> List[Tuple[float, float, float]]:
        """Use SmolVLM for intelligent content highlight detection"""
        if self.content_tags:
            logger.info(f"Analyzing video with content-aware SmolVLM for {', '.join(self.content_tags)} content...")
        else:
            logger.info("Analyzing video with generic SmolVLM highlight detection...")
        
        if not self.model:
            logger.info("SmolVLM not available, using enhanced motion detection")
            return self._enhanced_motion_analysis()
        
        # First, identify the content type with a quick sample
        self.content_type = self._identify_content_type()
        logger.info(f"Detected content type: {self.content_type}")
        
        # Use SmolVLM selectively - only on promising segments
        # First pass: quick motion analysis to find candidate segments
        candidate_segments = self._find_candidate_segments()
        
        logger.info(f"Found {len(candidate_segments)} candidate segments, analyzing with SmolVLM...")
        
        # Second pass: SmolVLM analysis on candidates only
        highlights = []
        for i, (start, end, motion_score) in enumerate(candidate_segments):
            # Use SmolVLM to refine the score
            smolvlm_score = self._analyze_segment_with_smolvlm(start, end)
            
            # Combine motion and SmolVLM scores
            final_score = (
                motion_score * ANALYSIS_CONFIG["MOTION_WEIGHT_FOR_SVLM_COMBO"]
                + smolvlm_score * ANALYSIS_CONFIG["SMOLVLM_WEIGHT"]
            )
            
            logger.info(f"Segment {start}-{end}s: motion={motion_score:.3f}, smolvlm={smolvlm_score:.3f}, final={final_score:.3f}")
            
            if final_score > ANALYSIS_CONFIG["SMOLVLM_FINAL_THRESHOLD"]:
                highlights.append((start, end, final_score))
                logger.info(f"Content highlight confirmed: {start}-{end}s (score: {final_score:.3f})")
            
            if i % PROGRESS_LOG_EVERY == 0:
                progress = (i / len(candidate_segments)) * 100
                logger.info(f"SmolVLM analysis progress: {progress:.1f}%")
        
        logger.info(f"Found {len(highlights)} content highlights")
        return highlights
    
    def _identify_content_type(self) -> str:
        """Identify the content type using SmolVLM based on provided tags"""
        if not self.model:
            return "unknown"
        
        try:
            # Extract a few sample frames from different parts of the video
            sample_times = [self.duration * t for t in ANALYSIS_CONFIG["SAMPLE_CONTENT_TIMES"]]
            frames = []
            
            for time_pos in sample_times:
                frame = self._extract_frame_at_time(time_pos)
                if frame is not None:
                    frames.append(frame)
            
            if not frames:
                return "unknown"
            
            # Create content type detection prompt based on tags
            if not self.content_tags:
                prompt = CONTENT_DETECTION_PROMPTS["generic"]
            else:
                for tag in ["gaming", "cooking", "sports", "music", "tutorial"]:
                    if tag in self.content_tags:
                        prompt = CONTENT_DETECTION_PROMPTS[tag]
                        break
                else:
                    primary_tag = self.content_tags[0]
                    prompt = CONTENT_DETECTION_TEMPLATE_OTHER.format(tag=primary_tag)

            # Analyze the first frame to determine content type
            content_type = self._query_smolvlm_single_frame(frames[0], prompt)
            
            # Clean up the response
            content_type = content_type.strip().lower().replace(" ", "_")
            
            return content_type if content_type else "unknown"
            
        except Exception as e:
            logger.warning(f"Content type detection failed: {e}")
            return "unknown"
    

    def _extract_key_frames_from_segment(self, start_time: float, end_time: float, num_frames: int = 3) -> List[np.ndarray]:
        """Extract key frames from a video segment"""
        frames = []
        
        # Extract frames at different points in the segment
        time_points = np.linspace(start_time, end_time, num_frames)
        
        for time_point in time_points:
            frame = self._extract_frame_at_time(time_point)
            if frame is not None:
                frames.append(frame)
        
        return frames
    
    def _extract_frame_at_time(self, time_seconds: float) -> Optional[np.ndarray]:
        """Extract a single frame at a specific time using base class capture"""
        frame_number = int(time_seconds * self.fps)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        return frame if ret else None
    
    def _analyze_frames_for_content(self, frames: List[np.ndarray]) -> float:
        """Analyze frames for highlight content using SmolVLM"""
        if not frames:
            return 0.0
        
        # Create content-specific prompts based on tags
        prompts = self._get_content_specific_prompts()
        
        total_score = 0.0
        
        for frame in frames:
            frame_score = 0.0
            
            # Analyze frame with multiple content-specific questions
            for prompt in prompts:
                response = self._query_smolvlm_single_frame(frame, prompt)
                score = self._parse_content_response(response)
                frame_score += score
            
            # Average the scores from different prompts
            frame_score = frame_score / len(prompts) if prompts else 0.0
            total_score += frame_score
        
        # Average across all frames
        return total_score / len(frames) if frames else 0.0
    
    def _get_content_specific_prompts(self) -> List[str]:
        """Get content analysis prompts based on content tags"""
        if not self.content_tags:
            return CONTENT_PROMPTS["generic"]
        for tag in ["gaming", "cooking", "sports", "music", "tutorial"]:
            if tag in self.content_tags:
                return CONTENT_PROMPTS[tag]
        primary_tag = self.content_tags[0]
        return [t.format(tag=primary_tag) for t in CONTENT_PROMPT_TEMPLATES]
    
    def _query_smolvlm_single_frame(self, frame: np.ndarray, prompt: str) -> str:
        """Query SmolVLM with a single frame and prompt"""
        try:
            import torch
            
            # Convert frame to PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # Format the prompt properly for SmolVLM
            formatted_prompt = f"{PROMPT_FORMAT['IMAGE_PREFIX']}{prompt}"
            
            # Prepare inputs with proper format
            inputs = self.processor(
                text=formatted_prompt,
                images=pil_image,
                return_tensors="pt"
            )
            
            # Move to appropriate device
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=ANALYSIS_CONFIG["GEN_MAX_NEW_TOKENS"],
                    do_sample=False,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # Decode response
            generated_text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
            
            # Extract only the response part (after the prompt)
            # SmolVLM often repeats the prompt, so we need to extract just the answer
            response = generated_text
            
            # Remove the formatted prompt if it exists
            if formatted_prompt in response:
                response = response.split(formatted_prompt)[-1].strip()
            
            # Clean up common artifacts
            response = response.replace("<image>", "").strip()
            
            # Extract just the first meaningful word/phrase (before any repetition)
            lines = response.split('\n')
            if lines:
                # Take the first non-empty line
                for line in lines:
                    clean_line = line.strip()
                    if clean_line and not clean_line.startswith('answer:'):
                        # Remove "answer:" prefix if present
                        if clean_line.lower().startswith('answer:'):
                            clean_line = clean_line[7:].strip()
                        # Take just the first word/phrase before any repetition
                        first_part = clean_line.split('.')[0].split(' ')[0].strip()
                        if first_part:
                            return first_part
            
            return "5"
            
        except Exception as e:
            logger.warning(f"SmolVLM query failed: {e}")
            return "0"
    
    def _parse_content_response(self, response: str) -> float:
        """Parse SmolVLM response to extract numeric score"""
        try:
            # Clean the response - remove the prompt part and common phrases
            clean_response = response.replace("<image>", "").strip()
            # Remove common prompt phrases
            for phrase in ["Rate 1-10", "Respond with just the number", "Just give the number"]:
                clean_response = clean_response.replace(phrase, "").strip()
            
            # Look for decimal numbers first, then integers
            import re
            decimal_numbers = re.findall(r'\d+\.\d+', clean_response)
            integer_numbers = re.findall(r'\b\d+\b', clean_response)
            
            # Try decimal numbers first
            if decimal_numbers:
                score = float(decimal_numbers[0])
                return min(score / 10.0, 1.0)
            elif integer_numbers:
                score = int(integer_numbers[0])
                return min(score / 10.0, 1.0)
            
            # Look for YES/NO responses
            response_lower = response.lower()
            if "yes" in response_lower and "no" not in response_lower:
                return 0.8
            elif "no" in response_lower:
                return 0.2
            
            # Default higher score if unclear - be more lenient
            return DEFAULT_SVLM_PARSE_SCORE
            
        except Exception as e:
            logger.warning(f"Failed to parse response '{response}': {e}")
            return DEFAULT_SVLM_PARSE_SCORE
    
    def _analyze_audio_for_gaming_reactions(self, start_time: float, end_time: float) -> float:
        """Analyze audio for gaming reactions and excitement"""
        try:
            import librosa
            
            # Load audio segment
            y, sr = librosa.load(
                self.video_path,
                sr=ANALYSIS_CONFIG["AUDIO_SR"],
                offset=start_time,
                duration=end_time - start_time
            )
            
            if len(y) == 0:
                return 0.0
            
            # Calculate audio features that indicate gaming excitement
            # RMS energy (volume/intensity)
            rms = librosa.feature.rms(y=y)[0]
            avg_energy = np.mean(rms)
            
            # Zero crossing rate (speech activity)
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            avg_zcr = np.mean(zcr)
            
            # Spectral rolloff (voice excitement/pitch changes)
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            avg_rolloff = np.mean(rolloff)
            
            # Combine features for gaming reaction score
            energy_score = min(avg_energy * ANALYSIS_CONFIG["AUDIO_ENERGY_SCALE"], 1.0)
            speech_score = min(avg_zcr * ANALYSIS_CONFIG["AUDIO_ZCR_SCALE"], 1.0)
            excitement_score = min(avg_rolloff / ANALYSIS_CONFIG["AUDIO_ROLLOFF_DIV"], 1.0)
            
            # Weighted combination
            audio_score = (
                energy_score * ANALYSIS_CONFIG["AUDIO_ENERGY_WEIGHT"]
                + speech_score * ANALYSIS_CONFIG["AUDIO_SPEECH_WEIGHT"]
                + excitement_score * ANALYSIS_CONFIG["AUDIO_EXCITEMENT_WEIGHT"]
            )
            
            return min(audio_score, 1.0)
            
        except Exception as e:
            logger.warning(f"Audio analysis failed: {e}")
            return AUDIO_EXCEPTION_DEFAULT_SCORE
    
    def _fallback_motion_analysis(self, start_time: float, end_time: float) -> float:
        """Fallback motion analysis when SmolVLM is not available"""
        start_frame = int(start_time * self.fps)
        end_frame = int(end_time * self.fps)
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        motion_scores = []
        prev_frame = None
        
        # Sample every N frames for better motion detection
        for frame_num in range(start_frame, min(end_frame, start_frame + MOTION_SAMPLE_MAX_SPAN_FRAMES), ANALYSIS_CONFIG["MOTION_SAMPLE_STEP"]):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = self.cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, ANALYSIS_CONFIG["MOTION_RESIZE"])
            
            if prev_frame is not None:
                diff = cv2.absdiff(prev_frame, gray)
                motion_score = np.mean(diff)
                motion_scores.append(motion_score)
            
            prev_frame = gray
        
        if motion_scores:
            avg_motion = np.mean(motion_scores)
            # Enhanced normalization for gaming content
            normalized_score = min(avg_motion / ANALYSIS_CONFIG["MOTION_NORM_DIVISOR"], 1.0)
            
            # Add slight randomness for variety when using fallback
            low, high = ANALYSIS_CONFIG["FALLBACK_RANDOM_RANGE"]
            normalized_score += random.uniform(low, high)
            
            return min(normalized_score, 1.0)
        
        return ANALYSIS_CONFIG["FALLBACK_DEFAULT_SCORE"]  # Default score
    
    def _enhanced_motion_analysis(self) -> List[Tuple[float, float, float]]:
        """Enhanced motion analysis for gaming content when SmolVLM is not available"""
        logger.info("Using enhanced motion analysis for gaming highlights...")
        
        segments = []
        segment_length = ANALYSIS_CONFIG["SEGMENT_LENGTH"]  # seconds
        
        for start in range(0, int(self.duration), segment_length):
            end = min(start + segment_length, self.duration)
            
            # Enhanced motion scoring for gaming
            motion_score = self._calculate_enhanced_motion_score(start, end)
            audio_score = self._analyze_audio_for_gaming_reactions(start, end)
            
            # Combine scores with gaming-focused weights
            combined_score = (
                motion_score * ANALYSIS_CONFIG["ENHANCED_COMBINED_MOTION_WEIGHT"]
                + audio_score * ANALYSIS_CONFIG["ENHANCED_COMBINED_AUDIO_WEIGHT"]
            )
            
            if combined_score > ANALYSIS_CONFIG["ENHANCED_KEEP_THRESHOLD"]:  # Keep segments with good combined score
                segments.append((start, end, combined_score))
                logger.info(f"Enhanced highlight: {start}-{end}s (motion: {motion_score:.3f}, audio: {audio_score:.3f}, combined: {combined_score:.3f})")
        
        logger.info(f"Found {len(segments)} enhanced motion highlights")
        return segments
    
    def _find_candidate_segments(self) -> List[Tuple[float, float, float]]:
        """Quick motion analysis to find candidate segments for SmolVLM analysis"""
        candidates = []
        segment_length = ANALYSIS_CONFIG["SEGMENT_LENGTH"]
        
        for start in range(0, int(self.duration), segment_length):
            end = min(start + segment_length, self.duration)
            
            # Quick motion analysis
            motion_score = self._calculate_enhanced_motion_score(start, end)
            
            # Only consider segments with decent motion for SmolVLM analysis
            if motion_score > ANALYSIS_CONFIG["CANDIDATE_MOTION_MIN"]:
                candidates.append((start, end, motion_score))
        
        # Sort by motion score and take top candidates to limit SmolVLM calls
        candidates.sort(key=lambda x: x[2], reverse=True)
        max_candidates = min(10, len(candidates))  # Limit to 10 SmolVLM calls
        
        return candidates[:max_candidates]
    
    def _calculate_enhanced_motion_score(self, start_time: float, end_time: float) -> float:
        """Enhanced motion calculation optimized for gaming content"""
        start_frame = int(start_time * self.fps)
        end_frame = int(end_time * self.fps)
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        motion_scores = []
        edge_scores = []
        prev_frame = None
        
        # Sample every N frames for good balance of speed and accuracy
        for frame_num in range(start_frame, min(end_frame, start_frame + MOTION_SAMPLE_MAX_SPAN_FRAMES), ANALYSIS_CONFIG["MOTION_SAMPLE_STEP"]):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = self.cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, ANALYSIS_CONFIG["MOTION_RESIZE"])
            
            # Calculate edge density (indicates UI elements, text, detailed graphics)
            edges = cv2.Canny(gray, ANALYSIS_CONFIG["EDGE_CANNY_LO"], ANALYSIS_CONFIG["EDGE_CANNY_HI"])
            edge_density = np.mean(edges) / 255.0
            edge_scores.append(edge_density)
            
            if prev_frame is not None:
                # Motion detection
                diff = cv2.absdiff(prev_frame, gray)
                motion_score = np.mean(diff)
                motion_scores.append(motion_score)
            
            prev_frame = gray
        
        if motion_scores and edge_scores:
            avg_motion = np.mean(motion_scores)
            avg_edges = np.mean(edge_scores)
            
            # Gaming content often has high edge density (UI, text, detailed graphics)
            # Combine motion and edge information
            motion_component = min(avg_motion / ANALYSIS_CONFIG["MOTION_NORM_DIVISOR"], 1.0)
            edge_component = min(avg_edges * 2.0, 1.0)  # UI elements boost score
            
            # Weighted combination favoring motion but considering UI complexity
            enhanced_score = (
                motion_component * ANALYSIS_CONFIG["MOTION_WEIGHT"]
                + edge_component * ANALYSIS_CONFIG["EDGE_DENSITY_WEIGHT"]
            )
            
            return min(enhanced_score, 1.0)
        
        return 0.3
    
    def _analyze_segment_with_smolvlm(self, start_time: float, end_time: float) -> float:
        """Analyze a single segment with SmolVLM (simplified for efficiency)"""
        try:
            # Extract just one key frame from the middle of the segment
            mid_time = start_time + (end_time - start_time) / 2
            frame = self._extract_frame_at_time(mid_time)
            
            if frame is None:
                return 0.0
            
            # Use a simpler, more lenient prompt
            if self.content_tags:
                primary_tag = self.content_tags[0]
                prompt = RATING_PROMPTS["templated"].format(tag=primary_tag)
            else:
                prompt = RATING_PROMPTS["generic"]
            
            response = self._query_smolvlm_single_frame(frame, prompt)
            score = self._parse_content_response(response)
            
            logger.info(f"SmolVLM response: '{response}' -> score: {score}")
            
            return score
            
        except Exception as e:
            logger.warning(f"SmolVLM segment analysis failed: {e}")
            return 0.0