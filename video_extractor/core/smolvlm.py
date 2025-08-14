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
import base64
import io

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
            logger.info(f"Loading SmolVLM model for {', '.join(self.content_tags)} content analysis...")
            from transformers import AutoProcessor, AutoModelForImageTextToText
            import torch
            
            model_name = "HuggingFaceTB/SmolVLM-Instruct"
            
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModelForImageTextToText.from_pretrained(
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
        logger.info(f"Analyzing video with content-aware SmolVLM for {', '.join(self.content_tags)} content...")
        
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
            final_score = motion_score * 0.3 + smolvlm_score * 0.7
            
            logger.info(f"Segment {start}-{end}s: motion={motion_score:.3f}, smolvlm={smolvlm_score:.3f}, final={final_score:.3f}")
            
            if final_score > 0.4:  # Lower threshold for combined score
                highlights.append((start, end, final_score))
                logger.info(f"Content highlight confirmed: {start}-{end}s (score: {final_score:.3f})")
            
            if i % 2 == 0:
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
            sample_times = [self.duration * 0.2, self.duration * 0.5, self.duration * 0.8]
            frames = []
            
            for time_pos in sample_times:
                frame = self._extract_frame_at_time(time_pos)
                if frame is not None:
                    frames.append(frame)
            
            if not frames:
                return "unknown"
            
            # Create content type detection prompt based on tags
            primary_tag = self.content_tags[0] if self.content_tags else "general"
            
            if "gaming" in self.content_tags:
                prompt = """Look at this video content and identify the gaming genre:
Options: FPS, Battle Royale, MOBA, Racing, Fighting, Sports, Strategy, RPG, Other
Just respond with the genre name."""
            elif "cooking" in self.content_tags:
                prompt = """Look at this cooking video and identify the type:
Options: Recipe Tutorial, Baking, Grilling, Fine Dining, Fast Cooking, Meal Prep, Other
Just respond with the type."""
            elif "sports" in self.content_tags:
                prompt = """Look at this sports content and identify the sport:
Options: Football, Basketball, Soccer, Tennis, Baseball, Hockey, Combat Sports, Other
Just respond with the sport name."""
            elif "music" in self.content_tags:
                prompt = """Look at this music content and identify the type:
Options: Live Performance, Music Video, Studio Recording, DJ Set, Concert, Other
Just respond with the type."""
            elif "tutorial" in self.content_tags:
                prompt = """Look at this tutorial content and identify the type:
Options: Tech Tutorial, DIY, Educational, How-to, Software Demo, Other
Just respond with the type."""
            else:
                prompt = f"""Look at this {primary_tag} video content and describe what type of content this is.
Just give a brief 1-2 word description."""

            # Analyze the first frame to determine content type
            content_type = self._query_smolvlm_single_frame(frames[0], prompt)
            
            # Clean up the response
            content_type = content_type.strip().lower().replace(" ", "_")
            
            return content_type if content_type else "unknown"
            
        except Exception as e:
            logger.warning(f"Content type detection failed: {e}")
            return "unknown"
    
    def _analyze_gaming_segment(self, start_time: float, end_time: float) -> float:
        """Analyze a video segment for gaming highlights using SmolVLM"""
        if not self.model:
            return self._fallback_motion_analysis(start_time, end_time)
        
        try:
            # Extract key frames from the segment
            key_frames = self._extract_key_frames_from_segment(start_time, end_time)
            
            if not key_frames:
                return 0.0
            
            # Analyze frames with gaming-specific prompts
            gaming_score = self._analyze_frames_for_gaming_content(key_frames)
            
            # Combine with audio analysis for emotional context
            audio_score = self._analyze_audio_for_gaming_reactions(start_time, end_time)
            
            # Weighted combination: 70% visual gaming content, 30% audio reactions
            final_score = gaming_score * 0.7 + audio_score * 0.3
            
            return min(final_score, 1.0)
            
        except Exception as e:
            logger.warning(f"SmolVLM analysis failed for segment {start_time}-{end_time}: {e}")
            return self._fallback_motion_analysis(start_time, end_time)
    
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
        """Extract a single frame at a specific time"""
        cap = cv2.VideoCapture(self.video_path)
        
        frame_number = int(time_seconds * self.fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        
        ret, frame = cap.read()
        cap.release()
        
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
        primary_tag = self.content_tags[0] if self.content_tags else "general"
        
        if "gaming" in self.content_tags:
            return [
                "Rate this gaming moment's excitement level from 1-10. Consider action, skill demonstration, and dramatic tension. Respond with just the number.",
                "Is this a highlight-worthy gaming moment that players would want to clip and share? Rate from 1-10.",
                "Does this show eliminations, clutch plays, or intense action? Rate excitement 1-10.",
            ]
        elif "cooking" in self.content_tags:
            return [
                "Rate this cooking moment's interest level from 1-10. Consider technique demonstration, final reveals, and key steps. Respond with just the number.",
                "Is this a highlight-worthy cooking moment that viewers would want to see? Rate from 1-10.",
                "Does this show important cooking techniques, ingredient reveals, or finished dishes? Rate interest 1-10.",
            ]
        elif "sports" in self.content_tags:
            return [
                "Rate this sports moment's excitement level from 1-10. Consider goals, amazing plays, and dramatic moments. Respond with just the number.",
                "Is this a highlight-worthy sports moment that fans would want to replay? Rate from 1-10.",
                "Does this show scoring, incredible athleticism, or game-changing moments? Rate excitement 1-10.",
            ]
        elif "music" in self.content_tags:
            return [
                "Rate this music moment's energy level from 1-10. Consider performance peaks, crowd reactions, and musical highlights. Respond with just the number.",
                "Is this a highlight-worthy music moment that fans would want to share? Rate from 1-10.",
                "Does this show powerful vocals, instrumental solos, or crowd engagement? Rate energy 1-10.",
            ]
        elif "tutorial" in self.content_tags:
            return [
                "Rate this tutorial moment's importance from 1-10. Consider key explanations, demonstrations, and results. Respond with just the number.",
                "Is this a highlight-worthy tutorial moment that learners would want to reference? Rate from 1-10.",
                "Does this show important steps, before/after comparisons, or key insights? Rate importance 1-10.",
            ]
        else:
            return [
                f"Rate this {primary_tag} moment's interest level from 1-10. Consider engagement, importance, and shareability. Respond with just the number.",
                f"Is this a highlight-worthy {primary_tag} moment that viewers would want to see? Rate from 1-10.",
                f"Does this show important or exciting {primary_tag} content? Rate interest 1-10.",
            ]
    
    def _query_smolvlm_single_frame(self, frame: np.ndarray, prompt: str) -> str:
        """Query SmolVLM with a single frame and prompt"""
        try:
            import torch
            
            # Convert frame to PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # Format the prompt properly for SmolVLM
            formatted_prompt = f"<image>\n{prompt}"
            
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
                    max_new_tokens=50,
                    do_sample=False,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # Decode response
            generated_text = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the response part (after the prompt)
            if formatted_prompt in generated_text:
                response = generated_text.replace(formatted_prompt, "").strip()
            else:
                response = generated_text.strip()
            
            # Clean up the response further
            response = response.replace("<image>", "").strip()
            
            return response if response else "5"
            
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
            return 0.7
            
        except Exception as e:
            logger.warning(f"Failed to parse response '{response}': {e}")
            return 0.7
    
    def _analyze_audio_for_gaming_reactions(self, start_time: float, end_time: float) -> float:
        """Analyze audio for gaming reactions and excitement"""
        try:
            import librosa
            
            # Load audio segment
            y, sr = librosa.load(
                self.video_path,
                sr=22050,
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
            energy_score = min(avg_energy * 15, 1.0)  # High energy indicates excitement
            speech_score = min(avg_zcr * 25, 1.0)     # Speech activity
            excitement_score = min(avg_rolloff / 4000, 1.0)  # Higher pitch = excitement
            
            # Weighted combination
            audio_score = energy_score * 0.5 + speech_score * 0.3 + excitement_score * 0.2
            
            return min(audio_score, 1.0)
            
        except Exception as e:
            logger.warning(f"Audio analysis failed: {e}")
            return 0.3
    
    def _fallback_motion_analysis(self, start_time: float, end_time: float) -> float:
        """Fallback motion analysis when SmolVLM is not available"""
        cap = cv2.VideoCapture(self.video_path)
        
        start_frame = int(start_time * self.fps)
        end_frame = int(end_time * self.fps)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        motion_scores = []
        prev_frame = None
        
        # Sample every 30 frames for better motion detection
        for frame_num in range(start_frame, min(end_frame, start_frame + 300), 30):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (320, 180))
            
            if prev_frame is not None:
                diff = cv2.absdiff(prev_frame, gray)
                motion_score = np.mean(diff)
                motion_scores.append(motion_score)
            
            prev_frame = gray
        
        cap.release()
        
        if motion_scores:
            avg_motion = np.mean(motion_scores)
            # Enhanced normalization for gaming content
            normalized_score = min(avg_motion / 25.0, 1.0)
            
            # Add slight randomness for variety when using fallback
            import random
            normalized_score += random.uniform(0.05, 0.15)
            
            return min(normalized_score, 1.0)
        
        return 0.4  # Default score
    
    def _enhanced_motion_analysis(self) -> List[Tuple[float, float, float]]:
        """Enhanced motion analysis for gaming content when SmolVLM is not available"""
        logger.info("Using enhanced motion analysis for gaming highlights...")
        
        segments = []
        segment_length = 30  # 30-second segments
        
        for start in range(0, int(self.duration), segment_length):
            end = min(start + segment_length, self.duration)
            
            # Enhanced motion scoring for gaming
            motion_score = self._calculate_enhanced_motion_score(start, end)
            audio_score = self._analyze_audio_for_gaming_reactions(start, end)
            
            # Combine scores with gaming-focused weights
            combined_score = motion_score * 0.6 + audio_score * 0.4
            
            if combined_score > 0.5:  # Keep segments with good combined score
                segments.append((start, end, combined_score))
                logger.info(f"Enhanced highlight: {start}-{end}s (motion: {motion_score:.3f}, audio: {audio_score:.3f}, combined: {combined_score:.3f})")
        
        logger.info(f"Found {len(segments)} enhanced motion highlights")
        return segments
    
    def _find_candidate_segments(self) -> List[Tuple[float, float, float]]:
        """Quick motion analysis to find candidate segments for SmolVLM analysis"""
        candidates = []
        segment_length = 30
        
        for start in range(0, int(self.duration), segment_length):
            end = min(start + segment_length, self.duration)
            
            # Quick motion analysis
            motion_score = self._calculate_enhanced_motion_score(start, end)
            
            # Only consider segments with decent motion for SmolVLM analysis
            if motion_score > 0.4:
                candidates.append((start, end, motion_score))
        
        # Sort by motion score and take top candidates to limit SmolVLM calls
        candidates.sort(key=lambda x: x[2], reverse=True)
        max_candidates = min(10, len(candidates))  # Limit to 10 SmolVLM calls
        
        return candidates[:max_candidates]
    
    def _calculate_enhanced_motion_score(self, start_time: float, end_time: float) -> float:
        """Enhanced motion calculation optimized for gaming content"""
        cap = cv2.VideoCapture(self.video_path)
        
        start_frame = int(start_time * self.fps)
        end_frame = int(end_time * self.fps)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        motion_scores = []
        edge_scores = []
        prev_frame = None
        
        # Sample every 30 frames for good balance of speed and accuracy
        for frame_num in range(start_frame, min(end_frame, start_frame + 300), 30):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (320, 180))
            
            # Calculate edge density (indicates UI elements, text, detailed graphics)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.mean(edges) / 255.0
            edge_scores.append(edge_density)
            
            if prev_frame is not None:
                # Motion detection
                diff = cv2.absdiff(prev_frame, gray)
                motion_score = np.mean(diff)
                motion_scores.append(motion_score)
            
            prev_frame = gray
        
        cap.release()
        
        if motion_scores and edge_scores:
            avg_motion = np.mean(motion_scores)
            avg_edges = np.mean(edge_scores)
            
            # Gaming content often has high edge density (UI, text, detailed graphics)
            # Combine motion and edge information
            motion_component = min(avg_motion / 25.0, 1.0)
            edge_component = min(avg_edges * 2.0, 1.0)  # UI elements boost score
            
            # Weighted combination favoring motion but considering UI complexity
            enhanced_score = motion_component * 0.7 + edge_component * 0.3
            
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
            
            # Use a simpler, more lenient content-aware prompt
            primary_tag = self.content_tags[0] if self.content_tags else "general"
            prompt = f"Is this an exciting {primary_tag} moment? Rate 1-10 where 5+ means it's worth watching. Just give the number."
            
            response = self._query_smolvlm_single_frame(frame, prompt)
            score = self._parse_content_response(response)
            
            logger.info(f"SmolVLM response: '{response}' -> score: {score}")
            
            return score
            
        except Exception as e:
            logger.warning(f"SmolVLM segment analysis failed: {e}")
            return 0.0