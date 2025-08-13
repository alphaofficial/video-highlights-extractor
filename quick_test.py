#!/usr/bin/env python3
"""
Quick test version - creates highlights from fixed intervals for testing
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path
import cv2
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_quick_highlights(video_path: str, num_highlights: int = None, min_duration: int = 90):
    """Create highlights from evenly spaced intervals for quick testing"""
    
    # Get video info
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    cap.release()
    
    logger.info(f"Video: {duration:.1f}s, {fps:.1f}fps, {total_frames} frames")
    
    # Auto-determine number of highlights if not specified
    if num_highlights is None:
        highlight_coverage = 0.7  # Use 70% of video for highlights
        usable_duration = duration * highlight_coverage
        num_highlights = int(usable_duration / min_duration)
        num_highlights = max(2, min(num_highlights, 20))  # Between 2-20 highlights
    
    logger.info(f"Creating {num_highlights} highlights of {min_duration}s each")
    
    # Create output directory
    output_dir = Path(video_path).parent / f"{Path(video_path).stem}_highlights"
    output_dir.mkdir(exist_ok=True)
    
    # Create evenly spaced highlights
    segment_duration = min_duration
    highlights = []
    
    # Calculate spacing between highlights to avoid overlap
    available_time = duration - (num_highlights * segment_duration)
    spacing = available_time / (num_highlights + 1) if num_highlights > 1 else 0
    
    for i in range(num_highlights):
        # Start each highlight with proper spacing
        start_time = spacing * (i + 1) + (segment_duration * i)
        end_time = start_time + segment_duration
        
        # Ensure we don't exceed video duration
        if end_time > duration:
            end_time = duration
            start_time = max(0, end_time - segment_duration)
        
        if start_time < duration and end_time - start_time >= min_duration:
            highlights.append((start_time, end_time))
    
    # Create clips
    output_files = []
    for i, (start, end) in enumerate(highlights):
        output_file = output_dir / f"highlight_{i+1}.mp4"
        
        logger.info(f"Creating highlight {i+1}: {start:.1f}s - {end:.1f}s ({end-start:.1f}s duration)")
        
        # Use FFmpeg for efficient processing with better compatibility
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
            '-pix_fmt', 'yuv420p',  # Ensure compatibility
            '-movflags', '+faststart',  # Web optimization
            '-profile:v', 'baseline',  # Better compatibility
            '-level', '3.0',
            str(output_file)
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            output_files.append(str(output_file))
            logger.info(f"Created: {output_file}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create highlight {i+1}: {e}")
            logger.error(f"FFmpeg stderr: {e.stderr}")
    
    return output_files

def main():
    parser = argparse.ArgumentParser(description='Quick test - create highlights from fixed intervals')
    parser.add_argument('video_path', help='Path to input video file')
    parser.add_argument('--num-highlights', type=int, default=None, help='Number of highlights (auto-calculated if not specified)')
    parser.add_argument('--min-duration', type=int, default=90, help='Minimum duration per highlight (default: 90)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video_path):
        logger.error(f"Video file not found: {args.video_path}")
        sys.exit(1)
    
    try:
        output_files = create_quick_highlights(args.video_path, args.num_highlights, args.min_duration)
        
        logger.info(f"Successfully created {len(output_files)} highlight clips:")
        for file in output_files:
            print(f"  {file}")
            
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()