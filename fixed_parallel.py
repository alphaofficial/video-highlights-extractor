#!/usr/bin/env python3
"""
Fixed Parallel Video Highlights Extractor
Actually works with proper parallel processing and tracking.
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
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_single_clip(clip_data):
    """Create a single clip - standalone function for ProcessPoolExecutor"""
    clip_index, start_time, end_time, video_path, output_dir = clip_data
    
    output_file = Path(output_dir) / f"highlight_{clip_index:02d}.mp4"
    
    cmd = [
        'ffmpeg', '-y', '-loglevel', 'error',
        '-i', str(video_path),
        '-ss', str(start_time),
        '-t', str(end_time - start_time),
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
    
    start_process_time = time.time()
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        process_time = time.time() - start_process_time
        return {
            'success': True,
            'file': str(output_file),
            'index': clip_index,
            'duration': end_time - start_time,
            'process_time': process_time
        }
    except subprocess.CalledProcessError as e:
        return {
            'success': False,
            'index': clip_index,
            'error': str(e.stderr),
            'process_time': time.time() - start_process_time
        }

class FixedParallelExtractor:
    def __init__(self, video_path: str, min_duration: int = 30):
        self.video_path = video_path
        self.min_duration = min_duration
        self.output_dir = Path(video_path).parent / f"{Path(video_path).stem}_highlights"
        self.output_dir.mkdir(exist_ok=True)
        
    def analyze_video(self) -> List[Tuple[float, float]]:
        """Analyze video and return highlight time ranges"""
        logger.info("Analyzing video for highlights...")
        
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        cap.release()
        
        logger.info(f"Video: {duration:.1f}s, {fps:.1f}fps, {total_frames} frames")
        
        # Calculate number of highlights based on duration
        highlight_coverage = 0.7
        usable_duration = duration * highlight_coverage
        num_highlights = int(usable_duration / self.min_duration)
        num_highlights = max(3, min(num_highlights, 12))  # Between 3-12 highlights
        
        logger.info(f"Will create {num_highlights} highlights of {self.min_duration}s each")
        
        # Audio analysis for intelligent placement
        audio_peaks = self._get_audio_peaks(duration)
        
        if audio_peaks:
            highlights = self._create_highlights_from_peaks(audio_peaks, duration, num_highlights)
        else:
            highlights = self._create_evenly_spaced_highlights(duration, num_highlights)
        
        return highlights
    
    def _get_audio_peaks(self, duration: float) -> List[float]:
        """Get audio peak times for intelligent highlight placement"""
        try:
            logger.info("Analyzing audio for peak detection...")
            y, sr = librosa.load(self.video_path, sr=22050)
            
            # RMS energy in 2-second windows
            hop_length = sr * 2
            rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
            
            # Find peaks above 70th percentile
            threshold = np.percentile(rms, 70)
            peaks, _ = find_peaks(rms, height=threshold, distance=3)  # 6 second minimum distance
            
            # Convert to time
            peak_times = [peak * hop_length / sr for peak in peaks]
            logger.info(f"Found {len(peak_times)} audio peaks")
            return peak_times
            
        except Exception as e:
            logger.warning(f"Audio analysis failed: {e}")
            return []
    
    def _create_highlights_from_peaks(self, peaks: List[float], duration: float, num_highlights: int) -> List[Tuple[float, float]]:
        """Create highlights centered around audio peaks"""
        highlights = []
        used_ranges = []
        
        # Sort peaks by intensity (we'll use the original order as proxy)
        for peak_time in peaks[:num_highlights * 2]:  # Consider more peaks than needed
            # Center highlight around peak
            start_time = max(0, peak_time - self.min_duration / 2)
            end_time = min(duration, start_time + self.min_duration)
            
            # Adjust if we hit boundaries
            if end_time - start_time < self.min_duration:
                if start_time == 0:
                    end_time = min(duration, self.min_duration)
                else:
                    start_time = max(0, duration - self.min_duration)
            
            # Check for overlap with existing highlights
            overlap = False
            for used_start, used_end in used_ranges:
                if not (end_time <= used_start or start_time >= used_end):
                    overlap = True
                    break
            
            if not overlap and end_time - start_time >= self.min_duration:
                highlights.append((start_time, end_time))
                used_ranges.append((start_time, end_time))
                
                if len(highlights) >= num_highlights:
                    break
        
        # Fill remaining slots with evenly spaced highlights if needed
        while len(highlights) < num_highlights:
            remaining = num_highlights - len(highlights)
            spacing = duration / (remaining + 1)
            
            for i in range(remaining):
                candidate_start = spacing * (i + 1) - self.min_duration / 2
                candidate_start = max(0, candidate_start)
                candidate_end = min(duration, candidate_start + self.min_duration)
                
                # Check for overlap
                overlap = False
                for used_start, used_end in used_ranges:
                    if not (candidate_end <= used_start or candidate_start >= used_end):
                        overlap = True
                        break
                
                if not overlap:
                    highlights.append((candidate_start, candidate_end))
                    used_ranges.append((candidate_start, candidate_end))
                    break
            else:
                break  # Couldn't find non-overlapping slot
        
        return highlights
    
    def _create_evenly_spaced_highlights(self, duration: float, num_highlights: int) -> List[Tuple[float, float]]:
        """Create evenly spaced highlights as fallback"""
        logger.info("Creating evenly spaced highlights")
        highlights = []
        
        # Calculate spacing to avoid overlap
        total_highlight_time = num_highlights * self.min_duration
        available_space = duration - total_highlight_time
        spacing = available_space / (num_highlights + 1) if num_highlights > 1 else 0
        
        current_time = spacing
        for i in range(num_highlights):
            start_time = current_time
            end_time = start_time + self.min_duration
            
            if end_time <= duration:
                highlights.append((start_time, end_time))
                current_time = end_time + spacing
            else:
                break
        
        return highlights
    
    def create_highlights_parallel(self, highlights: List[Tuple[float, float]], max_workers: int = None) -> dict:
        """Create highlight clips in parallel and return detailed results"""
        if not highlights:
            return {'success': False, 'error': 'No highlights to create'}
        
        max_workers = max_workers or min(len(highlights), mp.cpu_count())
        logger.info(f"Creating {len(highlights)} highlights in parallel using {max_workers} workers")
        
        # Prepare clip data for parallel processing
        clip_data_list = []
        for i, (start_time, end_time) in enumerate(highlights):
            clip_data = (i + 1, start_time, end_time, self.video_path, self.output_dir)
            clip_data_list.append(clip_data)
            logger.info(f"  Highlight {i+1}: {start_time:.1f}s - {end_time:.1f}s ({end_time-start_time:.1f}s)")
        
        # Track timing
        overall_start_time = time.time()
        results = []
        
        # Execute in parallel
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_index = {executor.submit(create_single_clip, clip_data): clip_data[0] 
                             for clip_data in clip_data_list}
            
            # Collect results as they complete
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    if result['success']:
                        logger.info(f"✓ Highlight {result['index']} completed in {result['process_time']:.1f}s")
                    else:
                        logger.error(f"✗ Highlight {result['index']} failed: {result['error']}")
                        
                except Exception as e:
                    logger.error(f"✗ Highlight {index} crashed: {e}")
                    results.append({
                        'success': False,
                        'index': index,
                        'error': str(e),
                        'process_time': 0
                    })
        
        overall_time = time.time() - overall_start_time
        
        # Analyze results
        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]
        
        total_video_duration = sum(r.get('duration', 0) for r in successful)
        total_process_time = sum(r.get('process_time', 0) for r in results)
        
        summary = {
            'success': len(successful) > 0,
            'total_clips': len(highlights),
            'successful_clips': len(successful),
            'failed_clips': len(failed),
            'total_video_duration': total_video_duration,
            'total_process_time': total_process_time,
            'overall_time': overall_time,
            'parallel_efficiency': total_process_time / overall_time if overall_time > 0 else 0,
            'output_files': [r['file'] for r in successful],
            'failed_indices': [r['index'] for r in failed]
        }
        
        return summary

def main():
    parser = argparse.ArgumentParser(description='Fixed parallel video highlight extraction')
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
        extractor = FixedParallelExtractor(args.video_path, args.min_duration)
        
        # Analyze video
        highlights = extractor.analyze_video()
        if not highlights:
            logger.error("No highlights found")
            sys.exit(1)
        
        # Create highlights in parallel
        results = extractor.create_highlights_parallel(highlights, args.max_workers)
        
        # Report results
        if results['success']:
            logger.info(f"\n🎉 SUCCESS! Created {results['successful_clips']}/{results['total_clips']} highlights")
            logger.info(f"📊 Performance Stats:")
            logger.info(f"   Total video duration: {results['total_video_duration']:.1f}s")
            logger.info(f"   Total processing time: {results['total_process_time']:.1f}s")
            logger.info(f"   Overall wall time: {results['overall_time']:.1f}s")
            logger.info(f"   Parallel efficiency: {results['parallel_efficiency']:.1f}x")
            
            if results['failed_clips'] > 0:
                logger.warning(f"⚠️  {results['failed_clips']} clips failed: {results['failed_indices']}")
            
            logger.info(f"\n📁 Output files:")
            for file in sorted(results['output_files']):
                print(f"  {file}")
        else:
            logger.error("❌ Failed to create highlights")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"💥 Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()