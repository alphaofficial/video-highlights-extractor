#!/usr/bin/env python3
"""
Video Highlights Extractor - Main CLI Entry Point

Usage:
    python -m video_highlights video.mp4 --mode basic --min-duration 30
    python -m video_highlights video.mp4 --mode ml --min-duration 30  
    python -m video_highlights video.mp4 --mode smolvlm --min-duration 30
"""

import argparse
import sys
import os
from pathlib import Path

# Fix tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def create_parser():
    """Create the argument parser with all supported flags"""
    parser = argparse.ArgumentParser(
        description="Extract vertical 9:16 highlights from videos using AI-powered scene detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic mode (audio + visual analysis)
  python -m video_highlights video.mp4 --mode basic --min-duration 30

  # ML mode (with machine learning models)  
  python -m video_highlights video.mp4 --mode ml --min-duration 30

  # SmolVLM mode (with vision-language model)
  python -m video_highlights video.mp4 --mode smolvlm --min-duration 30

Installation:
  pip install -e .              # Basic mode only
  pip install -e .[ml]          # Basic + ML mode
  pip install -e .[smolvlm]     # Basic + SmolVLM mode  
  pip install -e .[all]         # All modes
        """
    )
    
    # Required arguments
    parser.add_argument(
        "video_path",
        help="Path to the input video file"
    )
    
    # Mode selection
    parser.add_argument(
        "--mode",
        choices=["basic", "ml", "smolvlm"],
        default="basic",
        help="Extraction mode to use (default: basic)"
    )
    
    # Common options (maintaining existing flags)
    parser.add_argument(
        "--min-duration",
        type=int,
        default=30,
        help="Minimum duration for each highlight in seconds (default: 30)"
    )
    
    parser.add_argument(
        "--max-highlights", 
        type=int,
        help="Maximum number of highlights to extract (auto-calculated if not specified)"
    )
    
    parser.add_argument(
        "--tags",
        nargs="+",
        default=["general"],
        help="Content tags to optimize highlight detection (e.g., gaming, cooking, sports, music, tutorial, vlog)"
    )
    
    return parser

def validate_dependencies(mode: str) -> bool:
    """Check if required dependencies are installed for the selected mode"""
    try:
        if mode in ["ml", "smolvlm"]:
            import torch
            import transformers
            
        if mode == "smolvlm":
            from PIL import Image
            import sentencepiece
            
        return True
        
    except ImportError as e:
        missing_dep = str(e).split("'")[1] if "'" in str(e) else str(e)
        print(f"Error: Missing dependency '{missing_dep}' for {mode} mode")
        print(f"Install with: pip install -e .[{mode}]")
        return False

def main():
    """Main entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    # Validate input file
    if not Path(args.video_path).exists():
        print(f"Error: Video file not found: {args.video_path}")
        sys.exit(1)
    
    # Validate dependencies
    if not validate_dependencies(args.mode):
        sys.exit(1)
    
    # Import and create the appropriate extractor
    try:
        if args.mode == "basic":
            from .core.basic import BasicHighlightExtractor
            extractor = BasicHighlightExtractor(
                video_path=args.video_path,
                min_duration=args.min_duration,
                max_highlights=args.max_highlights,
                mode="basic",
                content_tags=args.tags
            )
            
        elif args.mode == "ml":
            from .core.ml import MLHighlightExtractor
            extractor = MLHighlightExtractor(
                video_path=args.video_path,
                min_duration=args.min_duration,
                max_highlights=args.max_highlights,
                mode="ml",
                content_tags=args.tags
            )
            
        elif args.mode == "smolvlm":
            from .core.smolvlm import SmolVLMHighlightExtractor
            extractor = SmolVLMHighlightExtractor(
                video_path=args.video_path,
                min_duration=args.min_duration,
                max_highlights=args.max_highlights,
                mode="smolvlm",
                content_tags=args.tags
            )
            
    except ImportError as e:
        print(f"Error importing {args.mode} extractor: {e}")
        print(f"Make sure you have installed the required dependencies: pip install -e .[{args.mode}]")
        sys.exit(1)
    
    # Extract highlights
    try:
        output_files = extractor.extract_highlights()
        
        if output_files:
            print(f"\n✓ Successfully created {len(output_files)} highlights:")
            for file_path in output_files:
                print(f"  - {Path(file_path).name}")
        else:
            print("\n⚠ No highlights were extracted from the video")
            
    except Exception as e:
        print(f"Error during extraction: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()