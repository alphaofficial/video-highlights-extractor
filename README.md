# Video Highlights Extractor

A CLI tool that automatically extracts 9:16 vertical highlights from videos using AI-powered scene detection.

## Features

- **Smart Highlight Detection**: Uses audio energy analysis and visual motion detection
- **9:16 Aspect Ratio**: Perfect for social media (Instagram, TikTok, YouTube Shorts)
- **Minimum Duration**: Ensures each highlight is at least 90 seconds (configurable)
- **Center Cropping**: Automatically centers the crop for best composition
- **Multiple Formats**: Supports MP4, AVI, MOV, MKV, WebM, and more

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Install FFmpeg (required for video processing):
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

## Available Versions

### 1. SmolVLM Version (`smolvlm_highlights.py`) - **RECOMMENDED**
Uses SmolVLM vision-language model for intelligent gaming highlight detection:
```bash
# Install ML dependencies
pip install -r requirements_smolvlm.txt

# Run SmolVLM version
python smolvlm_highlights.py path/to/your/video.mp4 --min-duration 30
```

### 2. Basic Version (`video_highlights.py`)
Audio energy analysis and visual motion detection:
```bash
python video_highlights.py path/to/your/video.mp4 --min-duration 30
```

### 3. Quick Test (`quick_test.py`)
Fast processing for testing:
```bash
python quick_test.py path/to/your/video.mp4 --min-duration 30
```

### Options

- `--min-duration`: Duration for each highlight in seconds (default: 30)
- `--max-highlights`: Maximum number of highlights (auto-calculated if not specified)

## How SmolVLM Version Works

**Most intelligent option for gaming content:**

1. **Vision Analysis**: SmolVLM analyzes video frames to identify:
   - Combat and action sequences
   - Explosions and gunfire  
   - Fast movement and intense gameplay
   - Exciting gaming moments

2. **Audio Analysis**: Detects energy spikes and audio intensity
3. **Smart Scoring**: Combines visual and audio analysis for optimal highlights
4. **Intelligent Selection**: Avoids overlapping clips and selects best moments

## Output

Highlights are saved in a folder named `{original_filename}_highlights/` with files named:
- `highlight_1.mp4` (highest scored)
- `highlight_2.mp4`
- etc.

Each output video is:
- 1080x1920 resolution (9:16 aspect ratio)
- H.264 encoded for compatibility
- Center-cropped from the original video