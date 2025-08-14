# Video Highlights Extractor

A Python library that automatically extracts 9:16 vertical highlights from videos using AI-powered scene detection.

## Features

- **Smart Highlight Detection**: Multiple AI-powered modes for different use cases
- **9:16 Aspect Ratio**: Perfect for social media (Instagram, TikTok, YouTube Shorts)
- **Configurable Duration**: Set minimum duration for each highlight (default: 30 seconds)
- **Center Cropping**: Automatically centers the crop for best composition
- **Multiple Formats**: Supports MP4, AVI, MOV, MKV, WebM, and more

## Installation

### Quick Start (Basic Mode)
```bash
pip install -e .
```

### With Optional AI Features
```bash
# For ML-powered analysis
pip install -e .[ml]

# For SmolVLM vision-language model (content-aware)
pip install -e .[smolvlm]

# For all features
pip install -e .[all]
```

### FFmpeg Requirement
FFmpeg is required for video processing:
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

## Usage

The library provides a unified command-line interface with different analysis modes:

### Basic Mode (Audio + Visual Analysis)
```bash
python -m video_extractor video.mp4 --mode basic --min-duration 30
```

### ML Mode (Machine Learning Models)
```bash
python -m video_extractor video.mp4 --mode ml --min-duration 30
```

### SmolVLM Mode (Vision-Language Model) - **Content-Aware**
```bash
python -m video_extractor video.mp4 --mode smolvlm --min-duration 30 --tags gaming
```

### Content-Specific Examples
```bash
# Gaming content
python -m video_extractor gameplay.mp4 --mode smolvlm --tags gaming

# Cooking videos
python -m video_extractor recipe.mp4 --mode smolvlm --tags cooking

# Sports highlights
python -m video_extractor match.mp4 --mode smolvlm --tags sports

# Music performances
python -m video_extractor concert.mp4 --mode smolvlm --tags music

# Tutorial content
python -m video_extractor howto.mp4 --mode smolvlm --tags tutorial

# Multiple tags
python -m video_extractor video.mp4 --mode smolvlm --tags gaming sports
```

### Available Options

- `--mode`: Analysis mode (`basic`, `ml`, `smolvlm`) - default: `basic`
- `--min-duration`: Minimum duration for each highlight in seconds - default: `30`
- `--max-highlights`: Maximum number of highlights (auto-calculated if not specified)
- `--tags`: Content tags to optimize detection (`gaming`, `cooking`, `sports`, `music`, `tutorial`, etc.) - default: `general`

## Analysis Modes

### 1. Basic Mode
- **Audio Energy Analysis**: Detects volume spikes and audio intensity
- **Visual Motion Detection**: Identifies fast movement and scene changes
- **Lightweight**: No AI dependencies required
- **Best for**: General content, quick processing

### 2. ML Mode
- **Image Classification**: Uses ResNet-50 for action detection
- **Advanced Audio Analysis**: Multiple audio features (energy, brightness, activity)
- **Action Recognition**: Identifies sports, games, and dynamic content
- **Best for**: Sports, action videos, general content

### 3. SmolVLM Mode ⭐ **Content-Aware AI**
- **Vision-Language Understanding**: Uses SmolVLM for intelligent scene analysis
- **Content-Optimized**: Adapts to different content types using tags (gaming, cooking, sports, etc.)
- **Context-Aware**: Understands content-specific moments and highlights
- **Smart Scoring**: Combines visual understanding with audio analysis
- **GPU Accelerated**: Automatically uses CUDA if available for faster processing
- **Best for**: Any content type when you specify appropriate tags

#### Supported Content Tags:
- **`gaming`**: FPS, MOBA, Battle Royale, Racing games - detects eliminations, clutch plays, victories
- **`cooking`**: Recipe tutorials, baking, grilling - highlights technique demonstrations, reveals
- **`sports`**: Football, basketball, soccer - finds goals, amazing plays, dramatic moments  
- **`music`**: Live performances, concerts - captures performance peaks, crowd reactions
- **`tutorial`**: How-to, educational - identifies key explanations, demonstrations, results
- **`general`**: Default for any other content type

## Output

Highlights are saved in a folder named `{original_filename}_highlights/` with files named:
- `highlight_01.mp4` (highest scored)
- `highlight_02.mp4`
- etc.

Each output video is:
- **1080x1920 resolution** (9:16 aspect ratio)
- **H.264 encoded** for compatibility with all platforms
- **Center-cropped** from the original video for optimal framing
- **Preserves audio** with AAC encoding

## Performance Tips

- **GPU Acceleration**: SmolVLM and ML modes automatically use CUDA if available
- **Processing Time**: Expect ~1-2 minutes per minute of video (varies by mode and hardware)
- **Memory Usage**: Ensure sufficient RAM for longer videos (8GB+ recommended)
- **Storage**: Output files are typically 10-20% the size of the original

## Development

To contribute or modify the library:

```bash
# Clone and install in development mode
git clone <repository-url>
cd video-highlights-extractor
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e .[all]
```

The library uses a modular architecture with separate extractors for each mode in `video_extractor/core/`.# video-highlights-extractor
