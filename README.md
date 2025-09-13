<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>

<!-- ABOUT THE PROJECT -->
## About The Project

Identifies and extracts the most engaging moments from videos, converting them into vertical 9:16 format perfect for social media platforms like TikTok, Instagram Reels, and YouTube Shorts.

<!-- GETTING STARTED -->
## Getting Started

### Prerequisites

* Python 3.8 or higher
* FFmpeg (for video processing)
  ```sh
  # macOS
  brew install ffmpeg

  # Ubuntu/Debian
  sudo apt update && sudo apt install ffmpeg

  # Windows - Download from https://ffmpeg.org/download.html
  ```

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/alphaofficial/video-highlights-extractor.git
   cd video-highlights-extractor
   ```

2. Create a virtual environment
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the package
   ```sh
   # Basic installation (audio + visual analysis)
   pip install -e .

   # With ML capabilities
   pip install -e .[ml]

   # With SmolVLM (content-aware AI)
   pip install -e .[smolvlm]

   # Full installation (all features)
   pip install -e .[all]
   ```


<!-- USAGE EXAMPLES -->
## Usage

### Basic Usage

Extract highlights using default settings:
```bash
python -m video_extractor video.mp4
```

### Content-Specific Extraction

Optimize extraction for specific content types:

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
python -m video_extractor tutorial.mp4 --mode smolvlm --tags tutorial

# Multiple tags
python -m video_extractor video.mp4 --mode smolvlm --tags gaming sports
```

### Advanced Options

```bash
python -m video_extractor video.mp4 \
  --mode smolvlm \
  --tags gaming \
  --min-duration 45 \
  --max-highlights 3
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--mode` | Analysis mode (`basic`, `ml`, `smolvlm`) | `basic` |
| `--tags` | Content tags for optimization (optional) | None |
| `--min-duration` | Minimum highlight duration in seconds | `30` |
| `--max-highlights` | Maximum number of highlights | Auto-calculated |


<!-- ANALYSIS MODES -->
## Analysis Modes

### 1. Basic Mode
- **Audio Energy Analysis**: Detects volume spikes and audio intensity
- **Visual Motion Detection**: Identifies movement and scene changes
- **Lightweight**: No AI dependencies required
- **Best for**: Quick processing, general content

### 2. ML Mode
- **Image Classification**: Uses ResNet-50 for action detection
- **Advanced Audio Analysis**: Multiple audio features analysis
- **Action Recognition**: Identifies dynamic content patterns
- **Best for**: Sports, action videos, general content

### 3. SmolVLM Mode
- **Vision-Language Understanding**: Uses SmolVLM for intelligent scene analysis
- **Content-Aware**: Adapts to different content types using tags
- **Context Understanding**: Recognizes content-specific highlight moments
- **GPU Accelerated**: Automatic CUDA support for faster processing
- **Best for**: Any content type with appropriate tags


<!-- CONTENT TAGS -->
## Content Tags

Content tags help the AI understand what to look for in your videos:

| Tag | Optimized For | Detects |
|-----|---------------|---------|
| `gaming` | Video games, esports | Eliminations, clutch plays, victories, skill demonstrations |
| `cooking` | Recipe tutorials, food prep | Technique demonstrations, ingredient reveals, final dishes |
| `sports` | Athletic competitions | Goals, amazing plays, dramatic moments, celebrations |
| `music` | Concerts, performances | Performance peaks, crowd reactions, musical highlights |
| `tutorial` | Educational content | Key explanations, demonstrations, before/after results |

**Note**: Tags are completely optional. Without tags, the system uses generic highlight detection.

<!-- ROADMAP -->
## Roadmap

- [x] Basic audio/visual analysis
- [x] ML-powered highlight detection
- [x] SmolVLM integration for content understanding
- [x] Content-aware tagging system
- [ ] Web interface for easier usage
- [ ] Batch processing for multiple videos
- [ ] Custom model training capabilities
- [ ] Real-time streaming highlight detection
- [ ] Integration with social media APIs


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



