#!/usr/bin/env python3
"""
Setup script for Video Highlights Extractor
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Core dependencies (always required)
core_requirements = [
    "opencv-python>=4.8.0",
    "moviepy>=1.0.3", 
    "librosa>=0.10.0",
    "scikit-learn>=1.3.0",
    "scipy>=1.11.0",
    "numpy>=1.24.0",
    "soundfile>=0.12.0",
    "ffmpeg-python>=0.2.0",
]

# Optional dependencies for different modes
ml_requirements = [
    "torch>=2.0.0",
    "torchvision>=0.15.0",
    "transformers>=4.40.0",
    "accelerate>=0.20.0",
]

smolvlm_requirements = ml_requirements + [
    "pillow>=9.0.0",
    "sentencepiece>=0.1.99",
]

setup(
    name="video-highlights-extractor",
    version="1.0.0",
    author="Video Highlights Extractor",
    description="A Python library for extracting vertical 9:16 highlights from videos using AI-powered scene detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=core_requirements,
    extras_require={
        "basic": [],  # No extra deps for basic mode
        "ml": ml_requirements,
        "smolvlm": smolvlm_requirements,
        "all": smolvlm_requirements,  # SmolVLM includes everything
    },
    entry_points={
        "console_scripts": [
            "video-highlights=video_highlights.__main__:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Multimedia :: Video",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="video highlights extraction ai computer-vision",
    project_urls={
        "Source": "https://github.com/your-username/video-highlights-extractor",
        "Bug Reports": "https://github.com/your-username/video-highlights-extractor/issues",
    },
)