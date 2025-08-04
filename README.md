# VideoBooth: Cross Frame Attention Implementation

A PyTorch implementation of the cross frame attention and cross attention mechanisms from the [VideoBooth paper](https://arxiv.org/abs/2312.00777): "VideoBooth: Diffusion-based Video Generation with Image Prompts".

## Overview

This project implements the core attention mechanisms that enable **temporally consistent video generation** with image prompts. The VideoBooth framework addresses key challenges in video generation:

- **Cross Frame Attention**: Maintains temporal consistency across video frames
- **Cross Attention**: Incorporates text and image prompt conditioning
- **Image Prompt Integration**: Uses reference images to guide video generation


## File Structure

```
video_booth/
├── attention.py      # Core attention mechanisms
├── transformer.py    # VideoBoothBlock implementation  
├── main.py          # Usage example
├── requirements.txt  # Python dependencies
└── README.md        # This file
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip

### Setup Virtual Environment

```bash
# Clone or download the project
# cd video_booth

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

Run the example to see the cross frame attention in action:

```bash
python main.py
```

**Expected Output:**
```
torch.Size([2, 64, 512]) torch.Size([2, 64, 512]) torch.Size([2, 64, 512])
```

This demonstrates processing 3 video frames with cross frame attention, where:
- Frame 0 attends to image prompts and establishes temporal anchors
- Frames 1-2 maintain consistency by attending to Frame 0's keys/values

