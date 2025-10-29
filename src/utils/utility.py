import re
import os

DATA_DIR = "data"

def extract_video_id(url: str) -> str:
    """Extract the YouTube video ID from any valid YouTube URL."""
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",  # normal or embed URLs
    ]
    for p in patterns:
        match = re.search(p, url)
        if match:
            return match.group(1)
    raise ValueError(f"❌ Could not extract video ID from URL: {url}")


def init_directories(video_id: str):
    """Create and return structured directories for a given video ID."""
    raw_dir = os.path.join(DATA_DIR, "raw_videos", video_id)
    frames_dir = os.path.join(DATA_DIR, "frames", video_id)
    faces_dir = os.path.join(DATA_DIR, "faces", video_id)
    outputs_dir = os.path.join(DATA_DIR, "outputs", video_id)

    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(faces_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)

    print(f"✅ Initialized directories for video ID: {video_id}")
    print(f"📂 Raw video folder: {raw_dir}")

    return raw_dir, frames_dir, faces_dir, outputs_dir
