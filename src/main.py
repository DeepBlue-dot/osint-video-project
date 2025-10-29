import sys
from ingest.youtube_downloader import download_video
from utils.utility import extract_video_id
from utils.utility import init_directories


def process_video(url: str):
    """Full video processing pipeline."""
    video_id = extract_video_id(url)
    raw_dir, frames_dir, faces_dir, outputs_dir = init_directories(video_id)
    print(f"🎥 Processing video from URL: {url}")
    download_video(url, raw_dir)
    print(f"✅ Done downloading {video_id}")


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "https://www.youtube.com/watch?v=YzcawvDGe4Y"
    process_video(url)
