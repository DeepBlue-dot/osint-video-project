import os
import sys
from ingest.youtube_downloader import download_video
from utils.utility import extract_video_id
from utils.utility import init_directories
from video_pipeline.frame_extractor import extract_frames
from video_pipeline.face_detector import detect_faces
from audio_pipeline.audio_extractor import extract_audio
from audio_pipeline.speech_to_text import transcribe_audio
from video_pipeline.object_detector import detect_objects


def process_video(url: str):
    video_id = extract_video_id(url)
    raw_dir, frames_dir, faces_dir, outputs_dir, AUDIO_PATH, TRANSCRIPT_PATH = init_directories(video_id)
    video_path = os.path.join(raw_dir, f"{video_id}.mp4")

    print(f"🎥 Processing video from URL: {url}")
    download_video(url, raw_dir)
    print(f"✅ Done downloading {video_id}")

    extract_frames(video_path, frames_dir, frame_rate=1)

    # 🧩 NEW: Object detection
    detect_objects(frames_dir, outputs_dir, video_id)

    detect_faces(frames_dir, outputs_dir, faces_dir, video_id)

    extract_audio(video_path, AUDIO_PATH)
    transcribe_audio(AUDIO_PATH, TRANSCRIPT_PATH)



if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "https://www.youtube.com/watch?v=YzcawvDGe4Y"
    process_video(url)
