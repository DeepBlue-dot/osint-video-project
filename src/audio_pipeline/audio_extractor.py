import os
import ffmpeg

def extract_audio(video_path, output_path):
    """
    Extract audio from a video file and save as WAV.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    (
        ffmpeg
        .input(video_path)
        .output(output_path, format='wav', acodec='pcm_s16le', ac=1, ar='16k')
        .run(quiet=True, overwrite_output=True)
    )
    print(f"🎵 Audio extracted → {output_path}")
