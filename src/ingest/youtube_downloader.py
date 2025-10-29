import os
from yt_dlp import YoutubeDL


def download_video(video_url: str, output_dir: str, format: str = "best") -> dict:
    """
    Download a YouTube video (and metadata) using yt-dlp.

    Args:
        video_url (str): The full YouTube video URL.
        output_dir (str): Directory where the video and metadata will be saved.
        format (str): Desired format (default: "best").

    Returns:
        dict: Metadata dictionary returned by yt-dlp (includes id, title, duration, etc.)
    """
    os.makedirs(output_dir, exist_ok=True)

    ydl_opts = {
        "format": format,
        "outtmpl": os.path.join(output_dir, "%(id)s.%(ext)s"),
        "writeinfojson": True,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "progress_hooks": [lambda d: print_progress(d)],
    }

    try:
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)

        video_id = info.get("id")
        title = info.get("title", "Untitled")
        filepath = os.path.join(output_dir, f"{video_id}.{info.get('ext', 'mp4')}")

        print(f"\n✅ Download complete: {title} ({video_id})")
        print(f"📂 Saved to: {filepath}")

        return info

    except Exception as e:
        print(f"❌ Failed to download video: {e}")
        return {}
    

def print_progress(d):
    """Optional progress hook to show download progress."""
    if d["status"] == "downloading":
        percent = d.get("_percent_str", "").strip()
        speed = d.get("_speed_str", "")
        eta = d.get("eta", 0)
        print(f"\r⬇️  {percent} ({speed}) ETA: {eta}s", end="")
    elif d["status"] == "finished":
        print("\n📦 Download finished, finalizing...")
