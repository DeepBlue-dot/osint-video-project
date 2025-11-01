import cv2, os

def extract_frames(video_path, output_dir, frame_rate=1):
    """Extract frames from video every N seconds."""
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = int(fps * frame_rate)
    count = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            frame_name = os.path.join(output_dir, f"frame_{saved:05d}.jpg")
            cv2.imwrite(frame_name, frame)
            saved += 1
        count += 1
    cap.release()
    print(f"Extracted {saved} frames → {output_dir}")
