import cv2
import os
import json
from ultralytics import YOLO
from tqdm import tqdm

# === CONFIGURATION ===
VIDEO_PATH = "data/raw_videos/sample_video.mp4"  # change this
OUTPUT_FRAMES = "data/frames"
OUTPUT_FACES = "data/faces"
OUTPUT_JSON = "data/outputs/face_detections.json"
FRAME_INTERVAL = 30  # every 30th frame (~1 sec at 30 FPS)

# === PREPARE DIRECTORIES ===
os.makedirs(OUTPUT_FRAMES, exist_ok=True)
os.makedirs(OUTPUT_FACES, exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)

# === LOAD YOLOv8 FACE MODEL ===
# Tip: try a face-specific model like "yolov8n-face.pt" if available
model = YOLO("models/yolov8n-face.pt")  # change this if needed

# === LOAD VIDEO ===
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

detections = []

print(f"Processing video: {VIDEO_PATH}")
print(f"Total frames: {frame_count}, FPS: {fps}")

frame_id = 0
saved_id = 0
face_id = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_id % FRAME_INTERVAL == 0:
        timestamp = round(frame_id / fps, 2)

        # Run detection
        results = model.predict(source=frame, conf=0.4, verbose=False)

        # Extract detections
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy()

            for box, conf, cls in zip(boxes, confs, clss):
                x1, y1, x2, y2 = [int(x) for x in box]
                label = model.names[int(cls)]

                # Crop the detected region (face/person)
                cropped_face = frame[y1:y2, x1:x2]
                if cropped_face.size == 0:
                    continue

                face_filename = f"face_{face_id:05d}.jpg"
                face_path = os.path.join(OUTPUT_FACES, face_filename)
                cv2.imwrite(face_path, cropped_face)

                # Save detection data
                detections.append({
                    "frame_id": saved_id,
                    "timestamp": timestamp,
                    "bbox": [x1, y1, x2, y2],
                    "confidence": float(conf),
                    "label": label,
                    "face_image": face_filename
                })

                # Draw bounding box on frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                face_id += 1

        # Save annotated frame
        frame_filename = f"frame_{saved_id:05d}.jpg"
        cv2.imwrite(os.path.join(OUTPUT_FRAMES, frame_filename), frame)
        saved_id += 1

    frame_id += 1

cap.release()
cv2.destroyAllWindows()

# === SAVE RESULTS AS JSON ===
with open(OUTPUT_JSON, "w") as f:
    json.dump(detections, f, indent=2)

print(f"\n✅ Done!")
print(f" - Frames saved: {saved_id}")
print(f" - Faces cropped: {face_id}")
print(f" - Results JSON: {OUTPUT_JSON}")
