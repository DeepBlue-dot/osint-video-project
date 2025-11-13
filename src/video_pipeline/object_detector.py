# video_pipeline/object_detector.py

import os
import json
from ultralytics import YOLO
from tqdm import tqdm
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "yolov8n.pt")

def detect_objects(frames_dir, outputs_dir, video_id, conf_threshold=0.4):
    """
    Detect objects in video frames using YOLOv8.
    Saves JSON results with bounding boxes and confidence scores.
    """

    model = YOLO(MODEL_PATH)

    results_data = []
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(('.jpg', '.png'))])

    print(f"🔍 Running object detection on {len(frame_files)} frames...")

    for frame_file in tqdm(frame_files):
        frame_path = os.path.join(frames_dir, frame_file)
        results = model.predict(frame_path, conf=conf_threshold, verbose=False)

        # YOLOv8 returns detections in results[0].boxes
        frame_result = {
            "frame": frame_file,
            "detections": []
        }

        for box in results[0].boxes:
            label = model.names[int(box.cls)]
            conf = float(box.conf)
            xyxy = [float(x) for x in box.xyxy[0].tolist()]  # [x1, y1, x2, y2]

            frame_result["detections"].append({
                "label": label,
                "confidence": round(conf, 3),
                "bbox": xyxy
            })

        results_data.append(frame_result)

    output_path = os.path.join(outputs_dir, f"{video_id}_objects.json")
    os.makedirs(outputs_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)

    print(f"✅ Object detection completed. Results saved to {output_path}")
    return output_path
