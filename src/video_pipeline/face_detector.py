import os
import cv2
import json
import uuid
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "yolov8n-face.pt")

def detect_faces(frames_dir, output_dir, output_crop_dir, video_id):
    """
    Detect faces using YOLOv8n-face model.
    Saves cropped face images and outputs detections in JSON format.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_crop_dir, exist_ok=True)

    # Use a pretrained YOLOv8 face detection model
    model = YOLO(MODEL_PATH)
    
    output_json = os.path.join(output_dir, f"{video_id}_detections.json")
    detections = []

    for frame_file in sorted(os.listdir(frames_dir)):
        frame_path = os.path.join(frames_dir, frame_file)
        img = cv2.imread(frame_path)
        if img is None:
            continue

        # Run detection
        results = model.predict(img, conf=0.5, verbose=False)

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0].item())

            # Crop the face
            face_crop = img[y1:y2, x1:x2]
            crop_name = f"face_{uuid.uuid4().hex[:8]}.jpg"
            crop_path = os.path.join(output_crop_dir, crop_name)
            cv2.imwrite(crop_path, face_crop)

            detections.append({
                "detection_id": crop_name.split('.')[0],
                "type": "face",
                "frame_path": frame_path,
                "bbox": [x1, y1, x2, y2],
                "confidence": conf,
                "crop_path": crop_path,
                "extra": {}
            })

    with open(output_json, "w") as f:
        json.dump({"detections": {"faces": detections}}, f, indent=2)

    print(f"✅ Saved YOLOv8n-face detections → {output_json}")
    print(f"🖼️ Cropped faces saved to → {output_crop_dir}")
