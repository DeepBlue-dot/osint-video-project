# video_pipeline/ocr_extractor.py

import os
import json
from tqdm import tqdm
import easyocr

def extract_text_from_frames(frames_dir, outputs_dir, video_id, languages=['en']):
    """
    Extract visible text from each frame using EasyOCR.
    Saves results as a JSON file.
    """
    reader = easyocr.Reader(languages, gpu=True)

    frames = sorted([f for f in os.listdir(frames_dir) if f.endswith(('.jpg', '.png'))])
    results_data = []

    print(f"🔠 Running OCR on {len(frames)} frames...")

    for frame_file in tqdm(frames):
        frame_path = os.path.join(frames_dir, frame_file)
        ocr_results = reader.readtext(frame_path, detail=1)

        frame_texts = []
        for (bbox, text, confidence) in ocr_results:
            # Convert numpy.int32 → float for JSON compatibility
            bbox_clean = [[float(x), float(y)] for x, y in bbox]

            frame_texts.append({
                "text": text,
                "confidence": round(float(confidence), 3),
                "bbox": bbox_clean
            })

        results_data.append({
            "frame": frame_file,
            "texts": frame_texts
        })

    os.makedirs(outputs_dir, exist_ok=True)
    output_path = os.path.join(outputs_dir, f"{video_id}_ocr.json")

    # ✅ Use safe JSON dump
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)

    print(f"✅ OCR extraction completed. Results saved to {output_path}")
    return output_path
