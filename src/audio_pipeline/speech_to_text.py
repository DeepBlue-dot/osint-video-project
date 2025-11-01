import os
import json
import whisper

def transcribe_audio(audio_path, output_json):
    """
    Transcribe speech in the audio file using OpenAI Whisper.
    """
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    model = whisper.load_model("base")  
    result = model.transcribe(audio_path)

    transcript_data = []
    for segment in result["segments"]:
        transcript_data.append({
            "timestamp": segment["start"],
            "text": segment["text"].strip(),
            "confidence": segment.get("avg_logprob", None)
        })

    with open(output_json, "w") as f:
        json.dump({"transcript": transcript_data}, f, indent=2)

    print(f"🗣️ Transcript saved → {output_json}")
