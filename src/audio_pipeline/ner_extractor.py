# audio_pipeline/ner_extractor.py

import json
import os
import spacy

def extract_entities(transcript_path, outputs_dir, video_id, lang_model="en_core_web_sm"):
    """
    Extract named entities (people, orgs, locations, etc.) from a transcript.
    Saves results in a structured JSON file.
    """
    print(f"🧠 Loading language model: {lang_model}")
    nlp = spacy.load(lang_model)

    # Read the transcript
    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript_text = f.read().strip()

    print(f"🔍 Analyzing transcript for named entities...")
    doc = nlp(transcript_text)

    # Collect entities
    entities = []
    for ent in doc.ents:
        entities.append({
            "text": ent.text,
            "label": ent.label_,     # PERSON, GPE, ORG, DATE, etc.
            "start_char": ent.start_char,
            "end_char": ent.end_char
        })

    # Group by entity type
    grouped_entities = {}
    for e in entities:
        grouped_entities.setdefault(e["label"], []).append(e["text"])

    # Deduplicate
    for label in grouped_entities:
        grouped_entities[label] = list(set(grouped_entities[label]))

    # Save to JSON
    os.makedirs(outputs_dir, exist_ok=True)
    output_path = os.path.join(outputs_dir, f"{video_id}_entities.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "video_id": video_id,
            "entities": grouped_entities
        }, f, indent=2, ensure_ascii=False)

    print(f"✅ Entity extraction completed. Results saved to {output_path}")
    return output_path
