import io
import re
import json
import os
import unicodedata
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import login

hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)

XLSR_MODEL    = "sulaimank/wav2vec2-xlsr-luganda"
WHISPER_EN    = "openai/whisper-small.en"
LID_MODEL     = "facebook/mms-lid-256"
SAMPLE_RATE   = 16000
MIN_DURATION  = 1.0
PAD_SECONDS   = 0.3
LID_THRESHOLD = 0.90

app = FastAPI(title="Luganda ASR API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

import torch
import librosa
from transformers import (
    Wav2Vec2ForCTC, Wav2Vec2ProcessorWithLM,
    WhisperProcessor, WhisperForConditionalGeneration,
    Wav2Vec2ForSequenceClassification, AutoFeatureExtractor,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

print(f"Loading Luganda ASR model ({XLSR_MODEL})...")
xlsr_processor = Wav2Vec2ProcessorWithLM.from_pretrained(XLSR_MODEL)
xlsr_model     = Wav2Vec2ForCTC.from_pretrained(XLSR_MODEL).eval().to(device)

print(f"Loading English Whisper model ({WHISPER_EN})...")
whisper_processor = WhisperProcessor.from_pretrained(WHISPER_EN)
whisper_model     = WhisperForConditionalGeneration.from_pretrained(WHISPER_EN).eval().to(device)

print(f"Loading Language ID model ({LID_MODEL})...")
lid_extractor = AutoFeatureExtractor.from_pretrained(LID_MODEL)
lid_model     = Wav2Vec2ForSequenceClassification.from_pretrained(LID_MODEL).eval().to(device)
label2id      = {v: k for k, v in lid_model.config.id2label.items()}
lug_id        = label2id.get("lug")
eng_id        = label2id.get("eng")

print(f"All models ready on {device}.")


def normalise(text):
    if not text: return ""
    text = text.lower()
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"['\u02bb\u02bc\u02bd\u2018\u2019]", "'", text)
    text = re.sub(r'\d', '', text)
    text = re.sub(r"[^a-zŋ\s']", '', text)
    return " ".join(text.split())


def detect_language(array):
    import torch.nn.functional as F
    inputs = lid_extractor(
        array, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True
    ).to(device)
    with torch.no_grad():
        logits = lid_model(**inputs).logits
    probs    = F.softmax(logits, dim=-1)[0]
    lug_prob = probs[lug_id].item() if lug_id is not None else 0.0
    eng_prob = probs[eng_id].item() if eng_id is not None else 0.0
    total    = lug_prob + eng_prob
    if total == 0:
        return "lug-eng"
    lug_norm = lug_prob / total
    eng_norm = eng_prob / total
    if lug_norm >= LID_THRESHOLD:
        return "lug"
    elif eng_norm >= LID_THRESHOLD:
        return "eng"
    else:
        return "lug-eng"


def transcribe_luganda(array):
    inputs = xlsr_processor(
        array, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True
    ).to(device)
    with torch.no_grad():
        logits = xlsr_model(**inputs).logits
    return xlsr_processor.batch_decode(logits.cpu().numpy()).text[0].strip()


def transcribe_english(array):
    inputs = whisper_processor(
        array, sampling_rate=SAMPLE_RATE, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        predicted_ids = whisper_model.generate(inputs["input_features"])
    return whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()


@app.get("/health")
def health():
    return {"status": "ok", "device": device}


@app.post("/transcribe")
async def transcribe(
    audio_file: UploadFile = File(...),
    segments:   str        = Form(...),
):
    audio_bytes = await audio_file.read()
    data        = json.loads(segments)

    if isinstance(data, list):
        raw_segs = data
    else:
        raw_segs = data.get("results", {}).get("speaker_segments", [])

    segs = []
    for s in raw_segs:
        segs.append({
            "speaker_id": s.get("speaker_id") or s.get("speaker"),
            "start":      s["start"],
            "end":        s["end"],
            "gender":     s.get("gender", ""),
        })

    full_audio, _ = librosa.load(io.BytesIO(audio_bytes), sr=SAMPLE_RATE, mono=True)
    total_duration = len(full_audio) / SAMPLE_RATE
    results = []

    for seg in segs:
        start    = seg["start"]
        end      = seg["end"]
        duration = end - start

        if duration < MIN_DURATION:
            results.append({
                "speaker":  seg["speaker_id"],
                "start":    start,
                "end":      end,
                "language": "",
                "text":     "",
                "skipped":  True,
            })
            continue

        start_p = max(0.0, start - PAD_SECONDS)
        end_p   = min(total_duration, end + PAD_SECONDS)
        array   = full_audio[int(start_p * SAMPLE_RATE):int(end_p * SAMPLE_RATE)]

        try:
            lang = detect_language(array)

            if lang == "eng":
                text = transcribe_english(array)
            elif lang == "lug":
                text = transcribe_luganda(array)
            else:
                text_lug = transcribe_luganda(array)
                text_eng = transcribe_english(array)
                text = text_lug if len(text_lug) >= len(text_eng) else text_eng

        except Exception as e:
            print(f"  [ERROR] {seg['speaker_id']} @ {start:.2f}s — {e}")
            lang = ""
            text = ""

        results.append({
            "speaker":  seg["speaker_id"],
            "start":    start,
            "end":      end,
            "language": lang,
            "text":     text,
            "skipped":  False,
        })

        torch.cuda.empty_cache()

    speaker_segments = [r for r in results if not r.get("skipped")]

    return {
        "transcript_text":  " ".join([s["text"] for s in speaker_segments]),
        "speaker_segments": speaker_segments,
    }