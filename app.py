import io
import re
import json
import os
import asyncio
import unicodedata
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import login

hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)

SUNBIRD_MODEL = "Sunbird/asr-whisper-large-v3-salt"  # Luganda
WHISPER_EN    = "nyrahealth/CrisperWhisper"          # English (verbatim — keeps fillers)
LID_MODEL     = "facebook/mms-lid-256"
SAMPLE_RATE   = 16000
MIN_DURATION  = 1.0
PAD_SECONDS   = 0.3
LID_THRESHOLD = 0.90

SALT_LANGUAGE_TOKENS_WHISPER = {
    "eng": 50259,
    "lug": 50355,
}

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
    AutoModelForSpeechSeq2Seq, AutoProcessor,
    Wav2Vec2ForSequenceClassification, AutoFeatureExtractor,
    pipeline as hf_pipeline,
)

device_asr  = "cuda" if torch.cuda.is_available() else "cpu"
device_lid  = "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
print(f"ASR device: {device_asr} ({torch_dtype}) | LID device: {device_lid}")

print(f"Loading Sunbird ASR model ({SUNBIRD_MODEL})...")
sunbird_processor = AutoProcessor.from_pretrained(SUNBIRD_MODEL)
sunbird_model     = AutoModelForSpeechSeq2Seq.from_pretrained(
    SUNBIRD_MODEL,
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True,
    use_safetensors=True,
).eval().to(device_asr)

SUNBIRD_LANG_STRINGS = {
    code: sunbird_processor.tokenizer.decode(token_id)
    for code, token_id in SALT_LANGUAGE_TOKENS_WHISPER.items()
}

print(f"Loading CrisperWhisper ({WHISPER_EN})...")
whisper_processor = AutoProcessor.from_pretrained(WHISPER_EN)
whisper_model     = AutoModelForSpeechSeq2Seq.from_pretrained(
    WHISPER_EN,
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True,
    use_safetensors=True,
).eval().to(device_asr)
whisper_pipe = hf_pipeline(
    "automatic-speech-recognition",
    model=whisper_model,
    tokenizer=whisper_processor.tokenizer,
    feature_extractor=whisper_processor.feature_extractor,
    chunk_length_s=30,
    return_timestamps="word",
    torch_dtype=torch_dtype,
    device=device_asr,
)

print(f"Loading Language ID model ({LID_MODEL})...")
lid_extractor = AutoFeatureExtractor.from_pretrained(LID_MODEL)
lid_model     = Wav2Vec2ForSequenceClassification.from_pretrained(LID_MODEL).eval().to(device_lid)
label2id      = {v: k for k, v in lid_model.config.id2label.items()}
lug_id        = label2id.get("lug")
eng_id        = label2id.get("eng")

print(f"All models ready — ASR on {device_asr}, LID on {device_lid}.")


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
    ).to(device_lid)
    with torch.no_grad():
        logits = lid_model(**inputs).logits
    probs    = F.softmax(logits, dim=-1)[0]
    lug_prob = probs[lug_id].item() if lug_id is not None else 0.0
    eng_prob = probs[eng_id].item() if eng_id is not None else 0.0
    del inputs, logits, probs
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
    inputs = sunbird_processor(
        array, sampling_rate=SAMPLE_RATE, return_tensors="pt"
    )
    input_features = inputs["input_features"].to(device_asr, dtype=torch_dtype)
    with torch.no_grad():
        predicted_ids = sunbird_model.generate(
            input_features,
            language=SUNBIRD_LANG_STRINGS["lug"],
            task="transcribe",
            no_repeat_ngram_size=3,
            forced_decoder_ids=None,
        )
    text = sunbird_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
    del inputs, input_features, predicted_ids
    return text


def transcribe_english(array):
    from utils import adjust_pauses_for_hf_pipeline_output
    hf_output = whisper_pipe({"array": array, "sampling_rate": SAMPLE_RATE})
    result    = adjust_pauses_for_hf_pipeline_output(hf_output)
    return result["text"].strip()


@app.get("/health")
def health():
    return {
        "status":     "ok",
        "asr_device": device_asr,
        "lid_device": device_lid,
    }


# Serialize ASR work: only one transcribe pipeline runs the GPU at a time,
# but the event loop stays free so /health and other endpoints stay responsive
# even while a long transcribe is in flight.
asr_lock = asyncio.Lock()


def _run_transcribe_pipeline(audio_bytes, segs):
    full_audio, _ = librosa.load(io.BytesIO(audio_bytes), sr=SAMPLE_RATE, mono=True)
    total_duration = len(full_audio) / SAMPLE_RATE

    # Pass 1: extract arrays and run language ID for every usable segment.
    prep = []
    for seg in segs:
        start    = seg["start"]
        end      = seg["end"]
        duration = end - start

        if duration < MIN_DURATION:
            prep.append({"seg": seg, "skipped": True, "array": None, "detected": None})
            continue

        start_p = max(0.0, start - PAD_SECONDS)
        end_p   = min(total_duration, end + PAD_SECONDS)
        array   = full_audio[int(start_p * SAMPLE_RATE):int(end_p * SAMPLE_RATE)]

        if len(array) < int(SAMPLE_RATE * MIN_DURATION):
            prep.append({"seg": seg, "skipped": True, "array": None, "detected": None})
            continue

        try:
            detected = detect_language(array)
        except Exception as e:
            print(f"  [LID ERROR] {seg['speaker_id']} @ {start:.2f}s — {e}")
            detected = None

        prep.append({"seg": seg, "skipped": False, "array": array, "detected": detected})

    # Decide the dominant language from segments LID was confident about.
    lug_count = sum(1 for p in prep if p["detected"] == "lug")
    eng_count = sum(1 for p in prep if p["detected"] == "eng")
    dominant  = "eng" if eng_count > lug_count else "lug"
    print(f"  [LID summary] lug={lug_count}, eng={eng_count}, uncertain/unknown={sum(1 for p in prep if not p['skipped'] and p['detected'] not in ('lug', 'eng'))} → dominant={dominant}")

    # Pass 2: transcribe each segment using its detected language, or the dominant
    # language as the fallback for uncertain ("lug-eng") and LID-failed segments.
    results = []
    for p in prep:
        seg   = p["seg"]
        start = seg["start"]
        end   = seg["end"]

        if p["skipped"]:
            results.append({
                "speaker":  seg["speaker_id"],
                "start":    start,
                "end":      end,
                "language": "",
                "text":     "",
                "skipped":  True,
            })
            continue

        detected = p["detected"]
        resolved = detected if detected in ("lug", "eng") else dominant
        array    = p["array"]

        try:
            if resolved == "eng":
                text = transcribe_english(array)
            else:
                text = transcribe_luganda(array)
        except Exception as e:
            print(f"  [ERROR] {seg['speaker_id']} @ {start:.2f}s — {e}")
            resolved = ""
            text     = ""

        results.append({
            "speaker":  seg["speaker_id"],
            "start":    start,
            "end":      end,
            "language": resolved,
            "text":     text,
            "skipped":  False,
        })

        if device_asr == "cuda":
            torch.cuda.empty_cache()

    speaker_segments = [r for r in results if not r.get("skipped")]

    return {
        "transcript_text":  " ".join([s["text"] for s in speaker_segments]),
        "speaker_segments": speaker_segments,
    }


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

    async with asr_lock:
        return await asyncio.to_thread(_run_transcribe_pipeline, audio_bytes, segs)