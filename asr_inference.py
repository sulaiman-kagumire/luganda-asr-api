import time
import argparse
import requests
import json
from pathlib import Path

ENDPOINT   = "http://100.104.98.125:8011/transcribe"

HERE = Path(__file__).parent

# Defaults used when --audio / --segments aren't provided (handy for quick smoke tests).
DEFAULT_AUDIO = HERE / "0001__academic-pressure_depression_phone.wav"
DEFAULT_SEGMENTS = {
    "analysis_type": "speaker_diarization",
    "results": {
        "speaker_segments": [
            {"speaker": "SPEAKER_00", "start": 1.2,    "end": 2.87},
            {"speaker": "SPEAKER_01", "start": 2.956,  "end": 7.594},
            {"speaker": "SPEAKER_00", "start": 9.006,  "end": 12.864},
            {"speaker": "SPEAKER_01", "start": 12.986, "end": 15.427},
            {"speaker": "SPEAKER_00", "start": 15.99,  "end": 20.031},
        ]
    }
}

parser = argparse.ArgumentParser(description="Call the Luganda ASR API and print/save transcripts.")
parser.add_argument("--audio",    help=f"Path to the audio file to transcribe (default: {DEFAULT_AUDIO.name}).")
parser.add_argument("--segments", help="Path to a speaker-diarization JSON file (e.g. the output from diarization API)")
parser.add_argument("--output",   help="Save the raw JSON response to this path (ready to feed into downstream models).")
args = parser.parse_args()

AUDIO_FILE = Path(args.audio) if args.audio else DEFAULT_AUDIO

if args.segments:
    with open(args.segments, "r") as f:
        segments = json.load(f)
else:
    segments = DEFAULT_SEGMENTS

# Pull out the segment list regardless of whether the JSON is a flat list or
# wrapped under {"results": {"speaker_segments": [...]}} (server accepts both).
if isinstance(segments, list):
    seg_list = segments
else:
    seg_list = segments.get("results", {}).get("speaker_segments", [])

speakers = sorted({s.get("speaker") or s.get("speaker_id") for s in seg_list if s.get("speaker") or s.get("speaker_id")})
print(f"Audio    : {AUDIO_FILE.name}")
print(f"Segments : {len(seg_list)}  (speakers: {', '.join(speakers)})")
print(f"Endpoint : {ENDPOINT}")
print()

print("Sending request...")
t0 = time.time()

with open(AUDIO_FILE, "rb") as f:
    response = requests.post(
        ENDPOINT,
        files={"audio_file": f},
        data={"segments": json.dumps(segments)},
        timeout=600,
    )

elapsed = time.time() - t0
print(f"Got response in {elapsed:.1f}s")
print(f"HTTP status: {response.status_code}")
if response.status_code != 200:
    print(response.text)
    raise SystemExit(1)

result = response.json()

if args.output:
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved JSON response to {args.output}")

print(f"\nTranscript:\n{result.get('transcript_text', '<missing>')}")

print(f"\nSpeaker segments returned ({len(result.get('speaker_segments', []))}):")
for seg in result.get("speaker_segments", []):
    lang = seg.get("language", "N/A")
    print(f"  [{seg['speaker']} | {seg['start']:.2f}→{seg['end']:.2f}s | lang: {lang}]")
    print(f"    {seg['text']}")
