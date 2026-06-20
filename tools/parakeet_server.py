#!/usr/bin/env python3
"""
Parakeet TDT 0.6B v3 transcription server for Adapsis.

Drop-in replacement for the Whisper server that TelegramBot.ax already
talks to. Honors the existing Adapsis contract:

    POST /transcribe   multipart field "file"  ->  {"text": "..."}

Also exposes an OpenAI-compatible alias:

    POST /v1/audio/transcriptions  field "file" ->  {"text": "..."}

Parakeet v3 auto-detects language across 25 EU languages (incl. de/en), so no
language hint is needed. Telegram sends OGG/Opus; we transcode to 16 kHz mono
WAV with ffmpeg before inference.

Run:
    pip install onnx-asr[gpu] fastapi uvicorn python-multipart
    # ffmpeg must be on PATH (apt install ffmpeg)
    PARAKEET_HOST=127.0.0.1 PARAKEET_PORT=8090 python parakeet_server.py

Footprint: ONNX INT8 ~ <1 GB. Fits a 4 GB GPU with room to spare, or runs
CPU-only. For maximum accuracy on a bigger box, swap the loader for NeMo
(see NEMO note below).
"""

import os
import shutil
import subprocess
import tempfile

import onnx_asr
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

HOST = os.environ.get("PARAKEET_HOST", "127.0.0.1")
PORT = int(os.environ.get("PARAKEET_PORT", "8090"))
MODEL_ID = os.environ.get("PARAKEET_MODEL", "nemo-parakeet-tdt-0.6b-v3")
# quantization: "int8" (smallest, ~700MB) or "fp32"/"fp16"
QUANT = os.environ.get("PARAKEET_QUANT", "int8")
# VAD: Parakeet caps a single forward pass at ~20-30s. For longer audio we run
# Silero VAD to split into speech segments and transcribe each, then join.
VAD_ENABLED = os.environ.get("PARAKEET_VAD", "1") not in ("0", "false", "no")
VAD_NAME = os.environ.get("PARAKEET_VAD_MODEL", "silero")
# Clips at/under this many seconds skip VAD (single fast pass).
VAD_THRESHOLD_SECS = float(os.environ.get("PARAKEET_VAD_THRESHOLD_SECS", "20"))

app = FastAPI(title="parakeet-adapsis")

# Loaded once at startup. onnx-asr pulls the ONNX export from HF on first run.
print(f"[parakeet] loading {MODEL_ID} ({QUANT}) ...")
model = onnx_asr.load_model(MODEL_ID, quantization=QUANT)

# Load VAD once and build a VAD-wrapped view of the model. Kept separate from
# the plain model so short clips still use the cheaper single-pass path.
vad_model = None
if VAD_ENABLED:
    try:
        print(f"[parakeet] loading VAD ({VAD_NAME}) ...")
        _vad = onnx_asr.load_vad(VAD_NAME)
        vad_model = model.with_vad(_vad)
        print("[parakeet] VAD ready")
    except Exception as e:  # noqa: BLE001 - degrade gracefully, don't crash ASR
        print(f"[parakeet] VAD unavailable ({e}); long audio may truncate")
        vad_model = None
print(f"[parakeet] ready on {HOST}:{PORT}")


def _to_wav16k_mono(src_path: str) -> str:
    """Transcode any input (OGG/Opus, mp3, m4a, ...) to 16kHz mono WAV."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH")
    dst_path = src_path + ".16k.wav"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-i", src_path,
         "-ac", "1", "-ar", "16000", "-f", "wav", dst_path],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return dst_path


def _wav_duration_secs(wav_path: str) -> float:
    """Duration of a WAV via its RIFF header (no extra deps). 0.0 if unknown."""
    try:
        import wave
        with wave.open(wav_path, "rb") as w:
            frames = w.getnframes()
            rate = w.getframerate() or 16000
            return frames / float(rate)
    except Exception:  # noqa: BLE001
        return 0.0


def _recognize(wav: str) -> str:
    """Route by length: short clips single-pass, long clips via VAD segments.

    Parakeet v3 auto-detects language; no hint passed. With VAD, recognize()
    yields one SegmentResult per detected speech span — we join their text.
    """
    dur = _wav_duration_secs(wav)
    use_vad = vad_model is not None and (dur == 0.0 or dur > VAD_THRESHOLD_SECS)
    if use_vad:
        parts = []
        for res in vad_model.recognize(wav):
            seg = getattr(res, "text", None)
            seg = seg if seg is not None else str(res)
            seg = seg.strip()
            if seg:
                parts.append(seg)
        return " ".join(parts).strip()
    return (model.recognize(wav) or "").strip()


async def _transcribe(file: UploadFile) -> dict:
    suffix = os.path.splitext(file.filename or "audio")[1] or ".ogg"
    raw = None
    wav = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(await file.read())
            raw = tmp.name
        wav = _to_wav16k_mono(raw)
        return {"text": _recognize(wav)}
    finally:
        for p in (raw, wav):
            if p and os.path.exists(p):
                os.remove(p)


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    """Contract used by TelegramBot.ax -> http_upload(..., "file", "")."""
    try:
        return JSONResponse(await _transcribe(file))
    except Exception as e:  # keep the Adapsis side simple: it checks for "text"
        return JSONResponse({"text": "", "error": str(e)}, status_code=500)


@app.post("/v1/audio/transcriptions")
async def openai_transcriptions(file: UploadFile = File(...)):
    """OpenAI-compatible alias, same response shape ({"text": ...})."""
    try:
        return JSONResponse(await _transcribe(file))
    except Exception as e:
        return JSONResponse({"text": "", "error": str(e)}, status_code=500)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_ID,
        "quant": QUANT,
        "vad": vad_model is not None,
        "vad_threshold_secs": VAD_THRESHOLD_SECS,
    }


# --- NEMO note -------------------------------------------------------------
# For top-of-leaderboard accuracy on a machine with more VRAM, replace the
# loader + recognize() with NeMo:
#
#   import nemo.collections.asr as nemo_asr
#   model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v3")
#   text = model.transcribe([wav])[0].text
#
# FP16 footprint ~1.2 GB. The HTTP contract above stays identical.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
