"""FunASR wrapper: paraformer-zh + ct-punc, offline ASR on full utterance.

Accepts either raw int16 PCM @ 16 kHz, or a compressed audio container
(WebM/Opus from MediaRecorder, WAV, etc.) — auto-decoded via PyAV.
"""

import asyncio
import io
import os
import sys
import wave
from pathlib import Path

import av
import numpy as np

from config import MODELS_DIR, SAMPLE_RATE_ASR

os.environ.setdefault("MODELSCOPE_CACHE", MODELS_DIR)
os.environ.setdefault("HF_HOME", MODELS_DIR)

_model = None
_punc_model = None

# Set env DEBUG_ASR=1 to write the last received utterance to ./debug_asr.wav
# (after decode + resample to 16k mono).
_DEBUG_DUMP = os.environ.get("DEBUG_ASR") == "1"
_DEBUG_PATH = Path(__file__).parent / "debug_asr.wav"


def _dump_wav(pcm_int16: bytes):
    try:
        with wave.open(str(_DEBUG_PATH), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE_ASR)
            w.writeframes(pcm_int16)
        print(f"[ASR] dumped {len(pcm_int16)} bytes -> {_DEBUG_PATH}", file=sys.stderr)
    except Exception as e:
        print(f"[ASR] dump failed: {e}", file=sys.stderr)


def load_model():
    """Eagerly load FunASR model + run warmup. Call once at server startup."""
    global _model, _punc_model
    if _model is not None:
        return _model

    os.makedirs(MODELS_DIR, exist_ok=True)
    from funasr import AutoModel

    print(f"[ASR] loading paraformer-zh into {MODELS_DIR}...", file=sys.stderr)
    _model = AutoModel(
        model="paraformer-zh",
        cache_dir=MODELS_DIR,
        disable_update=True,
        device="cpu",
        ncpu=4,
    )
    print("[ASR] loading ct-punc (separate AutoModel)...", file=sys.stderr)
    # Load ct-punc as its own AutoModel so we can call .generate(input=text)
    # on it directly. (When passed as `punc_model=` to the ASR model, FunASR
    # 1.3.1 loads it but does NOT auto-invoke it during generate.)
    _punc_model = AutoModel(
        model="ct-punc",
        cache_dir=MODELS_DIR,
        disable_update=True,
        device="cpu",
        ncpu=4,
    )
    print("[ASR] warmup inference (1s silence + 1 punc call)...", file=sys.stderr)
    silent = np.zeros(SAMPLE_RATE_ASR, dtype=np.float32)
    _model.generate(input=silent, batch_size_s=300)
    try:
        _punc_model.generate(input="你 好 世 界")
    except Exception as e:
        print(f"[ASR] punc warmup failed: {e}", file=sys.stderr)
    print("[ASR] ready.", file=sys.stderr)
    return _model


def _apply_punc(text: str) -> str:
    """Run ct-punc over the raw ASR text. Falls back to bare-text-with-spaces-
    stripped if punc fails."""
    if not text or _punc_model is None:
        return text.replace(" ", "")
    try:
        res = _punc_model.generate(input=text)
        if res and isinstance(res[0], dict):
            punctuated = res[0].get("text", "")
            if punctuated:
                return punctuated
    except Exception as e:
        print(f"[ASR] punc apply failed: {e}", file=sys.stderr)
    # Fall back: at least strip the inter-char spaces.
    return text.replace(" ", "")


def _decode_with_pyav(data: bytes) -> np.ndarray:
    """Decode arbitrary audio container (WebM/Opus, WAV, MP3, ...) into a
    mono float32 numpy array resampled to SAMPLE_RATE_ASR."""
    buf = io.BytesIO(data)
    out_chunks = []
    with av.open(buf) as container:
        astream = next((s for s in container.streams if s.type == "audio"), None)
        if astream is None:
            return np.zeros(0, dtype=np.float32)
        resampler = av.AudioResampler(
            format="s16", layout="mono", rate=SAMPLE_RATE_ASR
        )
        for frame in container.decode(astream):
            for out_frame in resampler.resample(frame):
                arr = out_frame.to_ndarray()  # shape (1, N) int16 for mono s16
                out_chunks.append(arr.reshape(-1))
        # Flush resampler tail
        for out_frame in resampler.resample(None):
            arr = out_frame.to_ndarray()
            out_chunks.append(arr.reshape(-1))
    if not out_chunks:
        return np.zeros(0, dtype=np.float32)
    pcm16 = np.concatenate(out_chunks).astype(np.int16)
    return pcm16.astype(np.float32) / 32768.0


def _is_raw_pcm(data: bytes) -> bool:
    """Heuristic: WebM begins with EBML magic 0x1A45DFA3; WAV with 'RIFF';
    Opus container, Ogg with 'OggS'. Raw PCM has none of these."""
    if len(data) < 4:
        return True
    head = data[:4]
    if head == b"RIFF" or head == b"OggS":
        return False
    if data[:4] == b"\x1A\x45\xDF\xA3":
        return False
    return True


def _transcribe_sync(audio: bytes) -> str:
    if _model is None:
        raise RuntimeError("FunASR model not loaded; call load_model() at startup")
    if not audio:
        return ""

    if _is_raw_pcm(audio):
        arr = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
    else:
        try:
            arr = _decode_with_pyav(audio)
        except Exception as e:
            print(f"[ASR] decode failed: {e}", file=sys.stderr)
            return ""

    if arr.size == 0:
        return ""

    if _DEBUG_DUMP:
        pcm16 = (np.clip(arr, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
        _dump_wav(pcm16)

    duration_s = arr.size / SAMPLE_RATE_ASR
    peak = float(np.max(np.abs(arr)))
    rms = float(np.sqrt(np.mean(arr * arr)))
    print(
        f"[ASR] input: {arr.size} samples ({duration_s:.2f}s) peak={peak:.3f} rms={rms:.4f}",
        file=sys.stderr,
    )
    result = _model.generate(input=arr, batch_size_s=300)
    if not result:
        return ""
    raw_text = result[0].get("text", "") if isinstance(result[0], dict) else ""
    raw_text = raw_text.strip()
    if not raw_text:
        return ""
    return _apply_punc(raw_text).strip()


async def transcribe(audio: bytes) -> str:
    """Transcribe a complete utterance.

    Accepts either raw int16 PCM mono @ 16 kHz, or a compressed audio
    container (WebM/Opus from MediaRecorder, WAV, MP3, etc.).
    """
    return await asyncio.to_thread(_transcribe_sync, audio)
