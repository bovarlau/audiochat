"""Async TTS streaming: MiniMax /v1/t2a_v2 SSE -> PCM bytes generator."""

import json
import sys
from typing import AsyncIterator

import httpx

from config import API_KEY, SAMPLE_RATE_TTS, TTS_MODEL, TTS_URL, TTS_VOICE


async def synthesize(text: str) -> AsyncIterator[bytes]:
    """
    Stream PCM (int16 mono @ SAMPLE_RATE_TTS) chunks for the given text.

    Mirrors chatbot.py's parsing exactly, including the status==2 dedup.
    """
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    payload = {
        "model": TTS_MODEL,
        "text": text,
        "stream": True,
        "voice_setting": {
            "voice_id": TTS_VOICE,
            "speed": 1,
            "vol": 1,
            "pitch": 0,
            "emotion": "happy",
        },
        "audio_setting": {
            "sample_rate": SAMPLE_RATE_TTS,
            "bitrate": 128000,
            "format": "pcm",
            "channel": 1,
        },
        "subtitle_enable": False,
    }

    timeout = httpx.Timeout(60.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream(
            "POST", TTS_URL, headers=headers, json=payload
        ) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                snippet = body.decode("utf-8", errors="replace")[:300]
                print(f"[TTS HTTP {resp.status_code}] {snippet}", file=sys.stderr)
                return
            async for raw in resp.aiter_lines():
                if not raw:
                    continue
                line = raw.lstrip()
                if not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if not data_str or data_str == "[DONE]":
                    continue
                try:
                    obj = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                data_obj = obj.get("data") or {}
                # MiniMax sends a final status==2 event whose audio is the full
                # concatenated payload — skip to avoid playing each sentence twice.
                if data_obj.get("status") == 2:
                    continue
                audio_hex = data_obj.get("audio") or ""
                if not audio_hex:
                    continue
                try:
                    pcm = bytes.fromhex(audio_hex)
                except ValueError:
                    continue
                if pcm:
                    yield pcm
