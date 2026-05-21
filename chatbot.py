"""
Multi-turn CLI chatbot: MiniMax LLM streaming + streaming TTS playback.

- LLM: MiniMax Anthropic-compatible endpoint (MiniMax-M2.7), streaming.
- TTS: MiniMax /v1/t2a_v2 with stream=True, PCM output.
- Playback: sounddevice RawOutputStream (32kHz mono int16).
"""

import json
import os
import queue
import sys
import threading

import anthropic
import requests
import sounddevice as sd
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ.get("MINIMAX_API_KEY")
LLM_BASE_URL = os.environ.get("ANTHROPIC_BASE_URL", "https://api.minimaxi.com/anthropic")
TTS_URL = "https://api.minimaxi.com/v1/t2a_v2"

LLM_MODEL = "MiniMax-M2.7"
TTS_MODEL = "speech-2.8-hd"
TTS_VOICE = "male-qn-qingse"
SAMPLE_RATE = 32000

SYSTEM_PROMPT = "You are a helpful assistant. 用中文回答用户。"
SENTENCE_ENDS = set("。！？!?；;\n")

client = anthropic.Anthropic(api_key=API_KEY, base_url=LLM_BASE_URL)


class TTSPlayer:
    """Sentence-level streaming TTS player. Single TTS worker preserves order."""

    _SENTINEL = None

    def __init__(self):
        self.sentence_queue: "queue.Queue[str | None]" = queue.Queue()
        self.audio_queue: "queue.Queue[bytes | None]" = queue.Queue()
        self._buffer = ""
        self._stream = sd.RawOutputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="int16"
        )
        self._tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
        self._play_thread = threading.Thread(target=self._play_worker, daemon=True)

    def start(self):
        self._stream.start()
        self._tts_thread.start()
        self._play_thread.start()

    def feed(self, text: str):
        """Accumulate streamed text; flush sentences whenever punctuation appears."""
        self._buffer += text
        while True:
            cut = -1
            for i, ch in enumerate(self._buffer):
                if ch in SENTENCE_ENDS:
                    cut = i
                    break
            if cut < 0:
                return
            sentence = self._buffer[: cut + 1].strip()
            self._buffer = self._buffer[cut + 1 :]
            if sentence:
                self.sentence_queue.put(sentence)

    def flush(self):
        tail = self._buffer.strip()
        self._buffer = ""
        if tail:
            self.sentence_queue.put(tail)

    def close(self):
        """Signal workers and wait until all queued audio has played."""
        self.sentence_queue.put(self._SENTINEL)
        self._tts_thread.join()
        self.audio_queue.put(self._SENTINEL)
        self._play_thread.join()
        try:
            self._stream.stop()
            self._stream.close()
        except Exception:
            pass

    def _tts_worker(self):
        while True:
            sentence = self.sentence_queue.get()
            if sentence is self._SENTINEL:
                return
            try:
                self._synthesize(sentence)
            except Exception as e:
                print(f"\n[TTS error] {e}", file=sys.stderr)

    def _synthesize(self, text: str):
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
                "sample_rate": SAMPLE_RATE,
                "bitrate": 128000,
                "format": "pcm",
                "channel": 1,
            },
            "subtitle_enable": False,
        }
        with requests.post(
            TTS_URL, headers=headers, json=payload, stream=True, timeout=60
        ) as resp:
            if resp.status_code != 200:
                print(
                    f"\n[TTS HTTP {resp.status_code}] {resp.text[:300]}",
                    file=sys.stderr,
                )
                return
            for raw in resp.iter_lines(decode_unicode=True):
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
                # MiniMax 流式末尾会再发一个 status==2 的事件，里面 audio 是完整拼接，
                # 跳过它，避免每句播两遍。
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
                    self.audio_queue.put(pcm)

    def _play_worker(self):
        while True:
            chunk = self.audio_queue.get()
            if chunk is self._SENTINEL:
                return
            try:
                self._stream.write(chunk)
            except Exception as e:
                print(f"\n[Audio error] {e}", file=sys.stderr)


def chat_once(messages: list, user_text: str):
    messages.append(
        {"role": "user", "content": [{"type": "text", "text": user_text}]}
    )

    player = TTSPlayer()
    player.start()

    assistant_text = ""
    in_response = False

    try:
        stream = client.messages.create(
            model=LLM_MODEL,
            max_tokens=2000,
            system=SYSTEM_PROMPT,
            messages=messages,
            stream=True,
        )

        print("\n" + "=" * 60)
        print("Assistant:")
        print("=" * 60)

        for chunk in stream:
            if chunk.type == "content_block_delta":
                delta = getattr(chunk, "delta", None)
                if not delta:
                    continue
                if delta.type == "thinking_delta":
                    thinking = delta.thinking or ""
                    if thinking:
                        # Thinking only printed; not sent to TTS.
                        print(thinking, end="", flush=True)
                elif delta.type == "text_delta":
                    text = delta.text or ""
                    if text:
                        if not in_response:
                            in_response = True
                            print("\n" + "-" * 60)
                            print("Response:")
                            print("-" * 60)
                        print(text, end="", flush=True)
                        assistant_text += text
                        player.feed(text)
        print()
    except Exception as e:
        print(f"\n[LLM error] {e}", file=sys.stderr)
    finally:
        player.flush()
        player.close()

    if assistant_text:
        messages.append(
            {"role": "assistant", "content": [{"type": "text", "text": assistant_text}]}
        )
    else:
        # 没拿到回复就回滚 user 消息，避免历史里悬空一条 user
        messages.pop()


def main():
    if not API_KEY:
        print("MINIMAX_API_KEY 未设置（检查 .env）", file=sys.stderr)
        sys.exit(1)

    print("多轮对话已启动。输入 /exit 退出，空行忽略。\n")
    messages: list = []

    while True:
        try:
            user_text = input("你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            return
        if not user_text:
            continue
        if user_text.lower() in ("/exit", "/quit", "exit", "quit"):
            print("再见。")
            return
        try:
            chat_once(messages, user_text)
        except KeyboardInterrupt:
            print("\n[已中断本轮]")
            continue


if __name__ == "__main__":
    main()
