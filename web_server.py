"""FastAPI + WebSocket server for browser-based voice chat.

Run: uvicorn web_server:app --host 127.0.0.1 --port 8000
"""

import asyncio
import json
import struct
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import web_asr
from config import API_KEY, SAMPLE_RATE_TTS, SENTENCE_ENDS
from web_llm import stream_reply
from web_tts import synthesize

HERE = Path(__file__).parent
STATIC_DIR = HERE / "static"

# Binary frame header layout (little-endian):
#   bytes 0-1: magic 0xA1 0xB2
#   byte 2:    version (0x01)
#   bytes 3-6: turn_id (uint32 LE)
#   bytes 7-10: seq (uint32 LE)
#   byte 11:   flags (bit0 = final frame for this seq)
#   bytes 12-15: reserved (0)
#   bytes 16+: PCM int16 LE mono @ SAMPLE_RATE_TTS
_HEADER_STRUCT = struct.Struct("<BBBIIB4s")
_MAGIC0 = 0xA1
_MAGIC1 = 0xB2
_VERSION = 0x01


def make_audio_frame(turn_id: int, seq: int, flags: int, pcm: bytes) -> bytes:
    return _HEADER_STRUCT.pack(_MAGIC0, _MAGIC1, _VERSION, turn_id, seq, flags, b"\x00\x00\x00\x00") + pcm


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not API_KEY:
        print("MINIMAX_API_KEY 未设置（检查 .env）", file=sys.stderr)
    # Load FunASR (downloads models on first run; large).
    await asyncio.to_thread(web_asr.load_model)
    yield


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


class ConnCtx:
    """Per-WebSocket connection state."""

    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.messages: list = []
        self.turn_id: int = 0
        self.current_task: asyncio.Task | None = None
        self.audio_buf = bytearray()
        self.outbound: asyncio.Queue = asyncio.Queue(maxsize=200)
        self.sender_task: asyncio.Task | None = None
        self.closed = False

    async def send_json(self, payload: dict):
        await self.outbound.put(("json", payload))

    async def send_bin(self, frame: bytes):
        await self.outbound.put(("bin", frame))

    async def _sender_loop(self):
        try:
            while True:
                kind, data = await self.outbound.get()
                if kind == "json":
                    await self.ws.send_text(json.dumps(data, ensure_ascii=False))
                else:
                    await self.ws.send_bytes(data)
        except (WebSocketDisconnect, RuntimeError, asyncio.CancelledError):
            return

    async def cancel_current(self):
        task = self.current_task
        self.current_task = None
        if task and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass


async def _handle_turn(ctx: ConnCtx, user_text: str):
    """Run one user turn: feed LLM, pipeline sentences to TTS, stream PCM out."""
    turn_id = ctx.turn_id
    ctx.messages.append({"role": "user", "content": [{"type": "text", "text": user_text}]})

    state = {"buffer": "", "seq": 0, "assistant_text": ""}
    sentence_queue: asyncio.Queue = asyncio.Queue()

    async def llm_producer():
        async def on_text(text: str):
            state["assistant_text"] += text
            state["buffer"] += text
            await ctx.send_json({"type": "llm_delta", "text": text, "turn_id": turn_id})
            while True:
                buf = state["buffer"]
                cut = -1
                for i, ch in enumerate(buf):
                    if ch in SENTENCE_ENDS:
                        cut = i
                        break
                if cut < 0:
                    return
                sentence = buf[: cut + 1].strip()
                state["buffer"] = buf[cut + 1 :]
                if sentence:
                    state["seq"] += 1
                    await sentence_queue.put((state["seq"], sentence))

        async def on_thinking(text: str):
            await ctx.send_json(
                {"type": "thinking_delta", "text": text, "turn_id": turn_id}
            )

        try:
            await stream_reply(ctx.messages, on_text, on_thinking)
        finally:
            tail = state["buffer"].strip()
            state["buffer"] = ""
            if tail:
                state["seq"] += 1
                await sentence_queue.put((state["seq"], tail))
            await sentence_queue.put(None)  # sentinel
            await ctx.send_json({"type": "llm_done", "turn_id": turn_id})

    async def tts_consumer():
        while True:
            item = await sentence_queue.get()
            if item is None:
                return
            seq_num, sentence = item
            try:
                async for pcm in synthesize(sentence):
                    await ctx.send_bin(make_audio_frame(turn_id, seq_num, 0, pcm))
                # mark sentence final
                await ctx.send_bin(make_audio_frame(turn_id, seq_num, 1, b""))
                await ctx.send_json(
                    {"type": "tts_sentence_done", "seq": seq_num, "turn_id": turn_id}
                )
            except asyncio.CancelledError:
                raise
            except Exception as e:
                await ctx.send_json(
                    {"type": "error", "where": "tts", "message": str(e)}
                )

    try:
        try:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(llm_producer())
                tg.create_task(tts_consumer())
        except* asyncio.CancelledError:
            pass
        except* Exception as eg:
            for e in eg.exceptions:
                try:
                    await ctx.send_json(
                        {"type": "error", "where": "llm", "message": str(e)}
                    )
                except Exception:
                    pass
    finally:
        # Roll back the user turn if we produced nothing (matches chatbot.py).
        if state["assistant_text"]:
            ctx.messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": state["assistant_text"]}],
                }
            )
        else:
            if ctx.messages and ctx.messages[-1]["role"] == "user":
                ctx.messages.pop()


async def _handle_audio(ctx: ConnCtx):
    """ASR on accumulated buffer, then run a turn with the recognized text."""
    pcm = bytes(ctx.audio_buf)
    ctx.audio_buf.clear()
    turn_id = ctx.turn_id
    try:
        text = await web_asr.transcribe(pcm)
    except Exception as e:
        await ctx.send_json({"type": "error", "where": "asr", "message": str(e)})
        return
    await ctx.send_json({"type": "asr_result", "text": text, "turn_id": turn_id})
    if not text:
        return
    await _handle_turn(ctx, text)


async def _start_new_turn(ctx: ConnCtx, runner):
    """Cancel any in-flight turn, bump turn_id, schedule the new turn."""
    await ctx.cancel_current()
    ctx.turn_id += 1
    ctx.current_task = asyncio.create_task(runner(ctx))


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    ctx = ConnCtx(ws)
    ctx.sender_task = asyncio.create_task(ctx._sender_loop())
    try:
        while True:
            msg = await ws.receive()
            mtype = msg.get("type")
            if mtype == "websocket.disconnect":
                break
            if "bytes" in msg and msg["bytes"] is not None:
                # incoming PCM (mic capture, int16 16 kHz mono)
                ctx.audio_buf.extend(msg["bytes"])
                continue
            text = msg.get("text")
            if not text:
                continue
            try:
                obj = json.loads(text)
            except json.JSONDecodeError:
                continue
            t = obj.get("type")
            if t == "text_input":
                user_text = (obj.get("text") or "").strip()
                if not user_text:
                    continue

                async def text_runner(c: ConnCtx, _ut=user_text):
                    await _handle_turn(c, _ut)

                await _start_new_turn(ctx, text_runner)
            elif t == "audio_start":
                # Barge-in implicit: cancel any in-flight turn so audio bytes
                # for the new utterance accumulate cleanly.
                await ctx.cancel_current()
                ctx.audio_buf.clear()
            elif t == "audio_end":

                async def audio_runner(c: ConnCtx):
                    await _handle_audio(c)

                await _start_new_turn(ctx, audio_runner)
            elif t == "cancel":
                await ctx.cancel_current()
            # unknown types ignored
    except WebSocketDisconnect:
        pass
    finally:
        ctx.closed = True
        await ctx.cancel_current()
        if ctx.sender_task:
            ctx.sender_task.cancel()
            try:
                await ctx.sender_task
            except Exception:
                pass
        try:
            await ws.close()
        except Exception:
            pass
