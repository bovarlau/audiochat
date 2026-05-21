"""Shared constants for both CLI (chatbot.py) and web stack."""

import os

from dotenv import load_dotenv

# override=True so values in .env take precedence over the shell's existing
# environment (e.g. a stray ANTHROPIC_BASE_URL pointing at api.anthropic.com).
load_dotenv(override=True)

API_KEY = os.environ.get("MINIMAX_API_KEY")
LLM_BASE_URL = os.environ.get("ANTHROPIC_BASE_URL", "https://api.minimaxi.com/anthropic")
TTS_URL = "https://api.minimaxi.com/v1/t2a_v2"

LLM_MODEL = "MiniMax-M2.7"
TTS_MODEL = "speech-2.8-hd"
TTS_VOICE = "male-qn-qingse"

SAMPLE_RATE_TTS = 32000
SAMPLE_RATE_ASR = 16000

SYSTEM_PROMPT = "You are a helpful assistant. 用中文回答用户。"
SENTENCE_ENDS = set("。！？!?；;\n")

LLM_MAX_TOKENS = 2000

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
