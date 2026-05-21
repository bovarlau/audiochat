"""
MiniMax Text-to-Speech test using speech-2.8-hd model.
"""

import os
from dotenv import load_dotenv

load_dotenv()

import requests

API_URL = "https://api.minimaxi.com/v1/t2a_v2"
API_KEY = os.environ.get("MINIMAX_API_KEY")


def generate_speech(text: str, output_path: str = "output.mp3") -> bool:
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "speech-2.8-hd",
        "text": text,
        "stream": False,
        "voice_setting": {
            "voice_id": "male-qn-qingse",
            "speed": 1,
            "vol": 1,
            "pitch": 0,
            "emotion": "happy",
        },
        "audio_setting": {
            "sample_rate": 32000,
            "bitrate": 128000,
            "format": "mp3",
            "channel": 1,
        },
        "subtitle_enable": False,
    }

    try:
        print(f"Calling MiniMax API: {API_URL} (speech-2.8-hd, voice: male-qn-qingse)")

        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)

        if response.status_code == 200:
            audio_hex = response.json().get("data", {}).get("audio", "")
            if not audio_hex:
                print("Error: No audio data in response")
                print(f"Response: {response.text[:500]}")
                return False

            audio_bytes = bytes.fromhex(audio_hex)
            with open(output_path, "wb") as f:
                f.write(audio_bytes)
            print(f"Success! Saved to {output_path} ({len(audio_bytes)} bytes)")
            return True
        else:
            print(f"Error: HTTP {response.status_code}")
            print(f"Response: {response.text[:500]}")
            return False

    except Exception as e:
        print(f"Request failed: {e}")
        return False


if __name__ == "__main__":
    text = input("请输入要生成语音的中文文本: ").strip()
    if text:
        generate_speech(text, "output.mp3")
    else:
        print("Text cannot be empty")
