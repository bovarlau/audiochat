"""
MiniMax LLM streaming response test.
"""

import os
from dotenv import load_dotenv

load_dotenv()

import anthropic

API_KEY = os.environ.get("MINIMAX_API_KEY")
BASE_URL = os.environ.get("ANTHROPIC_BASE_URL", "https://api.minimaxi.com/anthropic")
client = anthropic.Anthropic(api_key=API_KEY, base_url=BASE_URL)


def chat(prompt: str) -> bool:
    print("Starting stream response...\n")
    print("=" * 60)
    print("Thinking Process:")
    print("=" * 60)

    reasoning_buffer = ""
    text_buffer = ""

    try:
        stream = client.messages.create(
            model="MiniMax-M2.7",
            max_tokens=1000,
            system="You are a helpful assistant.",
            messages=[
                {"role": "user", "content": [{"type": "text", "text": prompt}]}
            ],
            stream=True,
        )

        for chunk in stream:
            if chunk.type == "content_block_start":
                if hasattr(chunk, "content_block") and chunk.content_block:
                    if chunk.content_block.type == "text":
                        print("\n" + "=" * 60)
                        print("Response Content:")
                        print("=" * 60)

            elif chunk.type == "content_block_delta":
                if hasattr(chunk, "delta") and chunk.delta:
                    if chunk.delta.type == "thinking_delta":
                        new_thinking = chunk.delta.thinking
                        if new_thinking:
                            print(new_thinking, end="", flush=True)
                            reasoning_buffer += new_thinking
                    elif chunk.delta.type == "text_delta":
                        new_text = chunk.delta.text
                        if new_text:
                            print(new_text, end="", flush=True)
                            text_buffer += new_text

        print("\n")
        return True

    except Exception as e:
        print(f"Request failed: {e}")
        return False


if __name__ == "__main__":
    prompt = input("请输入对话内容: ").strip()
    if prompt:
        chat(prompt)
    else:
        print("Prompt cannot be empty")
