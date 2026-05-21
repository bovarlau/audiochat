"""Async LLM streaming using anthropic.AsyncAnthropic against MiniMax endpoint."""

from typing import Awaitable, Callable

import anthropic

from config import API_KEY, LLM_BASE_URL, LLM_MAX_TOKENS, LLM_MODEL, SYSTEM_PROMPT

_client = anthropic.AsyncAnthropic(api_key=API_KEY, base_url=LLM_BASE_URL)


async def stream_reply(
    messages: list,
    on_text_delta: Callable[[str], Awaitable[None]],
    on_thinking_delta: Callable[[str], Awaitable[None]] | None = None,
) -> str:
    """
    Stream a reply from the LLM. Returns accumulated assistant text.

    Cancellation: the caller's task can be cancelled at any time; the
    `async with stream(...)` context manager will close the HTTP response.
    """
    assistant_text = ""
    async with _client.messages.stream(
        model=LLM_MODEL,
        max_tokens=LLM_MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=messages,
    ) as stream:
        async for event in stream:
            etype = getattr(event, "type", None)
            if etype != "content_block_delta":
                continue
            delta = getattr(event, "delta", None)
            if not delta:
                continue
            dtype = getattr(delta, "type", None)
            if dtype == "thinking_delta":
                thinking = getattr(delta, "thinking", "") or ""
                if thinking and on_thinking_delta:
                    await on_thinking_delta(thinking)
            elif dtype == "text_delta":
                text = getattr(delta, "text", "") or ""
                if text:
                    assistant_text += text
                    await on_text_delta(text)
    return assistant_text
