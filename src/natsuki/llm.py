"""Thin wrapper around the Mistral chat API.

Note: this SDK build (mistralai==2.6.0) ships with no root-level
`mistralai/__init__.py` -- `from mistralai import Mistral` fails. The
working import is `from mistralai.client import Mistral`.
"""

from __future__ import annotations

from natsuki.config import MISTRAL_CHAT_MODEL, require_mistral_key

_client = None


def _get_client():
    global _client
    if _client is None:
        from mistralai.client import Mistral

        _client = Mistral(api_key=require_mistral_key())
    return _client


def chat(
    messages: list[dict[str, str]],
    model: str | None = None,
    temperature: float = 0.0,
) -> str:
    client = _get_client()
    response = client.chat.complete(
        model=model or MISTRAL_CHAT_MODEL,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content
