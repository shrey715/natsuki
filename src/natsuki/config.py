"""Environment/config loading. Never hardcode secrets here."""

import os

from dotenv import load_dotenv

load_dotenv()

MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")
MISTRAL_EMBED_MODEL = os.environ.get("MISTRAL_EMBED_MODEL", "mistral-embed")
MISTRAL_CHAT_MODEL = os.environ.get("MISTRAL_CHAT_MODEL", "mistral-small-latest")


def require_mistral_key() -> str:
    if not MISTRAL_API_KEY:
        raise RuntimeError(
            "MISTRAL_API_KEY is not set. Add it to a .env file in the project root."
        )
    return MISTRAL_API_KEY
