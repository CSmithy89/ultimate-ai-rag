from __future__ import annotations


API_KEY_PREFIXES = {
    "openai": "sk-",
    "anthropic": "sk-ant-",
    "openrouter": "sk-or-",
}
MIN_KEY_LENGTH_DEFAULT = 32
MIN_GEMINI_KEY_LENGTH = 32


def requires_api_key(provider: str) -> bool:
    return provider not in {"ollama"}


def validate_api_key(provider: str, api_key: str) -> bool:
    if not requires_api_key(provider):
        return True
    if not api_key:
        return False
    if provider == "gemini":
        return api_key.startswith("AIza") and len(api_key.strip()) >= MIN_GEMINI_KEY_LENGTH
    prefix = API_KEY_PREFIXES.get(provider)
    if prefix:
        return api_key.startswith(prefix)
    return len(api_key.strip()) >= MIN_KEY_LENGTH_DEFAULT
