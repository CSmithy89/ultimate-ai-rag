from __future__ import annotations


API_KEY_PREFIXES = {
    "openai": "sk-",
    "anthropic": "sk-ant-",
    "openrouter": "sk-or-",
}


def requires_api_key(provider: str) -> bool:
    return provider not in {"ollama"}


def validate_api_key(provider: str, api_key: str) -> bool:
    if not requires_api_key(provider):
        return True
    if not api_key:
        return False
    prefix = API_KEY_PREFIXES.get(provider)
    if prefix:
        return api_key.startswith(prefix)
    return len(api_key.strip()) >= 10
