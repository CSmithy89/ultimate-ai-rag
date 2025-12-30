"""Unit tests for trace encryption utilities."""

import secrets

from agentic_rag_backend.ops.trace_crypto import TraceCrypto


def test_encrypt_decrypt_roundtrip() -> None:
    crypto = TraceCrypto(secrets.token_hex(32))
    payload = crypto.encrypt("hello world")

    assert payload.startswith("enc:")
    assert crypto.decrypt(payload) == "hello world"


def test_decrypt_passthrough_for_plaintext() -> None:
    crypto = TraceCrypto(secrets.token_hex(32))
    assert crypto.decrypt("plain text") == "plain text"


def test_decrypt_invalid_payload_returns_placeholder() -> None:
    crypto = TraceCrypto(secrets.token_hex(32))
    result = crypto.decrypt("enc:invalid")

    assert result.startswith("<encrypted:")
