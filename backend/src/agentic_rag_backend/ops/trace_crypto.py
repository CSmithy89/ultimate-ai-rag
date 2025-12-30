from __future__ import annotations

import base64
import os

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import structlog


_PREFIX = "enc:"
logger = structlog.get_logger(__name__)


class TraceCrypto:
    def __init__(self, key_hex: str) -> None:
        key = bytes.fromhex(key_hex)
        if len(key) != 32:
            raise ValueError("TRACE_ENCRYPTION_KEY must be 32 bytes (64 hex chars).")
        self._aesgcm = AESGCM(key)

    def encrypt(self, plaintext: str) -> str:
        nonce = os.urandom(12)
        ciphertext = self._aesgcm.encrypt(
            nonce,
            plaintext.encode("utf-8"),
            None,
        )
        payload = base64.urlsafe_b64encode(nonce + ciphertext).decode("ascii")
        return _PREFIX + payload

    def decrypt(self, payload: str) -> str:
        if not payload.startswith(_PREFIX):
            return payload
        try:
            data = base64.urlsafe_b64decode(payload[len(_PREFIX):])
            nonce = data[:12]
            ciphertext = data[12:]
            plaintext = self._aesgcm.decrypt(nonce, ciphertext, None)
            return plaintext.decode("utf-8")
        except Exception as exc:
            logger.warning(
                "trace_decrypt_failed",
                error=str(exc),
                error_type=exc.__class__.__name__,
            )
            return f"<encrypted: {exc.__class__.__name__}>"
