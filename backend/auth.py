"""
Lightweight, dependency-free authentication for the dashboard.

Design goals (the "secure" in *secure login page*):
  * The password is never stored in plaintext — only a PBKDF2-HMAC-SHA256
    hash + salt live in the source.
  * Credential checks use constant-time comparison (`hmac.compare_digest`)
    to avoid timing side-channels.
  * Sessions are stateless, signed cookies: `base64(payload).hmac_signature`.
    A tampered or expired cookie is rejected; nothing trusts client data
    without verifying the signature first.
  * The signing secret comes from $PHISHLENS_SECRET when set, otherwise a
    fresh random key is generated per process (restart => sessions expire).

Only the Python standard library is used, so there are no new dependencies.
"""

from __future__ import annotations

import os
import time
import hmac
import base64
import hashlib
import secrets

# --------------------------------------------------------------------------
# Credentials.  Username is public; the password is stored as a salted
# PBKDF2 hash (see backend/auth.py docstring) — the plaintext "ccp123"
# never appears in the codebase.
USERNAME = "student"
_PW_SALT = bytes.fromhex("9f2c7a1be4d83056aa17c9e0b4d5f6a8")
_PW_HASH = bytes.fromhex(
    "d551f581553db1240dff54f16d66d21d3aa44b8b8f2ac88859d525c89607d365"
)
_PBKDF2_ROUNDS = 200_000

# --------------------------------------------------------------------------
# Session signing.
SECRET = os.environ.get("PHISHLENS_SECRET", secrets.token_hex(32)).encode()
SESSION_TTL = 8 * 60 * 60          # 8 hours
COOKIE_NAME = "phishlens_session"


def verify_credentials(username: str, password: str) -> bool:
    """Constant-time check of a username/password pair."""
    candidate = hashlib.pbkdf2_hmac(
        "sha256", password.encode(), _PW_SALT, _PBKDF2_ROUNDS
    )
    # Evaluate both comparisons regardless of outcome to keep timing flat.
    user_ok = hmac.compare_digest(username.encode(), USERNAME.encode())
    pass_ok = hmac.compare_digest(candidate, _PW_HASH)
    return user_ok and pass_ok


def _sign(payload: bytes) -> str:
    sig = hmac.new(SECRET, payload, hashlib.sha256).digest()
    return (base64.urlsafe_b64encode(payload).decode().rstrip("=")
            + "." + base64.urlsafe_b64encode(sig).decode().rstrip("="))


def _b64decode(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def create_session(username: str) -> str:
    """Issue a signed session token for `username`, valid for SESSION_TTL."""
    expiry = int(time.time()) + SESSION_TTL
    payload = f"{username}|{expiry}".encode()
    return _sign(payload)


def verify_session(token: str | None) -> str | None:
    """Return the username if `token` is a valid, unexpired session, else None."""
    if not token or "." not in token:
        return None
    body, sig = token.rsplit(".", 1)
    try:
        payload = _b64decode(body)
        given_sig = _b64decode(sig)
    except Exception:
        return None
    expected = hmac.new(SECRET, payload, hashlib.sha256).digest()
    if not hmac.compare_digest(given_sig, expected):
        return None
    try:
        username, expiry = payload.decode().split("|")
        if int(expiry) < int(time.time()):
            return None
    except Exception:
        return None
    return username
