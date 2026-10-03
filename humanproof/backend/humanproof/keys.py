"""Which keys a deployment uses.

Kept separate from context.py (which pulls in the whole server) so that offline
tools such as ml/dataset/export.py can resolve the same keys with few dependencies.
"""
from __future__ import annotations

import base64
import binascii

from .config import Settings
from .security.crypto import KeyRing, SigningKeySet, keys_from_master, load_or_create_dev_keys


def master_bytes(value: str) -> bytes:
    """HP_MASTER_SECRET as key material.

    Either base64 of 32 or more random bytes, or any random string of 32 or more
    characters (a password manager's output works). The same value always gives
    the same keys, so it must be entered identically everywhere it is used.
    """
    value = value.strip()
    try:
        raw = base64.b64decode(value, validate=True)
        if len(raw) >= 32:
            return raw
    except (binascii.Error, ValueError):
        pass
    if len(value) < 32:
        raise RuntimeError("HP_MASTER_SECRET is too short: use 32 random bytes in base64, "
                           "or a random string of at least 32 characters")
    return value.encode()


def resolve_keys(settings: Settings) -> tuple[KeyRing, SigningKeySet, bytes]:
    """The encryption keyring, signing keys and pairwise secret for these settings."""
    keyring = signing = pairwise = None
    if settings.master_secret:
        keyring, signing, pairwise = keys_from_master(master_bytes(settings.master_secret))
    elif settings.env != "prod":
        keyring, signing, pairwise = load_or_create_dev_keys(settings.data_dir)
    # Individually configured keys take precedence over derived or generated ones.
    if settings.kek_keyring:
        keyring = KeyRing.from_spec(settings.kek_keyring)
    if settings.signing_keys:
        signing = SigningKeySet.from_spec(settings.signing_keys)
    if settings.pairwise_secret:
        pairwise = base64.b64decode(settings.pairwise_secret)
    if keyring is None or signing is None or pairwise is None:
        raise RuntimeError("No keys configured: set HP_MASTER_SECRET (or the three individual keys)")
    if len(pairwise) < 32:
        raise RuntimeError("HP_PAIRWISE_SECRET must be at least 32 bytes")
    return keyring, signing, pairwise
