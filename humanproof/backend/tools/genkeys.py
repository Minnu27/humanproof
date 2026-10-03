"""Generate production key material. Run once, store the output in your secret manager.

    python -m tools.genkeys --out /secure/dir

Creates:
  signing-1.pem       Ed25519 private key for attestation tokens (chmod 600)
  and prints HP_KEK_KEYRING, HP_SIGNING_KEYS and HP_PAIRWISE_SECRET values.
Rotate by generating a new key, putting it FIRST in the list, and keeping the
old one after it until everything encrypted/signed with it has expired.
"""
from __future__ import annotations

import argparse
import base64
import os
import secrets
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--kid", default=f"k{secrets.token_hex(3)}")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    pem = Ed25519PrivateKey.generate().private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
    )
    key_path = args.out / f"signing-{args.kid}.pem"
    fd = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as fh:
        fh.write(pem)
    kek = base64.b64encode(secrets.token_bytes(32)).decode()
    pairwise = base64.b64encode(secrets.token_bytes(32)).decode()
    print(f"HP_KEK_KEYRING={args.kid}:{kek}")
    print(f"HP_SIGNING_KEYS={args.kid}:{key_path}")
    print("# or, where secrets must be environment variables (e.g. Vercel):")
    print(f"HP_SIGNING_KEYS={args.kid}:b64:{base64.b64encode(pem).decode()}")
    print(f"HP_PAIRWISE_SECRET={pairwise}")


if __name__ == "__main__":
    main()
