import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from humanproof.security import tokens
from humanproof.security.crypto import SigningKey, SigningKeySet

SDK = Path(__file__).resolve().parents[2] / "sdk"
sys.path.insert(0, str(SDK))

from humanproof_verify import InvalidToken, Verifier  # noqa: E402


def _keys():
    return SigningKeySet([SigningKey("s1", Ed25519PrivateKey.generate())])


def test_python_sdk_accepts_valid_and_rejects_invalid():
    ks = _keys()
    tok = tokens.sign(ks, {"aud": "bank.example", "sub": "x", "hp_human": True}, 60, "https://hp.example")
    v = Verifier("https://hp.example", "bank.example", jwks=ks.jwks())
    assert v.verify(tok)["sub"] == "x"
    with pytest.raises(InvalidToken):
        Verifier("https://hp.example", "other.example", jwks=ks.jwks()).verify(tok)
    with pytest.raises(InvalidToken):
        Verifier("https://hp.example", "bank.example", jwks=_keys().jwks()).verify(tok)
    h, p, s = tok.split(".")
    with pytest.raises(InvalidToken):
        v.verify(f"{h}.{p}.{s[:-4]}AAAA")


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_js_sdk(tmp_path):
    ks = _keys()
    good = tokens.sign(ks, {"aud": "bank.example", "sub": "y", "hp_human": True}, 60, "http://localhost:1")
    bad_aud = tokens.sign(ks, {"aud": "evil.example", "hp_human": True}, 60, "http://localhost:1")
    script = tmp_path / "t.mjs"
    script.write_text(f"""
import {{ createVerifier }} from {json.dumps(str(SDK / 'verify.mjs'))};
globalThis.fetch = async () => ({{ ok: true, json: async () => ({json.dumps(ks.jwks())}) }});
const v = createVerifier({{ issuer: "http://localhost:1", audience: "bank.example" }});
const c = await v({json.dumps(good)});
if (c.sub !== "y") throw new Error("claims");
let rejected = false;
try {{ await v({json.dumps(bad_aud)}); }} catch {{ rejected = true; }}
if (!rejected) throw new Error("audience not enforced");
console.log("ok");
""")
    out = subprocess.run(["node", str(script)], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "ok"
