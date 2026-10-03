# Threat model

HumanProof answers one question for a relying party (a bank, a dating app, a
video-call platform): *is a live human being operating this device right now?*
It does **not** prove who that human is (that is identity verification), and a
verified human can still choose to commit fraud.

## Assets

| Asset | Why it matters |
|---|---|
| Attestation-token signing key (Ed25519) | Whoever holds it can mint "verified human" tokens for anyone. |
| Key-encryption keys (AES-256) | Protect stored voiceprints and research features. |
| Pairwise secret | Derives per-app pseudonyms; leaking it lets apps link users across services. |
| Stored voiceprints | Biometric templates; regulated data (BIPA, GDPR Art. 9). |
| Trained models | Tampering could silently accept bots. |

## Attackers and what stops them

| # | Attack | Defence | Where |
|---|---|---|---|
| A1 | **Replay a recorded human** (video, mouse trace, audio) | Every challenge (dot path, curve, five words) is fresh CSPRNG output, revealed only when the checkpoint starts; responses must follow *this* challenge | `challenges.py`, per-jump eye displacement vs an unpredictable dot path (`scoring/gaze.py`), motor adherence, phrase match |
| A2 | **Pre-compute responses** | Challenge hidden until `/start`; submissions accepted only inside a time window (not faster than the challenge can physically be performed); one submission per checkpoint | `api.py` `_begin_submit` |
| A3 | **Scripted bot drives the API directly** with synthetic signals | Motor model trained on real human movement vs five bot families; gaze timing (saccade latency, micro-motion); voice anti-spoof on *raw audio*; face deepfake model on *raw pixels*; fused decision with per-checkpoint floors | `scoring/*`, `fusion.py` |
| A4 | **Real-time deepfake face + cloned voice** (the $25M case) | Face-crop deepfake detector; voice anti-spoofing model; lip-sync check across face and voice; random words defeat pre-generated TTS | `face.py`, `voice.py` |
| A5 | **Virtual camera / emulator injection** | iOS App Attest and Android Play Integrity (device + app integrity); every submission signed by the attested key over the exact body; web/desktop get lower "web" assurance with a stricter pass bar | `security/attestation.py` |
| A6 | **Token forgery / algorithm confusion** | Single algorithm (EdDSA) and token type; unknown `kid`, `alg:none`, HS256 all rejected; `aud`, `iss`, `exp`, `nbf` enforced | `security/tokens.py`, `sdk/` |
| A7 | **Token theft and reuse at another app** | Tokens are audience-scoped and expire in 15 min; `sub` is a per-app pseudonym | `api.py` `_issue_token` |
| A8 | **Cross-app tracking** by relying parties | Pairwise pseudonymous subject per relying party | `crypto.pairwise_subject` |
| A9 | **Database theft** | No raw video/audio/IP stored; voiceprints envelope-encrypted with record-bound AAD; keys outside the DB | `storage.py`, `crypto.KeyRing` |
| A10 | **Tampered model file** | SHA-256 pinned in `manifest.json`, checked at load; ONNX only (no pickle, no code execution on load) | `models.py` |
| A11 | **Audit-log tampering** | HMAC hash chain; editing/deleting a row breaks verification | `storage.audit` |
| A12 | **Abuse / enumeration** | Per-client rate limits (keyed hash, not stored IPs), per-client hourly session cap, body-size limits, strict schemas that reject unknown fields and never echo values | `security/http.py`, `schemas.py` |
| A13 | **Malicious/insecure deployment config** | Production refuses to start without real keys, HTTPS, explicit origins, ASR, and all trained models; heuristic fallback forbidden | `config.validate_for_env`, `context.py` |
| A14 | **XSS / supply chain in the app** | Strict build-time CSP (no inline script, no remote code, no eval), WASM and models bundled locally with pinned hashes, no CDN | `client/vite.config.ts`, `scripts/fetch-models.mjs` |

## Known limits (be honest with customers about these)

* **A human puppeteer passes.** If a person sits at the device and performs the
  checks, the system correctly says "a human is present". Stopping *that* person
  from impersonating someone else needs identity binding (passkeys, ID documents).
* **Web clients can be scripted.** A browser cannot prove it is unmodified. Web
  results are labelled `hp_assurance: web`; high-value flows should require
  `device_attested`.
* **Detection models age.** New voice-cloning and face-swap generators appear
  constantly. Re-train on fresh attack data on a schedule and track the
  cross-dataset metrics in `docs/MODELS.md`, not the in-dataset ones.
* **In-memory state.** Rate limits and pending passkey challenges live in one
  process. Run one API worker, or move them to Redis before scaling out.
* **Certification is external.** Liveness claims to banks need an accredited
  ISO/IEC 30107-3 presentation-attack-detection test (e.g. iBeta), plus a
  penetration test. This repository prepares for those; it does not replace them.
