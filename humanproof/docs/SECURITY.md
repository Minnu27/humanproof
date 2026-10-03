# Security controls

This maps the implementation to the OWASP Application Security Verification
Standard (ASVS 4.0.3, Level 2) and Mobile ASVS where it applies. "Test" names
the automated test that checks the control; all run in CI (`.github/workflows`).

## What runs on every push

| Check | Tool | Workflow |
|---|---|---|
| Unit, API, attestation and SDK tests (80% coverage gate) | pytest | `ci.yml` |
| Python static analysis | bandit | `ci.yml` |
| Python dependency CVEs | pip-audit (strict) | `ci.yml` |
| JS dependency CVEs | npm audit (high+) | `ci.yml` |
| Code scanning (Python, TypeScript) | CodeQL security-extended | `security.yml` |
| Committed secrets | gitleaks | `security.yml` |
| Container CVEs | Trivy (high+, fail) | `security.yml` |
| Dynamic API scan | OWASP ZAP API scan | `security.yml` |

## ASVS mapping

| ASVS | Control | Implementation | Test |
|---|---|---|---|
| V1.1 | Threat model | `docs/THREAT_MODEL.md` | – |
| V2.1/2.2 | Authenticators resist replay | WebAuthn passkeys (UV required), App Attest counters | `test_apple_assertions_and_replay` |
| V3.x | Session management | 256-bit random session IDs, 5 min TTL, single-use checkpoints, no cookies | `test_session_expiry`, `test_double_submit_rejected` |
| V4.1 | Access control on flow state | Checkpoint order enforced; passkeys only after pass | `test_checkpoints_must_run_in_order`, `test_passkey_requires_pass` |
| V5.1 | Input validation | Pydantic strict schemas, bounds on every number, `extra=forbid` | `test_unknown_fields_and_bounds_rejected` |
| V5.2 | Sanitisation / no reflection | Validation errors report field names, never values | same |
| V5.5 | Safe deserialisation | JSON only; images via Pillow with size + pixel limits; WAV parser strict; models ONNX (no pickle) | `test_wav_validation`, `test_tampered_model_refused` |
| V6.2 | Approved crypto | AES-256-GCM, Ed25519, HMAC-SHA256, SHA-256 (pyca/cryptography) | `test_crypto.py` |
| V6.4 | Key management | Keys outside DB; key IDs and rotation; prod refuses dev keys | `test_key_rotation`, `test_prod_refuses_insecure_config` |
| V7.1 | Logging without sensitive data | Audit log stores truncated session IDs, scores, decisions; no biometrics, no IPs | `storage.audit` |
| V7.3 | Log integrity | HMAC hash chain | `test_audit_chain_detects_tampering` |
| V8.1–8.3 | Data protection | No raw video stored; audio and face snapshots stored only with separate opt-in (or in tester sessions), encrypted per row, expiring, deletable by receipt; voiceprints encrypted; `Cache-Control: no-store` | `test_nothing_is_stored_without_opt_in`, `test_sample_cannot_be_decrypted_as_another_row`, `test_person_can_delete_their_data_with_the_receipt`, `test_samples_expire` |
| V8.3 | Sensitive data not retained longer than needed | Retention limit on samples; abandoned attempts purged; passkey deletion removes samples | `test_abandoned_attempts_are_not_kept`, `test_passkey_bind_reverify_and_delete` |
| V1.14 / V4.2 | State shared safely between instances | Sessions, passkey challenges and audit log in Postgres; session updates are compare-and-swap; audit chain serialised by an advisory lock | `test_storage.py` |
| V9.1 | TLS | Prod requires https base URL; HSTS preload header; Caddy auto-TLS | `test_prod_refuses_insecure_config` |
| V10.3 | Integrity of code/artefacts | Model SHA-256 manifest; client model SHA-256 lock; pinned deps | `test_tampered_model_refused` |
| V11.1 | Business-logic limits | Submission time windows; per-client hourly cap | `test_submission_too_fast_is_burned`, `test_rate_limit` |
| V12.1 | File upload limits | 2.5 MB body cap enforced even without Content-Length | `test_body_size_limit` |
| V13.1 | API security | Strict CORS allowlist, no credentials; OpenAPI hidden in prod | `test_cors_only_allowed_origins` |
| V14.4 | Security headers | HSTS, CSP (`default-src 'none'` on API), XFO, nosniff, Referrer-Policy, COOP/CORP | `test_security_headers` |
| MASVS-RESILIENCE | App/device integrity | App Attest, Play Integrity with per-request binding | `test_attestation.py` |
| MASVS-NETWORK | No cleartext | Android `usesCleartextTraffic=false`; iOS ATS default | `scripts/patch-native.mjs` |
| MASVS-STORAGE | No backups of app data | Android `allowBackup=false`; app stores nothing locally | same |

## Operating it safely

* Keys: either one `HP_MASTER_SECRET` (encryption, signing and pairwise keys are
  derived from it; simplest, but it cannot be rotated piecemeal), or separate keys
  from `python -m tools.genkeys`. Keep them in a secret manager or KMS/HSM and back
  them up: stored samples and voiceprints are unreadable without the encryption key.
  Never commit `.env` or `secrets/`.
* Rotate (separate keys): add the new key first in `HP_KEK_KEYRING` /
  `HP_SIGNING_KEYS`, keep the old one after it until tokens and ciphertexts using
  it have expired.
* Use Postgres (`HP_DATABASE_URL`) whenever more than one instance runs. Per-minute
  rate limits are still per process; add Redis before relying on them at scale.
* Training data: keep `HP_COLLECTION_KEY` private and rotate it when a tester
  leaves; treat every dataset export as plaintext biometric data (`docs/DATA.md`).
* Before selling to regulated customers: accredited PAD test (ISO/IEC 30107-3),
  external penetration test, and SOC 2 Type II for the operation around it.

## Reporting a vulnerability

Email security@humanproof.example with steps to reproduce. Please do not open
public issues for security problems.
