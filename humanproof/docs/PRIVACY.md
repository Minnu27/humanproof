# Privacy and data handling

Biometric data is regulated (Illinois BIPA, Texas CUBI, Washington, GDPR
Art. 9, CCPA/CPRA). HumanProof is built to minimise what it touches.

| Data | Where it is processed | Stored? |
|---|---|---|
| Camera video | On the device (MediaPipe) | Never leaves the device |
| Eye/head/mouth measurements (numbers per frame) | Server | Kept with the session for up to 1 h after expiry, then purged; kept longer only as anonymous features if the user opted in |
| 4 face snapshots (224 px) | Server, in memory | No, discarded after scoring |
| Voice clip (≤ 8 s) | Server, in memory | No, discarded after scoring |
| Voice signature (192 numbers) | Server | Only if the check passes; AES-256-GCM encrypted; deletable by the user with their passkey |
| Pointer/touch trace | Server | Session only (features), unless opted in |
| IP address | Server | Never stored; a keyed hash is used for rate limiting |

## Consent

* The app blocks until the user ticks explicit consent to biometric processing
  (`consent.biometric_processing: true`, versioned `2026-09-v1`). The API rejects
  sessions without it.
* Research use is a separate, unticked-by-default opt-in. Only feature vectors
  are kept, never images or audio.

## Rights

* **Deletion:** `POST /v1/passkey/assert/options {"purpose": "delete"}` then
  `/v1/passkey/assert/verify` removes the subject and their voiceprint.
* **Unlinkability:** each relying party sees a different pseudonym for the same
  person, so apps cannot combine their records through HumanProof.

## Before launch (needs a lawyer, not code)

* A published biometric retention policy and destruction schedule (BIPA §15(a)).
* Written-release wording reviewed for each launch jurisdiction.
* A data-processing agreement for relying parties.
* A DPIA if serving EU users.
