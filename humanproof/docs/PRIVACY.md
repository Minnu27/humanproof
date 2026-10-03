# Privacy and data handling

Biometric data is regulated (Illinois BIPA, Texas CUBI, Washington, GDPR
Art. 9, CCPA/CPRA). HumanProof is built to minimise what it touches.

| Data | Where it is processed | Stored by default? |
|---|---|---|
| Camera video | On the device (MediaPipe) | Never leaves the device |
| Eye/head/mouth measurements (numbers per frame) | Server | Kept with the session for up to 1 h after expiry, then purged |
| 4 face snapshots (224 px) | Server, in memory | No, discarded after scoring |
| Voice clip (≤ 8 s) | Server, in memory | No, discarded after scoring |
| Voice signature (192 numbers) | Server | Only if the check passes; AES-256-GCM encrypted; deletable by the user with their passkey |
| Pointer/touch trace | Server | Session only |
| IP address | Server | Never stored; a keyed hash is used for rate limiting |

## Optional storage for training

Off unless the operator enables it (`docs/DATA.md`). When enabled, a person can
choose, separately and unticked by default:

* **Keep the measurements** from this attempt: the per-frame numbers, the pointer
  trace, the challenge and the scores. No audio, no images.
* **Also keep the recordings:** the voice clip and the four face snapshots.

What is stored is encrypted per row with a key held outside the database, carries
no session ID, name or IP address, expires after `HP_DATA_RETENTION_DAYS`
(default 365), and is used only to test and improve the checks. Ordinary users'
data is unlabelled and is not used to train models. The API tells the app exactly
what it will keep (`storing`), and the consent screen offers these boxes only when
the server can actually store.

Tester sessions (private code required) always store everything, with the label
the tester gives; the consent screen says so before the camera opens.

## Consent

* The app blocks until the user ticks explicit consent to biometric processing
  (`consent.biometric_processing: true`, versioned `2026-10-v2`). The API rejects
  sessions without it.
* Storage for training is separate consent (`research_opt_in`, `media_opt_in`),
  and declining does not affect the result.

## Rights

* **Deletion:** `POST /v1/passkey/assert/options {"purpose": "delete"}` then
  `/v1/passkey/assert/verify` removes the subject, their voiceprint and any
  samples stored from their attempts.
* **Deleting one attempt's stored data:** the result screen shows a receipt and a
  delete button; `POST /v1/data/delete {"receipt": "..."}` works later too. Only a
  hash of the receipt is stored, so the rows cannot be tied to the holder by
  someone reading the database.
* **Abandoned attempts** are not kept: their samples are purged within about an hour.
* **Unlinkability:** each relying party sees a different pseudonym for the same
  person, so apps cannot combine their records through HumanProof.

## Before launch (needs a lawyer, not code)

* A published biometric retention policy and destruction schedule (BIPA §15(a)).
  Storing recordings for training makes this mandatory, not optional: state the
  purpose, the retention period you configured and how to delete.
* Decide who may hold `HP_MASTER_SECRET` and run exports; an export is unencrypted
  biometric data on someone's disk.
* Written-release wording reviewed for each launch jurisdiction.
* A data-processing agreement for relying parties.
* A DPIA if serving EU users.
