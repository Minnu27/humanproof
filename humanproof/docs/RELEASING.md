# Releasing the apps

Push a tag (`git tag v1.0.0 && git push --tags`) and `.github/workflows/release-apps.yml`
builds everything. Configure these first in GitHub → Settings → Secrets and variables.

## Repository variables

| Name | Example |
|---|---|
| `API_BASE` | Full API URL including its path: `https://your-domain/api` on Vercel, `https://api.humanproof.example` for a dedicated API host |
| `RELYING_PARTY` | `humanproof` |
| `PLAY_CLOUD_PROJECT` | Google Cloud project number linked to Play Integrity |

## Secrets

| Platform | Secrets | How to get them |
|---|---|---|
| macOS (signed + notarised) | `APPLE_CERTIFICATE`, `APPLE_CERTIFICATE_PASSWORD`, `APPLE_SIGNING_IDENTITY`, `APPLE_ID`, `APPLE_PASSWORD`, `APPLE_TEAM_ID` | Apple Developer Program ($99/yr): "Developer ID Application" certificate exported as base64 .p12; app-specific password for notarisation |
| Windows | optional code-signing certificate (see Tauri docs) | Without it, SmartScreen warns on first launch |
| Android | `ANDROID_KEYSTORE_B64`, `ANDROID_KEYSTORE_PASSWORD`, `ANDROID_KEY_ALIAS`, `ANDROID_KEY_PASSWORD` | `keytool -genkey -v -keystore release.jks -keyalg RSA -keysize 4096 -validity 10000 -alias humanproof`, then base64 it. Google Play developer account ($25 one-time) |
| iOS | `IOS_DIST_P12_B64`, `IOS_DIST_P12_PASSWORD`, `IOS_PROFILE_B64`, `APPLE_TEAM_ID` | "Apple Distribution" certificate + App Store provisioning profile with the **App Attest** capability enabled |

## Server-side attestation setup

* **iOS:** download Apple's App Attestation root CA
  (`Apple_App_Attestation_Root_CA.pem` from apple.com/certificateauthority) into
  `HP_DATA_DIR`, and set `HP_APPLE_TEAM_ID` / `HP_APPLE_BUNDLE_ID`.
* **Android:** in Play Console → App integrity, link a Cloud project, create a
  service account with Play Integrity access, mount its JSON key and set
  `HP_GOOGLE_SERVICE_ACCOUNT_FILE` and `HP_ANDROID_PACKAGE_NAME`.

Without these, mobile apps still work, at the lower "web" assurance level.

## Store review notes

Both stores require a clear explanation of camera, microphone and biometric use.
The usage strings are set by `client/scripts/patch-native.mjs`; the in-app consent
screen and `docs/PRIVACY.md` are what reviewers will read.
