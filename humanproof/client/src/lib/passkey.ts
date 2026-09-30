// Passkey binding: after passing, the verified human can register a device-bound
// passkey so later re-verification is one biometric tap (see /v1/passkey/assert/*).
import { browserSupportsWebAuthn, startRegistration } from "@simplewebauthn/browser";
import * as api from "./api";
import { detectPlatform } from "./platform";

export function passkeysSupported(): boolean {
  // Embedded webviews (Capacitor, Tauri) do not expose WebAuthn reliably; the
  // native apps use their platform attestation key instead.
  return detectPlatform() === "web" && browserSupportsWebAuthn();
}

export async function bindPasskey(sessionId: string): Promise<void> {
  const options = await api.passkeyOptions(sessionId);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const credential = await startRegistration({ optionsJSON: options as any });
  await api.passkeyVerify(sessionId, credential);
}
