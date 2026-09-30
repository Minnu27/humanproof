// Platform detection and device attestation (Apple App Attest / Play Integrity).
// Web and desktop builds cannot attest and run at the lower "web" assurance level.
import { Capacitor, registerPlugin } from "@capacitor/core";

export type Platform = "web" | "ios" | "android" | "desktop";

interface AttestPlugin {
  isSupported(): Promise<{ supported: boolean }>;
  // iOS: generates an App Attest key and attests it over SHA-256(challenge).
  attest(opts: { challenge: string }): Promise<{
    keyId?: string;
    attestation?: string;
    token?: string;
  }>;
  // iOS: assertion over SHA-256(clientData). Android: integrity token for requestHash.
  bind(opts: {
    clientData?: string;
    requestHash?: string;
    cloudProjectNumber?: string;
  }): Promise<{ assertion?: string; token?: string }>;
}

const Attest = registerPlugin<AttestPlugin>("HumanProofAttest");

export function detectPlatform(): Platform {
  const p = Capacitor.getPlatform();
  if (p === "ios" || p === "android") return p;
  if ("__TAURI_INTERNALS__" in window) return "desktop";
  return "web";
}

export function b64urlToBytes(s: string): Uint8Array {
  const b64 = s.replace(/-/g, "+").replace(/_/g, "/") + "=".repeat((4 - (s.length % 4)) % 4);
  return Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
}

export function bytesToB64url(b: ArrayBuffer | Uint8Array): string {
  const bytes = b instanceof Uint8Array ? b : new Uint8Array(b);
  let s = "";
  for (const x of bytes) s += String.fromCharCode(x);
  return btoa(s).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

async function sha256(data: Uint8Array | string): Promise<ArrayBuffer> {
  const bytes = typeof data === "string" ? new TextEncoder().encode(data) : data;
  return crypto.subtle.digest("SHA-256", bytes as BufferSource);
}

async function sha256Hex(data: string): Promise<string> {
  return [...new Uint8Array(await sha256(data))].map((b) => b.toString(16).padStart(2, "0")).join("");
}

/** Must match backend security/attestation.py request_binding(). */
export async function requestBinding(sessionId: string, body: string): Promise<string> {
  return JSON.stringify({ s: sessionId, b: await sha256Hex(body) });
}

const CLOUD_PROJECT = (import.meta.env.VITE_PLAY_CLOUD_PROJECT as string | undefined) ?? "";

/** Returns the payload for POST /attest, or null when this device cannot attest. */
export async function attestDevice(platform: Platform, challengeB64url: string): Promise<object | null> {
  if (platform !== "ios" && platform !== "android") return null;
  try {
    const { supported } = await Attest.isSupported();
    if (!supported) return null;
    if (platform === "ios") {
      const r = await Attest.attest({ challenge: challengeB64url });
      return { kind: "apple_app_attest", key_id: r.keyId, payload: r.attestation };
    }
    const requestHash = bytesToB64url(await sha256(b64urlToBytes(challengeB64url)));
    const r = await Attest.bind({ requestHash, cloudProjectNumber: CLOUD_PROJECT });
    return { kind: "play_integrity", payload: r.token };
  } catch {
    return null; // attestation unavailable: continue at web assurance
  }
}

export async function deviceBindingHeaders(
  platform: Platform,
  sessionId: string,
  body: string,
): Promise<Record<string, string>> {
  const binding = await requestBinding(sessionId, body);
  if (platform === "ios") {
    const r = await Attest.bind({ clientData: binding });
    return { "x-hp-assertion": r.assertion ?? "" };
  }
  if (platform === "android") {
    const r = await Attest.bind({
      requestHash: bytesToB64url(await sha256(binding)),
      cloudProjectNumber: CLOUD_PROJECT,
    });
    return { "x-hp-integrity": r.token ?? "" };
  }
  return {};
}
