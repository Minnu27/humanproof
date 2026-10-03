// Thin, typed client for the HumanProof API.
// Every checkpoint body is serialised exactly once; on attested devices the
// same bytes are hashed and signed, so the server can prove they came from the
// genuine app and were not modified in transit.
import { deviceBindingHeaders, type Platform } from "./platform";

// Web builds call the API on the same origin under /api (Vercel routes /api/* to the
// backend service; `npm run dev` proxies it). Desktop and mobile builds are not
// served from that origin, so they are built with an absolute VITE_API_BASE,
// e.g. https://humanproof.example/api.
export const API_BASE = ((import.meta.env.VITE_API_BASE as string | undefined) || "/api").replace(/\/$/, "");
export const RELYING_PARTY = (import.meta.env.VITE_RELYING_PARTY as string | undefined) ?? "humanproof";
export const CONSENT_VERSION = "2026-10-v2";

export type Checkpoint = "gaze" | "motor" | "voice";

export type AttackType =
  | "replay_video"
  | "photo"
  | "face_swap"
  | "synthetic_face"
  | "tts_voice"
  | "voice_clone"
  | "audio_replay"
  | "bot_pointer"
  | "scripted_client"
  | "other";

/** What the person agreed to have kept beyond the check itself. Both are optional. */
export interface SharingChoice {
  measurements: boolean;
  recordings: boolean;
}

/** A trusted tester recording a session with a known answer (see docs/DATA.md). */
export interface TesterLabel {
  code: string;
  participant: string;
  label: "human" | "attack";
  attack_type?: AttackType;
}

export interface Session {
  session_id: string;
  expires_in: number;
  steps: Checkpoint[];
  assurance: "web" | "device_attested";
  attestation_challenge: string;
  // What the server will actually keep from this attempt (it may be less than was
  // offered, e.g. when storage is switched off), and the receipt that deletes it.
  storing: "nothing" | "measurements" | "measurements_and_media";
  data_receipt: string | null;
  labelled: boolean;
}

export interface GazeKeyframe {
  start: number;
  end: number;
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

export interface GazeChallenge {
  duration_ms: number;
  keyframes: GazeKeyframe[];
  capture_ms: number[];
}
export interface MotorChallenge {
  duration_ms: number;
  polyline: [number, number][];
}
export interface VoiceChallenge {
  words: string[];
  max_duration_ms: number;
}

export interface CheckpointResult {
  checkpoint: Checkpoint;
  passed: boolean;
  score: number;
  next: Checkpoint | null;
}

export interface Decision {
  decision: "pass" | "step_up" | "reject";
  score: number;
  assurance: string;
  checkpoints: Record<string, number>;
  reasons: string[];
  attestation_token: string | null;
  subject_hint: string | null;
  // Present on demo (non-production) servers only: the measurements behind each score.
  debug?: Record<string, { score?: number; reasons?: string[]; features?: unknown; info?: unknown }> | null;
  data_stored: boolean;
}

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

async function request<T>(path: string, body?: string, headers: Record<string, string> = {}): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "content-type": "application/json", ...headers },
    body,
    credentials: "omit",
    cache: "no-store",
    referrerPolicy: "no-referrer",
  });
  if (!res.ok) {
    let detail = `Request failed (${res.status})`;
    try {
      const j = await res.json();
      if (typeof j.detail === "string") detail = j.detail;
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(res.status, detail);
  }
  return (await res.json()) as T;
}

export function createSession(platform: Platform, sharing: SharingChoice, tester?: TesterLabel): Promise<Session> {
  return request<Session>(
    "/v1/sessions",
    JSON.stringify({
      platform,
      relying_party: RELYING_PARTY,
      consent: {
        version: CONSENT_VERSION,
        biometric_processing: true,
        research_opt_in: sharing.measurements,
        media_opt_in: sharing.recordings,
      },
      ...(tester ? { collection: tester } : {}),
    }),
  );
}

export interface DataPolicy {
  sharing: boolean; // the server can store data that people choose to share
  tester_mode: boolean; // labelled tester sessions are enabled
  retention_days: number;
}

/** What this server is able to keep. Unknown or unreachable means "offer nothing". */
export async function getDataPolicy(): Promise<DataPolicy> {
  try {
    const res = await fetch(`${API_BASE}/v1/data/policy`, { credentials: "omit", cache: "no-store" });
    if (res.ok) return (await res.json()) as DataPolicy;
  } catch {
    /* fall through */
  }
  return { sharing: false, tester_mode: false, retention_days: 0 };
}

/** Deletes everything stored from one attempt. The receipt is the only key to it. */
export function deleteStoredData(receipt: string): Promise<{ deleted: number }> {
  return request("/v1/data/delete", JSON.stringify({ receipt }));
}

export function attest(sessionId: string, payload: object): Promise<{ assurance: string }> {
  return request(`/v1/sessions/${encodeURIComponent(sessionId)}/attest`, JSON.stringify(payload));
}

export async function startCheckpoint<T>(sessionId: string, cp: Checkpoint): Promise<T> {
  const r = await request<{ challenge: T }>(
    `/v1/sessions/${encodeURIComponent(sessionId)}/checkpoints/${cp}/start`,
  );
  return r.challenge;
}

export async function submitCheckpoint(
  sessionId: string,
  cp: Checkpoint,
  payload: object,
  platform: Platform,
  attested: boolean,
): Promise<CheckpointResult> {
  const body = JSON.stringify(payload);
  const headers = attested ? await deviceBindingHeaders(platform, sessionId, body) : {};
  return request<CheckpointResult>(`/v1/sessions/${encodeURIComponent(sessionId)}/checkpoints/${cp}`, body, headers);
}

export function finalize(sessionId: string): Promise<Decision> {
  return request<Decision>(`/v1/sessions/${encodeURIComponent(sessionId)}/finalize`);
}

export function passkeyOptions(sessionId: string): Promise<Record<string, unknown>> {
  return request(`/v1/sessions/${encodeURIComponent(sessionId)}/passkey/options`);
}

export function passkeyVerify(sessionId: string, credential: object): Promise<{ status: string }> {
  return request(`/v1/sessions/${encodeURIComponent(sessionId)}/passkey/verify`, JSON.stringify({ credential }));
}
