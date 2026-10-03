import { useCallback, useEffect, useRef, useState } from "react";
import * as api from "./lib/api";
import { attestDevice, detectPlatform, type Platform } from "./lib/platform";
import { loadFaceLandmarker, openCamera } from "./lib/face";
import type { FaceLandmarker } from "@mediapipe/tasks-vision";
import { Welcome } from "./steps/Welcome";
import { Consent, type TesterForm } from "./steps/Consent";
import { CameraCheck } from "./steps/CameraCheck";
import { GazeStep } from "./steps/GazeStep";
import { MotorStep } from "./steps/MotorStep";
import { VoiceStep } from "./steps/VoiceStep";
import { Result } from "./steps/Result";

type Stage = "welcome" | "consent" | "camera" | "gaze" | "motor" | "voice" | "checking" | "result" | "error";

export interface Ctx {
  platform: Platform;
  session: api.Session;
  attested: boolean;
  video: HTMLVideoElement;
  landmarker: FaceLandmarker;
  tester?: api.TesterLabel; // set when this attempt is a labelled tester session
  retentionDays: number;
}

// Tester mode is opened with ?collect=1. It still needs the private code to do anything.
const TESTER_MODE = new URLSearchParams(window.location.search).get("collect") === "1";

const STEP_LABELS: Record<string, string> = { gaze: "Eyes", motor: "Movement", voice: "Voice" };

export default function App() {
  const [stage, setStage] = useState<Stage>("welcome");
  const [error, setError] = useState<string>("");
  const [decision, setDecision] = useState<api.Decision | null>(null);
  const [busy, setBusy] = useState(false);
  const ctxRef = useRef<Ctx | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const platform = useRef<Platform>(detectPlatform()).current;
  const [policy, setPolicy] = useState<api.DataPolicy>({ sharing: false, tester_mode: false, retention_days: 0 });
  // Kept for the life of the page (never written to storage) so a tester can record several sessions in a row.
  const [tester, setTester] = useState<TesterForm | null>(
    TESTER_MODE ? { code: "", participant: "", label: "human", attackType: "replay_video" } : null,
  );

  useEffect(() => {
    let live = true;
    void api.getDataPolicy().then((p) => live && setPolicy(p));
    return () => {
      live = false;
    };
  }, []);

  const stopCamera = useCallback(() => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
  }, []);

  useEffect(() => stopCamera, [stopCamera]);

  const fail = (e: unknown) => {
    stopCamera();
    const msg =
      e instanceof DOMException && e.name === "NotAllowedError"
        ? "Camera or microphone access was blocked. Allow access in your browser or device settings, then try again."
        : e instanceof api.ApiError
          ? e.message
          : "Something went wrong. Please try again.";
    setError(msg);
    setStage("error");
  };

  const begin = async (sharing: api.SharingChoice, label?: api.TesterLabel) => {
    setBusy(true);
    try {
      const openMedia = async () => {
        const [stream, landmarker] = await Promise.all([openCamera(), loadFaceLandmarker()]);
        streamRef.current = stream;
        return landmarker;
      };
      let session: api.Session;
      let landmarker: FaceLandmarker;
      if (label) {
        // Tester mode: check the code before asking for the camera.
        session = await api.createSession(platform, sharing, label);
        landmarker = await openMedia();
      } else {
        // Normal use: a refused camera prompt should not use up an attempt.
        landmarker = await openMedia();
        session = await api.createSession(platform, sharing);
      }
      const video = videoRef.current!;
      video.srcObject = streamRef.current;
      await video.play();
      let attested = false;
      const att = await attestDevice(platform, session.attestation_challenge);
      if (att) {
        try {
          attested = (await api.attest(session.session_id, att)).assurance === "device_attested";
        } catch {
          attested = false;
        }
      }
      ctxRef.current = { platform, session, attested, video, landmarker, tester: label, retentionDays: policy.retention_days };
      setStage("camera");
    } catch (e) {
      fail(e);
    } finally {
      setBusy(false);
    }
  };

  const next = async (r: api.CheckpointResult) => {
    if (r.next) {
      setStage(r.next);
      return;
    }
    setStage("checking");
    try {
      const d = await api.finalize(ctxRef.current!.session.session_id);
      stopCamera();
      setDecision(d);
      setStage("result");
    } catch (e) {
      fail(e);
    }
  };

  const restart = () => {
    stopCamera();
    ctxRef.current = null;
    setDecision(null);
    setError("");
    setStage("consent");
  };

  const inFlow = stage === "gaze" || stage === "motor" || stage === "voice";

  return (
    <div className={`app ${inFlow ? "app--flow" : ""}`}>
      {!inFlow && (
        <header className="brand">
          <img src="/icon.svg" alt="" width={28} height={28} />
          <span>HumanProof</span>
          {tester && <span className="badge badge--warn brand-tag">Tester mode</span>}
        </header>
      )}
      {inFlow && (
        <ol className="stepper" aria-label="Progress">
          {(["gaze", "motor", "voice"] as const).map((s) => (
            <li key={s} aria-current={stage === s ? "step" : undefined} className={stage === s ? "on" : ""}>
              {STEP_LABELS[s]}
            </li>
          ))}
        </ol>
      )}

      {/* The camera element lives for the whole flow so the stream is shared by steps. */}
      <video
        ref={videoRef}
        className={`camera ${stage === "camera" || stage === "voice" ? "camera--visible" : ""}`}
        playsInline
        muted
        aria-hidden="true"
      />

      <main className="main">
        {stage === "welcome" && <Welcome onStart={() => setStage("consent")} />}
        {stage === "consent" && (
          <Consent busy={busy} policy={policy} tester={tester} onTesterChange={setTester} onAgree={begin} />
        )}
        {stage === "camera" && ctxRef.current && <CameraCheck ctx={ctxRef.current} onReady={() => setStage("gaze")} />}
        {stage === "gaze" && ctxRef.current && <GazeStep ctx={ctxRef.current} onDone={next} onError={fail} />}
        {stage === "motor" && ctxRef.current && <MotorStep ctx={ctxRef.current} onDone={next} onError={fail} />}
        {stage === "voice" && ctxRef.current && <VoiceStep ctx={ctxRef.current} onDone={next} onError={fail} />}
        {stage === "checking" && (
          <section className="card center" aria-live="polite">
            <div className="spinner" aria-hidden="true" />
            <h1>Checking your results</h1>
            <p className="muted">This takes a few seconds.</p>
          </section>
        )}
        {stage === "result" && decision && ctxRef.current && (
          <Result decision={decision} ctx={ctxRef.current} onRetry={restart} />
        )}
        {stage === "error" && (
          <section className="card" role="alert">
            <h1>We couldn't finish the check</h1>
            <p>{error}</p>
            <button className="btn" onClick={restart}>
              Try again
            </button>
          </section>
        )}
      </main>
    </div>
  );
}
