import { useEffect, useRef, useState } from "react";
import type { Ctx } from "../App";
import * as api from "../lib/api";
import { Recorder, toBase64 } from "../lib/audio";
import { analyse, onFrames } from "../lib/face";
import { useOnceJob } from "../lib/useOnceJob";

interface Props {
  ctx: Ctx;
  onDone: (r: api.CheckpointResult) => void;
  onError: (e: unknown) => void;
}

const MIN_S = 1.6;

export function VoiceStep({ ctx, onDone, onError }: Props) {
  const [words, setWords] = useState<string[] | null>(null);
  const [maxMs, setMaxMs] = useState(8000);
  const [state, setState] = useState<"idle" | "rec" | "send">("idle");
  const [elapsed, setElapsed] = useState(0);
  const rec = useRef<Recorder | null>(null);
  const mouth = useRef<{ t: number; mouth: number; face: boolean }[]>([]);
  const tl0 = useRef(0);
  const stopFrames = useRef<() => void>(() => {});
  const timer = useRef(0);
  const alive = useRef(true);

  useOnceJob(async (h) => {
    try {
      const ch = await api.startCheckpoint<api.VoiceChallenge>(ctx.session.session_id, "voice");
      if (!h.alive()) return;
      setWords(ch.words);
      setMaxMs(ch.max_duration_ms);
    } catch (e) {
      if (h.alive()) onError(e);
    }
  });

  useEffect(
    () => () => {
      alive.current = false;
      stopFrames.current();
      window.clearInterval(timer.current);
    },
    [],
  );

  const start = async () => {
    try {
      const r = new Recorder();
      mouth.current = [];
      tl0.current = performance.now();
      stopFrames.current = onFrames(ctx.video, (ft) => {
        const s = analyse(ctx.landmarker, ctx.video, ft);
        const t = ft - tl0.current;
        if (t >= 0 && mouth.current.length < 1200) {
          mouth.current.push({ t: Math.round(t * 100) / 100, mouth: Math.round(s.mouth * 1000) / 1000, face: s.face });
        }
      });
      await r.start();
      rec.current = r;
      setState("rec");
      timer.current = window.setInterval(() => {
        const s = rec.current?.seconds ?? 0;
        setElapsed(s);
        if (s * 1000 >= maxMs - 250) void finish();
      }, 100);
    } catch (e) {
      onError(e);
    }
  };

  const finish = async () => {
    const r = rec.current;
    if (!r) return;
    rec.current = null;
    window.clearInterval(timer.current);
    stopFrames.current();
    setState("send");
    try {
      const out = await r.stop();
      if (out.seconds < MIN_S) {
        setState("idle");
        return;
      }
      const result = await api.submitCheckpoint(
        ctx.session.session_id,
        "voice",
        {
          audio_wav_b64: toBase64(out.wav),
          audio_offset_ms: Math.max(-2000, Math.min(2000, Math.round(out.startedAt - tl0.current))),
          mouth_frames: mouth.current,
        },
        ctx.platform,
        ctx.attested,
      );
      if (alive.current) onDone(result);
    } catch (e) {
      if (alive.current) onError(e);
    }
  };

  return (
    <section className="stage">
      <div className="stage-head">
        <h1>Read these words aloud</h1>
        <p>Speak naturally, in your normal voice, facing the camera.</p>
      </div>
      <p className="words" aria-live="polite">
        {words ? words.join("  ·  ") : "…"}
      </p>
      <div className="voice-controls">
        {state === "idle" && (
          <button className="btn btn--primary btn--big" disabled={!words} onClick={start}>
            Start recording
          </button>
        )}
        {state === "rec" && (
          <>
            <div className="rec-indicator" role="status">
              <span className="rec-dot" aria-hidden="true" /> Recording {elapsed.toFixed(1)} s
            </div>
            <button className="btn" disabled={elapsed < MIN_S} onClick={finish}>
              Done
            </button>
          </>
        )}
        {state === "send" && (
          <div className="rec-indicator" role="status">
            <span className="spinner spinner--small" aria-hidden="true" /> Checking…
          </div>
        )}
      </div>
    </section>
  );
}
