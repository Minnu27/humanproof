import { useRef, useState } from "react";
import type { Ctx } from "../App";
import * as api from "../lib/api";
import { analyse, cropFace, onFrames } from "../lib/face";
import { useOnceJob } from "../lib/useOnceJob";

/** Same semantics as backend challenges.GazeChallenge.target_at(). */
export function targetAt(kfs: api.GazeKeyframe[], t: number): [number, number] {
  let cur = kfs[0];
  for (const k of kfs) {
    if (k.start <= t) cur = k;
    else break;
  }
  if (t < cur.end) {
    const a = (t - cur.start) / (cur.end - cur.start);
    return [cur.x0 + a * (cur.x1 - cur.x0), cur.y0 + a * (cur.y1 - cur.y0)];
  }
  return [cur.x1, cur.y1];
}

const r3 = (v: number) => Math.round(v * 1000) / 1000;

interface Props {
  ctx: Ctx;
  onDone: (r: api.CheckpointResult) => void;
  onError: (e: unknown) => void;
}

export function GazeStep({ ctx, onDone, onError }: Props) {
  const [count, setCount] = useState(3);
  const [phase, setPhase] = useState<"ready" | "run" | "send">("ready");
  const dotRef = useRef<HTMLDivElement>(null);
  const areaRef = useRef<HTMLDivElement>(null);
  useOnceJob(async ({ alive, onCleanup }) => {
    try {
      const ch = await api.startCheckpoint<api.GazeChallenge>(ctx.session.session_id, "gaze");
      for (let i = 3; i > 0; i--) {
        setCount(i);
        await new Promise((r) => setTimeout(r, 700));
      }
      if (!alive()) return;
      setPhase("run");
      const frames: object[] = [];
      const crops: string[] = [];
      const pending = [...ch.capture_ms];
      const t0 = performance.now();

      let raf = 0;
      const draw = () => {
        const t = performance.now() - t0;
        const [x, y] = targetAt(ch.keyframes, t);
        const area = areaRef.current;
        if (dotRef.current && area) {
          dotRef.current.style.transform = `translate(${x * area.clientWidth}px, ${y * area.clientHeight}px)`;
        }
        if (t < ch.duration_ms + 200) raf = requestAnimationFrame(draw);
      };
      raf = requestAnimationFrame(draw);
      onCleanup(() => cancelAnimationFrame(raf));

      let stopFrames = () => {};
      await new Promise<void>((resolve) => {
        stopFrames = onFrames(ctx.video, (ft) => {
          const t = ft - t0;
          if (t < 0) return;
          const s = analyse(ctx.landmarker, ctx.video, ft);
          frames.push({
            t: r3(t), ix: r3(s.ix), iy: r3(s.iy), yaw: r3(s.yaw), pitch: r3(s.pitch), roll: r3(s.roll),
            ear: r3(s.ear), mouth: r3(s.mouth), face: s.face,
          });
          if (pending.length && t >= pending[0] && s.landmarks) {
            pending.shift();
            const c = cropFace(ctx.video, s.landmarks);
            if (c) crops.push(c);
          }
          if (t > ch.duration_ms + 150) resolve();
        });
        onCleanup(stopFrames);
      });
      stopFrames();
      if (!alive()) return;
      setPhase("send");
      const area = areaRef.current!;
      const result = await api.submitCheckpoint(
        ctx.session.session_id,
        "gaze",
        {
          frames: frames.slice(0, 1500),
          face_crops: crops.slice(0, 4),
          viewport: { w: Math.max(200, area.clientWidth), h: Math.max(200, area.clientHeight) },
        },
        ctx.platform,
        ctx.attested,
      );
      if (alive()) onDone(result);
    } catch (e) {
      if (alive()) onError(e);
    }
  });

  return (
    <section className="stage" aria-live="polite">
      <div className="gaze-area" ref={areaRef}>
        {phase === "run" && <div className="gaze-dot" ref={dotRef} />}
        {phase === "ready" && (
          <div className="stage-msg">
            <h1>Follow the dot with your eyes</h1>
            <p>Keep your head still. Starting in {count}…</p>
          </div>
        )}
        {phase === "send" && (
          <div className="stage-msg">
            <div className="spinner" aria-hidden="true" />
            <p>Checking…</p>
          </div>
        )}
      </div>
    </section>
  );
}
