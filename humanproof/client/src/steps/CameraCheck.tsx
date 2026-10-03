import { useEffect, useRef, useState } from "react";
import type { Ctx } from "../App";
import { analyse, onFrames } from "../lib/face";

/** Waits until a single, well-lit, centred face is steadily visible. */
export function CameraCheck({ ctx, onReady }: { ctx: Ctx; onReady: () => void }) {
  const [hint, setHint] = useState("Looking for your face…");
  const [ok, setOk] = useState(false);
  const good = useRef(0);

  useEffect(() => {
    const stop = onFrames(ctx.video, (t) => {
      const s = analyse(ctx.landmarker, ctx.video, t);
      if (!s.face || !s.landmarks) {
        good.current = 0;
        setHint("Looking for your face…");
        setOk(false);
        return;
      }
      const xs = s.landmarks.map((p) => p.x);
      const width = Math.max(...xs) - Math.min(...xs);
      const cx = (Math.max(...xs) + Math.min(...xs)) / 2;
      if (width < 0.22) setHint("Move a little closer");
      else if (width > 0.7) setHint("Move back a little");
      else if (Math.abs(cx - 0.5) > 0.18) setHint("Centre your face");
      else if (Math.abs(s.yaw) > 20 || Math.abs(s.pitch) > 20) setHint("Face the screen");
      else {
        good.current += 1;
        setHint("Perfect. Hold still…");
      }
      if (good.current > 20) setOk(true);
    });
    return stop;
  }, [ctx]);

  return (
    <section className="card center">
      <h1>Position your face</h1>
      <p className="muted" aria-live="polite">
        {hint}
      </p>
      <p className="fineprint">
        {ctx.attested ? "Device verified." : "Running in browser mode."} Next, a dot will jump around the
        screen. Just look at it each time it moves.
      </p>
      <button className="btn btn--primary" disabled={!ok} onClick={onReady}>
        I'm ready
      </button>
    </section>
  );
}
