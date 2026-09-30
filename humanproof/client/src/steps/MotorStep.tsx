import { useCallback, useEffect, useRef, useState } from "react";
import type { Ctx } from "../App";
import * as api from "../lib/api";
import { useOnceJob } from "../lib/useOnceJob";

interface Props {
  ctx: Ctx;
  onDone: (r: api.CheckpointResult) => void;
  onError: (e: unknown) => void;
}

type Sample = { t: number; x: number; y: number; pressure: number };
const r2 = (v: number) => Math.round(v * 100) / 100;
const KEY_STEP = 8;

export function MotorStep({ ctx, onDone, onError }: Props) {
  const [challenge, setChallenge] = useState<api.MotorChallenge | null>(null);
  const [msg, setMsg] = useState("Press on the green dot and trace the line to the flag without lifting.");
  const [sending, setSending] = useState(false);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const samples = useRef<Sample[]>([]);
  const motion = useRef<{ t: number; ax: number; ay: number; az: number }[]>([]);
  const tracking = useRef(false);
  const pointerType = useRef<"mouse" | "touch" | "pen" | "keyboard">("mouse");
  const t0 = useRef(0);
  const keyPos = useRef<[number, number] | null>(null);
  const alive = useRef(true);

  useOnceJob(async (h) => {
    try {
      const ch = await api.startCheckpoint<api.MotorChallenge>(ctx.session.session_id, "motor");
      if (h.alive()) setChallenge(ch);
    } catch (e) {
      if (h.alive()) onError(e);
    }
  });
  useEffect(() => () => void (alive.current = false), []);

  const pts = useCallback((): [number, number][] => {
    const c = canvasRef.current;
    if (!c || !challenge) return [];
    return challenge.polyline.map(([x, y]) => [x * c.clientWidth, y * c.clientHeight]);
  }, [challenge]);

  const draw = useCallback(() => {
    const c = canvasRef.current;
    if (!c || !challenge) return;
    const dpr = window.devicePixelRatio || 1;
    c.width = c.clientWidth * dpr;
    c.height = c.clientHeight * dpr;
    const g = c.getContext("2d")!;
    g.scale(dpr, dpr);
    const css = getComputedStyle(document.documentElement);
    const p = pts();
    g.lineCap = "round";
    g.lineJoin = "round";
    g.strokeStyle = css.getPropertyValue("--track").trim() || "#c9d6cf";
    g.lineWidth = 22;
    g.beginPath();
    p.forEach(([x, y], i) => (i ? g.lineTo(x, y) : g.moveTo(x, y)));
    g.stroke();
    if (samples.current.length > 1) {
      g.strokeStyle = css.getPropertyValue("--ink-trace").trim() || "#1f5c4a";
      g.lineWidth = 4;
      g.beginPath();
      samples.current.forEach((s, i) => (i ? g.lineTo(s.x, s.y) : g.moveTo(s.x, s.y)));
      g.stroke();
    }
    const [sx, sy] = p[0];
    const [ex, ey] = p[p.length - 1];
    g.fillStyle = "#2e9e6f";
    g.beginPath();
    g.arc(sx, sy, 16, 0, Math.PI * 2);
    g.fill();
    g.fillStyle = css.getPropertyValue("--ink").trim() || "#10231c";
    g.beginPath();
    g.moveTo(ex - 2, ey + 16);
    g.lineTo(ex - 2, ey - 18);
    g.lineTo(ex + 18, ey - 10);
    g.lineTo(ex - 2, ey - 2);
    g.fill();
    if (keyPos.current) {
      g.fillStyle = "#e0a400";
      g.beginPath();
      g.arc(keyPos.current[0], keyPos.current[1], 8, 0, Math.PI * 2);
      g.fill();
    }
  }, [challenge, pts]);

  useEffect(() => {
    draw();
    const onResize = () => draw();
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, [draw]);

  useEffect(() => {
    const onMotion = (e: DeviceMotionEvent) => {
      const a = e.accelerationIncludingGravity;
      if (!tracking.current || !a || a.x == null || a.y == null || a.z == null) return;
      if (motion.current.length < 3000) {
        const clip = (v: number) => Math.max(-99, Math.min(99, v));
        motion.current.push({ t: r2(e.timeStamp - t0.current), ax: clip(a.x), ay: clip(a.y), az: clip(a.z) });
      }
    };
    window.addEventListener("devicemotion", onMotion);
    return () => window.removeEventListener("devicemotion", onMotion);
  }, []);

  const local = (e: { clientX: number; clientY: number }) => {
    const r = canvasRef.current!.getBoundingClientRect();
    return [e.clientX - r.left, e.clientY - r.top] as const;
  };

  const push = (t: number, x: number, y: number, pressure = 0.5) => {
    if (samples.current.length < 4000) {
      samples.current.push({ t: r2(Math.max(0, t - t0.current)), x: r2(x), y: r2(y), pressure: r2(pressure) });
    }
  };

  const nearEnd = (x: number, y: number) => {
    const p = pts();
    const [ex, ey] = p[p.length - 1];
    return Math.hypot(x - ex, y - ey) < 40;
  };

  const submit = async () => {
    const c = canvasRef.current!;
    setSending(true);
    setMsg("Checking…");
    try {
      const r = await api.submitCheckpoint(
        ctx.session.session_id,
        "motor",
        {
          pointer_type: pointerType.current,
          samples: samples.current,
          viewport: { w: Math.max(200, Math.round(c.clientWidth)), h: Math.max(200, Math.round(c.clientHeight)) },
          device_motion: motion.current,
        },
        ctx.platform,
        ctx.attested,
      );
      if (alive.current) onDone(r);
    } catch (e) {
      if (alive.current) onError(e);
    }
  };

  const reset = (why: string) => {
    tracking.current = false;
    samples.current = [];
    motion.current = [];
    keyPos.current = null;
    setMsg(why);
    draw();
  };

  const onDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (sending) return;
    const [x, y] = local(e);
    const [sx, sy] = pts()[0];
    if (Math.hypot(x - sx, y - sy) > 36) {
      setMsg("Start on the green dot.");
      return;
    }
    e.currentTarget.setPointerCapture(e.pointerId);
    tracking.current = true;
    samples.current = [];
    motion.current = [];
    pointerType.current = (e.pointerType as "mouse" | "touch" | "pen") || "mouse";
    t0.current = e.timeStamp;
    push(e.timeStamp, x, y, e.pressure);
    setMsg("Keep going to the flag…");
  };

  const onMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (!tracking.current) return;
    const events = e.nativeEvent.getCoalescedEvents?.() ?? [e.nativeEvent];
    for (const ev of events) {
      const [x, y] = local(ev);
      push(ev.timeStamp, x, y, ev.pressure);
    }
    draw();
  };

  const onUp = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (!tracking.current) return;
    tracking.current = false;
    const [x, y] = local(e);
    push(e.timeStamp, x, y, e.pressure);
    if (!nearEnd(x, y) || samples.current.length < 20) {
      reset("You lifted before reaching the flag. Start again from the green dot.");
      return;
    }
    void submit();
  };

  // Keyboard alternative (accessibility): arrow keys move a marker along; Enter finishes.
  const onKey = (e: React.KeyboardEvent<HTMLCanvasElement>) => {
    if (sending || !challenge) return;
    const dirs: Record<string, [number, number]> = {
      ArrowLeft: [-KEY_STEP, 0], ArrowRight: [KEY_STEP, 0], ArrowUp: [0, -KEY_STEP], ArrowDown: [0, KEY_STEP],
    };
    if (e.key === "Enter" && keyPos.current) {
      e.preventDefault();
      if (nearEnd(...keyPos.current) && samples.current.length >= 20) void submit();
      else setMsg("Move the yellow marker to the flag, then press Enter.");
      return;
    }
    const d = dirs[e.key];
    if (!d) return;
    e.preventDefault();
    if (!keyPos.current) {
      keyPos.current = pts()[0];
      pointerType.current = "keyboard";
      samples.current = [];
      t0.current = e.timeStamp;
      push(e.timeStamp, ...keyPos.current);
    }
    keyPos.current = [keyPos.current[0] + d[0], keyPos.current[1] + d[1]];
    push(e.timeStamp, ...keyPos.current);
    draw();
  };

  return (
    <section className="stage">
      <div className="stage-head">
        <h1>Trace the line</h1>
        <p aria-live="polite">{msg}</p>
      </div>
      <canvas
        ref={canvasRef}
        className="trace"
        tabIndex={0}
        aria-label="Tracing area. Use a mouse or finger, or arrow keys to move a marker from the green dot to the flag and press Enter."
        onPointerDown={onDown}
        onPointerMove={onMove}
        onPointerUp={onUp}
        onPointerCancel={() => reset("Tracing was interrupted. Start again from the green dot.")}
        onKeyDown={onKey}
      />
    </section>
  );
}
