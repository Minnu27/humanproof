// On-device face tracking with MediaPipe Face Landmarker (478 landmarks incl. irises).
// Only derived numbers leave the device (iris offsets, head angles, eye/mouth
// openness) plus four small face crops at server-chosen moments. No video is sent.
import { FaceLandmarker, FilesetResolver, type NormalizedLandmark } from "@mediapipe/tasks-vision";

export interface FaceSignals {
  face: boolean;
  ix: number;
  iy: number;
  yaw: number;
  pitch: number;
  roll: number;
  ear: number;
  mouth: number;
  landmarks?: NormalizedLandmark[];
}

let landmarkerPromise: Promise<FaceLandmarker> | null = null;

export function loadFaceLandmarker(): Promise<FaceLandmarker> {
  landmarkerPromise ??= (async () => {
    const fileset = await FilesetResolver.forVisionTasks("/mediapipe/wasm");
    const make = (delegate: "GPU" | "CPU") =>
      FaceLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: "/models/face_landmarker.task", delegate },
        runningMode: "VIDEO",
        numFaces: 1,
        outputFacialTransformationMatrixes: true,
        outputFaceBlendshapes: false,
      });
    try {
      return await make("GPU");
    } catch {
      return await make("CPU");
    }
  })();
  return landmarkerPromise;
}

const dist = (a: NormalizedLandmark, b: NormalizedLandmark) => Math.hypot(a.x - b.x, a.y - b.y);
const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

// Eye landmark indices (MediaPipe face mesh). Corners, upper/lower lids, iris centre.
const EYES = [
  { c1: 33, c2: 133, top: 159, bottom: 145, iris: 468, v: [160, 144, 158, 153] },
  { c1: 362, c2: 263, top: 386, bottom: 374, iris: 473, v: [385, 380, 387, 373] },
];

function eyeOffsets(lm: NormalizedLandmark[]) {
  let ix = 0;
  let iy = 0;
  let ear = 0;
  for (const e of EYES) {
    const [a, b] = lm[e.c1].x < lm[e.c2].x ? [lm[e.c1], lm[e.c2]] : [lm[e.c2], lm[e.c1]];
    const iris = lm[e.iris];
    ix += ((iris.x - a.x) / (b.x - a.x + 1e-6) - 0.5) * 2;
    iy += ((iris.y - lm[e.top].y) / (lm[e.bottom].y - lm[e.top].y + 1e-6) - 0.5) * 2;
    ear += (dist(lm[e.v[0]], lm[e.v[1]]) + dist(lm[e.v[2]], lm[e.v[3]])) / (2 * dist(lm[e.c1], lm[e.c2]) + 1e-6);
  }
  return { ix: clamp(ix / 2, -1.99, 1.99), iy: clamp(iy / 2, -1.99, 1.99), ear: clamp(ear / 2, 0, 1.49) };
}

function headAngles(m: number[]) {
  // 4x4 column-major rigid transform -> Euler angles (degrees).
  const r = (i: number, j: number) => m[j * 4 + i];
  const deg = 180 / Math.PI;
  const pitch = Math.atan2(r(2, 1), r(2, 2)) * deg;
  const yaw = Math.asin(clamp(-r(2, 0), -1, 1)) * deg;
  const roll = Math.atan2(r(1, 0), r(0, 0)) * deg;
  return { yaw: clamp(yaw, -89, 89), pitch: clamp(pitch, -89, 89), roll: clamp(roll, -89, 89) };
}

let lastTs = 0;

export function analyse(lm: FaceLandmarker, video: HTMLVideoElement, tsMs: number): FaceSignals {
  // MediaPipe requires strictly increasing timestamps across every call on this instance.
  const ts = Math.max(tsMs, lastTs + 1);
  lastTs = ts;
  const res = lm.detectForVideo(video, ts);
  const face = res.faceLandmarks?.[0];
  if (!face || face.length < 478) {
    return { face: false, ix: 0, iy: 0, yaw: 0, pitch: 0, roll: 0, ear: 0, mouth: 0 };
  }
  const m = res.facialTransformationMatrixes?.[0]?.data;
  const angles = m ? headAngles(m) : { yaw: 0, pitch: 0, roll: 0 };
  const mouth = clamp(dist(face[13], face[14]) / (dist(face[61], face[291]) + 1e-6), 0, 2.9);
  return { face: true, ...eyeOffsets(face), ...angles, mouth, landmarks: face };
}

/** Square face crop with 25% margin, 224 px JPEG (same geometry as ml/face/extract_faces.py). */
export function cropFace(video: HTMLVideoElement, lm: NormalizedLandmark[]): string | null {
  const w = video.videoWidth;
  const h = video.videoHeight;
  let minX = 1, minY = 1, maxX = 0, maxY = 0;
  for (const p of lm) {
    minX = Math.min(minX, p.x);
    maxX = Math.max(maxX, p.x);
    minY = Math.min(minY, p.y);
    maxY = Math.max(maxY, p.y);
  }
  const cx = ((minX + maxX) / 2) * w;
  const cy = ((minY + maxY) / 2) * h;
  const side = Math.max((maxX - minX) * w, (maxY - minY) * h) * 1.5;
  const x0 = Math.max(0, cx - side / 2);
  const y0 = Math.max(0, cy - side / 2);
  const x1 = Math.min(w, cx + side / 2);
  const y1 = Math.min(h, cy + side / 2);
  if (x1 - x0 < 64 || y1 - y0 < 64) return null;
  const canvas = document.createElement("canvas");
  canvas.width = 224;
  canvas.height = 224;
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;
  ctx.drawImage(video, x0, y0, x1 - x0, y1 - y0, 0, 0, 224, 224);
  const url = canvas.toDataURL("image/jpeg", 0.85);
  return url.slice(url.indexOf(",") + 1);
}

export async function openCamera(): Promise<MediaStream> {
  return navigator.mediaDevices.getUserMedia({
    video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 } },
    audio: false,
  });
}

/** Calls cb for each new camera frame with a timestamp on the performance.now() clock. */
export function onFrames(video: HTMLVideoElement, cb: (t: number) => void): () => void {
  let stopped = false;
  const v = video as HTMLVideoElement & {
    requestVideoFrameCallback?: (f: (now: number, meta: { captureTime?: number }) => void) => number;
  };
  if (v.requestVideoFrameCallback) {
    const loop = (now: number, meta: { captureTime?: number }) => {
      if (stopped) return;
      cb(meta.captureTime ?? now);
      v.requestVideoFrameCallback!(loop);
    };
    v.requestVideoFrameCallback(loop);
  } else {
    let last = -1;
    const loop = () => {
      if (stopped) return;
      if (video.currentTime !== last) {
        last = video.currentTime;
        cb(performance.now());
      }
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  }
  return () => {
    stopped = true;
  };
}
