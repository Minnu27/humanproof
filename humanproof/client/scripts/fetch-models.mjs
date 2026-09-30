// Downloads the MediaPipe Face Landmarker model into public/models and pins its
// SHA-256 in models.lock.json. After the first download the hash is enforced:
// a changed file (tampering, or an upstream update you have not reviewed) fails the build.
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";

const MODELS = {
  "face_landmarker.task":
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
};
const LOCK = "models.lock.json";
const lock = existsSync(LOCK) ? JSON.parse(readFileSync(LOCK, "utf8")) : {};
mkdirSync("public/models", { recursive: true });

for (const [name, url] of Object.entries(MODELS)) {
  const dest = `public/models/${name}`;
  let data;
  if (existsSync(dest)) {
    data = readFileSync(dest);
  } else {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`download failed: ${name} (${res.status})`);
    data = Buffer.from(await res.arrayBuffer());
  }
  const digest = createHash("sha256").update(data).digest("hex");
  if (lock[name] && lock[name] !== digest) {
    throw new Error(`SHA-256 mismatch for ${name}: expected ${lock[name]}, got ${digest}`);
  }
  if (!existsSync(dest)) writeFileSync(dest, data);
  if (!lock[name]) {
    lock[name] = digest;
    console.log(`pinned ${name} sha256=${digest} (review and commit ${LOCK})`);
  }
  console.log(`ok ${name}`);
}
writeFileSync(LOCK, JSON.stringify(lock, null, 2) + "\n");
