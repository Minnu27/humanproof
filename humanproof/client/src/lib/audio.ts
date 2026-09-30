// Microphone capture -> 16 kHz mono 16-bit PCM WAV.
// Browser voice processing (noise suppression, AGC, echo cancellation) is turned
// off: it reshapes the signal in ways that can look synthetic to the anti-spoof
// model, and the server wants the voice as the microphone heard it.

export interface Recording {
  wav: Uint8Array;
  startedAt: number; // performance.now() when the first sample was captured
  seconds: number;
}

export class Recorder {
  private ctx?: AudioContext;
  private stream?: MediaStream;
  private node?: AudioWorkletNode;
  private chunks: Float32Array[] = [];
  private rate = 48000;
  startedAt = 0;

  async start(): Promise<void> {
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false },
      video: false,
    });
    this.ctx = new AudioContext();
    this.rate = this.ctx.sampleRate;
    await this.ctx.audioWorklet.addModule("/recorder-worklet.js");
    const src = this.ctx.createMediaStreamSource(this.stream);
    this.node = new AudioWorkletNode(this.ctx, "recorder");
    this.node.port.onmessage = (e: MessageEvent<Float32Array>) => {
      if (this.chunks.length === 0) this.startedAt = performance.now() - (e.data.length / this.rate) * 1000;
      this.chunks.push(e.data);
    };
    src.connect(this.node);
  }

  get seconds(): number {
    return this.chunks.reduce((n, c) => n + c.length, 0) / this.rate;
  }

  async stop(): Promise<Recording> {
    this.node?.disconnect();
    this.stream?.getTracks().forEach((t) => t.stop());
    await this.ctx?.close();
    const all = new Float32Array(this.chunks.reduce((n, c) => n + c.length, 0));
    let o = 0;
    for (const c of this.chunks) {
      all.set(c, o);
      o += c.length;
    }
    const pcm = resampleTo16k(all, this.rate);
    return { wav: encodeWav(pcm, 16000), startedAt: this.startedAt, seconds: pcm.length / 16000 };
  }
}

/** Windowed-sinc low-pass + linear interpolation; adequate for speech at 16 kHz. */
export function resampleTo16k(x: Float32Array, rate: number): Float32Array {
  if (rate === 16000) return x;
  const ratio = rate / 16000;
  const taps = 32;
  const fc = 0.9 * (8000 / rate);
  const kernel = new Float32Array(taps * 2 + 1);
  let sum = 0;
  for (let i = -taps; i <= taps; i++) {
    const sinc = i === 0 ? 2 * fc : Math.sin(2 * Math.PI * fc * i) / (Math.PI * i);
    const w = 0.54 - 0.46 * Math.cos((2 * Math.PI * (i + taps)) / (2 * taps));
    kernel[i + taps] = sinc * w;
    sum += sinc * w;
  }
  for (let i = 0; i < kernel.length; i++) kernel[i] /= sum;
  const n = Math.floor(x.length / ratio);
  const out = new Float32Array(n);
  for (let j = 0; j < n; j++) {
    const pos = j * ratio;
    const c = Math.floor(pos);
    let acc = 0;
    for (let k = -taps; k <= taps; k++) {
      const idx = c + k;
      if (idx >= 0 && idx < x.length) acc += x[idx] * kernel[k + taps];
    }
    out[j] = acc;
  }
  return out;
}

export function encodeWav(pcm: Float32Array, rate: number): Uint8Array {
  const buf = new ArrayBuffer(44 + pcm.length * 2);
  const v = new DataView(buf);
  const str = (o: number, s: string) => [...s].forEach((ch, i) => v.setUint8(o + i, ch.charCodeAt(0)));
  str(0, "RIFF");
  v.setUint32(4, 36 + pcm.length * 2, true);
  str(8, "WAVE");
  str(12, "fmt ");
  v.setUint32(16, 16, true);
  v.setUint16(20, 1, true);
  v.setUint16(22, 1, true);
  v.setUint32(24, rate, true);
  v.setUint32(28, rate * 2, true);
  v.setUint16(32, 2, true);
  v.setUint16(34, 16, true);
  str(36, "data");
  v.setUint32(40, pcm.length * 2, true);
  for (let i = 0; i < pcm.length; i++) {
    const s = Math.max(-1, Math.min(1, pcm[i]));
    v.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Uint8Array(buf);
}

export function toBase64(bytes: Uint8Array): string {
  let s = "";
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    s += String.fromCharCode(...bytes.subarray(i, i + chunk));
  }
  return btoa(s);
}
