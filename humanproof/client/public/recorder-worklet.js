// Collects raw microphone samples on the audio thread and posts them in chunks.
class RecorderProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buf = [];
    this.len = 0;
  }
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (ch) {
      this.buf.push(ch.slice(0));
      this.len += ch.length;
      if (this.len >= 4096) {
        const out = new Float32Array(this.len);
        let o = 0;
        for (const b of this.buf) {
          out.set(b, o);
          o += b.length;
        }
        this.port.postMessage(out, [out.buffer]);
        this.buf = [];
        this.len = 0;
      }
    }
    return true;
  }
}
registerProcessor("recorder", RecorderProcessor);
