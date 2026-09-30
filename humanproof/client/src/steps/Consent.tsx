import { useId, useState } from "react";

export function Consent({ busy, onAgree }: { busy: boolean; onAgree: (research: boolean) => void }) {
  const [agree, setAgree] = useState(false);
  const [research, setResearch] = useState(false);
  const agreeId = useId();
  const researchId = useId();
  return (
    <section className="card">
      <p className="eyebrow">Before we start</p>
      <h1>What this check uses</h1>
      <dl className="facts">
        <dt>Camera</dt>
        <dd>
          Your face is analysed <strong>on this device</strong>. We receive eye and head measurements and four
          small face snapshots, which are checked for signs of deepfakes and then discarded.
        </dd>
        <dt>Microphone</dt>
        <dd>A few seconds of you reading five words. The audio is analysed and discarded, never stored.</dd>
        <dt>Movement</dt>
        <dd>How your pointer or finger moves while you trace a line.</dd>
        <dt>What we keep</dt>
        <dd>
          If you pass: an encrypted voice signature (numbers, not a recording) so you can re-verify later, and
          the result. You can delete it at any time.
        </dd>
      </dl>
      <div className="check">
        <input id={agreeId} type="checkbox" checked={agree} onChange={(e) => setAgree(e.target.checked)} />
        <label htmlFor={agreeId}>I agree to this biometric processing for the purpose of this check.</label>
      </div>
      <div className="check">
        <input id={researchId} type="checkbox" checked={research} onChange={(e) => setResearch(e.target.checked)} />
        <label htmlFor={researchId}>
          Optional: help improve detection by keeping my anonymous measurements (never images or audio).
        </label>
      </div>
      <button className="btn btn--primary" disabled={!agree || busy} onClick={() => onAgree(research)}>
        {busy ? "Starting camera…" : "Agree and continue"}
      </button>
      <p className="fineprint">Your browser will ask for camera and microphone access next.</p>
    </section>
  );
}
