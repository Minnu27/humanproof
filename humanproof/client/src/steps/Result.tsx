import { useState } from "react";
import type { Ctx } from "../App";
import * as api from "../lib/api";
import { bindPasskey, passkeysSupported } from "../lib/passkey";

interface Props {
  decision: api.Decision;
  ctx: Ctx;
  onRetry: () => void;
}

const LABELS: Record<string, string> = { gaze: "Eyes", face: "Face", motor: "Movement", voice: "Voice" };

export function Result({ decision, ctx, onRetry }: Props) {
  const [copied, setCopied] = useState(false);
  const [passkey, setPasskey] = useState<"idle" | "busy" | "done" | "failed">("idle");
  const pass = decision.decision === "pass";

  const copy = async () => {
    if (!decision.attestation_token) return;
    await navigator.clipboard.writeText(decision.attestation_token);
    setCopied(true);
  };

  const handOff = () => {
    // If a relying party opened this window, return the token only to that party's origin.
    const rp = api.RELYING_PARTY;
    if (window.opener && decision.attestation_token && rp.includes(".")) {
      window.opener.postMessage({ type: "humanproof:token", token: decision.attestation_token }, `https://${rp}`);
      window.close();
    }
  };

  const addPasskey = async () => {
    setPasskey("busy");
    try {
      await bindPasskey(ctx.session.session_id);
      setPasskey("done");
    } catch {
      setPasskey("failed");
    }
  };

  return (
    <section className="card">
      <div className={`badge ${pass ? "badge--ok" : decision.decision === "step_up" ? "badge--warn" : "badge--bad"}`}>
        {pass ? "Verified human" : decision.decision === "step_up" ? "Almost there" : "Not verified"}
      </div>
      <h1>
        {pass
          ? "You're verified"
          : decision.decision === "step_up"
            ? "We need one more try"
            : "We couldn't verify you this time"}
      </h1>
      <ul className="scores" aria-label="Checkpoint results">
        {Object.entries(decision.checkpoints).map(([k, v]) => (
          <li key={k}>
            <span>{LABELS[k] ?? k}</span>
            <meter min={0} max={1} low={0.35} high={0.6} optimum={1} value={v} aria-label={`${LABELS[k] ?? k} score`} />
          </li>
        ))}
      </ul>

      {pass ? (
        <>
          <p className="muted">
            Assurance: {decision.assurance === "device_attested" ? "verified device" : "browser"}. Your proof is valid
            for 15 minutes.
          </p>
          <div className="row">
            {window.opener ? (
              <button className="btn btn--primary" onClick={handOff}>
                Return to the app
              </button>
            ) : (
              <button className="btn btn--primary" onClick={copy}>
                {copied ? "Copied" : "Copy proof token"}
              </button>
            )}
            {passkeysSupported() && passkey !== "done" && (
              <button className="btn" onClick={addPasskey} disabled={passkey === "busy"}>
                {passkey === "busy" ? "Waiting for passkey…" : "Save a passkey for quick re-checks"}
              </button>
            )}
          </div>
          {passkey === "done" && <p className="ok">Passkey saved. Next time, one tap re-verifies you.</p>}
          {passkey === "failed" && <p className="muted">The passkey wasn't saved. You can skip this.</p>}
        </>
      ) : (
        <>
          <p>
            {decision.decision === "step_up"
              ? "Your result was close. Good light, a steady head and a quiet room help."
              : "Please try again in good light, facing the camera, and speaking in your normal voice."}
          </p>
          <button className="btn btn--primary" onClick={onRetry}>
            Try again
          </button>
        </>
      )}
    </section>
  );
}
