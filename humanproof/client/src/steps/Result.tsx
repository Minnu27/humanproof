import { useState } from "react";
import type { Ctx } from "../App";
import * as api from "../lib/api";
import { getCapture } from "../lib/diagnostics";
import { bindPasskey, passkeysSupported } from "../lib/passkey";
import { ATTACK_TYPES } from "./Consent";

interface Props {
  decision: api.Decision;
  ctx: Ctx;
  onRetry: () => void;
}

const LABELS: Record<string, string> = { gaze: "Eyes", face: "Face", motor: "Movement", voice: "Voice" };

export function Result({ decision, ctx, onRetry }: Props) {
  const [copied, setCopied] = useState(false);
  const [passkey, setPasskey] = useState<"idle" | "busy" | "done" | "failed">("idle");
  const [diag, setDiag] = useState<"idle" | "copied" | "failed">("idle");
  const [stored, setStored] = useState<"kept" | "busy" | "deleted" | "failed">("kept");
  const pass = decision.decision === "pass";
  const tester = ctx.tester;
  // For a tester deliberately attacking the check, a rejection is the good outcome.
  const attackTest = tester?.label === "attack";
  const good = attackTest ? !pass : pass;
  const receipt = decision.data_stored ? ctx.session.data_receipt : null;

  const deleteStored = async () => {
    if (!receipt) return;
    setStored("busy");
    try {
      await api.deleteStoredData(receipt);
      setStored("deleted");
    } catch {
      setStored("failed");
    }
  };
  const round3 = (_: string, v: unknown) => (typeof v === "number" ? Math.round(v * 1000) / 1000 : v);

  const copyDiagnostics = async () => {
    const payload = {
      decision: { ...decision, attestation_token: undefined, subject_hint: undefined },
      capture: getCapture(),
      screen: { w: window.innerWidth, h: window.innerHeight, dpr: window.devicePixelRatio },
      agent: navigator.userAgent,
    };
    try {
      await navigator.clipboard.writeText(JSON.stringify(payload, round3));
      setDiag("copied");
    } catch {
      setDiag("failed");
    }
  };

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
      <div className={`badge ${good ? "badge--ok" : decision.decision === "step_up" ? "badge--warn" : "badge--bad"}`}>
        {attackTest
          ? pass
            ? "Attack got through"
            : "Attack blocked"
          : pass
            ? "Verified human"
            : decision.decision === "step_up"
              ? "Almost there"
              : "Not verified"}
      </div>
      <h1>
        {attackTest
          ? pass
            ? "This attack was not caught"
            : "This attack was caught"
          : pass
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

      {tester ? (
        <>
          <p>
            Recorded as{" "}
            <strong>
              {tester.label === "human"
                ? "a real person"
                : `an attack: ${ATTACK_TYPES.find((a) => a.value === tester.attack_type)?.label ?? tester.attack_type}`}
            </strong>{" "}
            for participant <strong>{tester.participant}</strong>. The scores above are what an ordinary user would
            have got; tester sessions never issue a proof token.
          </p>
          {decision.reasons.length > 0 && (
            <ul className="reasons" aria-label="What the check noticed">
              {decision.reasons.map((r) => (
                <li key={r}>{r}</li>
              ))}
            </ul>
          )}
          <button className="btn btn--primary" onClick={onRetry}>
            Record another session
          </button>
        </>
      ) : pass ? (
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
          {decision.reasons.length > 0 && (
            <ul className="reasons" aria-label="What the check noticed">
              {decision.reasons.map((r) => (
                <li key={r}>{r}</li>
              ))}
            </ul>
          )}
          <button className="btn btn--primary" onClick={onRetry}>
            Try again
          </button>
        </>
      )}
      {receipt && (
        <div className="stored" aria-live="polite">
          {stored === "deleted" ? (
            <p className="ok">Deleted. Nothing from this attempt is stored any more.</p>
          ) : (
            <>
              <h2>Data kept from this attempt</h2>
              <p className="muted">
                {ctx.session.storing === "measurements_and_media"
                  ? "The measurements, your voice recording and the face snapshots"
                  : "The measurements (numbers only, no audio or images)"}{" "}
                are stored encrypted for up to {ctx.retentionDays} days. This receipt is the only way to delete them
                later, so keep it if you might want to:
              </p>
              <code className="receipt">{receipt}</code>
              <div className="row">
                <button className="btn" onClick={deleteStored} disabled={stored === "busy"}>
                  {stored === "busy" ? "Deleting…" : "Delete my data from this attempt"}
                </button>
              </div>
              {stored === "failed" && (
                <p className="muted" role="alert">
                  The data could not be deleted just now. Try again in a moment.
                </p>
              )}
            </>
          )}
        </div>
      )}
      {decision.debug && (
        <details className="diag">
          <summary>Diagnostics (demo mode)</summary>
          <p className="fineprint">
            The measurements behind each score, and this attempt's raw numbers (no images or audio). Copy them to
            share with whoever is tuning the checks.
          </p>
          <button className="btn" onClick={copyDiagnostics}>
            {diag === "copied" ? "Copied" : "Copy diagnostics"}
          </button>
          {diag === "failed" && <p className="muted">Copying was blocked. Select the text below instead.</p>}
          <pre>{JSON.stringify(decision.debug, round3, 1)}</pre>
        </details>
      )}
    </section>
  );
}
