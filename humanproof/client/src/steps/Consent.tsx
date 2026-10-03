import { useId, useState } from "react";
import type { AttackType, DataPolicy, SharingChoice, TesterLabel } from "../lib/api";

export const ATTACK_TYPES: { value: AttackType; label: string; fakes: string }[] = [
  { value: "replay_video", label: "Video of a real person played to the camera", fakes: "eyes" },
  { value: "photo", label: "Photo held up to the camera", fakes: "eyes and face" },
  { value: "face_swap", label: "Live face-swap or deepfake filter", fakes: "face" },
  { value: "synthetic_face", label: "Fully generated face or avatar", fakes: "eyes and face" },
  { value: "tts_voice", label: "Text-to-speech voice", fakes: "voice" },
  { value: "voice_clone", label: "Cloned voice", fakes: "voice" },
  { value: "audio_replay", label: "Recording of a real voice played back", fakes: "voice" },
  { value: "bot_pointer", label: "Script or bot moving the pointer", fakes: "movement" },
  { value: "scripted_client", label: "Fully scripted, no person at all", fakes: "eyes and movement" },
  { value: "other", label: "Something else (stored, not used for training)", fakes: "nothing specific" },
];

export interface TesterForm {
  code: string;
  participant: string;
  label: "human" | "attack";
  attackType: AttackType;
}

interface Props {
  busy: boolean;
  policy: DataPolicy;
  tester: TesterForm | null; // non-null when the page was opened in tester mode
  onTesterChange: (t: TesterForm) => void;
  onAgree: (sharing: SharingChoice, tester?: TesterLabel) => void;
}

export function Consent({ busy, policy, tester, onTesterChange, onAgree }: Props) {
  const [agree, setAgree] = useState(false);
  const [measurements, setMeasurements] = useState(false);
  const [recordings, setRecordings] = useState(false);
  const id = useId();
  const days = policy.retention_days;

  const participantOk = !tester || /^[A-Za-z0-9_-]{1,40}$/.test(tester.participant);
  const testerOk = !tester || (tester.code.length > 0 && participantOk);

  const submit = () => {
    if (!tester) {
      onAgree({ measurements: measurements || recordings, recordings });
      return;
    }
    onAgree(
      { measurements: true, recordings: true },
      {
        code: tester.code,
        participant: tester.participant,
        label: tester.label,
        ...(tester.label === "attack" ? { attack_type: tester.attackType } : {}),
      },
    );
  };

  return (
    <section className="card">
      <p className="eyebrow">{tester ? "Tester session" : "Before we start"}</p>
      <h1>{tester ? "Record a labelled session" : "What this check uses"}</h1>

      {tester && (
        <>
          <p className="lead">
            This attempt is recorded in full and used to train the detectors, so it must be labelled honestly. It
            never issues a proof token.
          </p>
          {!policy.tester_mode && (
            <p className="notice" role="alert">
              Tester mode is not switched on for this server, so this session cannot be recorded. It needs a
              database, storage enabled and a tester code of at least 12 characters.
            </p>
          )}
          <div className="field">
            <label htmlFor={`${id}-code`}>Tester code</label>
            <input
              id={`${id}-code`}
              type="password"
              autoComplete="off"
              value={tester.code}
              onChange={(e) => onTesterChange({ ...tester, code: e.target.value })}
            />
          </div>
          <div className="field">
            <label htmlFor={`${id}-who`}>Participant ID</label>
            <input
              id={`${id}-who`}
              type="text"
              autoComplete="off"
              autoCapitalize="off"
              spellCheck={false}
              placeholder="p01"
              value={tester.participant}
              aria-invalid={!participantOk}
              aria-describedby={`${id}-who-hint`}
              onChange={(e) => onTesterChange({ ...tester, participant: e.target.value })}
            />
            <span id={`${id}-who-hint`} className="hint">
              A made-up ID, not a name: letters, digits, - and _. Use the same one every time for the same person.
            </span>
          </div>
          <fieldset className="field">
            <legend>What is this session?</legend>
            <div className="check">
              <input
                id={`${id}-human`}
                type="radio"
                name={`${id}-label`}
                checked={tester.label === "human"}
                onChange={() => onTesterChange({ ...tester, label: "human" })}
              />
              <label htmlFor={`${id}-human`}>A real person doing the check normally</label>
            </div>
            <div className="check">
              <input
                id={`${id}-attack`}
                type="radio"
                name={`${id}-label`}
                checked={tester.label === "attack"}
                onChange={() => onTesterChange({ ...tester, label: "attack" })}
              />
              <label htmlFor={`${id}-attack`}>An attempt to fool the check</label>
            </div>
          </fieldset>
          {tester.label === "attack" && (
            <div className="field">
              <label htmlFor={`${id}-type`}>How is it being fooled?</label>
              <select
                id={`${id}-type`}
                value={tester.attackType}
                aria-describedby={`${id}-type-hint`}
                onChange={(e) => onTesterChange({ ...tester, attackType: e.target.value as AttackType })}
              >
                {ATTACK_TYPES.map((a) => (
                  <option key={a.value} value={a.value}>
                    {a.label}
                  </option>
                ))}
              </select>
              <span id={`${id}-type-hint`} className="hint">
                Counts as a fake example for: {ATTACK_TYPES.find((a) => a.value === tester.attackType)?.fakes}. Do
                the other tasks normally.
              </span>
            </div>
          )}
        </>
      )}

      <dl className="facts">
        <dt>Camera</dt>
        <dd>
          Your face is analysed <strong>on this device</strong>. We receive eye and head measurements and four
          small face snapshots, which are checked for signs of deepfakes.
        </dd>
        <dt>Microphone</dt>
        <dd>A few seconds of you reading five words aloud, checked for signs of a synthetic voice.</dd>
        <dt>Movement</dt>
        <dd>How your pointer or finger moves while you trace a line.</dd>
        <dt>What we keep</dt>
        {tester ? (
          <dd>
            Everything from this session: the measurements, the voice recording and the face snapshots, stored
            encrypted for up to {days} days with the label above. You get a receipt at the end that deletes it.
          </dd>
        ) : (
          <dd>
            If you pass: an encrypted voice signature (numbers, not a recording) so you can re-verify later, and
            the result. Snapshots and audio are discarded after the check
            {policy.sharing ? " unless you choose to share them below." : "."}
          </dd>
        )}
      </dl>

      <div className="check">
        <input id={`${id}-agree`} type="checkbox" checked={agree} onChange={(e) => setAgree(e.target.checked)} />
        <label htmlFor={`${id}-agree`}>
          {tester
            ? "I agree to this biometric processing and to this session being recorded for training."
            : "I agree to this biometric processing for the purpose of this check."}
        </label>
      </div>

      {!tester && policy.sharing && (
        <fieldset className="optional">
          <legend>Optional: help improve detection</legend>
          <div className="check">
            <input
              id={`${id}-m`}
              type="checkbox"
              checked={measurements || recordings}
              disabled={recordings}
              onChange={(e) => setMeasurements(e.target.checked)}
            />
            <label htmlFor={`${id}-m`}>Keep the measurements from this attempt (numbers only).</label>
          </div>
          <div className="check">
            <input id={`${id}-r`} type="checkbox" checked={recordings} onChange={(e) => setRecordings(e.target.checked)} />
            <label htmlFor={`${id}-r`}>Also keep my voice recording and face snapshots from this attempt.</label>
          </div>
          <p className="fineprint">
            Stored encrypted for up to {days} days and used only to test and improve this check. You can delete it
            straight after the check. Saying no changes nothing about your result.
          </p>
        </fieldset>
      )}

      <button
        className="btn btn--primary"
        disabled={!agree || busy || !testerOk || (!!tester && !policy.tester_mode)}
        onClick={submit}
      >
        {busy ? "Starting camera…" : tester ? "Agree and record" : "Agree and continue"}
      </button>
      <p className="fineprint">Your browser will ask for camera and microphone access next.</p>
    </section>
  );
}
