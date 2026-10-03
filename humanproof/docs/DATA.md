# Collecting data and retraining on it

The service can store verification attempts in a database and retrain its models
on them. This page covers what is stored, how it gets a trustworthy label, how
retraining decides whether a new model is better, and how to switch it all on.

## The short version

1. Attach a Postgres database and set four variables (below). Storage is off until you do.
2. Ordinary users are stored only if they tick an optional box. Their data is
   **unlabelled** and is never used to train anything.
3. Trusted testers open the app with `?collect=1`, enter a private code, and say
   whether the session is a real person or a specific kind of attack. These
   **labelled** sessions are the training data.
4. Once a week GitHub exports the labelled sessions, retrains the eye and movement
   models, and opens a pull request only if a new model beats the current one on
   people it never trained on. Nothing changes in production until you merge it.
5. The voice and face models are deep networks: they are trained on a GPU with
   `ml/colab_train.ipynb`, which pulls the same data.

## Why ordinary users' data is not used for training

The only "label" an ordinary attempt has is the system's own verdict. Training on
that teaches the model to agree with itself: an attack that slipped through once
would be learned as "human" forever. So unlabelled data is kept for monitoring
(pass rates, score drift) and labelled tester sessions do the teaching.

## What is stored

| | Ordinary user, no box ticked | "Keep the measurements" | "Also keep my recordings" | Tester session |
|---|---|---|---|---|
| Eye, pointer and mouth measurements, challenge, scores | no | yes | yes | yes |
| Voice recording (WAV, up to 8 s) | no | no | yes | yes |
| Four face snapshots (224 px JPEG) | no | no | yes | yes |
| Label | n/a | unlabelled | unlabelled | human / attack type |

Every sample is encrypted (AES-256-GCM) with a key that is not in the database and
is bound to its own row, so a database dump alone is unreadable. Rows carry no
session ID, IP address or name. Retention is `HP_DATA_RETENTION_DAYS` (default 365);
expired rows are deleted automatically. An attempt that is abandoned part-way is
deleted within about an hour, because the person never reached the receipt.

**Deleting:** the result screen shows a receipt code and a "Delete my data from
this attempt" button. The receipt is the only link between a person and their rows
(only its hash is stored), so it also works later: `POST /api/v1/data/delete
{"receipt": "..."}`. Deleting with a passkey removes that person's samples too.

## Tester sessions

Open the app with `?collect=1` (for example `https://your-app.vercel.app/?collect=1`).

* **Tester code:** the value of `HP_COLLECTION_KEY`. Share it only with people you
  trust to label honestly; anyone who has it can add training data.
* **Participant ID:** a made-up ID such as `p07`, the same one every time for the
  same person. It is how the data is split so nobody is in both training and test.
* **Real person or attack**, and for attacks, which kind. Labels are per check:

| Attack type | Counted as a fake example for |
|---|---|
| Video of a real person played to the camera | eyes |
| Photo held up to the camera | eyes, face |
| Live face-swap / deepfake filter | face |
| Fully generated face or avatar | eyes, face |
| Text-to-speech, cloned voice, replayed voice recording | voice |
| Script or bot moving the pointer | movement |
| Fully scripted client, no person | eyes, movement (its image and audio could be anything, so they are not used) |
| Something else | nothing (stored, not trained on) |

  Checks the attack does not touch are stored as `unknown` and left out of training:
  someone testing a cloned voice is still moving a real mouse.

A tester session shows the verdict an ordinary user would have got, but never
issues a proof token and never creates a stored identity.

**What to collect.** Variety matters more than volume: different people, laptops,
phones, lighting, glasses, accents. For attacks, use the tools a real attacker
would (OBS virtual camera with a recorded video, a face-swap app, ElevenLabs or
similar, an automation script). Retraining starts once a check has at least 40
(eyes) or 30 (movement) real-human sessions from at least 5 people, 8 of them
from held-out people.

## Switching it on (Vercel)

1. **Database:** Vercel project → Storage → Create → Neon (Postgres) → connect it
   to the project. That sets `DATABASE_URL`, which the backend picks up. Tables are
   created on first use.
2. **Environment variables** (Project → Settings → Environment Variables):

| Variable | Value |
|---|---|
| `HP_MASTER_SECRET` | A long random secret: 32 random bytes in base64, or any random string of 32+ characters. Every key is derived from it. **Back it up**: without it the stored data cannot be read and issued proofs cannot be checked. |
| `HP_DATA_COLLECTION_ENABLED` | `true` |
| `HP_COLLECTION_KEY` | the tester code, 12+ characters |
| `HP_DATA_RETENTION_DAYS` | optional, default `365` |

   To generate the secret, open **Git Bash** (installed with Git for Windows; any
   macOS/Linux terminal works too) and run `openssl rand -base64 32`. A password
   manager's generator set to 40+ characters is fine as well. Do not make one up.

3. Redeploy. `https://your-app/api/v1/status` (demo mode only) shows
   `storage: postgres` and `collection.active: true`, or the reason it is not active.

Storage refuses to switch on without a persistent key or (on Vercel) without a
database: samples encrypted with a throwaway key, or written to a disk that is
wiped, would be lost.

## Retraining (GitHub)

`.github/workflows/retrain.yml` runs every Monday and on demand (Actions → Retrain
models → Run workflow; tick "Report only" to measure without changing anything).

Set up once:

* Repository → Settings → Secrets and variables → Actions → add `HP_DATABASE_URL`
  (the same connection string) and `HP_MASTER_SECRET` (the same value).
* Settings → Actions → General → tick "Allow GitHub Actions to create and approve
  pull requests".

Each run:

1. `ml/dataset/export.py` decrypts the finished attempts onto the runner's
   temporary disk and splits them by participant. The dataset is deleted at the end
   of the job and is never uploaded or committed.
2. `ml/retrain.py` writes a report (in the run summary) and trains candidates:
   * **Eye check (`gaze.onnx`).** Real recordings measured against their own dot
     path are genuine examples. The same recordings measured against *other
     sessions'* paths are replay examples: a real person's video in the wrong
     session. Labelled eye attacks are added.
   * **Movement check (`motor.onnx`).** The public mouse recordings and synthetic
     bots, plus the collected tracing sessions and collected bot sessions.
3. A candidate replaces the current model only if, on held-out participants, its
   balanced accuracy is higher by a margin (0.01 eyes, 0.005 movement) **and** no
   single measure (real people accepted, replays rejected, bots rejected, ...) is
   more than 2 points worse. Otherwise the models folder is untouched.
4. If a model changed, the server's tests run against it and a pull request is
   opened with the report. Merging it deploys the model.

Run it by hand the same way:

```bash
export HP_DATABASE_URL=... HP_MASTER_SECRET=...
python ml/dataset/export.py --out /secure/hp-export
python ml/retrain.py --data /secure/hp-export --iitkgp /data/Mouse-Dynamics --dry-run
rm -rf /secure/hp-export
```

**Voice and face.** `ml/colab_train.ipynb` section 0 runs the same export inside
Colab and adds the labelled recordings and snapshots to the public datasets
(`train_antispoof.py --csv`, `train_face.py --extra`). The report says when there
is enough (about 200 genuine and 200 fake samples each) for that to be worthwhile.

## Limits to keep in mind

* **Verified with simulated sessions only.** The storage, export, retraining and
  pull-request logic have been run end to end on simulated testers, not on real
  recordings. Read the first real report critically.
* **Small test sets are noisy.** With 16 held-out sessions one session is 6 points.
  The report prints the counts next to every percentage.
* **Labels are trusted.** A tester who mislabels sessions poisons the training
  data. Keep the code private, rotate it (`HP_COLLECTION_KEY`) when a tester
  leaves, and review the per-attack-type table in each report for surprises.
* **The stored data is biometric data.** See `PRIVACY.md` for the legal steps
  that have to be in place before collecting from the public.
