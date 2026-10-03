# Deploying on Vercel

One Vercel project, two services, one domain. `vercel.json` lives at the
repository root (the folder that contains `humanproof/`).

| Service | What it is | Public path |
|---|---|---|
| `client` | The web app (static Vite build) | `/` (everything not under `/api`) |
| `backend` | The FastAPI service (`backend/app.py`, Python 3.12) | `/api/*` |

There are no bindings: the only caller of the backend is the web app running in
the visitor's browser, which reaches it through the public `/api` route. Nothing
server-side calls another service. `ml/` (training scripts) and
`client/src-tauri` (desktop shell) are not web services and are not deployed.

## How the paths line up

Vercel hands the backend the original path, so `/api/v1/sessions` arrives as
`/api/v1/sessions`. `backend/app.py` therefore mounts every route under `/api`
when it runs on Vercel (`HP_API_PREFIX`), and the web build calls `/api` on its
own origin (`client/src/lib/api.ts`). `backend/tests/test_vercel.py` fails if
`vercel.json`, the backend prefix and the client default drift apart.

Relying parties verify tokens against `https://<your-domain>/api/.well-known/jwks.json`.

## Environment variables (Vercel → Project → Settings → Environment Variables)

Set automatically from Vercel's own variables unless you override them:
`HP_API_PREFIX=/api`, `HP_PUBLIC_BASE_URL`, `HP_ALLOWED_ORIGINS`, `HP_RP_ID`
(the last three use the production domain). Override them when you attach a custom
domain, or to allow the desktop and mobile apps
(`HP_ALLOWED_ORIGINS=["https://your-domain","capacitor://localhost","https://localhost","tauri://localhost","http://tauri.localhost"]`).

You set these:

| Variable | Demo deployment | Production |
|---|---|---|
| `HP_ENV` | `dev` | `prod` |
| `HP_ALLOW_HEURISTIC_FALLBACK` | `true` | unset |
| `HP_KEK_KEYRING`, `HP_SIGNING_KEYS`, `HP_PAIRWISE_SECRET` | from `python -m tools.genkeys --out ./secrets` (use the `b64:` form of the signing key) | same, required |
| `HP_ASR_MODEL` | unset | required (see below) |

Set the three key variables even for a demo. Without them each server instance
generates its own keys, so a token signed by one instance fails verification on another.

## Limits of this deployment

* **Session storage is per instance.** Sessions, rate limits and pending passkey
  challenges live in SQLite under `/tmp` and in memory. Vercel can run several
  instances, and a verification that starts on one and continues on another fails
  with "Session not found". A quiet demo usually stays on one instance; real
  traffic needs a shared database (Postgres) and Redis first.
* **Nothing persists.** `/tmp` is wiped when an instance is recycled, so stored
  voice signatures, passkeys and the audit log are lost. Do not onboard real users.
* **Production mode needs more than Vercel gives by default.** It requires the
  trained voice and face models in `backend/models/` and speech recognition
  (`requirements-asr.txt`, kept out of the default install to stay well under the
  500 MB Python bundle limit; the Whisper model must also be bundled, because the
  filesystem is read-only apart from `/tmp`).
* **Request size:** Vercel caps bodies at 4.5 MB; the API's own cap is 2.5 MB.

For a production service, run the backend from `backend/Dockerfile` on a host with
a persistent disk or database, and keep Vercel for the web app.

## Local check

```bash
vercel dev -L        # from the repository root; runs both services behind one port
cd humanproof/backend
python -m tools.e2e_probe --kind bot --base http://localhost:3000/api --origin http://localhost:3000
```
