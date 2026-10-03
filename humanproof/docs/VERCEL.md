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
| `HP_MASTER_SECRET` | 32 random bytes, base64 (see `docs/DATA.md`) | same, required (or the three separate keys from `python -m tools.genkeys`, which allow rotation) |
| `DATABASE_URL` | set by Vercel when you attach a Postgres database (Storage → Neon) | same |
| `HP_DATA_COLLECTION_ENABLED`, `HP_COLLECTION_KEY` | only to store data for training (`docs/DATA.md`) | same |
| `HP_ASR_MODEL` | unset | required (see below) |

Set `HP_MASTER_SECRET` even for a demo. Without it each server instance generates
its own keys, so a token signed by one instance fails verification on another.

Also set automatically on Vercel: `HP_TRUSTED_IP_HEADER=x-real-ip`. Vercel's edge
sets that header itself, so each visitor gets their own rate limit and hourly
attempt allowance instead of everyone sharing the proxy's address.

## Limits of this deployment

* **Attach a database.** Without one, sessions live in SQLite under `/tmp` on a
  single instance: a verification that starts on one instance and continues on
  another fails with "Session not found", and everything is lost when the instance
  is recycled. With Postgres attached, sessions, passkeys, the audit log and
  stored samples are shared by all instances and persist.
* **Rate limits are still per instance** (in memory). The hourly cap on attempts
  per visitor is in the database and does hold across instances.
* **Production mode needs more than Vercel gives by default.** It requires the
  trained voice and face models in `backend/models/` and speech recognition
  (`requirements-asr.txt`, kept out of the default install to stay well under the
  500 MB Python bundle limit; the Whisper model must also be bundled, because the
  filesystem is read-only apart from `/tmp`).
* **Request size:** Vercel caps bodies at 4.5 MB; the API's own cap is 2.5 MB.

## Local check

```bash
vercel dev -L        # from the repository root; runs both services behind one port
cd humanproof/backend
python -m tools.e2e_probe --kind bot --base http://localhost:3000/api --origin http://localhost:3000
```
