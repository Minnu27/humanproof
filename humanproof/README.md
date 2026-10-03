# HumanProof

A verification layer that answers one question for any app, bank or video call:
**is a live human on the other end right now?** Before registering (or before a
risky action), the person completes three short checkpoints, about 40 seconds in total:

1. **Eyes:** follow a dot along a secret random path. On-device face tracking
   measures eye and head movement; the server checks that the eyes follow *this*
   path with human timing, and runs a deepfake detector on four face snapshots.
2. **Movement:** trace a random curve with a mouse, finger or arrow keys. A model
   trained on real human movement separates human motor dynamics from scripts.
3. **Voice:** read five random words. Speech recognition checks the words, an
   anti-spoofing model checks for synthetic, converted or replayed audio, and lip
   movement must match the sound. Passing stores an encrypted voice signature.

The scores are fused into one decision. On a pass, the app receives a signed,
15-minute token (Ed25519) that any relying party verifies against the public
keys at `/.well-known/jwks.json`. Each app sees a different pseudonym for the
same person, so apps cannot track users across services.

```
client/   React app → web, desktop (Tauri: Windows/macOS/Linux), mobile (Capacitor: iOS/Android)
backend/  FastAPI service: challenges, scoring, fusion, encryption, attestation, tokens
ml/       Training pipelines: motor (CPU), voice + face (GPU, Colab notebook included)
sdk/      Token verification for relying parties (Python, Node.js)
docs/     Threat model, security controls (OWASP ASVS map), privacy, models, releasing
```

## Run it locally (about 5 minutes)

```bash
# API (dev mode lets the voice/face checkpoints run on heuristics until you train those models)
cd backend
pip install -r requirements-dev.txt
HP_ENV=dev HP_ALLOW_HEURISTIC_FALLBACK=true HP_API_PREFIX=/api uvicorn humanproof.main:app --port 8000

# App (new terminal)
cd client
npm install
npm run dev        # downloads the face-tracking model once, then serves http://localhost:5173
```

The web app calls the API at `/api` on its own origin; `npm run dev` proxies that to
port 8000. `vercel dev` from the repository root runs both together the same way Vercel does.

The first `npm run dev` pins the face-tracking model's SHA-256 in
`client/models.lock.json`. Commit that file; from then on any changed model file fails the build.

Desktop app during development: `npm run desktop:dev`. Phones: `npm run cap:sync`,
then open `client/android` in Android Studio or `client/ios/App` in Xcode.

## Tests and checks

```bash
cd backend && pytest --cov=humanproof      # 82 tests, ~90% coverage
HP_IITKGP_DATA=/path/to/Mouse-Dynamics pytest -k real_held_out   # motor model on real unseen people
bandit -r humanproof tools && pip-audit -r requirements.txt
cd ../client && npm run typecheck && npm run build && npm audit
python -m tools.e2e_probe --kind bot --base http://127.0.0.1:8000/api   # from backend/, against a running server
```

CI (`.github/workflows/`) runs all of the above plus CodeQL, gitleaks, Trivy and
an OWASP ZAP API scan, retrains the motor model from public data, and on a
version tag builds installers for Windows, macOS, Linux, Android and iOS.

## Going to production

1. **Train the GPU models:** open `ml/colab_train.ipynb` in Colab Pro (see
   `docs/MODELS.md` for datasets, licences and what numbers to expect).
2. **Generate keys:** `python -m tools.genkeys --out ./secrets`; fill `.env` from
   `backend/.env.example`. Production refuses to start with dev keys, HTTP, wildcard
   origins, missing models, or missing speech recognition.
3. **Deploy:** `docker compose up -d` in `backend/` (API behind Caddy with automatic HTTPS),
   or web app + API together on Vercel (`docs/VERCEL.md`).
4. **Ship the apps:** set the repository variables and signing secrets listed in
   `docs/RELEASING.md`, then push a tag `v1.0.0`.
5. **Get certified before selling to banks:** an accredited ISO/IEC 30107-3
   liveness test (e.g. iBeta), an external penetration test, and SOC 2 later.
   `docs/THREAT_MODEL.md` and `docs/SECURITY.md` are written to hand to those auditors.

## Integrating as a relying party

```python
from humanproof_verify import Verifier
claims = Verifier("https://api.humanproof.example", audience="bank.example").verify(token)
if claims["hp_assurance"] != "device_attested":
    ...  # web/desktop result: fine for sign-ups, ask for more on a wire transfer
```

## Honest limits

A person sitting at the device passes; that is correct, because the product
proves presence, not identity. Browsers cannot prove they are unmodified, so web
results carry a lower assurance level. Detection models age as generators
improve and need scheduled retraining. Details are in `docs/THREAT_MODEL.md`.
