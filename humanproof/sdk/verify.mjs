// Verify a HumanProof attestation token in Node.js >= 20 (WebCrypto Ed25519), no dependencies.
//
//   import { createVerifier } from "./verify.mjs";
//   const verify = createVerifier({ issuer: "https://api.humanproof.example", audience: "bank.example" });
//   const claims = await verify(token); // throws on anything invalid
const TYP = "hp-attestation+jwt";
const dec = (s) => Buffer.from(s, "base64url");

export function createVerifier({ issuer, audience, cacheSeconds = 3600, leewaySeconds = 30 }) {
  issuer = issuer.replace(/\/$/, "");
  if (!issuer.startsWith("https://") && !/localhost|127\.0\.0\.1/.test(issuer)) throw new Error("issuer must use https");
  let keys = new Map();
  let fetchedAt = 0;

  async function key(kid) {
    const now = Date.now() / 1000;
    if ((!keys.has(kid) && now - fetchedAt > 60) || now - fetchedAt > cacheSeconds) {
      const res = await fetch(`${issuer}/.well-known/jwks.json`);
      if (!res.ok) throw new Error("could not fetch JWKS");
      const jwks = await res.json();
      keys = new Map();
      for (const k of jwks.keys ?? []) {
        if (k.kty === "OKP" && k.crv === "Ed25519") {
          keys.set(k.kid, await crypto.subtle.importKey("raw", dec(k.x), { name: "Ed25519" }, false, ["verify"]));
        }
      }
      fetchedAt = now;
    }
    const k = keys.get(kid);
    if (!k) throw new Error("unknown key id");
    return k;
  }

  return async function verify(token) {
    const parts = String(token).split(".");
    if (parts.length !== 3) throw new Error("malformed");
    const [h, p, s] = parts;
    const header = JSON.parse(dec(h).toString());
    const claims = JSON.parse(dec(p).toString());
    if (header.alg !== "EdDSA" || header.typ !== TYP) throw new Error("unexpected algorithm or type");
    const ok = await crypto.subtle.verify({ name: "Ed25519" }, await key(header.kid), dec(s), Buffer.from(`${h}.${p}`));
    if (!ok) throw new Error("bad signature");
    const now = Date.now() / 1000;
    if (claims.iss !== issuer) throw new Error("wrong issuer");
    if (claims.aud !== audience) throw new Error("wrong audience");
    if (claims.exp < now - leewaySeconds || claims.nbf > now + leewaySeconds) throw new Error("expired or not yet valid");
    if (claims.hp_human !== true) throw new Error("not a human attestation");
    return claims;
  };
}
