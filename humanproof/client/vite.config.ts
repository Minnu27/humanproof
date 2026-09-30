import react from "@vitejs/plugin-react";
import { defineConfig, loadEnv } from "vite";

// The Content-Security-Policy is generated at build time so connect-src names
// exactly one API origin. No inline scripts, no remote code, no eval
// (WebAssembly compilation is the only exception MediaPipe needs).
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "VITE_");
  const api = new URL(env.VITE_API_BASE || "http://localhost:8000").origin;
  const csp = [
    "default-src 'self'",
    "script-src 'self' 'wasm-unsafe-eval'",
    `connect-src 'self' ${api}`,
    "img-src 'self' data: blob:",
    "media-src 'self' blob: mediastream:",
    "worker-src 'self' blob:",
    "style-src 'self'",
    "font-src 'self'",
    "object-src 'none'",
    "base-uri 'none'",
    "form-action 'none'",
  ].join("; ");
  return {
    plugins: [
      react(),
      {
        name: "inject-csp",
        transformIndexHtml(html) {
          return html.replace("__CSP__", csp);
        },
      },
    ],
    build: { target: "es2022", sourcemap: false, assetsInlineLimit: 0 },
    server: { port: 5173, strictPort: true },
  };
});
