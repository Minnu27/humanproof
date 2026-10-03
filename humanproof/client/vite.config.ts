import react from "@vitejs/plugin-react";
import { defineConfig, loadEnv } from "vite";

// The Content-Security-Policy is generated at build time so connect-src names
// exactly one API origin. No inline scripts, no remote code, no eval
// (WebAssembly compilation is the only exception MediaPipe needs).
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "VITE_");
  // Relative base ("/api", the web default) is same-origin: 'self' already covers it.
  const apiBase = env.VITE_API_BASE || "/api";
  const apiOrigin = /^https?:\/\//.test(apiBase) ? ` ${new URL(apiBase).origin}` : "";
  const csp = [
    "default-src 'self'",
    "script-src 'self' 'wasm-unsafe-eval'",
    `connect-src 'self'${apiOrigin}`,
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
    server: {
      port: 5173,
      strictPort: true,
      // Same shape as production: the browser calls /api on its own origin.
      // Start the backend with HP_API_PREFIX=/api (or use `vercel dev`).
      proxy: { "/api": { target: env.VITE_DEV_API_TARGET || "http://localhost:8000", changeOrigin: false } },
    },
  };
});
