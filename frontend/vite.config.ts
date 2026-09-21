import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    host: "0.0.0.0",
    port: 5284,
    // Allow Cursor port-forwards and public demo tunnels (loca.lt, trycloudflare, etc.).
    allowedHosts: true,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8472",
        changeOrigin: true,
        // Avoid buffering SSE chat streams through the Vite proxy.
        configure: (proxy) => {
          proxy.on("proxyRes", (proxyRes, req) => {
            if (req.url?.includes("/chat/stream")) {
              proxyRes.headers["cache-control"] = "no-cache, no-transform";
              proxyRes.headers["x-accel-buffering"] = "no";
            }
          });
        },
      },
    },
  },
});
