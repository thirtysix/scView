import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import path from "path";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  // plotly.js source partials (our slim bundle) reference Node's `global`, which
  // doesn't exist in the browser — alias it to globalThis for both the prod build
  // (Rollup, via `define`) and the dev dependency pre-bundle (esbuild).
  define: {
    global: "globalThis",
  },
  optimizeDeps: {
    esbuildOptions: {
      define: { global: "globalThis" },
    },
  },
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  build: {
    // The vendor-plotly / vendor-deckgl chunks are intentionally large but lazy.
    chunkSizeWarningLimit: 5000,
    rollupOptions: {
      output: {
        // Split heavy vendors into their own long-lived chunks so an app code
        // change doesn't bust the cache for ~5 MB of libraries.
        manualChunks(id: string) {
          if (!id.includes("node_modules")) return undefined;
          if (id.includes("plotly")) return "vendor-plotly";
          if (/[\\/](deck\.gl|@deck\.gl|@luma\.gl|@math\.gl|@loaders\.gl)[\\/]/.test(id))
            return "vendor-deckgl";
          if (id.includes("apache-arrow")) return "vendor-arrow";
          if (/[\\/](react|react-dom|scheduler)[\\/]/.test(id)) return "vendor-react";
          return undefined;
        },
      },
    },
  },
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: process.env.VITE_API_URL || "http://localhost:8080",
        changeOrigin: true,
      },
      "/ws": {
        target: (process.env.VITE_API_URL || "http://localhost:8080").replace("http", "ws"),
        ws: true,
      },
    },
  },
});
