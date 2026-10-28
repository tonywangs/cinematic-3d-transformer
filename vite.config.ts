import { defineConfig } from "vite";
export default defineConfig({
  server: { port: 5174, proxy: { "/export": "http://127.0.0.1:5175" } },
  build: { rollupOptions: { output: { manualChunks: { three: ["three"] } } } },
});
