import adapter from "@sveltejs/adapter-static";
import { sveltekit } from "@sveltejs/kit/vite";
import { vitePreprocess } from "@sveltejs/vite-plugin-svelte";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [
    sveltekit({
      preprocess: vitePreprocess(),
      // Static SPA: no server runtime. All data comes from the existing Bun API
      // server (server.ts), which also serves these assets.
      adapter: adapter({ fallback: "index.html" }),
    }),
  ],
  server: { host: "127.0.0.1" },
});
