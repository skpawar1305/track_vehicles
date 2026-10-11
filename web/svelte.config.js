import adapter from "@sveltejs/adapter-static";
import { vitePreprocess } from "@sveltejs/vite-plugin-svelte";

/** @type {import('@sveltejs/kit').Config} */
const config = {
  preprocess: vitePreprocess(),
  kit: {
    // Static SPA: no server runtime. All data comes from the existing Bun API
    // server (server.ts), which also serves these assets.
    adapter: adapter({ fallback: "index.html" }),
  },
};

export default config;
