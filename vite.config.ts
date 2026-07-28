/// <reference types="vitest/config" />
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';
import { fileURLToPath, URL } from 'node:url';

// https://vite.dev/config/
export default defineConfig({
  /**
   * Where the built site will be served from.
   *
   * `/` is right for a custom domain, which is the plan — and wrong for the GitHub Pages project
   * URL the site lands on before one is attached, where everything sits under `/<repo>/`. Baking
   * either in makes the other deploy serve a blank page with four 404s in the console: a failure
   * that reads as a broken build and is not one.
   *
   * So it is an input, set by the deploy workflow from a repository variable, and `/` by default
   * because that is what `npm run dev`, `npm run preview` and the Playwright suite all assume.
   */
  base: process.env.BASE_PATH || '/',
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
    },
  },
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
    css: true,
    coverage: {
      provider: 'v8',
      reporter: ['text', 'html'],
      // The engine is the part that has to be right, but the store and the formatters sit
      // between it and the screen: a wrong clamp or a mis-scaled unit reaches a user exactly as
      // a wrong roofline would, and neither was measured while this listed only the engine.
      include: ['src/engine/**', 'src/store/**', 'src/lib/**', 'src/data/**'],
    },
    // Playwright specs live in e2e/ and run via `npm run test:e2e`.
    exclude: ['**/node_modules/**', '**/dist/**', 'e2e/**'],
  },
});
