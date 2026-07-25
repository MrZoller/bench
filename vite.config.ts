/// <reference types="vitest/config" />
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';
import { fileURLToPath, URL } from 'node:url';

// https://vite.dev/config/
export default defineConfig({
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
    // Playwright specs live in e2e/ and run via `npm run screenshots`.
    exclude: ['**/node_modules/**', '**/dist/**', 'e2e/**'],
  },
});
