import { defineConfig } from 'vite';

export default defineConfig({
  // PORT is honoured when the environment assigns one, so a dev server already
  // holding 5173 does not block a second one. Nothing in the app depends on a
  // particular origin — every fetch is relative — so any port serves it.
  // Without PORT the behaviour is unchanged: 5173, and fail rather than drift
  // somewhere the README does not name.
  server: { port: Number(process.env.PORT) || 5173, strictPort: true },
  build: { target: 'esnext' },
});
