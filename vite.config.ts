import { defineConfig } from 'vite';

export default defineConfig({
  // PORT is honoured when the environment assigns one, so a dev server already
  // holding 5173 does not block a second one. Nothing in the app depends on a
  // particular origin — every fetch is relative — so any port serves it.
  // Without PORT the behaviour is unchanged: 5173, and fail rather than drift
  // somewhere the README does not name.
  server: { port: Number(process.env.PORT) || 5173, strictPort: true },
  // Deploy base. '/' for the dev server and for a user/organisation Pages
  // site; '/<repo>/' for a project site, which is what
  // https://<user>.github.io/<repo>/ is. Taken from the environment so the
  // repository name is not baked into the source — the Pages workflow sets it
  // from the repo it is running in. Get this wrong and every asset 404s while
  // index.html loads fine, which reads as a blank page.
  base: process.env.BASE_PATH ?? '/',
  build: { target: 'esnext' },
});
