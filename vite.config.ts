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
  resolve: {
    alias: {
      // GLTFLoader, like everything under three/examples, imports from 'three'
      // — the WebGL-flavoured core entry — while the app imports
      // 'three/webgpu'. Both resolve to real modules, so it works, and three
      // then prints "Multiple instances of Three.js being imported" and the
      // bundle carries two copies of the core. The WebGPU entry re-exports all
      // of core, so pointing bare 'three' at it collapses them to one.
      //
      // Exact-match only ('three$'): 'three/tsl' and 'three/examples/...' must
      // keep resolving to their own entry points.
      three$: 'three/webgpu',
    },
  },
  build: { target: 'esnext' },
});
