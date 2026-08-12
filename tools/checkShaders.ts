/**
 * Static checks on the WGSL carried in template literals.
 *
 * Both failures here are silent at author time and expensive at run time: the
 * TypeScript compiler is happy, the page loads, and the first thing you learn
 * is a 500 from the pipeline or a blank screen. Both are decidable by reading
 * the text, which is what this does.
 *
 *   backticks    a backtick inside a WGSL template — usually in a comment,
 *                where it is idiomatic prose — terminates the template. The
 *                shader is then truncated, or TypeScript reports a parse error
 *                tens of lines away from the cause.
 *
 *   reserved     WGSL reserves ~130 words that are not keywords and have no
 *                legal use anywhere in a shader. Several of them are the
 *                obvious name for something in a renderer — `filter`, `sample`
 *                is not one but `set`, `shared`, `ref`, `type`, `patch`,
 *                `precise` and `smooth` are — so they get written as a local
 *                or a struct member and the module fails to compile as a
 *                whole. Nothing else in the toolchain looks at WGSL before the
 *                GPU does.
 *
 * The scan tracks template state rather than banning backticks outright, and
 * strips comments and `${…}` interpolations before looking for identifiers:
 * the prose in this codebase is full of "the type of", "a shared field" and
 * "where the slope", none of which is WGSL.
 *
 *   npm run check:shaders
 */

import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join } from 'node:path';

const ROOTS = ['src', 'src/shaders'];
const OPEN = /\/\* wgsl \*\/\s*`/;

/**
 * WGSL's reserved words, from the spec's "Reserved Words" table. These are not
 * keywords — they have no meaning — they are simply unavailable as identifiers
 * so that the language can grow into them later. Using one is always an error,
 * in any position, which is what makes this check exact rather than heuristic.
 */
const RESERVED = new Set(
  `NULL Self abstract active alignas alignof as asm asm_fragment async attribute auto await
   become binding_array cast catch class co_await co_return co_yield coherent column_major
   common compile compile_fragment concept const_cast consteval constexpr constinit crate
   debugger decltype delete demote demote_to_helper do dynamic_cast enum explicit export
   extends extern external fallthrough filter final finally friend from fxgroup get goto
   groupshared highp impl implements import inline instanceof interface layout lowp macro
   macro_rules match mediump meta mod module move mut mutable namespace new nil noexcept
   noinline nointerpolation non_coherent noncoherent noperspective null nullptr of operator
   package packoffset partition pass patch pixelfragment precise precision premerge priv
   protected pub public readonly ref regardless register reinterpret_cast require resource
   restrict self set shared sizeof smooth snorm static static_assert static_cast std
   subroutine super target template this thread_local throw trait try type typedef typeid
   typename typeof union unless unorm unsafe unsized use using varying virtual volatile wgsl
   where with writeonly yield`
    .trim()
    .split(/\s+/),
);

/** Scanner state that outlives a line: block comments and `${…}` both wrap. */
interface Scan {
  comment: boolean;
  interp: number;
}

/**
 * One line of a template with its comments and interpolations removed, so what
 * is left is WGSL and nothing else. An interpolation's *result* is not scanned
 * here and does not need to be: every emitted block in this project is itself
 * a wgsl-tagged template, and gets scanned where it is written.
 */
function code(line: string, st: Scan): string {
  let out = '';
  for (let i = 0; i < line.length; i++) {
    if (st.comment) {
      if (line.startsWith('*/', i)) {
        st.comment = false;
        i++;
      }
    } else if (st.interp > 0) {
      if (line[i] === '{') st.interp++;
      else if (line[i] === '}') st.interp--;
    } else if (line.startsWith('/*', i)) {
      st.comment = true;
      i++;
    } else if (line.startsWith('//', i)) {
      break;
    } else if (line.startsWith('${', i)) {
      st.interp = 1;
      i++;
    } else {
      out += line[i];
    }
  }
  return out;
}

const backticks: string[] = [];
const reserved: string[] = [];

for (const root of ROOTS) {
  for (const entry of readdirSync(root)) {
    const path = join(root, entry);
    if (!statSync(path).isFile() || !path.endsWith('.ts')) continue;

    let inTemplate = false;
    const st: Scan = { comment: false, interp: 0 };
    readFileSync(path, 'utf8')
      .split('\n')
      .forEach((line, i) => {
        if (!inTemplate) {
          if (OPEN.test(line)) {
            inTemplate = true;
            st.comment = false;
            st.interp = 0;
          }
          return;
        }
        // A closing delimiter is the first thing on its line: ` or `; or `);
        if (line.trimStart().startsWith('`')) {
          inTemplate = false;
          return;
        }
        if (line.includes('`')) backticks.push(`  ${path}:${i + 1}  ${line.trim()}`);
        for (const [word] of code(line, st).matchAll(/[A-Za-z_][A-Za-z0-9_]*/g)) {
          if (RESERVED.has(word)) reserved.push(`  ${path}:${i + 1}  ${word}    ${line.trim()}`);
        }
      });
  }
}

if (backticks.length > 0) {
  console.error('\n  Backtick inside a WGSL template — this truncates the shader:\n');
  console.error(backticks.join('\n'));
  console.error('');
}
if (reserved.length > 0) {
  console.error('\n  WGSL reserved word used as an identifier — the module will not compile:\n');
  console.error(reserved.join('\n'));
  console.error('');
}
if (backticks.length + reserved.length > 0) {
  process.exitCode = 1;
} else {
  console.log('  shader templates ok');
}
