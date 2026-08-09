/**
 * Guards a failure mode that has now bitten twice: a backtick inside a comment
 * that sits within a WGSL template literal silently terminates the template.
 * TypeScript then reports a parse error tens of lines away, or the file still
 * parses and a shader block is quietly truncated.
 *
 * Only backticks *inside* a template are a problem — in ordinary comments they
 * are idiomatic, so the scan tracks template state rather than banning them.
 *
 *   npm run check:shaders
 */

import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join } from 'node:path';

const ROOTS = ['src', 'src/shaders'];
const OPEN = /\/\* wgsl \*\/\s*`/;
const problems: string[] = [];

for (const root of ROOTS) {
  for (const entry of readdirSync(root)) {
    const path = join(root, entry);
    if (!statSync(path).isFile() || !path.endsWith('.ts')) continue;

    let inTemplate = false;
    readFileSync(path, 'utf8')
      .split('\n')
      .forEach((line, i) => {
        if (!inTemplate) {
          if (OPEN.test(line)) inTemplate = true;
          return;
        }
        // A closing delimiter is the first thing on its line: ` or `; or `);
        if (line.trimStart().startsWith('`')) {
          inTemplate = false;
          return;
        }
        if (line.includes('`')) problems.push(`  ${path}:${i + 1}  ${line.trim()}`);
      });
  }
}

if (problems.length > 0) {
  console.error('\n  Backtick inside a WGSL template — this truncates the shader:\n');
  console.error(problems.join('\n'));
  console.error('');
  process.exitCode = 1;
} else {
  console.log('  shader templates ok');
}
