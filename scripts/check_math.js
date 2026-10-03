// Renders every $...$ and $$...$$ in docs/, README, CLAUDE.md and notebook
// markdown with KaTeX, and reports anything that fails.
//
//   npm install katex && node scripts/check_math.js
//
// Exists because KaTeX implements a subset of LaTeX: \unicode, \shuffle,
// \mathscr and \DeclareMathOperator all parse in TeX and fail here.

const fs = require('fs'), path = require('path');
const katex = require('katex');            // resolved from ./node_modules
const args = process.argv.slice(2);
if (args.length && (args.length !== 2 || args[0] !== '--root')) {
  console.error('Usage: node scripts/check_math.js [--root DIRECTORY]');
  process.exit(2);
}
const root = args.length ? path.resolve(args[1]) : path.resolve(__dirname, '..');
function extract(text) {
  const out = [];
  text.replace(/\$\$([\s\S]*?)\$\$/g, (m,g) => { out.push([g,true]); return ' '; })
      .replace(/(?<!\\)\$([^\$\n]+?)(?<!\\)\$/g, (m,g) => { out.push([g,false]); return ''; });
  return out;
}
let files = [];
(function walk(d){ for (const e of fs.readdirSync(d,{withFileTypes:true})) {
  if (e.name==='.git'||e.name==='.venv'||e.name==='node_modules'||e.name==='.ipynb_checkpoints'||e.name==='__pycache__') continue;
  const f = path.join(d,e.name);
  if (e.isDirectory()) walk(f); else if (f.endsWith('.md')||f.endsWith('.ipynb')) files.push(f);
}})(root);
let bad=0, total=0;
for (const f of files) {
  let text;
  if (f.endsWith('.ipynb')) {
    const nb = JSON.parse(fs.readFileSync(f,'utf8'));
    text = nb.cells.filter(c=>c.cell_type==='markdown').map(c=>c.source.join('')).join('\n\n');
  } else text = fs.readFileSync(f,'utf8');
  for (const [expr,display] of extract(text)) {
    total++;
    try { katex.renderToString(expr,{displayMode:display,throwOnError:true}); }
    catch (err) { bad++;
      console.log('FAIL', path.relative(root,f));
      console.log('  expr:', expr.replace(/\n/g,' ').slice(0,110));
      console.log('  err :', String(err.message).split('\n')[0].slice(0,130)); }
  }
}
console.log(`\n${total} expressions checked, ${bad} failed`);
process.exitCode = bad ? 1 : 0;
