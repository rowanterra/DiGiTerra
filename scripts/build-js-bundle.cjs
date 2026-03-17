/**
 * Build a single JS bundle from the split app scripts (order matters).
 * Output: static/js/app.bundle.js
 * Run: npm run build
 */
const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..');
const STATIC_JS = path.join(ROOT, 'static', 'js');

const FILES = [
  'init.js',
  'core.js',
  'upload.js',
  'preprocess.js',
  'modeling.js',
  'inference.js',
  'app.js',
];

const outPath = path.join(STATIC_JS, 'app.bundle.js');
const parts = [];

for (const name of FILES) {
  const filePath = path.join(STATIC_JS, name);
  if (!fs.existsSync(filePath)) {
    console.error('Missing:', filePath);
    process.exit(1);
  }
  const content = fs.readFileSync(filePath, 'utf8');
  parts.push(`// === ${name} ===\n${content}`);
}

const bundle = parts.join('\n\n');
fs.writeFileSync(outPath, bundle, 'utf8');
console.log('Wrote', outPath, `(${(bundle.length / 1024).toFixed(1)} KB)`);
