/** @type {import('eslint').Linter.Config} */
// The seven files under static/js share one global scope in the browser and are
// concatenated into app.bundle.js. Linting them separately reports every
// cross-file reference as undefined, so no-undef runs against the bundle
// (see .eslintrc.bundle.cjs and the lint script in package.json) and the split
// files are checked for everything else.
module.exports = {
  root: true,
  env: { browser: true, es2022: true },
  extends: 'eslint:recommended',
  parserOptions: { ecmaVersion: 2022, sourceType: 'script' },
  rules: {
    'no-undef': 'off',
    'no-unused-vars': ['warn', { argsIgnorePattern: '^_', varsIgnorePattern: '^_' }],
    'no-console': 'off',
    // Deliberate empty branches and one nested helper; neither can throw.
    'no-empty': 'off',
    'no-inner-declarations': 'off',
    'no-alert': 'off',
  },
  ignorePatterns: ['build/', 'dist/', 'node_modules/', '*.min.js', 'static/js/app.bundle.js'],
};
