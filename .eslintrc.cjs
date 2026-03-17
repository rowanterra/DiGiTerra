/** @type {import('eslint').Linter.Config} */
module.exports = {
  root: true,
  env: { browser: true, es2022: true },
  parserOptions: { ecmaVersion: 2022, sourceType: 'script' },
  rules: {
    'no-unused-vars': ['warn', { argsIgnorePattern: '^_' }],
    'no-console': 'off',
    'no-alert': 'off',
  },
  ignorePatterns: ['build/', 'dist/', 'node_modules/', '*.min.js'],
};
