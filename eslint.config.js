import js from '@eslint/js';
import globals from 'globals';
import reactHooks from 'eslint-plugin-react-hooks';
import reactRefresh from 'eslint-plugin-react-refresh';
import tseslint from 'typescript-eslint';

export default tseslint.config(
  /**
   * `.claude/worktrees` holds a full checkout of this repo per background agent, and each carries its
   * own `tsconfig.json`. Without this, typescript-eslint sees several candidate roots and fails with
   * `No tsconfigRootDir was set, and multiple candidate TSConfigRootDirs are present` — as a
   * *parsing* error on every file it is given, so `npm run lint` reported 936 errors across the real
   * `src/` tree while an agent happened to be running, and none of them were about the code.
   */
  { ignores: ['dist', 'coverage', 'playwright-report', 'test-results', '.claude/worktrees'] },
  {
    extends: [js.configs.recommended, ...tseslint.configs.recommended],
    files: ['**/*.{ts,tsx}'],
    languageOptions: {
      ecmaVersion: 2022,
      globals: { ...globals.browser, ...globals.node },
    },
    plugins: {
      'react-hooks': reactHooks,
      'react-refresh': reactRefresh,
    },
    rules: {
      ...reactHooks.configs.recommended.rules,
      'react-refresh/only-export-components': ['warn', { allowConstantExport: true }],
    },
  }
);
