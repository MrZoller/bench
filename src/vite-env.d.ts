/// <reference types="vite/client" />

/**
 * The origin the deployed site is served from, baked in by `vite.config.ts` from the
 * `PAGES_SITE_ORIGIN` repository variable. Empty when nothing has set it — see
 * `src/lib/siteUrl.ts`, which is the only module allowed to read it.
 */
declare const __SITE_ORIGIN__: string;
