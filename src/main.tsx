import { StrictMode } from 'react';
import { createRoot, hydrateRoot } from 'react-dom/client';
import App from '@/App';
import '@/index.css';

const root = document.getElementById('root');
if (!root) throw new Error('Missing #root element');

const tree = (
  <StrictMode>
    <App />
  </StrictMode>
);

/**
 * Hydrate a prerendered page; render an empty one from scratch.
 *
 * It has to branch rather than always hydrate, because not every page this bundle boots on has
 * markup: `404.html` is the un-prerendered shell and Pages serves it at any unmatched path, and
 * `hydrateRoot` on an empty container is itself a mismatch — React would report a recoverable
 * error and re-render the whole tree on a page that was never wrong.
 *
 * The signal is an explicit attribute the prerender script writes, not `hasChildNodes()`. A
 * formatter, a comment, or a newline inside `<div id="root">` is a child node, so the cheap test
 * answers "yes" for a shell that has nothing to hydrate — a false positive that costs a full
 * re-render and looks fine in a browser.
 */
if (root.hasAttribute('data-prerendered')) {
  hydrateRoot(root, tree);
} else {
  createRoot(root).render(tree);
}
