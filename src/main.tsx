import { StrictMode } from 'react';
import { createRoot, hydrateRoot } from 'react-dom/client';
import App from '@/App';
import { shouldHydrate } from '@/store/url';
import '@/index.css';

const root = document.getElementById('root');
if (!root) throw new Error('Missing #root element');

const tree = (
  <StrictMode>
    <App />
  </StrictMode>
);

/**
 * Hydrate a prerendered page whose markup is the page this address asks for; render anything else
 * from scratch.
 *
 * It has to branch rather than always hydrate, because not every page this bundle boots on has
 * markup: `404.html` is the un-prerendered shell and Pages serves it at any unmatched path, and
 * `hydrateRoot` on an empty container is itself a mismatch — React would report a recoverable
 * error and re-render the whole tree on a page that was never wrong.
 *
 * The signal for that is an explicit attribute the prerender script writes, not `hasChildNodes()`.
 * A formatter, a comment, or a newline inside `<div id="root">` is a child node, so the cheap test
 * answers "yes" for a shell that has nothing to hydrate — a false positive that costs a full
 * re-render and looks fine in a browser.
 *
 * **The attribute alone was not enough, and this is the second half.** It is a bare boolean: it
 * says markup exists, not *which scenario* it holds — and the two come apart constantly, because
 * every shared link is the root path carrying a complete query. So {@link shouldHydrate} asks both
 * questions, and its docblock carries the argument for each: is there markup, and is it the right
 * markup. `createRoot` over a container holding the wrong markup replaces it cleanly, so there is
 * nothing to clear first.
 */
if (
  shouldHydrate(
    window.location.pathname,
    window.location.search,
    import.meta.env.BASE_URL,
    root.hasAttribute('data-prerendered')
  )
) {
  hydrateRoot(root, tree);
} else {
  createRoot(root).render(tree);
}
