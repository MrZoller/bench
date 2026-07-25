import { useEffect } from 'react';
import { useConfig, type Config } from './config';
import { configToSearch, sameScenario, searchToConfig } from './url';

/**
 * Keeps the address bar and the store in step, in both directions.
 *
 * Deliberately `replaceState` rather than `pushState`: a slider emits a value per frame while
 * being dragged, and pushing each one would bury the previous page under hundreds of history
 * entries. The trade is that the back button leaves the page rather than stepping through
 * scenarios — which is the right default for a tool people arrive at by link.
 *
 * The reverse direction still matters despite that, because `popstate` also fires when someone
 * edits the URL or follows a second link into the same page.
 */
export function useUrlSync(): void {
  const config = useConfig();
  const replace = useConfig((s) => s.replace);

  // Store -> URL.
  useEffect(() => {
    const search = configToSearch(config as Config);
    const next = `${window.location.pathname}${search}`;
    if (next !== `${window.location.pathname}${window.location.search}`) {
      window.history.replaceState(null, '', next);
    }
  }, [config]);

  // URL -> store.
  useEffect(() => {
    const onPop = () => {
      const fromUrl = searchToConfig(window.location.search);
      if (!sameScenario(fromUrl, useConfig.getState() as Config)) replace(fromUrl);
    };
    window.addEventListener('popstate', onPop);
    return () => window.removeEventListener('popstate', onPop);
  }, [replace]);
}
