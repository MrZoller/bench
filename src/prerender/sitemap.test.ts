import { describe, expect, it } from 'vitest';
import { prerenderRoutes, routePath } from '@/data/routes';
import { DEVICES, getDevice } from '@/data/catalog';
import { sitemapXml } from './sitemap';

const ORIGIN = 'https://mrzoller.github.io';

/** Every `<loc>` in a sitemap, in order. */
function locs(xml: string): string[] {
  return [...xml.matchAll(/<loc>([^<]*)<\/loc>/g)].map((match) => match[1]);
}

describe('sitemapXml', () => {
  it('lists a page for every route the build writes, absolute and base-prefixed', () => {
    const xml = sitemapXml(prerenderRoutes(), '/headroom/', ORIGIN)!;
    const advertised = prerenderRoutes().filter((route) => route.indexable);

    expect(locs(xml)).toEqual(
      advertised.map((route) => `${ORIGIN}${routePath(route, '/headroom/')}`)
    );
    for (const loc of locs(xml))
      expect(loc).toMatch(/^https:\/\/[^/]+\/headroom\/.*\/$|\/headroom\/$/);
  });

  it('is well-formed, and namespaced as the protocol requires', () => {
    const xml = sitemapXml(prerenderRoutes(), '/', ORIGIN)!;
    expect(xml.startsWith('<?xml version="1.0" encoding="UTF-8"?>\n')).toBe(true);
    expect(xml).toContain('<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">');
    expect(xml.trimEnd().endsWith('</urlset>')).toBe(true);
  });

  /**
   * The rumoured row: a page, and no invitation to it.
   *
   * The catalog's rule is that a pre-release spec stays visibly labelled, and a search result is
   * the one context that strips the label off. So the page is built — someone who has the address
   * gets a working page with its status on it — and the sitemap does not name it, nor any of the
   * pair pages it appears in.
   */
  it('leaves out hardware that is not shipping, and there is something to leave out', () => {
    const unshipped = DEVICES.filter((device) => device.status !== 'shipping');
    expect(unshipped.length).toBeGreaterThan(0);

    const xml = sitemapXml(prerenderRoutes(), '/headroom/', ORIGIN)!;
    for (const device of unshipped) {
      expect(xml).not.toContain(`/${device.id}/`);
    }

    const built = prerenderRoutes().filter((route) =>
      unshipped.some((device) => device.id === route.segments[0])
    );
    // One device page plus its pair pages — written, and every one of them omitted here.
    expect(built.length).toBe(prerenderRoutes().length - locs(xml).length);
    expect(built.length).toBeGreaterThan(1);
  });

  it('keeps every shipping device and every model', () => {
    const xml = sitemapXml(prerenderRoutes(), '/headroom/', ORIGIN)!;
    for (const device of DEVICES.filter((d) => d.status === 'shipping')) {
      expect(xml).toContain(`${ORIGIN}/headroom/${device.id}/</loc>`);
    }
    const models = prerenderRoutes().filter((route) => route.tier === 2);
    for (const route of models) {
      expect(xml).toContain(`${ORIGIN}${routePath(route, '/headroom/')}</loc>`);
    }
  });

  /**
   * **Nothing, rather than something invalid.** `<loc>` has to be a complete URL beginning with a
   * scheme, so a build that does not know where it will be published cannot write a lesser
   * sitemap — it can only write a file a crawler reports as a parse error. Failing the build
   * instead would break every fork and every pull-request build over a setting that only matters
   * once something is published, and guessing `<owner>.github.io` is what `PAGES_SITE_ORIGIN`
   * exists to stop.
   */
  it('writes no sitemap at all when the site has no known origin', () => {
    expect(sitemapXml(prerenderRoutes(), '/headroom/', '')).toBeNull();
  });

  it('escapes a loc rather than emitting invalid XML', () => {
    const [root] = prerenderRoutes();
    const xml = sitemapXml([{ ...root, segments: ['a&b'] }], '/', ORIGIN)!;
    expect(xml).toContain('<loc>https://mrzoller.github.io/a&amp;b/</loc>');
  });

  it('resolves every listed device path back to a catalogued row', () => {
    // The sitemap is what a crawler is handed, so a path in it that names nothing is a soft 404
    // reported against the site rather than a broken link somebody notices.
    const xml = sitemapXml(prerenderRoutes(), '/', ORIGIN)!;
    for (const route of prerenderRoutes().filter((r) => r.tier === 1 && r.indexable)) {
      expect(() => getDevice(route.segments[0])).not.toThrow();
      expect(locs(xml)).toContain(`${ORIGIN}/${route.segments[0]}/`);
    }
  });
});
