import { describe, expect, it } from 'vitest';
import { configFromPath, prerenderRoutes, routePath } from './routes';
import { DEVICES, getDevice } from './catalog';

describe('prerenderRoutes', () => {
  it('starts at the root', () => {
    const [root] = prerenderRoutes();
    expect(root.segments).toEqual([]);
    expect(root.tier).toBe(0);
    // The root asserts no scenario, so it renders whatever the defaults are on the day it builds.
    expect(root.config).toEqual({});
  });

  it('names a device per route below the root', () => {
    for (const route of prerenderRoutes().slice(1)) {
      expect(route.tier).toBe(1);
      expect(route.segments).toHaveLength(1);
      expect(route.config.deviceId).toBe(route.segments[0]);
    }
  });

  /**
   * The precondition the whole Phase 2 slice rests on. Three same-class devices would render
   * three near-identical pages, and the leak the slice exists to catch — one route rendering the
   * previous route's leftovers — would be invisible in the diff.
   */
  it('spans three device classes', () => {
    const classes = prerenderRoutes()
      .slice(1)
      .map((route) => getDevice(route.segments[0]).class);
    expect(new Set(classes).size).toBe(classes.length);
    expect(classes).toHaveLength(3);
  });

  it('prerenders only shipping hardware', () => {
    // A pre-release spec must stay visibly labelled, and a page of its own is the one place the
    // label has no surrounding context to carry it.
    for (const route of prerenderRoutes().slice(1)) {
      expect(getDevice(route.segments[0]).status).toBe('shipping');
    }
  });

  it('resolves every device id against the catalog', () => {
    for (const route of prerenderRoutes().slice(1)) {
      expect(() => getDevice(route.segments[0])).not.toThrow();
    }
  });

  it('gives every route a distinct path, title and description', () => {
    const routes = prerenderRoutes();
    for (const field of ['title', 'description'] as const) {
      expect(new Set(routes.map((route) => route[field])).size).toBe(routes.length);
    }
    expect(new Set(routes.map((route) => routePath(route, '/'))).size).toBe(routes.length);
  });

  it('names the device and its headline figures in the description', () => {
    const route = prerenderRoutes()[1];
    const device = getDevice(route.segments[0]);
    expect(route.title).toContain(device.name);
    expect(route.description).toContain(device.name);
    expect(route.description).toMatch(/\d+ GiB at \d+ GB\/s/);
  });
});

describe('routePath', () => {
  it('is the base itself for the root', () => {
    expect(routePath(prerenderRoutes()[0], '/headroom/')).toBe('/headroom/');
  });

  it('ends in a slash, because a directory of index.html files is what Pages serves', () => {
    const route = prerenderRoutes()[1];
    expect(routePath(route, '/headroom/')).toBe(`/headroom/${route.segments[0]}/`);
    expect(routePath(route, '/')).toBe(`/${route.segments[0]}/`);
  });

  it('tolerates a base without its trailing slash', () => {
    expect(routePath(prerenderRoutes()[0], '/headroom')).toBe('/headroom/');
  });
});

describe('configFromPath', () => {
  const deviceId = DEVICES[1].id;

  it('reads a device out of a one-segment path', () => {
    expect(configFromPath(`/${deviceId}/`, '/')).toEqual({ deviceId });
    expect(configFromPath(`/headroom/${deviceId}/`, '/headroom/')).toEqual({ deviceId });
  });

  it('reads the same device with or without a trailing slash', () => {
    expect(configFromPath(`/${deviceId}`, '/')).toEqual({ deviceId });
  });

  it('treats a directory and its index.html as the same page', () => {
    expect(configFromPath(`/${deviceId}/index.html`, '/')).toEqual({ deviceId });
  });

  it('claims nothing for the root', () => {
    expect(configFromPath('/', '/')).toEqual({});
    expect(configFromPath('/headroom/', '/headroom/')).toEqual({});
  });

  it('claims nothing for an unknown segment', () => {
    expect(configFromPath('/not-a-device/', '/')).toEqual({});
  });

  it('claims nothing for a path deeper than the routes it knows', () => {
    // 404.html answers at arbitrary depth, and it must boot as the default scenario rather than
    // as whichever segment happens to look like an id.
    expect(configFromPath(`/${deviceId}/some/model/`, '/')).toEqual({});
  });

  it('follows a device id that has been renamed', () => {
    expect(configFromPath('/rtx-a6000-ada/', '/')).toEqual({ deviceId: 'rtx-6000-ada' });
  });

  it('does not read a device out of a path outside the base', () => {
    // Not this site's page at all: `/rtx-5090/` under a `/headroom/` base is somebody else's.
    expect(configFromPath(`/${deviceId}/`, '/headroom/')).toEqual({});
  });

  it('survives a malformed escape rather than throwing out of the store initializer', () => {
    expect(() => configFromPath('/%/', '/')).not.toThrow();
    expect(configFromPath('/%/', '/')).toEqual({});
  });

  it('round-trips every route it emits', () => {
    for (const route of prerenderRoutes()) {
      expect(configFromPath(routePath(route, '/headroom/'), '/headroom/')).toEqual(route.config);
    }
  });
});
