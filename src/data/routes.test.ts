import { describe, expect, it } from 'vitest';
import { configFromPath, prerenderRoutes, routePath, type PrerenderRoute } from './routes';
import { DEVICES, MODELS, getDevice, getModel, modelSlug, modelsByPopularity } from './catalog';

/** The routes of one tier, which is how every claim below is scoped. */
function tier(n: number): readonly PrerenderRoute[] {
  return prerenderRoutes().filter((route) => route.tier === n);
}

describe('prerenderRoutes', () => {
  it('starts at the root', () => {
    const [root] = prerenderRoutes();
    expect(root.segments).toEqual([]);
    expect(root.tier).toBe(0);
    // The root asserts no scenario, so it renders whatever the defaults are on the day it builds.
    expect(root.config).toEqual({});
  });

  /**
   * The inventory, stated as arithmetic rather than as a number.
   *
   * 199 today. Written against the catalog so it moves with it — a device row added to
   * `devices.json` should change what the site publishes without anybody editing a test — while
   * still failing if a *tier* changes shape, which is the thing that would need a decision.
   */
  it('is every device, every model, and a bounded product of the two', () => {
    expect(tier(0)).toHaveLength(1);
    expect(tier(1)).toHaveLength(DEVICES.length);
    expect(tier(2)).toHaveLength(MODELS.length);
    expect(tier(3)).toHaveLength(120);
    expect(prerenderRoutes()).toHaveLength(1 + DEVICES.length + MODELS.length + 120);
  });

  it('names a device per route in the device tier, in the catalog’s own order', () => {
    expect(tier(1).map((route) => route.segments)).toEqual(DEVICES.map((device) => [device.id]));
    for (const route of tier(1)) {
      expect(route.segments).toHaveLength(1);
      expect(route.config).toEqual({ deviceId: route.segments[0] });
    }
  });

  it('names a model per route in the model tier, under a prefix of its own', () => {
    expect(tier(2).map((route) => route.segments)).toEqual(
      modelsByPopularity().map((model) => ['m', modelSlug(model)])
    );
    for (const route of tier(2)) {
      expect(route.config).toEqual({ modelId: getModel(route.config.modelId!).id });
    }
  });

  /**
   * The precondition the pair tier rests on, and the reason it is not a flat slice.
   *
   * `devices.json` is grouped by class in display order, so its first twelve rows are twelve
   * NVIDIA cards — a shortlist that answers one question at twelve price points and says nothing
   * about the unified-memory and CPU machines whose whole appeal is that a large model fits. Class
   * also drives the runtime filter, the quant fallback and whether a rig can shard, so a
   * single-class shortlist would render twelve near-identical pages.
   */
  it('spreads the pair tier across all three device classes', () => {
    const devices = [...new Set(tier(3).map((route) => route.segments[0]))];
    expect(devices).toHaveLength(12);

    const counts = new Map<string, number>();
    for (const id of devices) {
      const { class: deviceClass } = getDevice(id);
      counts.set(deviceClass, (counts.get(deviceClass) ?? 0) + 1);
    }
    expect(Object.fromEntries(counts)).toEqual({
      'discrete-gpu': 6,
      'unified-soc': 4,
      'cpu-ram': 2,
    });

    // The leading rows of each class, which is what `$comment-order` in devices.json states the
    // order means — not a sample, and not the first twelve of the file.
    for (const [deviceClass, count] of [
      ['discrete-gpu', 6],
      ['unified-soc', 4],
      ['cpu-ram', 2],
    ] as const) {
      const leading = DEVICES.filter((device) => device.class === deviceClass).slice(0, count);
      expect(devices.filter((id) => getDevice(id).class === deviceClass)).toEqual(
        leading.map((device) => device.id)
      );
    }
  });

  /**
   * The pair tier's models track Hugging Face rather than a list somebody maintains. 13 of 35 rows
   * changed rank across a single week's refresh, so a written-down shortlist is wrong within the
   * month — and being the *prefix* of the model tier is what makes that checkable here.
   */
  it('pairs the ten most-downloaded models, and nothing else', () => {
    const shortlist = modelsByPopularity().slice(0, 10);
    for (const device of new Set(tier(3).map((route) => route.segments[0]))) {
      expect(
        tier(3)
          .filter((route) => route.segments[0] === device)
          .map((route) => route.segments[1])
      ).toEqual(shortlist.map(modelSlug));
    }
  });

  it('names both halves of a pair in its config', () => {
    for (const route of tier(3)) {
      expect(route.config).toEqual({
        deviceId: route.segments[0],
        modelId: getModel(route.config.modelId!).id,
      });
    }
  });

  it('resolves every device id and every model slug against the catalog', () => {
    for (const route of prerenderRoutes()) {
      if (route.config.deviceId) expect(() => getDevice(route.config.deviceId!)).not.toThrow();
      if (route.config.modelId) expect(() => getModel(route.config.modelId!)).not.toThrow();
    }
  });

  /**
   * Rumoured hardware gets a page and no invitation to it.
   *
   * The page is written because it works — the status label renders in the device panel exactly as
   * it does everywhere else, so a visitor who lands on it sees what they are looking at. What must
   * not happen is a search result for a machine nobody can buy, stripped of the label that made it
   * honest. So `indexable` is false for every route naming a non-shipping row, and
   * `sitemap.xml` is the only thing that reads it.
   */
  it('writes a page for rumoured hardware but does not advertise it', () => {
    const unshipped = DEVICES.filter((device) => device.status !== 'shipping');
    expect(unshipped.length).toBeGreaterThan(0);

    for (const route of prerenderRoutes()) {
      const device = route.config.deviceId ? getDevice(route.config.deviceId) : undefined;
      expect(route.indexable).toBe(device === undefined || device.status === 'shipping');
    }
    // And it really is written: the rumoured row has a device page and its pair pages too.
    for (const device of unshipped) {
      expect(prerenderRoutes().some((route) => route.segments[0] === device.id)).toBe(true);
    }
  });

  it('gives every route a distinct path, title and description', () => {
    const routes = prerenderRoutes();
    for (const field of ['title', 'description'] as const) {
      expect(new Set(routes.map((route) => route[field])).size).toBe(routes.length);
    }
    expect(new Set(routes.map((route) => routePath(route, '/'))).size).toBe(routes.length);
  });

  it('names the device and its headline figures in a device description', () => {
    const route = tier(1)[0];
    const device = getDevice(route.segments[0]);
    expect(route.title).toContain(device.name);
    expect(route.description).toContain(device.name);
    expect(route.description).toMatch(/\d+ GiB at \d+ GB\/s/);
  });

  it('names the model and its size in a model description', () => {
    const route = tier(2)[0];
    const model = getModel(route.config.modelId!);
    expect(route.title).toContain(model.name);
    expect(route.description).toContain(model.org);
    expect(route.description).toMatch(/[\d.]+B parameters/);
  });

  it('names both halves in a pair description', () => {
    const route = tier(3)[0];
    const device = getDevice(route.segments[0]);
    const model = getModel(route.config.modelId!);
    expect(route.title).toContain(device.name);
    expect(route.title).toContain(model.name);
    expect(route.description).toContain(device.name);
    expect(route.description).toContain(model.name);
  });
});

describe('routePath', () => {
  it('is the base itself for the root', () => {
    expect(routePath(prerenderRoutes()[0], '/headroom/')).toBe('/headroom/');
  });

  it('ends in a slash, because a directory of index.html files is what Pages serves', () => {
    const route = tier(1)[0];
    expect(routePath(route, '/headroom/')).toBe(`/headroom/${route.segments[0]}/`);
    expect(routePath(route, '/')).toBe(`/${route.segments[0]}/`);
  });

  it('joins the segments of a two-level route', () => {
    const route = tier(3)[0];
    expect(routePath(route, '/headroom/')).toBe(
      `/headroom/${route.segments[0]}/${route.segments[1]}/`
    );
  });

  it('tolerates a base without its trailing slash', () => {
    expect(routePath(prerenderRoutes()[0], '/headroom')).toBe('/headroom/');
  });
});

describe('configFromPath', () => {
  const deviceId = DEVICES[1].id;
  const model = MODELS[0];
  const slug = modelSlug(model);

  it('reads a device out of a one-segment path', () => {
    expect(configFromPath(`/${deviceId}/`, '/')).toEqual({ deviceId });
    expect(configFromPath(`/headroom/${deviceId}/`, '/headroom/')).toEqual({ deviceId });
  });

  it('reads a model out of the model namespace', () => {
    expect(configFromPath(`/m/${slug}/`, '/')).toEqual({ modelId: model.id });
    expect(configFromPath(`/headroom/m/${slug}/`, '/headroom/')).toEqual({ modelId: model.id });
  });

  it('reads both out of a pair path', () => {
    expect(configFromPath(`/${deviceId}/${slug}/`, '/')).toEqual({ deviceId, modelId: model.id });
  });

  it('reads a slug whatever case it was typed in', () => {
    // The slug is lowercased by definition, and the id it is made from is not — so a hand-typed
    // `/m/Qwen--Qwen3-8B/` is the spelling somebody would reach for first.
    expect(configFromPath(`/m/${slug.toUpperCase()}/`, '/')).toEqual({ modelId: model.id });
  });

  it('reads the same device with or without a trailing slash', () => {
    expect(configFromPath(`/${deviceId}`, '/')).toEqual({ deviceId });
  });

  it('treats a directory and its index.html as the same page', () => {
    expect(configFromPath(`/${deviceId}/index.html`, '/')).toEqual({ deviceId });
    expect(configFromPath(`/m/${slug}/index.html`, '/')).toEqual({ modelId: model.id });
  });

  it('claims nothing for the root', () => {
    expect(configFromPath('/', '/')).toEqual({});
    expect(configFromPath('/headroom/', '/headroom/')).toEqual({});
  });

  it('claims nothing for an unknown segment', () => {
    expect(configFromPath('/not-a-device/', '/')).toEqual({});
    expect(configFromPath('/m/not-a-model/', '/')).toEqual({});
  });

  /**
   * Half a route is not a route. Answering `/rtx-5090/nonsense/` with the RTX 5090's figures would
   * put a page the reader did not ask for under the address they did — the same failure the device
   * aliases exist to prevent, arrived at from the other end.
   */
  it('claims nothing when only one half of a pair resolves', () => {
    expect(configFromPath(`/${deviceId}/not-a-model/`, '/')).toEqual({});
    expect(configFromPath(`/not-a-device/${slug}/`, '/')).toEqual({});
  });

  it('claims nothing for a path deeper than the routes it knows', () => {
    // 404.html answers at arbitrary depth, and it must boot as the default scenario rather than
    // as whichever segment happens to look like an id.
    expect(configFromPath(`/${deviceId}/${slug}/more/`, '/')).toEqual({});
  });

  it('follows a device id that has been renamed', () => {
    expect(configFromPath('/rtx-a6000-ada/', '/')).toEqual({ deviceId: 'rtx-6000-ada' });
    expect(configFromPath(`/rtx-a6000-ada/${slug}/`, '/')).toEqual({
      deviceId: 'rtx-6000-ada',
      modelId: model.id,
    });
  });

  it('does not read a device out of a path outside the base', () => {
    // Not this site's page at all: `/rtx-5090/` under a `/headroom/` base is somebody else's.
    expect(configFromPath(`/${deviceId}/`, '/headroom/')).toEqual({});
  });

  it('survives a malformed escape rather than throwing out of the store initializer', () => {
    expect(() => configFromPath('/%/', '/')).not.toThrow();
    expect(configFromPath('/%/', '/')).toEqual({});
    expect(configFromPath('/%/%/', '/')).toEqual({});
  });

  /**
   * The property the whole hydrate turns on, asserted over every route rather than sampled.
   *
   * The server is handed `route.config`; the browser recomputes it from the address alone. If
   * those two ever disagree for one route, React discards that page's markup — and this is the
   * cheap exhaustive check that they cannot, which is why the hydration suite next door samples
   * shapes instead of sweeping all 199.
   */
  it('round-trips every route it emits', () => {
    for (const route of prerenderRoutes()) {
      expect(configFromPath(routePath(route, '/headroom/'), '/headroom/')).toEqual(route.config);
      expect(configFromPath(routePath(route, '/'), '/')).toEqual(route.config);
    }
  });
});
