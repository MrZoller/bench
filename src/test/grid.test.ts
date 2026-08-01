import { describe, expect, it, vi } from 'vitest';
import { DEFAULT_CONFIG } from '@/store/config';
import {
  BOUNDED_CELLS,
  BOUNDED_DEVICE_IDS,
  BOUNDED_MODEL_IDS,
  boundedGrid,
  realComparisonGrid,
} from './grid';

// The mock every consumer of src/test/grid.ts declares — see its module docblock for why the
// declaration cannot ride the import.
vi.mock('@/data/catalog', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/data/catalog')>();
  return { ...actual, comparisonGrid: vi.fn(actual.comparisonGrid) };
});

/**
 * The fixture's own preconditions, since a fixture that quietly stops matching the catalog is the
 * failure the old single-file suite had twice in other forms — a sweep that filtered on a field
 * the type does not have, and an exemption list that matched nothing. Both reported compliance
 * over zero cases. Held here, beside the fixture, so the suites that render it can assume it.
 */
describe('the bounded grid the app-level suites render', () => {
  it('names rows and columns the catalog still has', () => {
    const { models, devices } = boundedGrid();
    expect(models.map((m) => m.id)).toEqual(expect.arrayContaining(BOUNDED_MODEL_IDS));
    expect(devices.map((d) => d.id)).toEqual(expect.arrayContaining(BOUNDED_DEVICE_IDS));
    expect(models).toHaveLength(BOUNDED_MODEL_IDS.length);
    expect(devices).toHaveLength(BOUNDED_DEVICE_IDS.length);
  });

  it('is a constant the catalog cannot grow', () => {
    // The property #101 asks for, stated as an assertion: the bounded grid is 12 cells whatever
    // the catalog does next, so a change that touches no component cannot fail CI on grid size.
    expect(BOUNDED_CELLS).toBe(12);
    const real = realComparisonGrid();
    // And it really is a reduction — a fixture equal to the catalog would satisfy everything
    // above while restoring the whole cost.
    expect(real.models.length * real.devices.length).toBeGreaterThan(BOUNDED_CELLS * 10);
  });

  it('spans the three device classes, in the catalog’s own order', () => {
    const classes = boundedGrid().devices.map((d) => d.class);
    expect(classes).toEqual(['discrete-gpu', 'discrete-gpu', 'unified-soc', 'cpu-ram']);
  });

  it('contains the default scenario, so a cell is marked', () => {
    const { models, devices } = boundedGrid();
    expect(models.map((m) => m.id)).toContain(DEFAULT_CONFIG.modelId);
    expect(devices.map((d) => d.id)).toContain(DEFAULT_CONFIG.deviceId);
  });
});
