import { beforeEach, vi } from 'vitest';
import { comparisonGrid } from '@/data/catalog';

/**
 * The bounded comparison grid every app-level suite renders by default, and the opt-in back to
 * the real one ([#101](https://github.com/MrZoller/bench/issues/101),
 * [#115](https://github.com/MrZoller/bench/issues/115)).
 *
 * The Matrix is models × devices and the app-level tests render the whole page, so every test
 * pays for the whole grid — 1,470 buttons, each with a full-sentence `aria-label` that
 * `getByRole` computes on every name-matched query. #78 and #77 grew both axes in one sweep and
 * took the old single-file suite from 42s to about fourteen minutes on CI, for changes that
 * touched no component; the per-test timeout was raised twice on the way and must not be raised
 * again.
 *
 * So the grid is bounded by default and the real one is opted into. **The polarity is the whole
 * design.** Bounding it by default makes the wall clock a constant — {@link BOUNDED_CELLS} cells
 * however large the catalog grows, which is the property #101 asks for: a change that touches no
 * component cannot fail CI on grid size. Opting in makes every full-grid case a written decision
 * rather than an accident, which matters because the full grid is what caught #52's roving tab
 * index and #64's rotated header, and a shrunken grid would have passed both.
 *
 * The line is not "does this test mention the Matrix" but **"does this test assert something
 * about the grid"** — see {@link atFullGrid}. Everything else merely renders it.
 *
 * **What an importing file must do itself**: declare the mock. `vi.mock` is hoisted per test
 * file by the transformer and cannot ride an import, so every consumer carries the same three
 * lines —
 *
 *     vi.mock('@/data/catalog', async (importOriginal) => {
 *       const actual = await importOriginal<typeof import('@/data/catalog')>();
 *       return { ...actual, comparisonGrid: vi.fn(actual.comparisonGrid) };
 *     });
 *
 * — and then calls {@link boundGridByDefault} once at the top level. `src/test/grid.test.ts` holds
 * the fixture's own preconditions, so a fixture that quietly stops matching the catalog fails
 * there rather than making some other file's assertion vacuous.
 */

/** The real extent, kept so {@link atFullGrid} restores it rather than re-deriving it. */
export const realComparisonGrid = (
  await vi.importActual<typeof import('@/data/catalog')>('@/data/catalog')
).comparisonGrid;

/**
 * The rows and columns the bounded grid renders, chosen so it is small and still not degenerate.
 *
 * Every one of these is here for a property some suite reads off the grid, and dropping any of
 * them makes an assertion vacuous rather than red:
 *
 * - `openai/gpt-oss-120b` on `dgx-spark` is `DEFAULT_CONFIG`, so the grid contains the marked
 *   cell — without it `isCurrent` is false everywhere and the selection mark, its legend key and
 *   `aria-current` all have nothing to test.
 * - Three device classes in file order, so the class bands are a run of three rather than one,
 *   and `separated` is non-empty.
 * - `epyc-9654` is a column vLLM cannot drive, so the struck-heading branch is reachable.
 * - `rtx-3060-12gb` at 12 GiB refuses most rows on counted bytes while `rtx-5090` holds them, so
 *   both the ran and the did-not-fit inks are painted.
 * - `Qwen/Qwen3-8B` is dense, so the MXFP4 default substitutes on it and the stand-in note fires.
 *
 * Ids rather than indices, and asserted to resolve in `grid.test.ts`: a slice of the catalog is
 * a fixture that silently becomes a different fixture the next time the catalog moves.
 */
export const BOUNDED_MODEL_IDS = [
  'openai/gpt-oss-120b',
  'Qwen/Qwen3-8B',
  'deepseek-ai/DeepSeek-R1',
];
export const BOUNDED_DEVICE_IDS = ['rtx-5090', 'rtx-3060-12gb', 'dgx-spark', 'epyc-9654'];
export const BOUNDED_CELLS = BOUNDED_MODEL_IDS.length * BOUNDED_DEVICE_IDS.length;

export function boundedGrid(): ReturnType<typeof realComparisonGrid> {
  const real = realComparisonGrid();
  return {
    // Filtered out of the real extent rather than looked up, so both lists keep the order the
    // component is entitled to assume — popularity for the rows, file order for the columns.
    models: real.models.filter((m) => BOUNDED_MODEL_IDS.includes(m.id)),
    devices: real.devices.filter((d) => BOUNDED_DEVICE_IDS.includes(d.id)),
  };
}

/**
 * Render this test's Matrix at the real catalog extent.
 *
 * Call it before `render`, in any test whose subject is the grid itself: its size, its column
 * headings, its bands, its tab sequence, or a sweep over the cells it paints. The rule of thumb
 * is whether the assertion would still mean something with twelve cells — if it would, leave it
 * bounded.
 */
export function atFullGrid() {
  vi.mocked(comparisonGrid).mockImplementation(realComparisonGrid);
}

/** Registers the bounded default for every test in the importing file. */
export function boundGridByDefault() {
  beforeEach(() => {
    vi.mocked(comparisonGrid).mockImplementation(boundedGrid);
  });
}
