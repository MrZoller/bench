import { expect, test, type Locator, type Page } from '@playwright/test';
import { marks } from '@/design/tokens';

/**
 * The budget bar's ceiling rule, which is always drawn on top of a fill.
 *
 * Issue #73. `scale = max(used, ceiling)` keeps an over-budget stack on screen, which is the right
 * rule — and it has two consequences the rule has to survive. The segments occupy the *whole* width
 * whenever the bar overflows, so there is no empty track left for the reference line to sit on; and
 * its position is a function of the overshoot, 50% in at 2x and 6.9% in at 14x, so which fill it
 * lands on is not something the design chooses. A 2px dashed critical-red line over `--color-weights`
 * blue is the worst pairing this palette contains, because the gaps between the dashes show the fill
 * — and this is the reference the whole bar exists to be measured against.
 *
 * jsdom cannot answer any of this: it has no layout engine, so every width here is 0 and a spec
 * written against it would assert 0 ≈ 6.9% of 0 and pass against unfixed markup. Hence a browser.
 *
 * **What it cannot answer either, stated rather than implied.** "Distinguishable against the
 * segment beneath it" is ultimately a claim about pixels, and reading them back would mean
 * decoding a screenshot — so this asserts the *mechanism* instead: a track-coloured halo of the
 * same width as the gap the segments already keep between themselves, on both sides of the rule,
 * with the rule still at the true ceiling position. That is falsifiable in the direction that
 * matters — restoring the bare 2px line fails on the halo's width and on its background — and it
 * is the same shape of compromise `hybrid-targets.spec.ts` records for the pointer queries.
 */

/**
 * Two scenarios, because one position cannot tell a measurement from a constant.
 *
 * The first is the issue's own URL, at 448 GiB against a 31 GiB ceiling. The second is half the
 * catalogue away — an M3 Ultra whose ceiling is 192 GiB — and puts the rule near the middle of the
 * bar. A spec that only ever looked at the 14x case would pass against a hardcoded `left`.
 */
const SCENARIOS = [
  {
    name: 'fourteen times over on a 5090',
    url: '/?m=deepseek-ai/DeepSeek-V3&q=q4_k_m&r=llama.cpp&d=rtx-5090&n=1&ctx=131072&u=8&p=8192&kv=fp16',
  },
  {
    name: 'twice over on an M3 Ultra',
    url: '/?m=deepseek-ai/DeepSeek-V3&q=q4_k_m&r=mlx&d=mac-studio-m3-ultra-256&n=1&ctx=32768&u=1&p=8192&kv=fp16',
  },
] as const;

const budget = (page: Page) => page.getByRole('region', { name: /memory budget/i });

/** The bar itself, which is the `role="img"` carrying the whole stack as a text alternative. */
const bar = (page: Page) => budget(page).getByRole('img', { name: /allocatable used/i });

/**
 * The halo, and the dashed rule inside it.
 *
 * Addressed structurally — the bar has exactly two children, the segment row and the rule — so
 * every test below confirms it found the rule by asserting the dashed border before trusting any
 * geometry. A spec that measures the wrong element passes for the wrong reason, which this suite
 * has already produced three of.
 */
const halo = (page: Page) => bar(page).locator(':scope > div').last();
const rule = (page: Page) => halo(page).locator(':scope > div');

/** Box and the computed properties that carry the separation, in one round trip. */
async function measure(locator: Locator) {
  return locator.evaluate((el) => {
    const style = getComputedStyle(el);
    const box = el.getBoundingClientRect();
    return {
      left: box.left,
      right: box.right,
      width: box.width,
      height: box.height,
      background: style.backgroundColor,
      borderLeftStyle: style.borderLeftStyle,
      borderLeftWidth: style.borderLeftWidth,
      borderLeftColor: style.borderLeftColor,
    };
  });
}

/**
 * The two figures the header states, which is what the expected position is derived from.
 *
 * Derived rather than hardcoded at 6.9%: the model's file size comes from a generated catalogue, so
 * a refresh that moves DeepSeek V3 by a gibibyte would fail a literal — a false alarm that teaches
 * people to re-bless the number instead of reading it. The header and the bar are drawn from one
 * pair of values, so the ratio between them is the position the rule must be at.
 */
async function statedFigures(page: Page) {
  const text = (await budget(page).locator('header p').innerText()).replace(/\s+/g, ' ');
  const match = text.match(/([\d.]+) GiB \/ ([\d.]+) GiB/);
  expect(match, `the header did not state a used/ceiling pair — read "${text}"`).not.toBeNull();
  return { used: Number(match![1]), ceiling: Number(match![2]) };
}

for (const { name, url } of SCENARIOS) {
  test.describe(name, () => {
    test.beforeEach(async ({ page }) => {
      await page.goto(url);
      await expect(bar(page)).toBeVisible();
    });

    /**
     * The precondition, asserted before anything is measured: the scenario really is over budget
     * *and* the rule really lands inside a fill. On a configuration that fits there is no rule at
     * all, so every assertion below would be measuring the segment row instead — and passing.
     */
    test('the rule is drawn, and it lands inside the weights fill', async ({ page }) => {
      const line = await measure(rule(page));
      expect(line.borderLeftStyle, 'this is not the ceiling rule').toBe('dashed');

      // Both scenarios put it inside the weights block — `--color-weights` blue is the fill the
      // dashes lose the most contrast against, so it is the one worth pinning. This is what makes
      // the halo load-bearing rather than decorative.
      const weights = await measure(
        bar(page).locator(':scope > div').first().locator(':scope > div').first()
      );
      expect(weights.width, 'the weights segment is not laid out').toBeGreaterThan(0);
      expect(line.left, 'the rule sits left of the fill').toBeGreaterThan(weights.left);
      expect(line.right, 'the rule sits right of the fill').toBeLessThan(weights.right);
    });

    test('the rule sits where the ceiling is, as a share of the whole bar', async ({ page }) => {
      const { used, ceiling } = await statedFigures(page);
      const track = await measure(bar(page));
      const line = await measure(rule(page));

      // `scale` is the larger of the two, which is `used` whenever the bar overflows — so the rule
      // is at ceiling/used of the bar's width. 6.9% in the issue's case; ~50% in the other.
      const expected = track.left + (ceiling / used) * track.width;
      /**
       * A few pixels of tolerance, and the reason is the derivation rather than the layout: the
       * header prints whole gibibytes above 10, so a ratio rebuilt from it carries up to ~0.15% of
       * the bar. That is nowhere near loose enough to hide the failure this measures — a rule at a
       * constant offset misses the other scenario by hundreds of pixels.
       */
      expect(
        Math.abs(line.left - expected),
        `expected the rule at ${((ceiling / used) * 100).toFixed(1)}% of the bar`
      ).toBeLessThan(3);

      // Full height, or it reads as a tick on one edge rather than a limit across the stack.
      expect(line.height).toBeCloseTo(track.height, 1);
    });

    test('the rule keeps a track-coloured halo on both sides of itself', async ({ page }) => {
      const track = await measure(bar(page));
      const band = await measure(halo(page));
      const line = await measure(rule(page));

      expect(line.borderLeftStyle).toBe('dashed');
      expect(line.borderLeftWidth).toBe(`${marks.lineWidth}px`);
      expect(line.width, 'the rule is not the declared line weight').toBeCloseTo(
        marks.lineWidth,
        1
      );

      // One gap either side of the line — the same 2px the segments keep between themselves, so
      // the dashes read against the track wherever the rule falls.
      expect(band.width, 'the halo is missing or the wrong width').toBeCloseTo(
        marks.lineWidth + marks.gap * 2,
        1
      );
      expect(line.left - band.left, 'no halo left of the rule').toBeCloseTo(marks.gap, 1);
      expect(band.right - line.right, 'no halo right of the rule').toBeCloseTo(marks.gap, 1);

      // Read off the bar rather than written as a hex, so it cannot drift from the token the
      // segment gaps already reveal.
      expect(band.background, 'the halo is not the track colour').toBe(track.background);
      expect(band.background).not.toBe('rgba(0, 0, 0, 0)');

      /**
       * And nothing of it is lost to the bar's clip — in these two positions, which is as far as
       * the promise goes, so it is asserted as an overhang rather than as `>= track.left`.
       *
       * The halo is a fixed `lineWidth + 2·gap` centred on the rule, and the bar clips its children,
       * so it survives whole exactly while the rule is one gap clear of each edge. Both scenarios
       * here are (6.9% and ~50%). A stack 0.5% over the ceiling is not: the rule lands at 99.5% and
       * the right 1–4px of separation is clipped away, and one hundreds of times over does the same
       * on the left. That is what a fixed-width mark does near an edge, and the alternative is to
       * draw the ceiling where the ceiling is not — so a scenario added there should not read this
       * as a regression. The invariant that holds at every position is the pair of offsets above.
       */
      expect(
        Math.max(0, track.left - (line.left - marks.gap)),
        'the halo is clipped at the left edge'
      ).toBeLessThan(0.5);
      expect(
        Math.max(0, line.right + marks.gap - track.right),
        'the halo is clipped at the right edge'
      ).toBeLessThan(0.5);
    });
  });
}
