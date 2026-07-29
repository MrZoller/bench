import { expect, test } from '@playwright/test';

/**
 * The Matrix's device headers are rotated 45 degrees and taken out of flow, so the row's height is
 * a *computed guess* — `ceil(longest * 8 * sin(45)) + 20`, where 8px per character is an
 * estimate at this type size rather than a measurement. The comment in the component says it errs
 * long on purpose, because the cost of erring short is a header that clips.
 *
 * jsdom cannot check that: it has no layout, so every rotated label is zero by zero there and the
 * estimate is unfalsifiable. Whether 6.5 is still right after a font, a type scale or a catalogue
 * change is exactly the kind of thing that breaks silently.
 */

/**
 * At the default root, and again at 200% text.
 *
 * The second case is the one that was broken (#44). The labels are `text-xs`, so their width — and
 * therefore the height their rotation needs — scales with the root font size, while the reserved
 * height was a character count times a constant in *CSS pixels*. At a 32px root the text doubled
 * and the row it has to fit in did not, so the `overflow-x-auto` wrapper clipped the device names:
 * precisely the failure the rotation was introduced to prevent, reappearing at the one setting a
 * low-vision reader would be using. `headerBand` is in `rem` now — both of its lengths.
 *
 * Parameterised rather than copied, because the first version of this check existed only at the
 * default root and passed throughout — the estimate was never wrong at 16px.
 */
for (const root of [16, 32]) {
  test(`the header row is tall enough for the longest rotated label at a ${root}px root`, async ({
    page,
  }) => {
    await page.goto('/');
    if (root !== 16) {
      await page.evaluate((px) => {
        document.documentElement.style.fontSize = `${px}px`;
      }, root);
      // Confirms the scaling landed before anything is measured against it — otherwise the 32px
      // case silently re-runs the 16px one and passes for the wrong reason.
      await page.waitForFunction(
        (px) => getComputedStyle(document.documentElement).fontSize === `${px}px`,
        root
      );
    }

    const headers = page
      .getByRole('region', { name: /every model on every machine/i })
      .locator('thead th');
    await expect(headers.first()).toBeVisible();

    const count = await headers.count();
    expect(count).toBeGreaterThan(1);

    const headerBox = await headers.nth(1).boundingBox();
    expect(headerBox).not.toBeNull();

    // Every rotated label has to sit inside the row it belongs to. The labels are absolutely
    // positioned and rotated, so their painted box is what matters, not their text length.
    for (let i = 1; i < count; i++) {
      const label = headers.nth(i).locator('span');
      const box = await label.boundingBox();
      expect(box, `header ${i} has no laid-out label`).not.toBeNull();

      // The rotation origin is the label's bottom-right corner, so it extends *upward* from the
      // bottom of the cell. Clipping shows up as the label's top escaping above the row's top.
      expect(box!.y, `header ${i} label overflows the row`).toBeGreaterThanOrEqual(
        headerBox!.y - 1
      );
      expect(box!.height).toBeGreaterThan(0);
    }
  });
}

/**
 * And that erring long has not turned into erring *absurdly* long — a header taking half the
 * viewport would be its own defect, and nothing else would catch it.
 */
test('does not reserve more header height than the labels need', async ({ page }) => {
  await page.goto('/');

  const headers = page
    .getByRole('region', { name: /every model on every machine/i })
    .locator('thead th');
  await expect(headers.first()).toBeVisible();

  const headerBox = await headers.nth(1).boundingBox();
  const tallest = await headers.evaluateAll((cells) =>
    Math.max(
      ...cells.map((cell) => {
        const span = cell.querySelector('span');
        return span ? span.getBoundingClientRect().height : 0;
      })
    )
  );

  expect(headerBox!.height).toBeGreaterThanOrEqual(tallest);

  /**
   * A *ratio*, not a fixed slack, and that distinction cost a round to see.
   *
   * The reservation is font-independent — a per-character estimate times a character count — while
   * `tallest` is font-dependent. `tallest + 60` therefore tightens whenever the platform's
   * `system-ui` renders narrower than macOS's SF: at 247px reserved against 198.9px measured here,
   * a font 10% narrower would have taken the bound to 239 and failed on a header that is fine.
   *
   * As a ratio it moves with the measurement instead. 1.24 here, ~1.38 against a font 10% narrower,
   * and 1.6 still catches the failure this guards: someone raising the per-character constant "to
   * be safe" and reserving half the viewport.
   */
  expect(headerBox!.height).toBeLessThan(tallest * 1.6);
});

/**
 * The other axis, which is the half nothing was measuring (#64).
 *
 * A 45-degree rotation costs `sin(45) × label` of height and `cos(45) × label` of width, and only
 * the height was ever reserved — so the labels leaned up-and-*right* out of the scroll container.
 * At 1440 and 1024 the grid fitted its panel exactly and the container still reported 142px of
 * overflow, purely from header text: a grid that needs no horizontal scrolling got a scrollbar, and
 * the default view hid the last four device names — including the longest one, the one the 246px of
 * reserved height was calculated from. The labels now lean the other way, into the model column,
 * where the space is reserved and inside the container at every width.
 *
 * Three widths, because they catch different mistakes. 1280 is where the panel is at its widest and
 * the overflow was pure decoration. 1024 is the width the bug was filed at. 1060 is the interesting
 * one: the grid's own min-content is 865px (868 under a wider font) inside a 970px panel, so the
 * header's sideways extent and the columns are competing for the same 100px, and every version of
 * this fix that reserved a *trailing* lane for the labels — padding, or a `minmax(0, …)` grid track
 * fed by free space — either scrolled here or scrolled at some narrower width. Both were built and
 * measured before this one; the numbers are in the component and in ROADMAP.
 *
 * **Below about 950 the premise stops holding, and that is honest.** The grid's own min-content is
 * 865px, so a panel narrower than that scrolls because the *grid* does not fit, which is what a
 * scroll container is for. At 960 the premise holds by 4.6px here and by 2px under `'Courier New'`,
 * which is a measurement of this machine rather than of the layout — the sweep below covers those
 * widths instead, by the general rule rather than by a hard-coded fit.
 *
 * jsdom reports every one of these widths as 0, which is why this survived.
 */
for (const width of [1280, 1060, 1024]) {
  test(`the grid does not scroll sideways for a header leaning past it at ${width}px`, async ({
    page,
  }) => {
    await page.setViewportSize({ width, height: 800 });
    await page.goto('/');

    const scroller = page
      .getByRole('region', { name: /every model on every machine/i })
      .locator('div.overflow-x-auto');
    await expect(scroller).toBeVisible();

    const measured = await scroller.evaluate((el) => {
      const table = el.querySelector('table')!;
      const labels = [...el.querySelectorAll('thead th span[title]')];
      const box = el.getBoundingClientRect();
      return {
        client: el.clientWidth,
        scroll: el.scrollWidth,
        table: table.getBoundingClientRect().width,
        // The container's own right edge, at the resting scroll position — what a reader can see
        // without panning.
        visibleRight: box.left + el.clientWidth,
        rightmostLabel: Math.max(...labels.map((l) => l.getBoundingClientRect().right)),
        labelCount: labels.length,
      };
    });

    expect(measured.labelCount).toBeGreaterThan(1);

    /**
     * The premise, asserted rather than assumed: the grid itself fits the panel here. Without it
     * this test would pass on any layout wide enough to scroll for honest reasons, which is the
     * "measures the wrong thing and cannot fail" shape the roadmap keeps recording.
     */
    expect(measured.table, 'the grid no longer fits its panel at this width').toBeLessThanOrEqual(
      measured.client + 1
    );

    // So nothing may scroll. One pixel of tolerance for a fractional layout; the defect was 142.
    expect(measured.scroll, 'the grid scrolls sideways for its header alone').toBeLessThanOrEqual(
      measured.client + 1
    );

    // And the last thing the header paints is inside what the reader can see, rather than 116px
    // past it.
    expect(
      measured.rightmostLabel,
      'a device label is painted outside the scroll container'
    ).toBeLessThanOrEqual(measured.visibleRight + 1);
  });
}

/**
 * The bill for leaning the other way, and why it is a reservation rather than an assumption.
 *
 * Left-leaning labels spend `cos(45) × label` over the model-name column instead of past the last
 * device column. That column is in flow and inside the container, so the scrollbar problem goes away
 * at every width — but the failure mode if the room runs out is *worse* than the one it replaces.
 * Overflow to the right of a scroll container is at least reachable by panning. Overflow to the left
 * is not scrollable at all: the name is simply gone, at every scroll position, with nothing in the
 * layout to suggest it was ever there.
 *
 * So the model column carries an explicit `min-width` of the same length the band is derived from,
 * and this is its contract: the reservation covers the furthest any label actually leans. Deleting
 * the `min-width` fails the first assertion (a computed `auto` parses to NaN); shortening it below
 * what the labels need fails it by the shortfall; a label that escapes anyway fails the second.
 *
 * 1024 and 390, because the reservation only binds where the model names are narrower than it — at
 * 1440 the auto table layout hands the column 338px and the guard is inert, which would make this
 * pass without testing anything. At 390 the column sits at exactly its reserved minimum.
 */
for (const width of [1024, 390]) {
  test(`the header leans into the room reserved for it at ${width}px`, async ({ page }) => {
    await page.setViewportSize({ width, height: 800 });
    await page.goto('/');
    await expect(page.locator('table td button').first()).toBeVisible();

    const measured = await page
      .locator('table[role="grid"]')
      .evaluate((table: HTMLTableElement) => {
        const stub = table.querySelector('thead th')!;
        const labels = [...table.querySelectorAll('thead th span[title]')];
        const lean = labels.map((label) => {
          // The anchor is `right-1/2`, turned about its bottom-right corner: the label's own column
          // centre. How far left it reaches from there is the quantity the column has to cover.
          const column = label.parentElement!.getBoundingClientRect();
          return {
            name: label.textContent ?? '',
            reach: (column.left + column.right) / 2 - label.getBoundingClientRect().left,
            escapes: table.getBoundingClientRect().left - label.getBoundingClientRect().left,
          };
        });
        const furthest = lean.reduce((a, b) => (b.reach > a.reach ? b : a));
        const escaped = lean.reduce((a, b) => (b.escapes > a.escapes ? b : a));
        return {
          count: labels.length,
          reserved: parseFloat(getComputedStyle(stub).minWidth),
          column: stub.getBoundingClientRect().width,
          furthest,
          escaped,
        };
      });

    expect(measured.count).toBeGreaterThan(1);

    expect(
      measured.reserved,
      `the model column reserves ${measured.reserved}px and "${measured.furthest.name}" leans ${measured.furthest.reach.toFixed(1)}px over it`
    ).toBeGreaterThanOrEqual(measured.furthest.reach);

    expect(
      measured.escaped.escapes,
      `"${measured.escaped.name}" is painted past the container's left edge, where no scroll position reaches it`
    ).toBeLessThanOrEqual(1);
  });
}

/**
 * The same claim, generalised: **no scroll container on this page scrolls for its own decoration.**
 *
 * The Matrix header is where this defect was found, which is why the sweep lives here, but the page
 * carries three of these containers — the grid, the Envelope's plot, and the Envelope's table behind
 * its disclosure — and the mechanism that produced 142px of phantom overflow is available to all of
 * them. Anything painted out of flow (a rotation, an absolute label, a ring, a shadow) enlarges the
 * scrollable area without enlarging any child, and the container then offers to scroll to something
 * the layout never asked for.
 *
 * So: a container may scroll only as far as its **in-flow** content reaches — `scrollWidth` against
 * that reach, and *not* against `clientWidth`, which is the trap this sweep fell into first. Every
 * scrolling container on a phone has in-flow content wider than its box; comparing the content to the
 * box therefore passes on anything that scrolls at all, and the first version of this test was green
 * against the filed defect at 390px with 142px of decoration overflow in front of it. Half a
 * parameterised sweep carrying no signal is worse than no sweep, because it reads as coverage.
 *
 * Out-of-flow boxes are excluded from the reach deliberately, and that exclusion is the other half of
 * the test: `scrollWidth` on the children counts the rotated labels, because they are descendants of
 * the table, so a sweep built from child `scrollWidth` also passed against the defect. Decoration has
 * to fit inside the box; only layout may ask for a scrollbar.
 *
 * Three widths. 1280 is the wide case. 390 is where the rule earns its keep: the grid scrolls there
 * for honest reasons under every implementation, so this is the only assertion in the file that can
 * tell an honest 865px of scroll from 920px of it. 960 is the window the *previous* fix broke — free
 * space had run out, the grid fitted its container, and the container scrolled 50px anyway — and this
 * rule catches it without needing to know at what width the grid stops fitting, which is the property
 * that made the hard-coded geometry widths miss it.
 */
for (const [width, height] of [
  [1280, 720],
  [960, 800],
  [390, 844],
]) {
  test(`no panel scrolls sideways for its own decoration at ${width}px`, async ({ page }) => {
    await page.setViewportSize({ width, height });
    await page.goto('/');
    await expect(page.locator('table td button').first()).toBeVisible();

    /**
     * Every disclosure open first, because one of the three containers is behind one.
     * `aria-expanded="false"` rather than a name or a class: it is what a disclosure *is*, so a
     * fourth one added later is swept without anyone remembering to add it here.
     */
    const collapsed = page.locator('button[aria-expanded="false"]');
    for (let i = await collapsed.count(); i > 0; i = await collapsed.count()) {
      await collapsed.first().click();
      // A toggle that renders nothing new would loop forever otherwise.
      expect(await collapsed.count(), 'a disclosure did not expand when clicked').toBeLessThan(i);
    }

    const containers = await page.evaluate(() =>
      [...document.querySelectorAll<HTMLElement>('*')]
        .filter((el) => ['auto', 'scroll'].includes(getComputedStyle(el).overflowX))
        .map((el) => {
          const left = el.getBoundingClientRect().left;
          const inFlow = [...el.querySelectorAll<HTMLElement>('*')].filter(
            (child) => !['absolute', 'fixed'].includes(getComputedStyle(child).position)
          );
          return {
            where: el.className.toString().slice(0, 60),
            client: el.clientWidth,
            scroll: el.scrollWidth,
            // Measured from the container's own left edge, at the resting scroll position.
            content: Math.max(
              0,
              ...inFlow.map((child) => child.getBoundingClientRect().right - left)
            ),
          };
        })
    );

    // Three with the disclosures open — the grid, the Envelope's plot, the Envelope's table.
    // Finding one would make every assertion below it a description of nothing.
    expect(containers.length).toBeGreaterThanOrEqual(3);

    for (const c of containers) {
      // A container that does not scroll cannot scroll for decoration, and `scrollWidth` is never
      // less than `clientWidth`, so an unscrolled 300px of content in a 900px box is not a finding.
      if (c.scroll <= c.client + 1) continue;
      expect(
        c.scroll,
        `"${c.where}" scrolls to ${c.scroll} in a ${c.client} box, and its in-flow content reaches only ${c.content}`
      ).toBeLessThanOrEqual(c.content + 1);
    }
  });
}

/**
 * The reason the rotation exists at all: horizontally these clipped to "GeForc…" four times over,
 * and two Mac Studio variants differing only in a trailing capacity suffix became indistinguishable.
 * A header that cannot tell its own columns apart is worse than none.
 */
test('every device column is distinguishable by its label', async ({ page }) => {
  await page.goto('/');

  const headers = page
    .getByRole('region', { name: /every model on every machine/i })
    .locator('thead th');
  await expect(headers.first()).toBeVisible();

  // The first cell is the row-header stub, and its only span is `sr-only` — deliberately clipped to
  // a 1px box, so it would read as truncated on the geometry check below.
  const labels = await headers.evaluateAll((cells) =>
    cells.slice(1).map((cell) => {
      const span = cell.querySelector('span');
      if (!span) return { text: '', display: 'none', clipped: false };
      return {
        text: span.textContent ?? '',
        /**
         * Both properties return 0 on an element with no layout box, and Blink gives an inline box
         * no `clientWidth` at all — so `0 > 0 + 1` is false and the clipping check below silently
         * becomes a tautology. That is not a hypothetical shape: the regression this test guards
         * against is a revert to truncated in-flow headers, which is exactly what would put the
         * label back to `display: inline`. Asserted so the measurement fails loudly rather than
         * degrading into one that cannot fail — the defect this whole test was rewritten to escape.
         */
        display: getComputedStyle(span).display,
        /**
         * Clipping has to be read off the layout, not the text. `text-overflow: ellipsis` paints a
         * glyph the browser never writes back into the DOM, so `textContent` stays whole while the
         * column becomes unreadable — the assertion this replaces could not fail.
         *
         * A 1px tolerance because both properties are integers: a shrink-to-fit box whose content
         * measures 198.6px can round its scroll and client widths apart by one. Real truncation
         * overflows by a whole character, 6–8px at this type size.
         */
        clipped: span.scrollWidth > span.clientWidth + 1,
      };
    })
  );

  const visible = labels.filter((l) => l.text.trim().length > 0);
  expect(visible.length).toBeGreaterThan(1);
  expect(new Set(visible.map((l) => l.text)).size).toBe(visible.length);
  for (const label of visible) {
    expect(
      label.display,
      `the header "${label.text}" is inline, so the clipping check cannot measure it`
    ).not.toBe('inline');
    expect(label.clipped, `the header "${label.text}" is clipped by its own box`).toBe(false);
  }
});
