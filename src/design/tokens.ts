/**
 * Design tokens — the single source of truth for bench's visual language.
 *
 * A dark instrument-panel chassis: a sibling to `~/code/wavefront`, not a clone. Wavefront runs
 * a phosphor-green accent; bench runs violet, so the two read as the same family without being
 * mistaken for each other.
 *
 * These are mirrored as CSS custom properties in `index.css` — keep the two in sync. Canvas code
 * reads from here, because `var(--color-*)` does not resolve inside a 2D context.
 *
 * **The palette is computed, not chosen.** Every categorical value below comes from the dataviz
 * reference ramps and was run through `validate_palette.js` against *this* surface (`#0f1117`),
 * not the reference's `#1a1a19` — contrast and lightness results only mean anything against the
 * surface a chart actually renders on. Two candidate palettes were discarded by the validator
 * rather than by eye:
 *
 *   - magenta + aqua adjacent in the budget stack: ΔE 1.6 under deuteranopia, effectively one
 *     colour for a deuteranope, despite looking unambiguous to me.
 *   - a violet accent beside the weights blue: ΔE 9.8 to *normal* vision, under the 15 floor.
 *
 * If you change a colour here, re-run the validator before committing it:
 *
 *   node scripts/validate_palette.js "<hex,hex,…>" --mode dark --surface "#0f1117" --pairs all
 */

export const colors = {
  // --- The chassis -----------------------------------------------------------------------
  /** Page plane, behind the panels. */
  bg: '#08090d',
  /** Chart and panel surface. Every palette check below is against this value. */
  surface: '#0f1117',
  surfaceRaised: '#161923',
  border: '#232735',
  /** Hairline grid — recessive by design; never competes with a mark. */
  grid: '#1b1f2b',
  /**
   * Edge of an interactive control, distinct from the panel hairline above.
   *
   * `border` is deliberately recessive — 1.18:1 against the raised fill — which is right for a
   * panel edge and wrong for a select. A control's boundary is what identifies it as
   * interactive, so it needs the 3:1 non-text minimum *before* it is focused, not after. This
   * step measures 3.41:1 against the fill and 3.67:1 against the panel.
   */
  controlBorder: '#646d88',

  // --- Ink -------------------------------------------------------------------------------
  // Static content: labels, titles, units, axis captions, and figures. Text never wears a
  // series colour — a coloured mark beside it carries the identity instead.
  text: '#e8eaf2',
  textMuted: '#9aa0b5',
  /**
   * Raised from `#6f7690` (4.20:1). This token labels every control and verdict — normal-sized
   * load-bearing text, so the threshold is 4.5:1 rather than the 3:1 that applies to a mark.
   * At 5.10:1 it still sits clearly below `textMuted` at 7.25:1, so the hierarchy survives.
   */
  textFaint: '#7d84a0',

  // --- Accent = live / interactive --------------------------------------------------------
  /**
   * Violet marks what changes or responds: slider handles, the active nav item, live readouts,
   * links. Never on a static label, and never encoding data — that keeps it unambiguous next to
   * the series colours, which is also why it is not one of them.
   *
   * Sits 17.6 ΔE from the nearest series hue (weights blue), clear of the 15 floor. Deliberately
   * brighter than the categorical lightness band, which scopes to series colours; the gate that
   * applies to chrome is contrast, and this clears 3:1 on `surface`.
   */
  accent: '#c084fc',
  accentDim: '#6d3f96',

  // --- Series: the memory budget ----------------------------------------------------------
  /**
   * The stacked budget bar, in stack order. Validated as a set on the *all-pairs* list — a
   * stricter gate than the adjacent list a stack strictly needs — worst CVD ΔE 13.2, worst
   * normal-vision ΔE 19.3, all three ≥ 3:1 on `surface`.
   *
   * Colour is never the only channel: segments carry direct labels and a 2px surface gap.
   */
  weights: '#3987e5',
  kvCache: '#c98500',
  overhead: '#d55181',
  /** Headroom — the absence of a thing, so it recedes rather than reading as a fourth category. */
  free: '#1b1f2b',

  // --- Status: reserved, never reused as a series -----------------------------------------
  /**
   * Verdicts. These ship with an icon and a word, always — `kvCache` above is amber and so is
   * `warning`, which is a known and accepted proximity in the reference palette. Hue alone must
   * never be what tells a user their configuration does not fit.
   */
  good: '#0ca30c',
  warning: '#fab219',
  serious: '#ec835a',
  /**
   * Raised from the reference palette's `#d03b3b`, which measures 3.93:1 on this surface. That
   * clears the 3:1 bar for a *graphical* element, which is what the validator checks, but this
   * token is used on 14px text — "Will not run", the overflow warning — where the threshold is
   * 4.5:1. This step is the reference ramp's own red and measures 5.84:1.
   */
  critical: '#e66767',
} as const;

/**
 * Sequential ramp for magnitude — the Envelope's feasibility field and the Matrix heatmap.
 * One hue, light to dark, because a rainbow implies category boundaries that continuous data
 * does not have.
 */
export const sequential = [
  '#cde2fb',
  '#9ec5f4',
  '#6da7ec',
  '#3987e5',
  '#256abf',
  '#184f95',
  '#0d366b',
] as const;

/**
 * The sequential ramp as a *chart* has to read it: brightest is best.
 *
 * `sequential` runs light to dark for a light surface. On this chassis the darkest step is the one
 * that recedes into the panel, so higher values have to be *brighter* — otherwise the best cells
 * are the ones you cannot see. Both grids read this rather than reversing it themselves, which is
 * what the Matrix used to do alone.
 */
export const magnitudeRamp = [...sequential].reverse();

/**
 * Where a value sits on the ramp: one step of `magnitudeRamp`, log-placed inside a stated domain.
 *
 * **Log-placed, because these quantities span orders of magnitude.** Decode runs from ~2 tok/s on
 * a CPU host to ~300 on a B200 and first-token latency from a second to twelve minutes; a linear
 * ramp spends every step but the last on the top of the range and leaves the whole rest of the
 * grid in one indistinguishable band. The comparison people need is "is this twice as fast", not
 * "what fraction of the best is it".
 *
 * **The domain is the caller's, and the floor is the half that was wrong.** With `min: 0` this is
 * the expression the Matrix has always used, which is right for a grid whose worst cell really is
 * near zero — 25 devices from a desktop CPU to a B200. It is wrong wherever the interesting
 * variation sits on top of a large constant: the Envelope's headroom at the default scenario runs
 * 18% to 49% of one machine's ceiling, because 60 GiB of weights is in every cell, and zero-floored
 * that whole range lands on three steps of seven with 76% of the field sharing one. A domain of
 * `{min, max}` over the cells actually on the scale subtracts the constant and spends the ramp on
 * what differs, which is the only thing a field can show. Log placement keeps its other property
 * there too: `log1p` is concave, so equal steps of ramp cover smaller differences at the bottom of
 * the domain — where the wall is, and where the decision is.
 *
 * **A rank-relative domain is a rank-relative claim, and the caller owes the reader that.** A cell on
 * the darkest step is the lowest reading *on its own grid* and not necessarily a poor one: on
 * gpt-oss-20b against an EPYC 9755 every Envelope cell runs, headroom spans 0.726 to 0.991 of the
 * ceiling, and the bottom step lands on a corner with 1.05 TB of 1.45 TB free. So the Envelope keys
 * its ramp "less room / more room" and states the comparison class, where the Matrix — floored at
 * zero across a desktop CPU to a B200 — can honestly say "worse / better". Anchoring is not the
 * escape: with `min: 0` that same grid puts **55 of 56** cells on one step against 50 for the span
 * domain, and the default scenario 35 of 56 against 28. The floor buys nothing here and costs the
 * boundary.
 *
 * A degenerate domain — every cell identical, or a single cell — has nothing for a ramp to say and
 * takes the top step: with no variation each cell is simultaneously the best and the worst on its
 * own grid, and the brightest step is the one that does not imply a deficit that was never measured.
 */
export function magnitudeFill(value: number, domain: { min: number; max: number }): string {
  const steps = magnitudeRamp.length;
  const top = magnitudeRamp[steps - 1];
  // `log1p` rather than `log` so a zero floor — the Matrix's domain — is 0 rather than -Infinity,
  // and a zero value with it. Both are real: a cell that did not fit has no headroom at all.
  const span = Math.log1p(domain.max) - Math.log1p(domain.min);
  if (!(span > 0)) return top;
  const placed = (Math.log1p(value) - Math.log1p(domain.min)) / span;
  // Clamped at both ends: `placed` is 1 at the top of the domain, which would index past the ramp,
  // and a caller may legitimately ask about a value below its own floor.
  return magnitudeRamp[Math.max(0, Math.min(steps - 1, Math.floor(placed * steps)))];
}

/** Parse a `#rrggbb` token into an `[r, g, b]` tuple. */
function hexToRgb(hex: string): [number, number, number] {
  const h = hex.replace('#', '');
  return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
}

/**
 * Build an `rgba(...)` string from a token hex plus alpha.
 *
 * The sanctioned way for a `<canvas>` 2D context or a gradient stop to use a token colour: CSS
 * `var(--color-*)` does not resolve there, so without this the hex gets copied into a component
 * as a literal and the token stops being the single source of truth.
 */
export function withAlpha(hex: string, alpha: number): string {
  const [r, g, b] = hexToRgb(hex);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

/** Spacing scale, in px. Matches the 4px rhythm the mark specs assume. */
export const space = { xs: 4, sm: 8, md: 12, lg: 16, xl: 24, xxl: 32 } as const;

/** Mark geometry the dataviz specs pin down, kept here so charts cannot drift from each other. */
export const marks = {
  /** Rounded data-ends, anchored to the baseline. */
  radius: 4,
  /** Line weight for series. */
  lineWidth: 2,
  /** Surface-coloured gap between adjacent fills, so segments never bleed together. */
  gap: 2,
  /** Minimum hit target, comfortably above the mark itself. */
  hitTarget: 44,
} as const;

export type StatusTone = 'good' | 'warning' | 'serious' | 'critical';
