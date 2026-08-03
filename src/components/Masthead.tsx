import { useEffect, useRef, useState } from 'react';
import { useConfig, type Config } from '@/store/config';
import { configToShareSearch } from '@/store/url';
import { CATALOG_GENERATED_AT } from '@/data/catalog';
import { colors, space, withAlpha } from '@/design/tokens';
import { CopyButton } from './CopyButton';
import { useDevicePixelRatio } from './useDevicePixelRatio';

/**
 * The masthead.
 *
 * The page's front door, and the one surface here that is decoration rather than instrument: a dot
 * lattice and a soft accent bloom behind the wordmark, painted on a canvas and settled into place
 * once on load.
 *
 * It is deliberately shallow. The Bench below is the product's hero surface — pick a model and
 * hardware, watch the budget fill — and a masthead that pushed those controls under the fold would
 * be trading the thing people came for against a first impression. The vertical padding is capped
 * in `vw` for that reason as much as for the reflow sweep.
 *
 * Nothing painted here encodes anything. Every colour is an existing token dimmed through
 * `withAlpha`, so the surface gains no new palette and the accent keeps meaning "live" everywhere
 * it appears on a control.
 */

/** How long the intro takes. One shot, on load, then the canvas is static for the page's life. */
const INTRO_MS = 700;

/** Lattice pitch. `space.xl` so the backdrop shares the 4px rhythm the rest of the page is on. */
const DOT_PITCH = space.xl;

function easeOutCubic(t: number): number {
  return 1 - (1 - t) ** 3;
}

/**
 * Paint the backdrop at intro progress `t` — 0 is unlit, 1 is settled.
 *
 * Pure in its arguments so a resize can jump straight to the settled frame by calling it with 1,
 * which is what makes "animate once, never again" a single line at the call site.
 */
function paint(ctx: CanvasRenderingContext2D, width: number, height: number, t: number): void {
  ctx.clearRect(0, 0, width, height);

  // The light source sits behind the wordmark, which is left-aligned in the container above.
  const cx = width * 0.28;
  const cy = height * 0.45;
  const diagonal = Math.hypot(width, height);

  /*
   * Lattice first, so the bloom washes over the dots instead of being stippled by them.
   *
   * `textFaint` rather than `grid`: the grid token is tuned to sit on `surface`, and on the darker
   * page background it is very nearly invisible. This is chrome, not a gridline, so it borrows the
   * ink colour at an alpha that reads as texture.
   */
  ctx.fillStyle = withAlpha(colors.textFaint, 0.3);
  for (let y = DOT_PITCH / 2; y < height; y += DOT_PITCH) {
    for (let x = DOT_PITCH / 2; x < width; x += DOT_PITCH) {
      /*
       * Each dot lights as the expanding front reaches it, so the lattice radiates outward from
       * the wordmark rather than fading up as one flat sheet. The 1.4 overshoot lets the front
       * clear the far corner before `t` runs out; the 3 sharpens each dot's own fade so the edge
       * of the front stays legible instead of smearing across the whole width.
       */
      const reach = Math.hypot(x - cx, y - cy) / diagonal;
      const lit = Math.min(1, Math.max(0, (t * 1.4 - reach) * 3));
      if (lit <= 0) continue;
      ctx.globalAlpha = lit;
      // A 1px rect, not an arc: at this size a circle is the same pixel with a softer edge, and
      // the crisp one survives the retina downscale better.
      ctx.fillRect(x, y, 1, 1);
    }
  }
  ctx.globalAlpha = 1;

  // The bloom. Widens as it brightens, so the masthead opens out rather than simply lighting up.
  const bloom = ctx.createRadialGradient(cx, cy, 0, cx, cy, diagonal * (0.35 + 0.2 * t));
  bloom.addColorStop(0, withAlpha(colors.accent, 0.26 * t));
  bloom.addColorStop(0.5, withAlpha(colors.accent, 0.08 * t));
  bloom.addColorStop(1, withAlpha(colors.accent, 0));
  ctx.fillStyle = bloom;
  ctx.fillRect(0, 0, width, height);

  // A cooler, weaker second source off to the right, so the field has a direction to it and does
  // not read as one symmetrical blob centred on the title.
  const cool = ctx.createRadialGradient(
    width * 0.78,
    height * 0.15,
    0,
    width * 0.78,
    height * 0.15,
    diagonal * 0.45
  );
  cool.addColorStop(0, withAlpha(colors.weights, 0.1 * t));
  cool.addColorStop(1, withAlpha(colors.weights, 0));
  ctx.fillStyle = cool;
  ctx.fillRect(0, 0, width, height);

  /*
   * Dissolve into the page. Without this the backdrop stops dead at the bottom border and the
   * masthead reads as a pasted-on banner; with it the lattice thins out and the panels below start
   * from the page's own background.
   */
  const fade = ctx.createLinearGradient(0, height * 0.55, 0, height);
  fade.addColorStop(0, withAlpha(colors.bg, 0));
  fade.addColorStop(1, colors.bg);
  ctx.fillStyle = fade;
  ctx.fillRect(0, height * 0.55, width, height * 0.45);
}

export function Masthead() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Bumped by a ResizeObserver so the effect below redraws at the new size — the same arrangement
  // the Envelope uses, and for the same reason: the bitmap is stretched until something else
  // happens to trigger a redraw otherwise.
  const [resizeTick, setResizeTick] = useState(0);
  // A DPR-only change alters no CSS box, so the ResizeObserver above never fires for it — the
  // same staleness the Envelope had, on the page's other canvas (#129).
  const dpr = useDevicePixelRatio();
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || typeof ResizeObserver === 'undefined') return;
    const observer = new ResizeObserver(() => setResizeTick((n) => n + 1));
    observer.observe(canvas);
    return () => observer.disconnect();
  }, []);

  /**
   * Whether the intro has already finished, and when it started.
   *
   * Both are needed, because the paint effect re-runs on every resize and must not replay the
   * flourish. `introduced` covers the settled case: once it is done, a resize repaints at `t = 1`
   * directly.
   *
   * `introStart` covers the case in flight, which is not hypothetical — a ResizeObserver fires an
   * initial notification the moment it observes, so the very first tick lands about a frame into
   * the intro on every single load. Holding the start timestamp across effect runs means a re-run
   * *resumes* at the progress already reached instead of restarting from zero; without it the
   * flourish silently ran twice on load, and dragging a window during those 700ms would pin the
   * backdrop near-blank until the drag ended and then replay the whole thing.
   *
   * Setting `introduced` when the loop is *scheduled* rather than when it completes would be the
   * tempting one-liner and is wrong: that initial resize tick would then find it already true and
   * kill the intro before it drew a frame.
   */
  const introduced = useRef(false);
  const introStart = useRef<number | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    // jsdom has no 2D context, and a real browser can refuse one under memory pressure. This is a
    // backdrop behind text that says everything the masthead has to say, so losing it costs
    // nothing and must not take the heading down with it.
    if (!ctx) return;

    // Draw at device resolution so the lattice stays crisp on a retina display.
    const { width, height } = canvas.getBoundingClientRect();
    // A zero-sized box would give a zero-sized bitmap, which reads back as "never painted" rather
    // than as "not laid out yet". The ResizeObserver brings us straight back when it has a size.
    if (width === 0 || height === 0) return;
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    /**
     * The blanket `prefers-reduced-motion` rule in index.css reaches CSS animations and
     * transitions. It cannot reach a `requestAnimationFrame` loop — that is a JS timer, not a
     * declared animation — so this asks directly, the same way the Matrix does before it scrolls
     * something programmatically.
     */
    const reduce = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches ?? false;

    if (reduce || introduced.current) {
      paint(ctx, width, height, 1);
      introduced.current = true;
      return;
    }

    let frame = 0;
    const step = (now: number) => {
      // Kept in a ref, not a local: a resize mid-intro re-runs this effect, and a local would
      // restart the clock. See the declaration above.
      introStart.current ??= now;
      const t = Math.min(1, (now - introStart.current) / INTRO_MS);
      paint(ctx, width, height, easeOutCubic(t));
      /*
       * One shot. The loop stops on the settled frame instead of running for the life of the
       * page: nothing here animates after load, so a standing rAF would spend a frame budget
       * every 16ms repainting an identical picture — and this is the first one in the codebase,
       * so it is worth it staying that way.
       */
      if (t < 1) frame = requestAnimationFrame(step);
      else introduced.current = true;
    };
    frame = requestAnimationFrame(step);
    return () => cancelAnimationFrame(frame);
  }, [resizeTick, dpr]);

  return (
    <header className="relative isolate border-b border-[var(--color-border)]">
      {/*
        Decoration with no textual equivalent, so it is hidden rather than described — unlike the
        Envelope's canvas, which is `role="img"` and carries a sentence describing the plot.

        `overflow-hidden` sits on this wrapper rather than on the <header> deliberately. The reflow
        sweep in e2e/reflow.spec.ts skips any subtree whose overflow-x is clipped, so clipping at
        the header would quietly exempt the wordmark and the disclaimer from the check that they
        still fit at 320px — the one place this masthead is most likely to break.
      */}
      <div aria-hidden="true" className="absolute inset-0 overflow-hidden">
        <canvas ref={canvasRef} className="h-full w-full" />
      </div>

      <div className="relative mx-auto flex max-w-6xl flex-wrap items-end justify-between gap-x-6 gap-y-4 px-[min(1rem,4vw)] py-[min(4.5rem,14vw)] sm:px-6">
        <div className="min-w-0">
          {/*
            Both bounds are `rem`, so the wordmark tracks the root font size — what the 200%-text
            reflow test asserts. The `10vw` preferred term only governs between them: the 2rem floor
            applies at 320px and below, `vw` carries it from there to 800px, and the 5rem cap holds
            it steady on anything wider. Left in the ink colour rather than the accent: violet is
            reserved for what changes or responds, and a title does neither.
          */}
          {/*
            **The `1.5rem` floor is set by the wordmark's own length, not by taste.** At 320px and a
            200% root the `10vw` term is 32px, so the floor governs, and an unbreakable eight-letter
            word cannot be wrapped or tracked out of an overflow. `headroom` needs 348.8px of ink in
            DejaVu Sans Bold — the default sans on the CI runner and on most Linux — against the
            294.4px the column offers, and the document scrolled sideways at 362px against a 320px
            viewport. `bench` fit because five letters left 86px of slack.

            At the default text size this changes nothing at any viewport a browser actually
            presents: the `10vw` term is 32px at 320px wide, above the new 24px floor, so `clamp`
            returns exactly what it did before. The new floor would only govern at 100% below 240px,
            where the old one governed below 320px — and 320px is WCAG reflow's own floor, so no
            reader is in that band. Under text zoom the floor does govern, which is the room needed.
          */}
          <h1 className="hero-wordmark text-[clamp(1.5rem,10vw,5rem)] leading-none font-semibold tracking-[-0.03em]">
            headroom
          </h1>
          <p className="mt-3 text-sm text-[var(--color-text-muted)]">
            What runs on your hardware, and how comfortably.
          </p>
        </div>

        <div className="flex min-w-0 flex-col items-start gap-2 sm:items-end">
          <ShareLink />
          <p className="max-w-md text-xs leading-relaxed text-[var(--color-text-faint)] sm:text-right">
            Estimates from a roofline model calibrated against published measurements. Treat them as
            a band, not a promise. Model catalog generated{' '}
            <time dateTime={CATALOG_GENERATED_AT}>
              {new Date(CATALOG_GENERATED_AT).toISOString().slice(0, 10)}
            </time>
            .
          </p>
        </div>
      </div>
    </header>
  );
}

/**
 * Copies a link that names the scenario in full.
 *
 * Not `location.href`: the address bar is deliberately bare on an untouched default page, because
 * it claims nothing there. A copied link always claims something — it says "this is what I was
 * looking at" — so every field is written out and the link cannot drift when a default moves.
 * `configToShareSearch` is the same encoder the address bar uses, minus the empty case, so there
 * is still only one place that knows the format.
 */
function ShareLink() {
  const config = useConfig();

  /**
   * Derived, not captured.
   *
   * Holding the link in state froze it at the click that revealed the field: adjusting any
   * control afterwards left the still-visible input offering the previous scenario, so a manual
   * copy shared something the user was no longer looking at. That is the same class of bug the
   * full encoding exists to prevent — a link that means something other than it appears to.
   */
  const href = `${window.location.origin}${window.location.pathname}${configToShareSearch(
    config as Config
  )}`;

  /**
   * The confirmation, the overlapping-click guard, the per-attempt timer and the
   * no-clipboard-API fallback all live in {@link CopyButton} now — extracted when the launch panel
   * (#136) needed a second copy button, because each of those four is a bug that was found once
   * and a hand-written second copy is how a codebase comes to have one of them fixed and one not.
   * Its docblock is where the four are written up; nothing about this call site is special.
   */
  return (
    <CopyButton
      value={href}
      idleLabel="Copy link to this scenario"
      copiedLabel="Link copied"
      fallbackLabel="Copy it from here"
      fieldLabel="Link to this scenario"
    />
  );
}
