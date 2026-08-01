/**
 * Catalog-shaped assertions that happen to need a render (#115): what the pickers say about the
 * rows a reader is choosing between, where a caveat appears relative to the choice, and that
 * both surfaces show the order the files state. Split out of `App.test.tsx`, which keeps the
 * panels agreeing with each other.
 */
import { cleanup, render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';
import App from './App';
import { useConfig, DEFAULT_CONFIG, estimateConfig } from '@/store/config';
import { DEVICES, getDevice } from '@/data/catalog';
import { DEVICE_CLASS_LABELS, SETTING_LABELS, SETTING_NOTES, deviceCountNote } from '@/lib/stops';
// The one component this file mounts on its own, and only to sweep a renderer over all 43 catalog
// rows — see "leaves none of the markup in any note the catalog carries".
import { Select } from '@/components/Controls';
import { RUNTIMES, getRuntime, runtimeDrives } from '@/data/runtimes';
import { canShard, maxAllocatablePerDevice } from '@/engine/placement';
import { atFullGrid, boundGridByDefault } from '@/test/grid';

/**
 * The Matrix's extent is bounded by default and the real grid opted into — the whole design,
 * and the fixture itself, live in `src/test/grid.ts` (#101, #115). The mock is declared here
 * because `vi.mock` is hoisted per test file and cannot ride an import; the fixture's own
 * preconditions are held in `src/test/grid.test.ts`.
 */
vi.mock('@/data/catalog', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/data/catalog')>();
  return { ...actual, comparisonGrid: vi.fn(actual.comparisonGrid) };
});

boundGridByDefault();

afterEach(() => {
  cleanup();
  useConfig.setState(DEFAULT_CONFIG);
});

/**
 * What a control says about itself (#80).
 *
 * The five Usage controls drive every figure on the page, and the panel's entire text content at the
 * default scenario was the labels and the values: "Context per sequence 32K Concurrent users 1
 * Prompt length 8K KV precision FP16 Q8 Q4 Device count 1x". The argument those five make together —
 * context times users times bits per token is most of what the budget bar draws — was written in
 * `Envelope.tsx`'s docstring and nowhere a reader can see, and there was no mechanism to fix it per
 * call site: `StopSlider` and `Segmented` took no note at all, and `Select`'s `hint` was a dead
 * escape hatch no call site passed.
 *
 * **This is DOM, not layout, so it is here.** Whether a sentence is *reachable* — resolved through
 * `aria-describedby` rather than merely sitting nearby — is an attribute question jsdom answers
 * exactly. Whether five extra lines of prose change the panel's geometry is a browser question, and
 * `e2e/reflow.spec.ts` already sweeps this panel at 320px and at 200% text for a page that scrolls
 * sideways; the notes are wrapping paragraphs and add no min-content floor, so they need no new spec.
 */
describe('the controls that drive every figure explain what they are', () => {
  /**
   * The description a screen reader resolves for a control: the `aria-describedby` ids in order,
   * each one's text, joined.
   *
   * Written out rather than reaching for `toHaveAccessibleDescription`, because the sweep has to
   * *list* the controls that have none. A per-element matcher can only report the first failure, and
   * the point of a sweep is naming the instances nobody thought of.
   */
  const description = (el: Element) =>
    (el.getAttribute('aria-describedby') ?? '')
      .split(/\s+/)
      .filter(Boolean)
      .map((id) => el.ownerDocument.getElementById(id)?.textContent?.trim() ?? '')
      .join(' ')
      .trim();

  const usage = () => screen.getByRole('region', { name: 'Usage' });

  /**
   * Every control in a panel: the sliders, the selects, and the `fieldset` a set of radios lives in
   * — deliberately not the radios themselves, whose description hangs off the group.
   */
  const controlsOf = (panel: HTMLElement) => [
    ...panel.querySelectorAll<HTMLElement>('input[type="range"], select, fieldset'),
  ];

  /** `Prompt length`, enough to find the offender from the failure message. */
  const named = (el: Element) =>
    (el as HTMLInputElement).labels?.[0]?.textContent?.trim() ??
    el.querySelector('legend')?.textContent?.trim() ??
    `<${el.tagName.toLowerCase()}>`;

  /** The element a setting's description hangs off — the group for the radios, the input otherwise. */
  const controlFor = (key: keyof typeof SETTING_NOTES) =>
    key === 'kvPrecision'
      ? within(usage()).getByRole('group', { name: SETTING_LABELS[key] })
      : within(usage()).getByLabelText(SETTING_LABELS[key]);

  it('gives every Usage control a description, not just a label and a value', () => {
    render(<App />);
    const controls = controlsOf(usage());

    // Vacuity guards, and lower bounds rather than exact counts: what has to hold is that the sweep
    // below ran over something — a selector that stopped matching, or every control moving out of the
    // panel, cannot report a clean sweep over nothing. An *exact* four went red on a sixth control
    // that was wired perfectly, before the property under test was evaluated at all, which is a
    // failure about the count and not about the thing this test is named after. A control leaving the
    // panel is caught by the per-setting test below, which looks each one up inside the region.
    expect(
      controls.filter((c) => c.tagName === 'INPUT').length,
      'the sliders the sweep ran over'
    ).toBeGreaterThanOrEqual(4);
    expect(
      controls.filter((c) => c.tagName === 'FIELDSET').length,
      'the KV group'
    ).toBeGreaterThanOrEqual(1);

    const silent = controls.filter((c) => description(c) === '').map(named);
    expect(silent, 'Usage controls with no accessible description').toEqual([]);
  });

  /**
   * And that each one is wired to its *own* sentence. Five near-identical call sites is where a
   * copy-paste puts the context's sentence under the prompt slider, which reads as plausibly as the
   * right answer and is worse than no note at all.
   */
  it('wires each control to its own sentence', () => {
    render(<App />);

    // Four of the five the issue names. If a setting loses its note the sweep above catches the
    // control; this catches a note that is present and attached to the wrong thing.
    expect(Object.keys(SETTING_NOTES)).toHaveLength(4);

    for (const key of Object.keys(SETTING_NOTES) as (keyof typeof SETTING_NOTES)[]) {
      expect(description(controlFor(key)), `${SETTING_LABELS[key]}’s description`).toBe(
        SETTING_NOTES[key]
      );
    }

    // The fifth is not in the table, because what an extra device buys depends on the runtime. Same
    // assertion, resolved through the same runtime the store is on.
    expect(
      description(within(usage()).getByLabelText(SETTING_LABELS.deviceCount)),
      'Device count’s description'
    ).toBe(
      deviceCountNote(
        getRuntime(DEFAULT_CONFIG.runtimeId),
        // Derived rather than `true`, so this tracks a change of default rather than asserting
        // against the wrong branch if the default pairing ever becomes an undrivable one.
        runtimeDrives(getRuntime(DEFAULT_CONFIG.runtimeId), getDevice(DEFAULT_CONFIG.deviceId))
      )
    );
  });

  it('describes the KV group once rather than once per radio', () => {
    render(<App />);
    const group = within(usage()).getByRole('group', { name: SETTING_LABELS.kvPrecision });
    expect(description(group)).toBe(SETTING_NOTES.kvPrecision);

    // A description on each radio is re-announced on every arrow key — three sentences to move
    // between three options — which is how a description earns a reputation for being noise.
    const radios = within(group).getAllByRole('radio');
    expect(radios.length, 'the group rendered no options').toBeGreaterThan(1);
    expect(
      radios.filter((r) => r.getAttribute('aria-describedby') !== null).length,
      'radios carrying their own copy of the group description'
    ).toBe(0);
  });

  /**
   * The device-count branch, both ways round.
   *
   * The default machine shards, so a test written only against the default would pass just as
   * happily if the sentence had been added to the `!shardable` paragraph instead — which is exactly
   * where the panel's one pre-existing explanation already lived, visible only when there is nothing
   * to configure. So the assertion is that the note is reachable *through the slider*, and that in
   * the branch with no slider it is not on the page at all.
   */
  it('describes the device-count slider in the branch that has one', async () => {
    const user = userEvent.setup();
    render(<App />);

    const note = deviceCountNote(
      getRuntime(DEFAULT_CONFIG.runtimeId),
      runtimeDrives(getRuntime(DEFAULT_CONFIG.runtimeId), getDevice('dgx-spark'))
    );
    await user.selectOptions(screen.getByLabelText('Hardware'), 'dgx-spark');
    const slider = screen.getByLabelText(SETTING_LABELS.deviceCount);
    expect(description(slider)).toBe(note);

    // A Mac has no transport between chassis: no control, so nothing to describe. The panel says
    // why the control is absent instead, which is a different sentence for a different reason.
    await user.selectOptions(screen.getByLabelText('Hardware'), 'mac-studio-m3-ultra-256');
    expect(screen.queryByLabelText(SETTING_LABELS.deviceCount)).not.toBeInTheDocument();
    expect(screen.getByText(/needs a transport between them/i)).toBeInTheDocument();
    expect(screen.queryByText(note)).not.toBeInTheDocument();
  });

  /**
   * The sentence under the slider and the arithmetic the slider drives are one claim (found in
   * review).
   *
   * The issue's suggested copy — "shard the model across, tensor-parallel. Adds memory and
   * bandwidth, minus what the interconnect costs" — is true of vLLM and of nothing else the app
   * offers, and the default runtime is one of the others. `achievedBandwidth` and the FLOPS closure
   * in `speed.ts` both return the per-device figure and short-circuit before `effectiveDeviceCount`
   * whenever `parallelism === 'layer'`, which is exactly the derivation `docs/ROADMAP.md` records as
   * wrong-first and silent when it breaks: a layer split buys capacity, not speed.
   *
   * So this asserts the copy against a measurement rather than against itself. The model has to
   * *fit on one device* for the comparison to isolate bandwidth: with a spill in play a layer split
   * really does get faster with more cards — 14.25 tok/s to 190.11 for this model at Q4_K_M on one
   * 4090 versus four — because the extra card stops it spilling. That is the capacity arriving as
   * speed, and it is why the sentence says "buys capacity" rather than "makes no difference".
   */
  it('does not promise the device-count slider a speed-up the runtime cannot deliver', () => {
    const device = getDevice('dgx-spark');
    expect(canShard(device), 'the slider does not render at all without a link').toBe(true);

    const measured = RUNTIMES.filter((r) => runtimeDrives(r, device)).map((runtime) => {
      const at = (deviceCount: number) =>
        estimateConfig({
          ...DEFAULT_CONFIG,
          deviceId: device.id,
          runtimeId: runtime.id,
          deviceCount,
        });
      const one = at(1);
      expect(
        one.placement.fits,
        `${runtime.id} spills on one ${device.name}, so a speed change would be capacity, not bandwidth`
      ).toBe(true);
      return {
        id: runtime.id,
        // `true` is not an assumption: the map above filters to `runtimeDrives`, and the
        // unsupported branch has its own test below.
        note: deviceCountNote(runtime, true),
        // 1% rather than exact equality: what is being distinguished is "held still to the last
        // decimal" from "half again as fast", and neither side needs a tighter threshold than that.
        aggregates: at(4).decode.perUserTokensPerSec > one.decode.perUserTokensPerSec * 1.01,
      };
    });

    // Both sides of the distinction are present, or the assertion below measures one branch twice.
    expect(measured.filter((m) => m.aggregates).map((m) => m.id)).toEqual(['vllm']);
    expect(measured.filter((m) => !m.aggregates).map((m) => m.id)).toEqual(['llama.cpp']);

    const lying = measured.filter(
      (m) => /bandwidth as well as memory/.test(m.note) !== m.aggregates
    );
    expect(
      lying.map((m) => `${m.id}: ${m.note}`),
      'runtimes whose device-count sentence disagrees with their own throughput'
    ).toEqual([]);
  });

  /**
   * And that the sentence follows the runtime on screen, which is the whole reason it is a function.
   */
  it('rewrites the device-count sentence when the runtime changes what a device buys', async () => {
    const user = userEvent.setup();
    render(<App />);

    const slider = () => screen.getByLabelText(SETTING_LABELS.deviceCount);
    expect(description(slider())).toMatch(/buys capacity, not speed/);

    await user.selectOptions(screen.getByLabelText(SETTING_LABELS.runtimeId), 'vllm');
    expect(description(slider())).toMatch(/bandwidth as well as memory/);
  });

  /**
   * Both notes described something no evaluation reaches, in configurations two clicks from the
   * default. Codex found them on #80; each is a sentence that reads the wrong one of two inputs.
   *
   * **The device-count note read the hardware and not the runtime.** `canShard` is
   * `interconnect !== undefined`, so on a DGX Spark the slider renders under MLX — which cannot
   * drive that machine at all — and the note promised a layer split buying capacity directly below
   * the Runtime control's "Does not run on" warning.
   *
   * **The runtime note claimed every weight is dequantized.** BF16 is a real format here, and MLX
   * coerces to it, so there is nothing to dequantize in a configuration a reader reaches by picking
   * the one runtime this catalog exists to cover for Apple hardware.
   */
  it('does not describe a split for a runtime that cannot drive the machine', () => {
    const device = getDevice('dgx-spark');
    const mlx = getRuntime('mlx');
    expect(canShard(device), 'the slider renders, which is the whole problem').toBe(true);
    expect(runtimeDrives(mlx, device), 'this test needs an undrivable pairing').toBe(false);

    const note = deviceCountNote(mlx, runtimeDrives(mlx, device));
    expect(note).not.toMatch(/buys capacity, not speed/);
    expect(note).not.toMatch(/bandwidth as well as memory/);
    expect(note, 'the control still stores a value, so it has to say why nothing moves').toMatch(
      /does not run on this machine/
    );
  });

  it('says on screen that a device count buys nothing under an undrivable runtime', async () => {
    const user = userEvent.setup();
    render(<App />);

    const slider = () => screen.getByLabelText(SETTING_LABELS.deviceCount);
    expect(description(slider())).toMatch(/buys capacity, not speed/);

    // The default device is the DGX Spark, which MLX does not drive.
    await user.selectOptions(screen.getByLabelText(SETTING_LABELS.runtimeId), 'mlx');
    expect(description(slider())).not.toMatch(/buys capacity|bandwidth as well as memory/);
    expect(description(slider())).toMatch(/does not run on this machine/);
  });

  it('does not tell a BF16 reader that every weight is dequantized', async () => {
    const user = userEvent.setup();
    render(<App />);
    const runtimeNote = () => description(screen.getByLabelText(SETTING_LABELS.runtimeId)) ?? '';

    /**
     * Driven through the DOM rather than read off the `<option>`s: `Controls.tsx` renders **only the
     * selected** option's note, so a sweep over `option.textContent` finds nothing and passes
     * whatever the copy says. That was the first version of this test.
     */
    const dequantizing = RUNTIMES.filter((r) => !r.nativeLowPrecision);
    expect(dequantizing.length, 'nothing dequantizes, so this test has no subject').toBeGreaterThan(
      0
    );

    // llama.cpp on the default machine, and MLX on hardware it actually drives — otherwise MLX's
    // note is the "Does not run on" warning and the sentence under test never renders.
    expect(runtimeNote()).toMatch(/quantized checkpoint/);
    expect(runtimeNote(), 'false for BF16, which is a format this app offers').not.toMatch(
      /every weight/
    );

    await user.selectOptions(screen.getByLabelText('Hardware'), 'mac-studio-m3-ultra-96');
    await user.selectOptions(screen.getByLabelText(SETTING_LABELS.runtimeId), 'mlx');
    expect(runtimeNote(), 'MLX coerces to BF16, so this is the easiest place to check it').toMatch(
      /quantized checkpoint/
    );
    expect(runtimeNote()).not.toMatch(/every weight/);
  });

  /**
   * The instance one panel up (found in review).
   *
   * `Select` renders only the selected option's note, and `runtimeOptions` produced one for exactly
   * two states — hardware the runtime cannot drive, and a runtime that preallocates. Both are false
   * for llama.cpp on any machine it drives, so at the default scenario the Runtime picker emitted no
   * `aria-describedby` at all: byte-for-byte the `aria-describedby: null` #80 tabulated for the
   * Usage sliders, in the panel #80 held up as the counterexample. Switching to vLLM produced a
   * description and switching back removed it, which is the appear-and-vanish behaviour that got
   * `Select`'s `hint` prop deleted rather than wired.
   *
   * Every option, not just the default one, because "the description exists at the scenario the test
   * happens to render" is the shape of the bug.
   */
  it('describes the Runtime picker at every choice, not only when a caveat applies', async () => {
    const user = userEvent.setup();
    render(<App />);

    const select = screen.getByLabelText(SETTING_LABELS.runtimeId) as HTMLSelectElement;
    const options = within(select).getAllByRole('option') as HTMLOptionElement[];
    // Vacuity guard: the picker offers every runtime, so a loop over nothing is a green test.
    expect(options.length, 'runtimes the picker offered').toBe(RUNTIMES.length);

    const silent: string[] = [];
    for (const option of options) {
      await user.selectOptions(select, option.value);
      if (description(select) === '') silent.push(option.value);
    }
    expect(silent, 'runtimes that leave the picker with no accessible description').toEqual([]);
  });

  /**
   * The instance the issue did not name, and the reason the sweep is worth running: the Matrix's
   * measure switch already had its sentence on screen and never attached it to anything, so a
   * screen-reader user entering the group heard "Colour the grid by, Does it fit, pressed" and
   * nothing about what the colour means.
   */
  it('describes the grid’s measure switch, which had the sentence but never attached it', async () => {
    const user = userEvent.setup();
    render(<App />);

    const group = screen.getByRole('group', { name: /colour the grid by/i });
    expect(description(group)).toMatch(/headroom left/i);

    /**
     * It tracks the selection, which is what makes it this group's description rather than a static
     * caption: each measure means something different by a bright cell.
     *
     * Scoped to the group since #65 gave the Envelope a measure control of its own, reading the same
     * `MEASURES`. A page-wide `getByRole('button', { name: 'How fast' })` then finds two and throws —
     * and the failure is a real one about the query rather than about either control, because "the
     * grid" in this test's name is the Matrix and the Envelope's switch answers for a different
     * picture. Both surfaces having the toggle is the point of sharing the vocabulary.
     */
    await user.click(within(group).getByRole('button', { name: 'How fast' }));
    expect(description(group)).toMatch(/tokens per second/i);
  });

  /**
   * The Hardware picker, whose note was doing two jobs and neither of them well (#68).
   *
   * `[statusWarning, ceilingClause, row.note].join(' ')` fused a derived claim onto 40-180 words of
   * catalog provenance with a bare space, and handed the whole thing to the control as its
   * `aria-describedby`. Two consequences, and the DOM is where both are visible:
   *
   *   - **The punctuation.** "raiseable to 240 GiB The allocation ceiling reserves 16 GiB for
   *     macOS" reads as a parse error on the most prominent control on the page, and on the M5
   *     Ultra the sentence that ran on was the warning that its specs are rumour-grade.
   *   - **The audience.** A screen-reader user heard the entire derivation — `iogpu.wired_limit_mb`,
   *     unwired allocations, what the sysctl parses — every time focus landed on the picker, before
   *     they could choose anything.
   *
   * `src/data/catalog.test.ts` sweeps the composition across all 43 rows. These assert the wiring:
   * that the short claim is what the control is described by, that the provenance is still reachable,
   * and that it is reachable somewhere other than the description.
   */
  describe('the Hardware picker', () => {
    const hardware = () => screen.getByLabelText(SETTING_LABELS.deviceId);
    const toggle = () => screen.getByRole('button', { name: /the full hardware note/i });

    /** The disclosure's region, found through the button that controls it. */
    const detail = () => {
      const id = toggle().getAttribute('aria-controls');
      return id === null ? null : document.getElementById(id);
    };

    /**
     * Every row where a derived clause used to be followed immediately by a curated note — the nine
     * seams, derived from the catalog rather than listed, so a row added later joins the sweep.
     * The issue named seven of them and one of those (`ryzen-ai-max-395`, already at its own
     * ceiling) composes a single fragment and never had a seam at all.
     */
    const seams = DEVICES.filter(
      (d) =>
        d.note !== undefined &&
        (d.status !== 'shipping' ||
          (d.allocatableTunable === true && maxAllocatablePerDevice(d) > d.allocatableBytes))
    );

    it('describes the rumoured Mac with closed sentences instead of one fused figure', async () => {
      const user = userEvent.setup();
      render(<App />);
      await user.selectOptions(hardware(), 'mac-studio-m5-ultra-512');

      // The only three-fragment row, and the one the issue calls out: the rumour warning was fused
      // to a capacity figure, which is the sentence on that row that most needs to stand alone.
      const note = description(hardware());
      expect(note).toMatch(/^Rumoured — specs may change\. /);
      expect(note).toMatch(/384 GiB allocatable by default, raiseable to 480 GiB\.$/);

      // Fourteen words, against 146 before — and none of the derivation.
      expect(note.split(/\s+/)).toHaveLength(14);
      expect(note).not.toMatch(/iogpu|window server|sysctl|per-core rate|rumour-grade/i);
    });

    it.each(seams.map((d) => [d.id, d] as const))(
      'gives %s a description that is a claim, not a derivation',
      async (_id, device) => {
        const user = userEvent.setup();
        render(<App />);
        await user.selectOptions(hardware(), device.id);

        const note = description(hardware());
        // Something is still said about every one of these rows — this is not a deletion.
        expect(note).not.toBe('');
        // The claim ends as a sentence, so nothing that follows it can look like part of it.
        expect(note).toMatch(/[.!?…]$/);
        // And the curated prose is not in it. First 40 characters rather than the whole string,
        // because a substring is what a bare join produces.
        expect(note).not.toContain(device.note!.slice(0, 40));
      }
    );

    it('keeps the curated note reachable, and out of the description while open', async () => {
      const user = userEvent.setup();
      render(<App />);
      await user.selectOptions(hardware(), 'mac-studio-m3-ultra-512');

      // Collapsed: the provenance is hidden, not unmounted — `hidden` is display: none, so it
      // still sets no height on a grid cell whose row also holds the Quantization and Runtime
      // pickers, while the toggle's `aria-controls` keeps resolving to a real node (#131).
      expect(screen.getByText(/what the sysctl parses/i)).not.toBeVisible();

      await user.click(toggle());
      expect(screen.getByText(/what the sysctl parses/i)).toBeVisible();
      // Open, and still not part of the control's accessible description. A disclosure that got
      // wired into `aria-describedby` when expanded would be the same defect with a click in front
      // of it.
      expect(description(hardware())).not.toMatch(/sysctl/i);
    });

    it('renders the catalog’s prose rather than printing its markup', async () => {
      const user = userEvent.setup();
      render(<App />);
      // Five rows write `**strong**`, two write `*emphasis*` and nine write backticked identifiers;
      // nothing rendered any of them, so the picker printed literal asterisks. Moving the prose to
      // its own region without this would have moved the glitch with it.
      await user.selectOptions(hardware(), 'mac-studio-m3-ultra-96');
      await user.click(toggle());

      const region = detail();
      expect(region).not.toBeNull();
      expect(region!.querySelector('strong')?.textContent).toMatch(/60-core GPU/);
      expect(region!.querySelector('code')?.textContent).toBe('iogpu.wired_limit_mb');
      // Verbatim apart from the marks: the note is provenance, and losing a clause of it in a
      // renderer would be worse than printing the asterisks.
      expect(region!.textContent).toBe(
        getDevice('mac-studio-m3-ultra-96').note!.replace(/\*\*|\*|`/g, '')
      );

      // The single-asterisk register, which the first version of this renderer did not read: two
      // rows write their contrast with one mark rather than two, and both printed the asterisks in
      // the region this change created for them.
      await user.selectOptions(hardware(), 'rx-9070-xt');
      expect(detail()!.querySelector('em')?.textContent).toBe('matrix');
    });

    /**
     * And over the whole catalog, because "five rows write `**strong**`" is a fact about the file on
     * the day it was read.
     *
     * The property is that nothing of the markup reaches the reader as text: the region's text is
     * the note with its marks removed, exactly, which fails both ways — an unrendered mark shows up
     * as a stray asterisk, and a renderer that ate a clause shows up as missing prose. A fourth
     * mark, or a stray `*` in a figure, fails here rather than printing itself at a reader.
     *
     * `Select` on its own rather than the whole Bench, which is the one place in this file that
     * mounts a component instead of the app: the wiring from the catalog through `devicePickerNote`
     * to this control is what the tests above assert, on the real picker. What is swept here is the
     * renderer against every note in the file, and mounting the Matrix's 408 cells 29 times to read
     * one paragraph cost 19 seconds of a suite that runs in two minutes. One mount and one click
     * either way, since the disclosure deliberately stays open across a change of selection.
     */
    it('leaves none of the markup in any note the catalog carries', async () => {
      const user = userEvent.setup();
      const noted = DEVICES.filter((d) => d.note !== undefined);
      expect(noted.length, 'no row carries a note, so this sweep proves nothing').toBeGreaterThan(
        20
      );

      const picker = (value: string) => (
        <Select
          label={SETTING_LABELS.deviceId}
          value={value}
          onChange={() => {}}
          options={noted.map((d) => ({ value: d.id, label: d.name, detail: d.note }))}
        />
      );

      const { rerender } = render(picker(noted[0].id));
      await user.click(toggle());

      const wrong: string[] = [];
      for (const device of noted) {
        rerender(picker(device.id));
        if (detail()?.textContent !== device.note!.replace(/\*\*|\*|`/g, '')) wrong.push(device.id);
      }
      expect(wrong, 'notes whose markup reached the reader as text').toEqual([]);
    });

    /**
     * The other 34 rows, where the split leaves the control with no accessible description at all.
     *
     * That is what #68 asks for — the derivation was never a description of the control — but a
     * description that is deliberately absent and one that vanished by accident are the same DOM,
     * and nothing else in the suite looks at this panel's descriptions. So the sanctioned state is
     * pinned: no description, and the prose one click away in the disclosure. This row is the one
     * whose note was dropped from the picker entirely once before.
     */
    it('describes the control only where it has derived a claim', async () => {
      const user = userEvent.setup();
      render(<App />);
      await user.selectOptions(hardware(), 'rtx-3090');

      const device = getDevice('rtx-3090');
      expect(device.status).toBe('shipping');
      expect(device.allocatableTunable).toBeUndefined();
      expect(device.note).toBeDefined();

      expect(hardware()).not.toHaveAttribute('aria-describedby');
      expect(description(hardware())).toBe('');

      await user.click(toggle());
      expect(detail()!.textContent).toMatch(/NVLink/);
    });

    it('offers no disclosure for a row the catalog says nothing extra about', async () => {
      const user = userEvent.setup();
      render(<App />);

      // The 5090 carries no curated note, so there is nothing to disclose — and an empty
      // disclosure is a control that promises something and does nothing.
      await user.selectOptions(hardware(), 'rtx-5090');
      expect(getDevice('rtx-5090').note).toBeUndefined();
      expect(
        screen.queryByRole('button', { name: /the full hardware note/i })
      ).not.toBeInTheDocument();
    });
  });
});

/**
 * What a picker says *before* the choice, which is not the same string as what it says after (#69).
 *
 * `Controls.tsx` renders every option's label and only the **selected** option's note. So the
 * caveats that decide whether a row is worth choosing lived in the one string a `<select>` will not
 * show until the choice has been made: "Mac Studio M5 Ultra (512 GB) — 512 GiB" scrolled past as an
 * equal of the 512 GB M3 Ultra one line above it, which is real hardware with measured bandwidth,
 * and its `Rumoured — specs may change.` appeared only afterwards. CLAUDE.md states that one as a
 * requirement: pre-release specs must stay visibly labelled in the UI.
 *
 * **Swept over both pickers that share the component**, because the mechanism is the component's and
 * not the catalog's. The Runtime picker had the same shape and a harder consequence — on a Mac Studio
 * it offered llama.cpp, vLLM and MLX as three equals and produced "Does not run on …" only once vLLM
 * had been selected and every figure on the page had been replaced by a refusal.
 *
 * These read `option.textContent`, which is the one place a sweep like this is not vacuous: the file
 * already records that reading notes off the `<option>`s finds nothing and passes whatever the copy
 * says. Here the text under test really is the option's own.
 */
describe('a picker states its caveats where the choice is made', () => {
  /** Every option's own text, which is all a closed `<select>` has to distinguish its rows by. */
  const optionsOf = (label: string) =>
    Array.from((screen.getByLabelText(label) as HTMLSelectElement).options).map((o) => ({
      value: o.value,
      text: (o.textContent ?? '').trim(),
    }));

  /** The marker as a reader sees it, not as the code spells it — `devices.json` says `rumored`. */
  const PRE_RELEASE = /\s·\s(rumoured|announced)$/;

  it('marks every row whose specs are not final, in the option text', () => {
    render(<App />);

    const options = optionsOf(SETTING_LABELS.deviceId);
    expect(options.length, 'the picker offered no hardware at all').toBe(DEVICES.length);

    // From `status`, so a row added to the catalog as announced or rumoured fails this rather than
    // slipping through it. The named instance is one device; the class is the field.
    const preRelease = new Set(DEVICES.filter((d) => d.status !== 'shipping').map((d) => d.id));
    expect(
      preRelease.size,
      'no catalogued row is rumoured or announced, so this sweep proves nothing'
    ).toBeGreaterThan(0);

    expect(
      options
        .filter((o) => preRelease.has(o.value) && !PRE_RELEASE.test(o.text))
        .map((o) => o.text),
      'pre-release hardware offered as though it were shipping'
    ).toEqual([]);
    // The other half of it: a marker on every row would satisfy the assertion above and mean nothing.
    expect(
      options
        .filter((o) => !preRelease.has(o.value) && PRE_RELEASE.test(o.text))
        .map((o) => o.text),
      'shipping hardware carrying a pre-release marker'
    ).toEqual([]);
  });

  it('marks the rumoured Mac without spending the figures the row is chosen on', () => {
    render(<App />);

    // The string the issue quotes, plus the marker it was missing. Pinned whole because the defect
    // was not "the marker is absent" but "the label is indistinguishable from a shipping row": a
    // marker that displaced the name or the capacity would satisfy a regex and lose the comparison.
    expect(
      optionsOf(SETTING_LABELS.deviceId).find((o) => o.value === 'mac-studio-m5-ultra-512')?.text
    ).toBe('Mac Studio M5 Ultra (512 GB) — 512 GiB · rumoured');
  });

  it('still gives the chosen row the fuller sentence, rather than moving it into the label', async () => {
    const user = userEvent.setup();
    render(<App />);

    // The marker is a tag on a row being scanned; this is the clause for the row that was picked, and
    // it is the control's accessible description. Adding the first must not cost the second.
    await user.selectOptions(
      screen.getByLabelText(SETTING_LABELS.deviceId),
      'mac-studio-m5-ultra-512'
    );
    expect(screen.getByLabelText(SETTING_LABELS.deviceId)).toHaveAccessibleDescription(
      /^Rumoured — specs may change\./
    );
  });

  it('marks a runtime that cannot drive the machine currently selected', async () => {
    const user = userEvent.setup();
    render(<App />);

    /**
     * Two machines, because the marked runtime differs between them: vLLM does not run on Apple
     * unified memory and MLX runs on nothing else. A marker hard-coded to either would pass on one
     * device and fail on the other, which is why the expectation is derived from `runtimeDrives`.
     */
    for (const deviceId of ['mac-studio-m3-ultra-256', 'rtx-5090']) {
      await user.selectOptions(screen.getByLabelText(SETTING_LABELS.deviceId), deviceId);
      const device = getDevice(deviceId);

      const options = optionsOf(SETTING_LABELS.runtimeId);
      expect(options.length, 'the picker offered no runtimes').toBe(RUNTIMES.length);
      expect(
        RUNTIMES.filter((r) => !runtimeDrives(r, device)).length,
        `every runtime drives ${deviceId}, so it marks nothing`
      ).toBeGreaterThan(0);

      const wrong = options.filter(
        (o) =>
          runtimeDrives(getRuntime(o.value), device) ===
          /does not run on this hardware/.test(o.text)
      );
      expect(
        wrong.map((o) => `${deviceId}: “${o.text}”`),
        'runtime options whose marker disagrees with whether the runtime drives this machine'
      ).toEqual([]);
    }

    // And the note still names the machine, which the marker deliberately does not — it is what a
    // screen-reader user hears on the control once the choice has been made.
    await user.selectOptions(screen.getByLabelText(SETTING_LABELS.runtimeId), 'mlx');
    expect(screen.getByLabelText(SETTING_LABELS.runtimeId)).toHaveAccessibleDescription(
      /Does not run on GeForce RTX 5090/
    );
  });

  it('says what a runtime will not run on, rather than leaving the reader to supply it', async () => {
    const user = userEvent.setup();
    render(<App />);

    /**
     * Pinned whole, because the first version of this marker said "does not run here" and an option's
     * own text is *all* that is announced for a row nobody has selected — so "here" resolved to
     * nothing, and the string that names the machine is the selected option's note, which is exactly
     * the dependency the marker exists to remove (found in review). The referent has to be inside the
     * option: "this hardware" is the control one row up.
     */
    await user.selectOptions(
      screen.getByLabelText(SETTING_LABELS.deviceId),
      'mac-studio-m3-ultra-256'
    );
    expect(optionsOf(SETTING_LABELS.runtimeId).find((o) => o.value === 'vllm')?.text).toBe(
      'vLLM · does not run on this hardware'
    );
  });
});

/**
 * The list order, on the two surfaces that render it (#79).
 *
 * `devices.json`'s row order *is* the display order — `catalog.ts` maps the file straight through and
 * neither surface sorts — and until this landed nothing anywhere said so, enforced it, or showed it.
 * The picker was a flat list of 43 options, so scrolling from `rtx-3090` to `rtx-pro-6000-blackwell`
 * to `h100-sxm` crossed two segment boundaries in silence; the Matrix ran the three classes together
 * as one 42-column strip and its caption explained the arrow keys while naming neither axis.
 *
 * `catalog.test.ts` owns the file's own structure — class runs, vendor runs, and the prose that states
 * them. What is asserted here is the other half of the issue: that the structure reaches a reader.
 * Every claim below is about rendered markup and fails against the flat version.
 */
describe('the catalog shows the order it is listed in', () => {
  const hardware = () => screen.getByLabelText(SETTING_LABELS.deviceId) as HTMLSelectElement;

  /**
   * The bands a given set of rows has, in file order, paired with the heading each expects.
   *
   * Parameterised because the two surfaces render different sets: the picker offers the whole catalog,
   * the Matrix only the shipping rows. They happen to produce the same three bands today, and
   * "happen to" is what this file keeps recording as the thing that stops being true.
   */
  const expectedBands = (rows: readonly (typeof DEVICES)[number][]) => {
    const bands: { label: string; ids: string[] }[] = [];
    for (const device of rows) {
      const label = DEVICE_CLASS_LABELS[device.class];
      const last = bands.at(-1);
      if (last && last.label === label) last.ids.push(device.id);
      else bands.push({ label, ids: [device.id] });
    }
    return bands;
  };

  it('gives the Hardware picker a heading per class band, over the rows the file already grouped', () => {
    render(<App />);

    const groups = [...hardware().querySelectorAll('optgroup')].map((group) => ({
      label: group.label,
      ids: [...group.querySelectorAll('option')].map((option) => option.value),
    }));

    // The premise, so a picker that grew one group and lost the rest cannot pass the comparison below
    // for the wrong reason.
    expect(groups.length, 'the picker renders no optgroups at all').toBe(3);
    // Whole thing at once — headings *and* membership *and* sequence. Grouping by filtering the
    // catalog three times would satisfy a check on the headings while quietly owning the order.
    expect(groups).toEqual(expectedBands(DEVICES));
  });

  /**
   * That the grouping cannot reorder the list, demonstrated on a list where reordering would show.
   *
   * Asserting the catalog's own order against the rendered options proves nothing here: `DEVICES` is
   * already class-grouped in the declared band order, so a `Select` that built its groups by
   * filtering the list three times would emit the same sequence and pass. This is the mechanism
   * instead — `optionRuns` splits on a *change* of group, so an interleaved list renders as two runs
   * with one heading rather than being tidied into one, and the sequence the call site passed survives
   * untouched. That distinction is why the Hardware picker can be grouped at all without taking
   * ownership of `devices.json`'s order.
   */
  it('groups a Select by adjacency, so no call site loses the order it passed', () => {
    const options = [
      { value: 'a', label: 'A', group: 'First' },
      { value: 'b', label: 'B', group: 'Second' },
      { value: 'c', label: 'C', group: 'First' },
      { value: 'd', label: 'D' },
    ];
    render(<Select label="Interleaved" value="a" onChange={() => {}} options={options} />);

    const select = screen.getByLabelText('Interleaved') as HTMLSelectElement;
    expect([...select.options].map((o) => o.value)).toEqual(['a', 'b', 'c', 'd']);
    expect(
      [...select.querySelectorAll('optgroup')].map((g) => ({
        label: g.label,
        ids: [...g.querySelectorAll('option')].map((o) => o.value),
      }))
    ).toEqual([
      { label: 'First', ids: ['a'] },
      { label: 'Second', ids: ['b'] },
      { label: 'First', ids: ['c'] },
    ]);
    // And an option with no group is rendered outside every heading rather than swept into the last
    // one — the two forms compose, which is what lets the other three pickers stay ungrouped.
    expect(select.querySelector('option[value="d"]')!.closest('optgroup')).toBeNull();
  });

  it('marks every column that opens a class band on the Matrix, and only those', () => {
    atFullGrid();
    render(<App />);

    const matrix = screen.getByRole('region', { name: /every model on every machine/i });
    const shipping = DEVICES.filter((d) => d.status === 'shipping');
    // Adjacency, like the component: the first column opens the first band and needs no separator,
    // since there is nothing to its left to be separated from.
    const expected = shipping.filter((d, i) => i > 0 && d.class !== shipping[i - 1].class);
    expect(expected.length, 'the shipping catalog spans one class, so this proves nothing').toBe(2);

    /**
     * Every column's heading, paired with whether it carries the band gap.
     *
     * By `data-band-start` rather than by the utility class that draws the gap. The first version of
     * this read `classList.contains()` on the border utility, and that border is a
     * `calc(var(--spacing) * 2)` now — the same length in the unit the columns are measured in — so a
     * class-name assertion would have gone quietly false while the markup got *more* correct. The
     * attribute is what the component promises; the border is how it currently looks.
     */
    const separated = [...matrix.querySelectorAll('thead th')]
      .slice(1)
      .map((th, i) => ({ id: shipping[i].id, gap: th.hasAttribute('data-band-start') }));
    expect(separated.filter((c) => c.gap).map((c) => c.id)).toEqual(expected.map((d) => d.id));

    // And down the grid, not only across the header — the gap is a full-height channel or it is a
    // decoration on the labels. One body row is enough: the class is a property of the column.
    const firstRow = matrix.querySelectorAll('tbody tr')[0];
    const cells = [...firstRow.querySelectorAll('td')].map((td, i) => ({
      id: shipping[i].id,
      gap: td.hasAttribute('data-band-start'),
    }));
    expect(cells.filter((c) => c.gap).map((c) => c.id)).toEqual(expected.map((d) => d.id));
  });

  /**
   * The gap, keyed where a sighted reader can find it.
   *
   * The band gap shipped named only inside the `sr-only` caption: a screen-reader user was told the
   * columns are grouped, and a sighted reader met two channels of whitespace with nothing on the page
   * saying what divided them — while the legend beside it keys every other mark on the surface. That is
   * #73's asymmetry, on the same surface and in the same direction, and the caption assertion below
   * passes happily with it live, which is why this is a separate case.
   */
  it('keys the band gap on the page, not only in the caption', () => {
    render(<App />);

    const matrix = screen.getByRole('region', { name: /every model on every machine/i });
    const key = within(matrix).getByText(/a gap between columns/i);
    // Outside the caption, which is the whole claim: `sr-only` text would satisfy a text query and
    // leave the sighted channel exactly as unkeyed as it was.
    expect(key.closest('caption')).toBeNull();
    // And it answers the question the gap raises rather than only labelling it — the bands, in order,
    // in the words the picker's headings use.
    expect(key).toHaveTextContent(
      expectedBands(DEVICES.filter((d) => d.status === 'shipping'))
        .map((band) => band.label)
        .join(', ')
    );
  });

  it('names both of the Matrix’s axes in its caption, which its headings cannot', () => {
    render(<App />);

    const matrix = screen.getByRole('region', { name: /every model on every machine/i });
    const caption = matrix.querySelector('caption')!.textContent ?? '';

    // The bands, in order and by the same names the picker's headings use — a reader who met
    // "Discrete GPUs" in the Hardware control should hear the same words here. Off the *shipping*
    // rows, which is what this grid renders.
    expect(caption).toContain(
      expectedBands(DEVICES.filter((d) => d.status === 'shipping'))
        .map((band) => band.label)
        .join(', ')
    );
    // And the row axis, which is the one fact the Matrix cannot state anywhere else: its row headings
    // are name-only, so 35 rows appeared in an order with no stated basis.
    expect(caption).toMatch(/rows run most-downloaded first/i);
  });

  /**
   * The other channel (#135). #79 gave the sort and the membership to the sr-only caption, and the
   * visual channel got neither — a sighted reader scanning 35 rows could not tell top-N from
   * curation, and so could not tell whether a missing model was refused or simply never asked.
   * That last misreading is the harmful one on a grid whose whole point is refusals with reasons.
   */
  it('states how a model earns a row, and what absence means, on both channels', () => {
    render(<App />);

    const statements = screen.getAllByText(/not one found unable to run/i);
    // One in the sr-only caption, one where sighted readers scan — parity, not a swap: the
    // original defect was one channel getting less than the other, in either direction.
    expect(statements.some((el) => el.closest('caption') !== null)).toBe(true);
    const visible = statements.find((el) => el.closest('caption') === null);
    expect(visible).toBeDefined();
    // The visible sentence carries all three facts: the criterion, the sort, the absence rule.
    expect(visible!.textContent).toMatch(/curated set, not a top-N/i);
    expect(visible!.textContent).toMatch(/most-downloaded first/i);
  });
});
