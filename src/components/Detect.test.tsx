import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { Detect } from './Detect';
import { DEFAULT_CONFIG, useConfig } from '@/store/config';
import { GIB } from '@/engine/types';

/**
 * The detection affordance's own behaviour (#137).
 *
 * `detect.test.ts` owns the mapping. What is left for here is the contract the issue is actually
 * about: **detection never selects.** Every path either offers buttons the reader presses or says
 * why it cannot, and the store is untouched until one is pressed.
 *
 * A stubbed `navigator.gpu`, because jsdom has none — which is itself one of the cases under test.
 */

/** Enough of the WebGPU surface to be read, shaped like what a real adapter returns. */
function stubAdapter(info: Record<string, unknown> | null, limits?: Record<string, number>) {
  Object.defineProperty(navigator, 'gpu', {
    configurable: true,
    value: {
      requestAdapter: () => Promise.resolve(info === null ? null : { info, limits }),
    },
  });
}

beforeEach(() => {
  useConfig.getState().replace(DEFAULT_CONFIG);
});

afterEach(() => {
  Reflect.deleteProperty(navigator, 'gpu');
});

const button = () => screen.getByRole('button', { name: /what can my machine run/i });

describe('when the browser exposes nothing', () => {
  it('names the picker rather than reporting a failure', async () => {
    // jsdom has no `navigator.gpu`, which is also Safari behind a flag and any hardened browser. A
    // browser doing what it was configured to do is not an error state.
    const user = userEvent.setup();
    render(<Detect />);
    await user.click(button());

    const said = await screen.findByText(/exposes no graphics adapter/i);
    expect(said).toBeInTheDocument();
    // Announced too: this path inserts only a paragraph, and there is not even a new focusable
    // control to meet by tabbing, so without a live region a screen-reader user learns nothing.
    expect(said.getAttribute('aria-live')).toBe('polite');
    expect(screen.queryByRole('heading')).not.toBeInTheDocument();
  });

  it('leaves the hardware selection alone', async () => {
    const user = userEvent.setup();
    render(<Detect />);
    await user.click(button());
    await screen.findByText(/exposes no graphics adapter/i);

    expect(useConfig.getState().deviceId).toBe(DEFAULT_CONFIG.deviceId);
  });
});

describe('when the signals narrow to a shortlist', () => {
  it('offers the candidates as buttons and selects none of them', async () => {
    const user = userEvent.setup();
    stubAdapter({ vendor: 'intel', architecture: 'xe-2hpg' });
    render(<Detect />);
    await user.click(button());

    const panel = await screen.findByRole('region', { name: /which of these is yours/i });
    const choices = within(panel).getAllByRole('button');
    expect(choices.length).toBeGreaterThan(0);

    // The contract, and the whole reason the feature is shaped this way: nothing is applied on the
    // reader's behalf. A wrong guess silently applied is invented data wearing the chassis of a
    // measurement.
    expect(useConfig.getState().deviceId).toBe(DEFAULT_CONFIG.deviceId);
  });

  it('says which candidate was chosen, once one is', async () => {
    // The Hardware select that otherwise reflects the choice is off screen in a long list, so
    // pressing a button left the reader on an unchanged control with no sign anything happened.
    const user = userEvent.setup();
    stubAdapter({ vendor: 'intel', architecture: 'xe-2hpg' });
    render(<Detect />);
    await user.click(button());

    const panel = await screen.findByRole('region', { name: /which of these is yours/i });
    const choices = within(panel).getAllByRole('button');
    expect(choices.every((b) => b.getAttribute('aria-pressed') === 'false')).toBe(true);

    await user.click(choices[0]);
    expect(choices[0].getAttribute('aria-pressed')).toBe('true');
  });

  it('sets the hardware only when a candidate is pressed', async () => {
    const user = userEvent.setup();
    stubAdapter({ vendor: 'intel', architecture: 'xe-2hpg' });
    render(<Detect />);
    await user.click(button());

    const panel = await screen.findByRole('region', { name: /which of these is yours/i });
    await user.click(within(panel).getAllByRole('button')[0]);

    expect(useConfig.getState().deviceId).not.toBe(DEFAULT_CONFIG.deviceId);
  });

  it('shows what it read, so the shortlist can be judged rather than trusted', async () => {
    const user = userEvent.setup();
    stubAdapter({ vendor: 'intel', architecture: 'xe-2hpg' }, { maxBufferSize: 4 * GIB });
    render(<Detect />);
    await user.click(button());

    const panel = await screen.findByRole('region', { name: /which of these is yours/i });
    expect(within(panel).getByText(/xe-2hpg/i)).toBeInTheDocument();
  });
});

describe('when the signals contradict each other', () => {
  it('offers the adapter’s rows and says a reading was ignored', async () => {
    // An Intel Mac: Chrome ships WebGPU on Metal for those, so the adapter is Intel on a macOS
    // platform. The first version rendered "Which of these is yours?" over an empty list.
    const user = userEvent.setup();
    stubAdapter({ vendor: 'intel' });
    Object.defineProperty(navigator, 'userAgentData', {
      configurable: true,
      value: { platform: 'macOS' },
    });
    try {
      render(<Detect />);
      await user.click(button());

      const panel = await screen.findByRole('region', { name: /which of these is yours/i });
      expect(within(panel).getAllByRole('button').length).toBeGreaterThan(0);
      expect(within(panel).getByText(/cannot both be true/i)).toBeInTheDocument();
    } finally {
      Reflect.deleteProperty(navigator, 'userAgentData');
    }
  });
});

describe('when the signals do not narrow enough', () => {
  it('says the list is long and offers it anyway', async () => {
    /**
     * A redacting browser: vendor and nothing else, which is seventeen shipping NVIDIA rows. The
     * first version hid them and pointed at the picker — **discarding the narrowing exactly where
     * it was worth most**, since the picker is the unfiltered forty-three. Seventeen buttons is
     * still far better than that, so the reader is told it is a long list and given it.
     */
    const user = userEvent.setup();
    stubAdapter({ vendor: 'nvidia' });
    render(<Detect />);
    await user.click(button());

    const panel = await screen.findByRole('region', { name: /which of these is yours/i });
    expect(within(panel).getByText(/more than a shortlist/i)).toBeInTheDocument();
    expect(within(panel).getAllByRole('button').length).toBeGreaterThan(6);
  });

  it('says why a Mac cannot be narrowed, which is the platform it matters most on', async () => {
    // Apple GPUs report no DeviceID through Metal, so the architecture is a feature family and
    // every Apple silicon Mac reports one of three. Ten rows, and the reason is the useful part.
    const user = userEvent.setup();
    stubAdapter({ vendor: 'apple', architecture: 'common-3' });
    render(<Detect />);
    await user.click(button());

    const panel = await screen.findByRole('region', { name: /which of these is yours/i });
    expect(within(panel).getByText(/Metal feature family/i)).toBeInTheDocument();
    expect(within(panel).getByText(/How much memory does it have/i)).toBeInTheDocument();
  });

  it('announces the result, since the read is asynchronous and inserts only visual content', async () => {
    // Without a status role a screen-reader user who pressed the button gets no indication that
    // anything happened — there is not even a new focusable control on the unavailable path.
    const user = userEvent.setup();
    stubAdapter({ vendor: 'nvidia' });
    render(<Detect />);
    await user.click(button());

    const panel = await screen.findByRole('region', { name: /which of these is yours/i });
    // `aria-live`, not `role="status"` — the latter replaces the implicit role and would take this
    // element out of the landmark it is named by, which is what the locator above proves.
    expect(panel.getAttribute('aria-live')).toBe('polite');
  });
});
