import { render, screen, within } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { Launch } from './Launch';
import { DEFAULT_CONFIG, evaluateConfig, type Config } from '@/store/config';

/**
 * The launch panel's own rules, which are about *presentation of refusals* rather than about
 * commands — `src/lib/launch.test.ts` owns the strings.
 *
 * The one judgement this component makes is which refusals deserve a reader's attention. A
 * launcher whose both forms refuse for one reason has a real problem to state; a launcher refusing
 * only the *other* form is simply the other kind of binary, and rendering that as a warning box
 * would put three problem-shaped panels on a page with no problem on it. Both paths are asserted
 * here because the difference is invisible from the emitter's side — it returns a refusal either
 * way.
 *
 * Rendered directly rather than through `<App>`: this panel does not depend on the grid, and the
 * #101 lesson is that a test which renders the whole page pays for 1,470 cells to look at one
 * section.
 */

const panel = () => screen.getByRole('region', { name: /run it/i });

function renderAt(over: Partial<Config> = {}) {
  const config = { ...DEFAULT_CONFIG, ...over };
  return render(<Launch config={config} placement={evaluateConfig(config).placement} />);
}

describe('the launch panel', () => {
  it('names each launcher of the selected runtime, because one row can be two surfaces', () => {
    renderAt({ runtimeId: 'llama.cpp' });

    // The catalog row is labelled "llama.cpp / Ollama" and they are different command surfaces, so
    // a placement carries no signal for which one the reader runs. Both, labelled, never a guess.
    for (const name of [/llama-server/i, /Ollama/i, /llama-bench/i]) {
      expect(within(panel()).getByRole('heading', { name, level: 3 })).toBeInTheDocument();
    }
  });

  it('gives every emitted command a copy button of its own', () => {
    renderAt({ runtimeId: 'llama.cpp' });

    const copies = within(panel()).getAllByRole('button', { name: /^copy the .+ command$/i });
    // Serve for llama-server and Ollama, measure for llama-bench — three commands, three buttons.
    expect(copies).toHaveLength(3);
  });

  it('states a real refusal once, as a note, rather than twice as two problems', () => {
    // MLX cannot load Q4_K_M: the figures are a stand-in by width and the checkpoint does not
    // exist, so both forms refuse for one reason. The panel says it once.
    renderAt({
      runtimeId: 'mlx',
      deviceId: 'mac-studio-m3-ultra-256',
      modelId: 'Qwen/Qwen3-8B',
      quantId: 'q4_k_m',
    });

    const notes = within(panel()).getAllByRole('note');
    expect(notes).toHaveLength(1);
    expect(notes[0].textContent).toMatch(/MLX does not load/i);
    // And no command survived it, in either direction.
    expect(within(panel()).queryByRole('button', { name: /copy/i })).not.toBeInTheDocument();
  });

  it('does not dress "this is the other binary" up as a problem', () => {
    renderAt({ runtimeId: 'llama.cpp' });

    // llama-server refuses to measure and llama-bench refuses to serve, and neither is a finding —
    // the command each points at is on this same panel. A `role="note"` on those would put three
    // warning boxes on a page whose configuration is fine.
    expect(within(panel()).queryByRole('note')).not.toBeInTheDocument();
    expect(within(panel()).getByText(/does not measure/i)).toBeInTheDocument();
    expect(within(panel()).getByText(/does not serve/i)).toBeInTheDocument();
  });

  it('carries each launcher’s provenance, since a command is a claim about flags', () => {
    renderAt({ runtimeId: 'llama.cpp' });

    const links = within(panel()).getAllByRole('link', { name: /upstream documentation/i });
    expect(links).toHaveLength(3);
    for (const link of links) {
      expect(link.getAttribute('href')).toMatch(/^https:\/\//);
      // A new tab needs the opener severed; this is the only place on the page with one.
      expect(link.getAttribute('rel')).toContain('noreferrer');
    }
  });

  it('drops the Ollama command at a concurrency it cannot express, without alarm', () => {
    // #171: Ollama takes parallelism as a daemon setting, so a Modelfile this surface can write
    // sizes memory for one user against a panel that priced eight. The serving form refuses — but
    // llama-server emits the same scenario one heading up, so this is navigation and not a
    // problem, and it must not render as a third warning box on a page with nothing wrong on it.
    renderAt({ runtimeId: 'llama.cpp', concurrency: 8 });

    expect(within(panel()).queryByRole('note')).not.toBeInTheDocument();
    expect(within(panel()).getByText(/OLLAMA_NUM_PARALLEL/)).toBeInTheDocument();
    // Two commands rather than three: llama-server serves it, llama-bench measures it, Ollama has
    // nothing to copy.
    expect(within(panel()).getAllByRole('button', { name: /^copy the .+ command$/i })).toHaveLength(
      2
    );
  });

  it('renders the notes rather than hiding them behind a disclosure', () => {
    // Each note is a place the command and the panel above it could be read as agreeing when they
    // do not, and a caveat nobody opens is a caveat nobody has.
    renderAt({ runtimeId: 'llama.cpp' });

    expect(within(panel()).getByText(/-c is the whole cache/i)).toBeVisible();
    expect(within(panel()).queryByRole('button', { expanded: false })).not.toBeInTheDocument();
  });
});
