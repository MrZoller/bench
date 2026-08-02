import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it } from 'vitest';
import { Calibrate } from './Calibrate';
import { DEFAULT_CONFIG, evaluateConfig, useConfig } from '@/store/config';

/**
 * The calibration panel (#139).
 *
 * `calibrate.test.ts` owns the parser and the comparison. What is left for here is the two claims
 * the *surface* makes: that the pasted text stays put, and that a pair which is not comparable is
 * never rendered as a hit or a miss. A delta printed beside two different jobs is noise wearing a
 * data point's chassis, and it is the surface that would make it look like evidence.
 */

/**
 * A paste that really is comparable with the default scenario: its own prompt length (8,192), and
 * its own format — gpt-oss at MXFP4, which is what `DEFAULT_CONFIG` selects. Both matter, and the
 * format one is the easier to get wrong: a fixture naming a different quantization is marked "not
 * comparable" by design, so a test built on one would be asserting the mismatch path while claiming
 * to assert the comparison.
 */
const MEASUREMENT = `
| model | size | params | backend | ngl | test | t/s |
| ----- | ---: | -----: | ------- | --: | ---: | --: |
| gpt-oss 120B MXFP4 | 59.0 GiB | 116.8 B | CUDA | 37 | pp8192 | 7285.68 ± 100.06 |
`;

const panel = () => screen.getByRole('region', { name: /check these numbers/i });

beforeEach(() => {
  useConfig.getState().replace(DEFAULT_CONFIG);
});

function renderPanel() {
  const config = useConfig.getState();
  return render(<Calibrate evaluation={evaluateConfig(config)} />);
}

describe('the panel', () => {
  it('keeps the field behind a disclosure', async () => {
    // The one panel that asks the reader for something rather than telling them something. Open by
    // default it is a large empty textarea on every page view.
    renderPanel();
    const region = within(panel())
      .getByLabelText(/llama-bench output/i)
      .closest('div');

    expect(region?.hasAttribute('hidden')).toBe(true);
    await userEvent
      .setup()
      .click(within(panel()).getByRole('button', { name: /paste a measurement/i }));
    expect(region?.hasAttribute('hidden')).toBe(false);
  });

  it('names the band it is asking to be checked against', () => {
    renderPanel();
    expect(within(panel()).getByText(/±30%/)).toBeInTheDocument();
  });

  it('says the paste stays in the browser before asking for it', async () => {
    // No backend and no telemetry is the whole shape of the feature, and a reader pasting the
    // output of a command they ran on their own machine deserves to be told so *before* they do it
    // — which is why the sentence sits with the field rather than with the submission link. It was
    // under the link at first, where it reassured people who had already pasted.
    renderPanel();
    await userEvent
      .setup()
      .click(within(panel()).getByRole('button', { name: /paste a measurement/i }));

    expect(within(panel()).getByText(/never leaves this page/i)).toBeInTheDocument();
    expect(within(panel()).queryByRole('table')).not.toBeInTheDocument();
  });
});

describe('a pasted measurement', () => {
  const paste = async (text: string) => {
    const user = userEvent.setup();
    renderPanel();
    await user.click(within(panel()).getByRole('button', { name: /paste a measurement/i }));
    await user.click(within(panel()).getByLabelText(/llama-bench output/i));
    await user.paste(text);
  };

  it('shows both figures rather than only the verdict', async () => {
    await paste(MEASUREMENT);

    const row = within(panel()).getByRole('row', { name: /prefill/i });
    expect(within(row).getByText('7285.7')).toBeInTheDocument();
    // And the prediction beside it, which is what makes this a comparison rather than a readout.
    expect(within(row).getAllByRole('cell').length).toBeGreaterThanOrEqual(3);
  });

  it('offers the submission only once there is a scenario to name', async () => {
    renderPanel();
    await userEvent
      .setup()
      .click(within(panel()).getByRole('button', { name: /paste a measurement/i }));

    // Nothing pasted: no link, because a measurement that cannot name its scenario is unusable.
    expect(within(panel()).queryByRole('link', { name: /submit/i })).not.toBeInTheDocument();
  });

  it('carries the scenario link into the pre-filled issue', async () => {
    await paste(MEASUREMENT);

    const link = within(panel()).getByRole('link', { name: /submit this pair/i });
    const href = link.getAttribute('href') ?? '';
    expect(href).toContain('/issues/new?');

    const body = new URL(href).searchParams.get('body') ?? '';
    // The querystring already round-trips a scenario, and that is the reproducible half of a data
    // point. A description of one is not.
    expect(body).toMatch(/\?.*d=/);
    expect(body).toContain('Scenario:');
  });

  it('names a paste it could not read rather than failing silently', async () => {
    await paste('this is not llama-bench output');

    expect(within(panel()).getByText(/No benchmark rows in that/i)).toBeInTheDocument();
    expect(within(panel()).queryByRole('table')).not.toBeInTheDocument();
  });

  it('refuses to grade a pair that is not comparable', async () => {
    /**
     * The claim the surface owns. `pp512` against a prediction made at the default scenario's
     * 8,192 tokens is two different jobs, and printing a percentage beside them would make the
     * difference between two jobs look like a disagreement about the model.
     */
    await paste(
      `| gpt-oss 120B MXFP4 | 59.0 GiB | 116.8 B | CUDA | 37 | pp512 | 7285.68 ± 100.06 |`
    );

    const row = within(panel()).getByRole('row', { name: /prefill/i });
    expect(within(row).getByText('—')).toBeInTheDocument();
    expect(within(row).getByText(/not comparable/i)).toBeInTheDocument();
    expect(within(panel()).getByText(/where the prediction is for/i)).toBeInTheDocument();
  });

  it('states the verdict in words as well as in colour', async () => {
    // Same rule as every other graded figure here: a verdict carried by hue alone is not a verdict
    // for everyone.
    await paste(MEASUREMENT);

    const row = within(panel()).getByRole('row', { name: /prefill/i });
    expect(within(row).getByText(/the ±30% band/)).toBeInTheDocument();
  });
});
