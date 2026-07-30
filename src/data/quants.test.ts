import { describe, expect, it } from 'vitest';
import { QUANTS } from './quants';
import { getRuntime } from './runtimes';

/**
 * The format list's own order, which is the order the picker shows (#79).
 *
 * `Bench.tsx` filters `QUANTS` to the formats a scenario admits and maps it. There is no sort anywhere
 * in the repo and no rank field, so this file is the sequence a reader gets — the same contract
 * `devices.json` carries in `$comment-order`, which nothing stated or checked here until the review of
 * #79 audited the class rather than the instance it was filed about.
 *
 * **One boundary is derivable and is checked; the other two are prose.** The docblock on `QUANTS` states
 * three families — the formats a framework loads natively, the GGUF K- and I-quants, and the calibrated
 * packer a serving stack reads — and no field records which family a row is in. Two of those boundaries
 * are therefore only checkable by adding a field that restates the order, which is the trap
 * `$comment-order` names on the device side. The GGUF run is different: it is exactly the set of formats
 * llama.cpp can load and vLLM cannot, which is a fact about two runtimes' `weightFormats` rather than
 * about this file's sequence. That makes it a real check, and it is the one a bpw sort trips over.
 */
describe('the format catalog is listed in the order it states', () => {
  const llamaCpp = getRuntime('llama.cpp');
  const vllm = getRuntime('vllm');

  /** The GGUF formats, derived from the two runtimes' containers rather than from `QUANTS`. */
  const gguf = new Set(
    llamaCpp.weightFormats.filter((format) => !vllm.weightFormats.includes(format))
  );

  it('keeps the GGUF formats in one run, widest first', () => {
    // The premise, twice over: a set of one is contiguous by construction, and a set of everything
    // makes the contiguity claim vacuous. Against today's catalog this is six of twelve.
    expect(gguf.size, 'no format is GGUF-only, so this measures nothing').toBeGreaterThan(1);
    expect(gguf.size).toBeLessThan(QUANTS.length);

    const positions = QUANTS.map((quant, index) => ({ quant, index })).filter(({ quant }) =>
      gguf.has(quant.id)
    );
    // Contiguity as a sequence of consecutive indices, so a K-quant filed among the vendor formats —
    // or a global bits-descending sort, which lifts `q8_0` above `fp8` and drops `q3_k_m` below
    // `awq_4bit` — fails here rather than merely looking odd in the picker.
    expect(
      positions.map(({ index }) => index),
      `GGUF rows at ${positions.map(({ quant }) => quant.id).join(', ')}`
    ).toEqual(positions.map((_, i) => positions[0].index + i));

    // And the ladder inside the run, which is the half a reader uses: scanning down the picker moves
    // towards smaller and lossier. Strictly descending, since two GGUF formats of identical effective
    // width would be a curation question rather than an ordering one.
    const widths = positions.map(({ quant }) => quant.bpw);
    expect(widths).toEqual([...widths].sort((a, b) => b - a));
    expect(new Set(widths).size, `GGUF widths: ${widths.join(', ')}`).toBe(widths.length);
  });

  it('opens at reference precision, whatever the runtime', () => {
    // The first row is the widest format in the file, so every picker — each of which is this list
    // filtered — opens on "the weights as trained" and descends from there. BF16 is in all three
    // runtimes' `weightFormats`, which is what makes the claim hold after filtering as well as before.
    expect(QUANTS[0].bpw).toBe(Math.max(...QUANTS.map((quant) => quant.bpw)));
    for (const runtime of [llamaCpp, vllm, getRuntime('mlx')]) {
      expect(runtime.weightFormats, `${runtime.id} cannot load the list's first row`).toContain(
        QUANTS[0].id
      );
    }
  });
});
