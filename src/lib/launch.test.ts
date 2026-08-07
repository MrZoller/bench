import { describe, expect, it } from 'vitest';
import { artifactFor, launchCommands, type Emission, type LaunchInput } from './launch';
import { planPlacement } from '@/engine/placement';
import {
  DEEPSEEK_V3,
  DGX_SPARK,
  GEMMA_3_12B,
  GPT_OSS_20B,
  GPT_OSS_120B,
  LLAMA_31_8B,
  LLAMA_32_3B,
  LLAMA_CPP,
  MAC_STUDIO_M3_ULTRA_256,
  MLX,
  QWEN3_32B,
  RTX_4090,
  RTX_5090,
  VLLM,
} from '@/engine/fixtures';
import { isSlidingLayer } from '@/engine/kv';
import { weightBreakdown } from '@/engine/weights';
import { getQuant } from '@/data/quants';
import { MODELS, getDevice } from '@/data/catalog';
import type { ModelSpec, QuantSpec, RuntimeSpec, UsageSpec } from '@/engine/types';

/**
 * The launch emitter (#136).
 *
 * The thing under test is mostly *refusals*, which is the shape the feature turned out to have: a
 * command is a claim that something can be run, and the catalog can name exactly one checkpoint
 * per model. So the assertions below are as much about what is absent from a string as what is in
 * it — a template that quietly names the source repo for a Q4_K_M selection would read perfectly
 * and start a different model than the one the panel priced.
 *
 * Unit tests only, per the issue's verification note: nothing here is layout.
 */

const usage = (over: Partial<UsageSpec> = {}): UsageSpec => ({
  contextTokens: 8192,
  concurrency: 1,
  kvPrecision: 'fp16',
  ...over,
});

function input(
  model: ModelSpec,
  quant: QuantSpec,
  runtime: RuntimeSpec,
  device = RTX_5090,
  count = 1,
  u: UsageSpec = usage()
): LaunchInput {
  const rig = { device, count };
  return {
    model,
    quant,
    runtime,
    rig,
    usage: u,
    placement: planPlacement(model, quant, u, rig, runtime),
  };
}

/** Every emitted command in a scenario, keyed by launcher — the shape most assertions want. */
function commands(i: LaunchInput): Record<string, { serve: Emission; measure: Emission }> {
  return Object.fromEntries(
    launchCommands(i).map((c) => [c.launcher.id, { serve: c.serve, measure: c.measure }])
  );
}

const text = (e: Emission): string => {
  if (!e.ok) throw new Error(`expected a command, got a refusal: ${e.reason}`);
  return e.text;
};
const reason = (e: Emission): string => {
  if (e.ok) throw new Error(`expected a refusal, got a command: ${e.text}`);
  return e.reason;
};

describe('the catalog can name exactly one checkpoint per model', () => {
  it('names the source repo at the format the repo actually ships', () => {
    // gpt-oss ships MXFP4, and `nativeQuant` says so — so that one pairing has a real artifact.
    expect(artifactFor(GPT_OSS_20B, 'mxfp4')).toBe(GPT_OSS_20B.id);
    // A model with no `nativeQuant` ships unquantized, which is the catalog's `bf16` row.
    expect(artifactFor(LLAMA_31_8B, 'bf16')).toBe(LLAMA_31_8B.id);
  });

  it('names nothing for a conversion published somewhere else', () => {
    // The whole point. A Q4_K_M GGUF of Llama 3.1 8B certainly exists; it is not in this catalog,
    // and the failure mode of guessing is a working-looking command for a different checkpoint.
    expect(artifactFor(LLAMA_31_8B, 'q4_k_m')).toBeUndefined();
    expect(artifactFor(LLAMA_31_8B, 'awq_4bit')).toBeUndefined();
    // Including the model's *own* non-native formats: gpt-oss at BF16 is a conversion too.
    expect(artifactFor(GPT_OSS_20B, 'bf16')).toBeUndefined();
  });

  it('makes every format unnameable on a model whose native format it does not recognise', () => {
    // The direction this fails in. A `quant_method` string with no matching QuantSpec id must not
    // make some *other* format nameable — it makes all of them unnameable.
    const odd: ModelSpec = { ...LLAMA_31_8B, nativeQuant: 'some-future-scheme' };
    for (const quant of ['bf16', 'q4_k_m', 'fp8', 'awq_4bit']) {
      expect(artifactFor(odd, quant), quant).toBeUndefined();
    }
  });

  it('agrees with the shipped catalog, so the rule is not a fixture artefact', () => {
    for (const model of MODELS) {
      const native = model.nativeQuant ?? 'bf16';
      expect(artifactFor(model, native), model.id).toBe(model.id);
    }
  });
});

describe('llama.cpp: one catalog row, three launchers', () => {
  const i = input(LLAMA_31_8B, getQuant('q4_k_m'), LLAMA_CPP);

  it('emits a server, an Ollama Modelfile and a benchmark client', () => {
    expect(Object.keys(commands(i))).toEqual(['llama-server', 'ollama', 'llama-bench']);
  });

  it('never names a checkpoint, because -m is a path on the reader’s disk', () => {
    // The one place a placeholder is honest, and the reason it is angle-bracketed: pasting it
    // unedited has to fail in the shell rather than half-work.
    const served = text(commands(i)['llama-server'].serve);
    expect(served).toContain('<path to your');
    expect(served).toContain('Q4_K_M');
    // And it must not have quietly reached for the source repo instead.
    expect(served).not.toContain(LLAMA_31_8B.id);
  });

  describe('-c is the whole cache, divided among the slots', () => {
    it('multiplies the window by the users, because llama.cpp divides it back', () => {
      const eight = input(
        LLAMA_31_8B,
        getQuant('q4_k_m'),
        LLAMA_CPP,
        RTX_5090,
        1,
        usage({ contextTokens: 4096, concurrency: 8 })
      );
      const served = text(commands(eight)['llama-server'].serve);

      expect(served).toContain('-c 32768');
      expect(served).toContain('-np 8');
      // The failure this guards: passing one user's window would give each of eight slots 512.
      expect(served).not.toContain('-c 4096');
    });

    it('leaves the slot flag off a single-user scenario', () => {
      const served = text(commands(i)['llama-server'].serve);
      expect(served).toContain('-c 8192');
      expect(served).not.toContain('-np');
    });
  });

  describe('-ngl is a layer count, and the layer count has three cases', () => {
    it('adds one for the output tensor when everything is resident', () => {
      // llama.cpp's `n_gpu_layers` counts a position past the repeating blocks, so `layers` alone
      // is one short of the whole model — it sheds layer 0 off the front and keeps the output
      // tensor. Verified against llama-model.cpp at commit 360e134, not recalled (#202).
      const served = text(commands(i)['llama-server'].serve);
      expect(i.placement.offloadFraction).toBe(0);
      expect(served).toContain(`-ngl ${LLAMA_31_8B.layers + 1}`);
    });

    it('adds the same one when the placement spills, which is not a second rule', () => {
      /**
       * **The `+ 1` is not a fully-resident special case, and treating it as one was #204's first
       * defect.** The output tensor occupies a slot for any positive `-ngl`, so `-ngl N` loads
       * `N - 1` repeating layers whether or not anything spilled — the emitter passed the resident
       * count bare and every spilling command was one layer short of what the panel priced.
       */
      const spilled = input(QWEN3_32B, getQuant('bf16'), LLAMA_CPP, RTX_4090);
      expect(spilled.placement.offloadFraction).toBeGreaterThan(0);

      const emitted = commands(spilled)['llama-server'].serve;
      const resident = spilled.placement.assignment.residentLayers;
      expect(resident).toBeGreaterThan(0);
      expect(resident).toBeLessThan(QWEN3_32B.layers);
      expect(text(emitted)).toContain(`-ngl ${resident + 1}`);
      if (!emitted.ok) throw new Error('unreachable');
      expect(emitted.notes.join(' ')).toMatch(/not a fraction of the model/i);
      // And the sentence states the layer count rather than the flag, which are now different
      // numbers: a reader checking the note against the panel is checking the resident count.
      expect(emitted.notes.join(' ')).toContain(
        `-ngl ${resident + 1} keeps ${resident} of ${QWEN3_32B.layers} layers`
      );
    });

    it('has its own sentence at zero on a GPU rig, rather than subtracting the output slot', () => {
      /**
       * A GPU with no room is not a rig with no GPU, and the general spilling note cannot serve
       * both: it subtracts the output slot, which only occupies a position for a *positive*
       * `-ngl`. Written without this branch the note read "llama.cpp reads it as -1 repeating
       * layers", on 2,202 catalog configurations — reachable wherever the resident fraction
       * floors to zero, which `residentLayersOf`'s `Math.max(0, …)` permits by design.
       */
      /* A card small enough that no whole layer survives beside the cache, synthesised rather
         than named: the real rows that reach this are the 16 GB 50-series parts, and pinning a
         catalog id here would make this fail the day one is re-specced (#197). */
      const small = {
        ...RTX_4090,
        id: 'small-fixture',
        name: 'Small fixture',
        capacityBytes: 16 * 1024 ** 3,
        allocatableBytes: 15 * 1024 ** 3,
      };
      const nothingFits = input(
        LLAMA_32_3B,
        getQuant('bf16'),
        LLAMA_CPP,
        small,
        1,
        usage({ contextTokens: 131072 })
      );

      expect(nothingFits.placement.assignment.residentLayers).toBe(0);
      const emitted = commands(nothingFits)['llama-server'].serve;
      if (!emitted.ok) throw new Error('unreachable');
      const notes = emitted.notes.join(' ');

      expect(notes).not.toMatch(/-\d+ repeating layers/);
      expect(notes).toMatch(/puts nothing on the GPU/i);
      // Not the cpu-ram sentence: this machine has a GPU, it just has no room.
      expect(notes).not.toMatch(/no GPU to offload to/i);

      /**
       * And it must not claim to match the panel, which is the mistake this note made first.
       * `residentLayers` floors, so it hits zero while a fraction is still resident — across the
       * 4,302 configurations reaching this note the spill runs 80.5% to 99.9% and is **never**
       * 100%. So `-ngl 0`, which puts nothing on a GPU, is always a slower placement than the one
       * priced, and a sentence saying otherwise is false on every case it renders.
       */
      expect(nothingFits.placement.offloadFraction).toBeGreaterThan(0);
      expect(nothingFits.placement.offloadFraction).toBeLessThan(1);
      expect(notes).toMatch(/slower than the panel estimates/i);
      expect(notes).not.toMatch(/what the figures above price/i);
    });

    it('is zero on a machine with no GPU to offload to', () => {
      // The correction the engine deliberately does not make: a cpu-ram rig reports every layer
      // resident, truthfully — nothing spilled, because there was nowhere to spill from — while
      // the honest flag is 0.
      const cpu = {
        ...RTX_5090,
        id: 'epyc-fixture',
        name: 'EPYC fixture',
        class: 'cpu-ram' as const,
      };
      const onCpu = input(LLAMA_31_8B, getQuant('q4_k_m'), LLAMA_CPP, cpu);

      expect(onCpu.placement.assignment.residentLayers).toBe(LLAMA_31_8B.layers);
      expect(text(commands(onCpu)['llama-server'].serve)).toContain('-ngl 0');
    });
  });

  /**
   * The caveats on a correction that moves answers *optimistically* (#182).
   *
   * The engine takes the input embedding table off a discrete GPU's budget because llama.cpp pins
   * it to the host, which makes 55,200 of the catalog's single-card configurations lighter and
   * flips 174 of them from "will not run" to "fits". A reader acting on that has bought hardware,
   * so the two modes under which the accounting is not what the panel priced have to be printed
   * beside the command rather than filed in a docblock.
   */
  describe('the host-resident correction states its own caveats', () => {
    /**
     * **Every launcher off the `llama.cpp` row, not just the two that take its flags** (#209).
     *
     * The correction is a property of the placement, and the panel above all three blocks shows the
     * one reduced budget — so an Ollama reader is acting on the same optimistic figure. This loop
     * ran over llama-server and llama-bench only, and the Ollama block, which builds its notes
     * independently, silently carried none of it.
     *
     * The claim is shared and the vocabulary is not: `-sm row` and `--no-mmap` are llama.cpp flags,
     * and an Ollama reader has neither to type. So what crosses is the host-memory requirement,
     * and the flags are asserted absent from the block that cannot use them.
     */
    it('says what came off the card, on every launcher the runtime row emits', () => {
      // Untied, so a whole `vocab x hidden` table is host-resident and the card budget is the file
      // less that. Llama 3.1 8B at Q4_K_M is 4.58 GiB of file against 318 MB of table.
      const untied = input(LLAMA_31_8B, getQuant('q4_k_m'), LLAMA_CPP, RTX_5090);
      expect(untied.placement.deviceWeightBytes).toBeLessThan(untied.placement.totalWeightBytes);

      for (const launcher of ['llama-server', 'ollama', 'llama-bench'] as const) {
        const emitted =
          commands(untied)[launcher][launcher === 'llama-bench' ? 'measure' : 'serve'];
        if (!emitted.ok) throw new Error('unreachable');
        const notes = emitted.notes.join(' ');

        expect(notes, launcher).toMatch(/lighter than the file/i);
        expect(notes, launcher).toMatch(/input embedding table in host RAM/i);
        // The figure carries its unit. It is a host-memory requirement, and "0.3 of committed RAM"
        // is not one (#209).
        expect(notes, launcher).toMatch(/0\.3 GiB/);
        // The panel has no host-RAM input, so every form of the note says the requirement is
        // unchecked rather than implying the budget above covers it.
        expect(notes, launcher).toMatch(/host memory this page does not check/i);

        if (launcher === 'ollama') {
          // Neither flag exists on this surface, and naming one would be a command a reader
          // cannot run. Ollama has no split-mode equivalent at all, and its mmap control is
          // undocumented and conditional — see `ollamaResidencyNote`.
          expect(notes, launcher).not.toMatch(/-sm row/);
          expect(notes, launcher).not.toMatch(/--no-mmap/);
          expect(notes, launcher).not.toMatch(/use_mmap/);
        } else {
          // The two flags, named. Neither is a flag Headroom emits, which is exactly why the note
          // has to: a reader who adds one is running something the panel did not price.
          expect(notes, launcher).toMatch(/-sm row/);
          expect(notes, launcher).toMatch(/--no-mmap/);
        }
      }
    });

    it('stays silent where nothing came off the card', () => {
      // A tied model holds the table on the card too — llama.cpp materialises it twice — so the
      // file and the card budget agree and there is nothing to caveat. Synthesised from the same
      // row so the tie is the only thing that differs (#197).
      const tied = input(
        { ...LLAMA_31_8B, id: 'test/tied', tiedEmbeddings: true },
        getQuant('q4_k_m'),
        LLAMA_CPP,
        RTX_5090
      );
      expect(tied.placement.deviceWeightBytes).toBe(tied.placement.totalWeightBytes);

      const emitted = commands(tied)['llama-server'].serve;
      if (!emitted.ok) throw new Error('unreachable');
      expect(emitted.notes.join(' ')).not.toMatch(/lighter than the file/i);
    });

    it('stays silent on unified memory, where the host is the rig', () => {
      const spark = input(LLAMA_31_8B, getQuant('q4_k_m'), LLAMA_CPP, DGX_SPARK);
      expect(spark.placement.deviceWeightBytes).toBe(spark.placement.totalWeightBytes);

      const emitted = commands(spark)['llama-server'].serve;
      if (!emitted.ok) throw new Error('unreachable');
      expect(emitted.notes.join(' ')).not.toMatch(/lighter than the file/i);
    });
  });

  describe('-ts carries the packing, and only when the packing is uneven', () => {
    /**
     * **The combination the first version of this file never tested, and got wrong.**
     *
     * `-ts` does not distribute the model — it distributes the `-ngl` window, because llama.cpp
     * puts the last `ngl` layers on GPUs and splits *those* by these proportions. So emitting the
     * *assigned* counts beside a resident `-ngl` hands llama.cpp two numbers from different scopes
     * and lets it re-derive a per-device split that is neither: on a rig packing 7,7,6,6 layers and
     * keeping 2,2,6,6 resident, `-ngl 16 -ts 7,7,6,6` spreads sixteen layers slightly in favour of
     * the two cards Headroom sized for two — which are the constrained cards precisely because their
     * cache already fills them. That is an OOM on load, from a command.
     *
     * There were a spilled test and a sharded test, and the defect lived only in their product.
     */
    it('proportions the resident layers, not the assigned ones, on a rig that spills', () => {
      /**
       * Llama 3.1 8B at BF16, 128K over 8 users, five 5090s: the packing assigns 7,7,6,6,6 layers
       * and keeps 5,5,6,6,6 of them resident.
       *
       * A **uniform** model, and that is now load-bearing rather than incidental — see the hybrid
       * test below. It is also the harder case for the flag to get right: the two counts differ
       * only on the cards that spill, so an implementation reading assigned counts looks correct on
       * three of the five entries.
       */
      const spilled = input(
        LLAMA_31_8B,
        getQuant('bf16'),
        LLAMA_CPP,
        RTX_5090,
        5,
        usage({ contextTokens: 131072, concurrency: 8 })
      );

      const shares = spilled.placement.assignment.shares;
      const assigned = shares.flatMap((s) => Array(s.deviceCount).fill(s.layers) as number[]);
      const resident = shares.flatMap(
        (s) => Array(s.deviceCount).fill(s.residentLayers) as number[]
      );

      // The premises, asserted rather than assumed. Without them the assertion below is satisfied
      // by the implementation it was written to reject.
      expect(spilled.placement.offloadFraction, 'nothing spilled').toBeGreaterThan(0);
      expect(resident.join(','), 'resident equals assigned').not.toBe(assigned.join(','));

      const served = text(commands(spilled)['llama-server'].serve);
      // The resident counts with the output tensor's slot on the last card, which is where
      // llama.cpp puts it — see the `upper_bound` suite below for why the slot has to be stated.
      const ratios = [...resident.slice(0, -1), resident[resident.length - 1] + 1];
      expect(ratios[ratios.length - 1]).toBeGreaterThan(1);
      expect(served).toContain(`-ts ${ratios.join(',')}`);
      expect(served).not.toContain(`-ts ${assigned.join(',')}`);

      // And the two flags have to agree, which is the property that makes the pair safe: `-ts`
      // proportions the window `-ngl` opens, so the proportions must sum to it — the *window*,
      // which is one slot wider than the layer count.
      expect(resident.reduce((a, b) => a + b, 0)).toBe(spilled.placement.assignment.residentLayers);
      expect(served).toContain(`-ngl ${ratios.reduce((a, b) => a + b, 0)}`);
      expect(served).toContain(`-ngl ${spilled.placement.assignment.residentLayers + 1}`);
    });

    /**
     * **A count is a faithful description of the packing only where the layers are
     * interchangeable** (raised by Codex on #164, P1), and this was the case the flag looked most
     * useful for.
     *
     * `-ts` partitions llama.cpp's ordered `-ngl` suffix into contiguous device ranges, while
     * `layerSplitBins` assigns *individual* layers by greedy combined load — so a share can be a
     * non-contiguous mixture of full-attention and sliding layers. On Gemma at 128K a full layer
     * caches ~128x a sliding one, so equal counts do not reproduce equal loads: llama.cpp would
     * hand a card a different set of expensive layers than `planPlacement` priced, and the fit the
     * panel reported would not be the fit the command produces.
     *
     * **So the flag stays off and the panel says what it packed instead** (#166). This test was
     * `emits nothing for a hybrid model` and asserted an absence, which is the shape the issue
     * asked to be replaced: an absence is satisfied by an emitter that has nothing to say as
     * happily as by one that has been told to be quiet. What is asserted now is the two lists —
     * the layer counts *and* their composition — that `DeviceShare.layerIndices` made expressible,
     * because the second is the fact the first was hiding.
     */
    it('states the packing it cannot flag, layer counts and composition both', () => {
      const gemma = input(
        GEMMA_3_12B,
        getQuant('q4_k_m'),
        LLAMA_CPP,
        RTX_5090,
        5,
        usage({ contextTokens: 131072, concurrency: 8 })
      );

      const shares = gemma.placement.assignment.shares;
      const counts = shares.map((s) => s.layers);
      const unbounded = shares.map(
        (s) => s.layerIndices.filter((layer) => !isSlidingLayer(GEMMA_3_12B, layer)).length
      );

      // The premises, asserted rather than assumed. The packing really is lopsided here — 2,2,2,21,21
      // against a composition of 2,2,2,1,1, which is the 19-layer spread #166 was filed on — and
      // without them the assertions below would pass against an even split saying nothing.
      expect(Math.max(...counts) - Math.min(...counts), 'the split is even').toBeGreaterThan(1);
      expect(new Set(unbounded).size, 'every card holds the same mixture').toBeGreaterThan(1);

      const emitted = commands(gemma)['llama-server'].serve;
      if (!emitted.ok) throw new Error('unreachable');
      const notes = emitted.notes.join(' ');

      // The positive claim: both lists, in the panel, as the numbers the engine actually packed.
      expect(notes).toContain(`Headroom packed ${counts.join(',')} layers`);
      expect(notes).toContain(`${unbounded.join(',')} of them attending over the whole context`);

      // And why there is no flag for it, which is the half a silent omission cannot say. `-ot`
      // looks like the answer and is not: it overrides where a *weight* lives, while a layer's KV
      // cache follows the device `-ngl` and `-ts` put the layer on — and the cache is the entire
      // reason this packing is uneven.
      expect(notes).toMatch(/llama\.cpp cannot be given that assignment/i);
      expect(notes).toMatch(/-ot names individual tensors/i);
      expect(notes).toMatch(/KV cache follows the device -ngl and -ts put the layer on/i);
      // A floor to plan against rather than a prediction of what the command produces: llama.cpp's
      // contiguous split sometimes lands the same composition, and finding out which would be this
      // module deriving llama.cpp's placement rather than formatting Headroom's.
      expect(notes).toMatch(/Treat the busiest card above as a floor/i);

      // The flag itself still stays off, which is what makes the sentences above the whole of what
      // Headroom claims here.
      expect(text(emitted)).not.toContain('-ts');
    });

    /**
     * **The same sweep the flag itself needed on this launcher.** `-ts` was missing from
     * `llama-bench` once already, because a sharded measurement run at llama.cpp's default split
     * times a placement other than the one priced — and that is as true when Headroom *cannot* express
     * its split as when it declines to repeat an even one. The serving command carried the
     * explanation and the measurement command, whose whole purpose is to produce a number for the
     * calibration record, carried nothing.
     */
    it('qualifies the benchmark run too, since it measures the split it cannot ask for', () => {
      const gemma = input(
        GEMMA_3_12B,
        getQuant('q4_k_m'),
        LLAMA_CPP,
        RTX_5090,
        5,
        usage({ contextTokens: 131072, concurrency: 8 })
      );
      const counts = gemma.placement.assignment.shares.map((s) => s.layers);
      const measured = commands(gemma)['llama-bench'].measure;

      expect(text(measured)).not.toContain('-ts');
      if (!measured.ok) throw new Error('unreachable');
      expect(measured.notes.join(' ')).toContain(`Headroom packed ${counts.join(',')} layers`);
      expect(measured.notes.join(' ')).toMatch(/llama\.cpp cannot be given that assignment/i);
    });

    /**
     * **The product this flag has already been caught on once.** The ROADMAP's note about `-ts`
     * records that the suite had a spilled case and a sharded case and never their conjunction,
     * which is where the defect lived; the packing sentence has the same two axes and a third,
     * because the model must also be hybrid for it to be emitted at all.
     *
     * gpt-oss 120B at Q4_K_M, 128K over 8 users on four 4090s packs 8,8,10,10 layers and keeps
     * 6,6,7,7 of them resident. The note describes the *assignment*, so it is the first list — a
     * sentence built from `residentLayers` would offer a subset of the packing as the packing, and
     * would read as correct on every rig where nothing spills, which is most of them.
     */
    it('describes the packing rather than the resident subset when the rig also spills', () => {
      const spilled = input(
        GPT_OSS_120B,
        getQuant('q4_k_m'),
        LLAMA_CPP,
        RTX_4090,
        4,
        usage({ contextTokens: 131072, concurrency: 8 })
      );
      const shares = spilled.placement.assignment.shares;
      const packed = shares.map((s) => s.layers);
      const resident = shares.map((s) => s.residentLayers);

      // The premises. Without the second this test passes against the wrong quantity.
      expect(spilled.placement.offloadFraction, 'nothing spilled').toBeGreaterThan(0);
      expect(resident.join(','), 'resident equals packed').not.toBe(packed.join(','));

      const emitted = commands(spilled)['llama-server'].serve;
      if (!emitted.ok) throw new Error('unreachable');
      expect(emitted.notes.join(' ')).toContain(`Headroom packed ${packed.join(',')} layers`);
      expect(emitted.notes.join(' ')).not.toContain(`Headroom packed ${resident.join(',')} layers`);
    });

    /**
     * **Being hybrid is a property of the model; caching unequal amounts is a property of the
     * context** — and the flag turns on the second. Below its shortest window every one of Gemma's
     * layers holds the whole context, so the packing hands out equal loads and a count describes it
     * exactly. `hasSlidingLayers` refused the flag here and then explained the refusal with an
     * imbalance that does not exist at this context.
     */
    it('gives a hybrid model the split once its context is inside every window', () => {
      const short = input(
        GEMMA_3_12B,
        getQuant('q4_k_m'),
        LLAMA_CPP,
        RTX_5090,
        5,
        // 1,024 is Gemma's own sliding window, so every layer caches 1,024 tokens.
        usage({ contextTokens: 1024 })
      );
      const counts = short.placement.assignment.shares.map((s) => s.residentLayers);
      const emitted = commands(short)['llama-server'].serve;

      // The premises: 48 layers over five cards is indivisible, so there is a real split to state,
      // and it is one llama.cpp's memory-proportional default does not produce unaided.
      expect(new Set(counts).size, 'the split is even, so this proves nothing').toBeGreaterThan(1);
      expect(counts.reduce((a, b) => a + b, 0)).toBe(GEMMA_3_12B.layers);

      // The output tensor's slot rides on the last card's share; see the `upper_bound` suite below.
      expect(text(emitted)).toContain(
        `-ts ${[...counts.slice(0, -1), counts[counts.length - 1] + 1].join(',')}`
      );
      // And no packing sentences, because there is no refusal left to explain.
      if (!emitted.ok) throw new Error('unreachable');
      expect(emitted.notes.join(' ')).not.toMatch(/Headroom packed/i);
    });

    /**
     * **The guard `tensorSplit` has and the note was written without.** Where the whole rig spilled
     * there is nothing on a GPU at all, `-ngl 0` says so, and two sentences about how llama.cpp
     * will divide the cards' layers between them describe a division that does not happen.
     * gpt-oss 120B at BF16 on two 4090s at 128K over 8 users is the reachable case — 96% of the
     * weights spill — and it is a runnable placement rather than a refused one, which is what keeps
     * the emitter running long enough to say it.
     */
    it('says nothing about a packing when the whole rig spilled and -ngl is 0', () => {
      const spilledOut = input(
        GPT_OSS_120B,
        getQuant('bf16'),
        LLAMA_CPP,
        RTX_4090,
        2,
        usage({ contextTokens: 131072, concurrency: 8 })
      );
      const emitted = commands(spilledOut)['llama-server'].serve;

      // The premises. Without them this passes on any rig that simply refuses.
      expect(spilledOut.placement.impossible, 'refused before the emitter ran').toBe(false);
      expect(spilledOut.placement.assignment.residentLayers, 'something stayed resident').toBe(0);

      expect(text(emitted)).toContain('-ngl 0');
      // And no `-ts` either, which is the shape #209's rule has to be checked against: the extra
      // slot goes to the last share whatever its count, so a rig with nothing resident anywhere
      // would emit a lone `-ts 0,0` with a 1 on the end — ratios for a window that does not exist.
      // `tensorSplit` returns before that on the all-zero guard, and this pins it.
      expect(text(emitted)).not.toContain('-ts');
      if (!emitted.ok) throw new Error('unreachable');
      expect(emitted.notes.join(' ')).not.toMatch(/Headroom packed/i);

      // Both launchers, because both carry the sentences and a guard on one of them is the half-fix
      // this pair has already been through once.
      const measured = commands(spilledOut)['llama-bench'].measure;
      expect(text(measured)).toContain('-ngl 0');
      if (!measured.ok) throw new Error('unreachable');
      expect(measured.notes.join(' ')).not.toMatch(/Headroom packed/i);
    });

    it('says nothing about a packing on a rig with one card, which packs nothing', () => {
      // The note is about a split, so a single device has no subject. Gemma is hybrid on every
      // rig; without this the two sentences above would appear beside a command that is exact.
      const one = input(
        GEMMA_3_12B,
        getQuant('q4_k_m'),
        LLAMA_CPP,
        RTX_5090,
        1,
        usage({ contextTokens: 131072 })
      );
      const emitted = commands(one)['llama-server'].serve;

      if (!emitted.ok) throw new Error('unreachable');
      expect(emitted.notes.join(' ')).not.toMatch(/Headroom packed/i);
    });

    it('gives the benchmark client the same split as the server', () => {
      // A sharded measurement run at llama.cpp's default even split times a different placement
      // than the serving command reproduces, which makes the number unusable for calibration —
      // the one thing the measurement form exists for.
      const dense = input(
        LLAMA_31_8B,
        getQuant('q4_k_m'),
        LLAMA_CPP,
        RTX_5090,
        5,
        usage({ contextTokens: 32768 })
      );
      const counts = dense.placement.assignment.shares.map((s) => s.residentLayers);

      const ts = [...counts.slice(0, -1), counts[counts.length - 1] + 1].join(',');

      expect(new Set(counts).size, 'the split is even, so this proves nothing').toBeGreaterThan(1);
      expect(text(commands(dense)['llama-bench'].measure)).toContain(`-ts ${ts}`);
      expect(text(commands(dense)['llama-server'].serve)).toContain(`-ts ${ts}`);
    });

    it('emits the layer counts an indivisible split actually produced', () => {
      // 32 layers over five cards is 7,7,6,6,6 — uneven, exactly expressible, and not what
      // llama.cpp's memory-proportional default would do on identical cards.
      const dense = input(
        LLAMA_31_8B,
        getQuant('q4_k_m'),
        LLAMA_CPP,
        RTX_5090,
        5,
        usage({ contextTokens: 32768 })
      );
      const served = text(commands(dense)['llama-server'].serve);
      const counts = dense.placement.assignment.shares.map((s) => s.residentLayers);

      expect(new Set(counts).size, 'the split is even, so this proves nothing').toBeGreaterThan(1);
      expect(counts.reduce((a, b) => a + b, 0)).toBe(LLAMA_31_8B.layers);
      // 7,7,7,6,5 layers, emitted as 7,7,7,6,6 slots: the fifth card holds five layers *and* the
      // output tensor, and llama.cpp normalises the ratios over a window that counts it (#204).
      //
      // **The packing is 7,7,7,6,5 rather than the 7,7,6,6,6 #204 was filed on, and #182 is why.**
      // The output projection is 318 MB against a 266 MB combined per-layer load here, so seeding
      // the last bin with it costs that card a layer — which is the layout llama.cpp produces, and
      // was previously smeared across all five.
      expect(served).toContain(`-ts 7,7,7,6,6`);
      expect(served).toContain(`-ngl ${LLAMA_31_8B.layers + 1}`);
    });

    it('stays silent on a single device, where there is nothing to split', () => {
      expect(text(commands(i)['llama-server'].serve)).not.toContain('-ts');
    });

    it('stays silent when the split is even, which is what llama.cpp would do unaided', () => {
      // A dense model divides cleanly, so the flag would say "split this evenly" — which is the
      // default. Emitting it anyway would make the flag noise and hide the case that matters.
      //
      // **At 128K rather than 4K, and the difference is #182's seed.** The output projection now
      // seeds the last bin, so an evenly divisible rig only splits evenly when that block weighs
      // less than one layer's combined load — which it does once the cache is the larger term.
      // Llama 3.1 8B's table is 318 MB against 149 MB of layer at 4K and 669 MB at 128K, so the
      // same rig is 9,9,8,6 there and 8,8,8,8 here.
      const dense = input(
        LLAMA_31_8B,
        getQuant('q4_k_m'),
        LLAMA_CPP,
        RTX_5090,
        4,
        usage({ contextTokens: 131072 })
      );
      const counts = dense.placement.assignment.shares.map((s) => s.layers);
      expect(new Set(counts).size, 'the split is uneven, so this proves nothing').toBe(1);
      expect(text(commands(dense)['llama-server'].serve)).not.toContain('-ts');
    });
  });

  /**
   * **The flags are checked against llama.cpp's own placement rule rather than against a string**
   * (#204), and that is the whole point of this block.
   *
   * Every other assertion above pins what the emitter writes. None of them could have caught either
   * of #204's defects, because both emitted a perfectly well-formed flag carrying the number the
   * panel displayed — a spilling `-ngl N` that loads `N - 1` layers, and `-ts` ratios summing to
   * `L` against a window of `L + 1` slots. The string was what Headroom meant; it was not what
   * llama.cpp does with it. So this suite runs the emitted pair through {@link placeSlots} and
   * asserts the *placement*: the layer counts that end up on each card, and which card gets the
   * output tensor.
   *
   * The reference implementation is about twenty lines, which is what makes this cheap enough to be
   * worth having.
   */
  describe('the emitted -ngl/-ts pair lands the placement Headroom sized', () => {
    /**
     * `llama_model::load_tensors`' device assignment, ported verbatim.
     *
     * Read from `src/llama-model.cpp:1285-1343` at ggml-org/llama.cpp commit `360e134`: the
     * `splits` cumulative-sum normalisation, `i_gpu_start`, `act_gpu_layers`, the
     * `get_layer_buft_list` guard and its `std::upper_bound`, and `dev_output =
     * get_layer_buft_list(n_layer_all)`. Not recalled — the file was fetched at that commit.
     *
     * Two details are load-bearing and neither is obvious from the flag's documentation:
     *
     *   - **The window is `n_layer_all + 1` slots**, the repeating blocks plus the output tensor's
     *     own position, and `-ts` is normalised over that window rather than over the layers. So
     *     ratios summing to the layer count are stretched across one slot more than they describe.
     *   - **`upper_bound` is a strict `>`**, so a boundary falls to the *next* device. With integer
     *     ratios whose sum equals `act_gpu_layers` the comparison is exact at every boundary, which
     *     is what makes the corrected emission land on the nose rather than nearly.
     *
     * `Math.fround` at each step because upstream computes the splits and the key in `float`. It
     * makes no difference to any case here, and it is what keeps this a port rather than a
     * paraphrase of one.
     */
    function placeSlots(
      nLayerAll: number,
      nGpuLayers: number,
      tensorSplit: readonly number[] | null,
      nDevices: number
    ): { layerDevice: number[]; outputDevice: number } {
      const f = Math.fround;
      const allZero = tensorSplit === null || tensorSplit.every((x) => x === 0);
      // The default is by free memory; on the identical cards a `Rig` describes, that is equal.
      const raw = allZero ? Array.from({ length: nDevices }, () => 1) : tensorSplit;

      let sum = 0;
      const splits = raw.slice(0, nDevices).map((x) => (sum = f(sum + x)));
      for (let i = 0; i < nDevices; i++) splits[i] = f(splits[i] / sum);

      const iGpuStart = Math.max(nLayerAll + 1 - nGpuLayers, 0);
      const actGpuLayers = nDevices === 0 ? 0 : Math.min(nGpuLayers, nLayerAll + 1);

      /** The device slot `il` lands on, or -1 for the host. */
      const deviceFor = (il: number): number => {
        if (il < iGpuStart || il - iGpuStart >= actGpuLayers) return -1;
        const key = f(f(il - iGpuStart) / actGpuLayers);
        const at = splits.findIndex((boundary) => boundary > key);
        // Unreachable: the largest key is `(actGpuLayers - 1) / actGpuLayers < 1` and the last
        // boundary is exactly 1. Asserted rather than assumed, since a silent -1 here would read
        // as "on the host" and quietly satisfy every count below.
        expect(at, `no device covers slot ${il}`).toBeGreaterThanOrEqual(0);
        return at;
      };

      return {
        layerDevice: Array.from({ length: nLayerAll }, (_, il) => deviceFor(il)),
        outputDevice: deviceFor(nLayerAll),
      };
    }

    /** What llama.cpp will actually do with a command, read back out of the command itself. */
    function place(i: LaunchInput, command: string) {
      const ngl = Number(/-ngl (\d+)/.exec(command)?.[1]);
      expect(ngl, 'no -ngl in the command').not.toBeNaN();
      const ts = /-ts ([\d,]+)/.exec(command)?.[1];
      const devices = i.rig.count;
      const { layerDevice, outputDevice } = placeSlots(
        i.model.layers,
        ngl,
        ts === undefined ? null : ts.split(',').map(Number),
        devices
      );
      const perDevice = Array.from(
        { length: devices },
        (_, d) => layerDevice.filter((x) => x === d).length
      );
      return { ngl, ts, outputDevice, perDevice, onGpu: layerDevice.filter((x) => x >= 0).length };
    }

    it('worked by hand: 32 layers over five cards, and the fifth keeps the table', () => {
      /**
       * The example #204 is filed on, with #182's packing under it. The packing is 7,7,7,6,5 — the
       * fifth card gives up a layer to the output projection it holds — and emitting the counts bare
       * as `-ngl 33 -ts 7,7,7,6,5` resolves to `8,7,7,6,4`: the first card gains the slot the ratios
       * did not account for, and the last card pays for it. Stating the slot instead
       * (`-ts 7,7,7,6,6`) makes the sum equal `-ngl` and the arithmetic land exactly.
       *
       * #204 was filed against a 7,7,6,6,6 packing, which is what this rig produced while the fixed
       * block was smeared across all five cards. The defect and its fix are unchanged by the move;
       * only the counts they operate on are.
       */
      const five = input(
        LLAMA_31_8B,
        getQuant('q4_k_m'),
        LLAMA_CPP,
        RTX_5090,
        5,
        usage({ contextTokens: 32768 })
      );
      const sized = five.placement.assignment.shares.flatMap(
        (s) => Array(s.deviceCount).fill(s.residentLayers) as number[]
      );
      expect(sized).toEqual([7, 7, 7, 6, 5]);

      const served = text(commands(five)['llama-server'].serve);
      expect(served).toContain('-ngl 33');
      expect(served).toContain('-ts 7,7,7,6,6');

      const got = place(five, served);
      expect(got.perDevice).toEqual([7, 7, 7, 6, 5]);
      expect(got.outputDevice).toBe(4);

      // And the emission this replaced, run through the same reference: the defect, reproduced.
      expect(placeDelivered(five, 33, [7, 7, 7, 6, 5])).toEqual([8, 7, 7, 6, 4]);
    });

    /**
     * **The seeded bin spilled to nothing and kept the table anyway** (#209), end to end.
     *
     * The two configurations #209 was filed on, reproduced from the frozen fixtures so they do not
     * move when the catalog refreshes. The first is that issue's example verbatim — Llama 3.2 3B at
     * Q6_K on four RTX 5080s at 128K over four users, packing `7,7,7,7` layers and keeping
     * `4,4,4,0` of them. The second is its Kimi K2 case, whose `31/1, 30/0` shape DeepSeek V3 at
     * BF16 reaches on two 4090s with the same counts.
     *
     * Both spent the extra slot on the last share *with a layer on it*, which is a different card
     * from the one `planPlacement` charged the output projection to the moment the seeded bin
     * floors — and the seeded bin is the first one to floor, since it carries a fixed block its
     * layer count does not. So the reader was told to put a table on a card that had never been
     * sized for it, beside a card the panel had priced for exactly that and which llama.cpp would
     * then leave empty.
     */
    it('keeps the table on the card the panel priced it onto, even at no layers (#209)', () => {
      const four = input(
        LLAMA_32_3B,
        getQuant('q6_k'),
        LLAMA_CPP,
        getDevice('rtx-5080'),
        4,
        usage({ contextTokens: 131072, concurrency: 4 })
      );
      expect(four.placement.assignment.shares.map((s) => [s.layers, s.residentLayers])).toEqual([
        [7, 4],
        [7, 4],
        [7, 4],
        [7, 0],
      ]);

      const served = text(commands(four)['llama-server'].serve);
      expect(served).toContain('-ngl 13');
      expect(served).toContain('-ts 4,4,4,1');

      const got = place(four, served);
      expect(got.perDevice).toEqual([4, 4, 4, 0]);
      expect(got.outputDevice, 'the table is on the card sized for it').toBe(3);

      // The emission this replaced: card 2 takes a table it was never charged for, and card 3 —
      // which the panel priced it onto — is handed nothing at all.
      const before = placeSlots(four.model.layers, 13, [4, 4, 5, 0], 4);
      expect(before.outputDevice).toBe(2);

      // Two cards, one resident layer between them, and the table on the empty one.
      const two = input(
        DEEPSEEK_V3,
        getQuant('bf16'),
        LLAMA_CPP,
        RTX_4090,
        2,
        usage({ contextTokens: 32768 })
      );
      expect(two.placement.assignment.shares.map((s) => [s.layers, s.residentLayers])).toEqual([
        [31, 1],
        [30, 0],
      ]);

      const servedTwo = text(commands(two)['llama-server'].serve);
      expect(servedTwo).toContain('-ngl 2');
      expect(servedTwo).toContain('-ts 1,1');

      const gotTwo = place(two, servedTwo);
      expect(gotTwo.perDevice).toEqual([1, 0]);
      expect(gotTwo.outputDevice).toBe(1);

      // Before: `-ts 2,0` put both the layer and the table on card 0 and left card 1 idle.
      const beforeTwo = placeSlots(two.model.layers, 2, [2, 0], 2);
      expect(beforeTwo.outputDevice).toBe(0);
    });

    /** The per-card layer counts a given pair produces, for asserting against a *rejected* pair. */
    function placeDelivered(i: LaunchInput, ngl: number, ts: readonly number[]): number[] {
      const { layerDevice } = placeSlots(i.model.layers, ngl, ts, i.rig.count);
      return Array.from(
        { length: i.rig.count },
        (_, d) => layerDevice.filter((x) => x === d).length
      );
    }

    it('puts the output on the seeded bin — the last share — whatever its layer count', () => {
      /**
       * **A zero share is not a card to skip, and reading it as one is what #204 got wrong.**
       *
       * `planPlacement` seeds the *last* bin with the output projection, so that bin is the one
       * priced for the table however few layers it keeps — and it is the first bin to reach zero
       * of them, because its overflow is clamped against a `weightBytes` carrying the table while
       * the layer count divides a `layerWeightBytes` that is not. Giving the extra slot to the last
       * *non-zero* share hands the table to a card that was never charged for it and leaves the
       * card that was idle.
       *
       * Both shapes below take the slot on the last entry, which is the rule `tensorSplit` now
       * emits, and llama.cpp's own `upper_bound` puts the table where the packing priced it: a
       * ratio of `c + 1` is at least 1 whatever `c` is, so the seeded bin's cumulative boundary is
       * the only one that clears the final slot's key.
       */
      const slot = (sized: readonly number[]) =>
        sized.map((c, i) => (i === sized.length - 1 ? c + 1 : c));

      // A packing that spilled the front cards entirely: the leading zeros hold nothing and the
      // last card holds both its layers and the table.
      const leading = placeSlots(26, 13, slot([0, 0, 6, 6]), 4);
      expect(
        Array.from({ length: 4 }, (_, d) => leading.layerDevice.filter((x) => x === d).length)
      ).toEqual([0, 0, 6, 6]);
      expect(leading.outputDevice).toBe(3);

      // The mirror, and the case this rule exists for: the seeded bin spilled to no layers at all,
      // and still gets the table rather than passing it to card 1.
      const trailing = placeSlots(26, 13, slot([6, 6, 0, 0]), 4);
      expect(
        Array.from({ length: 4 }, (_, d) => trailing.layerDevice.filter((x) => x === d).length)
      ).toEqual([6, 6, 0, 0]);
      expect(trailing.outputDevice, 'the table went to a card that was never sized for it').toBe(3);

      // Both triage examples from #209, worked through the same port. The first is the shape a
      // 2-card rig reaches; the second interleaves a zero the emitter must not treat specially.
      const two = placeSlots(61, 2, slot([1, 0]), 2);
      expect(
        Array.from({ length: 2 }, (_, d) => two.layerDevice.filter((x) => x === d).length)
      ).toEqual([1, 0]);
      expect(two.outputDevice).toBe(1);

      const four = placeSlots(12, 3, slot([1, 1, 0, 0]), 4);
      expect(
        Array.from({ length: 4 }, (_, d) => four.layerDevice.filter((x) => x === d).length)
      ).toEqual([1, 1, 0, 0]);
      expect(four.outputDevice).toBe(3);
    });

    /**
     * **The sweep, which is the assertion that would have caught both defects.**
     *
     * Every command over a cross-section of the shipped catalog, run through the reference. Two
     * claims, and the second is the one no string test can make: the repeating layers that end up
     * on a GPU are the count the placement sized, and where `-ts` is emitted, each card's share is
     * the one it was packed with and the output tensor is on the card that was given the extra slot.
     *
     * The coverage counts are asserted rather than hoped for. Without them a sweep that reached
     * neither a spilling placement nor a `-ts` command would pass while proving nothing, which is
     * how this file's own `-ts` defect survived — there was a spilled test and a sharded test and
     * never their product.
     *
     * **The output-tensor assertion is derived from the placement, and its first version was
     * derived from the emitter** (#209). It recomputed "the last non-zero share" out of the same
     * `sized` list `tensorSplit` reads and compared the emitter against that, which is not a test —
     * it passes by construction for *any* self-consistent rule, and 1,055 commands carrying the
     * table on a card the engine never charged for it swept straight through. What the flag has to
     * agree with is `planPlacement`: a bin's weight is its layers plus whatever fixed block was
     * seeded onto it, so the bin whose excess is `outputBytes` is the one the panel priced the
     * table onto, and that is read off `shares` here rather than restated.
     */
    it('holds across the shipped catalog, spilled and sharded', () => {
      const quants = [getQuant('q4_k_m'), getQuant('q8_0'), getQuant('bf16')];
      let spilling = 0;
      let sharded = 0;
      let resident = 0;
      let tableOnAnEmptyCard = 0;

      for (const model of MODELS) {
        for (const quant of quants) {
          for (const count of [1, 2, 4, 5, 8]) {
            for (const contextTokens of [8192, 32768, 131072]) {
              const scenario = input(
                model,
                quant,
                LLAMA_CPP,
                RTX_5090,
                count,
                usage({ contextTokens, concurrency: 4 })
              );
              const emitted = commands(scenario)['llama-server'].serve;
              if (!emitted.ok) continue;

              const where = `${model.id} ${quant.id} x${count} @${contextTokens}`;
              const sizedTotal = Math.min(
                scenario.placement.assignment.residentLayers,
                model.layers
              );
              const got = place(scenario, emitted.text);

              // `-ngl 0` is the one case #204 left open: it declines the placement rather than
              // expressing it, so there is nothing here to check it against. See `gpuLayers`.
              if (got.ngl === 0) continue;

              if (sizedTotal < model.layers) spilling++;
              else resident++;
              expect(got.onGpu, `layers on a GPU: ${where}`).toBe(sizedTotal);

              if (got.ts === undefined) continue;
              sharded++;
              const shares = scenario.placement.assignment.shares;
              const sized = shares.flatMap(
                (s) => Array(s.deviceCount).fill(s.residentLayers) as number[]
              );
              expect(got.perDevice, `per-card split: ${where}`).toEqual(sized);
              /**
               * The extra slot went to the card the **engine** charged the output projection to.
               *
               * A bin's `weightBytes` is its layers plus whatever fixed block was seeded onto it,
               * so the bin whose excess is `outputBytes` is the one the panel priced the table
               * onto — nothing here reads the rule `tensorSplit` applies. Bin 0's own seed comes
               * off first: a vision tower lands there (`clip.cpp` takes the first GPU, never
               * `tensor_split`) and would otherwise be a second candidate. The input embedding is
               * on neither, because this rig is a discrete GPU under llama.cpp.
               *
               * One match is required rather than the first one taken, so a shape that made the
               * derivation ambiguous fails here instead of quietly picking a side.
               */
              const { layerBytes, outputBytes, towerBytes } = weightBreakdown(model, quant);
              const perLayer = layerBytes / model.layers;
              const excess = shares.map(
                (s, at) => s.weightBytes - s.layers * perLayer - (at === 0 ? towerBytes : 0)
              );
              const holdsTable = excess.flatMap((e, at) =>
                Math.abs(e - outputBytes) < 1024 ? [at] : []
              );
              expect(holdsTable, `bins charged the output table: ${where}`).toHaveLength(1);
              expect(got.outputDevice, `output tensor: ${where}`).toBe(holdsTable[0]);
              if (sized[holdsTable[0]] === 0) tableOnAnEmptyCard++;
              // The pair agrees with itself: `-ts` proportions the window, so it sums to `-ngl`.
              expect(
                got.ts.split(',').reduce((a, b) => a + Number(b), 0),
                `-ts sums to -ngl: ${where}`
              ).toBe(got.ngl);
            }
          }
        }
      }

      // The premises. Each of these was zero in some earlier draft of this sweep.
      expect(resident, 'no fully-resident placement swept').toBeGreaterThan(400);
      expect(spilling, 'no spilling placement swept — this is defect 1').toBeGreaterThan(400);
      expect(sharded, 'no -ts emitted — this is defect 2').toBeGreaterThan(200);
      // And the premise #209 needed: a seeded bin that spilled to no layers still holds the table,
      // which is the state every earlier draft of this sweep asserted its way straight past.
      expect(
        tableOnAnEmptyCard,
        'no card held the table without a layer — this is #209 unswept'
      ).toBeGreaterThan(0);
    });
  });

  it('maps the KV precision to the cache type flags rather than the label', () => {
    const q8 = input(
      LLAMA_31_8B,
      getQuant('q4_k_m'),
      LLAMA_CPP,
      RTX_5090,
      1,
      usage({ kvPrecision: 'q8' })
    );
    expect(text(commands(q8)['llama-server'].serve)).toContain('-ctk q8_0 -ctv q8_0');

    const q4 = input(
      LLAMA_31_8B,
      getQuant('q4_k_m'),
      LLAMA_CPP,
      RTX_5090,
      1,
      usage({ kvPrecision: 'q4' })
    );
    expect(text(commands(q4)['llama-server'].serve)).toContain('-ctk q4_0 -ctv q4_0');
  });

  describe('the measurement form is a different binary, and it prices the same workload', () => {
    it('carries this scenario’s own prompt, and decodes at the top of its window', () => {
      const scenario = usage({ contextTokens: 8192, promptTokens: 6000 });
      const measured = input(LLAMA_31_8B, getQuant('q4_k_m'), LLAMA_CPP, RTX_5090, 1, scenario);
      const command = text(commands(measured)['llama-bench'].measure);

      // Two invocations: `-p` and `-n` are separate tests, and the generation one does not inherit
      // the prompt as cache depth — so a single command measures decoding from an empty cache.
      expect(command).toContain('-p 6000');
      expect(command).toContain('-n 0');
      expect(command).toContain('-p 0');
      /**
       * The decode run is `-d 8064 -n 128` and **not** `-d 6000 -n 2192`, which is what #180 was
       * filed against: the depth is the window less what the run needs to answer, because
       * `estimateDecode` charges every step at `contextTokens` rather than at the cache generation
       * happens to start from. The length is llama-bench's own 128 — deliberately its default here,
       * since a steady-state rate does not sharpen with a longer sample and the remainder of an 8K
       * window is two thousand tokens of wall clock for nothing.
       */
      expect(command).toContain('-d 8064');
      expect(command).toContain('-n 128');
      expect(command).not.toContain('-d 6000');
      expect(command).not.toContain('-n 2192');
      // llama-bench's default prompt, which is what a reader would otherwise measure prefill at.
      expect(command).not.toContain('-p 512');
    });

    /**
     * The #139 trap, answered by a flag that turned out to exist. `estimatePrefill` charges an
     * agent turn's attention against a resident prefix, so a measurement of a standalone prompt is
     * a measurement of a different workload than the prediction it would be compared against.
     * `llama-bench -d` runs the test at a stated context depth, which is exactly that state.
     *
     * **The prefix is the prefill run's depth and not the decode run's**, which is the half of this
     * #180 corrected. Prefill charges the turn's attention against what is already resident, so the
     * prefix is the state to reproduce; decode charges every step at the whole window, and the
     * prefix is *inside* that window rather than a smaller state to measure at. So the two runs
     * take different depths here for the same reason they take different depths on a standalone
     * prompt — the archetype moves one of them and not the other.
     */
    it('reproduces a resident prefix with -d, so the measured workload is the priced one', () => {
      const agentish = usage({
        contextTokens: 65536,
        promptTokens: 16384,
        cachedPrefixTokens: 47616,
      });
      const measured = input(LLAMA_31_8B, getQuant('q4_k_m'), LLAMA_CPP, RTX_5090, 1, agentish);
      const emitted = commands(measured)['llama-bench'].measure;
      const [prefillRun, decodeRun] = text(emitted).split('\n\n');

      expect(prefillRun).toContain('-d 47616');
      // The window less the 128 tokens the run generates — not the 64,000 resident when generation
      // starts, which is the depth this emitted before #180.
      expect(decodeRun).toContain('-d 65408');
      expect(decodeRun).not.toContain('-d 64000');
      if (!emitted.ok) throw new Error('unreachable');
      expect(emitted.notes.join(' ')).toMatch(/already in the cache/i);
    });

    it('leaves -d off the prefill run of a standalone prompt, and never off the decode one', () => {
      /**
       * The two runs want different depths, which is the whole reason there are two. A standalone
       * prefill is measured against an empty cache — passing `-d 0` would say the same thing more
       * loudly — while decode is *always* measured against a nearly-full one, because
       * `estimateDecode` charges every step at the whole window whatever the prompt and the
       * archetype's prefix happen to be.
       */
      const [prefillRun, decodeRun] = text(commands(i)['llama-bench'].measure).split('\n\n');

      expect(prefillRun).not.toContain('-d ');
      expect(decodeRun ?? '').toMatch(/-d \d+/);
    });

    it('admits that it cannot reproduce concurrency', () => {
      const many = input(
        LLAMA_31_8B,
        getQuant('q4_k_m'),
        LLAMA_CPP,
        RTX_5090,
        1,
        usage({ concurrency: 16 })
      );
      const emitted = commands(many)['llama-bench'].measure;
      if (!emitted.ok) throw new Error('unreachable');

      // A server up is not a measurement, and a measurement of the wrong workload is worse than
      // none — so the one thing llama-bench cannot do has to be said rather than left implied.
      expect(emitted.notes.join(' ')).toMatch(/no concurrency flag/i);
    });

    it('does not offer the server as a measurement, or the client as a server', () => {
      expect(reason(commands(i)['llama-server'].measure)).toMatch(/does not measure/i);
      expect(reason(commands(i)['llama-bench'].serve)).toMatch(/does not serve/i);
    });
  });

  describe('Ollama', () => {
    it('says what its Modelfile cannot be told, rather than inventing a parameter', () => {
      // Verified against the current Modelfile documentation: num_ctx and num_predict are on the
      // list, num_gpu is not. Which makes the layer split — the whole reason this feature exists —
      // the one thing this surface cannot take.
      const emitted = commands(i).ollama.serve;
      expect(text(emitted)).toContain('PARAMETER num_ctx 8192');
      expect(text(emitted)).not.toContain('num_gpu');
      if (!emitted.ok) throw new Error('unreachable');
      expect(emitted.notes.join(' ')).toMatch(/no parameter for the GPU layer count/i);
    });

    it('points its measurement at llama-bench rather than shipping a client it does not have', () => {
      expect(reason(commands(i).ollama.measure)).toMatch(/no benchmark client/i);
    });
  });
});

describe('vLLM', () => {
  const native = input(DEEPSEEK_V3, getQuant('fp8'), VLLM, RTX_5090, 8);

  it('serves the source repo when the selection is the checkpoint the repo ships', () => {
    const served = text(commands(native).vllm.serve);
    expect(served).toContain(`vllm serve ${DEEPSEEK_V3.id}`);
    expect(served).toContain('--max-model-len 8192');
    expect(served).toContain('--tensor-parallel-size 8');
  });

  it('refuses a format the catalog has no artifact for, and says what would have to exist', () => {
    // The trap #136 names outright: an AWQ selection under vLLM must not fall back to the source
    // checkpoint, because that is a command for a different model than the one priced.
    const awq = input(LLAMA_31_8B, getQuant('awq_4bit'), VLLM);
    const refused = reason(commands(awq).vllm.serve);

    expect(refused).toMatch(/no AWQ 4-bit checkpoint to name/i);
    expect(refused).toMatch(/would start a different model/i);
    expect(refused).toContain(LLAMA_31_8B.id);
  });

  it('refuses the measurement form for the same reason, not just the serving one', () => {
    const awq = input(LLAMA_31_8B, getQuant('awq_4bit'), VLLM);
    expect(reason(commands(awq).vllm.measure)).toMatch(/no AWQ 4-bit checkpoint/i);
  });

  it('states the memory fraction the figures were budgeted at rather than taking vLLM’s default', () => {
    // `preallocFraction` is 0.9 and vLLM's own default has moved to 0.92, so leaving it off would
    // run a slightly roomier machine than the panel priced.
    const emitted = commands(native).vllm.serve;
    expect(text(emitted)).toContain('--gpu-memory-utilization 0.9');
    if (!emitted.ok) throw new Error('unreachable');
    expect(emitted.notes.join(' ')).toMatch(/0\.92/);
  });

  it('maps an FP8 cache to the dtype flag and fp16 to auto', () => {
    const q8 = input(DEEPSEEK_V3, getQuant('fp8'), VLLM, RTX_5090, 8, usage({ kvPrecision: 'q8' }));
    expect(text(commands(q8).vllm.serve)).toContain('--kv-cache-dtype fp8');
    expect(text(commands(native).vllm.serve)).toContain('--kv-cache-dtype auto');
  });

  it('measures with vLLM’s own client at this scenario’s lengths', () => {
    const measured = text(commands(native).vllm.measure);
    expect(measured).toContain('vllm bench latency');
    expect(measured).toContain(`--model ${DEEPSEEK_V3.id}`);
    expect(measured).toMatch(/--input-len \d+/);
    expect(measured).toMatch(/--output-len \d+/);
  });
});

describe('MLX', () => {
  it('refuses every stand-in format loudly, because the checkpoint does not exist', () => {
    // #18 in copy-pasteable form. MLX's figures at Q4_K_M derive from a format MLX cannot read, so
    // a command naming it would send the reader after a file nobody published.
    const substituted = input(
      LLAMA_31_8B,
      getQuant('q4_k_m'),
      MLX,
      MAC_STUDIO_M3_ULTRA_256,
      1,
      usage()
    );
    const refused = reason(commands(substituted).mlx.serve);

    expect(refused).toMatch(/MLX does not load Q4_K_M/i);
    expect(refused).toMatch(/a file that does not exist/i);
    expect(refused).toMatch(/mlx_lm\.convert/);
    expect(refused).not.toContain(LLAMA_31_8B.id);
  });

  it('serves BF16, which is the one format the catalog and MLX agree on', () => {
    const bf16 = input(LLAMA_31_8B, getQuant('bf16'), MLX, MAC_STUDIO_M3_ULTRA_256);
    const served = text(commands(bf16).mlx.serve);
    expect(served.startsWith('mlx_lm.server')).toBe(true);
    expect(served).toContain(`--model ${LLAMA_31_8B.id}`);
  });

  it('refuses to serve at a cache precision the server cannot be told', () => {
    /**
     * A refusal rather than a warning, and the change came from review (#164, P1). `--kv-bits` is
     * on `mlx_lm.generate` and not on the server — checked against its argument list — so the
     * served command necessarily runs an fp16 cache. A long-context configuration that fits
     * *because* 8 bits halves the cache will OOM at fp16, and a note beside a copy button does not
     * stop that: the command is not a command for the placement the panel priced.
     */
    const q8 = input(
      LLAMA_31_8B,
      getQuant('bf16'),
      MLX,
      MAC_STUDIO_M3_ULTRA_256,
      1,
      usage({ kvPrecision: 'q8' })
    );

    expect(reason(commands(q8).mlx.serve)).toMatch(/no flag for it/i);
    // And the measurement form survives, because the client does take the precision — with the
    // threshold set to zero, or the CLI's own default of 5,000 would benchmark an fp16 cache.
    const measured = text(commands(q8).mlx.measure);
    expect(measured).toContain('--kv-bits 8');
    expect(measured).toContain('--quantized-kv-start 0');
  });
});

describe('a placement the engine refused produces no commands at all', () => {
  it('passes the engine’s own sentence through for an unsupported pairing', () => {
    // vLLM does not run on a Mac, and the refusal a reader needs is the one the engine already
    // wrote — not a second spelling of it.
    const wrong = input(LLAMA_31_8B, getQuant('bf16'), VLLM, MAC_STUDIO_M3_ULTRA_256);
    expect(wrong.placement.unsupported).toBeDefined();

    for (const c of launchCommands(wrong)) {
      expect(reason(c.serve)).toBe(wrong.placement.unsupported);
      expect(reason(c.measure)).toBe(wrong.placement.unsupported);
    }
  });

  it('refuses an impossible placement, where no flag rescues the arithmetic', () => {
    // Cache and activations alone over the ceiling: offload cannot move them, so a command would
    // OOM on load however it were flagged.
    const over = input(
      DEEPSEEK_V3,
      getQuant('q4_k_m'),
      LLAMA_CPP,
      RTX_4090,
      1,
      usage({ contextTokens: 131072, concurrency: 64 })
    );
    expect(over.placement.impossible).toBe(true);

    for (const c of launchCommands(over)) {
      expect(reason(c.serve)).toMatch(/does not run/i);
    }
  });

  it('emits nothing for a runtime with no launcher registered', () => {
    const unknown: RuntimeSpec = { ...LLAMA_CPP, id: 'not-a-runtime' };
    expect(launchCommands(input(LLAMA_31_8B, getQuant('q4_k_m'), unknown))).toEqual([]);
  });
});

describe('what review found, kept as tests', () => {
  it('refuses the measurement form when the prompt leaves no room to answer', () => {
    // The prompt slider goes up to the whole context, and the first version floored generation at
    // one token — emitting `--input-len <ctx> --output-len 1 --max-model-len <ctx>`, a command that
    // exceeds its own stated limit. A scenario with no room to answer has nothing to measure.
    const full = input(
      LLAMA_31_8B,
      getQuant('q4_k_m'),
      LLAMA_CPP,
      RTX_5090,
      1,
      usage({ contextTokens: 8192, promptTokens: 8192 })
    );

    expect(reason(commands(full)['llama-bench'].measure)).toMatch(/no room left to generate/i);
  });

  it('counts the resident prefix as occupying the window too', () => {
    /**
     * 47,616 of prefix under a 16,384 prompt in a 65,536 window leaves 1,536 — not the 49,152 a
     * prompt-only subtraction produced.
     *
     * Asserted on vLLM because that is where the figure is still a flag. `llama-bench` asks for a
     * short generation at the top of the window since #180, so the room the scenario leaves only
     * *gates* its measurement form; `vllm bench latency` still runs the scenario's own answer end
     * to end, and `--output-len` is what a prompt-only subtraction would overrun.
     */
    const agentish = input(
      LLAMA_31_8B,
      getQuant('bf16'),
      VLLM,
      RTX_5090,
      1,
      usage({ contextTokens: 65536, promptTokens: 16384, cachedPrefixTokens: 47616 })
    );

    expect(text(commands(agentish).vllm.measure)).toContain('--output-len 1536');
  });

  it('names the right cause when weights alone are over a rig that cannot spill', () => {
    /**
     * `impossible` covers two failures and the first draft named only one. On unified memory
     * anything over budget is impossible, including an oversized checkpoint whose *cache* is
     * nowhere near the ceiling — and telling that reader to lower the context is advice that will
     * not work at any context.
     */
    const oversized = input(
      GPT_OSS_120B,
      getQuant('bf16'),
      MLX,
      MAC_STUDIO_M3_ULTRA_256,
      1,
      usage({ contextTokens: 4096 })
    );

    expect(oversized.placement.impossible).toBe(true);
    expect(oversized.placement.floorBytesPerDevice).toBeLessThan(
      oversized.placement.allocatableBytesPerDevice
    );
    const refused = reason(commands(oversized).mlx.serve);
    expect(refused).toMatch(/weights are over the ceiling/i);
    expect(refused).toMatch(/Lowering the context will not help/i);
    expect(refused).not.toMatch(/cache and activations alone/i);
  });

  it('still blames the cache when the cache really is what is over', () => {
    // The other arm, so the split above is a split rather than a rewording.
    const cacheBound = input(
      DEEPSEEK_V3,
      getQuant('q4_k_m'),
      LLAMA_CPP,
      RTX_4090,
      1,
      usage({ contextTokens: 131072, concurrency: 64 })
    );

    expect(cacheBound.placement.floorBytesPerDevice).toBeGreaterThan(
      cacheBound.placement.allocatableBytesPerDevice
    );
    expect(reason(commands(cacheBound)['llama-server'].serve)).toMatch(/cache and activations/i);
  });

  it('never writes to a bare Modelfile, which is the reader’s own', () => {
    // `cat >` truncates unconditionally, and the directory an Ollama user runs this from is exactly
    // the one likely to already hold a Modelfile of their own.
    const emitted = commands(input(LLAMA_31_8B, getQuant('q4_k_m'), LLAMA_CPP)).ollama.serve;
    const written = text(emitted);

    expect(written).not.toMatch(/cat > Modelfile/);
    expect(written).toMatch(/set -C/);
    expect(written).toMatch(/headroom-[a-z0-9-]+\.Modelfile/);
  });

  it('pins the commit the catalog derived the model from, where the runtime can', () => {
    // Naming only the repo id resolves the mutable default branch, so after an upstream push the
    // command loads a checkpoint the displayed figures do not describe.
    const withRevision = { ...DEEPSEEK_V3, revision: 'abc1234567deadbeef' };
    const pinned = input(withRevision, getQuant('fp8'), VLLM, RTX_5090, 8);

    expect(text(commands(pinned).vllm.serve)).toContain('--revision abc1234567deadbeef');
    expect(text(commands(pinned).vllm.measure)).toContain('--revision abc1234567deadbeef');
  });

  it('says so rather than pinning, on a runtime with no revision flag', () => {
    // mlx_lm.server takes no revision, so the honest form is to name the commit the figures came
    // from instead of emitting a flag that does not parse.
    const withRevision = { ...LLAMA_31_8B, revision: 'abc1234567deadbeef' };
    const emitted = commands(input(withRevision, getQuant('bf16'), MLX, MAC_STUDIO_M3_ULTRA_256))
      .mlx.serve;

    expect(text(emitted)).not.toContain('--revision');
    if (!emitted.ok) throw new Error('unreachable');
    expect(emitted.notes.join(' ')).toMatch(/takes no revision flag/i);
    expect(emitted.notes.join(' ')).toMatch(/abc1234567/);
  });

  it('offloads to the CPU when the weights do not fit, or the command OOMs', () => {
    /**
     * `planPlacement` reports a *runnable* placement whenever only the weights are over — so both
     * commands were emitted for a configuration vLLM cannot start, since it defaults
     * `--cpu-offload-gb` to zero and tries to keep every weight on the cards.
     */
    const spilled = input(DEEPSEEK_V3, getQuant('fp8'), VLLM, RTX_5090, 2);
    expect(spilled.placement.offloadFraction).toBeGreaterThan(0);
    expect(spilled.placement.impossible).toBe(false);

    const served = text(commands(spilled).vllm.serve);
    const flag = /--cpu-offload-gb (\d+)/.exec(served);
    expect(flag, 'no offload flag on a spilled placement').not.toBeNull();

    // Per GPU, which is how vLLM documents it, and rounded up: too little offload is an OOM on
    // load and too much is only slower.
    // `deviceWeightBytes`, which is what `offloadFraction` is a fraction of since #182 — the same
    // number as the file on any vLLM placement, and the right operand rather than a lucky one.
    const perGpu =
      (spilled.placement.offloadFraction * spilled.placement.deviceWeightBytes) / 2 / 1024 ** 3;
    expect(Number(flag![1])).toBe(Math.ceil(perGpu));
    expect(spilled.placement.deviceWeightBytes).toBe(spilled.placement.totalWeightBytes);
    // The measurement form needs it too, or it starts a machine the serving form does not.
    expect(text(commands(spilled).vllm.measure)).toContain(`--cpu-offload-gb ${flag![1]}`);
  });

  it('leaves the offload flag off a placement that fits', () => {
    const resident = input(LLAMA_31_8B, getQuant('bf16'), VLLM, RTX_5090, 1);
    expect(resident.placement.offloadFraction).toBe(0);
    expect(text(commands(resident).vllm.serve)).not.toContain('--cpu-offload-gb');
  });

  it('states the Ollama daemon’s cache precision as a requirement rather than commanding it', () => {
    /**
     * `OLLAMA_KV_CACHE_TYPE` is read at server startup and defaults to f16, so a long-context
     * configuration that fits only because 8 bits halves the cache would OOM against a Modelfile
     * that cannot say otherwise. The first version therefore emitted
     * `OLLAMA_KV_CACHE_TYPE=q8_0 ollama serve &` — which does not wait for the daemon and does not
     * notice an existing one, so the block carried on against a server still on f16 (#171). The
     * honest fix for that is a readiness poll and a running-daemon check, and this block's value is
     * that it is one readable invocation. So the requirement is stated, and the block does not
     * manage a daemon at all.
     */
    const q8 = input(
      LLAMA_31_8B,
      getQuant('q4_k_m'),
      LLAMA_CPP,
      RTX_5090,
      1,
      usage({ kvPrecision: 'q8' })
    );
    const emitted = commands(q8).ollama.serve;
    if (!emitted.ok) throw new Error('unreachable');

    const note = emitted.notes.join(' ');
    expect(note).toContain('OLLAMA_KV_CACHE_TYPE=q8_0');
    expect(note).toMatch(/daemon setting rather than a Modelfile parameter/i);
    expect(note).toMatch(/server startup/i);
    expect(note).toMatch(/already-running daemon will not pick it up/i);
    expect(note).toMatch(/defaulting to f16/i);
    // The reader has to know the figures describe *that* daemon, or the note is trivia.
    expect(note).toMatch(/figures above are sized for a q8_0 cache/i);

    // And the block itself neither starts a server nor sets the variable — the two halves that
    // would make it a lifecycle script rather than a command.
    expect(emitted.text).not.toContain('ollama serve');
    expect(emitted.text).not.toContain('OLLAMA_KV_CACHE_TYPE');

    // Nothing at all on the default precision, where the daemon already does the right thing —
    // neither in the block nor as a note about a variable this scenario does not need.
    const f16 = commands(input(LLAMA_31_8B, getQuant('q4_k_m'), LLAMA_CPP)).ollama.serve;
    if (!f16.ok) throw new Error('unreachable');
    expect(f16.text).not.toContain('ollama serve');
    expect(f16.text).not.toContain('OLLAMA_KV_CACHE_TYPE');
    expect(f16.notes.join(' ')).not.toContain('OLLAMA_KV_CACHE_TYPE');
  });

  it('refuses the Ollama form when the scenario prices more than one user', () => {
    /**
     * `planPlacement` charges KV and activations for every simultaneous user, and the Modelfile
     * carries only `num_ctx`. Ollama's parallelism is `OLLAMA_NUM_PARALLEL`, read by the daemon at
     * startup — so a Modelfile this surface can write sizes memory for one user against a panel
     * that priced eight (#171). Same polarity as the MLX cache-precision refusal: a command that
     * cannot reproduce the placement is not a command.
     */
    const many = input(
      LLAMA_31_8B,
      getQuant('q4_k_m'),
      LLAMA_CPP,
      RTX_5090,
      1,
      usage({ concurrency: 8 })
    );

    const refused = reason(commands(many).ollama.serve);
    expect(refused).toContain('OLLAMA_NUM_PARALLEL');
    // And it points at the launcher that does take the count on the one command, rather than
    // leaving the reader with a surface and no way to use it.
    expect(refused).toContain('llama-server');
    expect(refused).toContain('-np 8');

    // The sibling launcher still emits, which is what makes the refusal navigation rather than a
    // dead end: the same scenario has a command, on the surface that can express it.
    expect(text(commands(many)['llama-server'].serve)).toContain('-np 8');

    // The measurement half is untouched by the user count — it refuses for its own reason.
    expect(reason(commands(many).ollama.measure)).toMatch(/no benchmark client/i);
  });

  it('still emits the Ollama block for a single user, quantized cache and all', () => {
    // The refusal is keyed on concurrency alone. A quantized cache at one user is the case the
    // note exists for, and it must not have been swept up by the parallelism refusal.
    const one = input(
      LLAMA_31_8B,
      getQuant('q4_k_m'),
      LLAMA_CPP,
      RTX_5090,
      1,
      usage({ concurrency: 1, kvPrecision: 'q8' })
    );
    const emitted = commands(one).ollama.serve;
    if (!emitted.ok) throw new Error('unreachable');

    expect(emitted.text).toMatch(/ollama create .+ && ollama run /);
    expect(emitted.notes.join(' ')).toContain('OLLAMA_KV_CACHE_TYPE=q8_0');

    // The quantized-cache branch is the one the `ollama serve &` line was removed from, so the
    // four fixes it sat above are asserted here as well as on the f16 block: the subshell, `set -e`,
    // `set -C` and the Headroom-specific filename.
    const lines = emitted.text.split('\n');
    expect(lines[0]).toBe('(');
    expect(lines[1]).toBe('set -e');
    expect(emitted.text.trimEnd().endsWith(')')).toBe(true);
    expect(emitted.text).toMatch(/\(set -C; cat > headroom-[a-z0-9-]+\.Modelfile\)/);
  });

  it('does not run a stale Modelfile after refusing to overwrite one', () => {
    /**
     * `set -C` makes the redirection fail on the second run for the same model and quantization,
     * and an unchained `ollama create` then builds and runs the previous `num_ctx`.
     *
     * **`&&` alone was not enough**, which was this block's own previous fix: it chains create to
     * run, but the heredoc above them is a separate command and a heredoc cannot be `&&`-chained to
     * what follows it — so the refusal still fell through to a create against the old file. `set -e`
     * aborts the block on any failure, which is the shell's answer to exactly this.
     */
    const written = text(commands(input(LLAMA_31_8B, getQuant('q4_k_m'), LLAMA_CPP)).ollama.serve);
    expect(written).toMatch(/ollama create .+ && ollama run /);
    // In a subshell, so `set -e` guards the block and does not survive into the reader's own
    // interactive shell — where it stays enabled and can close their terminal on a later failure.
    expect(written.split('\n')[0]).toBe('(');
    expect(written.split('\n')[1]).toBe('set -e');
    expect(written.trimEnd().endsWith(')')).toBe(true);
  });

  it('admits MLX cannot reproduce a multi-user measurement either', () => {
    const many = input(
      LLAMA_31_8B,
      getQuant('bf16'),
      MLX,
      MAC_STUDIO_M3_ULTRA_256,
      1,
      usage({ concurrency: 8 })
    );
    const emitted = commands(many).mlx.measure;
    if (!emitted.ok) throw new Error('unreachable');

    expect(emitted.notes.join(' ')).toMatch(/no concurrency option/i);
  });

  it('measures vLLM at the configured user count, not the client’s default of 8', () => {
    const many = input(DEEPSEEK_V3, getQuant('fp8'), VLLM, RTX_5090, 8, usage({ concurrency: 16 }));
    expect(text(commands(many).vllm.measure)).toContain('--batch-size 16');
  });
});

describe('every template says where its flags came from', () => {
  it('carries a source and a date, the way a devices.json row does', () => {
    // "A command is a claim, and flags drift" — the trap #136 names. A template with no provenance
    // is a template nobody can re-check when a runtime renames something.
    for (const runtime of [LLAMA_CPP, VLLM, MLX]) {
      const device = runtime.id === 'mlx' ? MAC_STUDIO_M3_ULTRA_256 : RTX_5090;
      const quant = runtime.id === 'mlx' ? getQuant('bf16') : getQuant('q4_k_m');

      const emitted = launchCommands(input(LLAMA_31_8B, quant, runtime, device));
      expect(emitted.length, runtime.id).toBeGreaterThan(0);

      for (const { launcher } of emitted) {
        expect(launcher.source, launcher.id).toMatch(/^https:\/\//);
        expect(launcher.checkedOn, launcher.id).toMatch(/^\d{4}-\d{2}-\d{2}$/);
        expect(launcher.runtimeId, launcher.id).toBe(runtime.id);
      }
    }
  });
});
