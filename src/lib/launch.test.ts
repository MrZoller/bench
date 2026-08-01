import { describe, expect, it } from 'vitest';
import { artifactFor, launchCommands, type Emission, type LaunchInput } from './launch';
import { planPlacement } from '@/engine/placement';
import {
  DEEPSEEK_V3,
  GEMMA_3_12B,
  GPT_OSS_20B,
  GPT_OSS_120B,
  LLAMA_31_8B,
  LLAMA_CPP,
  MAC_STUDIO_M3_ULTRA_256,
  MLX,
  QWEN3_32B,
  RTX_4090,
  RTX_5090,
  VLLM,
} from '@/engine/fixtures';
import { getQuant } from '@/data/quants';
import { MODELS } from '@/data/catalog';
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
      // leaves the output tensor on the host. Verified against llama-model.cpp, not recalled.
      const served = text(commands(i)['llama-server'].serve);
      expect(i.placement.offloadFraction).toBe(0);
      expect(served).toContain(`-ngl ${LLAMA_31_8B.layers + 1}`);
    });

    it('drops to the resident count when weights spill, and says it is not a fraction', () => {
      const spilled = input(QWEN3_32B, getQuant('bf16'), LLAMA_CPP, RTX_4090);
      expect(spilled.placement.offloadFraction).toBeGreaterThan(0);

      const emitted = commands(spilled)['llama-server'].serve;
      const ngl = spilled.placement.assignment.residentLayers;
      expect(ngl).toBeLessThan(QWEN3_32B.layers);
      expect(text(emitted)).toContain(`-ngl ${ngl}`);
      if (!emitted.ok) throw new Error('unreachable');
      expect(emitted.notes.join(' ')).toMatch(/not a fraction of the model/i);
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

  describe('-ts carries the packing, and only when the packing is uneven', () => {
    /**
     * **The combination the first version of this file never tested, and got wrong.**
     *
     * `-ts` does not distribute the model — it distributes the `-ngl` window, because llama.cpp
     * puts the last `ngl` layers on GPUs and splits *those* by these proportions. So emitting the
     * *assigned* counts beside a resident `-ngl` hands llama.cpp two numbers from different scopes
     * and lets it re-derive a per-device split that is neither: on a rig packing 7,7,6,6 layers and
     * keeping 2,2,6,6 resident, `-ngl 16 -ts 7,7,6,6` spreads sixteen layers slightly in favour of
     * the two cards bench sized for two — which are the constrained cards precisely because their
     * cache already fills them. That is an OOM on load, from a command.
     *
     * There were a spilled test and a sharded test, and the defect lived only in their product.
     */
    it('proportions the resident layers, not the assigned ones, on a rig that spills', () => {
      /**
       * gpt-oss-120b at Q8_0, 128K over 8 users, four 4090s — chosen because it fails the old
       * implementation *twice*. The packing assigns 9,9,9,9, so the "only when uneven" gate read
       * against assigned counts suppresses `-ts` altogether; and the cards keep 3,3,4,4 of those
       * resident, so the counts that were being suppressed were also the wrong ones. Equal counts
       * of unequal layers is exactly the case the flag exists for.
       */
      const spilled = input(
        GPT_OSS_120B,
        getQuant('q8_0'),
        LLAMA_CPP,
        RTX_4090,
        4,
        usage({ contextTokens: 131072, concurrency: 8 })
      );

      const shares = spilled.placement.assignment.shares;
      const assigned = shares.flatMap((s) => Array(s.deviceCount).fill(s.layers) as number[]);
      const resident = shares.flatMap(
        (s) => Array(s.deviceCount).fill(s.residentLayers) as number[]
      );

      // The premises, asserted rather than assumed. Without all three the assertion below is
      // satisfied by the implementation it was written to reject.
      expect(spilled.placement.offloadFraction, 'nothing spilled').toBeGreaterThan(0);
      expect(new Set(assigned).size, 'the assigned counts differ, so the old gate would fire').toBe(
        1
      );
      expect(new Set(resident).size, 'the resident counts are even too').toBeGreaterThan(1);

      const served = text(commands(spilled)['llama-server'].serve);
      expect(served).toContain(`-ts ${resident.join(',')}`);

      // And the two flags have to agree, which is the property that makes the pair safe: `-ts`
      // proportions the window `-ngl` opens, so the proportions must sum to it.
      expect(resident.reduce((a, b) => a + b, 0)).toBe(spilled.placement.assignment.residentLayers);
      expect(served).toContain(`-ngl ${spilled.placement.assignment.residentLayers}`);
    });

    it('gives the benchmark client the same split as the server', () => {
      // A sharded measurement run at llama.cpp's default even split times a different placement
      // than the serving command reproduces, which makes the number unusable for calibration —
      // the one thing the measurement form exists for.
      const gemma = input(
        GEMMA_3_12B,
        getQuant('q4_k_m'),
        LLAMA_CPP,
        RTX_5090,
        5,
        usage({ contextTokens: 131072, concurrency: 8 })
      );
      const counts = gemma.placement.assignment.shares.map((s) => s.residentLayers);

      expect(new Set(counts).size, 'the split is even, so this proves nothing').toBeGreaterThan(1);
      expect(text(commands(gemma)['llama-bench'].measure)).toContain(`-ts ${counts.join(',')}`);
    });

    it('emits the layer counts a hybrid model actually produced', () => {
      // Gemma at 128K over five cards: the balanced split is wildly uneven in layer counts,
      // because one full-attention layer caches ~128x a sliding one. llama.cpp's default splits by
      // device memory, which on identical cards is even — the wrong answer for this model.
      const gemma = input(
        GEMMA_3_12B,
        getQuant('q4_k_m'),
        LLAMA_CPP,
        RTX_5090,
        5,
        usage({ contextTokens: 131072, concurrency: 8 })
      );
      const served = text(commands(gemma)['llama-server'].serve);
      const counts = gemma.placement.assignment.shares.map((s) => s.layers);

      expect(new Set(counts).size, 'the split is even, so this proves nothing').toBeGreaterThan(1);
      expect(served).toContain(`-ts ${counts.join(',')}`);
    });

    it('stays silent on a single device, where there is nothing to split', () => {
      expect(text(commands(i)['llama-server'].serve)).not.toContain('-ts');
    });

    it('stays silent when the split is even, which is what llama.cpp would do unaided', () => {
      // A dense model divides cleanly, so the flag would say "split this evenly" — which is the
      // default. Emitting it anyway would make the flag noise and hide the case that matters.
      const dense = input(
        LLAMA_31_8B,
        getQuant('q4_k_m'),
        LLAMA_CPP,
        RTX_5090,
        4,
        usage({ contextTokens: 4096 })
      );
      const counts = dense.placement.assignment.shares.map((s) => s.layers);
      expect(new Set(counts).size, 'the split is uneven, so this proves nothing').toBe(1);
      expect(text(commands(dense)['llama-server'].serve)).not.toContain('-ts');
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
    it('carries this scenario’s prompt and generation lengths, not llama-bench’s defaults', () => {
      const scenario = usage({ contextTokens: 8192, promptTokens: 6000 });
      const measured = input(LLAMA_31_8B, getQuant('q4_k_m'), LLAMA_CPP, RTX_5090, 1, scenario);
      const command = text(commands(measured)['llama-bench'].measure);

      expect(command).toContain('-p 6000');
      expect(command).toContain('-n 2192');
      // llama-bench's own defaults, which are what a reader would otherwise measure.
      expect(command).not.toContain('-p 512');
      expect(command).not.toContain('-n 128');
    });

    /**
     * The #139 trap, answered by a flag that turned out to exist. `estimatePrefill` charges an
     * agent turn's attention against a resident prefix, so a measurement of a standalone prompt is
     * a measurement of a different workload than the prediction it would be compared against.
     * `llama-bench -d` runs the test at a stated context depth, which is exactly that state.
     */
    it('reproduces a resident prefix with -d, so the measured workload is the priced one', () => {
      const agentish = usage({
        contextTokens: 65536,
        promptTokens: 16384,
        cachedPrefixTokens: 47616,
      });
      const measured = input(LLAMA_31_8B, getQuant('q4_k_m'), LLAMA_CPP, RTX_5090, 1, agentish);
      const emitted = commands(measured)['llama-bench'].measure;

      expect(text(emitted)).toContain('-d 47616');
      if (!emitted.ok) throw new Error('unreachable');
      expect(emitted.notes.join(' ')).toMatch(/already in the cache/i);
    });

    it('leaves -d off a standalone prompt rather than passing zero', () => {
      expect(text(commands(i)['llama-bench'].measure)).not.toContain('-d ');
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

  it('admits the server cannot be told the cache precision the figures assume', () => {
    // `--kv-bits` is on mlx_lm.generate and not on the server — checked against its argument list.
    // So an 8-bit-cache scenario cannot be *served* at the precision it was priced at.
    const q8 = input(
      LLAMA_31_8B,
      getQuant('bf16'),
      MLX,
      MAC_STUDIO_M3_ULTRA_256,
      1,
      usage({ kvPrecision: 'q8' })
    );
    const served = commands(q8).mlx.serve;
    if (!served.ok) throw new Error('unreachable');

    expect(served.text).not.toContain('--kv-bits');
    expect(served.notes.join(' ')).toMatch(/no KV quantization flag/i);
    // And the client, which does take it, does.
    expect(text(commands(q8).mlx.measure)).toContain('--kv-bits 8');
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
