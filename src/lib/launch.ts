import type { ModelSpec, QuantSpec, Rig, RuntimeSpec, UsageSpec } from '@/engine/types';
import type { Placement } from '@/engine/placement';
import { effectiveDeviceCount, effectivePromptTokens } from '@/engine/placement';
import { hasSlidingLayers } from '@/engine/kv';
import { substitutionFor } from '@/data/runtimes';

/**
 * The runnable command for a placement (#136).
 *
 * Everything here is a formatter over answers the engine has already committed to — the layer
 * assignment, the KV precision, the context, the shard count. What it is *not* is a second
 * derivation of any of them: `-ngl` reads `Placement.assignment.residentLayers`, `--max-model-len`
 * reads the same `contextTokens` the budget bar drew, and where a figure is not on the placement
 * it is not printed. A limit stated twice is a limit that will disagree with itself, and here the
 * disagreement would be copy-pasteable.
 *
 * ## The three refusals, and why a refusal is the feature
 *
 * A command is a claim that something can be run, so this module refuses more often than it emits:
 *
 *   1. **A configuration the engine turned away.** `unsupported` and `impossible` both describe a
 *      placement that does not exist, and a command for one is a command that OOMs on load.
 *   2. **A checkpoint the catalog cannot name.** `ModelSpec.id` is the *source* repo, and
 *      `QuantSpec` carries sizing and compute properties only — there is no GGUF, AWQ or MLX
 *      artifact anywhere in the data. So a Q4_K_M selection under vLLM has no repo to print, and
 *      naming the source checkpoint instead would be a command for a different model than the one
 *      the placement priced. See {@link artifactFor}.
 *   3. **A format the runtime does not actually load.** The #18 substitution: MLX's figures at
 *      Q4_K_M derive from a format MLX cannot read, so a command naming that checkpoint is a
 *      command for a file that does not exist. `substitutionFor` already knows; this refuses on it.
 *
 * ## Every flag was read from upstream, and each template says where
 *
 * "A command is a claim, and flags drift" is the trap #136 names, and every template below carries
 * a `source` and the date it was checked, in the posture `devices.json` takes for a device spec.
 * Four things in here are *not* what a from-memory implementation would have written, and each one
 * was found by reading the source rather than by recalling it:
 *
 *   - **llama.cpp's `-ngl` counts the output tensor.** `n_gpu_layers` is the repeating blocks plus
 *     one position, so a fully-resident 48-layer model wants `-ngl 49` and `-ngl 48` leaves the
 *     output tensor on the CPU.
 *   - **`-c` is the whole cache, divided among the `-np` slots.** `n_ctx_seq` is
 *     `n_ctx / n_seq_max` unless the KV buffer is unified, so eight users at 64K wants `-c 524288`.
 *     Passing the per-user figure would give each slot an eighth of it.
 *   - **Ollama's Modelfile has no `num_gpu` parameter.** The documented list is `num_ctx`,
 *     `num_predict` and the sampler knobs — so the layer split this feature exists to print is the
 *     one thing that surface cannot take, and the template says so rather than inventing a flag.
 *   - **`mlx_lm.server` has no KV quantization flag.** `--kv-bits` is on `mlx_lm.generate`; the
 *     server's argument list does not carry it, so an 8-bit-cache scenario cannot be served at the
 *     precision it was priced at.
 */

/** Where a launcher's flags were read from, and when. */
export interface Launcher {
  id: string;
  /** What the reader actually types, which is not always the runtime's own label. */
  label: string;
  runtimeId: string;
  source: string;
  /** The date the flags above were checked against `source`, ISO. */
  checkedOn: string;
}

/**
 * One command, or the reason there is none.
 *
 * A refusal is a first-class result rather than an absence, because the reason is the useful part:
 * "vLLM cannot be told which checkpoint to load, because the catalog has no AWQ repo for this
 * model" is what a reader needs, and an empty panel is not.
 */
export type Emission =
  | { readonly ok: true; readonly text: string; readonly notes: readonly string[] }
  | { readonly ok: false; readonly reason: string };

/**
 * A launcher's two forms.
 *
 * Both, because a server being up is not a measurement (#139): the serving command starts the
 * thing, and the measurement command is the one that reproduces the *priced workload* — the
 * scenario's prompt length, generation length and resident prefix in the flags of that runtime's
 * own client. `llama-bench` loads GGUF and cannot measure a vLLM or MLX placement, so the client
 * travels with its launcher rather than standing in for the others.
 */
export interface LauncherCommands {
  launcher: Launcher;
  serve: Emission;
  measure: Emission;
}

export interface LaunchInput {
  model: ModelSpec;
  quant: QuantSpec;
  runtime: RuntimeSpec;
  rig: Rig;
  usage: UsageSpec;
  placement: Placement;
}

/**
 * The checkpoint bench can actually name for a model at a format, or nothing.
 *
 * **The catalog knows exactly one artifact per model: its own repo, at its own checkpoint
 * format.** `nativeQuant` is the `quantization_config.quant_method` the generator read from
 * `config.json`, and its absence means the repo ships unquantized — which is the `bf16` row, whose
 * label is "BF16 / FP16" precisely because the catalog does not distinguish the two at 16 bits.
 *
 * Everything else is a *conversion* somebody else published: a Q4_K_M GGUF, an AWQ pack, an
 * `mlx-community` port. Those live in repos this catalog has never seen, and the failure mode of
 * guessing one is not a 404 — it is a plausible-looking command that loads a different model than
 * the placement priced. So the answer is `undefined` and the caller refuses.
 *
 * An unrecognised `nativeQuant` — a `quant_method` string with no matching `QuantSpec` id — makes
 * every format on that model unnameable rather than making the wrong one nameable, which is the
 * direction this whole module fails in.
 */
export function artifactFor(model: ModelSpec, quantId: string): string | undefined {
  return (model.nativeQuant ?? 'bf16') === quantId ? model.id : undefined;
}

/**
 * The commit every figure on the page was derived from.
 *
 * `build-catalog.ts` records `revision` on each row precisely so a suspicious number is
 * reproducible, and a command naming only the repo id resolves the *mutable default branch* — so
 * after an upstream push the copied command loads a checkpoint the displayed memory and speed
 * figures do not describe (raised by Codex on #164). vLLM takes `--revision`, which accepts a
 * commit id; MLX's server does not, and that is stated in its notes rather than papered over.
 */
export function revisionOf(model: ModelSpec): string | undefined {
  return model.revision;
}

/**
 * A scenario whose window has no room left to answer in.
 *
 * The measurement form refuses rather than manufacturing a token: the prompt slider goes up to the
 * whole context, and a `--output-len 1` beside `--max-model-len <context>` is a command that
 * exceeds its own stated limit.
 */
function noRoomToAnswer(contextTokens: number): string {
  return (
    `The prompt fills the whole ${contextTokens.toLocaleString('en-US')}-token window, so there is ` +
    `no room left to generate into and nothing to measure. Lower the prompt length above and the ` +
    `benchmark command comes back.`
  );
}

/** The sentence a refused artifact gets, naming what would have to exist. */
function noArtifact(model: ModelSpec, quant: QuantSpec, launcher: string): string {
  const native = model.nativeQuant ?? 'bf16';
  return (
    `bench has no ${quant.label} checkpoint to name for ${model.name}. The catalog carries the ` +
    `source repo — ${model.id}, which ships at ${native.toUpperCase()} — and no per-format ` +
    `artifact, so ${launcher} would have to be pointed at a conversion published elsewhere. ` +
    `Naming the source repo here would start a different model than the one priced above.`
  );
}

/** llama.cpp's `--cache-type-k`/`-v` names for the precisions the catalog offers. */
const LLAMA_KV_TYPE = { fp16: 'f16', q8: 'q8_0', q4: 'q4_0' } as const;
/** vLLM's `--kv-cache-dtype` values. `auto` is "the model's own dtype", which is what fp16 means. */
const VLLM_KV_DTYPE = { fp16: 'auto', q8: 'fp8', q4: undefined } as const;

/**
 * `-ngl`, which is a layer count and not a fraction.
 *
 * Two corrections sit between `assignment.residentLayers` and the flag, and the engine declines
 * both because it has no notion of a flag:
 *
 *   - **A CPU-only rig offloads nothing.** `residentLayers` reports every layer, truthfully —
 *     nothing spilled, because there was nowhere to spill *from*. The honest flag is 0.
 *   - **llama.cpp counts the output tensor one position past the repeating blocks.**
 *     `n_gpu_layers` defaults to `n_layer_all + 1`, and `i_gpu_start = max(n_layer_all + 1 - ngl, 0)`
 *     — so `-ngl 48` on a 48-layer model leaves the output tensor on the host. "All of them" is
 *     `layers + 1`.
 */
function gpuLayers(input: LaunchInput): number {
  if (input.rig.device.class === 'cpu-ram') return 0;
  const { residentLayers } = input.placement.assignment;
  return residentLayers >= input.model.layers ? input.model.layers + 1 : residentLayers;
}

/**
 * `-ts`, when the packing actually produced an uneven split.
 *
 * This is the one flag that exists *because* the assignment was surfaced. llama.cpp's default split
 * is proportional to each device's memory, which on a homogeneous rig is an equal number of layers —
 * and on a hybrid model that is the wrong split, because one full-attention layer caches up to
 * ~128x what a sliding one does. `layerSplitBins` balances the combined load instead, and on Gemma
 * 3 12B at 128K over five cards its layer counts land 19 apart.
 *
 * **The counts are the *resident* ones, not the assigned ones, and the first version of this
 * emitted the assigned ones** — which is the trap #136 names, reached from the far side. `-ts` does
 * not distribute the model; it distributes the `-ngl` window, because llama.cpp puts the last `ngl`
 * layers on GPUs and splits *those* by these proportions. So `-ngl` and `-ts` are read together,
 * and giving them counts from two different scopes lets llama.cpp re-derive a per-device split that
 * is neither.
 *
 * It is reachable and it OOMs. Ministral 3 3B at Q8_0, 131,072 tokens over 4 users on four RTX
 * 5080s packs 7,7,6,6 layers and keeps 2,2,6,6 of them resident — so `-ngl 16 -ts 7,7,6,6` asks
 * llama.cpp to spread sixteen layers slightly-in-favour-of the two cards bench sized for two, which
 * are the constrained cards precisely because their cache already fills them. `-ts 2,2,6,6` is the
 * split that was actually sized.
 *
 * Where nothing spills the two are identical, so this changes only the case it exists for.
 *
 * Emitted only when the counts really differ, so the flag never appears saying "split this evenly",
 * which is what llama.cpp would have done unaided — and that gate reads the resident counts too,
 * since a packing that assigns equal counts of *unequal* layers is exactly where the flag is needed.
 *
 * **And never on a hybrid model, which is the case the flag looked most useful for** (raised by
 * Codex on #164, P1). `-ts` partitions llama.cpp's ordered `-ngl` suffix into contiguous device
 * ranges; `layerSplitBins` assigns *individual* layers by greedy combined load, so a share can be a
 * non-contiguous mixture of full-attention and sliding layers. On a model where those cache ~128x
 * differently, equal counts do not reproduce equal loads — llama.cpp would hand a card a different
 * set of expensive layers than `planPlacement` priced, and the fit the panel reported would not be
 * the fit the command produces. A count is a faithful description of the packing only where the
 * layers are interchangeable, so that is the only place it is emitted.
 *
 * What remains is still worth having: a uniform model over a device count that does not divide it
 * (48 layers over 5 cards is 10,10,10,9,9) is both uneven and exactly expressible, and llama.cpp's
 * memory-proportional default gets it wrong.
 */
function tensorSplit(input: LaunchInput): string | undefined {
  const { parallelism, shares } = input.placement.assignment;
  if (parallelism !== 'layer' || effectiveDeviceCount(input.rig) <= 1) return undefined;
  if (hasSlidingLayers(input.model)) return undefined;

  const counts = shares.flatMap((s) =>
    Array.from({ length: s.deviceCount }, () => s.residentLayers)
  );
  if (counts.length <= 1) return undefined;
  // Nothing is on a GPU at all, so there is no window to proportion. `-ngl 0` already says it.
  if (counts.every((c) => c === 0)) return undefined;
  if (Math.max(...counts) === Math.min(...counts)) return undefined;
  return counts.join(',');
}

/**
 * The generation the scenario leaves room for — or nothing, when it leaves none.
 *
 * The window minus the prompt **and the resident prefix**, both of which occupy it. The first
 * version subtracted only the prompt and floored at 1 (raised by Codex on #164): a prompt slider at
 * the full window then produced `--input-len <context> --output-len 1 --max-model-len <context>`, a
 * command that exceeds its own stated limit, and a 47,616-token prefix under a 16,384-token prompt
 * in a 65,536-token window was handed another 49,152 tokens of output it has nowhere to put.
 *
 * `undefined` rather than a manufactured token: a scenario with no room to answer is one the
 * measurement form has to refuse, not one it can round into existence.
 */
function generationTokens(usage: UsageSpec): number | undefined {
  const room = usage.contextTokens - effectivePromptTokens(usage) - (usage.cachedPrefixTokens ?? 0);
  return room > 0 ? room : undefined;
}

/** A shell command written one flag per line, which is how anyone would paste it back. */
function shell(head: string, args: readonly (string | undefined)[]): string {
  const kept = args.filter((a): a is string => a !== undefined);
  if (kept.length === 0) return head;
  return [head, ...kept].join(' \\\n  ');
}

/**
 * The local file the reader downloaded, which is theirs to name.
 *
 * A placeholder is honest for exactly this and for nothing else. `-m` takes a path on the reader's
 * own disk, so no catalog could supply it and inventing one would be noise; the *artifact* is the
 * opposite case, where a made-up value looks like a working answer. Angle brackets rather than a
 * plausible-looking path, so pasting it unedited fails in the shell instead of half-working.
 */
function ggufPlaceholder(model: ModelSpec, quant: QuantSpec): string {
  return `<path to your ${model.name} ${quant.label} .gguf>`;
}

const LLAMA_SERVER: Launcher = {
  id: 'llama-server',
  label: 'llama.cpp (llama-server)',
  runtimeId: 'llama.cpp',
  source: 'https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md',
  checkedOn: '2026-08-01',
};

const LLAMA_BENCH: Launcher = {
  id: 'llama-bench',
  label: 'llama.cpp (llama-bench)',
  runtimeId: 'llama.cpp',
  source: 'https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md',
  checkedOn: '2026-08-01',
};

const OLLAMA: Launcher = {
  id: 'ollama',
  label: 'Ollama',
  runtimeId: 'llama.cpp',
  source: 'https://docs.ollama.com/modelfile',
  checkedOn: '2026-08-01',
};

const VLLM: Launcher = {
  id: 'vllm',
  label: 'vLLM',
  runtimeId: 'vllm',
  source: 'https://docs.vllm.ai/en/stable/configuration/engine_args/',
  checkedOn: '2026-08-01',
};

const MLX: Launcher = {
  id: 'mlx',
  label: 'MLX',
  runtimeId: 'mlx',
  source: 'https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/SERVER.md',
  checkedOn: '2026-08-01',
};

/**
 * One catalog row, three launchers.
 *
 * `RUNTIMES` labels its first row "llama.cpp / Ollama" because the two share an engine and
 * therefore share every figure this app computes — but they are different *command surfaces*, and
 * a placement carries no signal for which one the reader runs. So the row produces both, labelled,
 * rather than a guess between them; `llama-bench` is a third because the measurement form is a
 * separate binary from the server.
 */
const LAUNCHERS: Record<string, readonly Launcher[]> = {
  'llama.cpp': [LLAMA_SERVER, OLLAMA, LLAMA_BENCH],
  vllm: [VLLM],
  mlx: [MLX],
};

/**
 * The reason no launcher for this scenario can emit anything, or nothing.
 *
 * Checked once and applied to every launcher, because these are properties of the *placement* and
 * not of any command surface: a configuration the engine refused is refused whichever binary you
 * would have typed.
 */
function placementRefusal(input: LaunchInput): string | undefined {
  const { placement } = input;
  if (placement.unsupported !== undefined) return placement.unsupported;
  if (placement.impossible) {
    /**
     * **Two different failures wear one flag, and the first draft named only the first** (raised by
     * Codex on #164). `impossible` is set either when the cache and activations alone are over the
     * ceiling — where lowering context or concurrency really does help — *or*, on a rig with
     * nowhere to spill, whenever anything is over at all, which on unified memory and CPU RAM
     * includes an oversized checkpoint whose cache is nowhere near the limit. Telling that reader
     * to lower the context is advice that will not work at any context.
     *
     * `floorBytesPerDevice` is the quantity that splits them: it is the cache plus activations,
     * carried on `Placement` precisely so a sentence and its predicate read one value.
     */
    const machine = input.rig.device.name;
    return placement.floorBytesPerDevice > placement.allocatableBytesPerDevice
      ? `This configuration does not run on ${machine}: the cache and activations alone are over ` +
          `the ceiling, so no flag rescues it. Lower the context or the concurrency and the ` +
          `commands come back.`
      : `This configuration does not run on ${machine}: the weights are over the ceiling and ` +
          `there is nowhere slower to spill them to. Lowering the context will not help — a ` +
          `narrower format or a smaller model is what changes the answer.`;
  }
  return undefined;
}

export function launchCommands(input: LaunchInput): readonly LauncherCommands[] {
  const launchers = LAUNCHERS[input.runtime.id] ?? [];
  const refusal = placementRefusal(input);

  return launchers.map((launcher) => {
    if (refusal !== undefined) {
      return {
        launcher,
        serve: { ok: false, reason: refusal },
        measure: { ok: false, reason: refusal },
      };
    }
    return { launcher, ...EMITTERS[launcher.id](input) };
  });
}

type Pair = { serve: Emission; measure: Emission };

const EMITTERS: Record<string, (input: LaunchInput) => Pair> = {
  'llama-server': llamaServer,
  'llama-bench': llamaBench,
  ollama: ollama,
  vllm: vllm,
  mlx: mlx,
};

function llamaServer(input: LaunchInput): Pair {
  const { model, quant, usage } = input;
  const ngl = gpuLayers(input);
  const kv = LLAMA_KV_TYPE[usage.kvPrecision];
  const split = tensorSplit(input);
  /**
   * The product, and it is the flag most likely to be got wrong by hand. `n_ctx_seq` is
   * `n_ctx / n_seq_max` unless the KV buffer is unified, and passing `-np` explicitly is what
   * turns unification off — so `-c` has to carry the whole rig's cache for each slot to get the
   * window the panel above priced. It is also exactly the quantity `totalKvBytes` was sized from.
   */
  const totalContext = usage.contextTokens * usage.concurrency;

  const notes = [
    `-c is the whole cache, not one user's window: llama.cpp hands each of the -np slots ` +
      `n_ctx / n_parallel, so ${fmt(usage.contextTokens)} tokens for each of ${usage.concurrency} ` +
      `is ${fmt(totalContext)} here.`,
    nglNote(input, ngl),
    `-m takes a path on your own disk, which no catalog can supply — the placeholder is the one ` +
      `thing here you are meant to replace.`,
    ...(split === undefined
      ? []
      : [
          `-ts proportions the -ngl window, not the model: llama.cpp puts the last ${ngl} layers ` +
            `on GPUs and splits those by these ratios. ${split} is the split bench sized, against ` +
            `a default that divides by device memory and therefore evenly on identical cards — ` +
            `which is the wrong answer for a model whose layers cache different amounts.`,
        ]),
  ];

  return {
    serve: {
      ok: true,
      text: shell('llama-server', [
        `-m ${ggufPlaceholder(model, quant)}`,
        `-c ${totalContext}`,
        usage.concurrency > 1 ? `-np ${usage.concurrency}` : undefined,
        `-ngl ${ngl}`,
        `-ctk ${kv} -ctv ${kv}`,
        split === undefined ? undefined : `-ts ${split}`,
      ]),
      notes,
    },
    // The serving form's own client is a different binary, and it is `llama-bench` — which this
    // module offers as a launcher of its own rather than duplicating here.
    measure: {
      ok: false,
      reason: `llama-server serves; it does not measure. The llama-bench command below is the measurement form of this same placement.`,
    },
  };
}

function llamaBench(input: LaunchInput): Pair {
  const { model, quant, usage } = input;
  const ngl = gpuLayers(input);
  const kv = LLAMA_KV_TYPE[usage.kvPrecision];
  const prompt = effectivePromptTokens(usage);
  const gen = generationTokens(usage);
  const prefix = usage.cachedPrefixTokens ?? 0;
  /**
   * `-ts` belongs here too, and leaving it off was the same defect one file over. `llama-bench`
   * takes the flag, and a sharded measurement run against llama.cpp's default even split is not a
   * measurement of the placement the server command reproduces — which makes the number it prints
   * unusable for the one thing the measurement form exists for.
   */
  const split = tensorSplit(input);

  const notes = [
    `-p and -n are this scenario's own prompt and generation lengths, which is what makes the ` +
      `result comparable with the figures above rather than with llama-bench's defaults of 512 ` +
      `and 128.`,
    nglNote(input, ngl),
    ...(split === undefined
      ? []
      : [
          `-ts is the same split the serving command uses. Measuring against llama.cpp's default ` +
            `even split would time a different placement than the one priced above.`,
        ]),
    ...(prefix > 0
      ? [
          `-d runs the test with ${fmt(prefix)} tokens already in the cache, which is what this ` +
            `archetype's resident prefix means. Without it the measurement is a standalone prompt ` +
            `and the prediction above is not: the turn's attention is charged against the prefix.`,
        ]
      : []),
    ...(usage.concurrency > 1
      ? [
          `llama-bench measures one sequence and has no concurrency flag, so it cannot reproduce ` +
            `${usage.concurrency} users. The decode figure it prints is the single-user rate; the ` +
            `capacity above is what ${usage.concurrency} users cost in memory.`,
        ]
      : []),
  ];

  return {
    serve: {
      ok: false,
      reason: `llama-bench measures; it does not serve. The llama-server command above is the serving form of this same placement.`,
    },
    measure:
      gen === undefined
        ? { ok: false, reason: noRoomToAnswer(usage.contextTokens) }
        : {
            ok: true,
            text: shell('llama-bench', [
              `-m ${ggufPlaceholder(model, quant)}`,
              `-p ${prompt}`,
              `-n ${gen}`,
              prefix > 0 ? `-d ${prefix}` : undefined,
              `-ngl ${ngl}`,
              `-ctk ${kv} -ctv ${kv}`,
              split === undefined ? undefined : `-ts ${split}`,
              `-o md`,
            ]),
            notes,
          },
  };
}

function ollama(input: LaunchInput): Pair {
  const { model, quant, usage } = input;
  const tag = `bench-${slug(model.name)}-${slug(quant.label)}`;

  /**
   * **Ollama's Modelfile documents no parameter for the GPU layer count**, and that is the one
   * number this whole feature exists to print. `num_ctx`, `num_predict` and the sampler knobs are
   * the list; `num_gpu` is not on it. So the template says what it cannot say rather than emitting
   * a plausible-looking line — which is the same rule as the artifact refusal, applied to a flag.
   */
  const notes = [
    `Ollama's Modelfile has no parameter for the GPU layer count — num_ctx and num_predict are ` +
      `documented, num_gpu is not — so the ${gpuLayers(input)}-layer split above is the one thing ` +
      `this surface cannot be told. Ollama decides it. Use llama-server if you need to pin it.`,
    `num_ctx is per request here, unlike llama-server's -c, which is the whole cache across slots.`,
    `FROM takes a path on your own disk, absolute or relative to the Modelfile.`,
    `Written to ${tag}.Modelfile rather than to Modelfile, under set -C, so this cannot overwrite ` +
      `one you already have.`,
  ];

  return {
    serve: {
      ok: true,
      text: [
        // A bench-specific filename, never the bare `Modelfile` this first wrote to. `cat >`
        // truncates unconditionally, and the directory an Ollama user runs this from is exactly the
        // one likely to already hold a Modelfile of their own — a copy-pasteable block that
        // silently destroys their file (raised by Codex on #164). `set -C` refuses to clobber even
        // this name, so the worst case is an error rather than a lost file.
        `(set -C; cat > ${tag}.Modelfile) <<'EOF'`,
        `FROM ${ggufPlaceholder(model, quant)}`,
        `PARAMETER num_ctx ${usage.contextTokens}`,
        `EOF`,
        ``,
        `ollama create ${tag} -f ${tag}.Modelfile`,
        `ollama run ${tag}`,
      ].join('\n'),
      notes,
    },
    measure: {
      ok: false,
      reason:
        `Ollama ships no benchmark client. It runs llama.cpp, so the llama-bench command in this ` +
        `panel measures the same engine on the same GGUF — that is the form to submit.`,
    },
  };
}

function vllm(input: LaunchInput): Pair {
  const { model, quant, usage, rig } = input;
  const repo = artifactFor(model, quant.id);
  if (repo === undefined) {
    const reason = noArtifact(model, quant, 'vLLM');
    return { serve: { ok: false, reason }, measure: { ok: false, reason } };
  }

  const tp = effectiveDeviceCount(rig);
  const kv = VLLM_KV_DTYPE[usage.kvPrecision];
  const prompt = effectivePromptTokens(usage);
  const gen = generationTokens(usage);
  const revision = revisionOf(model);

  /**
   * `--gpu-memory-utilization` is emitted rather than left default, and the reason is that the two
   * do not agree: `RuntimeSpec.preallocFraction` is 0.9, which is what every vLLM figure on this
   * page was budgeted against, while vLLM's own default has moved to 0.92. Stating it makes the
   * command reproduce the placement the panel priced instead of a slightly roomier one.
   */
  const notes = [
    `--gpu-memory-utilization is stated rather than left to vLLM's default, which is 0.92 — the ` +
      `figures above are budgeted at 0.9, and this is what makes the command match them.`,
    ...(revision === undefined
      ? []
      : [
          `--revision pins ${revision.slice(0, 10)}, the commit this model's parameter counts and ` +
            `attention shape were read from. Without it vLLM resolves the repo's default branch, ` +
            `which can move.`,
        ]),
    `--max-model-len is one sequence's window; --max-num-seqs is how many of them vLLM will hold ` +
      `at once. Together they are the cache the panel above sized.`,
  ];
  /*
   * No "this rig cannot shard, so the command drives one device" note, and its absence is
   * deliberate rather than an omission. `planPlacement` marks any `count > 1` rig without an
   * interconnect `unsupported`, and `placementRefusal` turns every launcher away before an emitter
   * runs — so inside this function `effectiveDeviceCount(rig)` is always `rig.count`, and a note
   * keyed on the difference could never fire. The refusal upstream owns that case, with its own
   * sentence.
   */

  return {
    serve: {
      ok: true,
      text: shell(`vllm serve ${repo}`, [
        revision === undefined ? undefined : `--revision ${revision}`,
        `--max-model-len ${usage.contextTokens}`,
        tp > 1 ? `--tensor-parallel-size ${tp}` : undefined,
        kv === undefined ? undefined : `--kv-cache-dtype ${kv}`,
        `--gpu-memory-utilization 0.9`,
        `--max-num-seqs ${usage.concurrency}`,
      ]),
      notes,
    },
    measure:
      gen === undefined
        ? {
            ok: false,
            reason: noRoomToAnswer(usage.contextTokens),
          }
        : {
            ok: true,
            text: shell(`vllm bench latency`, [
              `--model ${repo}`,
              revision === undefined ? undefined : `--revision ${revision}`,
              `--input-len ${prompt}`,
              `--output-len ${gen}`,
              // The configured user count, not the client's own default of 8. Without it the
              // benchmark times a different batch from the one the panel priced, which makes the
              // number unusable for the thing a measurement is for (raised by Codex on #164).
              `--batch-size ${usage.concurrency}`,
              `--max-model-len ${usage.contextTokens}`,
              tp > 1 ? `--tensor-parallel-size ${tp}` : undefined,
              kv === undefined ? undefined : `--kv-cache-dtype ${kv}`,
              `--gpu-memory-utilization 0.9`,
            ]),
            notes: [
              `vllm bench latency times one batch offline, at this scenario's own prompt, ` +
                `generation length and user count — which is what makes the result comparable ` +
                `with the figures above rather than with the client's defaults of 32, 128 and 8.`,
              ...(revision === undefined
                ? []
                : [
                    `--revision pins the commit the catalog derived this model's shape from, so ` +
                      `the command cannot quietly load a newer checkpoint than the figures ` +
                      `describe.`,
                  ]),
              `Needs the bench extra: pip install vllm[bench].`,
            ],
          },
  };
}

function mlx(input: LaunchInput): Pair {
  const { model, quant, usage, runtime } = input;

  /**
   * The #18 refusal, and the sharp end of it. Every GGUF row in MLX's `weightFormats` is a
   * stand-in *by width* for MLX's own affine scheme — the figures are modelled, the checkpoint is
   * not one MLX loads. A command naming it would be a command for a file that does not exist, in
   * copy-pasteable form, which is exactly the failure the substitution marker was built to prevent.
   */
  const substitution = substitutionFor(runtime, quant.id);
  if (substitution !== undefined) {
    const reason =
      `MLX does not load ${quant.label}. ${substitution} That makes the figures above a modelled ` +
      `stand-in and this checkpoint a file that does not exist — so there is no command. Convert ` +
      `the weights yourself with mlx_lm.convert, or select BF16, which MLX reads natively.`;
    return { serve: { ok: false, reason }, measure: { ok: false, reason } };
  }

  const repo = artifactFor(model, quant.id);
  if (repo === undefined) {
    const reason = noArtifact(model, quant, 'MLX');
    return { serve: { ok: false, reason }, measure: { ok: false, reason } };
  }

  const prompt = effectivePromptTokens(usage);
  const gen = generationTokens(usage);
  /**
   * **`mlx_lm.server` has no KV quantization flag.** `--kv-bits`, `--kv-group-size` and
   * `--quantized-kv-start` are `mlx_lm.generate`'s; the server's argument list does not carry
   * them. So an 8-bit-cache scenario cannot be *served* at the precision it was priced at, and the
   * command says so rather than printing a flag that does not parse.
   */
  const quantizedCache = usage.kvPrecision !== 'fp16';

  return {
    /**
     * **A refusal rather than a warning, when the cache precision cannot be reproduced** (raised by
     * Codex on #164, P1). The first version emitted the server command with a loud note saying the
     * cache would be fp16 — but a long-context configuration that fits *because* an 8-bit cache
     * halves it will OOM when run at fp16, and a note beside a copy button does not stop that. The
     * command is not a command for the placement the panel priced, so there is no command.
     *
     * The measurement form survives, because `mlx_lm.generate` does take `--kv-bits`.
     */
    serve: quantizedCache
      ? {
          ok: false,
          reason:
            `The figures above are priced with an 8-bit cache, and mlx_lm.server has no flag for ` +
            `it — --kv-bits is on mlx_lm.generate, not the server. A served command would run an ` +
            `fp16 cache needing roughly twice the memory shown, which is a different placement ` +
            `from the one above. Select an FP16 cache to get a serving command, or use the ` +
            `measurement command below, which does take the precision.`,
        }
      : {
          ok: true,
          text: shell(`mlx_lm.server`, [
            `--model ${repo}`,
            ...(gen === undefined ? [] : [`--max-tokens ${gen}`]),
          ]),
          notes: [
            `MLX reads this repo directly; no conversion step, because BF16 is the one format the ` +
              `catalog and MLX agree on.`,
            // No `--revision` on this server, so the pin the vLLM command gets is unavailable here.
            ...(revisionOf(model) === undefined
              ? []
              : [
                  `mlx_lm.server takes no revision flag, so this resolves the repo's default ` +
                    `branch. The figures above were derived from ${revisionOf(model)!.slice(0, 10)}.`,
                ]),
          ],
        },
    measure:
      gen === undefined
        ? { ok: false, reason: noRoomToAnswer(usage.contextTokens) }
        : {
            ok: true,
            text: shell(`python -c "print('word ' * ${prompt})" | mlx_lm.generate`, [
              `--model ${repo}`,
              `--prompt -`,
              `--max-tokens ${gen}`,
              // `--quantized-kv-start` defaults to 5,000 on the CLI and the engine prices every token at
              // the selected precision, so a run finishing under that threshold would benchmark an
              // entirely fp16 cache against a Q8 prediction (raised by Codex on #164). Zero is what makes
              // the measured cache the priced one.
              ...(quantizedCache
                ? [`--kv-bits 8`, `--kv-group-size 64`, `--quantized-kv-start 0`]
                : []),
            ]),
            notes: [
              `mlx_lm.generate prints prompt and generation tokens/sec, which are the two figures above.`,
              `The prompt is piped rather than quoted because its *length* is what is being measured — ` +
                `${fmt(prompt)} tokens. "word " is roughly one token each, so treat the count as ` +
                `approximate and read the tokens-per-second the tool reports back.`,
              ...(quantizedCache
                ? [
                    `--kv-bits 8 is the precision the figures above assume, and unlike the server, ` +
                      `mlx_lm.generate takes it. --quantized-kv-start 0 is what makes it apply from the ` +
                      `first token: the CLI default is 5,000, and the engine prices every token at 8 bits.`,
                  ]
                : []),
            ],
          },
  };
}

/** The `-ngl` sentence, which differs in the three cases the flag has. */
function nglNote(input: LaunchInput, ngl: number): string {
  const { model, rig, placement } = input;
  if (rig.device.class === 'cpu-ram') {
    return `-ngl 0 because ${rig.device.name} has no GPU to offload to — every layer runs on the host, which is what the figures above price.`;
  }
  if (ngl > model.layers) {
    return `-ngl ${ngl} is all ${model.layers} layers plus one: llama.cpp counts the output tensor a position past the repeating blocks, so ${model.layers} would leave it on the host.`;
  }
  return (
    `-ngl ${ngl} of ${model.layers} layers is the split bench sized, not a fraction of the model: ` +
    `${percentish(placement.offloadFraction)} of the weights spill to host RAM, and which layers ` +
    `stay is what decides whether that is ${ngl} or ${ngl + 2}.`
  );
}

/** Thousands separators, since these are long token counts a reader has to check against a panel. */
function fmt(n: number): string {
  return n.toLocaleString('en-US');
}

function percentish(fraction: number): string {
  return `${Math.round(fraction * 100)}%`;
}

/** A shell- and Ollama-safe tag: lowercase, no spaces, no slashes. */
function slug(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '');
}
