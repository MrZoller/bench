# bench

Work out which open-weight LLMs run on your hardware, and how comfortably — across discrete
GPUs, unified-memory machines, and CPU+RAM.

Most VRAM calculators sort hardware by one number and apply one KV-cache formula to every
model. Both shortcuts break on the machines and models people actually care about. Hardware is
a **capacity / bandwidth / compute triangle**: a DGX Spark holds 128 GB at 273 GB/s and prefills
fast but decodes slowly; a Mac Studio M3 Ultra is the inverse; an RTX 5090 is quick at
everything inside 32 GB. And the naive `2 × layers × kv_heads × head_dim` formula overstates
DeepSeek-family models by several times (MLA caches one compressed latent per layer, ~70 KB per
token) and roughly doubles anything with sliding-window layers. At long context those errors are
tens of gigabytes — the difference between "buy another GPU" and "you're fine".

So `bench` computes rather than approximates. Model architectures come from each repo's own
`config.json` on Hugging Face and exact parameter counts from its safetensors index, so weights
and KV are derived per model instead of guessed from a size class. Throughput is a roofline
calibrated against published measurements at both ends of the hardware range, and it reports
prefill and decode separately, because a machine can be strong at one and weak at the other.

## Setup

```
npm install
npm run dev
```

## Usage

```ts
import { evaluate } from '@/engine';

const { placement, decode, prefill } = evaluate({
  model: GPT_OSS_120B,
  quant: getQuant('mxfp4'),
  usage: { contextTokens: 32768, concurrency: 4, kvPrecision: 'q8' },
  rig: { device: DGX_SPARK, count: 1 },
  runtime: LLAMA_CPP,
});

placement.fits; // does it load at all
decode.perUserTokensPerSec; // how fast it feels
prefill.ttftSeconds; // how long before the first token
```

## Development

- Test: `npm test`
- Lint/format: `npm run lint` / `npm run format`

The engine under `src/engine/` is pure — no React imports — so it can be pinned to published
reference values. Those tests are the point: `src/engine/*.test.ts` asserts against llama.cpp's
own published file sizes, DeepSeek's stated KV footprint, and measured throughput on a DGX Spark
and an EPYC 9654. Treat a failure there as the numbers being wrong, not the test.
