# Calls made overnight, 25 July 2026

Chris handed the queue over with "merge each PR after it comes back clean", which meant the
judgment calls I had been parking became mine. Each one below is recorded so it can be reversed
cheaply. Nothing here is expensive to undo; the reasoning matters more than the choice.

## Reversed a review's proposed remedy

**Decode's parameter basis** (Codex, PR #1). The finding was right — decode was charging the
input embedding table it only reads one row of — but the suggested fix, using `activeParams`,
would have swapped one error for another. When a model **ties** its embedding to the output
projection, that table _is_ a full vocab matmul every step, so subtracting it understates Gemma 3
12B by 5%.

Took a third path: `activeDenseParams`, with tied-ness read from the **absence of an
`lm_head.weight` tensor**, never from `config.tie_word_embeddings`. That config key is undefined
on both Gemma 3 repos despite them being tied, so the obvious route would have dropped a 1.0B
table decode genuinely reads.

## Declined, with reasons on the thread

**"Mark the catalog phase pending"** (Codex, PR #2). Two of its three assertions do not hold:
`npm run catalog` has been in `package.json` on `main` since the scaffold and PR #2 does not touch
that file, and bundling the catalog artifacts into a docs-only PR would duplicate PR #1 and
conflict on merge. The live concern was merge order, so the status row was reworded to be true
under either. Left open for Chris initially; resolved once the queue was handed over, with the
refutation on record.

**Reclassifying the RTX 3090 as NVLink-class** (Codex, PR #1). The bridge is real but two-way only
and an accessory most owners do not have, so promoting the row would overstate every unbridged
pair to fix the rarer one — and a flat flag would be wrong at count 4 in the other direction.
Recorded as a `note` instead. The correct fix is a count-aware interconnect, which is a schema
change with no measurement behind either value.

## Chose constants where none were measured

**Interconnect tiers.** `TP_SCALING` went from `{nvlink, pcie}` to
`{fabric: 0.95, pcie: 0.85, network: 0.7}`. Matching only `/nvlink/` put AMD's Infinity Fabric
and the Spark's 200GbE in the same bucket despite sitting on opposite sides of PCIe — and they
err in _opposite_ directions, so no single default could serve both. The ordering is defensible;
the values are not measured, and the code says so.

**Calibration was deliberately not retuned.** Fixing the per-token basis moved the DGX Spark
anchors from ~10% and ~6% _under_ to ~19% and ~10% _over_, while EPYC stayed within 1% — evidence
the old fit was partly absorbing those bugs. Re-centring a fudge factor immediately after removing
what it was masking is how the next error gets hidden. All three anchors remain inside the ±30%
band the tests assert. **This is the one most worth a second opinion.**

## Scope calls

**Multi-device is now discrete-GPU only.** Every unified-soc row except the Spark and every
cpu-ram row has no `interconnect`, so the PCIe tier was being applied to transports that do not
exist — an 8× Mac mini rig reported 170 tok/s for hardware that cannot shard a model between
chassis.

**Popularity for mirrored seeds comes from the canonical repo.** Llama 3.1 70B shipped with
NousResearch's 4,838 downloads against Meta's 1,235,788, ranking the best-known model in the
catalog dead last. Gating covers `/raw/` and `/resolve/` but not API metadata, so this needs no
token. `popularity.measuredOn` records the substitution rather than leaving it implicit.

**Any seed failure now blocks the catalog write**, with `--allow-partial` as the escape hatch.
The old threshold tolerated 5 of 17 failures and still overwrote the committed artifact, exiting 0.

## What I got wrong, and how it was caught

Worth reading, because the pattern repeated:

- **Twice I fixed a display bug one state short of the real condition.** Suppressing speed
  readouts for `unsupported` but not `impossible`; gating the teaching heading on `fits` but not
  on `unsupported`. Both times the second state was the more dangerous one — `impossible` configs
  compute as though every weight were resident, so 41 catalogued combinations painted a green
  "Fast" beside a red "Will not run".
- **I justified dropping the output projection as "noise: one token against a prompt of
  hundreds."** True in the limit, false at the short end — on a one-token prompt it is 16% of the
  work. Per-token minus, once-per-request plus, is exact at every length.
- **A comment claimed the projection was "0.58B against a 0.63B remainder"**, which compared it to
  the dense remainder and ignored the 2.39B expert term. The percentage was right; the arithmetic
  behind it was not, and the first version of the guarding test asserted the wrong ratio because
  of it.
- **The palette I would have shipped by eye was colourblind-unsafe.** Magenta beside aqua measures
  ΔE 1.6 under deuteranopia — one colour for a deuteranope — and looked perfectly distinct to me.
  The validator caught it; nothing else would have.

## Still open for Chris

- The calibration decision above.
- Whether `docs/ROADMAP.md`'s status table should track "work complete" or "landed on `main`" —
  currently the former, worded to be true under either merge order.
- `main` is unprotected: rulesets need GitHub Pro on a private repo, so "PRs only, threads
  resolved" is convention rather than enforcement.
