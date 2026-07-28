import { execFile } from 'node:child_process';
import { mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { promisify } from 'node:util';
import { describe, expect, it } from 'vitest';
import { compare } from './catalog-diff';

const run = promisify(execFile);

/**
 * What decides whether the weekly refresh opens a pull request.
 *
 * Tested rather than trusted because both wrong answers are expensive and neither is visible
 * when it happens. A false positive is a pull request every Monday with nothing in it, which
 * trains people to close the bot unread — and so to close it the week a real architecture change
 * lands. A false negative is a catalog that silently stops tracking the repos it claims to be
 * derived from, on a page whose whole thesis is that the figures are derived and not typed.
 */

const model = (id: string, extra: Record<string, unknown> = {}) => ({
  id,
  name: id,
  totalParams: 8e9,
  layers: 32,
  attention: { core: { kind: 'gqa', kvHeads: 8, headDim: 128 } },
  ...extra,
});

const catalog = (models: ReturnType<typeof model>[], failures: string[] = []) => ({
  generatedAt: '2026-07-25T03:59:07.334Z',
  failures,
  models,
});

describe('deciding whether a refresh is worth a pull request', () => {
  /**
   * The case the script exists for. `build-catalog.ts` stamps `generatedAt` on every write, so
   * the file differs after every run whether or not a figure moved — and `git diff --quiet`,
   * the obvious wiring, would open an empty pull request every week for the rest of the
   * project's life.
   */
  it('ignores the timestamp, which moves on every single run', () => {
    const before = catalog([model('Qwen/Qwen3-8B')]);
    const after = { ...catalog([model('Qwen/Qwen3-8B')]), generatedAt: '2099-01-01T00:00:00.000Z' };

    expect(compare(before, after).changed).toBe(false);
    expect(compare(before, after).summary).toMatch(/no change/i);
  });

  it('notices a figure that moved, and names the field', () => {
    const result = compare(
      catalog([model('Qwen/Qwen3-8B')]),
      catalog([model('Qwen/Qwen3-8B', { totalParams: 8.2e9 })])
    );

    expect(result.changed).toBe(true);
    expect(result.summary).toMatch(/totalParams/);
    expect(result.summary).toMatch(/Qwen\/Qwen3-8B/);
  });

  /**
   * Nested rather than top-level, because that is where the fields the engine most depends on
   * live — a changed `kvHeads` is the difference between a cache estimate that is right and one
   * that is off by a multiple, and a shallow `!==` on an object would miss it entirely.
   */
  it('sees a change inside the attention spec', () => {
    const result = compare(
      catalog([model('Qwen/Qwen3-8B')]),
      catalog([
        model('Qwen/Qwen3-8B', {
          attention: { core: { kind: 'gqa', kvHeads: 4, headDim: 128 } },
        }),
      ])
    );

    expect(result.changed).toBe(true);
    expect(result.summary).toMatch(/attention/);
  });

  it('reports an added model', () => {
    const result = compare(
      catalog([model('Qwen/Qwen3-8B')]),
      catalog([model('Qwen/Qwen3-8B'), model('brand/New-9B')])
    );

    expect(result.changed).toBe(true);
    expect(result.summary).toMatch(/\*\*Added\*\* \(1\): brand\/New-9B/);
  });

  /**
   * A removal is the one outcome that silently shrinks the product, so it gets a sentence rather
   * than a line item. The generator refuses a partial write for the same reason — but a seed
   * dropped from the list on purpose reaches here legitimately and still wants a second look.
   */
  it('calls out a removed model in words, not just in a list', () => {
    const result = compare(
      catalog([model('Qwen/Qwen3-8B'), model('Qwen/Qwen3-32B')]),
      catalog([model('Qwen/Qwen3-8B')])
    );

    expect(result.changed).toBe(true);
    expect(result.summary).toMatch(/\*\*Removed\*\* \(1\): Qwen\/Qwen3-32B/);
    expect(result.summary).toMatch(/disappears from the product/);
  });

  /**
   * A run that dropped seeds writes them into `failures`, and the artifact is what the app
   * loads — so a refresh that lost two models to a Hugging Face outage has to be legible as
   * that, rather than as two unexplained removals.
   */
  it('reports a change in seed failures', () => {
    const result = compare(
      catalog([model('Qwen/Qwen3-8B')]),
      catalog([model('Qwen/Qwen3-8B')], ['meta-llama/Llama-3.1-8B: 401'])
    );

    expect(result.changed).toBe(true);
    expect(result.summary).toMatch(/Seed failures.*0 → 1/);
  });

  /**
   * Order is not substance. The generator emits models in seed order, so this is not reachable
   * today — which is exactly why it is pinned: a future sort by popularity or name would
   * otherwise report the entire catalog as changed on the week it landed, and every week after
   * would look identical again with no way to tell the two apart from the summary.
   */
  it('treats a reordered catalog as changed, since the file really did move', () => {
    const a = model('a/One');
    const b = model('b/Two');

    expect(compare(catalog([a, b]), catalog([b, a])).changed).toBe(true);
    // But nothing about either model differs, so the summary must not invent edits.
    expect(compare(catalog([a, b]), catalog([b, a])).summary).not.toMatch(/\| `a\/One` \|/);
  });

  it('handles an empty or malformed catalog without throwing', () => {
    expect(compare({}, {}).changed).toBe(false);
    expect(compare({}, catalog([model('a/One')])).changed).toBe(true);
  });
});

/**
 * The `--exit-code` wiring, exercised through the actual command line.
 *
 * `catalog-refresh.yml` branches on this exit status to decide whether to commit to the open
 * refresh branch, and no test of `compare()` can see that: the bug it replaced was
 * `git diff --quiet`, which is shell, and its successor is an exit code, which is also shell.
 * The default invocation is pinned alongside it because the two conventions are deliberately
 * different — a scheduled job that exits non-zero for "nothing changed" ends up permanently red
 * and permanently ignored, so only the opted-in flag may do that.
 */
describe('the command line the refresh workflow drives', () => {
  const script = join(import.meta.dirname, 'catalog-diff.ts');

  const write = (name: string, body: unknown) => {
    const path = join(mkdtempSync(join(tmpdir(), 'catalog-')), name);
    writeFileSync(path, JSON.stringify(body));
    return path;
  };

  const same = () => catalog([model('a/One')]);
  const moved = () => catalog([model('a/One', { layers: 40 })]);

  it('exits 0 without the flag, whatever the answer', async () => {
    const a = write('a.json', same());
    for (const other of [same(), moved()]) {
      const result = await run('npx', ['tsx', script, a, write('b.json', other)]);
      expect(result.stdout.length).toBeGreaterThan(0);
    }
  });

  it('exits 0 with --exit-code when only the timestamp moved', async () => {
    const a = write('a.json', same());
    const b = write('b.json', { ...same(), generatedAt: '2099-01-01T00:00:00.000Z' });

    await expect(run('npx', ['tsx', script, a, b, '--exit-code'])).resolves.toBeTruthy();
  });

  it('exits 1 with --exit-code when a figure moved', async () => {
    const a = write('a.json', same());
    const b = write('b.json', moved());

    await expect(run('npx', ['tsx', script, a, b, '--exit-code'])).rejects.toMatchObject({
      code: 1,
    });
  });
});

/**
 * `changed` and the summary are two answers to one question, and they were derived from different
 * evidence: `changed` from whole-document `JSON.stringify` equality, the summary from set
 * differences on model id plus a per-field compare. Both halves of the summary are blind to array
 * order and key order; the equality test is not. So any reordering that left every figure intact
 * set `changed` with nothing at all to say — and the workflow interpolates that summary into both
 * the commit message and the pull request body.
 *
 * A pull request that says nothing is the same failure as one that is wrong every week, and worse
 * in one way: it asks for a review it gives no subject for.
 */
describe('what changed and what it says cannot disagree', () => {
  /**
   * The same row, written with its keys in a different order — every value identical.
   *
   * Built from `model` rather than hand-written, so it cannot drift into being a different row
   * than the one it is compared against. Key order is a serialization artifact; this is the shape
   * that made whole-document equality disagree with a summary that is blind to it.
   */
  const keysReordered = (id: string) => {
    const row = model(id);
    return {
      attention: row.attention,
      layers: row.layers,
      totalParams: row.totalParams,
      name: row.name,
      id: row.id,
    };
  };

  /** Every shape of difference the comparison can be handed, substantive or not. */
  const CASES: [string, Parameters<typeof compare>][] = [
    ['identical', [catalog([model('a/One')]), catalog([model('a/One')])]],
    [
      'timestamp only',
      [
        catalog([model('a/One')]),
        { ...catalog([model('a/One')]), generatedAt: '2099-01-01T00:00:00.000Z' },
      ],
    ],
    [
      'models reordered',
      [catalog([model('a/One'), model('b/Two')]), catalog([model('b/Two'), model('a/One')])],
    ],
    ['keys reordered within a row', [catalog([model('a/One')]), catalog([keysReordered('a/One')])]],
    ['model added', [catalog([model('a/One')]), catalog([model('a/One'), model('b/Two')])]],
    ['model removed', [catalog([model('a/One'), model('b/Two')]), catalog([model('a/One')])]],
    ['field edited', [catalog([model('a/One')]), catalog([model('a/One', { layers: 40 })])]],
    ['failures grew', [catalog([model('a/One')]), catalog([model('a/One')], ['b/Two: 401'])]],
    [
      'failures swapped at the same length',
      [catalog([model('a/One')], ['b/Two: 401']), catalog([model('a/One')], ['c/Three: 404'])],
    ],
    ['empty on both sides', [{}, {}]],
  ];

  /**
   * The property, stated once over every case rather than asserted case by case: whenever the
   * script says a refresh is worth a pull request, it has to be able to say what moved. This is
   * the invariant the workflow depends on and the one that was violated.
   */
  it.each(CASES)('%s: a claimed change always states what changed', (_label, [before, after]) => {
    const { changed, summary } = compare(before, after);

    expect(summary.trim()).not.toBe('');
    if (changed) {
      expect(summary).not.toMatch(/^no change/i);
    } else {
      expect(summary).toMatch(/no change/i);
    }
  });

  it('treats a pure reordering as a change, and says that is all it was', () => {
    const result = compare(
      catalog([model('a/One'), model('b/Two')]),
      catalog([model('b/Two'), model('a/One')])
    );

    expect(result.changed).toBe(true);
    expect(result.summary).toMatch(/reordered/i);
    // The point of the line: a reader must not have to guess whether a figure moved.
    expect(result.summary).toMatch(/no figure changed/i);
    expect(result.summary).not.toMatch(/\| `a\/One` \|/);
  });

  /**
   * Key order within a row is a serialization artifact rather than an ordering anyone chose, and
   * `changedFields` is already blind to it. Reporting it would be a pull request about whitespace.
   */
  it('treats a row whose keys moved as no change at all', () => {
    const result = compare(catalog([model('a/One')]), catalog([keysReordered('a/One')]));

    expect(result.changed).toBe(false);
    expect(result.summary).toMatch(/no change/i);
    // The two really are key-order variants of one row, not two different rows that happen to
    // compare equal — otherwise this passes for a reason that has nothing to do with the guard.
    expect(JSON.stringify(model('a/One'))).not.toBe(JSON.stringify(keysReordered('a/One')));
  });

  /**
   * Failures were reported only when the *length* moved, so one seed starting to fail as another
   * stopped read as a quiet week — the third route to an empty summary, and the one that matters
   * most, since the whole point of the list is to say what the catalog was written without.
   */
  it('reports failures swapping at the same length, and names both sides', () => {
    const result = compare(
      catalog([model('a/One')], ['b/Two: 401']),
      catalog([model('a/One')], ['c/Three: 404'])
    );

    expect(result.changed).toBe(true);
    expect(result.summary).toMatch(/Now failing: c\/Three: 404/);
    expect(result.summary).toMatch(/No longer failing: b\/Two: 401/);
  });

  it('never opens the summary on a blank line', () => {
    // The summary is interpolated into the commit message and the PR body, where a leading blank
    // line reads as a missing section.
    for (const [, [before, after]] of CASES) {
      expect(compare(before, after).summary).not.toMatch(/^\n/);
    }
  });

  /** `byId` collapses a repeated id, so a duplicated row is invisible to every other check. */
  it('catches the generator writing one id twice', () => {
    const result = compare(catalog([model('a/One')]), catalog([model('a/One'), model('a/One')]));

    expect(result.changed).toBe(true);
    expect(result.summary).toMatch(/repeated ids/i);
  });
});

/**
 * The three cases Codex raised on PR #59, each a way for the two halves to disagree again.
 *
 * All three share a shape: a check that asks about the *new* document alone, or about a coarser
 * property than the one it claims to test, and so either fires on a week where nothing happened
 * or stays silent on a week where something did.
 */
describe('a claim about a refresh is a claim about the difference', () => {
  it('does not re-report a duplicate id the committed catalog already had', () => {
    // Nothing happened this week: the generator reproduced exactly what is committed, duplicate
    // and all. Asking "does the new side contain a duplicate" would open a pull request every
    // Monday for as long as the duplicate survives — the weekly noise this file exists to avoid.
    const both = catalog([model('a/One'), model('a/One')]);
    const result = compare(both, catalog([model('a/One'), model('a/One')]));

    expect(result.changed).toBe(false);
    expect(result.summary).toMatch(/no change/i);
  });

  it('still reports a duplicate the moment one appears', () => {
    const result = compare(catalog([model('a/One')]), catalog([model('a/One'), model('a/One')]));

    expect(result.changed).toBe(true);
    expect(result.summary).toMatch(/repeated ids/i);
    expect(result.summary).toMatch(/a\/One/);
  });

  it('reports a duplicate being resolved, which is also a difference', () => {
    const result = compare(catalog([model('a/One'), model('a/One')]), catalog([model('a/One')]));

    expect(result.changed).toBe(true);
    expect(result.summary).toMatch(/repeated ids/i);
  });

  it('does not claim a duplicate when a model is merely added', () => {
    // An addition takes an id from zero occurrences to one, which a naive count difference reads
    // as a multiplicity change. `added` is the line that explains it.
    const result = compare(catalog([model('a/One')]), catalog([model('a/One'), model('b/Two')]));

    expect(result.summary).toMatch(/\*\*Added\*\*/);
    expect(result.summary).not.toMatch(/repeated ids/i);
  });

  /**
   * A seed-order edit landing in the same week as an upstream figure change is the collision that
   * makes the two lines contradict each other. The commit message and the PR body both carry this
   * text, so a summary asserting "No figure changed" above a table of changed fields is worse than
   * either line alone.
   */
  it('does not deny the edits it just listed when the order moved too', () => {
    const result = compare(
      catalog([model('a/One'), model('b/Two')]),
      catalog([model('b/Two', { layers: 40 }), model('a/One')])
    );

    expect(result.changed).toBe(true);
    expect(result.summary).toMatch(/\| `b\/Two` \| layers \|/);
    expect(result.summary).toMatch(/reordered/i);
    expect(result.summary).not.toMatch(/no figure changed/i);
  });

  it('still says no figure changed when none did', () => {
    const result = compare(
      catalog([model('a/One'), model('b/Two')]),
      catalog([model('b/Two'), model('a/One')])
    );

    expect(result.summary).toMatch(/no figure changed/i);
  });

  /**
   * `['a','a','b']` → `['a','b','b']`: same length, same set, different multiset. Both the length
   * check and the two set differences agree nothing happened, while the seed failing twice has
   * changed. The comment on that code claimed to handle multiplicity; only counting occurrences
   * actually does.
   */
  it('sees a failure list whose multiplicity moved under a constant length and set', () => {
    const result = compare(
      catalog([model('a/One')], ['x: 401', 'x: 401', 'y: 404']),
      catalog([model('a/One')], ['x: 401', 'y: 404', 'y: 404'])
    );

    expect(result.changed).toBe(true);
    expect(result.summary).toMatch(/seed failures/i);
  });

  it('leaves an unchanged failure list alone, however it repeats', () => {
    const failures = ['x: 401', 'x: 401', 'y: 404'];
    const result = compare(
      catalog([model('a/One')], [...failures]),
      catalog([model('a/One')], [...failures])
    );

    expect(result.changed).toBe(false);
  });
});

/**
 * The invariants, swept exhaustively rather than sampled.
 *
 * Every hand-written case above encodes a failure someone already found. This is the check that
 * does not depend on having thought of the case: it enumerates every catalog of up to three rows
 * drawn from a pool that includes two rows sharing an id, compares all 1,600 ordered pairs, and
 * asserts the four properties the workflow actually relies on.
 *
 * It was worth the trouble. Three of the bugs this file has now fixed — a committed duplicate
 * re-reported every week, a change to a shadowed row going silent, a summary asserting "No figure
 * changed" above a table of changed fields — were each found by a reviewer reading the code rather
 * than by a test, and each was invisible to the cases written at the time.
 */
describe('the invariants hold across every small catalog', () => {
  const POOL = [
    { id: 'a/One', layers: 1, attention: { kind: 'gqa', kvHeads: 8 } },
    // Same id, different figures: the shape that makes `byId`'s last-wins collapse observable.
    { id: 'a/One', layers: 2, attention: { kind: 'gqa', kvHeads: 8 } },
    { id: 'b/Two', layers: 1, attention: { kind: 'gqa', kvHeads: 8 } },
  ];

  const CATALOGS: Record<string, unknown>[][] = [];
  for (let size = 0; size <= 3; size++) {
    const build = (prefix: Record<string, unknown>[]) => {
      if (prefix.length === size) return void CATALOGS.push([...prefix]);
      for (const row of POOL) build([...prefix, row]);
    };
    build([]);
  }

  /** Substance: row order matters, key order does not — the same reading `compare` uses. */
  const deepSort = (value: unknown): unknown =>
    value && typeof value === 'object' && !Array.isArray(value)
      ? Object.fromEntries(
          Object.entries(value as Record<string, unknown>)
            .sort(([a], [b]) => (a < b ? -1 : 1))
            .map(([k, v]) => [k, deepSort(v)])
        )
      : value;
  const canonical = (models: Record<string, unknown>[]) =>
    JSON.stringify(models.map((m) => JSON.stringify(deepSort(m))));

  it('decides identically to a byte comparison that ignores only key order', () => {
    const falseNegatives: string[] = [];
    const falsePositives: string[] = [];

    for (const before of CATALOGS) {
      for (const after of CATALOGS) {
        const { changed } = compare(
          { models: before, failures: [] },
          { models: after, failures: [] }
        );
        const differs = canonical(before) !== canonical(after);

        if (differs && !changed) falseNegatives.push(`${canonical(before)} -> ${canonical(after)}`);
        if (!differs && changed) falsePositives.push(`${canonical(before)} -> ${canonical(after)}`);
      }
    }

    // A false negative is a catalog that silently stopped tracking what it claims to derive from;
    // a false positive is the weekly empty pull request. Both are named in the file's own header.
    expect(falseNegatives, `reported unchanged: ${falseNegatives.slice(0, 3).join(' | ')}`).toEqual(
      []
    );
    expect(falsePositives, `reported changed: ${falsePositives.slice(0, 3).join(' | ')}`).toEqual(
      []
    );
  });

  it('never claims a change it cannot describe, and never describes one it did not claim', () => {
    for (const before of CATALOGS) {
      for (const after of CATALOGS) {
        const { changed, summary } = compare(
          { models: before, failures: [] },
          { models: after, failures: [] }
        );

        expect(summary.trim()).not.toBe('');
        expect(/^no change/i.test(summary)).toBe(!changed);
      }
    }
  });

  it('never claims a reordering when the ids themselves moved', () => {
    for (const before of CATALOGS) {
      for (const after of CATALOGS) {
        const { summary } = compare(
          { models: before, failures: [] },
          { models: after, failures: [] }
        );
        const ids = (models: Record<string, unknown>[]) =>
          JSON.stringify(models.map((m) => String(m.id)).sort());

        // A reordering is the same rows in a different arrangement. If the id multiset moved,
        // something was added, removed or duplicated, and that is the line that should say so.
        if (/\*\*Reordered\*\*/.test(summary)) expect(ids(before)).toBe(ids(after));
      }
    }
  });

  it('never prints a summary that contradicts itself', () => {
    for (const before of CATALOGS) {
      for (const after of CATALOGS) {
        const { summary } = compare(
          { models: before, failures: [] },
          { models: after, failures: [] }
        );

        // "No figure changed" beneath a table of changed figures — the collision Codex raised.
        if (/no figure changed/i.test(summary)) {
          expect(summary).not.toMatch(/\*\*Changed\*\*/);
          expect(summary).not.toMatch(/\*\*Repeated ids\*\*/);
        }
      }
    }
  });

  it('decides the failure list on the same multiset the summary names', () => {
    const LISTS = [[], ['x'], ['y'], ['x', 'x'], ['x', 'y'], ['x', 'x', 'y'], ['x', 'y', 'y']];
    const models = [POOL[0]];

    for (const before of LISTS) {
      for (const after of LISTS) {
        const { changed, summary } = compare(
          { models, failures: before },
          { models, failures: after }
        );
        const differs = JSON.stringify([...before].sort()) !== JSON.stringify([...after].sort());

        expect(changed, `${JSON.stringify(before)} -> ${JSON.stringify(after)}`).toBe(differs);
        // And when it did change, the line says which seed moved rather than only a total —
        // `['x','x','y']` to `['x','y','y']` is 3 → 3 and has to name `x` and `y`.
        if (differs) expect(summary).toMatch(/seed failures/i);
      }
    }
  });
});

/**
 * The second round of Codex findings on PR #59, all consequences of comparing an id's *presence*
 * where the question was about its *rows*.
 */
describe('a repeated id is reported wherever it appears', () => {
  const one = (id: string, extra: Record<string, unknown> = {}) => model(id, extra);

  it('reports a newly added seed the generator wrote twice', () => {
    // The id is absent from the old side entirely, so a check over shared ids alone never sees it:
    // the summary said "Added (1)" and nothing about the row appearing twice.
    const result = compare(
      catalog([one('b/Two')]),
      catalog([one('b/Two'), one('a/One'), one('a/One')])
    );

    expect(result.changed).toBe(true);
    expect(result.summary).toMatch(/\*\*Added\*\* \(1\): a\/One/);
    expect(result.summary).toMatch(/repeated ids/i);
    expect(result.summary).toMatch(/\| `a\/One` \| 0 \| 2 \|/);
  });

  /**
   * The warning has to describe what a duplicate actually does, or it is worse than no warning:
   * it sends a reviewer looking for a row that vanished when what really happens is that the same
   * model appears twice. `MODELS` is `modelsJson.models.map(toModel)` and keeps every row, so both
   * the Bench picker and the Matrix — which spread `MODELS` — list it twice; only `getModel`, via
   * `MODELS_BY_ID`, is last-wins. That disagreement is the thing to look for.
   */
  it('describes what a duplicate really does to the product', () => {
    const result = compare(
      catalog([one('a/One')]),
      catalog([one('a/One'), one('a/One', { layers: 40 })])
    );

    expect(result.summary).toMatch(/picker/i);
    expect(result.summary).toMatch(/once per row/i);
    expect(result.summary).toMatch(/disagree/i);
    // Two claims that were wrong, in opposite directions. The extra rows are not invisible — they
    // are duplicated on screen — and the count is not always two: a refresh that *resolves* a
    // duplicate leaves one row, and a generator emitting three leaves three.
    expect(result.summary).not.toMatch(/invisible in the product/i);
    expect(result.summary).not.toMatch(/list it twice/i);
  });

  it('says what moved when only a shadowed row changed', () => {
    // Two rows on both sides, and the one the app does *not* load is the one that changed. `byId`
    // is last-wins, so `edited` compares the row that stayed put and finds nothing — leaving a
    // claimed change whose only evidence read "2 → 2".
    const loaded = one('a/One', { layers: 40 });
    const result = compare(
      catalog([one('a/One'), loaded]),
      catalog([one('a/One', { layers: 32, totalParams: 9e9 }), loaded])
    );

    expect(result.changed).toBe(true);
    expect(result.summary).toMatch(/\| `a\/One` \| 2 \| 2 \|/);
    expect(result.summary).toMatch(/beneath the one the app loads/i);
  });

  it('does not blame a shadowed row when the loaded one moved', () => {
    // `[base, v1] -> [base, v2]` is also 2 -> 2, and there the row that changed is the one the
    // app loads — so a same-count report that always says "beneath" is wrong half the time, and
    // wrong about the only thing a reviewer is trying to establish: what the product will show.
    const base = one('a/One');
    const result = compare(
      catalog([base, one('a/One', { layers: 40 })]),
      catalog([base, one('a/One', { layers: 48 })])
    );

    expect(result.changed).toBe(true);
    expect(result.summary).toMatch(/the row the app loads/i);
    expect(result.summary).not.toMatch(/beneath the one the app loads/i);
    // And the fields table still names it, since `byId` did see this one.
    expect(result.summary).toMatch(/\| `a\/One` \| layers \|/);
  });

  it('says so when both the loaded row and a shadowed one moved', () => {
    const result = compare(
      catalog([one('a/One', { layers: 8 }), one('a/One', { layers: 40 })]),
      catalog([one('a/One', { layers: 9 }), one('a/One', { layers: 48 })])
    );

    expect(result.summary).toMatch(/the row the app loads, and one beneath it/i);
  });

  it('describes a resolved duplicate as one row, not two', () => {
    const result = compare(
      catalog([one('a/One'), one('a/One', { layers: 40 })]),
      catalog([one('a/One')])
    );

    expect(result.summary).toMatch(/\| `a\/One` \| 2 \| 1 \|/);
    expect(result.summary).toMatch(/1 fewer row/i);
    expect(result.summary).not.toMatch(/twice/i);
  });

  it('counts three rows as three, not as a duplicate', () => {
    const result = compare(
      catalog([one('a/One')]),
      catalog([one('a/One'), one('a/One'), one('a/One')])
    );

    expect(result.summary).toMatch(/\| `a\/One` \| 1 \| 3 \|/);
    expect(result.summary).toMatch(/2 more rows/i);
  });

  it('does not call a row gaining a duplicate a reordering', () => {
    // `added` and `removed` track presence only, so `[a] → [a, a]` passes both while the id
    // sequences differ in length. Nothing was reordered; a row was duplicated.
    const result = compare(catalog([one('a/One')]), catalog([one('a/One'), one('a/One')]));

    expect(result.changed).toBe(true);
    expect(result.summary).toMatch(/repeated ids/i);
    expect(result.summary).not.toMatch(/\*\*Reordered\*\*/);
  });

  it('does not name a field whose nested keys merely moved', () => {
    // `canonical` treats these rows as identical, so `changedFields` must too — otherwise the
    // summary reports an edit to `attention` that never happened. `attention` is one of the two
    // nested objects the generator writes, which is why it is the field used here.
    const straight = model('a/One');
    const shuffled = {
      ...model('a/One'),
      attention: { core: { headDim: 128, kvHeads: 8, kind: 'gqa' } },
    };

    const result = compare(catalog([straight]), catalog([shuffled]));

    // The two really are key-order variants, not two different rows that compare equal anyway.
    expect(JSON.stringify(straight)).not.toBe(JSON.stringify(shuffled));
    expect(result.changed).toBe(false);
    expect(result.summary).not.toMatch(/attention/);
  });

  it('still names a field whose nested values really moved', () => {
    // The other direction, so key-order blindness cannot be satisfied by ignoring the field.
    const result = compare(
      catalog([model('a/One')]),
      catalog([model('a/One', { attention: { core: { kind: 'gqa', kvHeads: 4, headDim: 128 } } })])
    );

    expect(result.changed).toBe(true);
    expect(result.summary).toMatch(/attention/);
  });
});
