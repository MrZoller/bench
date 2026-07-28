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
    expect(result.summary).toMatch(/duplicate ids/i);
  });
});
