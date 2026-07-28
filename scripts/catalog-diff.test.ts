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
