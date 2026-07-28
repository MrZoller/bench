/**
 * Compares a freshly generated catalog against the committed one, on substance.
 *
 * Exists because `git diff --quiet` is the wrong question. `build-catalog.ts` stamps
 * `generatedAt` on every write, so the file differs after *every* run whether or not a single
 * figure moved — and a weekly job wired to that would open an empty pull request every Sunday
 * for the rest of the project's life. People stop reading a bot that is wrong 51 weeks a year,
 * which means they also stop reading it the week DeepSeek's config changes.
 *
 * So the comparison is over `models` and `failures` only, and the timestamp is treated as what
 * it is: a record of when the fetch ran, not a fact about any model.
 *
 * Usage:
 *   tsx scripts/catalog-diff.ts <committed.json> <fresh.json>
 *   tsx scripts/catalog-diff.ts <a.json> <b.json> --exit-code
 *
 * Writes `changed=true|false` and a markdown `summary` to `$GITHUB_OUTPUT` when running under
 * Actions, and prints the summary either way. Exit status is 0 for both answers — "nothing
 * changed" is a successful run, not a failure, and conflating them is how a scheduled job comes
 * to be permanently red and permanently ignored.
 *
 * `--exit-code` opts into the other convention, spelled and behaving as `git diff --exit-code`
 * does: 0 when the two agree, 1 when they differ. It exists for shell that has to branch on the
 * answer inline, where reading `$GITHUB_OUTPUT` back is not available — and matching git's flag
 * name is deliberate, since a reader already knows what it means.
 */

import { readFileSync, appendFileSync } from 'node:fs';

interface Catalog {
  generatedAt?: string;
  failures?: string[];
  models?: Record<string, unknown>[];
}

/** Every field that decides a figure on screen, which is everything except the timestamp. */
function substance(catalog: Catalog) {
  return { failures: catalog.failures ?? [], models: catalog.models ?? [] };
}

function byId(models: Record<string, unknown>[]): Map<string, Record<string, unknown>> {
  return new Map(models.map((m) => [String(m.id), m]));
}

/**
 * Which top-level fields of a model row differ.
 *
 * Deliberately shallow-with-JSON rather than a recursive walk: the rows are flat apart from
 * `attention` and `popularity`, and naming the field is enough to send a reader to the diff.
 * A per-leaf path would be more precise and less readable, and the PR carries the real diff
 * anyway.
 */
function changedFields(a: Record<string, unknown>, b: Record<string, unknown>): string[] {
  const keys = new Set([...Object.keys(a), ...Object.keys(b)]);
  return [...keys].filter((k) => JSON.stringify(a[k]) !== JSON.stringify(b[k])).sort();
}

/**
 * The whole decision, as a pure function, so it can be tested without a filesystem or a runner.
 *
 * Worth separating: this is what decides whether a pull request appears every Monday, and both
 * of its wrong answers are expensive in different ways. A false positive is weekly noise that
 * trains people to close the bot unread; a false negative is a catalog that silently stops
 * tracking the models it claims to derive from.
 */
export function compare(
  oldCatalog: Catalog,
  newCatalog: Catalog
): {
  changed: boolean;
  summary: string;
} {
  const before = substance(oldCatalog);
  const after = substance(newCatalog);

  const changed = JSON.stringify(before) !== JSON.stringify(after);

  const lines: string[] = [];
  if (!changed) {
    lines.push('No change: every model resolves to the figures already committed.');
  } else {
    const oldById = byId(before.models);
    const newById = byId(after.models);

    const added = [...newById.keys()].filter((id) => !oldById.has(id));
    const removed = [...oldById.keys()].filter((id) => !newById.has(id));
    const edited = [...newById.keys()]
      .filter((id) => oldById.has(id))
      .map((id) => ({ id, fields: changedFields(oldById.get(id)!, newById.get(id)!) }))
      .filter((m) => m.fields.length > 0);

    if (added.length) lines.push(`**Added** (${added.length}): ${added.join(', ')}`);
    /**
     * Called out hard, because it is the one outcome that silently shrinks the product. The
     * generator refuses a partial write without `--allow-partial` for the same reason, but a
     * seed deliberately dropped from the list reaches here legitimately and still deserves a
     * reader's attention rather than a line item.
     */
    if (removed.length) {
      lines.push(`**Removed** (${removed.length}): ${removed.join(', ')}`);
      lines.push('');
      lines.push('> A removed model disappears from the product. Confirm this was intended.');
    }
    if (edited.length) {
      lines.push('');
      lines.push('**Changed**');
      lines.push('');
      lines.push('| Model | Fields |');
      lines.push('| --- | --- |');
      for (const { id, fields } of edited) lines.push(`| \`${id}\` | ${fields.join(', ')} |`);
    }

    if (before.failures.length !== after.failures.length) {
      lines.push('');
      lines.push(
        `**Seed failures**: ${before.failures.length} → ${after.failures.length}. ` +
          'A non-empty list means the catalog was written without those models.'
      );
    }
  }

  return { changed, summary: lines.join('\n') };
}

function main() {
  const args = process.argv.slice(2);
  const exitCode = args.includes('--exit-code');
  const [oldPath, newPath] = args.filter((a) => !a.startsWith('--'));
  if (!oldPath || !newPath) {
    console.error('usage: tsx scripts/catalog-diff.ts <committed.json> <fresh.json> [--exit-code]');
    process.exit(2);
  }

  const { changed, summary } = compare(
    JSON.parse(readFileSync(oldPath, 'utf8')) as Catalog,
    JSON.parse(readFileSync(newPath, 'utf8')) as Catalog
  );
  console.log(summary);

  if (process.env.GITHUB_OUTPUT) {
    /**
     * A heredoc delimiter rather than `summary=...`, because the summary is multi-line and the
     * `key=value` form silently truncates at the first newline — which would have shipped a PR
     * body consisting of the word "**Added**" and nothing else.
     */
    appendFileSync(
      process.env.GITHUB_OUTPUT,
      `changed=${changed}\nsummary<<CATALOG_EOF\n${summary}\nCATALOG_EOF\n`
    );
  }

  if (exitCode && changed) process.exit(1);
}

// Guarded so the pure half above can be imported by a test without the script running itself.
if (process.argv[1] && /catalog-diff\.ts$/.test(process.argv[1])) main();
