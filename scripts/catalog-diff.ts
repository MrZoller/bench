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
 * A row as a string that ignores key order but nothing else.
 *
 * Key order is a serialization artifact — `changedFields` is already blind to it, and a run that
 * emitted the same figures with the keys shuffled has not changed the catalog. Everything else,
 * including array order inside a row, is substance.
 */
function canonical(row: unknown): string {
  return JSON.stringify(row, (_key, value: unknown) =>
    value && typeof value === 'object' && !Array.isArray(value)
      ? Object.fromEntries(
          Object.entries(value as Record<string, unknown>).sort(([a], [b]) => (a < b ? -1 : 1))
        )
      : value
  );
}

/**
 * Every row an id carries, canonicalised and ordered — the unit the comparison actually works in.
 *
 * `byId` is last-wins, so it cannot see a repeated id at all: two rows sharing one id collapse to
 * the later, and a change to the earlier is invisible to `changedFields`, to `added`, and to
 * `removed` alike. Comparing an id's whole row multiset instead makes the duplicate question and
 * the edited question one question, which is what stops the answer to one from having a blind spot
 * the other is expected to cover.
 */
function rowsById(models: Record<string, unknown>[]): Map<string, string[]> {
  const rows = new Map<string, string[]>();
  for (const row of models) {
    const id = String(row.id);
    const list = rows.get(id) ?? [];
    list.push(canonical(row));
    rows.set(id, list);
  }
  for (const list of rows.values()) list.sort();
  return rows;
}

/**
 * Which top-level fields of a model row differ.
 *
 * Deliberately shallow-with-JSON rather than a recursive walk: the rows are flat apart from
 * `attention` and `popularity`, and naming the field is enough to send a reader to the diff.
 * A per-leaf path would be more precise and less readable, and the PR carries the real diff
 * anyway.
 *
 * Through `canonical` so this agrees with every other comparison in the file about what a change
 * is. Raw `JSON.stringify` is key-order sensitive, and `attention` and `popularity` are the two
 * nested objects here — either re-emitted with its keys shuffled and every value intact would be
 * named as a changed field, which is a pull request reporting an edit that did not happen.
 */
function changedFields(a: Record<string, unknown>, b: Record<string, unknown>): string[] {
  const keys = new Set([...Object.keys(a), ...Object.keys(b)]);
  return [...keys].filter((k) => canonical(a[k]) !== canonical(b[k])).sort();
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

  const oldById = byId(before.models);
  const newById = byId(after.models);
  const oldRows = rowsById(before.models);
  const newRows = rowsById(after.models);

  const added = [...newRows.keys()].filter((id) => !oldRows.has(id));
  const removed = [...oldRows.keys()].filter((id) => !newRows.has(id));

  /** Every id either side carries, and whether its rows are the same rows. */
  const allIds = [...new Set([...oldRows.keys(), ...newRows.keys()])];
  const rowsFor = (rows: Map<string, string[]>, id: string) => rows.get(id) ?? [];
  const differing = allIds.filter(
    (id) => canonical(rowsFor(oldRows, id)) !== canonical(rowsFor(newRows, id))
  );

  /**
   * The fields that moved on the row the app actually loads.
   *
   * Asked of `byId`, which is last-wins — deliberately, because that is the row the product uses
   * when an id is repeated. So this stays sensitive to two rows sharing an id merely swapping
   * places: the multiset is unchanged, but which of them wins is not.
   */
  const edited = [...newById.keys()]
    .filter((id) => oldById.has(id))
    .map((id) => ({ id, fields: changedFields(oldById.get(id)!, newById.get(id)!) }))
    .filter((m) => m.fields.length > 0);

  /**
   * Ids the generator wrote more than once, on either side, whose rows moved.
   *
   * The other half of the question `edited` answers, and the half last-wins cannot reach: a change
   * to a *shadowed* row is invisible to `changedFields`, to `added` and to `removed` alike, so
   * without this the catalog could stop tracking a row with nothing said.
   *
   * Asked as a difference rather than as a property of the new document alone. A duplicate that is
   * already committed and faithfully reproduced is not this week's news, and reporting it would
   * open a pull request every Monday for as long as it survives — the weekly-noise failure this
   * file exists to prevent. Reported when one appears, when one is resolved, and when a row
   * changes underneath one.
   *
   * Over every id rather than only the ones both sides share, because a *newly added* seed written
   * twice is absent from the old side entirely: it would be reported as one added model with no
   * mention that the generator emitted it twice.
   */
  const repeated = differing.filter(
    (id) => rowsFor(oldRows, id).length > 1 || rowsFor(newRows, id).length > 1
  );

  /**
   * Failures are compared as a multiset, not by length and not as a set.
   *
   * Length alone let a same-size change of contents through silently — one seed starting to fail
   * as another stops is exactly the week someone needs to be told, and it read as a quiet week.
   * Set differences alone miss the other half: `['a','a','b']` becoming `['a','b','b']` has the
   * same length *and* the same set, so both tests agree nothing happened while the seed that is
   * failing twice has changed. Counting occurrences is the comparison that has neither blind
   * spot, and the two lists below still make the line say *which*.
   */
  const counts = (values: string[]): Map<string, number> => {
    const tally = new Map<string, number>();
    for (const value of values) tally.set(value, (tally.get(value) ?? 0) + 1);
    return tally;
  };
  const differingCounts = (a: Map<string, number>, b: Map<string, number>): boolean =>
    [...new Set([...a.keys(), ...b.keys()])].some((key) => (a.get(key) ?? 0) !== (b.get(key) ?? 0));

  const failuresBefore = counts(before.failures);
  const failuresAfter = counts(after.failures);
  const failuresChanged = differingCounts(failuresBefore, failuresAfter);
  /**
   * Named from the counts, not from `includes`.
   *
   * A set difference is empty in exactly the multiset case above, so a summary built from one
   * would decide on evidence it then declines to state — the failure this whole change is about,
   * one level down. Counts say both that something moved and which seed moved.
   */
  const failureMoves = [...new Set([...failuresBefore.keys(), ...failuresAfter.keys()])]
    .map((seed) => ({
      seed,
      from: failuresBefore.get(seed) ?? 0,
      to: failuresAfter.get(seed) ?? 0,
    }))
    .filter((f) => f.from !== f.to)
    .sort((a, b) => (a.seed < b.seed ? -1 : 1));

  /**
   * The same rows, in a different order.
   *
   * Kept as a change rather than folded away, because the file really did move and a future sort
   * by popularity or name should land visibly. What it must not do is set `changed` *silently* —
   * that is the defect — so it gets a line of its own saying no figure moved.
   *
   * Only asked once the membership is settled: when models are added or removed the order differs
   * as a matter of course, and saying so as well would be noise. Not guarded on `edited`, because
   * a seed-order edit really can land in the same week as an upstream figure change — the summary
   * says both, and the wording below is what stops the two lines contradicting each other.
   *
   * The case this does *not* have to carry is two rows sharing an id swapping places: the id
   * sequence is `[a, a]` either way, but `byId` is last-wins, so the swap changes which row the
   * app loads and `edited` reports it as the fields that moved.
   */
  // Compared as arrays rather than joined strings: any delimiter can in principle appear inside
  // an id or a row, and one that did would make a real reordering compare equal.
  const sequence = (values: string[]) => JSON.stringify(values);
  const idsBefore = before.models.map((m) => String(m.id));
  const idsAfter = after.models.map((m) => String(m.id));
  const rowsBefore = before.models.map(canonical);
  const rowsAfter = after.models.map(canonical);

  // Guarded on the id *multiset*, not on `added`/`removed` being empty. Those track only whether
  // an id is present at all, so `[a] → [a, a]` passes both while the id sequence differs in
  // length — and the catalog gaining a duplicate row is not a reordering of anything. The multiset
  // test subsumes the membership one and closes that case.
  const sameIds = sequence([...idsBefore].sort()) === sequence([...idsAfter].sort());

  const reordered =
    sameIds &&
    (sequence(idsBefore) !== sequence(idsAfter) ||
      // The remaining case, once the ids line up: rows sharing an id permuting among themselves.
      // Only meaningful when the rows are otherwise the same rows — if the multiset moved, that
      // is an edit, and `edited` or `repeated` is the line that says so.
      (sequence([...rowsBefore].sort()) === sequence([...rowsAfter].sort()) &&
        sequence(rowsBefore) !== sequence(rowsAfter)));

  /**
   * Decided from the same evidence the summary is built from, so the two cannot disagree.
   *
   * This was whole-document `JSON.stringify` equality, which is sensitive to array order and key
   * order while every line of the summary below is blind to both. Any reordering that left the
   * figures alone — reordering the seed list in `build-catalog.ts` is enough, since rows are
   * written in seed order — therefore set `changed` with nothing at all to say, and the job
   * committed and opened a pull request whose statement of what moved was a blank line.
   *
   * A reordering is still a change — see `reordered` above, which now says so in the summary
   * rather than leaving it to be inferred from a blank one. What does become a no-op is a row
   * whose *keys* moved with every value intact, which is a serialization artifact and not an
   * ordering anyone chose.
   */
  const changed =
    added.length > 0 ||
    removed.length > 0 ||
    edited.length > 0 ||
    repeated.length > 0 ||
    failuresChanged ||
    reordered;

  const lines: string[] = [];
  /** A blank separator only once there is something to separate from, so no summary opens on one. */
  const gap = () => {
    if (lines.length) lines.push('');
  };

  if (!changed) {
    lines.push('No change: every model resolves to the figures already committed.');
  } else {
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
      gap();
      lines.push('**Changed**');
      lines.push('');
      lines.push('| Model | Fields |');
      lines.push('| --- | --- |');
      for (const { id, fields } of edited) lines.push(`| \`${id}\` | ${fields.join(', ')} |`);
    }

    if (reordered) {
      gap();
      // "No figure changed" only when none did. A seed-order edit landing in the same week as an
      // upstream figure update would otherwise print that sentence directly beneath the table of
      // fields that changed, and a summary that contradicts itself is no better than a blank one.
      lines.push(
        edited.length || repeated.length
          ? '**Reordered**: the models are also written in a different order, beyond the changes above.'
          : [...newRows.values()].some((rows) => rows.length > 1)
            ? '**Reordered**: the same rows, written in a different order — and one id carries ' +
              'more than one row, so the order decides which of them the app loads.'
            : '**Reordered**: the same models, written in a different order. No figure changed.'
      );
    }

    if (repeated.length) {
      gap();
      lines.push('**Repeated ids**');
      lines.push('');
      // A fourth column, because the counts alone can be identical while the rows are not: an id
      // carrying two rows on both sides where the *shadowed* one changed reads "2 → 2" against a
      // claimed change and names nothing that moved. `edited` cannot cover that case — `byId` is
      // last-wins, so the row it compares is precisely the row that did not move.
      lines.push('| Model | Rows before | Rows after | What moved |');
      lines.push('| --- | --- | --- | --- |');
      for (const id of repeated) {
        const was = rowsFor(oldRows, id).length;
        const now = rowsFor(newRows, id).length;
        const plural = (n: number) => (n === 1 ? 'row' : 'rows');

        // Which row moved is a separate question from how many there are, and answering only the
        // second one gets the first wrong half the time: `[base, v1] -> [base, v2]` is also 2 → 2,
        // and there the row that changed is the one the app loads. Asked by taking the loaded row
        // out of each side and comparing what is left.
        const loadedBefore = oldById.has(id) ? canonical(oldById.get(id)) : undefined;
        const loadedAfter = newById.has(id) ? canonical(newById.get(id)) : undefined;
        const loadedMoved = loadedBefore !== loadedAfter;
        const shadowOf = (rows: string[], loaded: string | undefined) => {
          const rest = [...rows];
          const at = loaded === undefined ? -1 : rest.indexOf(loaded);
          if (at >= 0) rest.splice(at, 1);
          return canonical(rest);
        };
        const shadowMoved =
          shadowOf(rowsFor(oldRows, id), loadedBefore) !==
          shadowOf(rowsFor(newRows, id), loadedAfter);

        const moved = loadedMoved
          ? shadowMoved
            ? 'the row the app loads, and one beneath it'
            : 'the row the app loads — its fields are in the table above'
          : shadowMoved
            ? "a row's figures, beneath the one the app loads"
            : 'how many rows carry this id';
        const count =
          was === now
            ? ''
            : now > was
              ? `, ${now - was} more ${plural(now - was)}`
              : `, ${was - now} fewer ${plural(was - now)}`;

        lines.push(`| \`${id}\` | ${was} | ${now} | ${moved}${count} |`);
      }
      lines.push('');
      // Per row rather than "twice", which is wrong in both directions: a refresh that *resolves*
      // a duplicate leaves one row, and a generator emitting three leaves three.
      lines.push(
        '> `MODELS` keeps every row, so the model picker and the comparison grid each list an id ' +
          'once per row it carries, while `getModel` resolves it to whichever row comes last. ' +
          'Wherever that count is above one the two surfaces disagree about the same model, and ' +
          'the fields table above describes only the row that wins.'
      );
    }

    if (failuresChanged) {
      gap();
      lines.push(
        `**Seed failures**: ${before.failures.length} → ${after.failures.length}. ` +
          'A non-empty list means the catalog was written without those models.'
      );
      // Stated per seed, because the totals can be equal while the list is not — and a reader
      // looking at "3 → 3" has been told a decision was made on evidence they cannot see.
      for (const { seed, from, to } of failureMoves) {
        lines.push(
          to === 0
            ? `- No longer failing: ${seed}`
            : from === 0
              ? `- Now failing: ${seed}`
              : `- ${seed}: ${from} → ${to}`
        );
      }
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
