/**
 * A CSV parser.
 *
 * Yes, really. `split(",")` is the classic wrong answer here: it breaks on the
 * first quoted field containing a comma, which in a marketing export is
 * usually the very first row ("Smith, J."). It also breaks on quoted newlines,
 * on escaped quotes, and on files exported from Excel with a byte-order mark.
 *
 * This handles all four, because the files people will actually upload are
 * exports from systems nobody here controls.
 */

/**
 * Parse CSV text into a header array and an array of row arrays.
 *
 * The parser is a small state machine over the characters. It accepts both
 * `\n` and `\r\n` line endings, `""` as an escaped quote inside a quoted
 * field, and a leading UTF-8 BOM.
 *
 * @param {string} text Raw file contents.
 * @param {string} [delimiter] Field separator. Sniffed if not given.
 * @returns {{header: string[], rows: string[][], delimiter: string}}
 */
export function parseCsv(text, delimiter) {
  // Excel writes a byte-order mark; left in place it becomes part of the first
  // column's name, and the column dropdown then shows a header that does not
  // match what the user sees in their spreadsheet.
  let source = text.charCodeAt(0) === 0xfeff ? text.slice(1) : text;
  const sep = delimiter ?? sniffDelimiter(source);

  const rows = [];
  let row = [];
  let field = "";
  let quoted = false;
  let index = 0;

  while (index < source.length) {
    const char = source[index];

    if (quoted) {
      if (char === '"') {
        if (source[index + 1] === '"') {
          field += '"';
          index += 2;
          continue;
        }
        quoted = false;
        index += 1;
        continue;
      }
      field += char;
      index += 1;
      continue;
    }

    if (char === '"' && field === "") {
      quoted = true;
      index += 1;
      continue;
    }
    if (char === sep) {
      row.push(field);
      field = "";
      index += 1;
      continue;
    }
    if (char === "\n" || char === "\r") {
      row.push(field);
      field = "";
      rows.push(row);
      row = [];
      index += source[index] === "\r" && source[index + 1] === "\n" ? 2 : 1;
      continue;
    }
    field += char;
    index += 1;
  }

  if (field !== "" || row.length) {
    row.push(field);
    rows.push(row);
  }

  // Trailing blank lines are normal in exports and are not data.
  while (rows.length && rows[rows.length - 1].every((cell) => cell.trim() === "")) {
    rows.pop();
  }
  if (!rows.length) return { header: [], rows: [], delimiter: sep };

  const header = rows[0].map((name) => name.trim());
  return { header, rows: rows.slice(1), delimiter: sep };
}

/**
 * Guess the delimiter from the first line.
 *
 * Counts candidates outside quoted sections and takes the most frequent. Comma
 * wins ties, because it is the common case and a wrong guess is recoverable —
 * the user sees nonsense column names immediately.
 */
export function sniffDelimiter(text) {
  const firstLine = text.slice(0, text.indexOf("\n") === -1 ? text.length : text.indexOf("\n"));
  const candidates = [",", ";", "\t", "|"];
  let best = ",";
  let bestCount = 0;
  for (const candidate of candidates) {
    let count = 0;
    let quoted = false;
    for (const char of firstLine) {
      if (char === '"') quoted = !quoted;
      else if (char === candidate && !quoted) count += 1;
    }
    if (count > bestCount) {
      best = candidate;
      bestCount = count;
    }
  }
  return best;
}

/**
 * Work out what each column contains, so the mapping UI can pre-select sensibly.
 *
 * @param {string[]} header
 * @param {string[][]} rows
 * @returns {Array<{name: string, index: number, kind: string, unique: number, values: Set<string>, numeric: boolean}>}
 */
export function profileColumns(header, rows) {
  // Sampling rather than scanning everything: a 500k-row file would otherwise
  // freeze the tab for several seconds before the user has chosen anything.
  const sample = rows.length > 5000 ? rows.filter((_, i) => i % Math.ceil(rows.length / 5000) === 0) : rows;

  return header.map((name, index) => {
    const values = new Set();
    let numericCount = 0;
    let nonEmpty = 0;
    for (const row of sample) {
      const raw = (row[index] ?? "").trim();
      if (raw === "") continue;
      nonEmpty += 1;
      if (values.size < 50) values.add(raw);
      if (raw !== "" && Number.isFinite(Number(raw))) numericCount += 1;
    }
    const numeric = nonEmpty > 0 && numericCount / nonEmpty > 0.95;
    const unique = values.size;

    let kind = "other";
    if (unique <= 2 && unique > 0) kind = "binary";
    else if (numeric) kind = "numeric";
    else if (unique <= 20) kind = "category";

    return { name, index, kind, unique, values, numeric };
  });
}

/**
 * Guess which column is which, by name first and shape second.
 *
 * Only ever a starting point — the mapping dropdowns are always shown, because
 * a confident wrong guess is worse than no guess.
 */
export function guessMapping(columns) {
  /*
   * Try patterns in PRIORITY order, scanning all columns for each, rather than
   * scanning columns once and taking whatever matches first.
   *
   * The difference is not academic. A file with both `past_spend` (what they
   * spent before the campaign — a covariate) and `revenue` (what the campaign
   * earned — the outcome) will hand you `past_spend` under the naive version,
   * purely because it appears earlier in the file. Mapping a pre-campaign
   * column as the revenue outcome produces a confident, entirely wrong profit
   * figure, and nothing downstream can detect it.
   */
  const byName = (patterns, filter = () => true) => {
    for (const pattern of patterns) {
      const hit = columns.find((column) => pattern.test(column.name) && filter(column));
      if (hit) return hit.name;
    }
    return null;
  };

  // Columns whose name marks them as measured BEFORE the campaign. These are
  // covariates whatever else they look like, and must never be auto-selected
  // as an outcome.
  const isPrePeriod = (column) =>
    /(^|_)(past|prior|pre|previous|historic|baseline|last|lifetime|ltv)(_|$)/i.test(column.name);

  const treatment =
    byName([/^treatment$/i, /^treated$/i, /^group$/i, /^variant$/i, /^arm$/i, /^segment$/i, /e.?mail/i], (c) => c.kind === "binary") ??
    columns.find((column) => column.kind === "binary")?.name ??
    null;

  const outcome =
    byName([/^outcome$/i, /^conver/i, /^purchase/i, /^bought$/i, /^response$/i, /^y$/i], (c) => c.kind === "binary") ??
    columns.find((column) => column.kind === "binary" && column.name !== treatment)?.name ??
    null;

  // Revenue first, then the vaguer words. `spend` is last among them because
  // it is the one most likely to name a pre-campaign column.
  const spend = byName(
    [/^revenue$/i, /revenue/i, /^sales$/i, /^order_value$/i, /^amount$/i, /^value$/i, /spend/i],
    (column) => column.numeric && !isPrePeriod(column),
  );
  const score = byName(
    [/uplift/i, /^score$/i, /^tau$/i, /^cate$/i, /predicted/i],
    (column) => column.numeric,
  );

  return { treatment, outcome, spend, score };
}
