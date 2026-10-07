/**
 * DOM helpers.
 *
 * The pages build their markup in JavaScript, so without these every page
 * turns into a wall of `document.createElement` noise. There is no templating
 * library here on purpose: adding a framework to render eight static documents
 * would be more machinery than the job needs, and it would mean a build step,
 * which is the thing that makes a site stop deploying three years later.
 *
 * One rule worth knowing: `html` below sets innerHTML, so it must only ever be
 * called with strings this repository wrote. None of the content on this site
 * comes from a user or a URL, so that holds — but if you add an input that
 * echoes something a visitor typed, use `text()` for it, not `html`.
 */

/** Create an element with attributes and children in one call. */
export function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key === "html") node.innerHTML = value;
    else if (key === "dataset") Object.assign(node.dataset, value);
    else if (key.startsWith("on") && typeof value === "function") {
      node.addEventListener(key.slice(2).toLowerCase(), value);
    } else node.setAttribute(key, value === true ? "" : String(value));
  }
  for (const child of [].concat(children)) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

/** Create an SVG element. SVG needs its own namespace or nothing renders. */
export function svg(tag, attrs = {}, children = []) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "text") node.textContent = value;
    else if (key === "dataset") Object.assign(node.dataset, value);
    else node.setAttribute(key, String(value));
  }
  for (const child of [].concat(children)) {
    if (child) node.append(child);
  }
  return node;
}

/** Escape text for interpolation into an `html:` string. */
export function escapeHtml(value) {
  return String(value).replace(
    /[&<>"']/g,
    (char) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[char],
  );
}

/**
 * Markdown-ish prose: paragraphs, bold, inline code, links, lists and tables.
 *
 * This is NOT a markdown parser and is not trying to be. It handles the
 * handful of constructs the copy on this site actually uses, because pulling
 * in a parser to render bold text would be a 40 KB dependency for four
 * asterisks. If you need something it does not cover, build the element.
 */
export function prose(markdown) {
  const wrapper = el("div");
  const blocks = markdown.trim().split(/\n{2,}/);

  for (const block of blocks) {
    const trimmed = block.trim();
    if (!trimmed) continue;

    if (trimmed.startsWith("|")) {
      wrapper.append(proseTable(trimmed));
      continue;
    }
    if (/^[-*] /m.test(trimmed) && trimmed.split("\n").every((line) => /^[-*] /.test(line.trim()))) {
      wrapper.append(
        el(
          "ul",
          {},
          trimmed
            .split("\n")
            .map((line) => el("li", { html: inline(line.trim().replace(/^[-*] /, "")) })),
        ),
      );
      continue;
    }
    if (trimmed.startsWith("### ")) {
      wrapper.append(el("h3", { html: inline(trimmed.slice(4)) }));
      continue;
    }
    if (trimmed.startsWith("## ")) {
      wrapper.append(el("h2", { html: inline(trimmed.slice(3)) }));
      continue;
    }
    wrapper.append(el("p", { html: inline(trimmed.replace(/\n/g, " ")) }));
  }
  return wrapper;
}

/** Inline formatting: bold, code, links. Everything else is escaped. */
function inline(text) {
  return escapeHtml(text)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2">$1</a>');
}

/** Render a pipe table, using the second row only to detect alignment. */
function proseTable(block) {
  const lines = block
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);
  const cells = (line) =>
    line
      .replace(/^\||\|$/g, "")
      .split("|")
      .map((cell) => cell.trim());

  const header = cells(lines[0]);
  const body = lines.slice(lines[1] && /^[\s|:-]+$/.test(lines[1]) ? 2 : 1).map(cells);

  return el("div", { class: "table-wrap prose-table" }, [
    el("table", {}, [
      header.some(Boolean)
        ? el("thead", {}, [el("tr", {}, header.map((cell) => el("th", { html: inline(cell) })))])
        : null,
      el(
        "tbody",
        {},
        body.map((row) => el("tr", {}, row.map((cell) => el("td", { html: inline(cell) })))),
      ),
    ]),
  ]);
}

/** A coloured callout. Kind is one of info, good, warn, bad, or plain. */
export function note(kind, markdown) {
  const classes = kind === "plain" ? "note" : `note note--${kind}`;
  return el("div", { class: classes }, [prose(markdown)]);
}

/** A collapsed "what am I looking at?" block. */
export function details(summaryText, markdownOrNode) {
  const body =
    typeof markdownOrNode === "string" ? prose(markdownOrNode) : markdownOrNode;
  return el("details", {}, [
    el("summary", { text: summaryText }),
    el("div", { class: "details-body" }, [body]),
  ]);
}

/** A single metric card. */
export function metric({ label, value, note: footnote, tone }) {
  const toneClass = tone ? ` metric-value--${tone}` : "";
  return el("div", { class: "metric" }, [
    el("span", { class: "metric-label", text: label }),
    el("span", { class: `metric-value${toneClass}`, text: value }),
    footnote ? el("span", { class: "metric-note", text: footnote }) : null,
  ]);
}

/** A row of metric cards. */
export function metrics(items) {
  return el("div", { class: "metrics" }, items.map(metric));
}

/**
 * A data table.
 *
 * @param {object} spec
 * @param {string[]} spec.columns Header labels.
 * @param {Array<Array<string|Node>>} spec.rows Cell contents.
 * @param {string} [spec.caption] Shown above the table.
 * @param {number[]} [spec.numeric] Indices of right-aligned columns.
 */
export function table({ columns, rows, caption, numeric = [] }) {
  const isNum = new Set(numeric);
  return el("div", { class: "table-wrap" }, [
    el("table", {}, [
      caption ? el("caption", { text: caption }) : null,
      el(
        "thead",
        {},
        [
          el(
            "tr",
            {},
            columns.map((label, index) =>
              el("th", { class: isNum.has(index) ? "num" : null, scope: "col", text: label }),
            ),
          ),
        ],
      ),
      el(
        "tbody",
        {},
        rows.map((row) =>
          el(
            "tr",
            {},
            row.map((cell, index) => {
              const attrs = { class: isNum.has(index) ? "num" : null };
              return cell instanceof Node
                ? el("td", attrs, [cell])
                : el("td", { ...attrs, html: inline(String(cell)) });
            }),
          ),
        ),
      ),
    ]),
  ]);
}

/** Wrap a table in a disclosure, so the chart leads and the numbers back it up. */
export function tableDetails(summaryText, spec) {
  return details(summaryText, table(spec));
}

/** A pass/fail pill. */
export function pill(passed, labels = ["holds up", "does not hold up"]) {
  return el("span", {
    class: `pill pill--${passed ? "good" : "bad"}`,
    text: passed ? labels[0] : labels[1],
  });
}

/** A figure: chart, legend and caption, in the frame the CSS expects. */
export function figure({ legend, chart, caption }) {
  return el("figure", {}, [
    el("div", { class: "chart-frame" }, [
      legend ? legendRow(legend) : null,
      chart,
    ]),
    caption ? el("figcaption", { html: inline(caption) }) : null,
  ]);
}

/** A legend row. Items are {label, colour, shape}. */
export function legendRow(items) {
  return el(
    "div",
    { class: "legend" },
    items.map((item) =>
      el("span", { class: "legend-item" }, [
        el("span", {
          class: `legend-swatch${item.shape ? ` legend-swatch--${item.shape}` : ""}`,
          style:
            item.shape === "dash"
              ? `color:${item.colour}`
              : `background:${item.colour}`,
        }),
        item.label,
      ]),
    ),
  );
}

/** A segmented control. Calls onChange with the chosen value. */
export function segmented({ options, value, onChange, label }) {
  const wrap = el("div", { class: "segmented", role: "group", "aria-label": label });
  for (const option of options) {
    wrap.append(
      el("button", {
        type: "button",
        "aria-pressed": String(option.value === value),
        text: option.label,
        onclick: () => onChange(option.value),
      }),
    );
  }
  return wrap;
}

/** A labelled slider with a live readout. Calls onInput with the number. */
export function slider({ label, min, max, step, value, format, help, onInput }) {
  const readout = el("span", { class: "control-readout", text: format(value) });
  const input = el("input", {
    type: "range",
    min,
    max,
    step,
    value,
    "aria-label": label,
    oninput: (event) => {
      const next = Number(event.target.value);
      readout.textContent = format(next);
      onInput(next);
    },
  });
  return el("div", { class: "control" }, [
    el("div", { class: "control-head" }, [
      el("span", { class: "control-label", text: label }),
      readout,
    ]),
    input,
    help ? el("span", { class: "control-help", text: help }) : null,
  ]);
}

/** A labelled number field. Calls onInput with the number. */
export function numberField({ label, min, max, step, value, help, onInput }) {
  return el("div", { class: "field" }, [
    el("label", { text: label, for: `f-${label.replace(/\W+/g, "-")}` }),
    el("input", {
      type: "number",
      id: `f-${label.replace(/\W+/g, "-")}`,
      min,
      max,
      step,
      value,
      oninput: (event) => onInput(Number(event.target.value)),
    }),
    help ? el("span", { class: "control-help", text: help }) : null,
  ]);
}

/** Replace a node's children in one go. */
export function replaceChildren(node, children) {
  node.replaceChildren(...[].concat(children).filter(Boolean));
}
