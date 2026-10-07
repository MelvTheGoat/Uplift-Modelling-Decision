/**
 * SVG charts, hand-rolled.
 *
 * Why not a charting library? Three reasons, in order of how much they
 * mattered:
 *
 * 1. **Error bars.** Every chart on this site that reports a measurement has
 *    to show its uncertainty, because the central finding is "the number is
 *    real but the range includes zero". Most chart libraries treat error bars
 *    as a plugin or an afterthought, and a chart that hides uncertainty would
 *    actively misrepresent this study.
 * 2. **No build step and no CDN.** A `<script src="some-cdn">` is a dependency
 *    on someone else's uptime for a page that otherwise needs none.
 * 3. **It is about 400 lines.** Five chart types with fixed requirements is
 *    less code than configuring a library to do the same thing.
 *
 * Everything scales by viewBox, so there is no resize handling: the browser
 * does it. Charts are drawn once into a fixed coordinate space and the SVG
 * stretches.
 */

import { svg, el } from "./dom.js";

/** Nominal drawing space. The viewBox makes these units, not pixels. */
const W = 760;
const H = 300;

/** Colours read from CSS so the charts follow the theme toggle. */
function palette() {
  const style = getComputedStyle(document.documentElement);
  const read = (name, fallback) => style.getPropertyValue(name).trim() || fallback;
  return {
    s1: read("--series-1", "#2a78d6"),
    s2: read("--series-2", "#eb6834"),
    s3: read("--series-3", "#1baf7a"),
    danger: read("--danger", "#e34948"),
    ink: read("--ink", "#0b0b0b"),
    inkSoft: read("--ink-soft", "#52514e"),
    inkFaint: read("--ink-faint", "#8a8880"),
    surface: read("--surface-raised", "#ffffff"),
    sunken: read("--surface-sunken", "#f0efec"),
  };
}

/**
 * Parse a CSS colour into [r, g, b].
 *
 * Handles the forms the stylesheet actually uses — `#rgb`, `#rrggbb` and the
 * `rgb()` that `getComputedStyle` sometimes returns — and falls back to mid
 * grey on anything else rather than throwing inside a render loop.
 */
function parseColour(value) {
  const text = String(value).trim();
  const hex = text.match(/^#([0-9a-f]{3}|[0-9a-f]{6})$/i);
  if (hex) {
    const digits = hex[1];
    const full =
      digits.length === 3
        ? digits
            .split("")
            .map((char) => char + char)
            .join("")
        : digits;
    return [0, 2, 4].map((offset) => parseInt(full.slice(offset, offset + 2), 16));
  }
  const rgb = text.match(/rgba?\(([^)]+)\)/i);
  if (rgb) {
    const parts = rgb[1].split(/[,\s/]+/).filter(Boolean).slice(0, 3);
    return parts.map((part) => Math.round(parseFloat(part)));
  }
  return [128, 128, 128];
}

/** Mix two CSS colours, `amount` being how much of `to` ends up in the result. */
function mixColours(from, to, amount) {
  const a = parseColour(from);
  const b = parseColour(to);
  const t = Math.min(1, Math.max(0, amount));
  const channel = (index) => Math.round(a[index] + (b[index] - a[index]) * t);
  return `rgb(${channel(0)} ${channel(1)} ${channel(2)})`;
}

/** A linear scale from a data range to a pixel range. */
function scale(domain, range) {
  const [d0, d1] = domain;
  const [r0, r1] = range;
  const span = d1 - d0 || 1;
  const fn = (value) => r0 + ((value - d0) / span) * (r1 - r0);
  fn.domain = domain;
  fn.range = range;
  return fn;
}

/** Round, human-looking tick values covering a domain. */
function ticks([lo, hi], target = 5) {
  if (!isFinite(lo) || !isFinite(hi) || lo === hi) return [lo];
  const raw = (hi - lo) / target;
  const magnitude = Math.pow(10, Math.floor(Math.log10(raw)));
  const normalised = raw / magnitude;
  const step =
    (normalised >= 5 ? 10 : normalised >= 2 ? 5 : normalised >= 1 ? 2 : 1) * magnitude;
  const out = [];
  for (let value = Math.ceil(lo / step) * step; value <= hi + step * 1e-9; value += step) {
    out.push(Math.abs(value) < step * 1e-9 ? 0 : value);
  }
  return out;
}

/** Pad a domain so marks do not touch the frame, and always include zero. */
function niceDomain(values, { includeZero = true, pad = 0.08 } = {}) {
  const finite = values.filter((v) => isFinite(v));
  let lo = Math.min(...finite);
  let hi = Math.max(...finite);
  if (includeZero) {
    lo = Math.min(lo, 0);
    hi = Math.max(hi, 0);
  }
  if (lo === hi) {
    lo -= 1;
    hi += 1;
  }
  const margin = (hi - lo) * pad;
  return [lo - margin, hi + margin];
}

/** The shared <svg> root. */
function root(height = H) {
  return svg("svg", {
    class: "chart",
    viewBox: `0 0 ${W} ${height}`,
    preserveAspectRatio: "xMidYMid meet",
    role: "img",
  });
}

/** Horizontal gridlines plus the y-axis labels. */
function yAxis(y, { format = (v) => String(v), title, left = 0 } = {}) {
  const colours = palette();
  const group = svg("g", { class: "grid" });
  for (const value of ticks(y.domain)) {
    const py = y(value);
    group.append(svg("line", { x1: left, x2: W - 12, y1: py, y2: py }));
    group.append(
      svg("text", {
        x: left - 8,
        y: py + 4,
        "text-anchor": "end",
        text: format(value),
      }),
    );
  }
  if (title) {
    group.append(
      svg("text", {
        class: "axis-title",
        transform: `translate(12, ${(y.range[0] + y.range[1]) / 2}) rotate(-90)`,
        "text-anchor": "middle",
        fill: colours.inkSoft,
        text: title,
      }),
    );
  }
  return group;
}

/** The x-axis title, centred under the plot. */
function xTitle(text, height) {
  return svg("text", {
    class: "axis-title",
    x: W / 2,
    y: height - 6,
    "text-anchor": "middle",
    text,
  });
}

// --------------------------------------------------------------- tooltip

const tooltipNode = () => document.getElementById("tooltip");

/**
 * Attach a hover tooltip to a mark.
 *
 * `rows` is an array of [label, value] pairs. Attaching this to every mark is
 * the default rather than an enhancement: a chart in a browser that cannot be
 * interrogated is a picture of a chart.
 */
export function attachTooltip(node, title, rows) {
  const show = (event) => {
    const tip = tooltipNode();
    if (!tip) return;
    tip.replaceChildren(
      el("div", { class: "tooltip-title", text: title }),
      el(
        "dl",
        {},
        rows.flatMap(([label, value]) => [
          el("dt", { text: label }),
          el("dd", { text: value }),
        ]),
      ),
    );
    tip.dataset.visible = "true";
    const pad = 14;
    const box = tip.getBoundingClientRect();
    let x = event.clientX + pad;
    let y = event.clientY + pad;
    if (x + box.width > window.innerWidth - 8) x = event.clientX - box.width - pad;
    if (y + box.height > window.innerHeight - 8) y = event.clientY - box.height - pad;
    tip.style.left = `${Math.max(8, x)}px`;
    tip.style.top = `${Math.max(8, y)}px`;
  };
  const hide = () => {
    const tip = tooltipNode();
    if (tip) tip.dataset.visible = "false";
  };
  node.addEventListener("pointerenter", show);
  node.addEventListener("pointermove", show);
  node.addEventListener("pointerleave", hide);
  node.addEventListener("focus", show);
  node.addEventListener("blur", hide);
  return node;
}

// --------------------------------------------------------------- charts

/**
 * Horizontal bars with error bars: the model scoreboard.
 *
 * The error bar is the point of this chart. A bar twice as long as another
 * means very little when both error bars cross zero, and a reader shown bars
 * alone has no way to know that.
 *
 * @param {Array<{label: string, value: number, low: number, high: number, extra?: Array}>} rows
 */
export function barsWithIntervals(rows, { xTitle: axisTitle, format = (v) => v.toFixed(1) } = {}) {
  const colours = palette();
  const rowHeight = 44;
  const height = Math.max(160, rows.length * rowHeight + 60);
  const node = root(height);
  const left = 150;
  const x = scale(
    niceDomain(rows.flatMap((r) => [r.value, r.low, r.high])),
    [left, W - 24],
  );

  // Vertical gridlines.
  const grid = svg("g", { class: "grid" });
  for (const value of ticks(x.domain)) {
    grid.append(svg("line", { x1: x(value), x2: x(value), y1: 18, y2: height - 42 }));
    grid.append(
      svg("text", {
        x: x(value),
        y: height - 26,
        "text-anchor": "middle",
        text: format(value),
      }),
    );
  }
  node.append(grid);

  rows.forEach((row, index) => {
    const cy = 30 + index * rowHeight + rowHeight / 2 - 10;
    const zero = x(0);
    const value = x(row.value);
    const group = svg("g", { tabindex: "0", role: "listitem" });

    group.append(
      svg("rect", {
        class: "bar",
        x: Math.min(zero, value),
        y: cy - 9,
        width: Math.abs(value - zero),
        height: 18,
        rx: 4,
        fill: row.value >= 0 ? colours.s1 : colours.danger,
      }),
    );
    group.append(
      svg("line", { class: "whisker", x1: x(row.low), x2: x(row.high), y1: cy, y2: cy }),
    );
    for (const end of [row.low, row.high]) {
      group.append(
        svg("line", { class: "whisker", x1: x(end), x2: x(end), y1: cy - 6, y2: cy + 6 }),
      );
    }
    group.append(
      svg("text", {
        x: left - 10,
        y: cy + 4,
        "text-anchor": "end",
        fill: colours.ink,
        text: row.label,
      }),
    );
    attachTooltip(group, row.label, [
      ["Score", format(row.value)],
      ["Could be as low as", format(row.low)],
      ["Could be as high as", format(row.high)],
      ...(row.extra ?? []),
    ]);
    node.append(group);
  });

  node.append(
    svg("line", {
      class: "zero-line",
      x1: x(0),
      x2: x(0),
      y1: 18,
      y2: height - 42,
    }),
  );
  if (axisTitle) node.append(xTitle(axisTitle, height));
  return node;
}

/**
 * Vertical bars with error bars, plus a second series as dots: the decile chart.
 *
 * Bars are coloured by sign — blue where the email helped, red where it hurt.
 * That is a diverging encoding on a quantity where the midpoint genuinely
 * means "nothing happened", which is the one case diverging colour is correct.
 *
 * The dots are what the model predicted. Same unit as the bars, so they
 * legitimately share the axis; the gap between them is the finding.
 */
export function barsWithPredictions(rows, { yTitle, xTitle: axisTitle, format } = {}) {
  const colours = palette();
  const height = 320;
  const node = root(height);
  const left = 54;
  const bottom = height - 52;
  const fmt = format ?? ((v) => v.toFixed(2));

  const y = scale(
    niceDomain(rows.flatMap((r) => [r.low, r.high, r.predicted, r.value])),
    [bottom, 22],
  );
  const bandWidth = (W - left - 24) / rows.length;

  node.append(yAxis(y, { format: fmt, title: yTitle, left }));

  rows.forEach((row, index) => {
    const cx = left + bandWidth * (index + 0.5);
    const barWidth = Math.min(30, bandWidth * 0.55);
    const zero = y(0);
    const top = Math.min(zero, y(row.value));
    const group = svg("g", { tabindex: "0", role: "listitem" });

    group.append(
      svg("rect", {
        class: "bar",
        x: cx - barWidth / 2,
        y: top,
        width: barWidth,
        height: Math.max(1.5, Math.abs(y(row.value) - zero)),
        rx: 3,
        fill: row.value >= 0 ? colours.s1 : colours.danger,
      }),
    );
    group.append(
      svg("line", { class: "whisker", x1: cx, x2: cx, y1: y(row.low), y2: y(row.high) }),
    );
    group.append(
      svg("circle", {
        cx,
        cy: y(row.predicted),
        r: 5,
        fill: colours.s2,
        stroke: colours.surface,
        "stroke-width": 2,
      }),
    );
    group.append(
      svg("text", { x: cx, y: bottom + 18, "text-anchor": "middle", text: row.label }),
    );
    attachTooltip(group, row.tooltipTitle ?? `Group ${row.label}`, [
      ["Measured", fmt(row.value)],
      ["Could be as low as", fmt(row.low)],
      ["Could be as high as", fmt(row.high)],
      ["Model predicted", fmt(row.predicted)],
      ...(row.extra ?? []),
    ]);
    node.append(group);
  });

  node.append(svg("line", { class: "zero-line", x1: left, x2: W - 24, y1: y(0), y2: y(0) }));
  if (axisTitle) node.append(xTitle(axisTitle, height));
  return node;
}

/**
 * Two or more lines on one axis: the profit frontier.
 *
 * One axis, always. Two measures on two scales in a single frame is the most
 * misleading thing a chart can do, so where two quantities need comparing here
 * they are in the same unit — dollars — and the gap between the lines is
 * directly readable as money.
 *
 * @param {Array<{label: string, colour: string, points: Array<{x: number, y: number}>}>} series
 */
export function lines(series, { xTitle: axisTitle, yTitle, formatY, formatX, markers = [] } = {}) {
  const colours = palette();
  const height = 320;
  const node = root(height);
  const left = 62;
  const bottom = height - 52;
  const fmtY = formatY ?? ((v) => v.toFixed(0));
  const fmtX = formatX ?? ((v) => v.toFixed(0));

  const allPoints = series.flatMap((s) => s.points);
  const x = scale(niceDomain(allPoints.map((p) => p.x), { includeZero: false, pad: 0.02 }), [
    left,
    W - 24,
  ]);
  const y = scale(niceDomain(allPoints.map((p) => p.y)), [bottom, 22]);

  node.append(yAxis(y, { format: fmtY, title: yTitle, left }));

  const xGrid = svg("g", { class: "grid" });
  for (const value of ticks(x.domain)) {
    xGrid.append(
      svg("text", {
        x: x(value),
        y: bottom + 18,
        "text-anchor": "middle",
        text: fmtX(value),
      }),
    );
  }
  node.append(xGrid);
  node.append(svg("line", { class: "zero-line", x1: left, x2: W - 24, y1: y(0), y2: y(0) }));

  // Reference markers (the recommendation, the peak) go behind the lines.
  for (const marker of markers) {
    node.append(
      svg("line", {
        x1: x(marker.at),
        x2: x(marker.at),
        y1: 22,
        y2: bottom,
        stroke: marker.colour ?? colours.inkFaint,
        "stroke-width": 2,
        "stroke-dasharray": "3 3",
      }),
    );
    node.append(
      svg("text", {
        x: x(marker.at),
        y: 16,
        "text-anchor": "middle",
        fill: marker.colour ?? colours.inkFaint,
        text: marker.label,
      }),
    );
  }

  for (const line of series) {
    const path = line.points
      .map((point, index) => `${index === 0 ? "M" : "L"}${x(point.x)},${y(point.y)}`)
      .join(" ");
    node.append(
      svg("path", {
        d: path,
        fill: "none",
        stroke: line.colour,
        "stroke-width": 2.5,
        "stroke-linejoin": "round",
        "stroke-linecap": "round",
      }),
    );
  }

  // One hover target per x position, carrying every series' value at that x.
  const xs = [...new Set(allPoints.map((p) => p.x))].sort((a, b) => a - b);
  const step = xs.length > 1 ? (x(xs[1]) - x(xs[0])) / 2 : 12;
  for (const value of xs) {
    const hit = svg("rect", {
      class: "hit",
      x: x(value) - step,
      y: 22,
      width: step * 2,
      height: bottom - 22,
      fill: "transparent",
      tabindex: "0",
    });
    const rows = series.map((line) => {
      const point = line.points.find((p) => p.x === value);
      return [line.label, point ? fmtY(point.y) : "—"];
    });
    attachTooltip(hit, `${fmtX(value)}`, rows);
    node.append(hit);

    for (const line of series) {
      const point = line.points.find((p) => p.x === value);
      if (!point) continue;
      node.append(
        svg("circle", {
          cx: x(point.x),
          cy: y(point.y),
          r: 3,
          fill: line.colour,
          opacity: 0.001,
          "pointer-events": "none",
        }),
      );
    }
  }

  if (axisTitle) node.append(xTitle(axisTitle, height));
  return node;
}

/**
 * A strip of dots against a reference band: the placebo test.
 *
 * The question this chart answers is "is the real number distinguishable from
 * noise?", so the noise is drawn as the thing occupying space and the real
 * result as a single line against it.
 */
export function distributionStrip({ band, reference, points = [], formatX, labels }) {
  const colours = palette();
  const height = 170;
  const node = root(height);
  const left = 24;
  const bottom = 110;

  const values = [...points, band.low, band.high, reference, 0];
  const x = scale(niceDomain(values, { includeZero: true, pad: 0.12 }), [left + 20, W - 30]);

  const grid = svg("g", { class: "grid" });
  for (const value of ticks(x.domain)) {
    grid.append(svg("line", { x1: x(value), x2: x(value), y1: 26, y2: bottom }));
    grid.append(
      svg("text", {
        x: x(value),
        y: bottom + 18,
        "text-anchor": "middle",
        text: (formatX ?? ((v) => v.toFixed(1)))(value),
      }),
    );
  }
  node.append(grid);

  const bandRect = svg("rect", {
    x: x(band.low),
    y: 42,
    width: Math.max(2, x(band.high) - x(band.low)),
    height: bottom - 42,
    fill: colours.inkFaint,
    opacity: 0.22,
    tabindex: "0",
  });
  attachTooltip(bandRect, labels.band, [
    ["Low end", (formatX ?? ((v) => v.toFixed(1)))(band.low)],
    ["High end", (formatX ?? ((v) => v.toFixed(1)))(band.high)],
  ]);
  node.append(bandRect);

  node.append(
    svg("line", {
      x1: x(band.centre),
      x2: x(band.centre),
      y1: 42,
      y2: bottom,
      stroke: colours.inkFaint,
      "stroke-width": 2,
    }),
  );

  for (const point of points) {
    node.append(
      svg("circle", {
        cx: x(point),
        cy: bottom - 18,
        r: 4.5,
        fill: colours.inkFaint,
        stroke: colours.surface,
        "stroke-width": 1.5,
      }),
    );
  }

  node.append(svg("line", { class: "zero-line", x1: x(0), x2: x(0), y1: 26, y2: bottom }));

  const real = svg("line", {
    x1: x(reference),
    x2: x(reference),
    y1: 32,
    y2: bottom + 4,
    stroke: colours.s1,
    "stroke-width": 3.5,
    tabindex: "0",
  });
  attachTooltip(real, labels.reference, [["Score", (formatX ?? ((v) => v.toFixed(1)))(reference)]]);
  node.append(real);
  node.append(
    svg("text", {
      x: x(reference),
      y: 24,
      "text-anchor": "middle",
      fill: colours.s1,
      "font-weight": "600",
      text: labels.reference,
    }),
  );

  return node;
}

/**
 * A heatmap: how deep to email across every cost-and-margin combination.
 *
 * Sequential, single hue, light to dark, because this is a magnitude with no
 * meaningful middle. A diverging or rainbow scale here would invent a midpoint
 * that does not exist.
 */
export function heatmap({ rows, xKey, yKey, valueKey, xFormat, yFormat, valueFormat, tooltip }) {
  const colours = palette();
  const xs = [...new Set(rows.map((r) => r[xKey]))].sort((a, b) => a - b);
  const ys = [...new Set(rows.map((r) => r[yKey]))].sort((a, b) => b - a);

  const cell = Math.min(64, (W - 110) / xs.length);
  const height = ys.length * Math.min(44, cell) + 78;
  const cellH = Math.min(44, cell);
  const node = root(height);
  const left = 86;

  const values = rows.map((r) => r[valueKey]).filter((v) => isFinite(v));
  const lo = Math.min(...values);
  const hi = Math.max(...values);

  // Single-hue ramp: the series-1 blue mixed towards the page surface.
  // Interpolated in JS rather than with CSS `color-mix`, which is only
  // reliably honoured as a CSS property and not as an SVG `fill` attribute.
  const ramp = (value) => {
    const t = hi === lo ? 0.5 : (value - lo) / (hi - lo);
    return mixColours(colours.sunken, colours.s1, 0.18 + t * 0.82);
  };

  ys.forEach((yValue, rowIndex) => {
    node.append(
      svg("text", {
        x: left - 10,
        y: 32 + rowIndex * cellH + cellH / 2 + 4,
        "text-anchor": "end",
        text: yFormat(yValue),
      }),
    );
    xs.forEach((xValue, colIndex) => {
      const row = rows.find((r) => r[xKey] === xValue && r[yKey] === yValue);
      if (!row) return;
      const rect = svg("rect", {
        x: left + colIndex * cell,
        y: 32 + rowIndex * cellH,
        width: cell - 2,
        height: cellH - 2,
        rx: 3,
        fill: ramp(row[valueKey]),
        tabindex: "0",
      });
      attachTooltip(rect, `${xFormat(xValue)} · ${yFormat(yValue)}`, tooltip(row));
      node.append(rect);

      // Direct value labels. The palette's pale steps fall below 3:1 contrast
      // against the surface, which obliges a visible label rather than relying
      // on colour alone to carry the value.
      const t = hi === lo ? 0.5 : (row[valueKey] - lo) / (hi - lo);
      node.append(
        svg("text", {
          class: "value-label",
          x: left + colIndex * cell + (cell - 2) / 2,
          y: 32 + rowIndex * cellH + cellH / 2 + 4,
          "text-anchor": "middle",
          fill: t > 0.55 ? "#ffffff" : colours.ink,
          "pointer-events": "none",
          text: valueFormat(row[valueKey]),
        }),
      );
    });
  });

  xs.forEach((xValue, colIndex) => {
    node.append(
      svg("text", {
        x: left + colIndex * cell + (cell - 2) / 2,
        y: 32 + ys.length * cellH + 18,
        "text-anchor": "middle",
        text: xFormat(xValue),
      }),
    );
  });

  return node;
}

/** The colours the pages use when building a legend, so the two cannot drift. */
export function chartColours() {
  return palette();
}
