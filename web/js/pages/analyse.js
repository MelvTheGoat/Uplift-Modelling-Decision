/**
 * Run the study's analysis on your own campaign.
 *
 * This is the page that makes the site a tool rather than a write-up. The file
 * is read with the browser's FileReader and analysed in the tab; nothing is
 * uploaded, because nothing needs to be, and a page that asks a marketer to
 * post their customer list to a stranger's server deserves to be closed.
 *
 * The flow is deliberately three separate steps — load, map, analyse — rather
 * than one magic button. Column guessing is a convenience and it will
 * sometimes be wrong; showing the guess and letting it be corrected is the
 * difference between a tool and a slot machine.
 */

import {
  el,
  prose,
  note,
  details,
  table,
  figure,
  metrics,
  slider,
  pill,
  replaceChildren,
} from "../dom.js";
import { barsWithPredictions, lines, distributionStrip, chartColours } from "../charts.js";
import { parseCsv, profileColumns, guessMapping } from "../csv.js";
import { analyse, randomTargetingProfit } from "../analysis.js";
import { count, money, num, per100, percent, pValue, tinyP } from "../format.js";

/** How many rows to analyse. Past this the browser stalls long enough to look broken. */
const MAX_ROWS = 200000;

export function render() {
  const state = {
    header: [],
    rows: [],
    columns: [],
    mapping: { treatment: null, outcome: null, spend: null, score: null, covariates: [] },
    treatedValue: null,
    outcomeValue: null,
    cost: 0.1,
    margin: 0.3,
    fileName: null,
  };

  const mappingHost = el("div");
  const resultsHost = el("div");

  const onLoaded = (text, fileName) => {
    const { header, rows } = parseCsv(text);
    if (!header.length || !rows.length) {
      replaceChildren(mappingHost, [
        note("bad", "That file has no readable rows. Is it definitely a CSV?"),
      ]);
      return;
    }
    state.header = header;
    state.rows = rows.slice(0, MAX_ROWS);
    state.columns = profileColumns(header, state.rows);
    state.mapping = { ...guessMapping(state.columns), covariates: [] };
    state.treatedValue = null;
    state.outcomeValue = null;
    state.fileName = fileName;
    drawMapping();
    replaceChildren(resultsHost, []);
  };

  const drawMapping = () => replaceChildren(mappingHost, mappingSection(state, drawMapping, run));

  const run = () => {
    replaceChildren(resultsHost, [
      el("div", { class: "loading", text: "Analysing… this takes a few seconds on a large file." }),
    ]);
    // Yield to the browser so the message above actually paints before the
    // main thread is tied up. Without this the page simply freezes and the
    // user assumes it has crashed.
    setTimeout(() => {
      try {
        replaceChildren(resultsHost, resultsSection(state, run));
        resultsHost.scrollIntoView({ behavior: "smooth", block: "start" });
      } catch (error) {
        replaceChildren(resultsHost, [
          note("bad", `**That did not work.** ${error.message}`),
        ]);
        console.error(error);
      }
    }, 30);
  };

  return [
    el("h1", { text: "Analyse your own campaign" }),
    el("p", {
      class: "lede",
      text: "Upload a CSV of a randomised campaign and get the whole analysis back.",
    }),

    note(
      "info",
      `
**Your file never leaves this browser.** There is no server to send it to —
the page is a set of static files, and the analysis runs in this tab. You can
disconnect from the internet after the page loads and it will still work.
      `,
    ),

    prose(`
### What you need

A CSV with one row per customer and, at minimum, two columns:

- **Who got the campaign** — a yes/no column. Any two values work: 1/0,
  true/false, "Mens E-Mail"/"No E-Mail". You pick which one means "got it".
- **What happened** — another yes/no column. Did they buy, visit, renew,
  convert.

Those two get you the honest before-and-after and a read on what your test
could and could not have detected. Three optional columns unlock the rest:

- **Revenue** per customer — turns the analysis into money.
- **An uplift score** per customer, from a model you fit elsewhere — unlocks
  the whole targeting evaluation: ranking quality, decile table, profit
  frontier, shuffle test, sleeping dogs.
- **Customer characteristics** — lets the tool check your randomisation
  actually came out balanced.

### The one thing that must be true

**Assignment must have been random.** If somebody chose who got the campaign —
by segment, by score, by seniority, by anything — then none of these numbers
mean what they appear to mean. The study puts a figure on how bad that gets:
on invented data where the truth is known, letting a marketing team pick the
recipients made the simple before-and-after overstate the real effect by
**7.5 times**, with nothing in the output looking wrong.

No tool can detect this for you. Only you know how the split was made.
    `),

    uploadSection(onLoaded),
    mappingHost,
    resultsHost,
  ];
}

// ------------------------------------------------------------------ upload

function uploadSection(onLoaded) {
  const input = el("input", {
    type: "file",
    accept: ".csv,.tsv,.txt,text/csv",
    id: "csv-input",
    onchange: (event) => {
      const file = event.target.files?.[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = () => onLoaded(String(reader.result), file.name);
      reader.onerror = () => window.alert("That file could not be read.");
      reader.readAsText(file);
    },
  });

  return el("div", { class: "controls" }, [
    el("div", { class: "control" }, [
      el("span", { class: "control-label", text: "Your campaign file" }),
      input,
      el("span", {
        class: "control-help",
        text: "CSV, tab-separated or semicolon-separated. Up to 200,000 rows.",
      }),
    ]),
    el("div", { class: "control" }, [
      el("span", { class: "control-label", text: "No file to hand?" }),
      el("button", {
        class: "icon-button",
        type: "button",
        text: "Load a sample campaign",
        onclick: async (event) => {
          const button = event.target;
          button.textContent = "Loading…";
          button.disabled = true;
          try {
            const response = await fetch("sample-campaign.csv");
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            onLoaded(await response.text(), "sample-campaign.csv");
          } catch (error) {
            window.alert(`Could not load the sample: ${error.message}`);
          } finally {
            button.textContent = "Load a sample campaign";
            button.disabled = false;
          }
        },
      }),
      el("span", {
        class: "control-help",
        text:
          "12,000 invented customers with deliberately awkward column names, so " +
          "you can see the whole thing work before committing your own data.",
      }),
    ]),
  ]);
}

// ------------------------------------------------------------------ mapping

function mappingSection(state, redraw, run) {
  const { columns, mapping } = state;

  const picker = (label, key, filter, help, allowNone = true) => {
    const options = columns.filter(filter);
    return el("div", { class: "field" }, [
      el("label", { text: label, for: `map-${key}` }),
      el(
        "select",
        {
          id: `map-${key}`,
          onchange: (event) => {
            mapping[key] = event.target.value || null;
            if (key === "treatment") state.treatedValue = null;
            if (key === "outcome") state.outcomeValue = null;
            redraw();
          },
        },
        [
          allowNone ? el("option", { value: "", text: "— none —" }) : null,
          ...options.map((column) =>
            el("option", {
              value: column.name,
              text: `${column.name}  (${column.kind})`,
              selected: mapping[key] === column.name,
            }),
          ),
        ],
      ),
      el("span", { class: "control-help", text: help }),
    ]);
  };

  const valuePicker = (label, columnName, stateKey, help) => {
    const column = columns.find((c) => c.name === columnName);
    if (!column) return null;
    const values = [...column.values].sort();
    if (!state[stateKey]) {
      // Default to the value that reads as "yes": 1, or whichever sorts last
      // for text ("yes" after "no", "treated" after "control").
      state[stateKey] = values.includes("1") ? "1" : values[values.length - 1];
    }
    return el("div", { class: "field" }, [
      el("label", { text: label, for: `val-${stateKey}` }),
      el(
        "select",
        {
          id: `val-${stateKey}`,
          onchange: (event) => {
            state[stateKey] = event.target.value;
          },
        },
        values.map((value) =>
          el("option", { value, text: value, selected: state[stateKey] === value }),
        ),
      ),
      el("span", { class: "control-help", text: help }),
    ]);
  };

  const covariateBox = el(
    "div",
    { class: "controls" },
    [
      el("div", { class: "control" }, [
        el("span", { class: "control-label", text: "Customer characteristics to balance-check" }),
        el("span", {
          class: "control-help",
          text:
            "Tick anything measured BEFORE the campaign went out. Anything measured " +
            "after is affected by the campaign, and checking it proves nothing.",
        }),
        el(
          "div",
          { style: "display:flex;flex-wrap:wrap;gap:0.4rem 1rem;margin-top:0.6rem" },
          columns
            .filter(
              (column) =>
                column.numeric &&
                ![mapping.treatment, mapping.outcome, mapping.spend, mapping.score].includes(
                  column.name,
                ),
            )
            .map((column) =>
              el("label", { style: "display:inline-flex;gap:0.35rem;align-items:center;font-size:0.875rem" }, [
                el("input", {
                  type: "checkbox",
                  checked: mapping.covariates.includes(column.name),
                  onchange: (event) => {
                    if (event.target.checked) mapping.covariates.push(column.name);
                    else mapping.covariates = mapping.covariates.filter((n) => n !== column.name);
                  },
                }),
                column.name,
              ]),
            ),
        ),
      ]),
    ],
  );

  const ready = Boolean(mapping.treatment && mapping.outcome);

  return [
    el("h2", { text: `Step 2: which column is which?` }),
    el("p", {
      text:
        `Read ${count(state.rows.length)} rows and ${state.header.length} columns from ` +
        `${state.fileName}. Best guesses are filled in below — check them.`,
    }),

    el("div", { class: "field-row" }, [
      picker(
        "Who got the campaign *",
        "treatment",
        () => true,
        "The yes/no column saying who was contacted.",
        false,
      ),
      mapping.treatment
        ? valuePicker(
            "…and which value means they got it",
            mapping.treatment,
            "treatedValue",
            "Everything else counts as the control group.",
          )
        : null,
      picker(
        "What happened *",
        "outcome",
        () => true,
        "The yes/no column saying whether they converted.",
        false,
      ),
      mapping.outcome
        ? valuePicker(
            "…and which value means yes",
            mapping.outcome,
            "outcomeValue",
            "Everything else counts as “did not convert”.",
          )
        : null,
    ]),

    el("div", { class: "field-row" }, [
      picker(
        "Revenue per customer",
        "spend",
        (column) => column.numeric,
        "Optional. Without it, profit is measured in conversions rather than money.",
      ),
      picker(
        "Uplift score per customer",
        "score",
        (column) => column.numeric,
        "Optional, and the one that unlocks everything about targeting.",
      ),
    ]),

    covariateBox,

    ready
      ? el("button", {
          class: "icon-button",
          type: "button",
          style: "font-size:1rem;padding:0.6rem 1.2rem",
          text: "Analyse this campaign",
          onclick: run,
        })
      : note("warn", "Pick a campaign column and an outcome column to continue."),
  ];
}

// ------------------------------------------------------------------ results

/** Pull the mapped columns out of the raw rows as typed arrays. */
function extract(state) {
  const { rows, columns, mapping } = state;
  const indexOf = (name) => columns.find((column) => column.name === name)?.index ?? -1;

  const treatmentIdx = indexOf(mapping.treatment);
  const outcomeIdx = indexOf(mapping.outcome);
  const spendIdx = mapping.spend ? indexOf(mapping.spend) : -1;
  const scoreIdx = mapping.score ? indexOf(mapping.score) : -1;

  const keep = [];
  for (let i = 0; i < rows.length; i += 1) {
    const row = rows[i];
    const t = (row[treatmentIdx] ?? "").trim();
    const y = (row[outcomeIdx] ?? "").trim();
    if (t === "" || y === "") continue;
    if (scoreIdx >= 0 && !Number.isFinite(Number(row[scoreIdx]))) continue;
    keep.push(i);
  }

  const n = keep.length;
  const treatment = new Int32Array(n);
  const outcome = new Float64Array(n);
  const spend = spendIdx >= 0 ? new Float64Array(n) : null;
  const score = scoreIdx >= 0 ? new Float64Array(n) : null;

  for (let k = 0; k < n; k += 1) {
    const row = rows[keep[k]];
    treatment[k] = (row[treatmentIdx] ?? "").trim() === state.treatedValue ? 1 : 0;
    outcome[k] = (row[outcomeIdx] ?? "").trim() === state.outcomeValue ? 1 : 0;
    if (spend) spend[k] = Number(row[spendIdx]) || 0;
    if (score) score[k] = Number(row[scoreIdx]);
  }

  const covariates = mapping.covariates.map((name) => {
    const idx = indexOf(name);
    const values = new Float64Array(n);
    for (let k = 0; k < n; k += 1) values[k] = Number(rows[keep[k]][idx]) || 0;
    return { name, values };
  });

  return { treatment, outcome, spend, score, covariates, dropped: rows.length - n };
}

function resultsSection(state, rerun) {
  const data = extract(state);
  const colours = chartColours();

  const nTreated = data.treatment.reduce((total, value) => total + value, 0);
  if (nTreated === 0 || nTreated === data.treatment.length) {
    return [
      note(
        "bad",
        `**Everybody is in the same group.** With "${state.treatedValue}" as the
treated value, ${nTreated === 0 ? "nobody" : "everybody"} counts as contacted.
There is no comparison to make. Check the "which value means they got it"
dropdown above.`,
      ),
    ];
  }

  // Bootstrap and placebo both re-score the whole file repeatedly. Scaling the
  // replicate count down on big files keeps this to a few seconds rather than
  // a minute; the page says what it used so the number is never a mystery.
  const nBoot = Math.max(40, Math.min(300, Math.floor(1.5e6 / data.outcome.length)));
  const nPlacebo = Math.max(10, Math.min(30, Math.floor(2e5 / data.outcome.length)));

  const result = analyse({
    outcome: data.outcome,
    treatment: data.treatment,
    spend: data.spend,
    score: data.score,
    covariates: data.covariates,
    costPerContact: state.cost,
    marginRate: state.margin,
    nBoot,
    nPlacebo,
  });

  const out = [
    el("h2", { text: "Step 3: what your campaign shows" }),
    data.dropped
      ? note(
          "plain",
          `${count(data.dropped)} rows were skipped because the campaign, outcome
or score column was empty or non-numeric. ${count(result.n)} rows were analysed.`,
        )
      : null,
    ...didItWork(result, state),
  ];

  if (result.balance) out.push(...balanceBlock(result));
  if (result.qini) out.push(...targetingBlock(result, data, state, colours, nBoot, nPlacebo, rerun));

  if (result.skipped.length) {
    out.push(
      el("h3", { text: "What this file could not answer" }),
      el("ul", {}, result.skipped.map((reason) => el("li", { text: reason }))),
    );
  }
  return out;
}

function didItWork(result, state) {
  const naive = result.naive;
  const design = result.design;
  const significant = naive.pValue < 0.05;

  const blocks = [
    el("h3", { text: "Did the campaign work?" }),
    metrics([
      {
        label: "Contacted, converted at",
        value: `${per100(naive.rateTreated, 2, false)} per 100`,
        note: `${count(naive.nTreated)} customers`,
      },
      {
        label: "Not contacted, converted at",
        value: `${per100(naive.rateControl, 2, false)} per 100`,
        note: `${count(naive.nControl)} customers`,
      },
      {
        label: "Difference",
        value: `${per100(naive.absoluteDifference)} per 100`,
        tone: significant ? (naive.absoluteDifference > 0 ? "good" : "bad") : null,
        note: `could be ${per100(naive.ciLow)} to ${per100(naive.ciHigh)}`,
      },
      {
        label: "Fluke probability",
        value: naive.pValue < 0.001 ? tinyP(naive.pValue) : num(naive.pValue, 3),
        tone: significant ? "good" : "bad",
      },
    ]),
  ];

  if (significant && naive.absoluteDifference > 0) {
    blocks.push(
      note(
        "good",
        `**The campaign worked.** The gap is ${per100(naive.absoluteDifference)} per
100 customers, with an honest range of ${per100(naive.ciLow)} to
${per100(naive.ciHigh)} — entirely above zero. A relative lift of
${percent(naive.relativeLift)}.`,
      ),
    );
  } else if (significant) {
    blocks.push(
      note(
        "bad",
        `**The campaign appears to have hurt.** Contacted customers converted
${per100(Math.abs(naive.absoluteDifference), 2, false)} per 100 *less* often,
and the range (${per100(naive.ciLow)} to ${per100(naive.ciHigh)}) stays below
zero. Worth ruling out a data problem before accepting it — a mislabelled
treatment column produces exactly this.`,
      ),
    );
  } else {
    blocks.push(
      note(
        "warn",
        `**Cannot tell.** The measured gap is ${per100(naive.absoluteDifference)} per
100, but the honest range runs from ${per100(naive.ciLow)} to
${per100(naive.ciHigh)} and includes zero. That is not the same as "the
campaign did nothing" — see below.`,
      ),
    );
  }

  if (design) {
    blocks.push(
      el("h3", { text: "What this test could have detected" }),
      prose(`
This is the part most readouts skip, and it decides how to read everything
above.

With ${count(design.perArm)} customers in the smaller group and a base rate of
${percent(design.baseline, 2)}, the smallest effect this campaign could
**reliably** have detected was **${percent(design.relativeMde, 1)}** — that is
${per100(design.absoluteMde, 3, false)} per 100 customers.

"Reliably" means a 4-in-5 chance of spotting it. An effect a little under that
line is not undetectable; it is detectable less often than you would want. So
a significant result sitting just below the line is perfectly possible — this
test got lucky — and it is a reason to repeat the campaign before betting on
the size of the effect, not a contradiction.
      `),
      metrics([
        {
          label: "Smallest detectable effect",
          value: percent(design.relativeMde, 1),
          tone: design.relativeMde > 0.5 ? "bad" : design.relativeMde > 0.2 ? null : "good",
        },
        {
          label: "You measured",
          value: percent(Math.abs(naive.relativeLift), 1),
          // No tone. Green-or-red here invites the reading "my effect failed a
          // test", when all this compares is one effect size against a
          // planning threshold. The verdict belongs to the p-value above.
        },
        {
          label: "For a 20% lift you'd need",
          value: `${count(design.nPerArmForTwentyPercent)} per group`,
          note: `you had ${count(design.perArm)}`,
        },
        {
          label: "Conversions per group",
          value: count(design.expectedConversionsPerArm),
          tone: design.expectedConversionsPerArm < 30 ? "bad" : null,
          note: design.expectedConversionsPerArm < 30 ? "too few to trust" : "healthy",
        },
      ]),
    );

    if (!significant) {
      blocks.push(
        note(
          "info",
          `**So be careful how you report this.** A test that can only see
effects of ${percent(design.relativeMde, 1)} or more will return "no
significant difference" for any genuine improvement below that. The honest
conclusion is **"we could not tell"**, not "it does not work". Those two get
conflated constantly, and the second one kills initiatives that were working.`,
        ),
      );
    } else if (!design.observedEffectIsAboveMde) {
      blocks.push(
        note(
          "warn",
          `**A caveat on the size.** Your result is significant, but the effect
you measured (${percent(Math.abs(naive.relativeLift), 1)}) sits just below what
this campaign was reliably powered to find (${percent(design.relativeMde, 1)}).

That combination is a known trap. When a test is marginally powered, the
results that happen to reach significance are the ones where chance pushed the
estimate upward — so the measured effect tends to **overstate** the real one.
Trust the direction. Treat the size as an upper bound until a larger campaign
confirms it.`,
        ),
      );
    }

    if (design.expectedConversionsPerArm < 30) {
      blocks.push(
        note(
          "warn",
          `**Treat everything on this page as provisional.** With only
${count(design.expectedConversionsPerArm)} conversions per group, the maths
behind every confidence range here assumes more data than you have. It starts
to mislead below roughly thirty, usually in the optimistic direction.`,
        ),
      );
    }
  }

  return blocks;
}

function balanceBlock(result) {
  const worst = result.balance.reduce((a, b) => (b.absStdMeanDiff > a.absStdMeanDiff ? b : a));
  const failing = result.balance.filter((row) => !row.balanced);

  return [
    el("h3", { text: "Was the split actually fair?" }),
    prose(`
Random assignment is supposed to guarantee the two groups look alike. "Supposed
to" is not a measurement — so here is the measurement.

The yardstick is the gap between the groups expressed in units of how much
customers vary anyway. Below 0.10 is conventionally negligible.
    `),
    metrics([
      {
        label: "Largest imbalance",
        value: num(worst.absStdMeanDiff, 4),
        tone: worst.absStdMeanDiff < 0.1 ? "good" : "bad",
        note: `on ${worst.feature}`,
      },
      {
        label: "Characteristics out of line",
        value: `${failing.length} of ${result.balance.length}`,
        tone: failing.length ? "bad" : "good",
      },
    ]),
    failing.length
      ? note(
          "bad",
          `**${failing.length} characteristic${failing.length > 1 ? "s differ" : " differs"}
between the groups by more than chance comfortably explains**
(${failing.map((row) => row.feature).join(", ")}). That usually means something
went wrong operationally — a send that failed for one region, a suppression
list applied to one arm, an export that dropped rows unevenly. Until it is
explained, treat every effect above as potentially contaminated.`,
        )
      : note(
          "good",
          `**The split looks fair.** The worst imbalance was
${num(worst.absStdMeanDiff, 4)}, comfortably inside the 0.10 threshold. Nothing
here suggests the randomisation went wrong.`,
        ),
    details(
      "Show every characteristic",
      table({
        columns: [
          "Characteristic",
          "Average, contacted",
          "Average, not contacted",
          "Standardised difference",
          "Fluke probability",
          "Verdict",
        ],
        numeric: [1, 2, 3, 4],
        rows: result.balance.map((row) => [
          row.feature,
          num(row.meanTreated, 3),
          num(row.meanControl, 3),
          num(row.stdMeanDiff, 4),
          num(row.pValue, 3),
          pill(row.balanced, ["balanced", "out of line"]),
        ]),
      }),
    ),
  ];
}

function targetingBlock(result, data, state, colours, nBoot, nPlacebo, rerun) {
  const boot = result.bootstrap;
  const placebo = result.placebo;
  const dogs = result.sleepingDogs;
  const frontier = result.frontier;

  const peak = frontier.reduce((best, row) =>
    row.incrementalProfit > best.incrementalProfit ? row : best,
  );
  const randomAtPeak = randomTargetingProfit(
    data.outcome,
    data.treatment,
    data.spend ?? data.outcome,
    peak.fraction,
    { costPerContact: state.cost, marginRate: state.margin },
  );

  const decileRows = result.deciles.map((row) => ({
    label: String(row.decile),
    value: row.observedUplift * 100,
    low: row.ciLow * 100,
    high: row.ciHigh * 100,
    predicted: row.predictedUplift * 100,
    tooltipTitle: `Tenth ${row.decile}`,
    extra: [["People", count(row.n)]],
  }));

  const blocks = [
    el("h3", { text: "Is your ranking any good?" }),
    prose(`
Your score column was used only to **order** customers. Every number below
comes from counting what actually happened in each part of that order — never
from the scores themselves, which are systematically too large in every model
of this kind.
    `),
    metrics([
      {
        label: "Extra conversions vs random",
        value: num(boot.qiniCoefficient),
        tone: boot.beatsRandom ? "good" : "bad",
      },
      {
        label: "Honest range",
        value: `${num(boot.ciLow)} to ${num(boot.ciHigh)}`,
        note: `${count(nBoot)} resamples`,
      },
      {
        label: "Fluke probability",
        value: num(boot.pValueOneSided, 3),
        tone: boot.beatsRandom ? "good" : "bad",
      },
      {
        label: "Survives the shuffle test?",
        value: placebo.passes ? "yes" : "no",
        tone: placebo.passes ? "good" : "bad",
        note: `${percent(placebo.sharePlacebosExceedingReal)} of shuffles beat it`,
      },
    ]),

    boot.beatsRandom && placebo.passes
      ? note(
          "good",
          `**Your ranking is doing real work.** It beat a random shuffle by
${num(boot.qiniCoefficient)} conversions, the range stays above zero, and it
survived scoring against ${count(nPlacebo)} campaigns where the contact column
was deliberately scrambled.`,
        )
      : note(
          "bad",
          `**Not proven.** ${
            boot.beatsRandom
              ? ""
              : `The honest range (${num(boot.ciLow)} to ${num(boot.ciHigh)}) includes zero. `
          }${
            placebo.passes
              ? ""
              : `${percent(placebo.sharePlacebosExceedingReal)} of deliberately scrambled campaigns scored at least as well as your real one. `
          }That does not mean your model is worthless — it means this campaign is
not big enough, or the effect not varied enough, to show that it is worth
something. Exactly the situation the study's headline campaign was in.`,
        ),

    figure({
      legend: [
        { label: "Where scrambled campaigns landed", colour: colours.inkFaint },
        { label: "Your real campaign", colour: colours.s1, shape: "line" },
        { label: "Zero", colour: colours.ink, shape: "dash" },
      ],
      chart: distributionStrip({
        band: {
          low: placebo.meanQini - placebo.sdQini,
          high: placebo.meanQini + placebo.sdQini,
          centre: placebo.meanQini,
        },
        reference: placebo.realQini,
        labels: { band: "Scrambled campaigns", reference: "Your campaign" },
        formatX: (v) => num(v),
      }),
      caption:
        "The shuffle test. Scrambled campaigns contain nothing to find by " +
        "construction, so your real score has to clear what luck alone produces.",
    }),

    el("h3", { text: "The check a sceptic should ask for" }),
    prose(`
Your ranking cut into ten equal groups, best first. In each group, what the
contacted customers actually did against what the not-contacted ones did. No
model involved — this is counting.

If the ranking works, the bars slope downwards.
    `),
    figure({
      legend: [
        { label: "Campaign helped this group", colour: colours.s1 },
        { label: "Campaign hurt this group", colour: colours.danger },
        { label: "What your score predicted", colour: colours.s2 },
      ],
      chart: barsWithPredictions(decileRows, {
        yTitle: "Extra conversions per 100 people",
        xTitle: "Your ranking, best tenth first",
      }),
      caption:
        "Orange dots well above the bars is normal and important: these models " +
        "predict effects far larger than the data can show.",
    }),
    details(
      "Show the ten groups as a table",
      table({
        columns: [
          "Tenth",
          "People",
          "Your score predicted",
          "Actually measured",
          "As low as",
          "As high as",
        ],
        numeric: [0, 1, 2, 3, 4, 5],
        rows: result.deciles.map((row) => [
          row.decile,
          count(row.n),
          per100(row.predictedUplift),
          per100(row.observedUplift),
          per100(row.ciLow),
          per100(row.ciHigh),
        ]),
      }),
    ),
  ];

  // --- money
  blocks.push(
    el("h3", { text: "What it is worth" }),
    result.usedOutcomeAsSpend
      ? note(
          "plain",
          "No revenue column was mapped, so conversions are being valued at **$1 " +
            "each**. The shape of the curve is right; the dollar amounts are not. " +
            "Map a revenue column for real money.",
        )
      : null,
    el("div", { class: "controls" }, [
      slider({
        label: "Cost of contacting one customer",
        min: 0.01,
        max: 2,
        step: 0.01,
        value: state.cost,
        format: (v) => money(v, 2),
        help: "Everything one more contact costs you, including list fatigue.",
        onInput: (value) => {
          state.cost = value;
        },
      }),
      slider({
        label: "Profit kept per $1 of revenue",
        min: 0.05,
        max: 0.95,
        step: 0.05,
        value: state.margin,
        format: (v) => percent(v),
        help: "Your gross margin on the extra revenue the campaign causes.",
        onInput: (value) => {
          state.margin = value;
        },
      }),
      el("div", { class: "control" }, [
        el("span", { class: "control-label", text: "Apply" }),
        el("button", {
          class: "icon-button",
          type: "button",
          text: "Recalculate with these",
          onclick: rerun,
        }),
        el("span", {
          class: "control-help",
          text: "Re-runs the whole analysis, including the resampling.",
        }),
      ]),
    ]),
    figure({
      legend: [
        { label: "Contact your top picks", colour: colours.s1, shape: "line" },
        { label: "Contact the same number at random", colour: colours.s2, shape: "line" },
      ],
      chart: lines(
        [
          {
            label: "Contact your top picks",
            colour: colours.s1,
            points: frontier.map((row) => ({
              x: row.fraction * 100,
              y: row.incrementalProfit,
            })),
          },
          {
            label: "Contact the same number at random",
            colour: colours.s2,
            points: frontier.map((row) => ({
              x: row.fraction * 100,
              y: randomTargetingProfit(
                data.outcome,
                data.treatment,
                data.spend ?? data.outcome,
                row.fraction,
                { costPerContact: state.cost, marginRate: state.margin },
              ),
            })),
          },
        ],
        {
          xTitle: "Share of your list contacted (%)",
          yTitle: "Profit after costs ($)",
          formatY: (v) => money(v, 0),
          formatX: (v) => `${v.toFixed(0)}%`,
          markers: [{ at: peak.fraction * 100, label: "peak", colour: colours.danger }],
        },
      ),
      caption:
        "The gap between the lines is what your ranking is worth. If they sit on " +
        "top of each other, it is worth nothing.",
    }),
    metrics([
      {
        label: "Contact this share",
        value: percent(peak.fraction),
        note: "where profit peaks",
      },
      {
        label: "Profit there",
        value: money(peak.incrementalProfit),
        tone: peak.incrementalProfit > 0 ? "good" : "bad",
      },
      {
        label: "Same number at random",
        value: money(randomAtPeak),
      },
      {
        label: "Your ranking is worth",
        value: money(peak.incrementalProfit - randomAtPeak),
        tone: peak.incrementalProfit - randomAtPeak > 0 ? "good" : "bad",
      },
    ]),
    details(
      "Show every depth as a table",
      table({
        columns: [
          "Share contacted",
          "People",
          "Extra conversions",
          "Extra revenue",
          "Cost",
          "Profit",
        ],
        numeric: [0, 1, 2, 3, 4, 5],
        rows: frontier.map((row) => [
          percent(row.fraction, 1),
          count(row.nTargeted),
          num(row.incrementalConversions),
          money(row.incrementalRevenue),
          money(row.contactCost),
          money(row.incrementalProfit),
        ]),
      }),
    ),
  );

  // --- sleeping dogs
  if (dogs.nFlagged > 0) {
    const low = dogs.measuredUpliftCiLow;
    const high = dogs.measuredUpliftCiHigh;
    blocks.push(
      el("h3", { text: "Who your model says to leave alone" }),
      prose(`
Your score is negative for **${count(dogs.nFlagged)} customers**
(${percent(dogs.shareFlagged, 1)} of the file) — the model expects contacting
them to *cost* you sales.

That is a hypothesis, not a finding. Here is what the randomised data actually
says about that same group.
      `),
      metrics([
        { label: "Flagged", value: count(dogs.nFlagged), note: percent(dogs.shareFlagged, 1) },
        {
          label: "Model predicted",
          value: `${per100(dogs.meanPredictedUplift)} per 100`,
          tone: "bad",
        },
        {
          label: "Actually measured",
          value: `${per100(dogs.measuredUplift)} per 100`,
          tone: high < 0 ? "bad" : low > 0 ? "good" : null,
          note: `${per100(low)} to ${per100(high)}`,
        },
        {
          label: "Profit if you contact them",
          value: money(dogs.profitIfTreated),
          tone: dogs.profitIfTreated < 0 ? "bad" : "good",
        },
      ]),
      high < 0
        ? note(
            "good",
            `**Your model was right.** This group genuinely converted less when
contacted — the whole range sits below zero. Suppressing them is a clear and
immediate saving of ${money(Math.abs(dogs.profitIfTreated))}.`,
          )
        : low > 0
          ? note(
              "bad",
              `**Your model got the direction wrong.** It predicted harm; the data
shows these customers gained ${per100(dogs.measuredUplift)} per 100, with a
range entirely above zero. They are not sleeping dogs — the campaign helped
them, just less than it helped others. Do not suppress them on this evidence.`,
            )
          : note(
              "warn",
              `**Cannot tell.** The measured effect is ${per100(dogs.measuredUplift)}
per 100 with a range of ${per100(low)} to ${per100(high)}, which straddles
zero. Suppressing ${percent(dogs.shareFlagged, 1)} of your list on this is a
bet, not a decision.`,
            ),
    );
  }

  blocks.push(
    details(
      "Why are the predicted effects so much bigger than the measured ones?",
      `
Because every model of this kind produces its answer by subtracting one
prediction from another, and the two predictions' errors do not cancel when you
subtract — they add.

So the spread of predicted effects comes out wider than the spread of real
ones. The top of the list gets assigned effects far larger than the data can
support, and the bottom of the list crosses below zero even when no real effect
down there is negative. That second part is where phantom sleeping dogs come
from.

The rule that follows: use the ranking to decide **who goes first**, and use
counted outcomes — the decile table and the frontier above — to decide **how
much that is worth**. Never quote your model's predicted effect sizes as a
forecast.
      `,
    ),
  );

  return blocks;
}
