/** How this was built, what it cannot do, and how to run it yourself. */

import { el, prose, note, details, table, metrics } from "../dom.js";
import { DATA_SOURCE } from "../copy.js";
import { count, money, num, percent, studyLabel } from "../format.js";

export function render(data) {
  const main = data.hillstrom.mens_conversion;
  const validation = data.validation ?? {};
  const scenarios = Object.entries(validation);

  return [
    el("h1", { text: "How this was built" }),
    el("p", {
      class: "lede",
      text: "The data, the code, the honest limitations, and how to run it yourself.",
    }),

    el("h2", { text: "Where the data comes from" }),
    prose(DATA_SOURCE),
    metrics([
      { label: "Customers in the headline campaign", value: count(main.summary.n) },
      { label: "Got the email", value: count(main.summary.n_treated) },
      { label: "Got nothing", value: count(main.summary.n_control) },
      { label: "Characteristics on file", value: num(main.summary.n_features, 0) },
    ]),

    el("h2", { text: "How we know the code is right" }),
    prose(`
A pipeline that estimates something unobservable has an awkward property: you
cannot check its answers against reality, because reality does not record what
each person *would* have done in the other group. So a bug and a genuine finding
look identical.

The way around it is to build a world where the answer **is** known. The study
generates fake customers whose individual response to the email is written down
by construction, runs every model against them, and checks whether each one
recovers the truth it was given.
    `),
    scenarios.length
      ? [
          table({
            caption: "Four models against invented customers whose true effect is known",
            columns: [
              "Scenario",
              "Model",
              "Ranking agreement with truth",
              "Spread vs true spread",
              "Sleeping dogs found",
            ],
            numeric: [2, 3, 4],
            rows: scenarios.flatMap(([name, scenario]) =>
              scenario.estimators.map((row) => [
                name === "balanced_50_50" ? "Even split" : "Uneven split (15/85)",
                row.model,
                num(row.kendall_tau, 3),
                `×${num(row.sd_ratio, 2)}`,
                percent(row.sleeping_dogs_detected, 0),
              ]),
            ),
          }),
          prose(`
The fourth column is the one to read. A value of 1.00 would mean the model's
predicted effects vary exactly as much as the real ones do. Below 1 means the
model is **flattening** real differences between people; above 1 means it is
**inventing** differences that are not there.

Almost nothing sits near 1. That is not a bug — it is the behaviour described on
the [targeting page](#/targeting), reproduced on data where it can be measured
exactly. It is the reason this site never quotes a model's predicted effect size
as a forecast.
          `),
        ]
      : null,

    el("h2", { text: "What this cannot tell you" }),
    prose(`
Stated plainly, because a study that lists no limitations has not looked for any.
    `),
    table({
      columns: ["Limitation", "Why it matters"],
      rows: [
        [
          "**The data is from 2008.**",
          "Email behaviour has changed. Inbox filtering, promotion tabs and general fatigue are all different now. The methods transfer; the effect sizes should not be assumed to.",
        ],
        [
          "**One retailer, one category, two weeks.**",
          "Nothing here establishes what happens over a longer horizon, or in a different category. A two-week window also cannot see customers who were merely delayed rather than converted.",
        ],
        [
          "**The email cost and margin are assumptions.**",
          `$${num(main.policy.cost_per_contact, 2)} per send and ${percent(main.policy.margin_rate)} margin are not in the data; they were chosen. Their effect is tested across a grid on the money page, and the recommended depth swings across most of its range when they move.`,
        ],
        [
          "**Models were not heavily tuned.**",
          "Defaults with light adjustment. A tuning effort might find more signal — but given the campaign fails the shuffle test, tuning against that same score would mostly fit noise harder.",
        ],
        [
          "**No long-run or multi-campaign effects.**",
          "Each campaign is analysed on its own. Real email programmes interact: contacting someone today changes how they respond next month, and none of that is measurable here.",
        ],
      ],
    }),

    el("h2", { text: "The honest summary" }),
    note(
      "info",
      `
On the headline campaign, **no model proved it beats random targeting**, and the
ranking **failed the shuffle test**. The women's campaign passed everything,
which is what rules out a broken pipeline.

The decision-relevant finding was one nobody asked for: the budget cap costs
${money(main.policy.optimal_profit - main.policy.recommended_profit)} while the
targeting is worth ${money(main.policy.profit_vs_random)}. Raise the cap first.

A study that produced a confident model recommendation from this data would have
been more satisfying to read and wrong.
      `,
    ),

    el("h2", { text: "Running it yourself" }),
    prose(`
Everything is open source, runs locally, and needs no paid service or API key.
    `),
    el("pre", {}, [
      el("code", {
        text: `git clone https://github.com/MelvTheGoat/Uplift-Modelling-Decision
cd Uplift-Modelling-Decision

python -m venv .venv
source .venv/bin/activate        # Windows: .\\.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
pip install -e .

python -m src.cli all            # the full study, about 10 minutes
python -m src.cli all --quick    # a rougher version, about 2`,
      }),
    ]),
    prose(`
The dataset downloads on first run. Results land in \`results/\`, which is also
committed, so the numbers on this site are readable without running anything.

To rebuild this site's data after a fresh run:
    `),
    el("pre", {}, [
      el("code", {
        text: `python scripts/build_web_data.py   # regenerates web/data.json
python -m http.server -d web 8000  # then open http://localhost:8000`,
      }),
    ]),
    note(
      "plain",
      "A plain `open web/index.html` will not work. The site uses JavaScript " +
        "modules and `fetch`, both of which browsers refuse to run from a `file://` " +
        "path. It needs a web server, which is what the second command is.",
    ),

    el("h2", { text: "The checks" }),
    el("pre", {}, [
      el("code", {
        text: `pytest -q                          # the test suite
ruff check . && ruff format --check .
mypy                               # strict mode`,
      }),
    ]),
    prose(`
The test suite runs entirely on invented data, so it is fast and does not depend
on a third-party host staying up. It includes a check that the JavaScript
sample-size calculator on the [Plan your own test](#/plan-a-test) page agrees
with the Python one — two implementations of one formula is a bug waiting to
happen, so a test fails if they drift apart.
    `),

    details(
      "Why is there no chart library?",
      `
The charts are hand-written SVG, which is unusual enough to be worth explaining.

The deciding reason is **error bars**. The central finding of this study is "the
number is real but its honest range includes zero", and a chart that cannot show
uncertainty would actively misrepresent it. Most charting libraries treat error
bars as a plugin or an afterthought.

The second reason is that there is no build step and no external script. The site
is HTML, CSS and JavaScript modules, served as files. Nothing to compile, no CDN
to go down, and nothing that stops working in three years because a toolchain
moved on.
      `,
    ),

    details(
      "Why no accuracy score anywhere?",
      `
Because it would be the wrong measurement, and a good score on it would be a
warning sign.

Accuracy and AUC measure how well a model predicts **who buys**. This study
ranks people by **how much the email changed them**. Those are different
orderings over the same customers, and a model that nails the second will score
poorly on the first — because the people the email moves most are not the people
most likely to buy.

What gets measured instead is the ranking itself: take the people the model puts
at the top, and check in the real randomised data whether the email actually did
more for them. That check cannot be gamed by predicting purchases well.
      `,
    ),

    el("h2", { text: "Campaign summary" }),
    table({
      columns: [
        "Campaign",
        "Customers",
        "Rate, emailed",
        "Rate, not emailed",
        "Best model",
        "Everything holds up?",
      ],
      numeric: [1, 2, 3],
      rows: data.meta.studies.map((key) => {
        const entry = data.hillstrom[key];
        return [
          studyLabel(key),
          count(entry.summary.n),
          percent(entry.summary.outcome_rate_treated, 2),
          percent(entry.summary.outcome_rate_control, 2),
          entry.best_model,
          entry.robustness.overall_pass ? "yes" : "no",
        ];
      }),
    }),
  ];
}
