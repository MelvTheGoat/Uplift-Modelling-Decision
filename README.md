# Who buys *because* of the campaign?

A shop sends a promotional email. The people who got it buy more. The obvious
next move is to build a model that predicts who will buy, and email those
people.

That is the wrong model, and it is wrong in a way that looks like success. It
finds your best customers — the ones who were going to buy anyway. You spend
the budget taking credit for sales you already had.

The question worth asking is different: **who buys *because* of the email?**
Those are two different groups of people, and only the second is worth paying
for.

This repository works through that question on a real randomised campaign, and
is honest about where the answer runs out.

**→ [Analyse your own campaign](https://melvthegoat.github.io/Uplift-Modelling-Decision/#/analyse)** —
upload a CSV and get this entire analysis run on your data. It runs in your
browser; the file is never uploaded anywhere. There is a sample file built in
if you want to see it work first.

**→ [Read the study](https://melvthegoat.github.io/Uplift-Modelling-Decision/)** —
plain English, interactive, no statistics background needed.

**→ [Read the memo](MEMO.md)** — two pages, for someone who has to make the
decision.

---

## What we found

On the Hillstrom email trial: 64,000 customers, split by coin flip in 2008 into
two email campaigns and a no-email control group.

**1. The email works.** People who got it bought at 1.25 per 100, against 0.57
per 100 for people who got nothing. Roughly double, and nowhere near a fluke.

**2. Picking who to email is not proven to help.** Four different methods were
tried. The best earned about 10 extra sales over emailing the same number of
people at random — but the honest range on that figure runs from −2 to +22. It
includes zero. We cannot rule out that the ranking is worth nothing.

**3. The budget cap costs more than the targeting earns.** This is the finding
nobody asked for and the one worth acting on. Getting the targeting right is
worth about **$1,390**. Being allowed to email 85% of the list instead of 30%
is worth about **$3,170**. The campaign stays profitable well past the point
where the budget stops it.

**4. The result failed the test that most studies skip.** We re-ran everything
on deliberately broken data — same customers, same purchases, but with "who got
the email" shuffled at random. There is nothing to find in that data by
construction. One run in five scored *better* than the real campaign did.

That last point is reported here rather than buried. A model picked on this
evidence would have been picked on a coin flip.

### How we know the pipeline is not simply broken

The women's campaign, run through exactly the same code, passes every check: a
clear positive score, an honest range that excludes zero, and a shuffle test it
comfortably survives.

A broken pipeline fails everywhere. This one gives a clean answer on one
campaign and an honest shrug on another, which is what working machinery looks
like when the signal differs between datasets.

---

## Two demonstrations of why the obvious approach misleads

Both run on invented customers, where the true answer is known and can be
checked — which is impossible on real data, because you never find out what a
person would have done in the other group.

**Letting the marketing team choose who gets the email.** Change nothing else.
The simple before-and-after comparison then overstates the true effect by
**7.5 times**, and nothing in the output looks wrong.

**Ranking by "who will buy" instead of "who the email moves".** The first
approach emails **five times more** customers that the email actively puts off.
It emails them *because* they look likely to buy, which they are — right up
until the email arrives.

---

## What's here

| File | What it does |
|---|---|
| [`MEMO.md`](MEMO.md) | The deliverable. Two pages, plain language, for a marketing director. |
| [`METHODS.md`](METHODS.md) | The technical detail: what each method does, what was assumed, what can go wrong. |
| [`DEPLOY.md`](DEPLOY.md) | How to put the website online. Free, and mostly already set up. |
| `web/` | The website. Static HTML, CSS and JavaScript — no build step, no dependencies. |
| `src/simulate.py` | Invented customers with a **known** individual response to the email. Everything is checked here before it is trusted on real data. |
| `src/naive.py` | The wrong analysis, done carefully, plus the two demonstrations above. |
| `src/data.py` | Loads the Hillstrom dataset, with a mirror in case the original host is blocked. |
| `src/estimators.py` | The four methods: three hand-written, plus a causal forest from EconML. |
| `src/evaluation.py` | How rankings are scored, honestly and out of sample. **No accuracy score** — see below. |
| `src/policy.py` | Profit at every targeting depth, the budget constraint, and the cost of contacting people the email puts off. |
| `src/experiment.py` | How many customers a test needs, and how to need fewer. |
| `src/robustness.py` | The four attempts to break the finding. |
| `src/cli.py` | Runs the whole study and writes `results/`. |
| `web/js/analysis.js` | The analysis engine, in the browser. A port of the Python below, held to it by a parity test. |
| `scripts/build_web_data.py` | Packs `results/` into one file for the website. |
| `tests/` | 150 tests, all on invented data — no network, no dataset download. |

### The tool

The site is not only a write-up. **Analyse your own campaign** takes a CSV of
any randomised A/B campaign and runs the whole analysis on it:

- Did it work — rates, the gap, an honest range, a fluke probability.
- **What your test could have detected** — the check that separates "it does
  not work" from "we could not tell". Most readouts skip this one.
- Whether the split was actually fair, on whatever characteristics you map.
- If you supply uplift scores from a model: ranking quality with a resampled
  range, the decile table, the profit frontier against random targeting, the
  shuffle test, and a measured verdict on whoever your model wants suppressed.

Column names do not have to match anything — you map them with dropdowns, and
pick which value means "got the campaign". Everything runs in the browser.

`web/js/analysis.js` is a port of `src/evaluation.py`, `src/policy.py` and
`src/naive.py`. Two implementations of one formula is a bug waiting to happen,
so `tests/test_web_analysis_parity.py` generates a campaign, runs both, and
fails if any number disagrees.

---

## Running it yourself

Everything is free, open source, and runs locally. Nothing calls a paid API.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

python -m src.cli all          # the full study, about 10 minutes
```

Or one stage at a time, in the order they are meant to be read:

```bash
python -m src.cli validate     # check the methods against known truth  (~30s)
python -m src.cli naive        # the wrong answer, and why              (~5s)
python -m src.cli hillstrom    # the real study                         (~9m)
python -m src.cli experiment   # how big a test would need to be        (~10s)

python -m src.cli all --quick  # a rougher version                      (~2m)
```

Results land in `results/` as JSON and CSV. Every number in the memo and on the
website traces back to one of them.

The dataset downloads on first use and caches in `data/`. The original host is
plain HTTP and blocked by many corporate networks, so the loader falls back to
an identical copy automatically.

### The website

```bash
python scripts/build_web_data.py    # after re-running the study
python -m http.server -d web 8000   # then open http://localhost:8000
```

Opening `web/index.html` directly will not work — the page uses JavaScript
modules, which browsers refuse to load from a `file://` path. It needs a web
server, which is what the second command is. [`DEPLOY.md`](DEPLOY.md) covers
putting it online.

### Checks

```bash
pytest -q            # 150 tests
ruff check . && ruff format --check .
mypy                 # strict mode
```

All three are clean, and CI runs them on Python 3.10 to 3.12.

**The test suite never touches the real dataset.** It runs entirely on invented
customers whose true individual response is known, so a test can assert that a
method is *correct* rather than that last week's output has not changed. One
test also checks that the sample-size calculator on the website agrees with the
Python one, because two implementations of one formula is a bug waiting to
happen.

---

## How the study is organised, and why in that order

**1. Invented customers first.** On real data you can never see how one person
would have responded both ways, so a method that looks good there might just be
reproducing your own mistakes. `src/simulate.py` builds data where the answer
is written down in advance, and nothing is trusted on real data until it
recovers that known answer.

This is also how the known weaknesses get *measured* rather than asserted.
Against truth, on an even split:

| Method | Spread of predicted effects, vs the real spread | Ranking agreement with truth |
|---|---|---|
| Single-model | 0.59× — flattens real differences between people | 0.67 |
| Two-model | 1.77× — invents differences that are not there | 0.38 |
| Cross-fitted | 1.38× | 0.44 |
| Causal forest | 0.61× | 0.62 |

A method at 1.00× would predict effects that vary exactly as much as the real
ones do. Nothing is close, in either direction, and that is the single most
important fact about these tools: **their rankings are worth more than their
numbers.**

One received piece of wisdom did not survive contact with the data. The
cross-fitted method is usually recommended for campaigns where the two groups
are very different in size. It does beat the two-model method here — but by
*more* on the even split than on the uneven one (27% better on prediction error
versus 17%). On this data its advantage is general, not a product of the
imbalance it was designed for.

**2. The wrong analysis second**, because it frames everything else.

**3. Then methods, scoring, policy, robustness.** The order matters: a
recommendation is only as good as the scoring behind it, and the scoring is
only as good as the checks that try to break it.

### Two choices worth flagging

**No accuracy score anywhere.** Accuracy measures how well a model separates
buyers from non-buyers. An uplift model is not trying to do that, and the
*best* one would score badly — a customer who was always going to buy has zero
uplift and should rank low. A high accuracy score here would be evidence that
something had gone wrong.

**Money is always measured, never predicted.** Every profit figure comes from
counting what actually happened in the randomised data among the selected
customers. None of it comes from the model's own predicted effect sizes. Using
predictions as values is circular, and it is the most common way an uplift
analysis overstates itself — here it would have inflated the best group by
nearly three times.

---

## Assumptions you can change

Everything a business might argue with lives in `src/config.py`: cost per
email ($0.10), profit kept per dollar of sales (30%), and the share of the list
the budget allows (30%). Edit and re-run.

`src/robustness.py` sweeps the first two automatically, and the answer is
genuinely sensitive to them — the right targeting depth moves from 2.5% to 85%
of the list across the range tested. The memo says so out loud, and the
website lets you move the sliders yourself.

---

## Data

**Hillstrom MineThatData E-Mail Analytics Challenge** (2008) — 64,000
customers, split at random three ways: men's email, women's email, no email.
Outcomes recorded over the following two weeks: site visit, purchase, amount
spent. Public, no sign-up.

It is a real randomised experiment, freely available, and analysed enough times
that a wrong answer here would be noticed. That last property is the reason to
use it rather than something more exciting.

**Criteo-UPLIFT v2** is supported behind `load_criteo()` and optional
throughout; nothing depends on it.

## Licence

MIT.
