# Who should get the campaign — and what that's actually worth

**To:** Marketing Director
**Re:** Budget-constrained targeting for the e-mail programme
**Evidence:** 64,000-customer randomised trial (Hillstrom 2008), 42,613 in the two
arms analysed here. All figures reproducible from `results/`.

---

## The recommendation

**Send the men's creative, spend the entire budget, and use the model to pick who
gets it — that is worth about $3,070 in profit against $1,670 if you picked the
same number of people at random.** But the bigger number is elsewhere: the budget
cap is costing you more than bad targeting is. At a 30%-of-file budget you make
$3,070; the profit-maximising campaign mails 85% of the file and makes $6,240. So
**raising the budget is worth roughly $3,170 and better targeting is worth roughly
$1,390** — and of those two, only the budget number is on solid statistical
ground.

That last clause is the honest part of this memo, and it is explained below. The
targeting gain did not survive one of our four robustness checks. Treat it as a
promising bet to be tested, not a finding to be operationalised.

*(All figures are incremental profit on the 42,613-customer file, at $0.10 per
contact and a 30% gross margin on incremental revenue. Scale linearly for a
larger list.)*

---

## Why "the people we mailed converted more" is not evidence

Customers who got the men's e-mail converted at **1.25%**. Customers who got
nothing converted at **0.57%**. That is a 119% lift, and it is statistically
overwhelming.

It is also almost useless for deciding whom to mail, for two separate reasons that
get confused with each other.

**First — in this case the number is real, but only because someone randomised.**
This was a proper experiment: customers were assigned to arms by chance, so the
only systematic difference between them is the e-mail. In the campaign readouts
you normally see, nobody randomised. The list was built from the engaged segment,
the loyalty members, the people who opened last time — customers who would have
bought more anyway. We measured how bad that gets: on simulated data where we
*know* the true effect is 2.2 percentage points, a realistically selected list
produces a headline gap of **16.9 points — a 7.5× overstatement**. Same customers,
same true effect, entirely fictional result. If a campaign readout has no control
group, the number in it is not an effect size.

**Second — even when the number is real, it is an average, and averages hide the
decision.** A 0.68-point average lift is equally consistent with everyone
responding slightly, and with a quarter of the file responding strongly while
another quarter is actively annoyed into unsubscribing. Those two worlds call for
opposite campaigns, and the headline cannot tell them apart.

The expensive consequence follows from that. Once a campaign "works", the natural
next step is to build a model of who is most likely to convert and mail them more.
That model ranks people by **who buys**, not by **who buys because of the e-mail** —
and those are different people. Customers who were always going to purchase get
ranked at the top and contribute nothing incremental. On our simulated data, where
we can check against the truth, a conventional "likelihood to convert" model
targeting the top 20% captured **440** genuine incremental sales; an uplift model
targeting the same 20% captured **657**. Worse, the conventional model mailed
**1,345** customers whom the campaign actively harms, against **254** for the
uplift model — five times as many.

**Uplift modelling**, in one sentence: instead of predicting whether a customer
will buy, it predicts how much *more* likely they are to buy if contacted than if
left alone. That difference is the only thing a targeting budget should be spent
on.

---

## What the model found

We tested four estimators (S-, T- and X-learner, plus a causal forest), first
against simulated data where the true individual effect is known — all four
recovered it, with the causal forest and S-learner most accurate — and then on the
real trial.

**The heterogeneity is real but shallow.** On the men's campaign, the model's top
10% of customers show a measured lift of **1.36 percentage points** (95% CI: 0.71
to 2.02) against a file average of 0.68. That top slice is genuinely twice as
responsive as average, and that is where the targeting gain comes from.

**Below the top decile, the ordering falls apart.** Deciles two through ten show
measured lifts of 0.46, 0.90, 0.46, 0.89, 0.50, 0.64, 0.58, 0.37 and 0.63 points —
no trend at all. The model confidently predicts the bottom decile is *harmed*
(−1.5 points); the trial says that group actually gained **+0.63 points**. The
model's ranking is informative at the very top and noise everywhere else.

**We could not confirm sleeping dogs on this file, and on the men's campaign the
model was demonstrably wrong about them.** "Sleeping dogs" are customers whose
response *falls* when contacted — the unsubscribe, the annoyance, the reminder to
cancel. They are the strongest argument for uplift modelling, because they are the
one group where the right action is the opposite of the intuitive one, and a
conventional model cannot see them at all. Our simulation confirms these methods
find them when they genuinely exist.

On the men's campaign the model flags **7,916 customers (18.6%)** as harmed,
predicting an average of −0.87 points. The trial says that group actually gained
**+0.44 points (CI: +0.02 to +0.87)** — positive, and the interval excludes zero.
These are not sleeping dogs. On the women's campaign the flagged group
(8,029 customers, 18.8%) measures **−0.12 points (CI: −0.53 to +0.28)**, which
crosses zero: no harm demonstrated either way.

There is still a case for excluding the flagged group, but it is a different and
weaker case than the one the model makes. On the men's campaign those 7,916
customers generate only **$366** of incremental revenue — about $110 of margin —
against **$792** in contact costs. Dropping them saves roughly **$680**. They are
unprofitable, not harmed, and those call for different follow-up actions: an
unprofitable segment is a pricing and creative problem, a harmed segment is a
suppression problem.

---

## The budget frontier

Profit from mailing the top *N*% by predicted uplift, men's creative:

| Share of file mailed | Contacts | Cost | Incremental profit | Profit per 1,000 contacts |
|---|---|---|---|---|
| 5% | 2,131 | $213 | $983 | **$461** |
| 10% | 4,261 | $426 | $1,799 | $422 |
| **30% (current budget)** | **12,784** | **$1,278** | **$3,066** | **$240** |
| 50% | 21,306 | $2,131 | $4,384 | $206 |
| 70% | 29,829 | $2,983 | $6,115 | $205 |
| **85% (profit-maximising)** | **36,221** | **$3,622** | **$6,241** | **$172** |
| 100% | 42,613 | $4,261 | $5,580 | $131 |

Read it this way. Every contact is profitable on average — $0.13 each — so at
these economics the campaign should go to nearly everybody, and the last 15% is
the only genuinely unprofitable slice. **Efficiency per contact and total profit
point in opposite directions**: the top 5% is three and a half times as efficient
per contact as mailing everyone, but it only reaches 2,131 people and earns $983.

At the current 30% budget the recommendation is to spend all of it, choosing the
12,784 customers by predicted uplift: **$3,066 profit (95% CI: $782 to $5,535)**
against **$1,674** for a random 12,784. The bootstrap says the probability of
losing money at this depth is essentially zero.

But the constraint costs more than the model earns. Going from a 30% budget to a
70% budget doubles profit. If there is any argument for a larger e-mail budget,
this is it.

---

## The experiment to validate this

Do not roll this out as policy. Run it as a four-cell test.

- **Design.** Split the eligible file at random into a model-targeted cell and a
  business-as-usual cell. Inside each, hold back a randomly chosen 10% who receive
  nothing, so the incremental effect of each policy is *measured* rather than
  assumed. Without the holdbacks the test cannot answer the question.
- **Size.** To detect the difference we expect — 0.91% conversion under model
  targeting against 0.68% under the incumbent — at 95% confidence and 80% power:
  **40,900 customers per cell, 81,800 total, about five weeks at 20,000 mailable
  customers a week.**
- **Success criterion.** Set in advance: incremental profit per thousand contacts
  in the model cell exceeds the business-as-usual cell. Not "the model cell
  converted more" — that is the mistake this memo is about.
- **A cheaper version exists, but not here.** CUPED (subtracting off the part of
  each customer's behaviour that was predictable before the test began) can cut the
  required sample substantially. On a continuous metric like site visits it gave a
  **50% variance reduction — a doubling of effective sample size**. On a 1%-rate
  conversion metric it gave **1.7%**, which is nothing. Do not let anyone quote the
  first number to justify a smaller conversion test.

Note for calibration: each Hillstrom arm of 21,300 customers can only detect a
39% relative change in conversion. Under-powered e-mail tests are the norm.

---

## What would invalidate this

**The targeting gain failed a robustness check, and this is the main caveat.** We
refit the whole pipeline ten times on *randomly shuffled* treatment labels, where
there is no effect to find by construction. On the men's campaign the model
scored **9.4** on real data, while the fake-treatment runs averaged **2.8 with a
spread of 6.3** — and two of the ten fake runs scored *higher* than the real one.
The men's ranking is inside the range this method produces from pure noise. The
women's campaign passed the same check cleanly. Aggregate performance also swung
across random seeds (a Qini coefficient between −0.1 and 6.8), and reruns selected
only **44%** the same customers. The top decile is trustworthy; the ordering
beneath it is not, and the specific list should not be treated as stable.

**The economics are assumptions, not measurements.** $0.10 per contact and a 30%
margin are inputs. They move the answer a lot: at $0.25 per contact the optimal
depth collapses from 85% of the file to 35%, and at $0.50 mailing everyone *loses*
$11,500 while the optimal campaign is a marginal 2.5% of the file. **Before acting
on any depth in the table above, confirm the true fully-loaded cost per contact.**
That number matters more than the model does.

**Known limits.**
- The data is a 2008 US apparel retailer. Deliverability, inbox behaviour and
  fatigue have all changed. Every number here is a method demonstration on a real
  trial, not a forecast for your file.
- The outcome window is two weeks. Long-run effects — list fatigue, unsubscribes,
  brand erosion from over-mailing — are invisible in it, and they are precisely
  the mechanism that creates sleeping dogs. Our inability to confirm harm may be a
  two-week window problem rather than an absence of harm.
- Conversion runs at 0.9%. There are only 289 incremental conversions in the
  entire trial to learn heterogeneity from. That, not the choice of algorithm, is
  the binding constraint on everything above.
- Randomisation held: the arms are balanced on every covariate (largest
  standardised difference 0.014, well inside the 0.10 threshold). The experiment
  itself is sound; the limits are in what can be learned from it.

**What would change the recommendation:** a true contact cost above roughly $0.25;
a validation test where the model cell fails to beat business-as-usual; or a
longer outcome window showing that the flagged 18.6% are genuinely harmed, which
would make suppressing them worth considerably more than the $680 in saved contact
costs it is worth today.
