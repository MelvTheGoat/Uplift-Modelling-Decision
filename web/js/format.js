/**
 * Number formatting.
 *
 * Small module, but it exists for a reason: the same quantity gets shown in
 * three places on three pages, and a reader who sees "0.0044", "0.44%" and
 * "+0.44pp" for one number reasonably concludes the site is confused.
 *
 * The convention throughout: rates are shown PER 100 PEOPLE. "0.44 extra
 * purchases per 100 people emailed" is a sentence a marketer can act on.
 * "0.0044" is not.
 */

/** True for null, undefined and NaN, which the results files use interchangeably. */
export function missing(value) {
  return value === null || value === undefined || (typeof value === "number" && !isFinite(value));
}

/** Format a dollar amount, with a leading minus rather than brackets. */
export function money(value, decimals = 0) {
  if (missing(value)) return "n/a";
  const sign = value < 0 ? "-" : "";
  return `${sign}$${Math.abs(value).toLocaleString("en-US", {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  })}`;
}

/** Format a proportion as a count per 100 people, e.g. 0.0044 -> "+0.44". */
export function per100(rate, decimals = 2, signed = true) {
  if (missing(rate)) return "n/a";
  const scaled = rate * 100;
  const body = Math.abs(scaled).toFixed(decimals);
  if (!signed) return scaled < 0 ? `-${body}` : body;
  return `${scaled < 0 ? "-" : "+"}${body}`;
}

/** Format a proportion as a percentage, e.g. 0.3 -> "30%". */
export function percent(fraction, decimals = 0) {
  if (missing(fraction)) return "n/a";
  return `${(fraction * 100).toFixed(decimals)}%`;
}

/** Format a whole number of people with thousands separators. */
export function count(value) {
  if (missing(value)) return "n/a";
  return Math.round(value).toLocaleString("en-US");
}

/** Fixed decimals with a forced sign, for scores that can go either way. */
export function signed(value, decimals = 1) {
  if (missing(value)) return "n/a";
  return `${value < 0 ? "−" : "+"}${Math.abs(value).toFixed(decimals)}`;
}

/** Plain decimals, no sign forcing. */
export function num(value, decimals = 1) {
  if (missing(value)) return "n/a";
  return value.toFixed(decimals);
}

/**
 * A p-value with its meaning spelled out.
 *
 * A bare "0.056" invites the reader to round it down to 0.05 and move on.
 * Saying what it means makes that harder.
 */
export function pValue(p) {
  if (missing(p)) return "n/a";
  let verdict;
  if (p < 0.01) verdict = "well clear of the usual bar";
  else if (p < 0.05) verdict = "clears the usual bar";
  else if (p < 0.1) verdict = "just misses the usual bar";
  else verdict = "nowhere near the usual bar";
  return `${p.toFixed(3)} — ${verdict}`;
}

/** Very small p-values, in scientific notation a reader can still parse. */
export function tinyP(p) {
  if (missing(p)) return "n/a";
  if (p >= 0.001) return p.toFixed(4);
  const exponent = Math.floor(Math.log10(p));
  const mantissa = p / Math.pow(10, exponent);
  return `${mantissa.toFixed(1)} × 10^${exponent}`;
}

/** A difference in dollars, with a direction word instead of a sign. */
export function moneyDelta(value) {
  if (missing(value)) return "n/a";
  return `${money(Math.abs(value))} ${value >= 0 ? "better" : "worse"}`;
}

/** Plain-English display names for the study keys used in the results files. */
export const STUDY_LABELS = {
  mens_conversion: "Men's email → purchases",
  womens_conversion: "Women's email → purchases",
  mens_visit: "Men's email → site visits",
  womens_visit: "Women's email → site visits",
};

/** One line on what each study measured and why it is in here. */
export const STUDY_BLURBS = {
  mens_conversion:
    "The headline question. Purchases are what pay for the campaign, so this is the one that decides anything.",
  womens_conversion:
    "The same question for the other campaign. A useful second opinion: if a method works on one and collapses on the other, that tells you something.",
  mens_visit:
    "Did the email make people visit the site? Far more people visit than buy, which makes this an easier question — and a poor substitute for the one that matters.",
  womens_visit: "Site visits for the women's campaign.",
};

/**
 * Plain-English display names for the four models.
 *
 * Each ends in "method" rather than being a bare noun phrase, because these
 * names appear at the start of sentences. "Two models flagged 7,916 customers"
 * reads as a count of models; "Two-model method flagged 7,916 customers" does
 * not.
 */
export const MODEL_LABELS = {
  "s-learner": "Single-model method",
  "t-learner": "Two-model method",
  "x-learner": "Cross-fitted method",
  "causal-forest": "Causal forest",
};

/** How each model works, with no maths. */
export const MODEL_BLURBS = {
  "s-learner":
    "Fit one model on everybody, with “did they get the email?” as just another column. Ask it twice per person — once pretending they got it, once pretending they did not — and subtract.",
  "t-learner":
    "Fit two separate models: one on the people who got the email, one on the people who did not. Subtract their predictions.",
  "x-learner":
    "A two-stage trick. Use each group’s model to estimate what the other group missed out on, then blend the two answers. Built for campaigns where one group is much bigger than the other.",
  "causal-forest":
    "A forest of trees that split on “who does the email affect most?” rather than “who buys?”. Each tree holds back part of its data so its own estimates stay honest.",
};

/** Display name for a study key, falling back to the key itself. */
export function studyLabel(key) {
  return STUDY_LABELS[key] ?? key.replace(/_/g, " ");
}

/** Display name for a model, falling back to the raw name. */
export function modelLabel(name) {
  return MODEL_LABELS[name] ?? name;
}
