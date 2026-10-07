"""Bundle everything in ``results/`` into one JSON file for the web front end.

Why a build step at all, when the browser could fetch the result files directly?

* **One request instead of twenty-odd.** The findings are spread across four JSON
  files and twenty-seven CSVs. Fetching them individually on page load means
  twenty-seven round trips before anything renders.
* **The site becomes self-contained.** Everything the front end needs lives under
  ``web/``, so that directory can be dropped on any static host — GitHub Pages,
  Netlify, Cloudflare Pages, an S3 bucket — with nothing else alongside it.
* **CSV parsing stays in Python.** Browsers have no CSV parser, and hand-rolling
  one that handles quoting correctly is a waste of everyone's afternoon.

Run it after regenerating the study::

    python -m src.cli all
    python scripts/build_web_data.py

The output is committed, so a fresh clone can serve the site without running
either step. Regenerate and commit it whenever the results change.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import RESULTS_DIR, ROOT

#: Where the bundle is written. The front end fetches it as a sibling of index.html.
OUTPUT_PATH = ROOT / "web" / "data.json"

#: Top-level JSON artefacts, copied across as-is.
JSON_FILES = ("hillstrom.json", "naive.json", "validation.json", "experiment.json")

#: CSVs keyed by study. The ``{study}`` placeholder is filled from hillstrom.json's keys.
PER_STUDY_CSV = {
    "balance": "balance_{study}.csv",
    "cost_sensitivity": "cost_sensitivity_{study}.csv",
}

#: CSVs keyed by study *and* model.
PER_MODEL_CSV = {"deciles": "deciles_{study}_{model}.csv"}


def _clean(value: Any) -> Any:
    """Convert a value to something ``json.dump`` will accept.

    pandas hands back numpy scalars, and NaN is not valid JSON — ``json.dump``
    will happily emit the literal ``NaN``, which then throws a syntax error in
    ``JSON.parse`` on the other side. Both become ``null``.

    Args:
        value: A scalar from a frame or a nested structure.

    Returns:
        A JSON-safe equivalent.
    """
    if isinstance(value, dict):
        return {str(key): _clean(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(inner) for inner in value]
    if value is None:
        return None
    if isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        return None if math.isnan(value) or math.isinf(value) else value
    # numpy scalars and anything else with a Python equivalent.
    item = getattr(value, "item", None)
    return _clean(item()) if callable(item) else value


def _read_frame(name: str) -> list[dict[str, Any]] | None:
    """Read one CSV from ``results/`` as a list of row dictionaries.

    Args:
        name: File name inside ``results/``.

    Returns:
        The rows, or None when the file does not exist. A missing per-model CSV
        is normal — a run with ``--quick`` skips the slow estimators — so this
        is not an error.
    """
    path = RESULTS_DIR / name
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    return [_clean(row) for row in frame.to_dict(orient="records")]


def build() -> dict[str, Any]:
    """Assemble the bundle.

    Returns:
        A mapping with one key per source artefact, plus a ``meta`` block.

    Raises:
        FileNotFoundError: If ``hillstrom.json`` is missing, since every page
            depends on it.
    """
    if not (RESULTS_DIR / "hillstrom.json").exists():
        raise FileNotFoundError(
            f"{RESULTS_DIR / 'hillstrom.json'} not found. Run `python -m src.cli all` first."
        )

    bundle: dict[str, Any] = {}
    for name in JSON_FILES:
        path = RESULTS_DIR / name
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            bundle[path.stem] = _clean(json.load(handle))

    studies = list(bundle["hillstrom"].keys())

    for key, template in PER_STUDY_CSV.items():
        bundle[key] = {
            study: rows
            for study in studies
            if (rows := _read_frame(template.format(study=study))) is not None
        }

    for key, template in PER_MODEL_CSV.items():
        nested: dict[str, dict[str, Any]] = {}
        for study in studies:
            models = [row["model"] for row in bundle["hillstrom"][study]["leaderboard"]]
            per_model = {
                model: rows
                for model in models
                if (rows := _read_frame(template.format(study=study, model=model))) is not None
            }
            if per_model:
                nested[study] = per_model
        bundle[key] = nested

    # Validation CSVs carry the per-model recovery diagnostics for the synthetic
    # scenarios; the JSON holds the same numbers, so only the scenario names are
    # needed here to drive the page's selector.
    bundle["meta"] = {
        "studies": studies,
        "validation_scenarios": list(bundle.get("validation", {}).keys()),
    }
    return bundle


def main() -> None:
    """Write the bundle to ``web/data.json``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="Where to write the bundle (default: web/data.json)",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=0,
        help="JSON indentation. 0 writes the compact form, which is what ships.",
    )
    args = parser.parse_args()

    bundle = build()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        bundle,
        indent=args.indent or None,
        separators=(",", ":") if not args.indent else None,
        allow_nan=False,
    )
    args.output.write_text(text + "\n", encoding="utf-8")

    size_kb = args.output.stat().st_size / 1024.0
    print(f"wrote {args.output} ({size_kb:,.0f} KB)")
    print(f"  studies: {', '.join(bundle['meta']['studies'])}")
    print(f"  top-level keys: {', '.join(sorted(bundle))}")


if __name__ == "__main__":
    main()
