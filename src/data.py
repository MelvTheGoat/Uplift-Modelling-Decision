"""Loading and preparing the real datasets.

Primary: the Hillstrom MineThatData e-mail challenge. 64,000 customers who had
purchased in the previous twelve months, randomly split three ways two weeks
before the outcome window:

* ``Mens E-Mail``   — 21,307 customers, e-mail featuring men's merchandise
* ``Womens E-Mail`` — 21,387 customers, e-mail featuring women's merchandise
* ``No E-Mail``     — 21,306 customers, contacted with nothing

Because assignment was randomised, this file supports causal claims directly:
no adjustment is needed for the difference between arms to mean something. What
it does *not* hand us is the individual effect, which is why the estimators are
validated on synthetic data first.

Secondary (optional, behind a flag): Criteo-UPLIFT v2. Large and slow to fetch;
nothing in the study blocks on it.
"""

from __future__ import annotations

import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.arrays import Array
from src.config import (
    CRITEO_FILENAME,
    CRITEO_URL,
    DATA_DIR,
    HILLSTROM_FILENAME,
    HILLSTROM_MIRROR_URL,
    HILLSTROM_URL,
)

#: Arm labels exactly as they appear in the `segment` column of the raw file.
CONTROL_LABEL = "No E-Mail"
WOMENS_LABEL = "Womens E-Mail"
MENS_LABEL = "Mens E-Mail"

#: Columns used as model features, after encoding.
NUMERIC_FEATURES = ["recency", "history", "mens", "womens", "newbie"]
CATEGORICAL_FEATURES = ["zip_code", "channel"]


@dataclass(frozen=True)
class UpliftDataset:
    """A binary-treatment uplift dataset ready for modelling.

    Attributes:
        features: Encoded covariate matrix, shape ``(n, d)``.
        treatment: Binary assignment, 1 = mailed.
        outcome: Binary response used as the uplift target.
        spend: Revenue per customer over the outcome window.
        feature_names: Column names matching ``features``.
        outcome_name: Which column of the raw file ``outcome`` came from.
        treatment_name: Human-readable description of the treated arm.
        frame: The filtered raw rows, for diagnostics and balance checks.
    """

    features: Array
    treatment: Array
    outcome: Array
    spend: Array
    feature_names: list[str]
    outcome_name: str
    treatment_name: str
    frame: pd.DataFrame

    @property
    def n(self) -> int:
        """Number of rows."""
        return int(self.features.shape[0])

    @property
    def treated_share(self) -> float:
        """Fraction of rows in the treated arm."""
        return float(self.treatment.mean())


def _download(url: str, destination: Path, timeout: int = 120) -> None:
    """Fetch ``url`` to ``destination``, writing atomically.

    Args:
        url: Source URL.
        destination: Local path to write.
        timeout: Socket timeout in seconds.

    Raises:
        urllib.error.URLError: If the host is unreachable or blocked.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
        partial.write_bytes(response.read())
    partial.replace(destination)


def fetch_hillstrom(data_dir: Path = DATA_DIR, force: bool = False) -> Path:
    """Download the Hillstrom CSV if it is not already cached.

    Tries the canonical minethatdata.com URL first and falls back to a
    byte-identical GitHub mirror, because the canonical host is plain HTTP and is
    blocked by many corporate and CI egress policies.

    Args:
        data_dir: Directory to cache into.
        force: Re-download even if the file is present.

    Returns:
        Path to the cached CSV.

    Raises:
        RuntimeError: If neither source can be reached.
    """
    destination = data_dir / HILLSTROM_FILENAME
    if destination.exists() and not force:
        return destination

    errors: list[str] = []
    for url in (HILLSTROM_URL, HILLSTROM_MIRROR_URL):
        try:
            _download(url, destination)
        except (urllib.error.URLError, OSError, ValueError) as exc:  # noqa: PERF203
            errors.append(f"{url}: {exc}")
        else:
            return destination

    joined = "\n  ".join(errors)
    raise RuntimeError(
        "Could not download the Hillstrom dataset from either source:\n  "
        f"{joined}\n"
        f"Download it manually and place it at {destination}."
    )


def load_hillstrom_raw(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Read the raw Hillstrom file, downloading it on first use.

    Args:
        data_dir: Cache directory.

    Returns:
        The raw 64,000-row frame, unmodified.
    """
    return pd.read_csv(fetch_hillstrom(data_dir))


def encode_features(frame: pd.DataFrame) -> tuple[Array, list[str]]:
    """One-hot encode the categoricals and stack them with the numeric columns.

    ``history_segment`` is dropped: it is a coarse binning of ``history``, which is
    already included at full resolution, so keeping both adds collinearity without
    adding information.

    Args:
        frame: Raw or filtered Hillstrom rows.

    Returns:
        A tuple of the encoded matrix and its column names.
    """
    numeric = frame[NUMERIC_FEATURES].astype(np.float64)
    dummies = pd.get_dummies(
        frame[CATEGORICAL_FEATURES], prefix=CATEGORICAL_FEATURES, drop_first=False
    ).astype(np.float64)
    encoded = pd.concat([numeric, dummies], axis=1)
    return encoded.to_numpy(dtype=np.float64), list(encoded.columns)


def load_hillstrom(
    treatment_arm: str = "womens",
    outcome: str = "conversion",
    data_dir: Path = DATA_DIR,
) -> UpliftDataset:
    """Build a two-arm uplift dataset from the three-arm Hillstrom trial.

    The third arm is dropped rather than pooled: the men's and women's creatives
    are different campaigns with different effects, and averaging them would
    answer a question nobody asked. Pooling is available via ``treatment_arm="any"``
    for completeness, with the caveat that the resulting "treatment" is a mixture.

    Args:
        treatment_arm: One of ``"womens"``, ``"mens"``, or ``"any"``.
        outcome: Which column to use as the binary uplift target — ``"conversion"``
            (bought something, ~0.9% base rate) or ``"visit"`` (came to the site,
            ~14.7% base rate).
        data_dir: Cache directory.

    Returns:
        An :class:`UpliftDataset`.

    Raises:
        ValueError: On an unknown arm or outcome name.
    """
    if outcome not in {"conversion", "visit"}:
        raise ValueError(f"outcome must be 'conversion' or 'visit', got {outcome!r}")

    raw = load_hillstrom_raw(data_dir)
    arms = {"womens": WOMENS_LABEL, "mens": MENS_LABEL}

    if treatment_arm == "any":
        frame = raw.copy()
        treated_labels = {WOMENS_LABEL, MENS_LABEL}
        treatment_name = "any e-mail (men's and women's pooled)"
    elif treatment_arm in arms:
        label = arms[treatment_arm]
        frame = raw[raw["segment"].isin([label, CONTROL_LABEL])].copy()
        treated_labels = {label}
        treatment_name = label
    else:
        raise ValueError(f"treatment_arm must be 'womens', 'mens' or 'any', got {treatment_arm!r}")

    frame = frame.reset_index(drop=True)
    treatment = frame["segment"].isin(treated_labels).to_numpy().astype(np.int64)
    features, names = encode_features(frame)

    return UpliftDataset(
        features=features,
        treatment=treatment,
        outcome=frame[outcome].to_numpy().astype(np.int64),
        spend=frame["spend"].to_numpy().astype(np.float64),
        feature_names=names,
        outcome_name=outcome,
        treatment_name=treatment_name,
        frame=frame,
    )


def load_criteo(
    data_dir: Path = DATA_DIR,
    sample_rows: int | None = 500_000,
) -> UpliftDataset:
    """Load Criteo-UPLIFT v2, if the ~300 MB file can be fetched.

    Optional throughout: no part of the study depends on it. Criteo is a much
    larger randomised advertising trial (13M rows, 12 anonymised features) and is
    useful as a second opinion on whether the ranking methods generalise beyond a
    single retailer's e-mail file.

    Args:
        data_dir: Cache directory.
        sample_rows: Read only the first N rows to keep memory sane; ``None`` reads all.

    Returns:
        An :class:`UpliftDataset` with ``conversion`` as the outcome.

    Raises:
        RuntimeError: If the file cannot be downloaded.
    """
    destination = data_dir / CRITEO_FILENAME
    if not destination.exists():
        try:
            _download(CRITEO_URL, destination, timeout=600)
        except (urllib.error.URLError, OSError) as exc:
            raise RuntimeError(
                f"Criteo download failed ({exc}). This dataset is optional — "
                "the study runs without it."
            ) from exc

    frame = pd.read_csv(destination, nrows=sample_rows)
    feature_names = [c for c in frame.columns if c.startswith("f")]
    return UpliftDataset(
        features=frame[feature_names].to_numpy(dtype=np.float64),
        treatment=frame["treatment"].to_numpy().astype(np.int64),
        outcome=frame["conversion"].to_numpy().astype(np.int64),
        spend=frame["conversion"].to_numpy().astype(np.float64),
        feature_names=feature_names,
        outcome_name="conversion",
        treatment_name="criteo advertising treatment",
        frame=frame,
    )


def dataset_summary(dataset: UpliftDataset) -> dict[str, float | str]:
    """Headline counts and rates for a loaded dataset.

    Args:
        dataset: A loaded uplift dataset.

    Returns:
        Mapping of summary name to value.
    """
    treated = dataset.treatment == 1
    return {
        "treatment": dataset.treatment_name,
        "outcome": dataset.outcome_name,
        "n": float(dataset.n),
        "n_treated": float(treated.sum()),
        "n_control": float((~treated).sum()),
        "outcome_rate_treated": float(dataset.outcome[treated].mean()),
        "outcome_rate_control": float(dataset.outcome[~treated].mean()),
        "spend_treated": float(dataset.spend[treated].mean()),
        "spend_control": float(dataset.spend[~treated].mean()),
        "n_features": float(len(dataset.feature_names)),
    }
