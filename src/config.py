"""Shared paths, constants and economic assumptions.

Every number a business reader might argue with lives here rather than being
buried in a function body, so that a single edit re-runs the whole study under
different assumptions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

# ---------------------------------------------------------------- paths

ROOT: Final[Path] = Path(__file__).resolve().parent.parent
DATA_DIR: Final[Path] = ROOT / "data"
RESULTS_DIR: Final[Path] = ROOT / "results"

HILLSTROM_FILENAME: Final[str] = "hillstrom.csv"

#: Canonical source. Frequently unreachable from locked-down networks, hence the mirror.
HILLSTROM_URL: Final[str] = (
    "http://www.minethatdata.com/"
    "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
)
#: Byte-identical copy of the same file, used when the canonical host is blocked.
HILLSTROM_MIRROR_URL: Final[str] = (
    "https://raw.githubusercontent.com/bscan/uplift/master/"
    "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
)

#: Criteo-UPLIFT v2 is ~300 MB and gated behind a slow host; supported but never required.
CRITEO_URL: Final[str] = (
    "https://criteo-bucket.s3.eu-central-1.amazonaws.com/criteo-uplift-v2.1.csv.gz"
)
CRITEO_FILENAME: Final[str] = "criteo-uplift-v2.1.csv.gz"

# ---------------------------------------------------------------- reproducibility

RANDOM_SEED: Final[int] = 20080320  # the date on the Hillstrom file, for luck
N_BOOTSTRAP: Final[int] = 500
N_FOLDS: Final[int] = 5

# ---------------------------------------------------------------- economics
#
# Hillstrom records `spend` in dollars over the two weeks after the mailing.
# These are the assumptions the profit numbers rest on. They are stated in the
# memo and their sensitivity is tested in `robustness.py`.

#: Marginal cost of sending one e-mail (creative, ESP fees, list hygiene, deliverability risk).
COST_PER_CONTACT: Final[float] = 0.10

#: Gross margin retained per dollar of incremental revenue.
MARGIN_RATE: Final[float] = 0.30

#: Share of the file we can afford to contact in one campaign, if a budget is imposed.
DEFAULT_BUDGET_FRACTION: Final[float] = 0.30
