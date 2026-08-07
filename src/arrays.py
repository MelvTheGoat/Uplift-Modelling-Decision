"""A single array type alias, used for every numpy annotation in the project.

The dtype is intentionally left as ``Any``. These functions take covariates as
float arrays, treatment and binary outcomes as integer arrays, and outcomes that
are sometimes integer counts and sometimes continuous revenue — often through the
same parameter. Every one of them converts explicitly with ``np.asarray(...,
dtype=...)`` at the top of the body, so pinning a dtype in the signature would
document a restriction the code does not actually impose.

Writing it out as an alias rather than leaving ``np.ndarray`` bare keeps the
project clean under mypy's ``disallow_any_generics``, and makes the choice
visible and revisable instead of implicit.
"""

from __future__ import annotations

from typing import Any, TypeAlias

import numpy.typing as npt

#: An n-dimensional numpy array of unspecified dtype.
Array: TypeAlias = npt.NDArray[Any]
