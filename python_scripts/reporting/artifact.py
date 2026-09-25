"""Saving and loading fitted objects without executing what the file says.

joblib writes a pickle, and loading a pickle runs the instructions inside it. A
signature over the bytes says a file was written by an installation that holds
the key; it says nothing about a file whose signer was compromised, and nothing
at all about a file from before signing existed.

skops writes the same objects as data with their types named, and refuses to
build a type the caller has not allowed. This module holds the allowlist: the
exact scikit-learn, numpy, scipy and pandas type names a census of every
registered estimator produced (artifact_allowlist.txt), plus the few types of
this project that a fitted preprocessor refers to. A file naming anything else is
refused before a single object is built, and the message names what it asked for.

Format v2 bundles carry model.skops and load through here. Format v1 bundles carry
model.joblib; they are read only when DIGITERRA_ALLOW_LEGACY_PICKLE is set, since
reading one runs its contents.
"""

from __future__ import annotations

import io
import logging
import os

logger = logging.getLogger(__name__)

ARTIFACT_NAME = "model.skops"
LEGACY_ARTIFACT_NAME = "model.joblib"

# The types a DiGiTerra artifact may name. The list is exact, not a prefix: it was
# taken from a census of every registered estimator written through this module
# (scripts/artifact_type_census.py writes artifact_allowlist.txt), plus the few
# types of this project that a fitted preprocessor refers to. A prefix test was
# what the review of 2026-09-24 defeated: "builtins." admitted builtins.eval, and
# the loader then handed skops the file's own type list as trusted, so a crafted
# file ran code on load. Now the file is read against this list and skops is given
# this list, never the file's.
_ALLOWLIST_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifact_allowlist.txt")

OWN_TRUSTED_TYPES = (
    "python_scripts.preprocessing.utilities.apply_group_map",
    "python_scripts.models.regression_models.train_radius_neighbors_regressor.RadiusNeighborsWithFallback",
    "python_scripts.preprocessing.applicability.Applicability",
    "python_scripts.preprocessing.uncertainty.Conformal",
    "python_scripts.reporting.birch_centroids.BirchCentroids",
    "python_scripts.models.regression_models.train_polynomial.PolynomialRegression",
)


def _read_allowlist() -> frozenset[str]:
    names = set(OWN_TRUSTED_TYPES)
    try:
        with open(_ALLOWLIST_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    names.add(line)
    except OSError as e:
        logger.error("Artifact allowlist not read (%s); only the project's own types load", e)
    return frozenset(names)


TRUSTED_TYPES: frozenset[str] = _read_allowlist()

# Names that never load whatever a census says: each is a way to run code.
_NEVER = (
    "builtins.eval",
    "builtins.exec",
    "builtins.__import__",
    "builtins.compile",
    "builtins.open",
    "os.",
    "subprocess.",
)


# Estimators skops cannot describe as data. A bundle of one of these cannot be
# written without a pickle, so the export says so rather than falling back to one.
# Checked by scripts/check_artifact_roundtrip.py, which names any that appear.
# Birch was here until 2026-09-25; it is now written as BirchCentroids (below).
UNSUPPORTED_ESTIMATORS: tuple[str, ...] = ()


class ArtifactError(ValueError):
    """The file asks for something outside the allowlist, or is not readable."""


def legacy_pickle_allowed() -> bool:
    return os.environ.get("DIGITERRA_ALLOW_LEGACY_PICKLE", "").strip().lower() in {"1", "true", "yes", "on"}


def _is_trusted(name: str) -> bool:
    if any(name == n or name.startswith(n) for n in _NEVER):
        return False
    return name in TRUSTED_TYPES


def dumps(objects: dict) -> bytes:
    """Serialize the fitted objects. Raises ArtifactError when they cannot be."""
    import skops.io as sio

    objects = _describable(objects)
    try:
        return sio.dumps(objects)
    except Exception as e:  # a type skops cannot describe as data
        model_name = type(objects.get("model")).__name__ if isinstance(objects, dict) else "the model"
        if model_name in UNSUPPORTED_ESTIMATORS:
            raise ArtifactError(
                f"A {model_name} model cannot be written in a format that does not execute on load, so this run "
                "cannot be exported as a bundle. Its results, model card and manifest are unaffected."
            ) from e
        raise ArtifactError(f"The fitted objects could not be written without a pickle: {e}") from e


def _describable(objects: dict) -> dict:
    """Replace a fitted Birch, which skops refuses, with its data-only prediction state."""
    if not isinstance(objects, dict):
        return objects
    from sklearn.cluster import Birch

    model = objects.get("model")
    if isinstance(model, Birch):
        from python_scripts.reporting.birch_centroids import BirchCentroids

        return {**objects, "model": BirchCentroids.from_birch(model)}
    return objects


def untrusted_types(data: bytes) -> list[str]:
    """Types the file asks for that this application does not allow."""
    import skops.io as sio

    try:
        asked = sio.get_untrusted_types(data=data)
    except Exception as e:
        raise ArtifactError(f"The artifact is not readable: {e}") from e
    return sorted(name for name in asked if not _is_trusted(name))


def loads(data: bytes) -> dict:
    """Build the fitted objects, or refuse and name what the file asked for."""
    import skops.io as sio

    refused = untrusted_types(data)
    if refused:
        raise ArtifactError(
            "The artifact asks for types this application does not load: " + ", ".join(refused[:5]) + "."
        )
    try:
        # skops is given the allowlist, never the file's own type list.
        objects = sio.loads(data, trusted=sorted(TRUSTED_TYPES))
    except Exception as e:
        raise ArtifactError(f"The artifact could not be read: {e}") from e
    if not isinstance(objects, dict) or "model" not in objects:
        raise ArtifactError("The artifact does not hold DiGiTerra fitted objects.")
    return objects


def loads_legacy(data: bytes) -> dict:
    """Read a format v1 artifact, which is a pickle.

    Reading it runs code from the file, so it is off unless
    DIGITERRA_ALLOW_LEGACY_PICKLE is set. It exists to open bundles written before
    the format changed, from a source the operator already trusts.
    """
    if not legacy_pickle_allowed():
        raise ArtifactError(
            "This bundle uses the older format. Its artifact is a pickle, which runs code when it is read. "
            "Export the bundle again from the installation that made it. "
            "To open a bundle from a source you trust, set DIGITERRA_ALLOW_LEGACY_PICKLE to 1."
        )
    import joblib

    objects = joblib.load(io.BytesIO(data))
    if not isinstance(objects, dict) or "model" not in objects:
        raise ArtifactError("The artifact does not hold DiGiTerra fitted objects.")
    return objects


class _CountingSink:
    """A write-only file object that records the byte count and keeps nothing."""

    def __init__(self):
        self.n = 0

    def write(self, b) -> int:
        self.n += len(b)
        return len(b)

    def tell(self) -> int:
        return self.n

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass

    def seekable(self) -> bool:
        return False

    def writable(self) -> bool:
        return True


def serialized_size(objects: dict) -> int:
    """Bytes the objects take when written, without holding them.

    skops builds its output in memory, so the size is measured from the bytes it
    returns; joblib is used only as a fallback for objects skops cannot describe,
    and then only to measure, never to store.
    """
    try:
        return len(dumps(objects))
    except ArtifactError:
        import joblib

        sink = _CountingSink()
        joblib.dump(objects, sink)
        return sink.n
