"""Saving and loading fitted objects without executing what the file says.

joblib writes a pickle, and loading a pickle runs the instructions inside it. A
signature over the bytes says a file was written by an installation that holds
the key; it says nothing about a file whose signer was compromised, and nothing
at all about a file from before signing existed.

skops writes the same objects as data with their types named, and refuses to
build a type the caller has not allowed. This module holds the allowlist: the
scikit-learn and numpy types DiGiTerra actually fits, plus the one function of its
own that a fitted preprocessor refers to. A file naming anything else is refused
before a single object is built, and the message names what it asked for.

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

# The one callable of this project that a fitted object refers to: the group-map
# step of the preprocessor. Anything else named by a file is refused.
OWN_TRUSTED_TYPES = (
    "python_scripts.preprocessing.utilities.apply_group_map",
    "python_scripts.models.regression_models.train_radius_neighbors_regressor._NearestFallbackRadiusNeighbors",
    "numpy.dtype",
    "numpy.ufunc",
)

# Type names from these modules are the estimators, transformers, and arrays a run
# fits. skops reports each unknown type by its dotted path, so a prefix test is
# what decides, and it is deliberately narrow: nothing outside these trees loads.
TRUSTED_PREFIXES = (
    "sklearn.",
    "numpy.",
    "scipy.",
    "pandas.",
    "collections.OrderedDict",
    "builtins.",
)


class ArtifactError(ValueError):
    """The file asks for something outside the allowlist, or is not readable."""


def legacy_pickle_allowed() -> bool:
    return os.environ.get("DIGITERRA_ALLOW_LEGACY_PICKLE", "").strip().lower() in {"1", "true", "yes", "on"}


def _is_trusted(name: str) -> bool:
    if name in OWN_TRUSTED_TYPES:
        return True
    return any(name.startswith(prefix) for prefix in TRUSTED_PREFIXES)


def dumps(objects: dict) -> bytes:
    """Serialize the fitted objects. Raises ArtifactError when they cannot be."""
    import skops.io as sio

    try:
        return sio.dumps(objects)
    except Exception as e:  # a type skops cannot describe as data
        raise ArtifactError(f"The fitted objects could not be written without a pickle: {e}") from e


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
        asked = sio.get_untrusted_types(data=data)
        objects = sio.loads(data, trusted=asked)
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
            "This bundle is in the older format, whose artifact is a pickle that runs code when it is read. "
            "Set DIGITERRA_ALLOW_LEGACY_PICKLE=1 to open one from a source you trust, or export it again "
            "from the installation that made it."
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
