"""scikit-learn's own default for one parameter of an estimator class.

A trainer falls back to this value when the form leaves a parameter blank, so an
empty form fits the estimator scikit-learn would build and the default follows the
installed scikit-learn. A sentinel such as "deprecated" is scikit-learn's own
default and is passed through unchanged. The run seed is the one deliberate
exception: trainers set random_state from it so that a run reproduces.
"""

import inspect


def sk_default(cls, name: str):
    """The default of parameter `name` in `cls.__init__`."""
    return inspect.signature(cls.__init__).parameters[name].default
