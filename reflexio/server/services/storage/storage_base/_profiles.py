"""Residual base (ABC) profile module.

All abstract methods have been extracted into the ``profiles`` package
(``ProfileStoreMixin``, ``InteractionStoreMixin``, ``ProfileSearchMixin``). Only
the now-drained ``ProfileMixin`` class remains; Task 6 removes the class and its
composition.
"""


class ProfileMixin:
    """Drained residual mixin — all abstracts moved to the ``profiles`` package."""
