"""Label-flip, sign-flip, scaled-update, model-replacement attack clients."""

from __future__ import annotations

from .clients import ATTACK_TYPES, ByzantineClient

__all__ = ["ByzantineClient", "ATTACK_TYPES"]
