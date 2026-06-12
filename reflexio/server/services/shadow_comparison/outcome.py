"""Pure F1 outcome helpers — position randomization + win/loss/tie derivation.

These functions are deliberately DB-free and LLM-free so they can be unit
tested in milliseconds and exercised by mutmut.
"""

from __future__ import annotations

import random
from enum import StrEnum

from reflexio.models.api_schema.eval_overview_schema import ShadowComparisonVerdict


class Outcome(StrEnum):
    """Reflexio-relative outcome derived from a judge's position-randomized verdict.

    Uses StrEnum so values serialize transparently to JSON / SQL.
    """

    WIN = "win"
    LOSS = "loss"
    TIE = "tie"


def assign_positions(
    reflexio_response: str,
    shadow_response: str,
    rng: random.Random,
) -> tuple[str, str, bool]:
    """
    Randomize which response is shown as "Request 1" to the judge.

    The judge prompt is blind to the assignment — it sees the two strings
    labeled "REQUEST 1" and "REQUEST 2". We record the assignment so the
    dashboard can derive the Reflexio-relative outcome via
    `derive_reflexio_outcome`.

    Args:
        reflexio_response (str): The response generated with Reflexio rules in context.
        shadow_response (str): The response generated without Reflexio rules.
        rng (random.Random): Seeded random for reproducibility (tests inject a
            seed; production uses `random.Random()` per judge call).

    Returns:
        tuple[str, str, bool]: (request_1_response, request_2_response,
            reflexio_is_request_1). Storage records the third element on the
            verdict row.
    """
    reflexio_is_request_1 = rng.random() < 0.5
    if reflexio_is_request_1:
        return reflexio_response, shadow_response, True
    return shadow_response, reflexio_response, False


def derive_reflexio_outcome(verdict: ShadowComparisonVerdict) -> Outcome:
    """
    Map a position-randomized verdict to a Reflexio-relative outcome.

    Truth table:
        better=='tie'                                  -> TIE
        better=='1' AND reflexio_is_request_1==True    -> WIN
        better=='1' AND reflexio_is_request_1==False   -> LOSS
        better=='2' AND reflexio_is_request_1==True    -> LOSS
        better=='2' AND reflexio_is_request_1==False   -> WIN

    Args:
        verdict (ShadowComparisonVerdict): The stored verdict, including the
            judge's `better_request` choice and the `reflexio_is_request_1`
            position-randomization record.

    Returns:
        Outcome: WIN / LOSS / TIE relative to the Reflexio response.
    """
    if verdict.output.better_request == "tie":
        return Outcome.TIE
    is_reflexio_better = (
        verdict.output.better_request == "1"
    ) == verdict.reflexio_is_request_1
    return Outcome.WIN if is_reflexio_better else Outcome.LOSS
