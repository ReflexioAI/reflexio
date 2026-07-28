import pytest
from pydantic import BaseModel

from reflexio.server.llm._litellm_types import ModelProvenance
from reflexio.server.services.extraction.resumable_agent import (
    decode_committed_output,
    encode_committed_output,
)


class _Output(BaseModel):
    value: str


def test_committed_output_envelope_round_trips_provenance():
    provenance = ModelProvenance(
        model_name="served",
        provider="provider",
    )

    encoded = encode_committed_output(_Output(value="accepted"), provenance)

    output, decoded_provenance = decode_committed_output(encoded)
    assert output == {"value": "accepted"}
    assert decoded_provenance == provenance


def test_legacy_raw_output_with_colliding_keys_is_not_unwrapped():
    legacy = {
        "output": {"value": "legacy"},
        "model_provenance": {"provider": "not-an-envelope"},
    }

    output, provenance = decode_committed_output(legacy)

    assert output is legacy
    assert provenance is None


def test_v1_envelope_without_provenance_still_unwraps_output():
    output, provenance = decode_committed_output(
        {
            "_reflexio_envelope_version": 1,
            "output": {"value": "accepted"},
        }
    )

    assert output == {"value": "accepted"}
    assert provenance is None


@pytest.mark.parametrize(
    "model_provenance",
    ["not-an-object", {"provider": "provider", "unexpected": "field"}],
)
def test_v1_envelope_with_malformed_provenance_is_corrupt(model_provenance):
    with pytest.raises(
        ValueError, match="Corrupt v1 committed output envelope:.*model_provenance"
    ):
        decode_committed_output(
            {
                "_reflexio_envelope_version": 1,
                "output": {"value": "accepted"},
                "model_provenance": model_provenance,
            }
        )


@pytest.mark.parametrize("output", [None, "not-an-object"])
def test_v1_envelope_with_invalid_output_is_corrupt(output):
    with pytest.raises(ValueError, match="Corrupt v1 committed output envelope"):
        decode_committed_output(
            {
                "_reflexio_envelope_version": 1,
                "output": output,
            }
        )


def test_unknown_envelope_version_is_rejected():
    future = {
        "_reflexio_envelope_version": 2,
        "output": {"value": "future"},
        "model_provenance": None,
    }

    with pytest.raises(
        ValueError, match="Unsupported committed output envelope version: 2"
    ):
        decode_committed_output(future)
