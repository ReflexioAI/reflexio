"""Raw document representation protects numeric values at publication boundaries."""

from decimal import Decimal

import pytest

from reflexio.config_document import decode_config_document, encode_config_document


def test_exact_nested_numbers_and_boolean_integer_distinction():
    raw = '{"fraction":0.12345678901234567890123456789,"scaled":1.00,"huge":1e1000,"integer":123456789012345678901234567890,"bool":true,"nested":[{"value":0.12345678901234567890123456789}]}'
    before = decode_config_document(raw)
    after = decode_config_document(encode_config_document(before))
    assert after == before
    for key in ("fraction", "scaled", "huge"):
        assert type(after[key]) is Decimal
        assert after[key].as_tuple() == before[key].as_tuple()
    assert type(after["integer"]) is int
    assert type(after["bool"]) is bool and after["bool"] is True
    assert after["nested"][0]["value"].as_tuple() == before["fraction"].as_tuple()


@pytest.mark.parametrize(
    "value", [0.1, 1e-6, 1e20, -0.0, Decimal("1.00"), Decimal("1e-1000"), True, 1]
)
def test_model_numbers_have_idempotent_wire_representation(value):
    encoded = encode_config_document({"value": value})
    assert encode_config_document(decode_config_document(encoded)) == encoded
    assert encode_config_document({"value": True}) != encode_config_document(
        {"value": 1}
    )
    assert encode_config_document({"value": 1.0}) != encode_config_document(
        {"value": 1}
    )


@pytest.mark.parametrize(
    "raw",
    [
        '{"secret":"private-value","number":NaN}',
        '{"secret":"private-value","number":Infinity}',
        '{"secret":"private-value","number":-Infinity}',
        "private-value",
        "null",
        "[]",
        "true",
        "1",
    ],
)
def test_invalid_raw_documents_refuse_without_values(raw):
    with pytest.raises(ValueError) as error:
        decode_config_document(raw)
    assert "private" not in str(error.value)


@pytest.mark.parametrize(
    "value",
    [
        float("nan"),
        float("inf"),
        Decimal("NaN"),
        Decimal("sNaN"),
        Decimal("Infinity"),
        Decimal("-Infinity"),
        object(),
    ],
)
def test_nonfinite_or_non_json_values_cannot_be_published(value):
    with pytest.raises(ValueError) as error:
        encode_config_document({"secret": "private-value", "nested": [value]})
    assert "private" not in str(error.value)


@pytest.mark.parametrize("score", [0.1, 1e-6, 1.0])
def test_actual_model_float_and_opaque_decimal_have_stable_readback(score):
    from reflexio.models.config_schema import PlaybookOptimizerConfig

    payload = {
        "playbook_optimizer_config": PlaybookOptimizerConfig(
            early_stop_score=score
        ).model_dump(mode="json"),
        "extension": {"fraction": Decimal("0.12345678901234567890123456789")},
    }
    encoded = encode_config_document(payload)
    decoded = decode_config_document(encoded)
    assert encode_config_document(decoded) == encoded
    assert (
        PlaybookOptimizerConfig.model_validate(
            decoded["playbook_optimizer_config"]
        ).early_stop_score
        == score
    )
    assert (
        decoded["extension"]["fraction"].as_tuple()
        == payload["extension"]["fraction"].as_tuple()
    )


@pytest.mark.parametrize("accelerated", [False, True])
def test_parser_limits_are_sanitized_in_both_implementations(accelerated):
    import sys

    import simplejson

    vars(simplejson)["_toggle_speedups"](accelerated)
    try:
        with pytest.raises(
            ValueError, match="^Configuration document is not a finite JSON object$"
        ):
            decode_config_document(
                '{"private-secret":1e9999999999999999999999999999999}'
            )
        # The C decoder accepts this depth; the Python implementation reaches
        # its recursion boundary. Both are valid dependency behaviors.
        if not accelerated:
            depth = sys.getrecursionlimit() + 100
            raw = '{"private-secret":' + "[" * depth + "0" + "]" * depth + "}"
            with pytest.raises(
                ValueError, match="^Configuration document is not a finite JSON object$"
            ):
                decode_config_document(raw)
            nested = []
            for _ in range(depth):
                nested = [nested]
            with pytest.raises(
                ValueError, match="^Configuration document cannot be encoded safely$"
            ):
                encode_config_document({"private-secret": nested})
    finally:
        vars(simplejson)["_toggle_speedups"](True)
