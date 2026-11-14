"""Both persisted-digest encoders preserve Unicode scalar validation."""

from collections.abc import Callable, Iterator

import pytest

from reflexio.server.services.offline_tuner.evidence import manifests
from reflexio.server.services.playbook import publication


@pytest.mark.parametrize(
    "encode", [publication.canonical_json_bytes, manifests.canonical_json_bytes]
)
@pytest.mark.parametrize("as_key", [False, True])
def test_every_surrogate_is_refused(
    encode: Callable[[object], bytes], as_key: bool
) -> None:
    for codepoint in range(0xD800, 0xE000):
        text = f"prefix{chr(codepoint)}suffix"
        payload = {text: "value"} if as_key else {"key": text}
        # The manifest encoder historically rejects invalid keys during UTF-16
        # sorting, before its explicit string validator. Preserve that boundary.
        if as_key and encode is manifests.canonical_json_bytes:
            with pytest.raises(UnicodeEncodeError, match="surrogates not allowed"):
                encode(payload)
        else:
            with pytest.raises(
                ValueError,
                match="^RFC 8785 strings cannot contain surrogate code points$",
            ):
                encode(payload)


@pytest.mark.parametrize(
    "encode", [publication.canonical_json_bytes, manifests.canonical_json_bytes]
)
def test_valid_scalar_boundaries_and_utf16_key_order_are_unchanged(
    encode: Callable[[object], bytes],
) -> None:
    assert encode({"value": "\x00\ud7ff\ue000\uffff\U00010000\U0010ffff"}) == (
        '{"value":"\\u0000\ud7ff\ue000\uffff\U00010000\U0010ffff"}'.encode()
    )
    assert encode({"\ue000": "bmp", "\U00010000": "astral"}) == (
        '{"\U00010000":"astral","\ue000":"bmp"}'.encode()
    )


class _HiddenSurrogate(str):
    def __iter__(self) -> Iterator[str]:
        return iter("a")


class _InventedSurrogate(str):
    def __iter__(self) -> Iterator[str]:
        return iter("\ud800")


class _FalseASCII(str):
    def isascii(self) -> bool:
        return True


@pytest.mark.parametrize(
    "reject", [publication._reject_surrogates, manifests._reject_surrogates]
)
def test_subclass_iteration_and_ascii_override_keep_their_original_semantics(
    reject: Callable[[str], None],
) -> None:
    reject(_HiddenSurrogate("\ud800"))
    for value in (_InventedSurrogate("valid"), _FalseASCII("\ud800")):
        with pytest.raises(ValueError, match="surrogate"):
            reject(value)
