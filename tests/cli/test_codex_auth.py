"""Unit tests for ``reflexio.cli.codex_auth`` — PKCE, JWT decoding, token storage.

We don't exercise the full browser/callback flow here (that's an integration
concern). The tests below lock down the building blocks:

- PKCE verifier/challenge generation produces RFC-7636-compatible output.
- JWT payload extraction handles both well-formed and pathological inputs.
- ``CodexTokens`` round-trips through ``save_tokens`` / ``load_tokens_raw``.
- ``is_expired`` honours the lead-time threshold.
- ``_tokens_from_response`` populates metadata from JWT claims correctly.
"""

from __future__ import annotations

import base64
import hashlib
import json
import time
from pathlib import Path

import pytest

from reflexio.cli import codex_auth


def _b64url(data: bytes) -> str:
    """Base64url-encode without padding (test helper, mirrors the module's)."""
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _make_jwt(claims: dict) -> str:
    """Build an RS256-shaped JWT from a payload dict.

    The signature is fake (the module deliberately does not verify), so we
    can hand-craft tokens for the storage / refresh logic without involving
    cryptography. Header is the constant Codex uses.
    """
    header = _b64url(json.dumps({"alg": "RS256", "typ": "JWT"}).encode())
    payload = _b64url(json.dumps(claims).encode())
    sig = _b64url(b"fake-signature-not-verified")
    return f"{header}.{payload}.{sig}"


class TestPkce:
    def test_pair_shape(self) -> None:
        verifier, challenge = codex_auth._make_pkce_pair()
        # Both base64url, no padding.
        assert "=" not in verifier
        assert "=" not in challenge
        # 32-byte random source -> 43-char base64url.
        assert len(verifier) == 43
        # Challenge is base64url(SHA-256(verifier ASCII)).
        expected = _b64url(hashlib.sha256(verifier.encode("ascii")).digest())
        assert challenge == expected

    def test_pairs_are_unique(self) -> None:
        # Different invocations should not collide (32-byte entropy).
        pairs = {codex_auth._make_pkce_pair()[0] for _ in range(50)}
        assert len(pairs) == 50


class TestJwtDecoding:
    def test_decode_extracts_payload(self) -> None:
        claims = {"foo": "bar", "exp": 1234567890}
        jwt = _make_jwt(claims)
        out = codex_auth._decode_jwt_payload(jwt)
        assert out == claims

    def test_decode_handles_unpadded_b64(self) -> None:
        # Codex JWTs typically have no padding on the payload segment;
        # the decoder must restore it on the fly.
        claims = {"x": 1}
        jwt = _make_jwt(claims)
        # Strip any incidental trailing '=' just in case.
        assert "=" not in jwt
        assert codex_auth._decode_jwt_payload(jwt) == claims

    def test_decode_rejects_malformed(self) -> None:
        with pytest.raises(ValueError, match="not a JWT"):
            codex_auth._decode_jwt_payload("not.a.jwt.at.all")
        with pytest.raises(ValueError, match="not a JWT"):
            codex_auth._decode_jwt_payload("only-one-part")


class TestTokensFromResponse:
    def test_extracts_account_id_and_plan_type(self) -> None:
        # Mirror the JWT shape OpenAI issues: chatgpt_plan_type lives under
        # the namespaced ``https://api.openai.com/auth`` claim, email under
        # ``https://api.openai.com/profile``.
        claims = {
            "exp": int(time.time()) + 3600,
            "https://api.openai.com/auth": {
                "chatgpt_account_id": "acct-abc-123",
                "chatgpt_plan_type": "max-x20",
            },
            "https://api.openai.com/profile": {
                "email": "user@example.com",
            },
        }
        access = _make_jwt(claims)
        payload = {
            "access_token": access,
            "refresh_token": "rt_abc",
            "expires_in": 3600,
        }
        tokens = codex_auth._tokens_from_response(payload)
        assert tokens.access_token == access
        assert tokens.refresh_token == "rt_abc"
        assert tokens.account_id == "acct-abc-123"
        assert tokens.plan_type == "max-x20"
        assert tokens.email == "user@example.com"
        assert tokens.expires_at == claims["exp"]

    def test_falls_back_to_expires_in_when_jwt_lacks_exp(self) -> None:
        claims = {"https://api.openai.com/auth": {}}  # no exp
        access = _make_jwt(claims)
        before = int(time.time())
        tokens = codex_auth._tokens_from_response(
            {"access_token": access, "refresh_token": "rt", "expires_in": 600}
        )
        # Allow a small wall-time window (<2s) for the test runner.
        assert before + 600 <= tokens.expires_at <= before + 602

    def test_rejects_missing_required_fields(self) -> None:
        with pytest.raises(ValueError, match="missing access_token"):
            codex_auth._tokens_from_response({"refresh_token": "rt"})
        with pytest.raises(ValueError, match="missing access_token"):
            codex_auth._tokens_from_response({"access_token": _make_jwt({})})


class TestTokenStorage:
    def test_save_and_load_round_trip(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # Redirect storage to a temp dir so the test never touches the
        # developer's real ~/.reflexio/auth/.
        monkeypatch.setattr(codex_auth, "REFLEXIO_AUTH_DIR", tmp_path / "auth")
        monkeypatch.setattr(
            codex_auth,
            "REFLEXIO_CODEX_TOKENS_PATH",
            tmp_path / "auth" / "openai-codex.json",
        )

        tokens = codex_auth.CodexTokens(
            access_token="a-jwt",
            refresh_token="rt-1",
            account_id="acct-x",
            expires_at=1234,
            plan_type="max-x20",
            email="x@y.com",
        )
        path = codex_auth.save_tokens(tokens)
        assert path.exists()
        # File mode should be 0600 on POSIX (best-effort on platforms that
        # don't support it; we just check the round-trip below).
        loaded = codex_auth.load_tokens_raw()
        assert loaded == tokens

    def test_load_returns_none_when_missing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            codex_auth,
            "REFLEXIO_CODEX_TOKENS_PATH",
            tmp_path / "openai-codex.json",
        )
        assert codex_auth.load_tokens_raw() is None

    def test_load_returns_none_for_malformed_json(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        path = tmp_path / "openai-codex.json"
        path.write_text("{not valid json")
        monkeypatch.setattr(codex_auth, "REFLEXIO_CODEX_TOKENS_PATH", path)
        assert codex_auth.load_tokens_raw() is None


class TestExpiryCheck:
    def test_is_expired_lead_time(self) -> None:
        now = int(time.time())
        # 30 seconds in the future, default lead time 60 -> already "expired".
        t1 = codex_auth.CodexTokens(
            access_token="x",
            refresh_token="y",
            account_id="",
            expires_at=now + 30,
            plan_type="",
            email="",
        )
        assert t1.is_expired() is True

        # 600 seconds in the future, well outside any lead time.
        t2 = codex_auth.CodexTokens(
            access_token="x",
            refresh_token="y",
            account_id="",
            expires_at=now + 600,
            plan_type="",
            email="",
        )
        assert t2.is_expired() is False
        # Custom lead time can flip the result.
        assert t2.is_expired(lead_seconds=700) is True


class TestAuthorizeUrl:
    def test_url_contains_required_oauth_params(self) -> None:
        verifier, _ = codex_auth._make_pkce_pair()
        state = "csrf-state-abc"
        url, challenge = codex_auth.build_authorize_url(verifier, state)
        # Sanity-check the host + a handful of required params.
        assert url.startswith(codex_auth.CODEX_AUTHORIZE_URL + "?")
        for required in (
            f"client_id={codex_auth.CODEX_CLIENT_ID}",
            "response_type=code",
            "code_challenge_method=S256",
            f"state={state}",
            "scope=openid+profile+email+offline_access",
        ):
            assert required in url
        assert challenge in url
