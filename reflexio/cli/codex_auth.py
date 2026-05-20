"""Reflexio-native OAuth tokens for OpenAI Codex / ChatGPT subscription.

This module owns reflexio's own OAuth tokens against ``auth.openai.com``,
independent of OpenClaw or any other CLI. Tokens are stored at
``~/.reflexio/auth/openai-codex.json`` and the refresh-token flow is built
into the loader so callers always see a fresh access token.

Why a separate module: the token store is consumed by both the CLI
(``reflexio setup openai-codex``) and the runtime proxy (``codex_proxy.py``
in the enterprise tree). Putting it in one place keeps the file shape and
refresh policy in sync.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import json
import logging
import secrets
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse

from .paths import reflexio_home

logger = logging.getLogger(__name__)

# OAuth client + endpoints used by the Codex CLI. Values verified by
# inspecting the JWT payload of an existing OpenClaw-issued token
# (`client_id`, `iss` claims) and the codex-rs source.
CODEX_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
CODEX_AUTH_ISSUER = "https://auth.openai.com"
CODEX_AUTHORIZE_URL = f"{CODEX_AUTH_ISSUER}/oauth/authorize"
CODEX_TOKEN_URL = f"{CODEX_AUTH_ISSUER}/oauth/token"

# Codex CLI binds its callback server to this port; OpenAI's OAuth client
# config has ``http://localhost:1455/auth/callback`` registered as a valid
# redirect URI, so we reuse it.
CODEX_CALLBACK_HOST = "localhost"
CODEX_CALLBACK_PORT = 1455
CODEX_CALLBACK_PATH = "/auth/callback"
CODEX_REDIRECT_URI = (
    f"http://{CODEX_CALLBACK_HOST}:{CODEX_CALLBACK_PORT}{CODEX_CALLBACK_PATH}"
)

CODEX_SCOPES = "openid profile email offline_access"

# Refresh slightly before the access token actually expires so a slow
# downstream call doesn't cross the boundary mid-flight.
_REFRESH_LEAD_SECONDS = 60

REFLEXIO_AUTH_DIR = reflexio_home() / "auth"
REFLEXIO_CODEX_TOKENS_PATH = REFLEXIO_AUTH_DIR / "openai-codex.json"


@dataclass
class CodexTokens:
    """Persisted Codex OAuth tokens.

    Attributes:
        access_token (str): Bearer token used for ``api.openai.com`` and
            ``chatgpt.com/backend-api/codex`` calls.
        refresh_token (str): Long-lived token used to mint a new access
            token at ``/oauth/token``.
        account_id (str): ``ChatGPT-Account-ID`` header value (from the
            JWT's ``chatgpt_account_id`` claim).
        expires_at (int): Unix epoch seconds when ``access_token`` expires.
        plan_type (str): Cached ``chatgpt_plan_type`` from the JWT (e.g.
            ``"plus"``, ``"max-x20"``) for human-facing diagnostics.
        email (str): User email from the JWT, surfaced in CLI status.
    """

    access_token: str
    refresh_token: str
    account_id: str
    expires_at: int
    plan_type: str
    email: str

    def is_expired(self, lead_seconds: int = _REFRESH_LEAD_SECONDS) -> bool:
        """Return True if the access token will expire within ``lead_seconds``.

        Args:
            lead_seconds (int): Treat tokens with less than this much time
                remaining as already expired.

        Returns:
            bool: ``True`` if a refresh is needed.
        """
        return self.expires_at - lead_seconds <= int(time.time())


def _b64url(data: bytes) -> str:
    """Base64url-encode without padding (PKCE-style)."""
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _make_pkce_pair() -> tuple[str, str]:
    """Generate a (code_verifier, code_challenge) PKCE pair.

    Uses a 32-byte random verifier; SHA-256 + base64url for the challenge.

    Returns:
        tuple[str, str]: ``(verifier, challenge)``.
    """
    verifier = _b64url(secrets.token_bytes(32))
    challenge = _b64url(hashlib.sha256(verifier.encode("ascii")).digest())
    return verifier, challenge


def _decode_jwt_payload(jwt: str) -> dict[str, Any]:
    """Decode an unsigned JWT payload (no signature verification).

    Codex JWTs are issued by ``auth.openai.com`` with RS256; we don't have
    the public key locally and don't need to: storing tokens we receive over
    HTTPS from the issuer is sufficient. The payload is read for metadata
    (account_id, plan_type, email, exp).

    Args:
        jwt (str): A JWT in standard ``header.payload.signature`` form.

    Returns:
        dict[str, Any]: The JSON-parsed payload.

    Raises:
        ValueError: If the JWT is malformed.
    """
    parts = jwt.split(".")
    if len(parts) != 3:
        raise ValueError("not a JWT (expected three dot-separated parts)")
    payload_b64 = parts[1] + "=" * (-len(parts[1]) % 4)  # restore padding
    return json.loads(base64.urlsafe_b64decode(payload_b64))


def _tokens_from_response(payload: dict[str, Any]) -> CodexTokens:
    """Build a ``CodexTokens`` from an ``/oauth/token`` JSON response.

    Reads the access JWT to derive ``account_id``, ``plan_type``, ``email``,
    and ``expires_at``. Falls back to ``expires_in`` from the response if the
    JWT lacks an ``exp`` claim.

    Args:
        payload (dict): Decoded JSON body from an OAuth token endpoint.

    Returns:
        CodexTokens: Populated record.

    Raises:
        ValueError: Required fields missing.
    """
    access = payload.get("access_token")
    refresh = payload.get("refresh_token")
    if not access or not refresh:
        raise ValueError(
            f"OAuth response missing access_token / refresh_token: {payload}"
        )
    claims = _decode_jwt_payload(access)
    auth_claims = claims.get("https://api.openai.com/auth", {}) or {}
    profile_claims = claims.get("https://api.openai.com/profile", {}) or {}
    account_id = auth_claims.get("chatgpt_account_id", "") or ""
    plan_type = auth_claims.get("chatgpt_plan_type", "unknown") or "unknown"
    email = profile_claims.get("email", "") or ""
    if (exp := claims.get("exp")) is not None:
        expires_at = int(exp)
    else:
        expires_at = int(time.time()) + int(payload.get("expires_in", 0))
    return CodexTokens(
        access_token=access,
        refresh_token=refresh,
        account_id=account_id,
        expires_at=expires_at,
        plan_type=str(plan_type),
        email=str(email),
    )


def save_tokens(tokens: CodexTokens) -> Path:
    """Persist tokens to ``~/.reflexio/auth/openai-codex.json``.

    Creates the parent directory with restrictive permissions on first write.
    The token file itself is written with mode 0600 — bearer tokens shouldn't
    be world-readable.

    Args:
        tokens (CodexTokens): Tokens to persist.

    Returns:
        Path: Where the file was written.
    """
    REFLEXIO_AUTH_DIR.mkdir(parents=True, exist_ok=True)
    # Filesystems without POSIX permissions (e.g., FAT) won't honour chmod;
    # tolerate the failure rather than aborting the login.
    with contextlib.suppress(OSError):
        REFLEXIO_AUTH_DIR.chmod(0o700)
    payload = {
        "version": 1,
        "access_token": tokens.access_token,
        "refresh_token": tokens.refresh_token,
        "account_id": tokens.account_id,
        "expires_at": tokens.expires_at,
        "plan_type": tokens.plan_type,
        "email": tokens.email,
    }
    REFLEXIO_CODEX_TOKENS_PATH.write_text(json.dumps(payload, indent=2))
    with contextlib.suppress(OSError):
        REFLEXIO_CODEX_TOKENS_PATH.chmod(0o600)
    return REFLEXIO_CODEX_TOKENS_PATH


def load_tokens_raw() -> CodexTokens | None:
    """Load tokens from disk without refreshing.

    Returns:
        CodexTokens | None: Persisted tokens, or ``None`` if the file is
            missing or malformed.
    """
    if not REFLEXIO_CODEX_TOKENS_PATH.exists():
        return None
    try:
        data = json.loads(REFLEXIO_CODEX_TOKENS_PATH.read_text())
        return CodexTokens(
            access_token=data["access_token"],
            refresh_token=data["refresh_token"],
            account_id=data.get("account_id", ""),
            expires_at=int(data.get("expires_at", 0)),
            plan_type=data.get("plan_type", "unknown"),
            email=data.get("email", ""),
        )
    except (KeyError, json.JSONDecodeError, ValueError) as e:
        logger.warning("Bad reflexio codex tokens file: %s", e)
        return None


def refresh_tokens(tokens: CodexTokens) -> CodexTokens:
    """Exchange the refresh_token for a new (access, refresh) pair.

    POSTs to ``auth.openai.com/oauth/token`` with ``grant_type=refresh_token``.
    The new tokens are persisted to disk before returning.

    Args:
        tokens (CodexTokens): The current tokens; only ``refresh_token`` is read.

    Returns:
        CodexTokens: A fresh, persisted token record.

    Raises:
        urllib.error.HTTPError: If the token endpoint rejects the refresh
            (e.g., refresh_token revoked — caller should prompt re-login).
    """
    body = urlencode(
        {
            "grant_type": "refresh_token",
            "refresh_token": tokens.refresh_token,
            "client_id": CODEX_CLIENT_ID,
            "scope": CODEX_SCOPES,
        }
    ).encode("utf-8")
    req = urllib.request.Request(  # noqa: S310 - fixed https URL
        CODEX_TOKEN_URL,
        data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 - fixed https URL
        payload = json.loads(resp.read())
    new_tokens = _tokens_from_response(payload)
    save_tokens(new_tokens)
    logger.info(
        "Refreshed OpenAI Codex tokens; new access expires at %d (plan=%s)",
        new_tokens.expires_at,
        new_tokens.plan_type,
    )
    return new_tokens


def get_fresh_tokens() -> CodexTokens | None:
    """Return tokens, refreshing on disk if the access token has expired.

    Returns:
        CodexTokens | None: Fresh tokens, or ``None`` if no tokens are saved.
            Caller should run ``reflexio setup openai-codex`` if ``None``.
    """
    tokens = load_tokens_raw()
    if tokens is None:
        return None
    if tokens.is_expired():
        try:
            return refresh_tokens(tokens)
        except urllib.error.HTTPError as e:
            logger.warning(
                "Refresh failed (HTTP %d); re-login required via "
                "'reflexio setup openai-codex'",
                e.code,
            )
            return None
    return tokens


# ---------------------------------------------------------------------------
# Authorization-code login flow (browser + PKCE + local callback)
# ---------------------------------------------------------------------------


class _CallbackHandler(BaseHTTPRequestHandler):
    """One-shot HTTP handler that captures the OAuth callback.

    The handler stashes the parsed query parameters on the server instance
    (which a stricter typer would model as a custom HTTPServer subclass);
    the orchestrating function reads them back after ``handle_request``.

    Browsers expect a tidy success page; we serve a small HTML body so the
    user knows the CLI took control.
    """

    # Silence default access logs; this is a 1-shot interactive flow.
    def log_message(  # noqa: ANN401, ARG002 — signature dictated by stdlib
        self,
        format: str,  # noqa: A002, ARG002
        *args: Any,  # noqa: ARG002
    ) -> None:
        """No-op — suppress the default access log noise."""
        return

    def do_GET(self) -> None:  # noqa: N802 - dictated by stdlib
        """Capture the callback query and write a success page."""
        parsed = urlparse(self.path)
        if parsed.path != CODEX_CALLBACK_PATH:
            self.send_response(404)
            self.end_headers()
            return
        query = parse_qs(parsed.query)
        # Store on the server instance for the caller to read.
        self.server._captured = {  # type: ignore[attr-defined]
            "code": (query.get("code") or [""])[0],
            "state": (query.get("state") or [""])[0],
            "error": (query.get("error") or [""])[0],
            "error_description": (query.get("error_description") or [""])[0],
        }
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(
            b"<html><body style='font-family:system-ui;max-width:520px;margin:48px auto'>"
            b"<h2>Reflexio is now signed in.</h2>"
            b"<p>You can close this tab and return to the terminal.</p>"
            b"</body></html>"
        )


def _capture_oauth_callback(state: str, timeout_s: int) -> dict[str, str]:
    """Run a one-shot HTTP server and return the OAuth callback query.

    Args:
        state (str): The CSRF ``state`` value sent on the authorize call;
            verified to match here.
        timeout_s (int): Hard ceiling on how long to wait for the user to
            complete the browser flow.

    Returns:
        dict[str, str]: The captured query parameters
            (``code``, ``state``, ``error``, ``error_description``).

    Raises:
        TimeoutError: If the callback isn't received in time.
        ValueError: If the callback's state doesn't match the request's.
    """
    server = HTTPServer((CODEX_CALLBACK_HOST, CODEX_CALLBACK_PORT), _CallbackHandler)
    server._captured = None  # type: ignore[attr-defined]
    server.timeout = timeout_s
    server.handle_request()
    captured: dict[str, str] | None = getattr(server, "_captured", None)
    if captured is None:
        raise TimeoutError(
            f"OAuth callback not received within {timeout_s}s — open the URL "
            "yourself and complete the sign-in?"
        )
    if captured.get("state") != state:
        raise ValueError(
            "OAuth state mismatch — refusing to continue (possible CSRF)."
        )
    if err := captured.get("error"):
        raise ValueError(
            f"OAuth provider returned error '{err}': {captured.get('error_description', '')}"
        )
    return captured


def build_authorize_url(verifier: str, state: str) -> tuple[str, str]:
    """Build the authorization URL for the browser step of the OAuth flow.

    Args:
        verifier (str): PKCE code verifier (the random secret stored locally).
        state (str): CSRF state value to round-trip through the redirect.

    Returns:
        tuple[str, str]: ``(authorize_url, code_challenge)``. The challenge
            is returned for callers that want to display it; the URL is what
            actually goes in the browser.
    """
    challenge = _b64url(hashlib.sha256(verifier.encode("ascii")).digest())
    qs = urlencode(
        {
            "client_id": CODEX_CLIENT_ID,
            "response_type": "code",
            "redirect_uri": CODEX_REDIRECT_URI,
            "scope": CODEX_SCOPES,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "state": state,
        }
    )
    return f"{CODEX_AUTHORIZE_URL}?{qs}", challenge


def exchange_authorization_code(code: str, verifier: str) -> CodexTokens:
    """Exchange an OAuth authorization code for tokens.

    Args:
        code (str): The ``code`` query param the redirect delivered.
        verifier (str): The PKCE code verifier (must be the one used when
            building the authorize URL).

    Returns:
        CodexTokens: The persisted token record.

    Raises:
        urllib.error.HTTPError: If the token endpoint rejects the request.
    """
    body = urlencode(
        {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": CODEX_REDIRECT_URI,
            "client_id": CODEX_CLIENT_ID,
            "code_verifier": verifier,
        }
    ).encode("utf-8")
    req = urllib.request.Request(  # noqa: S310 - fixed https URL
        CODEX_TOKEN_URL,
        data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 - fixed https URL
        payload = json.loads(resp.read())
    return _tokens_from_response(payload)


def login_interactive(
    *,
    open_browser: bool = True,
    timeout_s: int = 300,
) -> CodexTokens:
    """Run the full PKCE OAuth flow against ``auth.openai.com``.

    Steps:
      1. Generate a fresh PKCE pair + CSRF state.
      2. Build the authorize URL and either open the user's browser or
         print the URL for them to open manually.
      3. Bind a one-shot HTTP server on ``localhost:1455`` to catch the
         callback.
      4. Exchange the returned auth code for tokens.
      5. Persist tokens to disk.

    Args:
        open_browser (bool): When True (default), call ``webbrowser.open``
            on the authorize URL. When False, just print it.
        timeout_s (int): Maximum wall time to wait for the callback before
            failing.

    Returns:
        CodexTokens: The persisted token record.
    """
    verifier, _challenge = _make_pkce_pair()
    state = _b64url(secrets.token_bytes(16))
    authorize_url, _ = build_authorize_url(verifier, state)

    if open_browser:
        # Lazy import — webbrowser pulls in tkinter on some platforms.
        import webbrowser

        opened = webbrowser.open(authorize_url, new=1)
        if not opened:
            print("Could not open browser automatically.")
    print()
    print("Open this URL to sign in to ChatGPT:")
    print(f"  {authorize_url}")
    print()
    print(f"Listening for callback on {CODEX_REDIRECT_URI} ...")

    captured = _capture_oauth_callback(state=state, timeout_s=timeout_s)
    code = captured.get("code") or ""
    if not code:
        raise ValueError("OAuth callback returned no authorization code.")

    tokens = exchange_authorization_code(code, verifier)
    save_tokens(tokens)
    return tokens
