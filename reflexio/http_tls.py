"""TLS verification settings shared by Reflexio's blocking httpx callers.

Kept free of heavy imports: CLI command modules load at startup and the
lightweight ``reflexio-client`` distribution ships this module.
"""

from __future__ import annotations

import os
import ssl
from pathlib import Path


def tls_verify_from_env() -> ssl.SSLContext | bool:
    """Return the ``verify`` value for a blocking httpx call.

    httpx reads ``SSL_CERT_FILE`` / ``SSL_CERT_DIR`` but not the CA-bundle
    variables Reflexio's former ``requests`` transport honored. Keep honoring
    ``REQUESTS_CA_BUNDLE`` (then ``CURL_CA_BUNDLE``), in the same precedence,
    so a deployment behind a TLS-inspecting proxy keeps verifying against its
    own bundle.

    Returns:
        ssl.SSLContext | bool: A context that trusts only the configured bundle
            file or directory, or ``True`` for httpx's default verification.
    """
    bundle = os.environ.get("REQUESTS_CA_BUNDLE") or os.environ.get("CURL_CA_BUNDLE")
    if not bundle:
        return True
    if Path(bundle).is_dir():
        return ssl.create_default_context(capath=bundle)
    return ssl.create_default_context(cafile=bundle)
