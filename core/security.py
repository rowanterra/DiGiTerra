"""Deployment profile, identity, CSRF, and request rate.

The same code serves two deployments with opposite assumptions. On a desktop the
only user is the person at the keyboard, and a login screen would be an obstacle
with nothing behind it. On a shared or EDX installation every request arrives from
somewhere unproven. Rather than leave that difference to a handful of separate
variables that an operator has to get right one at a time, DIGITERRA_PROFILE picks
a set:

    desktop (default)  no identity, no CSRF token, no rate limit.
    server             identity required, CSRF required, rate limits on.

Any single control can still be set on its own, which is what a pilot needs while
its proxy is being arranged.

Identity here is delegation, not a login system. EDX and comparable environments
terminate single sign-on at a reverse proxy and pass the result in a header, so
this module trusts a named header only when the request arrives from a listed
proxy address, and offers a shared secret for a deployment that has no such proxy
yet. Neither is a substitute for an account system with its own sessions, and the
security review says so.
"""

from __future__ import annotations

import hmac
import ipaddress
import logging
import os
import secrets
import threading
import time

from flask import g, jsonify, request

logger = logging.getLogger(__name__)

CSRF_COOKIE_NAME = "digiterra_csrf"
CSRF_HEADER_NAME = "X-DiGiTerra-CSRF"
IDENTITY_ATTR = "identity"

# Methods that change something. A GET may be expensive, which the rate limit
# covers, but it is not what a cross-site form can abuse.
UNSAFE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}

# Paths that answer before a session can hold a token, and the static files.
CSRF_EXEMPT_PATHS = {"/health"}


def _flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def profile() -> str:
    value = os.environ.get("DIGITERRA_PROFILE", "desktop").strip().lower()
    return value if value in {"desktop", "server"} else "desktop"


def is_server_profile() -> bool:
    return profile() == "server"


def auth_required() -> bool:
    return _flag("DIGITERRA_REQUIRE_AUTH", is_server_profile())


def csrf_required() -> bool:
    return _flag("DIGITERRA_REQUIRE_CSRF", is_server_profile())


def rate_limit_enabled() -> bool:
    return _flag("DIGITERRA_RATE_LIMIT", is_server_profile())


def identity_header() -> str:
    return os.environ.get("DIGITERRA_AUTH_HEADER", "X-Forwarded-User").strip()


def _trusted_proxies() -> list:
    raw = os.environ.get("DIGITERRA_TRUSTED_PROXIES", "").strip()
    nets = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            nets.append(ipaddress.ip_network(part, strict=False))
        except ValueError:
            logger.warning("Ignoring unreadable entry in DIGITERRA_TRUSTED_PROXIES: %s", part)
    return nets


def _from_trusted_proxy() -> bool:
    """True when the immediate peer is one of the listed proxy addresses.

    A header can be set by anyone who can reach the port, so it is read only when
    the connection itself comes from an address the operator named.
    """
    nets = _trusted_proxies()
    if not nets:
        return False
    peer = request.remote_addr or ""
    try:
        addr = ipaddress.ip_address(peer)
    except ValueError:
        return False
    return any(addr in net for net in nets)


def resolve_identity() -> str | None:
    """The user this request belongs to, or None when it cannot be established."""
    if not auth_required():
        return "local"
    header = identity_header()
    if header and _from_trusted_proxy():
        value = (request.headers.get(header) or "").strip()
        if value:
            return value[:200]
    secret = os.environ.get("DIGITERRA_SHARED_SECRET", "").strip()
    if secret:
        offered = (request.headers.get("X-DiGiTerra-Token") or "").strip()
        if offered and hmac.compare_digest(offered, secret):
            return "shared-secret"
    return None


# ---------------------------------------------------------------------------
# CSRF. The token is a cookie the browser returns and a header the page sends;
# a cross-site form can cause the cookie to be sent but cannot read it to set the
# header. The cookie is therefore readable by script on purpose.
def issue_csrf_token() -> str:
    return secrets.token_urlsafe(32)


def csrf_token_valid() -> bool:
    cookie = request.cookies.get(CSRF_COOKIE_NAME, "")
    sent = request.headers.get(CSRF_HEADER_NAME, "") or (request.form.get("csrf_token") or "")
    return bool(cookie) and bool(sent) and hmac.compare_digest(cookie, sent)


# ---------------------------------------------------------------------------
# Rate limiting. A token bucket per client, in this process, which is the same
# place the session state lives; a deployment behind more than one process needs a
# shared store, and the security review says so.
_lock = threading.Lock()
_buckets: dict[str, tuple[float, float]] = {}


def _limits() -> tuple[float, float]:
    """Requests per minute and burst size."""
    per_minute = float(os.environ.get("DIGITERRA_RATE_PER_MINUTE", "120") or 120)
    burst = float(os.environ.get("DIGITERRA_RATE_BURST", "40") or 40)
    return per_minute, burst


def _client_key() -> str:
    ident = getattr(g, IDENTITY_ATTR, None)
    if ident and ident not in {"local", "shared-secret"}:
        return f"user:{ident}"
    return f"addr:{request.remote_addr or 'unknown'}"


def rate_allow(key: str | None = None, cost: float = 1.0) -> bool:
    """Take `cost` tokens from this client's bucket, or refuse."""
    if not rate_limit_enabled():
        return True
    per_minute, burst = _limits()
    if per_minute <= 0:
        return True
    key = key or _client_key()
    now = time.monotonic()
    with _lock:
        tokens, last = _buckets.get(key, (burst, now))
        tokens = min(burst, tokens + (now - last) * (per_minute / 60.0))
        if tokens < cost:
            _buckets[key] = (tokens, now)
            return False
        _buckets[key] = (tokens - cost, now)
        # The map is bounded so a stream of addresses cannot grow it without end.
        if len(_buckets) > 4096:
            cutoff = now - 3600
            for k, (_, seen) in list(_buckets.items()):
                if seen < cutoff:
                    del _buckets[k]
        return True


def reset_rate_state() -> None:
    with _lock:
        _buckets.clear()


# ---------------------------------------------------------------------------
def install(app) -> None:
    """Attach the profile's controls to the application."""

    @app.before_request
    def _security_gate():  # noqa: ANN202 - Flask hook
        path = request.path or "/"
        if path.startswith("/static/") or path in CSRF_EXEMPT_PATHS:
            return None

        identity = resolve_identity()
        setattr(g, IDENTITY_ATTR, identity)
        if identity is None:
            return jsonify({"error": "This deployment requires an authenticated request."}), 401

        # A run costs far more than a page view, so it takes more of the bucket.
        cost = 10.0 if path.rstrip("/") in {"/process", "/predict", "/upload"} else 1.0
        if not rate_allow(cost=cost):
            return jsonify({"error": "Too many requests. Wait a moment and try again."}), 429

        if csrf_required() and request.method in UNSAFE_METHODS and not csrf_token_valid():
            return jsonify({"error": "Missing or stale CSRF token. Reload the page and try again."}), 403
        return None

    @app.after_request
    def _issue_csrf_cookie(response):  # noqa: ANN202 - Flask hook
        if not request.cookies.get(CSRF_COOKIE_NAME):
            secure = request.is_secure or _flag("DIGITERRA_COOKIE_SECURE", False)
            # Readable by script by design: the page must echo it in a header.
            response.set_cookie(CSRF_COOKIE_NAME, issue_csrf_token(), httponly=False, samesite="Lax", secure=secure)
        return response

    logger.info(
        "Security profile %s: auth=%s csrf=%s rate_limit=%s",
        profile(),
        auth_required(),
        csrf_required(),
        rate_limit_enabled(),
    )
