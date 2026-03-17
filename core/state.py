"""Session state and request-scoped helpers for DiGiTerra. Used by app and blueprints."""
from flask import g, request

SESSION_STORE = {}
SESSION_COOKIE_NAME = "digiterra_session"


def get_session_id() -> str:
    """Return the current session ID (from cookie or request context)."""
    sid = getattr(g, "session_id", None)
    if sid:
        return sid
    return request.cookies.get(SESSION_COOKIE_NAME) or "default"


def get_session_storage(session_id: str | None = None) -> dict:
    """Get or create session-scoped storage dict."""
    sid = session_id or get_session_id()
    return SESSION_STORE.setdefault(sid, {})
