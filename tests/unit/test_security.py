"""The deployment profile: identity, CSRF, and request rate.

The desktop build and a shared installation have opposite defaults, so each
control is checked in both profiles. Identity here is delegation to a proxy or a
shared secret, not an account system; these tests state what that does and does
not cover.
"""

import io
import uuid
import warnings

import pytest

from core import security

warnings.filterwarnings("ignore")


@pytest.fixture
def client():
    from app import app

    app.config["TESTING"] = True
    return app.test_client()


@pytest.fixture(autouse=True)
def clean_rate_state():
    security.reset_rate_state()
    yield
    security.reset_rate_state()


def _csv():
    return io.BytesIO(b"a,b,c\n1,2,3\n4,5,6\n")


def test_desktop_profile_asks_for_nothing(client, monkeypatch):
    monkeypatch.setenv("DIGITERRA_PROFILE", "desktop")
    assert not security.auth_required()
    assert not security.csrf_required()
    assert not security.rate_limit_enabled()
    r = client.post("/upload", data={"file": (_csv(), "x.csv")})
    assert r.status_code == 200


def test_server_profile_turns_on_all_three(monkeypatch):
    monkeypatch.setenv("DIGITERRA_PROFILE", "server")
    assert security.auth_required()
    assert security.csrf_required()
    assert security.rate_limit_enabled()


def test_an_unauthenticated_request_is_refused_on_the_server_profile(client, monkeypatch):
    monkeypatch.setenv("DIGITERRA_PROFILE", "server")
    r = client.get("/")
    assert r.status_code == 401
    assert "authenticated" in r.get_json()["error"]


def test_a_proxy_header_is_read_only_from_a_listed_address(client, monkeypatch):
    monkeypatch.setenv("DIGITERRA_PROFILE", "server")
    monkeypatch.setenv("DIGITERRA_REQUIRE_CSRF", "0")
    # Without a trusted-proxy list the header means nothing, whoever sends it.
    r = client.get("/", headers={"X-Forwarded-User": "rowan"})
    assert r.status_code == 401
    monkeypatch.setenv("DIGITERRA_TRUSTED_PROXIES", "127.0.0.1/32")
    r = client.get("/", headers={"X-Forwarded-User": "rowan"})
    assert r.status_code == 200
    # The header must carry a name; an empty one is not an identity.
    r = client.get("/", headers={"X-Forwarded-User": "  "})
    assert r.status_code == 401


def test_a_shared_secret_authenticates_where_there_is_no_proxy(client, monkeypatch):
    monkeypatch.setenv("DIGITERRA_PROFILE", "server")
    monkeypatch.setenv("DIGITERRA_REQUIRE_CSRF", "0")
    monkeypatch.setenv("DIGITERRA_SHARED_SECRET", "a-long-secret-value")
    assert client.get("/", headers={"X-DiGiTerra-Token": "wrong"}).status_code == 401
    assert client.get("/", headers={"X-DiGiTerra-Token": "a-long-secret-value"}).status_code == 200


def test_csrf_is_required_for_a_write_and_not_for_a_read(client, monkeypatch):
    monkeypatch.setenv("DIGITERRA_PROFILE", "server")
    monkeypatch.setenv("DIGITERRA_TRUSTED_PROXIES", "127.0.0.1/32")
    headers = {"X-Forwarded-User": "rowan"}
    r = client.get("/", headers=headers)
    assert r.status_code == 200
    token = client.get_cookie(security.CSRF_COOKIE_NAME).value

    # The cookie alone is not enough; the page must echo it in the header.
    r = client.post("/upload", data={"file": (_csv(), "x.csv")}, headers=headers)
    assert r.status_code == 403
    assert "CSRF" in r.get_json()["error"]

    r = client.post(
        "/upload",
        data={"file": (_csv(), "x.csv")},
        headers={**headers, security.CSRF_HEADER_NAME: token},
    )
    assert r.status_code == 200

    # A token from somewhere else does not match this browser's cookie.
    r = client.post(
        "/upload",
        data={"file": (_csv(), "x.csv")},
        headers={**headers, security.CSRF_HEADER_NAME: security.issue_csrf_token()},
    )
    assert r.status_code == 403


def test_the_rate_limit_refuses_a_burst_and_recovers(monkeypatch):
    monkeypatch.setenv("DIGITERRA_PROFILE", "server")
    monkeypatch.setenv("DIGITERRA_RATE_PER_MINUTE", "60")
    monkeypatch.setenv("DIGITERRA_RATE_BURST", "5")
    key = f"test-{uuid.uuid4().hex}"
    assert all(security.rate_allow(key) for _ in range(5))
    assert not security.rate_allow(key)

    # Tokens return at the configured rate; one second buys one request at 60/min.
    import time as _time

    real = _time.monotonic()
    monkeypatch.setattr(security.time, "monotonic", lambda: real + 1.2)
    assert security.rate_allow(key)


def test_a_run_costs_more_of_the_budget_than_a_page_view(client, monkeypatch):
    monkeypatch.setenv("DIGITERRA_PROFILE", "server")
    monkeypatch.setenv("DIGITERRA_TRUSTED_PROXIES", "127.0.0.1/32")
    monkeypatch.setenv("DIGITERRA_REQUIRE_CSRF", "0")
    monkeypatch.setenv("DIGITERRA_RATE_PER_MINUTE", "600")
    monkeypatch.setenv("DIGITERRA_RATE_BURST", "12")
    headers = {"X-Forwarded-User": "rowan"}
    # Two uploads take 20 of a 12-token bucket, so the second is refused while the
    # index page, at one token, still answers before them.
    assert client.get("/", headers=headers).status_code == 200
    first = client.post("/upload", data={"file": (_csv(), "x.csv")}, headers=headers)
    second = client.post("/upload", data={"file": (_csv(), "x.csv")}, headers=headers)
    assert first.status_code == 200
    assert second.status_code == 429


def test_static_files_and_health_are_not_gated(client, monkeypatch):
    monkeypatch.setenv("DIGITERRA_PROFILE", "server")
    r = client.get("/static/style.css")
    assert r.status_code == 200
