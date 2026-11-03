# Security

## Reporting

Report a suspected vulnerability by email to the maintainer listed in `CITATION.cff`.
Include the version or commit, the configuration, and the smallest input that shows
the problem. Please do not open a public issue for an unreported vulnerability.

There is no service to attack: DiGiTerra runs on a desktop or on a host an operator
controls. A report about the desktop application, about a served installation, or
about a dependency is all in scope.

## Supported versions

The `main` branch is the supported version. Fixes are made there and released from
there. Earlier tags receive no updates.

## What the application defends against, and what it does not

`docs/SECURITY_REVIEW.md` is the full statement, written against the code rather than
against intent, and `tests/unit/test_docs.py` fails when a `DIGITERRA_*` limit exists
in the code and appears in neither that file nor `docs/documentation.md`.

The short form. `DIGITERRA_PROFILE` selects a set of controls rather than leaving an
operator to set each one. The `desktop` default asks for no identity, no CSRF token
and no rate limit, because the only user is the person at the keyboard. The `server`
profile requires all three, delegates identity to a proxy through a trusted header,
bounds every upload before it is parsed, runs data checks before any estimator is
fit, and refuses a model bundle that is unsigned, altered or signed elsewhere.

The server profile needs a platform whose kernel applies `RLIMIT_AS`, which is Linux
and the BSDs. On a platform that does not, an isolated step is refused rather than
run without its memory limit.

What it does not do is stated in the Open items section of `docs/SECURITY_REVIEW.md`
and is not repeated here, so that the two cannot disagree. The most important of them
is that there is no account system: the session cookie identifies a browser, not a
person, so two people behind one proxy identity share a session's data. A network
deployment for untrusted users needs that gap closed first.

## Dependencies

`requirements.txt` pins every runtime version and `THIRD_PARTY_NOTICES.md` lists each
one with its licence. `python scripts/check_requirements.py` reports any that fail to
import. The frontend has no runtime JavaScript dependency; ESLint is a development
dependency only, so `npm audit --omit=dev` is the check that matters for a release.
