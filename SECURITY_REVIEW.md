# DiGiTerra Security Review

**Date:** January 27, 2026  
**Status:** Production-ready for desktop use; additional hardening recommended for multi-user web deployment

---

## Summary

DiGiTerra is a desktop application for data exploration, preprocessing, and machine learning model training. This document reviews security for both desktop and potential web deployment scenarios.

The application has addressed critical security vulnerabilities and is production-ready for desktop use. For multi-user web deployment, additional security measures are needed (see "Security Considerations for Web Deployment" below).

---

## Security Measures Already Implemented

### File Upload Security

1. **File Type Validation**
   - Only `.csv` files are accepted via `ALLOWED_EXTENSIONS = {'csv'}`
   - Validation performed using `allowed_file()` function
   - **Location:** `app.py` lines 150, 187-189, 452-453, 2429-2430

2. **Filename Sanitization**
   - All uploaded filenames sanitized using `werkzeug.utils.secure_filename()`
   - Prevents path traversal attacks through malicious filenames
   - **Location:** `app.py` lines 456, 2433

3. **File Size Warnings**
   - Advisory warnings for large files (>50MB) and high cell counts
   - Non-restrictive (does not block uploads)
   - **Location:** `app.py` lines 153-155, 464-477

### File Download Security

1. **Path Traversal Protection**
   - Download route (`/download/<path:filename>`) validates that requested files stay within `USER_VIS_DIR`
   - Uses `Path.resolve()` and `Path.relative_to()` to prevent directory traversal
   - Only files (not directories) are served
   - **Location:** `app.py` lines 370-383

2. **Filename Sanitization on Download**
   - Download filenames sanitized using `secure_filename()`
   - **Location:** `app.py` line 382

### Input Validation

1. **CSV Reading Error Handling**
   - Comprehensive try/except blocks for CSV parsing errors
   - Specific error types handled: `EmptyDataError`, `ParserError`
   - **Location:** `app.py` lines 357-366, 2442-2449

2. **Route Input Validation**
   - Validation on critical routes: `/preprocess`, model training, `/correlationMatrices`, `/pairplot`
   - **Location:** `app.py` lines 632-633, 663-669, 436-437, 578-589, 752

### Platform Security

1. **Cross-Platform Path Handling**
   - Platform-specific paths for Windows, Linux, and macOS
   - Prevents hardcoded path vulnerabilities
   - **Location:** `app.py` lines 34-40

2. **Container Security (Docker)**
   - Runs as non-root user (`app:app`)
   - Minimal base image (`python:3.11-slim`)
   - **Location:** `deploy/docker/Dockerfile` lines 8, 18

---

## Security Choices & Design Decisions

### Acceptable for Desktop Use

1. **Global `memStorage`**
   - **Decision:** Acceptable for single-user desktop application
   - **Rationale:** Desktop apps run in user's own environment; no multi-user risk
   - **Action Required:** Only if deploying as multi-user web service

2. **No Authentication**
   - **Decision:** Acceptable for desktop application
   - **Rationale:** Desktop apps run locally; OS-level security applies
   - **Action Required:** Only if deploying as web service

3. **Flask Development Server**
   - **Decision:** Acceptable for desktop application
   - **Rationale:** Desktop apps typically run on localhost only
   - **Action Required:** Use production WSGI server for web deployment

### Security Trade-offs

1. **File Size Warnings (Non-Restrictive)**
   - **Decision:** Warn but don't block large files
   - **Rationale:** User may legitimately need to process large datasets
   - **Risk:** Resource exhaustion if maliciously large files uploaded
   - **Mitigation:** Acceptable for desktop; add hard limits for web deployment

2. **No Dependency Version Pinning**
   - **Decision:** Flexibility in dependency versions
   - **Rationale:** Easier updates, but less reproducible builds
   - **Risk:** Potential for vulnerable or incompatible versions
   - **Mitigation:** Pin versions for production deployments

---

## Summary of Recommendations

### For Desktop Use (Current State)
**Status:** Production Ready

No additional security measures required. Current security measures are appropriate for single-user desktop deployment.

### For Web Deployment (Future)

Before deploying as a multi-user web service, implement:
1. Session-based storage (replace `memStorage`)
2. CSRF protection
3. Authentication & authorization
4. Rate limiting
5. Production WSGI server
6. Reverse proxy with HTTPS
7. Dependency version pinning

---

## References

- **Security Fixes:** See `ISSUES_FOUND.md` for historical security fixes
- **Developer Notes:** See `HANDOFF.md` for integration guidance
- **Current Status:** See `DEBUG_REPORT.md` for verification results

---

Next review: 1-3 months or after major changes
