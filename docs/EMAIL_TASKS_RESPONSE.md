# March requested edits (URL prefix and SVC)

## Scope

Run the app under a URL prefix (e.g. `http://127.0.0.1:5000/digiterra/`) for subpath hosting. All features must work at the prefixed path.

## Done

**URL prefix**
- Backend: route registration and response URLs use prefix. Pairplot, correlation, and download paths are prefixed.
- Frontend: `API_ROOT` and helpers applied to fetch, SSE, image paths, and download links. Template asset paths use prefix.
- Routes and static assets work at both root and prefixed paths. Template variables: `api_root`, `static_root`.

**SVC**
- Duplicate element id for SVC gamma input removed.
- Gamma field lookup corrected for rbf kernel.
- Empty `class_weight` normalized so sklearn accepts it.

## How to test

From project root:

```bash
export URL_PREFIX=digiterra
python app.py
```

Open `http://127.0.0.1:5000/digiterra/`. Upload a CSV, run correlation and pairplot, run training (progress stream), download a file, run SVC with rbf and poly kernels.

See docs/HANDOFF.md for more on URL prefix and deployment.
