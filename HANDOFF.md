# DiGiTerra: Handoff Notes for New Developers

Hi there, welcome to the project. These notes are here to make your life easier when you pick this up, integrate it into a website, or hand it off again. Think of this as “what we wish we’d had on day one.”

---

## Quick Start (Get It Running in ~2 Minutes)

```bash
pip install -r requirements.txt
python app.py
```

Then open **http://127.0.0.1:5000** in your browser. That’s the web UI. For the desktop app (pywebview wrapper), run `python desktop_app.py` instead.

---

## Repo Layout: Where Things Live

| Path | What it is |
|------|------------|
| **`app.py`** | Main Flask app. All HTTP routes, upload handling, model training orchestration, and download endpoints live here. It’s large; use the section comments at the top of each block to navigate. |
| **`desktop_app.py`** | Thin launcher that starts the Flask server in a background thread and opens a pywebview window. Handles “Save file” dialogs for the desktop build. |
| **`templates/index.html`** | Single-page UI. All tabs (Upload, Data Exploration, Model Preprocessing, Modeling, Inference) are in this one file. |
| **`static/client_side.js`** | All front-end logic: uploads, API calls, progress polling, result display, downloads. |
| **`static/style.css`** | Styles. |
| **`python_scripts/`** | Core ML and preprocessing. `helpers.py` has shared helpers (e.g. `preprocess_data`, `prediction`). `config.py` sets `VIS_DIR` (output folder). Model trainers live under `models/` (regression, classification, clustering). Pipelines and plotting are under `preprocessing/` and `plotting/`. |
| **`deploy/`** | Docker and Kubernetes (Helm) deployment. See `deploy/README.md` for Docker build/run and Helm install. |
| **`examples/`** | Example datasets: 3 classification, 3 regression, 3 clustering. See `examples/README.md`. Not bundled in the app. |
| **`docs/`** | Extra guides: build instructions, feature plans, git sync, etc. |

---

## Important Behaviors to Know

- **Single-user, in-memory state**  
  `app.py` uses a global `memStorage` dict for models, data, scalers, and feature order. Fine for desktop or a single-user lab tool. **If you put this behind a multi-user website**, you’ll need to replace this with something like session-based storage or a proper backend store.

- **CSV only**  
  Upload and prediction endpoints accept only `.csv`. Validation is in `allowed_file()` and we use `secure_filename()` for safe filenames. Keep that in mind if you add new upload types.

- **Outputs**  
  Plots, PDFs, and Excel files go to `USER_VIS_DIR` (see “Paths” below). The UI fetches them via `/user-visualizations/<filename>` (for display) and `/download/<path:filename>` (for download). Path traversal is blocked on the download route.

---

## Paths and Environment

- **`USER_VIS_DIR`** (visualizations, predictions, etc.):  
  - macOS: `~/Library/Application Support/DiGiTerra/user_visualizations/`  
  - Windows: `%APPDATA%\DiGiTerra\user_visualizations\`  
  - Linux: `~/.local/share/DiGiTerra/user_visualizations/`  

  Override with `DIGITERRA_OUTPUT_DIR`.

- **Uploads** go to `APP_SUPPORT_DIR / "uploads"` (same base as above, but `uploads` subdir).

- **Base directory** for templates/static can be overridden with `DIGITERRA_BASE_DIR` (useful for PyInstaller builds).

---

## Security & Hardening: What’s Done and What to Watch

**Already in place:**

- File type check: only `.csv` allowed; `allowed_file()` + `secure_filename()` on upload and prediction.
- Path traversal fix on **`/download/<path:filename>`**: we resolve paths and ensure the requested file stays under `USER_VIS_DIR` (see `download_visualization` in `app.py`).
- CSV read errors are caught and returned as clear HTTP responses instead of 500s.
- Input validation on important routes (preprocess, model training, correlation matrices, pairplot, etc.).

**If you deploy this as a public or multi-user web app:**

1. **`memStorage`**: Replace with per-user/session storage so one user’s models and data don’t leak to another.
2. **CSRF**: Flask doesn’t add CSRF by default. Add something like `Flask-WTF` or CSRF tokens for state-changing requests.
3. **Reverse proxy**: Run behind nginx or similar. Don’t expose the Flask dev server directly to the internet.
4. **HTTPS**: Use TLS at the proxy; the app itself doesn’t handle SSL.
5. **Dependencies**: `requirements.txt` currently omits version pins. Consider pinning versions (e.g. in a `requirements.lock` or similar) for reproducible, auditable builds.

**`ast.literal_eval`:**  
Used only for MLP `hidden_layer_sizes` (tuple of ints from the UI). Safer than `eval`, but if you ever accept less trusted input there, add validation (e.g. check type and value ranges) to avoid surprises.

---

## Integrating Into a Website

- The app is a standard Flask app. You can run it as a WSGI app (e.g. gunicorn + nginx) or keep using the Docker image and put the container behind your existing web stack.
- The UI is one HTML page + JS + CSS. You could embed it in an iframe, or reverse-proxy specific paths to this service. Just ensure `/`, `/static/`, `/user-visualizations/`, `/download/`, and the API routes your JS calls are all proxied correctly.
- CORS isn’t configured. If the front end is served from another origin, you’ll need to add CORS headers (e.g. `flask-cors`) for the relevant routes.

---

## Existing Docs to Use

- **`README.md`**: User-facing run instructions, desktop build, and config (ports, debug, log paths).
- **`docs/documentation.md`**: High-level workflow and concepts.
- **`docs/BUILD_INSTRUCTIONS.md`**: Cross-platform desktop build (PyInstaller).
- **`docs/FEATURE_ADDITION_PLAN.md`**: Ideas and suggested spots for new features.
- **`ISSUES_FOUND.md`**: Historical bug list and fixes; **`DEBUG_REPORT.md`**: Current status and checks.

---

## Handy Commands

```bash
# Dependency check (catches missing imports)
python scripts/check_requirements.py

# Docker build & run
docker build -f deploy/docker/Dockerfile -t digiterra:local .
docker run --rm -p 5000:5000 digiterra:local

# Kubernetes (Helm)
helm install digiterra deploy/helm/digiterra --set image.repository=<your-registry>/digiterra --set image.tag=latest
# With persistence: --set persistence.enabled=true --set persistence.size=5Gi
# See deploy/README.md for options.
```

---

## Summary

- **`app.py`** = HTTP API + orchestration; **`desktop_app.py`** = desktop wrapper.  
- **`memStorage`** = in-memory, single-user; change this for multi-user web.  
- **Security:** upload validation and download path traversal are addressed; add CSRF, reverse proxy, HTTPS, and dependency pinning for production web use.  
- **Paths:** `USER_VIS_DIR`, `DIGITERRA_OUTPUT_DIR`, and `DIGITERRA_BASE_DIR` control where things live.

If something’s unclear or you find a gotcha, consider dropping a note in `HANDOFF.md` or `ISSUES_FOUND.md` for the next person. Good luck with the website integration.
