# March edits: URL prefix and SVC

## Scope

Run the app under a URL prefix (for example `http://127.0.0.1:5000/digiterra/`) for subpath hosting, with every feature working at the prefixed path.

## Done

URL prefix: route registration and response URLs use the prefix. Pairplot, correlation, and download paths are prefixed. In the page, `API_ROOT` and the helpers in `core.js` apply the prefix to fetch calls, the SSE stream, image paths, and download links, and the template asset paths use the `api_root` and `static_root` variables. Routes and static assets answer at both the root and the prefixed path.

SVC: the duplicate element id for the gamma input is removed, the gamma lookup for the rbf kernel is corrected, and an empty `class_weight` is normalized so scikit-learn accepts the value.

## Check

From the project root:

```bash
export URL_PREFIX=digiterra
python app.py
```

Open `http://127.0.0.1:5000/digiterra/`. Upload a CSV, run correlation and pairplot, run training with the progress stream, download a file, and run SVC with the rbf and poly kernels.

`docs/documentation.md` describes the URL prefix and `deploy/README.md` the container deployment.
