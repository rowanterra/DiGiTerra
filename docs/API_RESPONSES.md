# API response shape

## Success

Routes return the payload at the top level of the JSON body. For example, `/upload` returns `{ "filename": "...", "numcols": 10, ... }`. There is no common wrapper, and each route defines its own fields.

Long-running work (training) reports progress through an `EventSource` stream at `/progress/<id>`. An unknown id returns 404.

## Errors

HTTP status codes: `400` for a bad request, `404` for a missing resource, `413` for a request body over the upload limit, and `500` for a server error. Every JSON error body has an `error` string:

```json
{ "error": "Human-readable message" }
```

The handlers in `core/errors.py` produce this shape for `ApiError`, `RequestEntityTooLarge`, other HTTP exceptions, and unhandled exceptions. The scripts in `static/js/` show `data.error` when present and a generic "See console for details" message otherwise.

## Main endpoints

| Route | Success | Error |
|-------|---------|-------|
| `POST /upload` | `filename`, `numcols`, `firstcol`, `lastcol`, ... | `{ "error": "..." }` |
| `POST /preprocess` | Preprocess result | `{ "error": "..." }` |
| `POST /process` | Model results (SSE progress, then JSON) | `{ "error": "..." }` |
| `POST /predict` | Prediction payload | `{ "error": "..." }` |

The request and response fields are defined in the handlers under `routes/` (`upload.py`, `preprocess.py`, `modeling.py`, `prediction.py`).
