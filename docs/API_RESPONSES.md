# API response contract

DiGiTerra’s JSON API uses a simple, consistent shape where possible.

## Success

- **Success with data:** Many routes return JSON with the payload at the top level (e.g. `{ "filename": "...", "numcols": 10, ... }` for `/upload`). There is no single `{ "ok": true, "data": ... }` wrapper; each route documents its own fields.
- **Streaming:** Progress and long-running flows may use `EventSource` or streaming responses; see the route implementation.

## Errors

- **HTTP status:** Use standard codes: `400` (validation/bad request), `404` (not found), `500` (server error).
- **JSON body:** When returning JSON for an error, include an `error` string so the client can show it:
  - `{ "error": "Human-readable message" }`
- The frontend (`static/client_side.js`) shows `data.error` when present and falls back to a generic “See console for details” message.

## Main endpoints (summary)

| Route / area        | Success (typical) | Error (JSON)   |
|---------------------|------------------|----------------|
| `POST /upload`      | `filename`, `numcols`, `firstcol`, `lastcol`, … | `{ "error": "..." }` |
| `POST /preprocess` | Preprocess result payload | `{ "error": "..." }` |
| `POST /process`     | Model results (SSE or JSON) | `{ "error": "..." }` |
| `POST /predict`     | Prediction payload | `{ "error": "..." }` |

For full request/response shapes, see the route handlers in `app.py`.
