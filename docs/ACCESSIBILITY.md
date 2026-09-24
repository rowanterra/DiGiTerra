# Accessibility

The guided page (`/guided`) is built to WCAG 2.2 level AA. This note says what that means in the page and what has been checked; it is not a conformance claim for the classic page.

## What the guided page does

- Landmarks: one `header`, one `main`, a `nav` for the link to the classic page, and one `section` per step, each labelled by its heading. A skip link is the first focusable element.
- Every control has a visible `label` tied to it by `for`; radio groups are inside a `fieldset` with a `legend`. The Guided and Full modes are a radio group, so a screen reader announces a choice of two, not an on-off switch.
- Errors are written to one region with `role="alert"` and `aria-live="assertive"`; upload, progress and prediction status are `aria-live="polite"`.
- Focus moves to the next step's heading only when the user's own action opened it (after an upload, and after a result). Defaults that fill steps 2 to 4 do not move focus.
- Steps that are not yet usable have `aria-disabled="true"` and their controls are disabled, so tab order stays on the current step.
- Colour never stands alone: data-check findings include the word "blocking" or "warning"; the done state of a step is a border and the step's content.
- Tables have captions and `scope` on header cells. The visible focus ring is a 3 px outline on every focusable element. No time limit, no motion beyond the progress bar, and `prefers-reduced-motion` turns transitions off.
- Text is served at 17 px on a warm off-white ground with ink-dark text; the smallest text is 0.8 rem in the raw result block, which is a collapsed `details` element.

## What has been checked

- `tests/integration/test_browser_guided.py` drives the page in Chromium: landmarks and labels, keyboard order from the skip link, the mode radio group and its persistence, a full run in Full mode with a typed hyperparameter read back from the fitted estimator, and an axe-core run with the WCAG 2.x A and AA rule sets, which must report no serious or critical violation. The test skips when Playwright's Chromium or axe-core (`npm install`) is absent; set `DIGITERRA_CHROMIUM` to a Chromium binary to run it against a pinned browser.
- A keyboard-only walk and a VoiceOver walk on macOS are part of the release checklist in `CONTRIBUTING.md` and are done by a person, not the test.

## What is not claimed

No VPAT has been written. The classic page (`/`) predates this work and has not been audited. A conformance statement waits on a session with a user of assistive technology.
