# Accessibility

The guided page (`/guided`) is built to WCAG 2.2 level AA. This note says what that means in the page and what has been checked; it is not a conformance claim for the classic page.

## What the guided page does

- Landmarks: one `header`, one `main`, a `nav` of step buttons (the current one marked `aria-current="step"`), a `nav` for the link to the classic page, and one `section` per step, each labelled by its heading. One step is shown at a time and fills the window, so nothing on the page scrolls; the hyperparameter table and the raw result block are the two bounded scrolling regions. A skip link is the first focusable element.
- Every control has a visible `label` tied to it by `for`; radio groups are inside a `fieldset` with a `legend`. The Guided and Full modes are a radio group, so a screen reader announces a choice of two, not an on-off switch.
- Errors are written to one region with `role="alert"` and `aria-live="assertive"`; upload, progress and prediction status are `aria-live="polite"`.
- Focus moves to a step's heading when the user opens it (after an upload, after a result, or with a step button). Defaults that fill steps 2 and 3 do not move focus.
- Every control has an explanation. The words of its label have a dotted underline and open a tooltip on hover; keyboard focus of the control opens the same tooltip, and Escape closes it. The tooltip stays open while the pointer is over it, is outside the label so the control's accessible name is the label alone, and is tied to the control by `aria-describedby`, so it is read with the control (WCAG 1.4.13: hoverable, dismissible, persistent). Hyperparameter rows and result metrics get the same treatment; a result metric's header takes a Tab stop of its own. Until 2026-09-25 a separate help button beside each label opened the tooltip.
- Steps that are not yet opened have their step buttons disabled, and only the current step is in the document flow, so tab order stays on it.
- Colour never stands alone: data-check findings include the word "blocking" or "warning"; the done state of a step is a border and the step's content.
- Tables have captions and `scope` on header cells. The visible focus ring is a 3 px outline on every focusable element. No time limit, no motion beyond the progress bar, and `prefers-reduced-motion` turns transitions off.
- Text is served at 17 px on a warm off-white ground with ink-dark text; the smallest text is 0.8 rem in the raw result block, which is a collapsed `details` element.

## What has been checked

- `tests/integration/test_browser_guided.py` runs the page in Chromium: landmarks and labels, keyboard order from the skip link, an explanation on every control, the mode radio group and its persistence, a full run in Full mode with a typed hyperparameter read back from the fitted estimator, no scrolling on any step in either mode at three laptop viewports, and an axe-core run with the WCAG 2.x A and AA rule sets, which must report no serious or critical violation. The test skips when Playwright's Chromium or axe-core (`npm install`) is absent; set `DIGITERRA_CHROMIUM` to a Chromium binary to run it against a pinned browser.
- `docs/guided_page_aria_transcript.txt` (2026-09-24, before the help buttons were replaced by the hover words) is the accessibility tree of every step in both modes, written out in Playwright's ARIA snapshot form (role, accessible name, state), with the text of every tooltip and a list of every Tab stop on the Full step 3. It is what Chromium exposes to assistive technology, which is what a screen reader reads from; it is not a recording of a screen reader session. Reading it found one defect on 2026-09-24: the help button was inside each label, so a control's accessible name was the label plus the button's label ("CSV, Excel or Parquet file What is CSV, Excel or Parquet file?"). The button now follows the label, and the browser test asserts that the file input's accessible name is the label alone.
- The browser review of 2026-09-24 found that a tooltip opened by keyboard focus did not close on Escape (the focus kept it open); Escape now dismisses it until focus leaves the control.
- A keyboard-only walk and a VoiceOver walk on macOS are part of the release checklist in `CONTRIBUTING.md` and are done by a person, not the test.

## What is not claimed

No VPAT has been written. The classic page (`/`) predates this work and has not been audited. A conformance statement waits on a session with a user of assistive technology.
