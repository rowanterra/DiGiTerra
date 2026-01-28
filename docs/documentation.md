# DiGiTerra Documentation

## Workflow Overview
DiGiTerra follows a consistent workflow:

1. **Upload** a CSV file to populate the working dataset.
2. **Data Exploration** builds correlation matrices, pairplots, and descriptive statistics.
3. **Model Preprocessing** selects targets/indicators, cleaning, and transformations.
4. **Modeling** trains the model and calculates metrics.
5. **Downloads** provide PDF visuals and spreadsheet summaries.

## Core Concepts
- **Indicators** are the input columns used to predict targets.
- **Targets** are the output columns you want to predict or cluster.
- **Transformers** encode non-numeric columns for modeling.
- **Stratification** balances splits or evaluation based on a selected variable.

## File Format & Column Entry Tips
- **CSV only:** DiGiTerra expects CSV files. If you have an Excel spreadsheet, open it and choose **File → Save As** to export a `.csv` file.
- **Column letters:** Columns are selected using spreadsheet-style letters (A, B, C...). Ranges look like `A-D` and multiple selections can be separated with commas (for example: `A-D, F, H`).
- **Indicators vs targets:** Indicators are the inputs; targets are the outputs you want to predict.

## Process Map (Flow Chart)

```
Upload  ->  Data Exploration  ->  Model Preprocessing  ->  Modeling  ->  Outputs
              |                        |                      |
              +------ Backend Pipelines +---------------------+
```

## System Touchpoints
- Front-end UI: `templates/index.html`, `static/client_side.js`
- Styling: `static/style.css`
- Model orchestration: `app.py`
- Preprocessing + plotting: `python_scripts/preprocessing/*`, `python_scripts/plotting/*`

## Accessibility Features

DiGiTerra includes comprehensive accessibility features to ensure the application is usable by individuals with disabilities and those using assistive technologies.

### ARIA Attributes and Semantic HTML

The application uses ARIA (Accessible Rich Internet Applications) attributes throughout the interface to provide context and state information to screen readers and other assistive technologies. Key implementations include:

- **ARIA roles**: Navigation elements use `role="navigation"`, status messages use `role="status"` or `role="alert"`, and interactive regions use `role="region"` with descriptive labels
- **ARIA live regions**: Dynamic content updates are announced to screen readers using `aria-live` attributes with appropriate priority levels (`polite` for general updates, `assertive` for critical errors)
- **ARIA labels**: Interactive elements without visible text labels include `aria-label` attributes to provide accessible names
- **ARIA describedby**: Form inputs reference help text using `aria-describedby` to provide additional context
- **ARIA required**: Required form fields are marked with `aria-required="true"` to indicate mandatory inputs
- **ARIA invalid**: Form validation errors set `aria-invalid="true"` to communicate validation state

### Keyboard Navigation

Full keyboard navigation support is provided throughout the application:

- **Skip link**: A "Skip to main content" link appears when focused, allowing keyboard users to bypass repetitive navigation elements
- **Focus management**: The application programmatically manages focus to guide users through workflows and direct attention to important updates or errors
- **Focus indicators**: All interactive elements have visible focus styles with high contrast outlines to clearly indicate keyboard focus position
- **Tab order**: Logical tab order ensures that interactive elements are reached in a meaningful sequence

### Screen Reader Support

Screen reader users receive comprehensive information about the application state:

- **Screen reader announcements**: Dynamic content changes, error messages, and status updates are automatically announced using live regions
- **Descriptive labels**: All form controls, buttons, and interactive elements have accessible names that clearly describe their purpose
- **Hidden text**: Contextual help text and additional descriptions are available to screen readers using the `sr-only` class while remaining visually hidden
- **Toggle enhancements**: Toggle switches and checkboxes automatically receive ARIA labels based on their associated text or structure

### Form Accessibility

Forms are designed to be accessible and provide clear feedback:

- **Required field indicators**: Required fields are marked both visually and programmatically
- **Error communication**: Validation errors are announced to screen readers and associated with the relevant form fields
- **Help text**: Input fields include accessible help text that explains expected formats and requirements
- **Field relationships**: Related form fields are grouped and labeled appropriately

### Visual Accessibility

The interface supports users with visual impairments:

- **High contrast focus indicators**: Focus states use high contrast colors and sufficient outline width for visibility
- **Semantic structure**: Content is organized using proper heading hierarchy and semantic HTML elements
- **Language declaration**: The HTML document declares its language (`lang="en"`) to assist screen readers with pronunciation

These accessibility features ensure that DiGiTerra can be effectively used with screen readers, keyboard-only navigation, and other assistive technologies, making the application accessible to a broader range of users.
