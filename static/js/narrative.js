/**
 * The application as a narrative: one rail down the left carrying navigation, the
 * loaded data and the choices for the run, with the views to its right.
 *
 * The rail is a sibling of the views, built once on DOMContentLoaded, so it is
 * there from a cold load rather than appearing after an upload. Its navigation is
 * the old top tab bar: the same button nodes are moved into it, so every id, every
 * listener and setActiveTab's NodeList keep working. Entries after Upload are
 * disabled until a file is loaded, and say why in their title.
 *
 * The five-step strip is not navigation. It stays inside the Modeling view and
 * reports where a run has got to.
 *
 * Everything is built from the page that is already there. The mode sections keep
 * their markup, their element ids and their listeners; nodes are moved, never
 * rebuilt, so nothing that reads a control by id is affected. The chosen model is read
 * from the model <select> elements rather than from a second list, so it cannot
 * drift from the registry. Depends on core.js.
 */

/* eslint-disable no-unused-vars */

// The five steps of a run, in order. The names match what the rest of the page
// calls them, so a person reading the strip and a person reading the tabs see
// the same words.
const RUN_STEPS = [
    { key: 'upload', label: 'Data' },
    { key: 'preprocess', label: 'Preprocessing' },
    { key: 'setup', label: 'Model setup' },
    { key: 'run', label: 'Run' },
    { key: 'results', label: 'Results' },
];

// The views the rail navigates, in the order a run visits them. These are the
// buttons the header used to carry; the nodes are moved, not recreated, so the
// data-tab values and the handlers core.js bound to them are untouched.
const RAIL_VIEWS = ['upload', 'processing', 'model-preprocessing', 'modeling', 'historic'];

// Everything after Upload needs a file. The reason is what the entry says when a
// person hovers a disabled one, so the rail explains itself rather than going dead.
const RAIL_NEEDS_DATA = 'Upload a file first.';

// The model select of each mode, by task. switchModelingMode shows one mode at a
// time; the choices list reads whichever select belongs to the visible mode.
const MODEL_SELECTS = {
    simple: { regression: 'simpleNModels', classification: 'simpleClassModels', cluster: 'simpleClModels' },
    advanced: { regression: 'advancedNModels', classification: 'advancedClassModels', cluster: 'advancedClModels' },
    automl: { regression: 'automlNModels', classification: 'automlClassModels', cluster: 'automlClModels' },
};

function narrativeCurrentMode() {
    for (const mode of ['automl', 'simple', 'advanced']) {
        const radio = document.getElementById(mode + 'Mode');
        if (radio && radio.checked) return mode;
    }
    return 'simple';
}

/** The visible model select for the current mode, whatever the task type. */
/** The task the run is for, from the choice the app recorded. */
function narrativeTaskKey() {
    const el = document.getElementById('outputType1');
    const value = el && el.value ? el.value : '';
    if (value === 'Classifier') return 'classification';
    if (value === 'Cluster') return 'cluster';
    if (value === 'Numeric') return 'regression';
    return '';
}

/**
 * The model select the choices list reads.
 *
 * The task comes from the recorded choice rather than from which select happens to
 * be on screen. Picking by visibility made the count follow the page: on Modeling a
 * classification select was visible and the chip read 23, and on Documentation
 * nothing was visible so the first entry won and the same session read 31.
 */
function narrativeActiveSelect() {
    const byTask = MODEL_SELECTS[narrativeCurrentMode()] || {};
    const task = narrativeTaskKey();
    if (task && byTask[task]) {
        const el = document.getElementById(byTask[task]);
        if (el) return el;
    }
    // No task chosen yet. The regression select is the empty state the rail shows.
    for (const id of Object.values(byTask)) {
        const el = document.getElementById(id);
        if (el && el.offsetParent !== null) return el;
    }
    for (const id of Object.values(byTask)) {
        const el = document.getElementById(id);
        if (el) return el;
    }
    return null;
}

/** Two initials for the tile of an app, as KBase draws them. */
function narrativeInitials(label) {
    const words = String(label).replace(/[()]/g, '').split(/[\s_-]+/).filter(Boolean);
    if (!words.length) return '??';
    if (words.length === 1) return words[0].slice(0, 2).toUpperCase();
    return (words[0][0] + words[1][0]).toUpperCase();
}

/** Whether the app has a loaded file, by the same test the views use. */
function narrativeHasData() {
    try {
        if (typeof checkHasDataForExploration === 'function') return Boolean(checkHasDataForExploration());
    } catch (err) {
        console.error('Data check failed:', err);
    }
    return Boolean(typeof uploadedFileName === 'string' && uploadedFileName);
}

/** Whether a task type has been chosen, which is what gives the model line a value. */
function narrativeTaskChosen() {
    const el = document.getElementById('outputType1');
    return Boolean(el && el.value);
}

/**
 * Put the page into a rail and a view column, once, on load.
 *
 * main's children are moved into the view column rather than re-created, so the
 * views keep their ids and their listeners. The rail becomes their sibling, which
 * is what lets it exist before any of them has anything in it.
 */
function narrativeBuildShell() {
    const main = document.getElementById('main-content');
    if (!main || document.getElementById('narrativePage')) return false;

    const page = document.createElement('div');
    page.id = 'narrativePage';
    page.className = 'narrative-page is-single';

    const rail = document.createElement('aside');
    rail.id = 'narrativeRail';
    rail.className = 'narrative-rail';
    rail.setAttribute('aria-label', 'Navigation and data');
    rail.innerHTML = `
        <nav class="narrative-nav" id="narrativeNav" aria-label="Main navigation"></nav>
        <section class="narrative-panel" id="narrativeDataPanel">
            <div class="narrative-panel-head">
                <h3>Data</h3>
                <span class="narrative-count" id="narrativeDataCount">0</span>
            </div>
            <div class="narrative-panel-body" id="narrativeDataBody">
                <p class="narrative-empty">No data loaded.</p>
            </div>
        </section>
    `;

    const views = document.createElement('div');
    views.id = 'narrativeViews';
    views.className = 'narrative-views';
    while (main.firstChild) views.appendChild(main.firstChild);

    page.appendChild(rail);
    page.appendChild(views);
    main.appendChild(page);
    return true;
}

/**
 * Move the tab bar into the rail and retire the bar itself.
 *
 * The buttons are moved, not rebuilt: core.js captured them in a NodeList at load
 * and bound a click handler to each, and setActiveTab still writes to those same
 * nodes. Documentation goes last, in its own group.
 */
function narrativeMoveNav() {
    const nav = document.getElementById('narrativeNav');
    if (!nav || nav.childElementCount) return;
    const bar = document.getElementById('appTabs');
    if (!bar) return;

    RAIL_VIEWS.forEach((tab) => {
        const btn = bar.querySelector(`.tab-button[data-tab="${tab}"]`);
        if (btn) nav.appendChild(btn);
    });
    const rest = Array.from(bar.querySelectorAll('.tab-button[data-tab]')).filter(
        (b) => b.dataset.tab !== 'documentation'
    );
    rest.forEach((b) => nav.appendChild(b));

    // Documentation sits in the header's right-hand group rather than in the bar,
    // so it is found on the page rather than inside it.
    // Documentation and Tutorial stay in the banner. They have to work with no data
    // loaded, and with no data there is no rail to put them in.
    // The bar keeps its node so core.js can still address it; empty, it draws nothing.
    bar.classList.add('is-retired');

    // The forward step lives at the foot of the rail, under the entry it leaves.
    const next = document.createElement('button');
    next.type = 'button';
    next.id = 'narrativeNext';
    next.className = 'success-button narrative-next hidden';
    next.addEventListener('click', () => {
        const source = narrativeNextSource();
        if (source && !source.disabled) source.click();
    });
    nav.appendChild(next);
}

// The in-page button that carries each view forward. The rail's forward button is
// a proxy for it: same label, same enabled state, same click. The page keeps its
// own button as the thing that holds the behaviour; the rail only repeats it.
const NEXT_SOURCE = {
    processing: 'continueToPreprocessButton',
    'model-preprocessing': 'continueToModelingButton',
    modeling: 'continueToInferenceButton',
};

function narrativeNextSource() {
    const active = document.querySelector('#narrativeNav .tab-button.active');
    const id = active ? NEXT_SOURCE[active.dataset.tab] : null;
    const source = id ? document.getElementById(id) : null;
    // A source inside a hidden panel is not on offer, so neither is the proxy.
    return source && !source.closest('.hidden') ? source : null;
}

/** Repeat the current view's forward button in the rail, or hide the proxy. */
function narrativeRefreshNext() {
    const next = document.getElementById('narrativeNext');
    if (!next) return;
    const source = narrativeNextSource();
    const show = Boolean(source);
    if (next.classList.contains('hidden') === show) next.classList.toggle('hidden', !show);
    if (!source) return;
    const label = source.textContent.trim();
    if (next.textContent !== label) next.textContent = label;
    if (next.disabled !== source.disabled) next.disabled = source.disabled;
    const title = source.getAttribute('title') || '';
    if ((next.getAttribute('title') || '') !== title) {
        if (title) next.setAttribute('title', title);
        else next.removeAttribute('title');
    }
}

/**
 * Disable what the app would refuse.
 *
 * Everything past Upload needs a file. A disabled button does not fire its click,
 * so the rail cannot reach a view the app would turn away, and the title says why.
 */
function narrativeGateNav() {
    const nav = document.getElementById('narrativeNav');
    if (!nav) return;
    const ready = narrativeHasData();
    nav.querySelectorAll('.tab-button[data-tab]').forEach((btn) => {
        const tab = btn.dataset.tab;
        const gated = tab !== 'upload' && tab !== 'documentation' && !ready;
        btn.disabled = gated;
        btn.classList.toggle('is-gated', gated);
        if (gated) {
            btn.setAttribute('title', RAIL_NEEDS_DATA);
            btn.setAttribute('aria-disabled', 'true');
        } else {
            btn.removeAttribute('title');
            btn.removeAttribute('aria-disabled');
        }
    });
}

/**
 * The run strip, inside the Modeling view.
 *
 * This is progress, not navigation: it says where a run has got to. It also takes
 * the page header, so Modeling carries one band rather than three.
 */
function narrativeBuildFlow() {
    const host = document.getElementById('columnSelection');
    if (!host || document.getElementById('narrativeFlow')) return false;

    const flow = document.createElement('nav');
    flow.id = 'narrativeFlow';
    flow.className = 'narrative-flow';
    flow.setAttribute('aria-label', 'Modeling and run progress');
    flow.innerHTML = `<ol>${RUN_STEPS.map(
        (s, i) =>
            `<li class="narrative-step" data-step="${s.key}"><span class="narrative-step-num">${i + 1}</span><span class="narrative-step-label">${s.label}</span></li>`
    ).join('')}</ol>`;

    host.insertBefore(flow, host.firstChild);
    // The page header is gone: the rail is the navigation and its active entry is the
    // title. The strip stays as one slim line of run progress at the top of the view.
    return true;
}

/** Mark the step the run has reached. Steps before it read as done. */
function narrativeSetStep(key) {
    const flow = document.getElementById('narrativeFlow');
    if (!flow) return;
    const order = RUN_STEPS.map((s) => s.key);
    const at = order.indexOf(key);
    flow.querySelectorAll('.narrative-step').forEach((li) => {
        const i = order.indexOf(li.dataset.step);
        li.classList.toggle('is-done', at > -1 && i < at);
        li.classList.toggle('is-current', i === at);
    });
    // Before a run the strip repeats the rail, so it is drawn only from the moment
    // a run starts: then it is progress the rail does not show.
    if (key === 'run' || key === 'results') flow.classList.add('is-live');
    if (key === 'upload' || key === 'preprocess' || key === 'setup') flow.classList.remove('is-live');
}

/** Draw the stages the server reports underneath the run strip.
 *
 * The five steps say where the run is; these say what it is doing inside the Run
 * step. The stage names and their state come from the same progress message the
 * loading area reads, so the two cannot disagree.
 */
function narrativeSetStages(progress) {
    const flow = document.getElementById('narrativeFlow');
    if (!flow) return;
    let host = document.getElementById('narrativeStages');
    if (!host) {
        host = document.createElement('div');
        host.id = 'narrativeStages';
        host.className = 'narrative-stages';
        flow.appendChild(host);
    }
    const stages = (progress && progress.stages) || {};
    const chips = Object.entries(stages)
        .filter(([, stage]) => stage.status !== 'skipped')
        .map(([name, stage]) => {
            const state = stage.status === 'running' ? 'running' : stage.status === 'completed' ? 'done' : 'waiting';
            const label = name.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
            const pct = state === 'running' ? ` ${Math.round(stage.progress || 0)}%` : '';
            return `<span class="narrative-stage is-${state}">${escapeHtml(label)}${pct}</span>`;
        });
    // The chips report a run in progress. Once every stage is done the strip's
    // Results step says the same thing, so the row goes away.
    const allDone = chips.length > 0 && Object.values(stages)
        .filter((stage) => stage.status !== 'skipped')
        .every((stage) => stage.status === 'completed');
    host.innerHTML = chips.length && !allDone ? chips.join('') : '';
    host.hidden = !chips.length || allDone;
}

/** Fill the data panel from what the page already knows about the upload. */
/** The letters a person typed, as a column range, and the name behind the first. */
function narrativeColumnLine() {
    const target = (document.getElementById('predictors') || {}).value || '';
    const indicators = (document.getElementById('indicators') || {}).value || '';
    if (!target && !indicators) return '';
    const named = (letters) => {
        const first = String(letters).trim().toUpperCase().match(/^[A-Z]+/);
        if (!first || typeof columnToIndex !== 'function') return '';
        const list = window.uploadedColumnNames;
        if (!Array.isArray(list)) return '';
        const name = list[columnToIndex(first[0])];
        return name ? ` (${name})` : '';
    };
    const parts = [];
    if (target) parts.push(`Target ${target.trim().toUpperCase()}${named(target)}`);
    if (indicators) parts.push(`indicators ${indicators.trim().toUpperCase()}`);
    return parts.join(', ');
}

function narrativeRefreshData() {
    const body = document.getElementById('narrativeDataBody');
    const count = document.getElementById('narrativeDataCount');
    if (!body) return;
    const name = typeof uploadedFileName === 'string' && uploadedFileName ? uploadedFileName : '';
    if (!name) {
        body.innerHTML = '<p class="narrative-empty">No data loaded.</p>';
        if (count) count.textContent = '0';
        return;
    }
    let shape = '';
    if (typeof uploadedShape === 'object' && uploadedShape && uploadedShape.rows) {
        shape = `${Number(uploadedShape.rows).toLocaleString()} \u00d7 ${uploadedShape.columns}`;
    }
    // The sentence the modeling page writes says the same thing at length. The panel
    // states the file, its shape, and which columns the run uses, once they are set.
    const columns = narrativeColumnLine();
    body.innerHTML = `
        <div class="narrative-item narrative-item-data">
            <span class="narrative-tile">CSV</span>
            <div>
                <strong>${escapeHtml(name)}</strong>
                <button type="button" class="narrative-replace" id="narrativeReplaceFile">Replace</button>
                ${shape ? `<p>${escapeHtml(shape)}</p>` : ''}
                ${columns ? `<p class="narrative-selection">${escapeHtml(columns)}</p>` : ''}
            </div>
        </div>
        ${narrativeChoicesHtml()}
    `;
    if (count) count.textContent = '1';
}

/* The choices the run will use, read from the controls on each page and listed
   under the file. The list is the same whichever page is open, so a person can
   see what Preprocessing decided while on Modeling. "change" opens the page and
   the setting. */
const CHOICE_PAGES = [
    { tab: 'model-preprocessing', label: 'Preprocessing', items: [
        { name: 'Blanks', read: () => {
            const drop = document.getElementById('dropMissing');
            const impute = document.getElementById('imputeStrategy');
            if (!drop) return '';
            const scope = { indicatorAndTarget: 'indicators and target', indicator: 'indicators', target: 'target', all: 'all columns' }[drop.value];
            if (!scope) return 'kept';
            const fill = impute && impute.value && impute.value !== 'none' ? (/^[0-9.]+$/.test(impute.value) ? `filled with ${impute.value}` : `filled with the ${impute.value}`) : 'rows removed';
            return `${fill} (${scope})`;
        }, ids: ['dataCleaningMissingWrapper', 'imputeDiv'] },
        { name: 'Text columns', read: () => {
            const use = document.getElementById('useTransformer');
            const cols = document.getElementById('transformerColumn');
            if (!use) return '';
            if (use.value !== 'Yes' && use.value !== 'yes') return 'none';
            return cols && cols.value ? `one-hot: ${cols.value}` : 'one-hot encoded';
        }, ids: ['transformerOptionWrapper', 'transformerYes'] },
        { name: 'Stratify', read: () => {
            const yn = document.getElementById('scalingYesNo');
            const col = document.getElementById('specificVariableSelect');
            if (!yn) return '';
            return (yn.value === 'Yes' || yn.value === 'yes') ? (col && col.value ? `by ${(col.options && col.options[col.selectedIndex]) ? col.options[col.selectedIndex].text : col.value}` : 'yes') : 'no';
        }, ids: ['scalingQuestion', 'scalingYes'] },
        { name: 'Scaler', read: () => { const e = document.getElementById('scaler'); return e ? (e.options[e.selectedIndex] || {}).text || e.value : ''; }, ids: ['preprocessScaler'] },
        { name: 'Split', read: () => { const e = document.getElementById('trainSize'); return e && e.value ? `${Math.round(Number(e.value) * 100)} / ${Math.round((1 - Number(e.value)) * 100)}` : ''; }, ids: ['preprocessSplit'] },
        { name: 'Seed', read: () => { const e = document.getElementById('seedValue'); return e && e.value ? String(e.value) : ''; }, ids: ['preprocessSeed'] },
    ] },
    // The model is chosen on Modeling, so its lines appear from Modeling onward.
    { tab: 'modeling', label: 'Modeling', views: ['modeling', 'historic'], items: [
        { name: 'Mode', read: () => ({ automl: 'AutoML', simple: 'Simple', advanced: 'Advanced' }[narrativeCurrentMode()] || ''), ids: ['modelingModeCard'] },
        { name: 'Model', read: () => {
            if (!narrativeTaskChosen()) return '';
            const sel = narrativeActiveSelect();
            if (!sel) return '';
            const opt = sel.value ? sel.options[sel.selectedIndex] : null;
            if (opt) return opt.textContent.trim();
            return narrativeCurrentMode() === 'automl' ? 'best of the compared candidates' : 'not chosen';
        }, ids: () => { const sel = narrativeActiveSelect(); return sel ? [sel.id] : []; } },
        { name: 'Search', read: () => {
            if (narrativeCurrentMode() !== 'advanced') return '';
            const e = document.getElementById('hyperparameterSearch');
            return e && e.value && e.value !== 'none' ? (e.options[e.selectedIndex] || {}).text || e.value : '';
        }, ids: ['hyperparameterSearch'] },
    ] },
];

const CHOICE_VIEWS = new Set(['model-preprocessing', 'modeling', 'historic']);

function narrativeChoicesHtml() {
    if (typeof preprocessColumnsChosen === 'function' && !preprocessColumnsChosen()) return '';
    // Upload and Data Exploration come before the choices are made; listing
    // them there confuses a first run, so the list starts at Model Preprocessing.
    const active = document.querySelector('#narrativeNav .tab-button.active');
    if (!active || !CHOICE_VIEWS.has(active.dataset.tab)) return '';
    let html = '';
    CHOICE_PAGES.forEach((page) => {
        if (page.views && !page.views.includes(active.dataset.tab)) return;
        const rows = page.items.map((it) => ({ it, value: it.read() })).filter((r) => r.value);
        if (!rows.length) return;
        html += `<li class="narrative-choice-page">${escapeHtml(page.label)}</li>`;
        rows.forEach(({ it, value }) => {
            html += `<li><strong>${escapeHtml(it.name)}</strong> ${escapeHtml(value)}<button type="button" class="narrative-choice-change" data-tab="${page.tab}" data-ids="${(typeof it.ids === 'function' ? it.ids() : it.ids).join(',')}">change</button></li>`;
        });
    });
    return html ? `<ul class="narrative-choices" aria-label="Choices for this run">${html}</ul>` : '';
}

document.addEventListener('click', (e) => {
    const btn = e.target.closest('.narrative-choice-change');
    if (!btn) return;
    const entry = document.querySelector(`#narrativeNav .tab-button[data-tab="${btn.dataset.tab}"]`);
    if (entry && !entry.disabled) entry.click();
    const ids = (btn.dataset.ids || '').split(',').filter(Boolean);
    if (typeof revealPreprocessBlocks === 'function') revealPreprocessBlocks(ids);
    const first = ids.map((id) => document.getElementById(id)).find((el) => el);
    if (first) {
        first.scrollIntoView({ block: 'center', behavior: 'smooth' });
        const ctl = first.matches('select, input, textarea') ? first : first.querySelector('select, input:not([type=hidden]), textarea');
        if (ctl) ctl.focus({ preventScroll: true });
    }
});

// The list follows the controls: any change on the preprocessing form redraws it.
document.addEventListener('change', (e) => {
    if (e.target.closest && e.target.closest('#preprocessform')) narrativeRefreshData();
});
document.addEventListener('input', (e) => {
    if (e.target.closest && e.target.closest('#preprocessform') && (e.target.id === 'trainSize' || e.target.id === 'seedValue')) narrativeRefreshData();
});
document.addEventListener('click', (e) => {
    if (e.target.closest && e.target.closest('#narrativeNav .tab-button, .narrative-next, #continueToPreprocessButton, #continueToModelingButton, #continueToInferenceButton')) {
        window.setTimeout(narrativeRefreshData, 80);
    }
});

/**
 * Show the rail once a file is loaded, and keep it from then on.
 *
 * With no data the rail would carry an empty data panel and five
 * entries that all refuse, so the welcome and upload screens take the whole page
 * instead. The shell is the same one either way: a class collapses the grid to one
 * column rather than a second layout being built.
 */
function narrativeApplyLayout() {
    const page = document.getElementById('narrativePage');
    if (!page) return;
    // narrativeGateNav reads the same answer, so the rail appears exactly when its
    // entries stop refusing.
    if (narrativeHasData()) page.classList.remove('is-single');
    else if (!page.dataset.railShown) page.classList.add('is-single');
    if (narrativeHasData()) page.dataset.railShown = 'true';
}

/** Bring the rail's parts up to date with whatever the app now knows. */
function narrativeSync() {
    narrativeApplyLayout();
    narrativeGateNav();
    narrativeRefreshNext();
    narrativeRefreshData();
}

function narrativeWire() {
    // The Data panel is redrawn, so the click is caught on the panel rather than
    // bound to a button that will not be the same node next time.
    const dataBody = document.getElementById('narrativeDataBody');
    if (dataBody) {
        dataBody.addEventListener('click', (e) => {
            if (!e.target.closest('#narrativeReplaceFile')) return;
            if (typeof openResetPopup === 'function') openResetPopup();
        });
    }

    ['indicators', 'predictors'].forEach((id) => {
        const el = document.getElementById(id);
        if (el) el.addEventListener('change', narrativeRefreshData);
    });

    ['automlMode', 'simpleMode', 'advancedMode'].forEach((id) => {
        const radio = document.getElementById(id);
        if (radio) {
            radio.addEventListener('change', () => {
                narrativeRefreshData();
            });
        }
    });
    Object.values(MODEL_SELECTS).forEach((byTask) => {
        Object.values(byTask).forEach((id) => {
            const el = document.getElementById(id);
            if (el) el.addEventListener('change', narrativeRefreshData);
        });
    });
}

/** Build the shell once, then keep it in step with the app. */
function narrativeInit() {
    try {
        if (!narrativeBuildShell()) return;
        narrativeMoveNav();
        narrativeWire();
        narrativeSync();
    } catch (err) {
        // A rail that fails to build must not take the application with it.
        console.error('Narrative shell not built:', err);
    }
}

document.addEventListener('DOMContentLoaded', () => {
    narrativeInit();

    // setActiveTab writes the active class on these same nodes, on every
    // navigation and once when an upload succeeds, so watching them is how the
    // rail learns that a file has arrived without anything having to call it.
    const nav = document.getElementById('narrativeNav');
    if (nav) {
        const watch = new MutationObserver(narrativeSync);
        nav.querySelectorAll('.tab-button[data-tab]').forEach((btn) =>
            watch.observe(btn, { attributes: true, attributeFilter: ['class'] })
        );
    }

    // The forward buttons change without navigation: Continue to Modeling is
    // enabled once a task type is chosen, and Continue to Inference is created when
    // a run finishes. Watching the views for those changes keeps the proxy honest.
    const views = document.getElementById('narrativeViews');
    if (views) {
        new MutationObserver(narrativeRefreshNext).observe(views, {
            subtree: true,
            childList: true,
            attributes: true,
            attributeFilter: ['disabled', 'class'],
        });
    }

    // The Modeling view is revealed by a class change rather than by navigation.
    // The run strip is built the first time it is shown and stays with it.
    const host = document.getElementById('columnSelection');
    if (!host) return;
    const build = () => {
        if (host.classList.contains('hidden')) return;
        narrativeBuildFlow();
        narrativeSetStep('setup');
        narrativeSync();
    };
    build();
    new MutationObserver(build).observe(host, { attributes: true, attributeFilter: ['class'] });
});


/* Documentation is reading, not the run, so it takes the whole page whether or
   not data is loaded. The section's own hidden class is the signal: the layout
   follows it, so every route into the documentation behaves the same. */
(function narrativeDocumentationLayout() {
    const start = () => {
        const doc = document.getElementById('documentation');
        const page = document.getElementById('narrativePage');
        if (!doc || !page) return false;
        const apply = () => page.classList.toggle('is-doc', !doc.classList.contains('hidden'));
        new MutationObserver(apply).observe(doc, { attributes: true, attributeFilter: ['class'] });
        apply();
        return true;
    };
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', () => { if (!start()) setTimeout(start, 0); });
    else if (!start()) setTimeout(start, 0);
})();
