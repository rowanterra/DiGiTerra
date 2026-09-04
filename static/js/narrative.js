/**
 * The application as a narrative: one rail down the left carrying navigation, the
 * loaded data and the model catalog, with the views to its right.
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
 * rebuilt, so nothing that reads a control by id is affected. The catalog is read
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
// time; the catalog writes to whichever select belongs to the visible mode.
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
 * The model select the catalog reads.
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

/** Whether a task type has been chosen, which is what makes the catalog live. */
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
    rail.setAttribute('aria-label', 'Navigation, data and models');
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
        <section class="narrative-panel narrative-apps" id="narrativeAppsPanel">
            <div class="narrative-panel-head">
                <h3>Models</h3>
                <span class="narrative-count" id="narrativeAppsCount">0</span>
            </div>
            <div class="narrative-search">
                <label class="sr-only" for="narrativeAppSearch">Search models</label>
                <input type="search" id="narrativeAppSearch" placeholder="Search models" autocomplete="off">
            </div>
            <div class="narrative-panel-body" id="narrativeAppsBody">
                <p class="narrative-empty">No models listed.</p>
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
    // The page stacked the tab bar, this page header and the strip. The bar is gone
    // to the rail; the header moves inside the strip. The node is moved, so
    // #backToModelPreprocess and #modelingHeaderActions keep their ids and handlers.
    const pageHeader = host.querySelector(':scope > .section-header');
    if (pageHeader) flow.insertBefore(pageHeader, flow.firstChild);
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
    const note = document.querySelector('.modeling-mode-section:not(.hidden) .modeling-selection-note');
    const selection = note && note.textContent.trim() ? note.textContent.trim() : '';
    let shape = '';
    if (typeof uploadedShape === 'object' && uploadedShape && uploadedShape.rows) {
        shape = `Tabular, ${Number(uploadedShape.rows).toLocaleString()} \u00d7 ${uploadedShape.columns}`;
    }
    body.innerHTML = `
        <div class="narrative-item narrative-item-data">
            <span class="narrative-tile">CSV</span>
            <div>
                <strong>${escapeHtml(name)}</strong>
                ${shape ? `<p>${escapeHtml(shape)}</p>` : ''}
                ${selection ? `<p class="narrative-selection">${escapeHtml(selection)}</p>` : ''}
            </div>
        </div>
    `;
    if (count) count.textContent = '1';
}

/** Rebuild the catalog from the visible mode's model select. */
function narrativeRefreshApps() {
    const body = document.getElementById('narrativeAppsBody');
    const count = document.getElementById('narrativeAppsCount');
    if (!body) return;
    const select = narrativeActiveSelect();
    if (!select) {
        body.innerHTML = '<p class="narrative-empty">No models listed.</p>';
        if (count) count.textContent = '0';
        return;
    }
    // Before a task type is known the catalog still shows the registry and its
    // group counts, so the rail reads as the application rather than as a gap.
    // The entries are inert: choosing one would write to a select the run does
    // not use yet.
    const live = narrativeTaskChosen();
    const filter = (document.getElementById('narrativeAppSearch')?.value || '').trim().toLowerCase();
    let total = 0;
    let html = '';
    const groups = select.querySelectorAll('optgroup');
    const render = (label, options) => {
        const items = options.filter(
            (o) => o.value && (!filter || o.textContent.toLowerCase().includes(filter) || o.value.toLowerCase().includes(filter))
        );
        if (!items.length) return '';
        total += items.length;
        return (
            `<p class="narrative-group">${escapeHtml(label)} <span>${items.length}</span></p>` +
            items
                .map(
                    (o) =>
                        `<button type="button" class="narrative-item narrative-app${o.value === select.value && live ? ' is-selected' : ''}${live ? '' : ' is-inert'}"${live ? '' : ' disabled title="Choose a task type in Model Preprocessing."'} data-model="${escapeHtml(o.value)}">` +
                        `<span class="narrative-tile">${escapeHtml(narrativeInitials(o.textContent))}</span>` +
                        `<span class="narrative-app-name">${escapeHtml(o.textContent.trim())}</span></button>`
                )
                .join('')
        );
    };
    if (groups.length) {
        groups.forEach((g) => {
            html += render(g.label, Array.from(g.querySelectorAll('option')));
        });
    } else {
        html += render('Models', Array.from(select.querySelectorAll('option')));
    }
    body.innerHTML = html || '<p class="narrative-empty">No model matches that search.</p>';
    if (count) count.textContent = String(total);
}

/**
 * Show the rail once a file is loaded, and keep it from then on.
 *
 * With no data the rail would carry an empty data panel, an inert catalog and five
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
    narrativeRefreshData();
    narrativeRefreshApps();
}

function narrativeWire() {
    const body = document.getElementById('narrativeAppsBody');
    if (body) {
        body.addEventListener('click', (e) => {
            const btn = e.target.closest('.narrative-app');
            if (!btn || btn.disabled) return;
            const select = narrativeActiveSelect();
            if (!select) return;
            select.value = btn.dataset.model;
            select.dispatchEvent(new Event('change', { bubbles: true }));
            narrativeRefreshApps();
            narrativeSetStep('setup');
        });
    }
    const search = document.getElementById('narrativeAppSearch');
    if (search) search.addEventListener('input', narrativeRefreshApps);

    ['automlMode', 'simpleMode', 'advancedMode'].forEach((id) => {
        const radio = document.getElementById(id);
        if (radio) {
            radio.addEventListener('change', () => {
                narrativeRefreshApps();
                narrativeRefreshData();
            });
        }
    });
    Object.values(MODEL_SELECTS).forEach((byTask) => {
        Object.values(byTask).forEach((id) => {
            const el = document.getElementById(id);
            if (el) el.addEventListener('change', narrativeRefreshApps);
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
