/**
 * The modeling page as a narrative: a data panel and an app catalog on the left,
 * the run as a numbered sequence of steps in the middle.
 *
 * The layout is built from the page that is already there. The mode sections keep
 * their markup, their element ids, and their event listeners; this module wraps
 * them in a two-column grid and moves the existing nodes into it, so nothing that
 * reads a control by id is affected. The catalog is read from the model <select>
 * elements rather than from a second list, so it cannot drift from the registry.
 * Depends on core.js.
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
function narrativeActiveSelect() {
    const byTask = MODEL_SELECTS[narrativeCurrentMode()] || {};
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

function narrativeBuildLayout() {
    const host = document.getElementById('columnSelection');
    if (!host || document.getElementById('narrativeWrap')) return false;
    const sections = Array.from(host.querySelectorAll('.modeling-mode-section'));
    if (!sections.length) return false;

    const wrap = document.createElement('div');
    wrap.id = 'narrativeWrap';
    wrap.className = 'narrative-wrap';

    const rail = document.createElement('aside');
    rail.id = 'narrativeRail';
    rail.className = 'narrative-rail';
    rail.setAttribute('aria-label', 'Data and models');
    rail.innerHTML = `
        <section class="narrative-panel" id="narrativeDataPanel">
            <div class="narrative-panel-head">
                <h3>Data</h3>
                <span class="narrative-count" id="narrativeDataCount">0</span>
            </div>
            <div class="narrative-panel-body" id="narrativeDataBody">
                <p class="narrative-empty">No file is loaded.</p>
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
                <p class="narrative-empty">Choose a task type in Model Preprocessing.</p>
            </div>
        </section>
    `;

    const main = document.createElement('div');
    main.id = 'narrativeMain';
    main.className = 'narrative-main';

    const flow = document.createElement('nav');
    flow.id = 'narrativeFlow';
    flow.className = 'narrative-flow';
    flow.setAttribute('aria-label', 'Run progress');
    flow.innerHTML = `<ol>${RUN_STEPS.map(
        (s, i) =>
            `<li class="narrative-step" data-step="${s.key}"><span class="narrative-step-num">${i + 1}</span><span class="narrative-step-label">${s.label}</span></li>`
    ).join('')}</ol>`;

    sections[0].parentNode.insertBefore(wrap, sections[0]);
    wrap.appendChild(rail);
    wrap.appendChild(main);
    main.appendChild(flow);
    // The mode selector reads as the first decision of the run, so it moves with
    // the sections rather than staying above the new layout.
    const modeCard = host.querySelector('.preprocess-card');
    if (modeCard && modeCard.querySelector('#simpleMode')) main.appendChild(modeCard);
    sections.forEach((s) => main.appendChild(s));
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
    host.innerHTML = chips.length ? chips.join('') : '';
    host.hidden = !chips.length;
}

/** Fill the data panel from what the page already knows about the upload. */
function narrativeRefreshData() {
    const body = document.getElementById('narrativeDataBody');
    const count = document.getElementById('narrativeDataCount');
    if (!body) return;
    const name = typeof uploadedFileName === 'string' && uploadedFileName ? uploadedFileName : '';
    if (!name) {
        body.innerHTML = '<p class="narrative-empty">No file is loaded.</p>';
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
        body.innerHTML = '<p class="narrative-empty">Choose a task type in Model Preprocessing.</p>';
        if (count) count.textContent = '0';
        return;
    }
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
                        `<button type="button" class="narrative-item narrative-app${o.value === select.value ? ' is-selected' : ''}" data-model="${escapeHtml(o.value)}">` +
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

function narrativeWire() {
    const body = document.getElementById('narrativeAppsBody');
    if (body) {
        body.addEventListener('click', (e) => {
            const btn = e.target.closest('.narrative-app');
            if (!btn) return;
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

/** Build the layout the first time the modeling page is shown. */
function narrativeInit() {
    try {
        if (!narrativeBuildLayout()) return;
        narrativeWire();
        narrativeRefreshData();
        narrativeRefreshApps();
        narrativeSetStep('setup');
    } catch (err) {
        // A layout failure must not take the modeling page with it.
        console.error('Narrative layout not built:', err);
    }
}

// The modeling page is revealed by class changes rather than by navigation, so
// the build waits for the section to exist and runs once.
document.addEventListener('DOMContentLoaded', () => {
    const host = document.getElementById('columnSelection');
    if (!host) return;
    if (!host.classList.contains('hidden')) narrativeInit();
    const observer = new MutationObserver(() => {
        if (!host.classList.contains('hidden')) {
            narrativeInit();
            narrativeRefreshData();
        }
    });
    observer.observe(host, { attributes: true, attributeFilter: ['class'] });
});
