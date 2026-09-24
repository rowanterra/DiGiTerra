/* The guided page. One script, no framework.
 *
 * Four steps, one on screen at a time, each filling the window: upload,
 * target, model and run, result. The Guided and Full modes show the same
 * steps; Full also shows every setting the training route accepts. A step
 * opens when the one before it has produced what it needs; the step buttons
 * in the header and the Back and Next buttons move between opened steps, and
 * focus lands on the shown step's heading. Every control has an explanation
 * that opens on hover or focus of its help button (and is read by a screen
 * reader through aria-describedby), dismissed with Escape. Errors go to one
 * alert region; progress goes to a polite live region.
 */
(function () {
  'use strict';
  const API = (window.DIGITERRA_API_ROOT || '').replace(/\/$/, '');
  const $ = (id) => document.getElementById(id);
  const state = { columns: [], types: [], target: null, predictors: [], task: null, model: null, models: null, hyper: {}, opened: 1, current: 1 };

  // ---- explanations, one per control (plain language, one or two sentences)
  const HELP = {
    modeGroup: 'Guided fills every setting with a default and shows only the four steps. Full opens every setting the training route accepts, in the same steps. The choice is remembered in this browser.',
    file: 'A table with one header row. CSV, Excel (.xlsx, .xlsm) and Parquet are read. Nothing is sent anywhere except this server.',
    target: 'The column the model predicts. A text column or a number with few distinct values is treated as a class; other numbers as a quantity to regress.',
    predictors: 'The columns the model reads. All columns except the target are used unless changed here. Identifier columns and anything that leaks the target should be removed.',
    task: 'Classification predicts a class label. Regression predicts a number. Clustering groups rows without a target. The page proposes one from the target column; change it if the proposal is wrong.',
    model: 'The estimator to fit, from scikit-learn. The Guided default is a random forest, which needs little tuning. Every model in the list is trained through the same route.',
    hyperTable: 'The parameters of the chosen estimator, read from scikit-learn itself. A blank keeps the default shown. A value typed here reaches the estimator and is recorded in the run manifest.',
    splitMethod: 'How rows are divided into training and test halves. Random is the usual choice. Time-ordered keeps the latest rows for the test. Grouped keeps every row of a group on one side, for data where several rows come from one site or sample.',
    splitColumn: 'The column a time-ordered or grouped split reads. Not used by the random split.',
    testSize: 'The fraction of rows held out as the test set. 0.20 keeps one row in five for testing.',
    stratify: 'Keeps the class mix of the chosen column the same in both halves of a random split. Off unless a column is chosen.',
    scaler: 'Rescales the predictors before fitting. Trees do not need it; distance and linear models often do. Standard centres to mean zero and unit variance; min-max maps to 0 to 1; robust uses the median and quartiles; quantile maps to a uniform distribution.',
    dropMissing: 'What to do with rows that have a blank in a modeling column: remove them, or fill the blank with the column mean, median or mode computed on the training rows only.',
    zeroHandle: 'What to do with zeros in the modeling columns, for data where zero means "not measured": keep them, remove the rows, or fill them with the mean or median.',
    seed: 'The random seed. The same seed and settings give the same split and the same fit. It is written to the run record.',
    sigfig: 'Decimal places shown for the metrics. The predictions file is always written at full precision.',
    featureSelection: 'Keeps a subset of predictors before fitting: the k with the strongest univariate score, recursive elimination, or the ones a model weights. The chosen columns are named in the result.',
    featureK: 'How many predictors to keep when feature selection is on.',
    outlierMethod: 'Finds and removes outlying training rows before fitting, by interquartile range, z-score, an isolation forest or the local outlier factor. The count removed is reported.',
    cvType: 'Cross-validation on the training rows, reported beside the single test score. Stratified K-fold keeps the class mix in every fold.',
    cvFolds: 'How many folds the cross-validation uses.',
    search: 'Searches over hyperparameter values with cross-validation and keeps the best set. Randomized tries a sample of combinations; grid tries all of them. The chosen values are written to the run record.',
    runBtn: 'Fits the model with the settings on this page. A blocking data check stops the run and says why.',
    result: 'The metrics on the held-out test rows, the baseline on the same rows, the data checks, and the parameters as fitted. The bundle link downloads the run record and the model.',
    predictFile: 'A table with the same predictor columns as the training file. Predictions are written with a novelty score and a note for rows outside the training range.',
    stepsNav: 'The four steps. A step opens when the one before it is done; any opened step can be revisited.',
  };
  const MODEL_HELP = {
    classification: 'Predicts a class label from the predictors.',
    regression: 'Predicts a number from the predictors.',
    cluster: 'Groups rows by similarity; the number of groups is a setting of the model.',
  };
  const PARAM_HELP = {
    n_estimators: 'Number of trees or rounds. More is slower and usually a little better, with diminishing returns.',
    max_depth: 'Deepest a tree may grow. Smaller values fit less closely to the training rows.',
    n_neighbors: 'How many nearest rows vote for a prediction. Small values follow the training rows closely.',
    weights: 'uniform gives every neighbour one vote; distance gives nearer neighbours more.',
    learning_rate: 'How much each boosting round changes the model. Smaller needs more rounds.',
    C: 'Inverse regularization strength. Smaller values regularize more.',
    alpha: 'Regularization strength. Larger values regularize more.',
    max_iter: 'Cap on solver iterations.',
    tol: 'Stopping tolerance for the solver.',
    random_state: 'Seed for this estimator. The run seed is used when blank.',
    class_weight: 'Reweights classes; balanced counters an uneven class mix.',
    criterion: 'The split quality measure the trees use.',
    min_samples_leaf: 'Fewest rows a leaf may hold; larger values smooth the tree.',
    min_samples_split: 'Fewest rows a node needs before it can split.',
    kernel: 'The kernel function of the support vector machine.',
    gamma: 'Kernel width for rbf, poly and sigmoid kernels.',
    solver: 'The optimization routine; the default suits most data.',
    penalty: 'The regularization type.',
    n_jobs: 'CPU cores to use; blank means one.',
    p: 'Power of the Minkowski distance: 1 is Manhattan, 2 is Euclidean.',
    metric: 'The distance measure between rows.',
  };

  // ---- helpers
  function csrfHeaders() {
    const m = document.cookie.match(/(?:^|;\s*)digiterra_csrf=([^;]*)/);
    return m ? { 'X-DiGiTerra-CSRF': decodeURIComponent(m[1]) } : {};
  }
  async function postJSON(path, body) {
    const r = await fetch(API + path, { method: 'POST', headers: Object.assign({ 'Content-Type': 'application/json' }, csrfHeaders()), body: JSON.stringify(body) });
    const j = await r.json().catch(() => ({ error: 'The server answered with something that is not JSON.' }));
    if (!r.ok || j.error) throw new Error(j.error || ('HTTP ' + r.status));
    return j;
  }
  async function getJSON(path) {
    const r = await fetch(API + path);
    const j = await r.json();
    if (!r.ok || j.error) throw new Error(j.error || ('HTTP ' + r.status));
    return j;
  }
  function showError(msg) { const e = $('errors'); e.textContent = msg; e.classList.remove('hidden'); e.focus(); }
  function clearError() { const e = $('errors'); e.textContent = ''; e.classList.add('hidden'); }
  function esc(t) { return String(t).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c])); }

  // ---- help buttons and tooltips (WCAG 1.4.13: hoverable, dismissible with Escape, persistent)
  let tipCounter = 0;
  function attachHelp(target, text, labelText) {
    if (!target || !text) return null;
    const id = 'tip' + (++tipCounter);
    const btn = document.createElement('button');
    btn.type = 'button'; btn.className = 'help'; btn.textContent = '?';
    btn.setAttribute('aria-label', 'What is ' + labelText + '?');
    btn.setAttribute('aria-describedby', id);
    const tip = document.createElement('span');
    tip.className = 'tip'; tip.id = id; tip.setAttribute('role', 'tooltip'); tip.textContent = text;
    const wrap = document.createElement('span'); wrap.className = 'helpwrap';
    wrap.appendChild(btn); wrap.appendChild(tip);
    // A label's text is the control's accessible name, so the help button goes
    // after a label, never inside it; other targets (legends, headings, table
    // header cells) take it as their last child.
    if (target.tagName === 'LABEL') target.insertAdjacentElement('afterend', wrap); else target.appendChild(wrap);
    btn.addEventListener('click', (ev) => { ev.preventDefault(); wrap.classList.toggle('open'); });
    return id;
  }
  document.addEventListener('keydown', (ev) => { if (ev.key === 'Escape') document.querySelectorAll('.helpwrap.open').forEach((w) => w.classList.remove('open')); });
  function helpForControl(id) {
    const ctl = $(id); if (!ctl) return;
    const label = document.querySelector('label[for="' + id + '"]') || (ctl.tagName === 'BUTTON' ? ctl : null);
    const tipId = attachHelp(label || ctl.parentElement, HELP[id], label ? label.textContent.replace('?', '').trim() : id);
    if (tipId && ctl.tagName !== 'BUTTON') ctl.setAttribute('aria-describedby', tipId);
  }
  ['file', 'target', 'predictors', 'model', 'splitMethod', 'splitColumn', 'testSize', 'stratify', 'scaler', 'dropMissing', 'zeroHandle',
    'seed', 'sigfig', 'featureSelection', 'featureK', 'outlierMethod', 'cvType', 'cvFolds', 'search', 'runBtn', 'predictFile'].forEach(helpForControl);
  attachHelp(document.querySelector('#modeGroup'), HELP.modeGroup, 'the mode');
  attachHelp(document.querySelector('#step3 fieldset legend'), HELP.task, 'the task');
  attachHelp(document.querySelector('#hyperBox h3'), HELP.hyperTable, 'the hyperparameter table');
  attachHelp(document.querySelector('#step4 h2'), HELP.result, 'the result');
  attachHelp(document.querySelector('nav.steps'), HELP.stepsNav, 'the step list');

  // ---- steps
  function showStep(n, focus) {
    state.current = n;
    document.querySelectorAll('section.step').forEach((s) => s.classList.toggle('current', s.id === 'step' + n));
    document.querySelectorAll('nav.steps button').forEach((b) => {
      const k = Number(b.dataset.step);
      b.disabled = k > state.opened;
      if (k === n) b.setAttribute('aria-current', 'step'); else b.removeAttribute('aria-current');
      b.classList.toggle('done', k < state.opened);
    });
    clearError();
    if (focus) $('h' + n + 's').focus();
  }
  function openStep(n) {
    if (n > state.opened) state.opened = n;
    document.querySelectorAll('button.next[data-go]').forEach((b) => { b.disabled = Number(b.dataset.go) > state.opened; });
    document.querySelectorAll('nav.steps button').forEach((b) => { b.disabled = Number(b.dataset.step) > state.opened; });
  }
  document.querySelectorAll('nav.steps button, button.back, button.next').forEach((b) => {
    b.addEventListener('click', () => {
      const n = Number(b.dataset.step || b.dataset.go);
      if (b.classList.contains('back') && b.dataset.go === '1' && b.textContent.startsWith('Start over')) {
        state.opened = 1; openStep(1); $('file').value = ''; $('uploadStatus').textContent = '';
      }
      if (n <= state.opened) showStep(n, true);
    });
  });

  // ---- mode
  function setMode(mode) {
    document.body.dataset.mode = mode;
    try { localStorage.setItem('digiterra.guided.mode', mode); } catch (e) { /* storage may be unavailable */ }
    document.querySelectorAll('input[name=mode]').forEach((r) => { r.checked = r.value === mode; });
  }
  document.querySelectorAll('input[name=mode]').forEach((r) => r.addEventListener('change', () => setMode(r.value)));
  try { const saved = localStorage.getItem('digiterra.guided.mode'); if (saved === 'full' || saved === 'guided') setMode(saved); } catch (e) { /* ignore */ }

  // ---- step 1: upload
  $('uploadForm').addEventListener('submit', async (ev) => {
    ev.preventDefault(); clearError();
    const f = $('file').files[0];
    if (!f) { showError('Choose a file first.'); return; }
    $('uploadStatus').textContent = 'Uploading ' + f.name + '.';
    const fd = new FormData(); fd.append('file', f);
    try {
      const r = await fetch(API + '/upload', { method: 'POST', headers: csrfHeaders(), body: fd });
      const j = await r.json();
      if (!r.ok || j.error) throw new Error(j.error || ('HTTP ' + r.status));
      state.columns = j.columns;
      const t = await getJSON('/column-types');
      state.types = t.columns;
      $('uploadStatus').textContent = j.filename + ': ' + j.rows + ' rows, ' + j.numcols + ' columns.' + (j.warnings ? ' ' + j.warnings.join(' ') : '');
      fillColumns();
      openStep(3);   // target and model are filled with proposals, so both open at once
      showStep(2, true);
    } catch (e) { showError('Upload failed. ' + e.message); }
  });

  function fillColumns() {
    const target = $('target'); target.innerHTML = '';
    const pred = $('predictors'); pred.innerHTML = '';
    const split = $('splitColumn'); split.innerHTML = '<option value="">None</option>';
    const strat = $('stratify'); strat.innerHTML = '<option value="">No</option>';
    state.types.forEach((c) => {
      const o = document.createElement('option'); o.value = c.index; o.textContent = c.name + (c.numeric ? '' : ' (text)'); target.appendChild(o);
      const p = document.createElement('option'); p.value = c.index; p.textContent = c.name; p.selected = true; pred.appendChild(p);
      const s = document.createElement('option'); s.value = c.index; s.textContent = c.name; split.appendChild(s);
      const st = document.createElement('option'); st.value = c.index; st.textContent = c.name; strat.appendChild(st);
    });
    // The last column is proposed as the target, which is where most files keep it.
    target.value = String(state.types.length - 1);
    target.dispatchEvent(new Event('change'));
  }

  // ---- step 2: target and predictors
  $('target').addEventListener('change', () => {
    const idx = Number($('target').value);
    const c = state.types[idx];
    state.target = idx;
    state.predictors = state.types.filter((x) => x.index !== idx).map((x) => x.index);
    Array.from($('predictors').options).forEach((o) => { o.selected = Number(o.value) !== idx; o.disabled = Number(o.value) === idx; });
    const task = c.looks_categorical ? 'classification' : 'regression';
    $('targetHint').textContent = c.name + ' has ' + c.distinct + ' distinct values' + (c.missing ? ', ' + c.missing + ' missing' : '') + '. Proposed task: ' + task + '.';
    document.querySelectorAll('input[name=task]').forEach((r) => { r.checked = r.value === task; });
    state.task = task;
    ensureModels().then(fillModels);
  });
  $('predictors').addEventListener('change', () => {
    state.predictors = Array.from($('predictors').selectedOptions).map((o) => Number(o.value)).filter((i) => i !== state.target);
  });
  document.querySelectorAll('input[name=task]').forEach((r) => r.addEventListener('change', () => { state.task = r.value; fillModels(); }));

  // ---- step 3: model, hyperparameters, run
  async function ensureModels() { if (!state.models) state.models = await getJSON('/models'); }
  const GUIDED_DEFAULT = { classification: 'RF_classifier', regression: 'RF', cluster: 'kmeans' };
  function fillModels() {
    const sel = $('model'); sel.innerHTML = '';
    const list = (state.models && state.models[state.task]) || [];
    list.forEach((m) => { const o = document.createElement('option'); o.value = m.name; o.textContent = m.name.replace(/_classifier$/, '').replace(/_/g, ' '); sel.appendChild(o); });
    const wanted = GUIDED_DEFAULT[state.task];
    if (list.some((m) => m.name === wanted)) sel.value = wanted;
    sel.dispatchEvent(new Event('change'));
  }
  $('model').addEventListener('change', async () => {
    state.model = $('model').value; state.hyper = {};
    $('runBtn').disabled = !state.model;
    try {
      const d = await getJSON('/model-defaults/' + encodeURIComponent(state.model));
      const tb = $('hyperTable').querySelector('tbody'); tb.innerHTML = '';
      const family = MODEL_HELP[state.task] || '';
      if (!d.editable) { $('hyperNote').textContent = (d.note || '') + ' ' + family; $('hyperTable').classList.add('hidden'); return; }
      $('hyperNote').textContent = d.estimator + ' parameters, from scikit-learn. A blank value keeps the default. ' + family;
      d.params.forEach((p) => {
        const tr = document.createElement('tr');
        const id = 'hp_' + p.name;
        tr.innerHTML = '<th scope="row"><label for="' + id + '">' + esc(p.name) + '</label></th><td>' + esc(p.default === null ? 'None' : p.default) + '</td><td></td>';
        let input;
        if (p.kind === 'bool') { input = document.createElement('select'); ['', 'true', 'false'].forEach((v) => { const o = document.createElement('option'); o.value = v; o.textContent = v || '(default)'; input.appendChild(o); }); }
        else { input = document.createElement('input'); input.type = p.kind === 'int' || p.kind === 'float' ? 'number' : 'text'; if (p.kind === 'float') input.step = 'any'; }
        input.id = id; input.addEventListener('change', () => {
          const v = input.value;
          if (v === '') { delete state.hyper[p.name]; return; }
          state.hyper[p.name] = p.kind === 'bool' ? (v === 'true') : (p.kind === 'int' ? parseInt(v, 10) : (p.kind === 'float' ? parseFloat(v) : v));
        });
        tr.lastElementChild.appendChild(input); tb.appendChild(tr);
        const text = PARAM_HELP[p.name] || ('Parameter ' + p.name + ' of ' + d.estimator + '; the scikit-learn default is shown.');
        const tipId = attachHelp(tr.firstElementChild, text, p.name);
        input.setAttribute('aria-describedby', tipId);
      });
      $('hyperTable').classList.remove('hidden');
    } catch (e) { $('hyperNote').textContent = 'Hyperparameters could not be read: ' + e.message; }
  });

  function buildRequest() {
    const full = document.body.dataset.mode === 'full';
    const missing = full ? $('dropMissing').value : 'none';
    const zeros = full ? $('zeroHandle').value : 'none';
    const req = {
      indicators: state.predictors, predictors: state.task === 'cluster' ? [] : [state.target], models: state.model,
      hyperparameters: full ? state.hyper : {}, nonreq: full && Object.keys(state.hyper).length > 0,
      scaler: full ? $('scaler').value : 'none', units: '', sigfig: full ? Number($('sigfig').value) : 3,
      seedValue: full ? Number($('seed').value) : 42, testSize: full ? String($('testSize').value) : '0.20',
      stratifyBool: full && $('stratify').value !== '' && Number($('stratify').value) !== state.target,
      dropMissing: 'indicatorAndTarget', imputeStrategy: missing === 'none' ? 'none' : missing,
      dropZero: zeros === 'none' ? 'none' : 'indicatorAndTarget', zeroHandle: zeros === 'none' ? 'none' : (zeros === 'drop' ? 'drop' : zeros),
      quantileBinDict: { quantile: 0, bin: 0, binsLabel: '' }, useTransformer: 'No', transformerCols: [],
      filename: ($('file').files[0] || {}).name || 'upload', modelingMode: full ? 'advanced' : 'simple',
      crossValidationType: full ? $('cvType').value : 'None', crossValidationFolds: full ? Number($('cvFolds').value) : 5,
      hyperparameterSearch: full ? $('search').value : 'none',
      featureSelectionMethod: full ? $('featureSelection').value : 'none', featureSelectionK: full ? Number($('featureK').value) : null,
      outlierMethod: full ? $('outlierMethod').value : 'none', outlierAction: 'remove',
      splitMethod: full ? $('splitMethod').value : 'random', splitColumn: full && $('splitColumn').value !== '' ? Number($('splitColumn').value) : '',
    };
    if (req.stratifyBool) req.stratifyColumn = Number($('stratify').value);
    return req;
  }
  $('runBtn').addEventListener('click', async () => {
    clearError();
    if (!state.model) { showError('Choose a model first.'); return; }
    $('runBtn').disabled = true;
    $('progressBox').classList.remove('hidden'); $('progress').value = 0; $('progressText').textContent = 'Starting.';
    try {
      // The preprocess route wants a stratify column that is not the target; the
      // first predictor stands in, and the run only stratifies when asked to.
      await postJSON('/preprocess', { indicators: state.predictors, predictors: state.task === 'cluster' ? [] : [state.target], stratify: state.predictors[0] });
      const j = await postJSON('/process', buildRequest());
      if (j.status === 'queued') $('progressText').textContent = j.message || 'Queued.';
      listen(j.session_id);
    } catch (e) { showError('The run did not start. ' + e.message); $('runBtn').disabled = false; }
  });
  function listen(sid) {
    const es = new EventSource(API + '/progress/' + sid);
    let finished = false;
    es.onmessage = (ev) => {
      let d;
      try { d = JSON.parse(ev.data); } catch (e) { es.close(); showError('The progress stream sent something that is not JSON: ' + e.message); $('runBtn').disabled = false; return; }
      if (d.error) { finished = true; es.close(); showError(d.error); $('runBtn').disabled = false; return; }
      if (d.type === 'result') {
        finished = true; es.close();
        try { showResult(d.data); } catch (e) { showError('The result could not be shown: ' + e.message); $('runBtn').disabled = false; }
        return;
      }
      const pct = Math.round(d.overall_progress || 0);
      $('progress').value = pct;
      const stage = d.current_stage || d.stage || '';
      const msg = d.message || (d.stages && stage && d.stages[stage] && d.stages[stage].message) || '';
      $('progressText').textContent = pct + '%' + (stage ? ', ' + stage.replace(/_/g, ' ') : '') + (msg ? ': ' + msg : '');
    };
    es.onerror = () => { if (finished) es.close(); /* otherwise EventSource reconnects on its own; the run continues on the server */ };
  }

  // ---- step 4: result
  const METRIC_LABELS = {
    valscore: 'Test R²', valrmse: 'Test RMSE', valmae: 'Test MAE', trainscore: 'Train R²', trainrmse: 'Train RMSE', trainmae: 'Train MAE',
    accuracy: 'Accuracy', precision: 'Precision (weighted)', recall: 'Recall (weighted)', f1score: 'F1 (weighted)',
    test_silhouette: 'Test silhouette', train_silhouette: 'Train silhouette', test_davies_bouldin: 'Test Davies-Bouldin', test_calinski_harabasz: 'Test Calinski-Harabasz',
  };
  const METRIC_HELP = {
    valscore: 'R² on the test rows: 1 is a perfect fit, 0 is no better than the mean, negative is worse than the mean.',
    valrmse: 'Root mean squared error on the test rows, in the units of the target.',
    valmae: 'Mean absolute error on the test rows, in the units of the target.',
    trainscore: 'R² on the training rows. Much higher than the test value means the model fits the training rows too closely.',
    trainrmse: 'Root mean squared error on the training rows.', trainmae: 'Mean absolute error on the training rows.',
    accuracy: 'Share of test rows whose class was predicted correctly.',
    precision: 'Of the rows predicted as each class, the share that were that class, averaged by class size.',
    recall: 'Of the rows of each class, the share that were found, averaged by class size.',
    f1score: 'The harmonic mean of precision and recall, averaged by class size.',
    test_silhouette: 'How well separated the clusters are, from -1 to 1; higher is better.',
    train_silhouette: 'Silhouette on the training rows.',
    test_davies_bouldin: 'Ratio of within-cluster spread to between-cluster distance; lower is better.',
    test_calinski_harabasz: 'Ratio of between-cluster to within-cluster dispersion; higher is better.',
  };
  function showResult(res) {
    $('runBtn').disabled = false;
    const out = [];
    const rows = Object.keys(METRIC_LABELS).filter((k) => res[k] !== undefined && res[k] !== null && res[k] !== 'N/A');
    if (rows.length) {
      out.push('<table><caption>Metrics</caption><thead><tr><th scope="col">Metric</th><th scope="col" class="num">Value</th></tr></thead><tbody>');
      rows.forEach((k) => out.push('<tr><th scope="row" data-metric="' + k + '">' + METRIC_LABELS[k] + '</th><td class="num">' + esc(res[k]) + '</td></tr>'));
      out.push('</tbody></table>');
    }
    if (res.train_n !== undefined) out.push('<p>' + esc(res.train_n) + ' training rows, ' + esc(res.test_n) + ' test rows.' + (res.split_note ? ' ' + esc(res.split_note) : '') + '</p>');
    if (res.baseline) {
      const b = res.baseline;
      out.push('<p>Baseline (' + esc(b.description || 'baseline') + '): ' + esc(b.metric || '') + ' ' + esc(b.baseline_value) + ' against the model’s ' + esc(b.model_value) + '. ' + (b.model_beats_baseline ? 'The model beat the baseline.' : 'The model did not beat the baseline.') + '</p>');
    }
    const dc = res.data_checks;
    if (dc && ((dc.blocking && dc.blocking.length) || (dc.warnings && dc.warnings.length))) {
      out.push('<h3>Data checks</h3><ul class="checks">');
      (dc.blocking || []).forEach((c) => out.push('<li><span class="block-sev">blocking</span>: ' + esc(c.message) + '</li>'));
      (dc.warnings || []).forEach((c) => out.push('<li><span class="warn-sev">warning</span>: ' + esc(c.message) + '</li>'));
      out.push('</ul>');
    } else if (dc) {
      out.push('<p>Data checks: no findings on ' + esc(dc.n_rows) + ' rows and ' + esc(dc.n_features) + ' predictors.</p>');
    }
    if (res.model_params) {
      out.push('<details><summary>Estimator parameters as fitted</summary><table><thead><tr><th scope="col">Parameter</th><th scope="col">Value</th></tr></thead><tbody>');
      Object.keys(res.model_params).forEach((k) => out.push('<tr><th scope="row">' + esc(k) + '</th><td>' + esc(res.model_params[k] === null ? 'None' : res.model_params[k]) + '</td></tr>'));
      out.push('</tbody></table></details>');
    }
    out.push('<p><a href="' + API + '/export-bundle">Download the run record and model (bundle)</a>' + (res.model_card_url ? ' &middot; <a href="' + esc(res.model_card_url) + '">Model card</a>' : '') + '</p>');
    $('result').innerHTML = out.join('');
    $('result').querySelectorAll('th[data-metric]').forEach((th) => attachHelp(th, METRIC_HELP[th.dataset.metric], th.textContent.replace('?', '').trim()));
    $('resultRaw').textContent = JSON.stringify(res, null, 1);
    $('predictFile').disabled = false; $('predictBtn').disabled = false;
    openStep(4); showStep(4, true);
  }

  // ---- predict
  $('predictForm').addEventListener('submit', async (ev) => {
    ev.preventDefault(); clearError();
    const f = $('predictFile').files[0];
    if (!f) { showError('Choose a file to predict on.'); return; }
    const fd = new FormData(); fd.append('predictFile', f); fd.append('preprocessMode', 'none');
    $('predictStatus').textContent = 'Predicting.';
    try {
      const r = await fetch(API + '/predict', { method: 'POST', headers: csrfHeaders(), body: fd });
      const j = await r.json();
      if (!r.ok || j.error) throw new Error(j.error || ('HTTP ' + r.status));
      const a = j.applicability_summary || {};
      $('predictStatus').innerHTML = 'Predictions written for ' + esc(a.rows !== undefined ? a.rows : '') + ' rows' + (a.out_of_range ? ', ' + esc(a.out_of_range) + ' outside the training range' : '') + '. <a href="' + API + '/download/predictions.csv">Download predictions.csv</a>';
    } catch (e) { showError('Prediction failed. ' + e.message); $('predictStatus').textContent = ''; }
  });

  showStep(1, false);
})();
