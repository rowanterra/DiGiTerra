/* The guided page. One script, no framework.
 *
 * Five steps in one direction: upload, target, model, run, result. The Guided
 * and Full modes show the same steps; Full also shows every setting the
 * training route accepts. Each step is enabled when the one before it has
 * produced what it needs, and keyboard focus moves to the new step's heading.
 * Errors go to one alert region; progress goes to a polite live region.
 */
(function () {
  'use strict';
  const API = (window.DIGITERRA_API_ROOT || '').replace(/\/$/, '');
  const $ = (id) => document.getElementById(id);
  const state = { columns: [], types: [], target: null, predictors: [], task: null, model: null, models: null, hyper: {}, resultSession: null };

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
  function showError(msg) { const e = $('errors'); e.textContent = msg; e.classList.remove('hidden'); e.focus && e.focus(); }
  function clearError() { const e = $('errors'); e.textContent = ''; e.classList.add('hidden'); }
  // Focus moves only where the user's own action opened the next step: after an
  // upload (to step 2) and after a result (to step 5). The defaults that fill
  // steps 2 to 4 on their own do not pull focus down the page.
  function enableStep(n, focus) {
    const s = $('step' + n); s.setAttribute('aria-disabled', 'false');
    s.querySelectorAll('select, input, button').forEach((el) => { el.disabled = false; });
    if (focus) $('h' + n + 's').focus();
  }
  function markDone(n) { $('step' + n).classList.add('done'); }
  function esc(t) { return String(t).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c])); }

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
      markDone(1); enableStep(2, true);
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
    // Propose the last column as the target, which is where most files keep it.
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
    enableStep(3); markDone(2);
  });
  $('predictors').addEventListener('change', () => {
    state.predictors = Array.from($('predictors').selectedOptions).map((o) => Number(o.value)).filter((i) => i !== state.target);
  });
  document.querySelectorAll('input[name=task]').forEach((r) => r.addEventListener('change', () => { state.task = r.value; fillModels(); }));

  // ---- step 3: model and hyperparameters
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
    if (state.model) { enableStep(4); markDone(3); }
    try {
      const d = await getJSON('/model-defaults/' + encodeURIComponent(state.model));
      const tb = $('hyperTable').querySelector('tbody'); tb.innerHTML = '';
      if (!d.editable) { $('hyperNote').textContent = d.note || ''; $('hyperTable').classList.add('hidden'); return; }
      $('hyperNote').textContent = d.estimator + ' parameters, from scikit-learn. A blank value keeps the default.';
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
      });
      $('hyperTable').classList.remove('hidden');
    } catch (e) { $('hyperNote').textContent = 'Hyperparameters could not be read: ' + e.message; }
  });

  // ---- step 4: run
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
      const req = buildRequest();
      const j = await postJSON('/process', req);
      if (j.status === 'queued') $('progressText').textContent = j.message || 'Queued.';
      listen(j.session_id);
    } catch (e) { showError('The run did not start. ' + e.message); $('runBtn').disabled = false; }
  });
  function listen(sid) {
    const es = new EventSource(API + '/progress/' + sid);
    es.onmessage = (ev) => {
      let d;
      try { d = JSON.parse(ev.data); } catch (e) { es.close(); showError('The progress stream sent something that is not JSON: ' + e.message); $('runBtn').disabled = false; return; }
      if (d.error) { es.close(); showError(d.error); $('runBtn').disabled = false; return; }
      if (d.type === 'result') {
        es.close();
        try { showResult(d.data); } catch (e) { showError('The result could not be shown: ' + e.message); $('runBtn').disabled = false; }
        return;
      }
      const pct = Math.round(d.overall_progress || 0);
      $('progress').value = pct;
      const stage = d.current_stage || d.stage || '';
      const msg = d.message || (d.stages && stage && d.stages[stage] && d.stages[stage].message) || '';
      $('progressText').textContent = pct + '%' + (stage ? ', ' + stage.replace(/_/g, ' ') : '') + (msg ? ': ' + msg : '');
    };
    es.onerror = () => { /* EventSource reconnects on its own; the run continues on the server */ };
  }

  // ---- step 5: result
  const METRIC_LABELS = {
    valscore: 'Test R²', valrmse: 'Test RMSE', valmae: 'Test MAE', trainscore: 'Train R²', trainrmse: 'Train RMSE', trainmae: 'Train MAE',
    accuracy: 'Accuracy', precision: 'Precision (weighted)', recall: 'Recall (weighted)', f1score: 'F1 (weighted)',
    test_silhouette: 'Test silhouette', train_silhouette: 'Train silhouette', test_davies_bouldin: 'Test Davies-Bouldin', test_calinski_harabasz: 'Test Calinski-Harabasz',
  };
  function showResult(res) {
    $('runBtn').disabled = false; markDone(4); enableStep(5, true);
    const out = [];
    const rows = Object.keys(METRIC_LABELS).filter((k) => res[k] !== undefined && res[k] !== null && res[k] !== 'N/A');
    if (rows.length) {
      out.push('<table><caption>Metrics</caption><thead><tr><th scope="col">Metric</th><th scope="col" class="num">Value</th></tr></thead><tbody>');
      rows.forEach((k) => out.push('<tr><th scope="row">' + METRIC_LABELS[k] + '</th><td class="num">' + esc(res[k]) + '</td></tr>'));
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
    $('resultRaw').textContent = JSON.stringify(res, null, 1);
    $('predictFile').disabled = false; $('predictBtn').disabled = false;
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
})();
