/**
 * Inference tab: prediction form, AutoML/advanced forms. Depends on core.js.
 */
predictionForm.addEventListener('submit', async (e) => {
    e.preventDefault()
    let predictionErrorDiv = document.getElementById('predictionErrorDiv')
    let uploadPredictDf = document.getElementById('uploadPredictDf')
    let predictionResults = document.getElementById('predictionResults')

    //const predictFile = document.getElementById('predictFile').files[0];
    const formData = new FormData(predictionForm);
    formData.append('indicators', 'indicators')
    const resultTimestamp = formatDateTimeForFilename() 
    const predictionDownloadName = `predictions_${resultTimestamp}.csv`

    try{
        const response = await fetch(withApiRoot('/predict'), {
            method: 'POST',
            body: formData,
        });
        let data = await response.json();
        if (response.ok === true) {
            predictionErrorDiv.classList.add('hidden')
            predictionResults.classList.remove('hidden')
            const summary = data.summary || [];
            const preview = data.predictions_preview || {};
            const trainingViz = data.training_visualization;
            const trainingVizVersion = data.training_visualization_version || Date.now();
            const inferenceViz = data.inference_visualization;
            const inferenceVizVersion = data.inference_visualization_version || Date.now();
            let trainingVizUrl = '';
            let trainingPerfUrl = '';
            let trainingPlotWithOverlayUrl = ''; // for regression: pred-vs-actual only (no composite) so overlay aligns
            let inferenceVizUrl = ''; // server-regenerated plot with train + test + inference points (no overlay)
            if (trainingViz) {
                const base = withApiRoot('/user-visualizations/' + trainingViz);
                const cacheQuery = '?v=' + encodeURIComponent(trainingVizVersion) + '&t=' + Date.now();
                trainingVizUrl = base + cacheQuery;
                let predActualName = trainingViz;
                const modelTypeForPerf = data.model_type || 'regression';
                if (modelTypeForPerf === 'regression') {
                    predActualName = predActualName
                        .replace('target_plot_1_advanced', 'target_plot_pred_actual_1_advanced')
                        .replace('target_plot_1', 'target_plot_pred_actual_1');
                    trainingPlotWithOverlayUrl = withApiRoot('/user-visualizations/' + predActualName) + cacheQuery;
                }
                trainingPerfUrl = withApiRoot('/user-visualizations/' + predActualName) + cacheQuery;
            }
            if (inferenceViz) {
                inferenceVizUrl = withApiRoot('/user-visualizations/' + inferenceViz) + '?v=' + encodeURIComponent(inferenceVizVersion) + '&t=' + Date.now();
            }
            const modelType = data.model_type || 'regression';
            const modelTypeLabel = modelType === 'classification' ? 'Classification' : modelType === 'cluster' ? 'Clustering' : 'Regression';

            let summaryTableHtml = '';
            if (summary.length > 0) {
                const numericKeys = ['column', 'n', 'min', 'max', 'mean', 'std', '25', '50', '75', '100'];
                const hasNumeric = summary.some(row => row.mean != null || row.min != null);
                if (hasNumeric) {
                    summaryTableHtml = `
                        <table class="stats-table model-stats-table">
                            <thead><tr>${numericKeys.map(k => `<th>${k === '25' ? '25%' : k === '50' ? '50%' : k === '75' ? '75%' : k === '100' ? '100%' : k}</th>`).join('')}</tr></thead>
                            <tbody>
                                ${summary.filter(r => r.mean != null || r.min != null).map(row => `
                                    <tr>${numericKeys.map(k => `<td>${row[k] != null ? (typeof row[k] === 'number' ? Number(row[k]).toFixed(4) : escapeHtml(String(row[k]))) : ''}</td>`).join('')}</tr>
                                `).join('')}
                            </tbody>
                        </table>`;
                } else {
                    summaryTableHtml = summary.map(row => `
                        <div style="margin-bottom: 12px;">
                            <strong>${escapeHtml(row.column)}</strong> (n=${row.n})
                            ${row.value_counts && row.value_counts.length ? `
                                <table class="stats-table model-stats-table" style="margin-top: 6px;">
                                    <tr><th>Value</th><th>Count</th></tr>
                                    ${row.value_counts.map(vc => `<tr><td>${escapeHtml(vc.value)}</td><td>${vc.count}</td></tr>`).join('')}
                                </table>
                            ` : ''}
                        </div>
                    `).join('');
                }
            } else {
                summaryTableHtml = '<p>No inference summary available.</p>';
            }

            let previewChartHtml = '';
            const firstCol = summary.length ? summary[0].column : null;
            const trainingSummaryForAxes = data.training_target_summary && data.training_target_summary.summary;
            const axisMin = (trainingSummaryForAxes && trainingSummaryForAxes[0] && (trainingSummaryForAxes[0].min != null || trainingSummaryForAxes[0]['25'] != null))
                ? (Number(trainingSummaryForAxes[0].min) ?? Number(trainingSummaryForAxes[0]['25']))
                : null;
            const axisMax = (trainingSummaryForAxes && trainingSummaryForAxes[0] && (trainingSummaryForAxes[0].max != null || trainingSummaryForAxes[0]['100'] != null))
                ? (Number(trainingSummaryForAxes[0].max) ?? Number(trainingSummaryForAxes[0]['100']))
                : null;
            if (firstCol && preview[firstCol] && preview[firstCol].length > 0 && typeof preview[firstCol][0] === 'number') {
                const vals = preview[firstCol];
                const min = Math.min(...vals);
                const max = Math.max(...vals);
                const lo = (axisMin != null && axisMax != null) ? Math.min(axisMin, min) : min;
                const hi = (axisMax != null && axisMin != null) ? Math.max(axisMax, max) : max;
                const range = hi - lo || 1;
                const bins = 10;
                const step = (max - min) / bins || 1;
                const counts = Array(bins).fill(0);
                vals.forEach(v => {
                    let i = Math.min(Math.floor((v - min) / step), bins - 1);
                    if (i < 0) i = 0;
                    counts[i]++;
                });
                const maxCount = Math.max(...counts, 1);
                const histPad = { left: 42, right: 16, top: 12, bottom: 36 };
                const histW = 280;
                const histH = 200;
                const histPlotW = histW - histPad.left - histPad.right;
                const histPlotH = histH - histPad.top - histPad.bottom;
                const histBarsSvg = counts.map((c, i) => {
                    const barH = maxCount > 0 ? (c / maxCount) * histPlotH : 0;
                    const x = histPad.left + (i / bins) * histPlotW;
                    const bw = Math.max(2, histPlotW / bins - 2);
                    const y = histPad.top + histPlotH - barH;
                    return `<rect x="${x}" y="${y}" width="${bw}" height="${barH}" fill="#7f8c9a" rx="2"/>`;
                }).join('');
                const histYTicks = [0, Math.round(maxCount / 2), maxCount].filter((v, i, a) => a.indexOf(v) === i);
                const histYTickLines = histYTicks.map(t => {
                    const y = histPad.top + histPlotH - (t / maxCount) * histPlotH;
                    return `<line x1="${histPad.left}" y1="${y}" x2="${histPad.left - 5}" y2="${y}" stroke="#444" stroke-width="1"/>`;
                }).join('');
                const xTickCount = 5;
                const xTickValues = [];
                for (let i = 0; i < xTickCount; i++) {
                    const v = min + (max - min) * (i / (xTickCount - 1));
                    xTickValues.push(v);
                }
                const xTickLines = xTickValues.map((v) => {
                    const t = (v - min) / (max - min || 1);
                    const x = histPad.left + t * histPlotW;
                    return `<line x1="${x}" y1="${histPad.top + histPlotH}" x2="${x}" y2="${histPad.top + histPlotH + 5}" stroke="#444" stroke-width="1"/>`;
                }).join('');
                const xTickTexts = xTickValues.map((v) => {
                    const t = (v - min) / (max - min || 1);
                    const x = histPad.left + t * histPlotW;
                    const label = (v === Math.floor(v) ? v : Number(v).toFixed(1)).toString();
                    return `<text x="${x}" y="${histH - 10}" text-anchor="middle" font-size="10" fill="#444">${label}</text>`;
                }).join('');
                const leftPlotHtml = (trainingViz || inferenceVizUrl) ? (() => {
                    // Prefer server-regenerated plot (train + test + inference) when present; else training image + overlay.
                    if (inferenceVizUrl) {
                        return `
                        <div style="position: relative; display: inline-block; max-width: 100%;">
                            <img src="${inferenceVizUrl}" alt="Predicted vs Actual (train, test, inference)" class="inference-model-graphic-img inference-training-plot-img" style="display: block; max-height: 300px; width: auto;">
                        </div>
                        <p style="margin: 4px 0 0 0; font-size: 0.8rem; color: #555;">Pink points = inference (predicted values only; we have no actuals for new data). Distribution to the right.</p>`;
                    }
                    const imgUrl = trainingPlotWithOverlayUrl || trainingVizUrl;
                    const toNormX = v => 10 + ((v - lo) / range) * 80;
                    const toNormY = v => 85 - ((v - lo) / range) * 75;
                    const overlayPoints = vals.map(v => {
                        const cx = toNormX(v);
                        const cy = toNormY(v);
                        return `<circle cx="${cx}" cy="${cy}" r="0.38" fill="#5a7a9a" opacity="0.5"/>`;
                    }).join('');
                    return `
                        <div style="position: relative; display: inline-block; max-width: 100%;">
                            <img src="${imgUrl}" alt="Training Predicted vs Actual" class="inference-model-graphic-img inference-training-plot-img" style="display: block; max-height: 300px; width: auto;">
                            <svg class="inference-overlay-svg" style="position: absolute; left: 0; top: 0; width: 100%; height: 100%; pointer-events: none;" viewBox="0 0 100 100" preserveAspectRatio="xMidYMid meet">${overlayPoints}</svg>
                        </div>
                        <p style="margin: 4px 0 0 0; font-size: 0.8rem; color: #555;">Same plot as Modeling page; inference points (gray) on diagonal.</p>`;
                })() : '';
                const inferenceDistLabel = 'Inference distribution';
                previewChartHtml = `
                    <h4 style="margin: 0 0 8px 0;">Inference visuals (${firstCol})</h4>
                    <p style="margin: 0 0 10px 0; font-size: 0.9rem; color: #666;">Training Predicted vs Actual with inference points overlaid (left); inference distribution (right).</p>
                    <div class="inference-visuals-row" style="display: flex; flex-wrap: wrap; gap: 20px; align-items: flex-start;">
                        ${(trainingViz || inferenceVizUrl) ? `<div class="inference-training-plot-wrap" style="flex: 0 1 320px; min-width: 200px;"><p style="margin: 0 0 6px 0; font-size: 0.85rem; font-weight: 600;">${inferenceVizUrl ? 'Predicted vs Actual (training + inference)' : 'Training plot + inference overlay'}</p>${leftPlotHtml}</div>` : ''}
                        <div class="inference-dist-wrap" style="flex: 1 1 280px; min-width: 200px;">
                            <p style="margin: 0 0 6px 0; font-size: 0.85rem; font-weight: 600;">${inferenceDistLabel}</p>
                            <svg width="${histW}" height="${histH}" viewBox="0 0 ${histW} ${histH}" class="inference-dist-svg" style="max-width: 100%; height: auto;">
                                ${histBarsSvg}
                                <line x1="${histPad.left}" y1="${histPad.top}" x2="${histPad.left}" y2="${histPad.top + histPlotH}" stroke="#333" stroke-width="1"/>
                                <line x1="${histPad.left}" y1="${histPad.top + histPlotH}" x2="${histPad.left + histPlotW}" y2="${histPad.top + histPlotH}" stroke="#333" stroke-width="1"/>
                                ${histYTickLines}
                                ${histYTicks.map(t => `<text x="${histPad.left - 8}" y="${histPad.top + histPlotH - (t / maxCount) * histPlotH + 4}" text-anchor="end" font-size="10" fill="#444">${t}</text>`).join('')}
                                ${xTickLines}${xTickTexts}
                                <text x="${histPad.left + histPlotW/2}" y="${histH - 2}" text-anchor="middle" font-size="10" fill="#444">Predicted units</text>
                                <text x="14" y="${histPad.top + histPlotH/2}" text-anchor="middle" font-size="10" fill="#444" transform="rotate(-90, 14, ${histPad.top + histPlotH/2})">Count</text>
                            </svg>
                        </div>
                    </div>
                    <p style="margin-top: 8px; font-size: 0.85rem; color: #666;">Range: ${min.toFixed(4)} – ${max.toFixed(4)} (${vals.length} points)</p>
                `;
            }
            // Classification: predicted class counts are already in the summary table; no duplicate "Inference distribution" block

            const numericKeysForPanels = ['column', 'n', 'min', 'max', 'mean', 'std', '25', '50', '75', '100'];
            const trainingSummaryForPanel = data.training_target_summary && data.training_target_summary.summary;

            const classificationNote = modelType === 'classification' ? `
                <p class="inference-classification-note" style="margin: 0 0 12px 0; font-size: 0.9rem; color: #555; font-style: italic;">No true labels for new data, so we only show predicted class counts below and the model's test-set performance (confusion matrix) on the right.</p>
            ` : '';

            function renderSummaryTable(rows, numericKeys) {
                if (!rows || rows.length === 0) return '<p>No data.</p>';
                const hasNumeric = rows.some(r => r.mean != null || r.min != null);
                if (hasNumeric) {
                    return `<table class="stats-table model-stats-table"><thead><tr>${numericKeys.map(k => `<th>${k === '25' ? '25%' : k === '50' ? '50%' : k === '75' ? '75%' : k === '100' ? '100%' : k}</th>`).join('')}</tr></thead><tbody>
                        ${rows.filter(r => r.mean != null || r.min != null).map(row => `
                            <tr>${numericKeys.map(k => `<td>${row[k] != null ? (typeof row[k] === 'number' ? Number(row[k]).toFixed(4) : escapeHtml(String(row[k]))) : ''}</td>`).join('')}</tr>
                        `).join('')}
                    </tbody></table>`;
                }
                return rows.map(row => `
                    <div style="margin-bottom: 8px;">
                        <strong>${escapeHtml(row.column)}</strong> (n=${row.n})
                        ${row.value_counts && row.value_counts.length ? `
                            <table class="stats-table model-stats-table" style="margin-top: 4px;"><tr><th>Value</th><th>Count</th></tr>
                                ${row.value_counts.map(vc => `<tr><td>${escapeHtml(vc.value)}</td><td>${vc.count}</td></tr>`).join('')}
                            </table>
                        ` : ''}
                    </div>
                `).join('');
            }
            const _numericKeys = ['column', 'n', 'min', 'max', 'mean', 'std', '25', '50', '75', '100'];
            const trainingSummaryTableHtml = trainingSummaryForPanel && trainingSummaryForPanel.length ? `
                <h3 style="margin: 0 0 8px 0;">Training data (target)</h3>
                <p style="margin: 0 0 12px 0; font-size: 0.95rem; color: #666;">Spread the model was built on.</p>
                <div class="model-stats-table-wrapper" style="margin-bottom: 20px;">${renderSummaryTable(trainingSummaryForPanel, numericKeysForPanels)}</div>
            ` : '';
            const modelGraphicSectionHtml = trainingViz ? `
                <h3 style="margin: 0 0 8px 0;">Model used (training performance)</h3>
                <p style="margin: 0 0 12px 0; font-size: 0.95rem; color: #666;">Performance graphic from the model you trained.</p>
                <img src="${trainingPerfUrl || trainingVizUrl}" alt="Training performance" class="inference-model-graphic-img" style="display: block; max-height: 400px; width: auto; object-fit: contain;">
            ` : '';
            const rightPanelHtml = (trainingSummaryTableHtml || modelGraphicSectionHtml) ? `
                <div class="inference-model-graphic" style="flex: 0 0 calc(50% - 12px); min-width: 200px; overflow: auto; align-self: flex-start;">
                    ${trainingSummaryTableHtml}
                    ${modelGraphicSectionHtml}
                </div>
            ` : '';

            predictionResults.innerHTML = `
                <h2>Inference Results</h2>
                <p class="inference-model-type" style="margin: 0 0 16px 0; font-size: 0.95rem; color: #555;"><strong>Model type:</strong> ${escapeHtml(modelTypeLabel)}</p>
                <p>Your inference results for '<strong>${escapeHtml(data.filename || 'file')}</strong>' are ready to download.</p>
                <div class="button-group" style="margin-bottom: 24px;">
                    <a href="${withApiRoot('/download/predictions.csv')}?download_name=${encodeURIComponent(predictionDownloadName)}" onclick="return downloadFile('predictions.csv', '${predictionDownloadName}')">
                        <button class="predictionresultButton export-button">Download Results CSV</button>
                    </a>
                    <button class="secondary-button" onclick="backToModel()">Back To Model</button>
                    <button class="secondary-button" onclick="newPredict()">Run inference on another dataset</button>
                </div>
                <div class="inference-results-content" style="display: flex; flex-wrap: nowrap; gap: 24px; align-items: flex-start;">
                    <div class="inference-summary-section" style="flex: 0 0 calc(50% - 12px); min-width: 200px; overflow: auto; align-self: flex-start;">
                        <h3 style="margin: 0 0 8px 0;">Inference summary</h3>
                        <p style="margin: 0 0 12px 0; font-size: 0.95rem; color: #666;">Descriptive statistics for inferred (predicted) values (same style as Data Exploration).</p>
                        ${classificationNote}
                        <div class="model-stats-table-wrapper" style="margin-bottom: 16px;">${summaryTableHtml}</div>
                        ${previewChartHtml ? `<div class="inference-preview-chart" style="margin-top: 16px;">${previewChartHtml}</div>` : ''}
                    </div>
                    ${rightPanelHtml}
                </div>
            `
            uploadPredictDf.classList.add('hidden')
        }
        else {
            predictionErrorDiv.classList.remove('hidden')
            // Sanitize error message to prevent XSS
            const safeError = escapeHtml(String(data.error || 'Unknown error'));
            predictionErrorDiv.innerHTML = `<p>Error: ${safeError}</p>`
            predictionResults.classList.add('hidden')
        }
    }
    catch (e) {
        predictionErrorDiv.classList.remove('hidden')
        // Sanitize error message to prevent XSS
        const safeError = escapeHtml(String(e));
        predictionErrorDiv.innerHTML = `<p>Error: ${safeError}</p>`
    }
    


});

// AutoML form submission handler
const automlForm = document.getElementById('automlForm');
if (automlForm) {
    automlForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // AutoML mode is selected - trigger processForm submission with AutoML defaults
        // The processForm handler will detect AutoML mode and set appropriate defaults
        const automlModeRadio = document.getElementById('automlMode');
        if (automlModeRadio) {
            automlModeRadio.checked = true;
            switchModelingMode('automl');
        }
        
        // Check if processForm exists
        if (!processForm) {
            const errorDiv = getCachedElement('errorDiv');
            if (errorDiv) {
                showError(errorDiv, 'Error: Unable to find model configuration. Please configure your data first.');
                errorDiv.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }
            return;
        }
        
        // Disable the submit button and show initial status
        const submitButton = document.getElementById('automlSubmitButton');
        if (submitButton) {
            submitButton.disabled = true;
            submitButton.textContent = 'Running AutoML...';
        }
        
        // Show loading indicator immediately with enhanced visibility
        const automlLoading = document.getElementById('automlLoading');
        if (automlLoading) {
            automlLoading.classList.remove('hidden');
            automlLoading.style.display = 'block';
            automlLoading.innerHTML = `
                <div style="padding: 24px; background-color: #fff3cd; border: 2px solid #ffc107; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                    <p style="font-size: 1.3em; font-weight: 700; margin-bottom: 12px; color: #856404; display: flex; align-items: center; gap: 10px;">
                        <span>AutoML is running...</span>
                    </p>
                    <p style="color: #856404; margin-bottom: 16px; font-size: 1.05em;">Setting up automated model selection and optimization. This may take several hours for large datasets.</p>
                    <div class="spinner" style="margin: 0 auto; border-top-color: #856404;"></div>
                    <p style="color: #666; margin-top: 16px; font-size: 0.95em; font-style: italic;">Progress updates will appear here as AutoML evaluates different models and configurations.</p>
                </div>
            `;
        }
        
        // Show stop button
        const stopButton = document.getElementById('stopAutomlButton');
        if (stopButton) stopButton.style.display = 'inline-block';
        
        // Programmatically trigger the processForm submission
        // processForm will detect AutoML mode and set defaults
        const submitEvent = new Event('submit', { bubbles: true, cancelable: true });
        processForm.dispatchEvent(submitEvent);
        
        // Note: Button will be re-enabled by processForm handler when complete or on error
    });
}

// Advanced Optimization form submission - triggers the same model training as processForm
if (advancedOptimizationForm) {
    advancedOptimizationForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // Switch to Advanced mode if not already selected
        const advancedModeRadio = document.getElementById('advancedMode');
        if (advancedModeRadio && !advancedModeRadio.checked) {
            advancedModeRadio.checked = true;
            switchModelingMode('advanced');
        }
        
        // Check if a model has been selected
        const advancedNModels = getCachedElement('advancedNModels');
        const advancedClModels = getCachedElement('advancedClModels');
        const advancedClassModels = getCachedElement('advancedClassModels');
        
        const selectedModel = advancedNModels?.value || advancedClModels?.value || advancedClassModels?.value;
        
        if (!selectedModel) {
            const errorDiv = getCachedElement('errorDiv');
            if (errorDiv) {
                showError(errorDiv, 'Please select a model before running with advanced options.');
                errorDiv.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }
            return;
        }
        
        // Check if processForm exists and is valid
        if (!processForm) {
            const errorDiv = getCachedElement('errorDiv');
            if (errorDiv) {
                showError(errorDiv, 'Error: Unable to find model configuration. Please configure your data first.');
                errorDiv.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }
            return;
        }
        
        // Disable the submit button to prevent double submission
        const submitButton = document.getElementById('advancedOptimizationSubmitButton');
        if (submitButton) {
            submitButton.disabled = true;
            submitButton.textContent = 'Running...';
        }
        
        // Programmatically trigger the processForm submission
        // This will use all the settings from both Modeling page and Advanced Optimization page
        const submitEvent = new Event('submit', { bubbles: true, cancelable: true });
        processForm.dispatchEvent(submitEvent);
        
        // Re-enable button after a delay (in case of error)
        setTimeout(() => {
            if (submitButton) {
                submitButton.disabled = false;
                submitButton.textContent = 'Run Model with Advanced Options';
            }
        }, 1000);
    });
}
