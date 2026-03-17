/**
 * Preprocess tab: handlePreprocessFormSubmit, preprocess-form and exploration UI. Depends on core.js.
 */
async function handlePreprocessFormSubmit(e) {
    if (!e.target || e.target.id !== 'preprocessform') return;
    const stratErrorDiv = document.getElementById('stratErrorDiv');
    const specificVariableSelect = document.getElementById('specificVariableSelect');
    const stratifyColumn = specificVariableSelect ? specificVariableSelect.value : '';
    const quantileBinsEl = document.getElementById('quantileBins');
    const quantileBins = quantileBinsEl ? quantileBinsEl.value : '';
    const binsEl = document.getElementById('bins');
    const bins = binsEl ? binsEl.value : '';
    const binsLabelEl = document.getElementById('binsLabel');
    const binsLabel = binsLabelEl ? binsLabelEl.value : '';
    const quantilesEl = document.getElementById('quantiles');
    const quantiles = quantilesEl ? quantilesEl.value : '';
    const indicators = indicatorsSelect ? indicatorsSelect.value : '';
    const predictors = predictorsSelect ? predictorsSelect.value : '';
    const outputType1El = document.getElementById('outputType1');
    let outputType = outputType1El ? outputType1El.value : ''

    //Error checking for if using stratify or quantiles/bins
    if ((quantileBins =='quantiles' || quantileBins =='Bins') && stratifyColumn.trim()==""){
        e.preventDefault();
        showError(stratErrorDiv, 'Must fill out stratify variable if using quantiles or Bins', false);
        // Set aria-invalid on the stratify field
        const stratifyInput = document.getElementById('specificVariableSelect');
        if (stratifyInput) {
            stratifyInput.setAttribute('aria-invalid', 'true');
            manageFocus(stratifyInput);
        }
    }

    else if (quantileBins =='quantiles' && quantiles==''){
        e.preventDefault();
        showError(stratErrorDiv, 'Must fill out quantiles', false);
        const quantilesInput = document.getElementById('quantiles');
        if (quantilesInput) {
            quantilesInput.setAttribute('aria-invalid', 'true');
            manageFocus(quantilesInput);
        }
    }

    else if (quantileBins =='Bins' && (bins=='' || binsLabel=='')){
        e.preventDefault();
        showError(stratErrorDiv, 'Must fill out bins thresholds and labels', false);
        const binsInput = document.getElementById('bins');
        const binsLabelInput = document.getElementById('binsLabel');
        if (binsInput && !binsInput.value) {
            binsInput.setAttribute('aria-invalid', 'true');
            manageFocus(binsInput);
        } else if (binsLabelInput && !binsLabelInput.value) {
            binsLabelInput.setAttribute('aria-invalid', 'true');
            manageFocus(binsLabelInput);
        }
    }

    else if (stratifyColumn.trim()!=='' && quantileBins=='None'){
        e.preventDefault();
        showError(stratErrorDiv, 'Must fill use bins or quantiles if using stratify value', false);
        const quantileBinsSelect = document.getElementById('quantileBins');
        if (quantileBinsSelect) {
            quantileBinsSelect.setAttribute('aria-invalid', 'true');
            manageFocus(quantileBinsSelect);
        }
    }
    else if (!predictors && outputType!=='Cluster'){
        e.preventDefault();
        showError(stratErrorDiv, 'Must have targets for Classification and Regression Models', false);
        const predictorsInput = document.getElementById('predictors');
        if (predictorsInput) {
            predictorsInput.setAttribute('aria-invalid', 'true');
            manageFocus(predictorsInput);
        }
    }
    else if (stratifyColumn.trim() !== '' && (quantileBins === 'quantiles' || quantileBins === 'Bins') && predictors.trim() !== '' && outputType !== 'Cluster') {
        const pCols = getColumnIndices(predictors.toUpperCase().replace(/\s/g, ''));
        const stratNum = columnToIndex(stratifyColumn.trim().toUpperCase());
        if (pCols.indexOf(stratNum) !== -1) {
            e.preventDefault();
            showError(stratErrorDiv, 'Do not stratify by your target column. That would leak target information into the train/test split. Choose a different column to stratify by.', false);
            const stratifyInput = document.getElementById('specificVariableSelect');
            if (stratifyInput) {
                stratifyInput.setAttribute('aria-invalid', 'true');
                manageFocus(stratifyInput);
            }
        }
    }

    // If required variables are filled in, send to route and only switch to Modeling on success
    if (!e.defaultPrevented) {
        e.preventDefault(); // prevent native form submit now that we're handling it
        // Clear any previous aria-invalid attributes since validation passed
        const formFields = [
            'specificVariableSelect', 'quantiles', 'bins', 'binsLabel', 
            'quantileBins', 'predictors', 'modelingType'
        ];
        formFields.forEach(fieldId => {
            const field = document.getElementById(fieldId);
            if (field) {
                field.removeAttribute('aria-invalid');
            }
        });

        stratErrorDiv.innerHTML = '';
        const predictorCols = getColumnIndices((predictors || '').toUpperCase().replace(/\s/g, ''));
        const indicatorCols = getColumnIndices((indicators || '').toUpperCase().replace(/\s/g, ''));
        stratifyColumnNumber = columnToIndex((stratifyColumn || '').toUpperCase());

        const requestData = {
            filename: uploadedFileName,
            indicators: indicatorCols,
            predictors: predictorCols,
            stratify: stratifyColumnNumber,
            outputType: outputType || ''
        };

        try {
            const response = await fetch('/preprocess', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(requestData),
            });
            const data = await response.json();

            if (!response.ok) {
                if (stratErrorDiv) showError(stratErrorDiv, data.error || 'Preprocess failed. Please check your selections.', false);
                else if (errorDiv) showError(errorDiv, data.error || 'Preprocess failed.', false);
                return;
            }

            // Success: now switch UI to Modeling
            const columnDiv = document.getElementById('columnsection');
            const fileuploaddiv = document.getElementById('fileuploaddiv');
            const userInputSection = document.getElementById('userInputSection');
            const columnSelection = document.getElementById('columnSelection');
            if (columnDiv) columnDiv.classList.add('hidden');
            if (fileuploaddiv) fileuploaddiv.classList.add('hidden');
            if (userInputSection) userInputSection.classList.add('hidden');
            if (columnSelection) {
                columnSelection.dataset.ready = 'true';
                columnSelection.style.display = 'block';
            }
            if (typeof showTab === 'function') showTab('modeling');

            stratifyStr = '';
            if (stratifyColumn && stratifyColumn.trim() !== '' && data.stratify) {
                stratifyStr = 'with stratification by ' + data.stratify + ' value';
            }

            const preds = data.predictors;
            let predictorsColNameString = Array.isArray(preds) ? preds.join(',').substring(0, 10) : (preds || '').substring(0, 10);
            if (predictorsColNameString.length >= 10) {
                predictorsColNameString += '...';
            }

            const IndicatorsPredictorsSection = document.getElementById('modelingHeaderActions');
            const outputTypeLabel = outputType === 'Numeric' ? 'Regression' : outputType;

            if (!IndicatorsPredictorsSection) {
                if (errorDiv) showError(errorDiv, 'Modeling header section not found.');
                return;
            }

            // Displaying what was selected to user
            if (outputType === 'Cluster') {
                const noteText = `<em>Columns ${escapeHtml((indicators || '').toUpperCase())} are selected as indicators for ${escapeHtml(outputTypeLabel)} based modeling.</em>`;
                const simpleNote = document.getElementById('simpleModelingSelectionNote');
                const advancedNote = document.getElementById('advancedModelingSelectionNote');
                const automlNote = document.getElementById('automlModelingSelectionNote');
                if (simpleNote) simpleNote.innerHTML = noteText;
                if (advancedNote) advancedNote.innerHTML = noteText;
                if (automlNote) automlNote.innerHTML = noteText;
                IndicatorsPredictorsSection.innerHTML = `
                <div class='columnOutputSection' style="display: flex; gap: 12px; align-items: center;">
                    <button class="success-button" onclick="predictionPage()">Move Forward to Apply this Model for Inferencing on New Data</button>
                </div>`;
            }
            else {
                const noteText = `<em>Columns ${escapeHtml((indicators || '').toUpperCase())} are selected as indicators to predict column(s) ${escapeHtml((predictors || '').toUpperCase())} (${escapeHtml(predictorsColNameString)}) by ${escapeHtml(outputTypeLabel)} based modeling.</em>`;
                const simpleNote = document.getElementById('simpleModelingSelectionNote');
                const advancedNote = document.getElementById('advancedModelingSelectionNote');
                const automlNote = document.getElementById('automlModelingSelectionNote');
                if (simpleNote) simpleNote.innerHTML = noteText;
                if (advancedNote) advancedNote.innerHTML = noteText;
                if (automlNote) automlNote.innerHTML = noteText;
                IndicatorsPredictorsSection.innerHTML = `
                <div class='columnOutputSection' style="display: flex; gap: 12px; align-items: center;">
                    <button class="success-button" onclick="predictionPage()">Move Forward to Apply this Model for Inferencing on New Data</button>
                </div>`;
            }
        } catch (error) {
            console.error('Preprocess submit error:', error);
            const msg = error && error.message ? error.message : 'Request failed. See console for details.';
            if (stratErrorDiv) showError(stratErrorDiv, msg, false);
            else if (errorDiv) showError(errorDiv, msg, false);
        }
    }
}

// Expose globally so inline onclick on the button can call it
window.handlePreprocessFormSubmit = handlePreprocessFormSubmit;

// Attach using delegation (capturing phase so we run before any other handler)
document.addEventListener('submit', handlePreprocessFormSubmit, true);

/// Section 4: displaying and hidding divs based on user selection

    // Handling displaying the 'how to replace missing values' user input
    document.getElementById('dropMissing').addEventListener('change', function(){
        let missingColSelection = this.value;
        let imputeDiv = document.getElementById('imputeDiv');
        if (missingColSelection=='none'){
            //impute div hidden
            imputeDiv.classList.add('hidden')
        }
        else{
            //impute div not hidden
            imputeDiv.classList.remove('hidden')
        }
    })

    document.getElementById('exploreDropMissing').addEventListener('change', function(){
        let missingColSelection = this.value;
        let imputeDiv = document.getElementById('exploreImputeDiv');
        if (missingColSelection=='none'){
            //impute div hidden
            imputeDiv.classList.add('hidden')
        }
        else{
            //impute div not hidden
            imputeDiv.classList.remove('hidden')
        }
    })

    // Handling displaying the stratifying options of bins or quantiles
    document.getElementById('scalingYesNo').addEventListener('change', function() {
        let scalingAnswer = this.value;
        let scalingYes = document.getElementById('scalingYes');
        if (scalingAnswer === 'Yes'){
            scalingYes.classList.remove('hidden')
        }
        else{
            scalingYes.classList.add('hidden');
            const stratifyColumnInput = document.getElementById('specificVariableSelect');
            const quantileBinsSelect = document.getElementById('quantileBins');
            const quantileInput = document.getElementById('quantileInput');
            const binInput = document.getElementById('binInput');
            const quantilesField = document.getElementById('quantiles');
            const binsField = document.getElementById('bins');
            const binsLabelField = document.getElementById('binsLabel');
            if (stratifyColumnInput) {
                stratifyColumnInput.value = '';
            }
            if (quantileBinsSelect) {
                quantileBinsSelect.value = 'None'; // Reset to "Neither" option
            }
            if (quantileInput) {
                quantileInput.classList.add('hidden');
            }
            if (binInput) {
                binInput.classList.add('hidden');
            }
            if (quantilesField) {
                quantilesField.value = '';
            }
            if (binsField) {
                binsField.value = '';
            }
            if (binsLabelField) {
                binsLabelField.value = '';
            }
        }
    });

    // Handle displaying the transformer user input for transformer columns
    document.getElementById('useTransformer').addEventListener('change', function(){
        let transformerAnswer = this.value;
        let transformerYes = document.getElementById('transformerYes');
        if (transformerAnswer === 'Yes'){
            transformerYes.classList.remove('hidden')
        }
        else{
            transformerYes.classList.add('hidden');
        }
    });

    // Auto-detect NaN or zeros in indicator/target columns (Data Cleaning)
    const autoDetectNanZerosBtn = document.getElementById('autoDetectNanZeros');
    const dataCleaningMissingWrapper = document.getElementById('dataCleaningMissingWrapper');
    const dataCleaningZerosWrapper = document.getElementById('dataCleaningZerosWrapper');
    const dataCleaningAutodetectMessage = document.getElementById('dataCleaningAutodetectMessage');
    const dropMissingSelect = document.getElementById('dropMissing');
    const drop0Select = document.getElementById('drop0');
    if (autoDetectNanZerosBtn && dataCleaningAutodetectMessage) {
        autoDetectNanZerosBtn.addEventListener('click', async function() {
            if (!indicatorsSelect || !indicatorsSelect.value.trim() || !predictorsSelect || !predictorsSelect.value.trim()) {
                alert('Please enter Indicator and Target columns first (e.g., A-D and A).');
                return;
            }
            const indicatorIndices = getColumnIndices(indicatorsSelect.value.toUpperCase().replace(/\s/g, ""));
            const predictorIndices = getColumnIndices(predictorsSelect.value.toUpperCase().replace(/\s/g, ""));
            if (indicatorIndices.length === 0 || predictorIndices.length === 0) {
                alert('Could not parse column indices. Use format like "A-D" or "A,B,C".');
                return;
            }
            try {
                autoDetectNanZerosBtn.disabled = true;
                autoDetectNanZerosBtn.textContent = 'Detecting...';
                const response = await fetch(withApiRoot('/auto-detect-nan-zeros'), {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ indicators: indicatorIndices, predictors: predictorIndices })
                });
                const data = await response.json();
                if (!response.ok) {
                    dataCleaningAutodetectMessage.textContent = data.error || 'Autodetect failed.';
                    if (dataCleaningMissingWrapper) dataCleaningMissingWrapper.classList.remove('autodetect-greyed');
                    if (dataCleaningZerosWrapper) dataCleaningZerosWrapper.classList.remove('autodetect-greyed');
                    if (dropMissingSelect) dropMissingSelect.disabled = false;
                    if (drop0Select) drop0Select.disabled = false;
                    return;
                }
                dataCleaningAutodetectMessage.textContent = data.message || '';
                // Grey out only the option(s) not needed: missing vs zeros independently
                if (dataCleaningMissingWrapper) {
                    if (data.needs_missing_handling) {
                        dataCleaningMissingWrapper.classList.remove('autodetect-greyed');
                        if (dropMissingSelect) dropMissingSelect.disabled = false;
                    } else {
                        dataCleaningMissingWrapper.classList.add('autodetect-greyed');
                        if (dropMissingSelect) { dropMissingSelect.value = 'none'; dropMissingSelect.disabled = true; }
                        document.getElementById('imputeDiv').classList.add('hidden');
                    }
                }
                if (dataCleaningZerosWrapper) {
                    if (data.needs_zero_handling) {
                        dataCleaningZerosWrapper.classList.remove('autodetect-greyed');
                        if (drop0Select) drop0Select.disabled = false;
                    } else {
                        dataCleaningZerosWrapper.classList.add('autodetect-greyed');
                        if (drop0Select) { drop0Select.value = 'none'; drop0Select.disabled = true; }
                    }
                }
            } catch (err) {
                console.error(err);
                dataCleaningAutodetectMessage.textContent = 'Autodetect failed. Check console.';
                if (dataCleaningMissingWrapper) dataCleaningMissingWrapper.classList.remove('autodetect-greyed');
                if (dataCleaningZerosWrapper) dataCleaningZerosWrapper.classList.remove('autodetect-greyed');
                if (dropMissingSelect) dropMissingSelect.disabled = false;
                if (drop0Select) drop0Select.disabled = false;
            } finally {
                autoDetectNanZerosBtn.disabled = false;
                autoDetectNanZerosBtn.textContent = 'Run';
            }
        });
    }

    // Auto-detect categorical columns for transformers
    const transformerOptionWrapper = document.getElementById('transformerOptionWrapper');
    const transformerAutodetectMessage = document.getElementById('transformerAutodetectMessage');
    const useTransformerSelect = document.getElementById('useTransformer');
    const transformerYesDiv = document.getElementById('transformerYes');

    function applyTransformerAutodetectResult(data, responseOk) {
        if (transformerAutodetectMessage) transformerAutodetectMessage.textContent = data.message || data.error || '';
        if (!responseOk || !data.transformer_indices) return;
        if (data.transformer_indices.length === 0) {
            transformerOptionWrapper.classList.add('autodetect-greyed');
            if (useTransformerSelect) { useTransformerSelect.value = 'No'; useTransformerSelect.disabled = true; }
            if (transformerYesDiv) transformerYesDiv.classList.add('hidden');
            const transformerColumnInput = document.getElementById('transformerColumn');
            if (transformerColumnInput) transformerColumnInput.value = '';
        } else {
            transformerOptionWrapper.classList.remove('autodetect-greyed');
            if (useTransformerSelect) { useTransformerSelect.disabled = false; useTransformerSelect.value = 'Yes'; }
            if (transformerYesDiv) transformerYesDiv.classList.remove('hidden');
            const transformerColumnInput = document.getElementById('transformerColumn');
            if (transformerColumnInput) {
                const letters = data.transformer_indices.map(idx => getColumnLetter(idx));
                const sorted = data.transformer_indices.slice().sort((a, b) => a - b);
                const isConsecutive = sorted.every((val, i, arr) => i === 0 || val === arr[i - 1] + 1);
                if (letters.length === 1) transformerColumnInput.value = letters[0];
                else if (isConsecutive) transformerColumnInput.value = `${getColumnLetter(sorted[0])}-${getColumnLetter(sorted[sorted.length - 1])}`;
                else transformerColumnInput.value = letters.join(', ');
            }
        }
    }

    const autoDetectTransformersTopBtn = document.getElementById('autoDetectTransformersTop');
    if (autoDetectTransformersTopBtn) {
        autoDetectTransformersTopBtn.addEventListener('click', async function() {
            if (!indicatorsSelect || !indicatorsSelect.value.trim()) {
                alert('Please enter indicator columns first (e.g., A-D).');
                return;
            }
            const indicatorIndices = getColumnIndices(indicatorsSelect.value.toUpperCase().replace(/\s/g, ""));
            if (indicatorIndices.length === 0) {
                alert('Could not parse indicator column indices. Use format like "A-D" or "A,B,C".');
                return;
            }
            try {
                autoDetectTransformersTopBtn.disabled = true;
                autoDetectTransformersTopBtn.textContent = 'Detecting...';
                const response = await fetch(withApiRoot('/auto-detect-transformers'), {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ indicators: indicatorIndices })
                });
                const data = await response.json();
                applyTransformerAutodetectResult(data, response.ok);
                if (response.ok && data.transformer_indices && data.transformer_indices.length > 0) {
                    if (transformerAutodetectMessage) transformerAutodetectMessage.textContent = data.message || '';
                }
            } catch (err) {
                console.error(err);
                if (transformerAutodetectMessage) transformerAutodetectMessage.textContent = 'Autodetect failed.';
                transformerOptionWrapper.classList.remove('autodetect-greyed');
                if (useTransformerSelect) useTransformerSelect.disabled = false;
            } finally {
                autoDetectTransformersTopBtn.disabled = false;
                autoDetectTransformersTopBtn.textContent = 'Run autodetect';
            }
        });
    }

    const autoDetectTransformersBtn = document.getElementById('autoDetectTransformers');
    if (autoDetectTransformersBtn) {
        autoDetectTransformersBtn.addEventListener('click', async function() {
            const indicatorsInput = document.getElementById('indicators');
            if (!indicatorsInput || !indicatorsInput.value.trim()) {
                alert('Please enter indicator columns first (e.g., A-D).');
                return;
            }
            const indicatorIndices = getColumnIndices(indicatorsInput.value.toUpperCase().replace(/\s/g, ""));
            if (indicatorIndices.length === 0) {
                alert('Could not parse indicator column indices. Please use format like "A-D" or "A,B,C".');
                return;
            }
            try {
                autoDetectTransformersBtn.disabled = true;
                autoDetectTransformersBtn.textContent = 'Detecting...';
                const response = await fetch(withApiRoot('/auto-detect-transformers'), {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ indicators: indicatorIndices })
                });
                const data = await response.json();
                applyTransformerAutodetectResult(data, response.ok);
                if (response.ok && data.transformer_indices && data.transformer_indices.length > 0) {
                    if (transformerAutodetectMessage) transformerAutodetectMessage.textContent = data.message || '';
                } else {
                    const message = data.message || data.error || 'No categorical columns detected.';
                    if (data.transformer_indices && data.transformer_indices.length === 0 && transformerAutodetectMessage) {
                        transformerAutodetectMessage.textContent = message;
                    }
                }
            } catch (error) {
                console.error('Error auto-detecting transformers:', error);
                if (transformerAutodetectMessage) transformerAutodetectMessage.textContent = 'Error detecting categorical columns.';
                transformerOptionWrapper.classList.remove('autodetect-greyed');
                if (useTransformerSelect) useTransformerSelect.disabled = false;
            } finally {
                autoDetectTransformersBtn.disabled = false;
                autoDetectTransformersBtn.textContent = 'Auto-detect';
            }
        });
    }

    // Handle displaying quantile and bins 
    document.getElementById('quantileBins').addEventListener('change', function() {
        let quantileBins = this.value;
        let quantileInput = document.getElementById("quantileInput");
        let binInput = document.getElementById("binInput");

        if (quantileBins=='None'){
            quantileInput.classList.add('hidden');
            binInput.classList.add('hidden');
        }
        else if (quantileBins=='quantiles'){
            quantileInput.classList.remove('hidden');
            binInput.classList.add('hidden');
        }
        else if (quantileBins=='Bins'){
            quantileInput.classList.add('hidden');
            binInput.classList.remove('hidden');
        }
    });


    // Handle displaying units if user selects yes
    document.getElementById('unitToggle').addEventListener('change', function() {
        let units = document.getElementById('units')
        let unitName = document.getElementById('unitName');
        if (this.checked) {
            units.classList.remove("hidden");
        } 
        else {
            units.classList.add("hidden");
            unitName.value=''
        }
    });

    // Advanced unit toggle handler
    const advancedUnitToggle = document.getElementById('advancedUnitToggle');
    if (advancedUnitToggle) {
        advancedUnitToggle.addEventListener('change', function() {
            let advancedUnits = document.getElementById('advancedUnits')
            let advancedUnitName = document.getElementById('advancedUnitName');
            if (advancedUnitName && advancedUnits) {
                if (this.checked) {
                    advancedUnits.classList.remove("hidden");
                } 
                else {
                    advancedUnits.classList.add("hidden");
                    advancedUnitName.value=''
                }
            }
        });
    }

    // AutoML unit toggle handler
    const automlUnitToggle = document.getElementById('automlUnitToggle');
    if (automlUnitToggle) {
        automlUnitToggle.addEventListener('change', function() {
            let automlUnits = document.getElementById('automlUnits')
            let automlUnitName = document.getElementById('automlUnitName');
            if (automlUnitName && automlUnits) {
                if (this.checked) {
                    automlUnits.classList.remove("hidden");
                } 
                else {
                    automlUnits.classList.add("hidden");
                    automlUnitName.value=''
                }
            }
        });
    }

    // Helper function to show/hide both regular and advanced field containers
    function toggleFieldVisibility(regularId, advancedId, show) {
        const regularField = document.getElementById(regularId);
        const advancedField = document.getElementById(advancedId);
        if (regularField) {
            if (show) {
                regularField.classList.remove("hidden");
            } else {
                regularField.classList.add("hidden");
            }
        }
        if (advancedField) {
            if (show) {
                advancedField.classList.remove("hidden");
            } else {
                advancedField.classList.add("hidden");
            }
        }
    }

    // Handle displaying the hyperparameters for specific models
    const outputType1Element = document.getElementById("outputType1");
    if (outputType1Element) {
        outputType1Element.addEventListener("change", function() {
            let outputType = this.value;
            
            // Hide all fields initially (both regular and advanced)
            toggleFieldVisibility('ridgeFields', 'advancedRidgeFields', false);
            toggleFieldVisibility('lassoFields', 'advancedLassoFields', false);
            toggleFieldVisibility('logisticFields', 'advancedLogisticFields', false);
            toggleFieldVisibility('polynomialFields', null, false);
            toggleFieldVisibility('elasticNetFields', null, false);
            toggleFieldVisibility('SVMFields', 'advancedSVMFields', false);
            toggleFieldVisibility('RFFields', 'advancedRFFields', false);
            toggleFieldVisibility('PerceptronFields', 'advancedPerceptronFields', false);
            toggleFieldVisibility('MLPFields', 'advancedMLPFields', false);
            toggleFieldVisibility('K-NearestFields', 'advancedK-NearestFields', false);
            toggleFieldVisibility('GradientBoostingFields', 'advancedGradientBoostingFields', false);

            updateOutputTypeDisplay(outputType);
            updateAutomlSettingsDisplay();
            // Update multi-output model availability when output type changes
            if (outputType === 'Numeric') {
                updateMultiOutputModelAvailability();
            }
        });
    }
    
    // Add listener for scaler changes to update AutoML display
    const scalerElement = getCachedElement('scaler');
    if (scalerElement) {
        scalerElement.addEventListener("change", function() {
            updateAutomlSettingsDisplay();
        });
    }

    // Function to update multi-output model availability based on target count
    function updateMultiOutputModelAvailability() {
        const predictorsInput = document.getElementById('predictors');
        if (!predictorsInput) return;
        
        const predictors = predictorsInput.value.trim();
        const predictorCols = predictors ? getColumnIndices(predictors.toUpperCase().replace(/\s/g, "")) : [];
        const isMultiOutput = predictorCols.length > 1;
        
        // Get all regression model dropdowns
        const modelSelects = [
            document.getElementById('simpleNModels'),
            document.getElementById('advancedNModels'),
            document.getElementById('automlNModels'),
            document.getElementById('headerNModels')
        ].filter(el => el !== null);
        
        modelSelects.forEach(select => {
            if (!select) return;
            
            // Find all options with data-requires-multi-output attribute
            Array.from(select.options).forEach(option => {
                if (option.hasAttribute('data-requires-multi-output')) {
                    if (isMultiOutput) {
                        // Enable for multi-output
                        option.disabled = false;
                        option.style.color = '';
                        option.style.opacity = '';
                    } else {
                        // Disable for single-output
                        option.disabled = true;
                        option.style.color = '#999';
                        option.style.opacity = '0.6';
                        
                        // If this option is currently selected, clear it
                        if (select.value === option.value) {
                            select.value = '';
                        }
                    }
                }
            });
        });
    }
    
    // Listen for changes to predictors input
    const predictorsInput = document.getElementById('predictors');
    if (predictorsInput) {
        predictorsInput.addEventListener('input', updateMultiOutputModelAvailability);
        predictorsInput.addEventListener('change', updateMultiOutputModelAvailability);
        // Initial check
        updateMultiOutputModelAvailability();
    }

    // Displays the numeric models (Simple mode)
    const nModelsElement = document.getElementById("simpleNModels");
    if (nModelsElement) {
        nModelsElement.addEventListener("change", function() {
            let selectedModel = this.value;
            
            // Hide all fields (both regular and advanced)
            toggleFieldVisibility('AgglomerativeFields', 'advancedAgglomerativeFields', false);
            toggleFieldVisibility('GaussianFields', 'advancedGaussianFields', false);
            toggleFieldVisibility('KmeansFields', 'advancedKmeansFields', false);
            toggleFieldVisibility('Logistic_classifierFields', 'advancedLogistic_classifierFields', false);
            toggleFieldVisibility('MLP_classifierFields', 'advancedMLP_classifierFields', false);
            toggleFieldVisibility('RF_classifierFields', 'advancedRF_classifierFields', false);
            toggleFieldVisibility('SVC_classifierFields', 'advancedSVC_classifierFields', false);
            toggleFieldVisibility('ridgeFields', 'advancedRidgeFields', false);
            toggleFieldVisibility('lassoFields', 'advancedLassoFields', false);
            toggleFieldVisibility('logisticFields', 'advancedLogisticFields', false);
            toggleFieldVisibility('polynomialFields', null, false);
            toggleFieldVisibility('elasticNetFields', null, false);
            toggleFieldVisibility('SVMFields', 'advancedSVMFields', false);
            toggleFieldVisibility('RFFields', 'advancedRFFields', false);
            toggleFieldVisibility('PerceptronFields', 'advancedPerceptronFields', false);
            toggleFieldVisibility('MLPFields', 'advancedMLPFields', false);
            toggleFieldVisibility('K-NearestFields', 'advancedK-NearestFields', false);
            toggleFieldVisibility('GradientBoostingFields', 'advancedGradientBoostingFields', false);

            // Show fields based on selection (both regular and advanced)
            if (selectedModel === "TerraFORMER") {
                //terraformerFields.classList.remove("hidden");
            } else if (selectedModel === "Ridge") {
                toggleFieldVisibility('ridgeFields', 'advancedRidgeFields', true);
            } else if (selectedModel === "Lasso") {
                toggleFieldVisibility('lassoFields', 'advancedLassoFields', true);
            } else if (selectedModel === "Logistic"){
                toggleFieldVisibility('logisticFields', 'advancedLogisticFields', true);
            } else if (selectedModel === "Polynomial") {
                toggleFieldVisibility('polynomialFields', null, true);
            } else if (selectedModel === "ElasticNet") {
                toggleFieldVisibility('elasticNetFields', null, true);
            } else if (selectedModel === "SVM") {
                toggleFieldVisibility('SVMFields', 'advancedSVMFields', true);
            } else if (selectedModel === "RF") {
                toggleFieldVisibility('RFFields', 'advancedRFFields', true);
            } else if (selectedModel === "Perceptron") {
                toggleFieldVisibility('PerceptronFields', 'advancedPerceptronFields', true);
            } else if (selectedModel === "MLP") {
                toggleFieldVisibility('MLPFields', 'advancedMLPFields', true);
            } else if (selectedModel === "K-Nearest") {
                toggleFieldVisibility('K-NearestFields', 'advancedK-NearestFields', true);
            } else if (selectedModel === "gradient_boosting") {
                toggleFieldVisibility('GradientBoostingFields', 'advancedGradientBoostingFields', true);
            }
        });
    }

    // Displays the cluster models
    const clModelsElement = document.getElementById("simpleClModels");
    if (clModelsElement) {
        clModelsElement.addEventListener("change", function() {
        let selectedModel = this.value;
        
        // Hide all fields (both regular and advanced)
        toggleFieldVisibility('AgglomerativeFields', 'advancedAgglomerativeFields', false);
        toggleFieldVisibility('GaussianFields', 'advancedGaussianFields', false);
        toggleFieldVisibility('KmeansFields', 'advancedKmeansFields', false);
        toggleFieldVisibility('Logistic_classifierFields', 'advancedLogistic_classifierFields', false);
        toggleFieldVisibility('MLP_classifierFields', 'advancedMLP_classifierFields', false);
        toggleFieldVisibility('RF_classifierFields', 'advancedRF_classifierFields', false);
        toggleFieldVisibility('SVC_classifierFields', 'advancedSVC_classifierFields', false);
        toggleFieldVisibility('ridgeFields', 'advancedRidgeFields', false);
        toggleFieldVisibility('lassoFields', 'advancedLassoFields', false);
        toggleFieldVisibility('logisticFields', 'advancedLogisticFields', false);
        toggleFieldVisibility('polynomialFields', null, false);
        toggleFieldVisibility('elasticNetFields', null, false);
        toggleFieldVisibility('SVMFields', 'advancedSVMFields', false);
        toggleFieldVisibility('RFFields', 'advancedRFFields', false);
        toggleFieldVisibility('PerceptronFields', 'advancedPerceptronFields', false);
        toggleFieldVisibility('MLPFields', 'advancedMLPFields', false);
        toggleFieldVisibility('K-NearestFields', 'advancedK-NearestFields', false);
        toggleFieldVisibility('GradientBoostingFields', 'advancedGradientBoostingFields', false);
        // Additional clustering models
        toggleFieldVisibility(null, 'advancedAffinityPropagationFields', false);
        toggleFieldVisibility(null, 'advancedBisectingKmeansFields', false);
        toggleFieldVisibility(null, 'advancedHDBSCANFields', false);
        toggleFieldVisibility(null, 'advancedMeanshiftFields', false);
        toggleFieldVisibility(null, 'advancedMinibatchKmeansFields', false);
        toggleFieldVisibility(null, 'advancedOPTICSFields', false);

        // Show fields based on selection (both regular and advanced)
        if (selectedModel === 'agglo'){
            toggleFieldVisibility('AgglomerativeFields', 'advancedAgglomerativeFields', true);
        }
        else if (selectedModel === 'gmm'){
            toggleFieldVisibility('GaussianFields', 'advancedGaussianFields', true);
        }
        else if (selectedModel === 'kmeans'){
            toggleFieldVisibility('KmeansFields', 'advancedKmeansFields', true);
        }
        else if (selectedModel === 'dbscan'){
            // DBSCAN fields already handled elsewhere
        }
        else if (selectedModel === 'birch'){
            // BIRCH fields already handled elsewhere
        }
        else if (selectedModel === 'spectral'){
            // Spectral fields already handled elsewhere
        }
        else if (selectedModel === 'affinity_propagation'){
            toggleFieldVisibility(null, 'advancedAffinityPropagationFields', true);
        }
        else if (selectedModel === 'bisecting_kmeans'){
            toggleFieldVisibility(null, 'advancedBisectingKmeansFields', true);
        }
        else if (selectedModel === 'hdbscan'){
            toggleFieldVisibility(null, 'advancedHDBSCANFields', true);
        }
        else if (selectedModel === 'meanshift'){
            toggleFieldVisibility(null, 'advancedMeanshiftFields', true);
        }
        else if (selectedModel === 'minibatch_kmeans'){
            toggleFieldVisibility(null, 'advancedMinibatchKmeansFields', true);
        }
        else if (selectedModel === 'optics'){
            toggleFieldVisibility(null, 'advancedOPTICSFields', true);
        }
        });
    }

    // Displays the classification models
    const classModelsElement = document.getElementById("simpleClassModels");
    if (classModelsElement) {
        classModelsElement.addEventListener("change", function() {
            let selectedModel = this.value;
            
            // Hide all fields (both regular and advanced)
            toggleFieldVisibility('AgglomerativeFields', 'advancedAgglomerativeFields', false);
            toggleFieldVisibility('GaussianFields', 'advancedGaussianFields', false);
            toggleFieldVisibility('KmeansFields', 'advancedKmeansFields', false);
            toggleFieldVisibility('Logistic_classifierFields', 'advancedLogistic_classifierFields', false);
            toggleFieldVisibility('MLP_classifierFields', 'advancedMLP_classifierFields', false);
            toggleFieldVisibility('RF_classifierFields', 'advancedRF_classifierFields', false);
            toggleFieldVisibility('SVC_classifierFields', 'advancedSVC_classifierFields', false);
            toggleFieldVisibility('ridgeFields', 'advancedRidgeFields', false);
            toggleFieldVisibility('lassoFields', 'advancedLassoFields', false);
            toggleFieldVisibility('logisticFields', 'advancedLogisticFields', false);
            toggleFieldVisibility('polynomialFields', null, false);
            toggleFieldVisibility('elasticNetFields', null, false);
            toggleFieldVisibility('SVMFields', 'advancedSVMFields', false);
            toggleFieldVisibility('RFFields', 'advancedRFFields', false);
            toggleFieldVisibility('PerceptronFields', 'advancedPerceptronFields', false);
            toggleFieldVisibility('MLPFields', 'advancedMLPFields', false);
            toggleFieldVisibility('K-NearestFields', 'advancedK-NearestFields', false);
            toggleFieldVisibility('GradientBoostingFields', 'advancedGradientBoostingFields', false);

            // Show fields based on selection (both regular and advanced)
            if (selectedModel === 'Logistic_classifier'){
                toggleFieldVisibility('Logistic_classifierFields', 'advancedLogistic_classifierFields', true);
            }
            else if (selectedModel === 'MLP_classifier'){
                toggleFieldVisibility('MLP_classifierFields', 'advancedMLP_classifierFields', true);
            }
            else if (selectedModel === 'RF_classifier'){
                toggleFieldVisibility('RF_classifierFields', 'advancedRF_classifierFields', true);
            }
            else if (selectedModel === 'SVC_classifier'){
                toggleFieldVisibility('SVC_classifierFields', 'advancedSVC_classifierFields', true);
            }
        });
    }

    // Displays the SVM hyperparamters based on selected kernal
    document.getElementById("kernel").addEventListener("change", function() {
        let kernel = this.value;
        let polykernelFields = document.getElementById("polykernelFields");
        let svmGamma = document.getElementById("svmGamma");

        if (kernel==='poly'){
            svmGamma.classList.remove('hidden');
            polykernelFields.classList.remove('hidden');
        }

        else if (kernel==='rbf'){
            polykernelFields.classList.add('hidden');
            svmGamma.classList.remove('hidden');
        }
        else {
            svmGamma.classList.add('hidden');
            polykernelFields.classList.add('hidden');
        }

    });

    // Displays the SVM classification hyperparamters based on selected kernal
    document.getElementById("Class_kernel").addEventListener("change", function() {
        let kernel = this.value;
        let class_polykernelFields = document.getElementById("Class_polykernelFields");
        let svcGamma = document.getElementById("SVCGammaContainer");

        if (kernel==='poly'){
            svcGamma.classList.remove('hidden');
            class_polykernelFields.classList.remove('hidden');
        }

        else if (kernel==='rbf'){
            class_polykernelFields.classList.add('hidden');
            svcGamma.classList.remove('hidden');
        }
        else {
            svcGamma.classList.add('hidden');
            class_polykernelFields.classList.add('hidden');
        }

    });

    // Displays or hides non required Ridge hyperparamters
    // Feature Selection toggle (works for both old and new locations)
    const featureSelectionMethod = document.getElementById('featureSelectionMethod');
    const featureSelectionParams = document.getElementById('featureSelectionParams');
    if (featureSelectionMethod && featureSelectionParams) {
        featureSelectionMethod.addEventListener('change', function() {
            if (this.value !== 'none') {
                featureSelectionParams.classList.remove('hidden');
            } else {
                featureSelectionParams.classList.add('hidden');
            }
        });
    }

    // Outlier Handling toggle (works for both old and new locations)
    const outlierMethod = document.getElementById('outlierMethod');
    const outlierActionDiv = document.getElementById('outlierActionDiv');
    if (outlierMethod && outlierActionDiv) {
        outlierMethod.addEventListener('change', function() {
            if (this.value !== 'none') {
                outlierActionDiv.classList.remove('hidden');
            } else {
                outlierActionDiv.classList.add('hidden');
            }
        });
    }

    // Hyperparameter Search toggle (works for both old and new locations)
    const hyperparameterSearch = document.getElementById('hyperparameterSearch');
    const hyperparameterSearchParams = document.getElementById('hyperparameterSearchParams');
    if (hyperparameterSearch && hyperparameterSearchParams) {
        hyperparameterSearch.addEventListener('change', function() {
            if (this.value !== 'none') {
                hyperparameterSearchParams.classList.remove('hidden');
            } else {
                hyperparameterSearchParams.classList.add('hidden');
            }
        });
    }

    document.getElementById('nonreqRidgeSlider').addEventListener('change', function() {
        let nonreqRidgeParams = document.getElementById('nonreqRidgeParams');
        if (this.checked) {
            nonreqRidgeParams.classList.remove("hidden");
        } 
        else{
            nonreqRidgeParams.classList.add("hidden");
            //reset all the values
        }
    });

    // Displays or hides non required Lasso hyperparamters
    document.getElementById('nonreqLassoSlider').addEventListener('change', function() {
        let nonreqLassoParams = document.getElementById('nonreqLassoParams');
        if (this.checked) {
            nonreqLassoParams.classList.remove("hidden");
        } 
        else{
            nonreqLassoParams.classList.add("hidden");
        }
    });

    // Displays or hides non required Logistic hyperparamters
    document.getElementById('nonreqLogisticSlider').addEventListener('change', function() {
        let nonreqLogisticParams = document.getElementById('nonreqLogisticParams');
        if (this.checked) {
            nonreqLogisticParams.classList.remove("hidden");
        } 
        else{
            nonreqLogisticParams.classList.add("hidden");
        }
    });

    // Displays or hides non required SVM hyperparamters
    document.getElementById('nonreqSVMSlider').addEventListener('change', function() {
        let nonreqSVMParams = document.getElementById('nonreqSVMParams');
        if (this.checked) {
            nonreqSVMParams.classList.remove("hidden");
        } 
        else{
            nonreqSVMParams.classList.add("hidden");
        }
    });

    // Displays or hides non required RF hyperparamters
    document.getElementById('nonreqRFSlider').addEventListener('change', function() {
        let nonreqRFParams = document.getElementById('nonreqRFParams');
        if (this.checked) {
            nonreqRFParams.classList.remove("hidden");
        } 
        else{
            nonreqRFParams.classList.add("hidden");
        }
    });

    // Displays or hides non required Perceptron hyperparamters
    document.getElementById('nonreqPerceptronSlider').addEventListener('change', function() {
        let nonreqPerceptronParams = document.getElementById('nonreqPerceptronParams');
        if (this.checked) {
            nonreqPerceptronParams.classList.remove("hidden");
        } 
        else{
            nonreqPerceptronParams.classList.add("hidden");
        }
    });

    // Displays or hides non required MLP hyperparamters
    document.getElementById('nonreqMLPSlider').addEventListener('change', function() {
        let nonreqMLPParams = document.getElementById('nonreqMLPParams');
        if (this.checked) {
            nonreqMLPParams.classList.remove("hidden");
        } 
        else{
            nonreqMLPParams.classList.add("hidden");
        }
    });

    // Displays or hides non required KNearest hyperparamters
    document.getElementById('nonreqKNearestSlider').addEventListener('change', function() {
        let nonreqKNearestParams = document.getElementById('nonreqKNearestParams');
        if (this.checked) {
            nonreqKNearestParams.classList.remove("hidden");
        } 
        else{
            nonreqKNearestParams.classList.add("hidden");
        }
    });

    // Displays or hides non required GB hyperparamters
    document.getElementById('nonreqGBSlider').addEventListener('change', function() {
        let nonreqGBParams = document.getElementById('nonreqGBParams');
        if (this.checked) {
            nonreqGBParams.classList.remove("hidden");
        } 
        else{
            nonreqGBParams.classList.add("hidden");
        }
    });

    // Displays or hides non required Logistic Classifier hyperparamters
    document.getElementById('nonreqLogisticClassifierSlider').addEventListener('change', function() {
        let nonreqLogisticClassifierParams = document.getElementById('nonreqLogisticClassifierParams');
        if (this.checked) {
            nonreqLogisticClassifierParams.classList.remove("hidden");
        } 
        else{
            nonreqLogisticClassifierParams.classList.add("hidden");
        }
    });

    // Displays or hides non required MLP Classifier hyperparamters
    document.getElementById('nonreqMLPClassifierSlider').addEventListener('change', function() {
        let nonreqMLPClassifierParams = document.getElementById('nonreqMLPClassifierParams');
        if (this.checked) {
            nonreqMLPClassifierParams.classList.remove("hidden");
        } 
        else{
            nonreqMLPClassifierParams.classList.add("hidden");
        }
    });

    // Displays or hides non required RF Classifier hyperparamters
    document.getElementById('nonreqRFClassifierSlider').addEventListener('change', function() {
        let nonreqRFClassifierParams = document.getElementById('nonreqRFClassifierParams');
        if (this.checked) {
            nonreqRFClassifierParams.classList.remove("hidden");
        } 
        else{
            nonreqRFClassifierParams.classList.add("hidden");
        }
    });

    // Displays or hides non required SVC hyperparamters
    document.getElementById('nonreqSVCClassifierSlider').addEventListener('change', function() {
        let nonreqSVCClassifierParams = document.getElementById('nonreqSVCClassifierParams');
        if (this.checked) {
            nonreqSVCClassifierParams.classList.remove("hidden");
        } 
        else{
            nonreqSVCClassifierParams.classList.add("hidden");
        }
    });

    // Displays or hides non required Agglo hyperparamters
    document.getElementById('nonreqAgglomerativeSlider').addEventListener('change', function() {
        let nonreqAgglomerativeParams = document.getElementById('nonreqAgglomerativeParams');
        if (this.checked) {
            nonreqAgglomerativeParams.classList.remove("hidden");
        } 
        else{
            nonreqAgglomerativeParams.classList.add("hidden");
        }
    });

    // Displays or hides non required Gaussian hyperparamters
    document.getElementById('nonreqGaussianSlider').addEventListener('change', function() {
        let nonreqGaussianParams = document.getElementById('nonreqGaussianParams');
        if (this.checked) {
            nonreqGaussianParams.classList.remove("hidden");
        } 
        else{
            nonreqGaussianParams.classList.add("hidden");
        }
    });

    // Displays or hides non required KMeans hyperparamters
    document.getElementById('nonreqKmeansSlider').addEventListener('change', function() {
        let nonreqKmeansParams = document.getElementById('nonreqKmeansParams');
        if (this.checked) {
            nonreqKmeansParams.classList.remove("hidden");
        } 
        else{
            nonreqKmeansParams.classList.add("hidden");
        }
    });
    
    // Advanced Optimization page event listeners (duplicate of above but for advanced IDs)
    // Helper function to set up advanced slider listeners
    function setupAdvancedSlider(sliderId, paramsId) {
        const slider = document.getElementById(sliderId);
        if (slider) {
            slider.addEventListener('change', function() {
                const params = document.getElementById(paramsId);
                if (params) {
                    if (this.checked) {
                        params.classList.remove("hidden");
                    } else {
                        params.classList.add("hidden");
                    }
                }
            });
        }
    }
    
    // Advanced SVM kernel change handler
    const advancedKernel = document.getElementById('advancedKernel');
    if (advancedKernel) {
        advancedKernel.addEventListener('change', function() {
            const polyFields = document.getElementById('advancedPolykernelFields');
            const gammaDiv = document.getElementById('advancedSvmGamma');
            if (this.value === 'poly') {
                if (polyFields) polyFields.classList.remove('hidden');
                if (gammaDiv) gammaDiv.classList.remove('hidden');
            } else {
                if (polyFields) polyFields.classList.add('hidden');
                if (this.value === 'rbf' || this.value === 'sigmoid') {
                    if (gammaDiv) gammaDiv.classList.remove('hidden');
                } else {
                    if (gammaDiv) gammaDiv.classList.add('hidden');
                }
            }
        });
    }
    
    // Set up all advanced slider listeners
    setupAdvancedSlider('advancedNonreqRidgeSlider', 'advancedNonreqRidgeParams');
    setupAdvancedSlider('advancedNonreqLassoSlider', 'advancedNonreqLassoParams');
    setupAdvancedSlider('advancedNonreqLogisticSlider', 'advancedNonreqLogisticParams');
    setupAdvancedSlider('advancedNonreqSVMSlider', 'advancedNonreqSVMParams');
    setupAdvancedSlider('advancedNonreqRFSlider', 'advancedNonreqRFParams');
    setupAdvancedSlider('advancedNonreqPerceptronSlider', 'advancedNonreqPerceptronParams');
    setupAdvancedSlider('advancedNonreqMLPSlider', 'advancedNonreqMLPParams');
    setupAdvancedSlider('advancedNonreqKNearestSlider', 'advancedNonreqKNearestParams');
    setupAdvancedSlider('advancedNonreqGBSlider', 'advancedNonreqGBParams');
    setupAdvancedSlider('advancedNonreqLogisticClassifierSlider', 'advancedNonreqLogisticClassifierParams');
    setupAdvancedSlider('advancedNonreqMLPClassifierSlider', 'advancedNonreqMLPClassifierParams');
    setupAdvancedSlider('advancedNonreqRFClassifierSlider', 'advancedNonreqRFClassifierParams');
    setupAdvancedSlider('advancedNonreqSVCClassifierSlider', 'advancedNonreqSVCClassifierParams');
    // Additional regression model sliders
    setupAdvancedSlider('advancedNonreqAdaBoostSlider', 'advancedNonreqAdaBoostParams');
    setupAdvancedSlider('advancedNonreqBaggingSlider', 'advancedNonreqBaggingParams');
    setupAdvancedSlider('advancedNonreqDecisionTreeSlider', 'advancedNonreqDecisionTreeParams');
    setupAdvancedSlider('advancedNonreqSGDSlider', 'advancedNonreqSGDParams');
    setupAdvancedSlider('advancedNonreqHistGBSlider', 'advancedNonreqHistGBParams');
    setupAdvancedSlider('advancedNonreqHuberSlider', 'advancedNonreqHuberParams');
    setupAdvancedSlider('advancedNonreqQuantileSlider', 'advancedNonreqQuantileParams');
    setupAdvancedSlider('advancedNonreqLinearSVRSlider', 'advancedNonreqLinearSVRParams');
    setupAdvancedSlider('advancedNonreqNuSVRSlider', 'advancedNonreqNuSVRParams');
    setupAdvancedSlider('advancedNonreqPassiveAggressiveSlider', 'advancedNonreqPassiveAggressiveParams');
    setupAdvancedSlider('advancedNonreqRANSACSlider', 'advancedNonreqRANSACParams');
    setupAdvancedSlider('advancedNonreqTheilSenSlider', 'advancedNonreqTheilSenParams');
    setupAdvancedSlider('advancedNonreqRadiusNeighborsSlider', 'advancedNonreqRadiusNeighborsParams');
    setupAdvancedSlider('advancedNonreqOMPSlider', 'advancedNonreqOMPParams');
    setupAdvancedSlider('advancedNonreqLARSSlider', 'advancedNonreqLARSParams');
    setupAdvancedSlider('advancedNonreqLARSCVSlider', 'advancedNonreqLARSCVParams');
    setupAdvancedSlider('advancedNonreqLassoCVSlider', 'advancedNonreqLassoCVParams');
    setupAdvancedSlider('advancedNonreqElasticNetCVSlider', 'advancedNonreqElasticNetCVParams');
    setupAdvancedSlider('advancedNonreqRidgeCVSlider', 'advancedNonreqRidgeCVParams');
    // Additional classification model sliders
    setupAdvancedSlider('advancedNonreqAdaBoostClassifierSlider', 'advancedNonreqAdaBoostClassifierParams');
    setupAdvancedSlider('advancedNonreqBaggingClassifierSlider', 'advancedNonreqBaggingClassifierParams');
    setupAdvancedSlider('advancedNonreqDecisionTreeClassifierSlider', 'advancedNonreqDecisionTreeClassifierParams');
    setupAdvancedSlider('advancedNonreqGradientBoostingClassifierSlider', 'advancedNonreqGradientBoostingClassifierParams');
    setupAdvancedSlider('advancedNonreqHistGradientBoostingClassifierSlider', 'advancedNonreqHistGradientBoostingClassifierParams');
    setupAdvancedSlider('advancedNonreqKNeighborsClassifierSlider', 'advancedNonreqKNeighborsClassifierParams');
    setupAdvancedSlider('advancedNonreqLDAClassifierSlider', 'advancedNonreqLDAClassifierParams');
    setupAdvancedSlider('advancedNonreqLinearSVCSlider', 'advancedNonreqLinearSVCParams');
    setupAdvancedSlider('advancedNonreqNuSVCSlider', 'advancedNonreqNuSVCParams');
    setupAdvancedSlider('advancedNonreqPassiveAggressiveClassifierSlider', 'advancedNonreqPassiveAggressiveClassifierParams');
    setupAdvancedSlider('advancedNonreqQDAClassifierSlider', 'advancedNonreqQDAClassifierParams');
    setupAdvancedSlider('advancedNonreqRidgeClassifierSlider', 'advancedNonreqRidgeClassifierParams');
    setupAdvancedSlider('advancedNonreqBernoulliNBSlider', 'advancedNonreqBernoulliNBParams');
    setupAdvancedSlider('advancedNonreqCategoricalNBSlider', 'advancedNonreqCategoricalNBParams');
    setupAdvancedSlider('advancedNonreqComplementNBSlider', 'advancedNonreqComplementNBParams');
    setupAdvancedSlider('advancedNonreqMultinomialNBSlider', 'advancedNonreqMultinomialNBParams');
    // Additional clustering model sliders
    setupAdvancedSlider('advancedNonreqAffinityPropagationSlider', 'advancedNonreqAffinityPropagationParams');
    setupAdvancedSlider('advancedNonreqBisectingKmeansSlider', 'advancedNonreqBisectingKmeansParams');
    setupAdvancedSlider('advancedNonreqHDBSCANSlider', 'advancedNonreqHDBSCANParams');
    setupAdvancedSlider('advancedNonreqMeanshiftSlider', 'advancedNonreqMeanshiftParams');
    setupAdvancedSlider('advancedNonreqMinibatchKmeansSlider', 'advancedNonreqMinibatchKmeansParams');
    setupAdvancedSlider('advancedNonreqOPTICSSlider', 'advancedNonreqOPTICSParams');
    setupAdvancedSlider('advancedNonreqAgglomerativeSlider', 'advancedNonreqAgglomerativeParams');
    setupAdvancedSlider('advancedNonreqGaussianSlider', 'advancedNonreqGaussianParams');
    setupAdvancedSlider('advancedNonreqKmeansSlider', 'advancedNonreqKmeansParams');
    
    // Advanced model selectors - show/hide hyperparameter fields
    const advancedNModelsElement = document.getElementById("advancedNModels");
    if (advancedNModelsElement) {
        advancedNModelsElement.addEventListener("change", function() {
            let selectedModel = this.value;
            
            // Hide all fields (both regular and advanced)
            toggleFieldVisibility('ridgeFields', 'advancedRidgeFields', false);
            toggleFieldVisibility('lassoFields', 'advancedLassoFields', false);
            toggleFieldVisibility('logisticFields', 'advancedLogisticFields', false);
            toggleFieldVisibility('polynomialFields', null, false);
            toggleFieldVisibility('elasticNetFields', null, false);
            toggleFieldVisibility('SVMFields', 'advancedSVMFields', false);
            toggleFieldVisibility('RFFields', 'advancedRFFields', false);
            toggleFieldVisibility('PerceptronFields', 'advancedPerceptronFields', false);
            toggleFieldVisibility('MLPFields', 'advancedMLPFields', false);
            toggleFieldVisibility('K-NearestFields', 'advancedK-NearestFields', false);
            toggleFieldVisibility('GradientBoostingFields', 'advancedGradientBoostingFields', false);
            // Additional regression models
            toggleFieldVisibility(null, 'advancedAdaBoostFields', false);
            toggleFieldVisibility(null, 'advancedBaggingFields', false);
            toggleFieldVisibility(null, 'advancedDecisionTreeFields', false);
            toggleFieldVisibility(null, 'advancedSGDFields', false);
            toggleFieldVisibility(null, 'advancedHistGradientBoostingFields', false);
            toggleFieldVisibility(null, 'advancedHuberFields', false);
            toggleFieldVisibility(null, 'advancedQuantileFields', false);
            toggleFieldVisibility(null, 'advancedLinearSVRFields', false);
            toggleFieldVisibility(null, 'advancedNuSVRFields', false);
            toggleFieldVisibility(null, 'advancedPassiveAggressiveFields', false);
            toggleFieldVisibility(null, 'advancedRANSACFields', false);
            toggleFieldVisibility(null, 'advancedTheilSenFields', false);
            toggleFieldVisibility(null, 'advancedRadiusNeighborsFields', false);
            toggleFieldVisibility(null, 'advancedOMPFields', false);
            toggleFieldVisibility(null, 'advancedLARSFields', false);
            toggleFieldVisibility(null, 'advancedLARSCVFields', false);
            toggleFieldVisibility(null, 'advancedLassoCVFields', false);
            toggleFieldVisibility(null, 'advancedElasticNetCVFields', false);
            toggleFieldVisibility(null, 'advancedRidgeCVFields', false);

            // Show fields based on selection (both regular and advanced)
            if (selectedModel === "Ridge") {
                toggleFieldVisibility('ridgeFields', 'advancedRidgeFields', true);
            } else if (selectedModel === "Lasso") {
                toggleFieldVisibility('lassoFields', 'advancedLassoFields', true);
            } else if (selectedModel === "Logistic"){
                toggleFieldVisibility('logisticFields', 'advancedLogisticFields', true);
            } else if (selectedModel === "Polynomial") {
                toggleFieldVisibility('polynomialFields', null, true);
            } else if (selectedModel === "ElasticNet") {
                toggleFieldVisibility('elasticNetFields', null, true);
            } else if (selectedModel === "SVM") {
                toggleFieldVisibility('SVMFields', 'advancedSVMFields', true);
            } else if (selectedModel === "RF") {
                toggleFieldVisibility('RFFields', 'advancedRFFields', true);
            } else if (selectedModel === "Perceptron") {
                toggleFieldVisibility('PerceptronFields', 'advancedPerceptronFields', true);
            } else if (selectedModel === "MLP") {
                toggleFieldVisibility('MLPFields', 'advancedMLPFields', true);
            } else if (selectedModel === "K-Nearest") {
                toggleFieldVisibility('K-NearestFields', 'advancedK-NearestFields', true);
            } else if (selectedModel === "gradient_boosting") {
                toggleFieldVisibility('GradientBoostingFields', 'advancedGradientBoostingFields', true);
            } else if (selectedModel === "AdaBoost") {
                toggleFieldVisibility(null, 'advancedAdaBoostFields', true);
            } else if (selectedModel === "Bagging") {
                toggleFieldVisibility(null, 'advancedBaggingFields', true);
            } else if (selectedModel === "DecisionTree") {
                toggleFieldVisibility(null, 'advancedDecisionTreeFields', true);
            } else if (selectedModel === "SGD") {
                toggleFieldVisibility(null, 'advancedSGDFields', true);
            } else if (selectedModel === "HistGradientBoosting") {
                toggleFieldVisibility(null, 'advancedHistGradientBoostingFields', true);
            } else if (selectedModel === "Huber") {
                toggleFieldVisibility(null, 'advancedHuberFields', true);
            } else if (selectedModel === "Quantile") {
                toggleFieldVisibility(null, 'advancedQuantileFields', true);
            } else if (selectedModel === "LinearSVR") {
                toggleFieldVisibility(null, 'advancedLinearSVRFields', true);
            } else if (selectedModel === "NuSVR") {
                toggleFieldVisibility(null, 'advancedNuSVRFields', true);
            } else if (selectedModel === "PassiveAggressive") {
                toggleFieldVisibility(null, 'advancedPassiveAggressiveFields', true);
            } else if (selectedModel === "RANSAC") {
                toggleFieldVisibility(null, 'advancedRANSACFields', true);
            } else if (selectedModel === "TheilSen") {
                toggleFieldVisibility(null, 'advancedTheilSenFields', true);
            } else if (selectedModel === "RadiusNeighbors") {
                toggleFieldVisibility(null, 'advancedRadiusNeighborsFields', true);
            } else if (selectedModel === "OMP") {
                toggleFieldVisibility(null, 'advancedOMPFields', true);
            } else if (selectedModel === "LARS") {
                toggleFieldVisibility(null, 'advancedLARSFields', true);
            } else if (selectedModel === "LARSCV") {
                toggleFieldVisibility(null, 'advancedLARSCVFields', true);
            } else if (selectedModel === "LassoCV") {
                toggleFieldVisibility(null, 'advancedLassoCVFields', true);
            } else if (selectedModel === "ElasticNetCV") {
                toggleFieldVisibility(null, 'advancedElasticNetCVFields', true);
            } else if (selectedModel === "RidgeCV") {
                toggleFieldVisibility(null, 'advancedRidgeCVFields', true);
            }
        });
    }
    
    const advancedClModelsElement = document.getElementById("advancedClModels");
    if (advancedClModelsElement) {
        advancedClModelsElement.addEventListener("change", function() {
            let selectedModel = this.value;
            
            // Hide all fields (both regular and advanced)
            toggleFieldVisibility('AgglomerativeFields', 'advancedAgglomerativeFields', false);
            toggleFieldVisibility('GaussianFields', 'advancedGaussianFields', false);
            toggleFieldVisibility('KmeansFields', 'advancedKmeansFields', false);
            toggleFieldVisibility('Logistic_classifierFields', 'advancedLogistic_classifierFields', false);
            toggleFieldVisibility('MLP_classifierFields', 'advancedMLP_classifierFields', false);
            toggleFieldVisibility('RF_classifierFields', 'advancedRF_classifierFields', false);
            toggleFieldVisibility('SVC_classifierFields', 'advancedSVC_classifierFields', false);
            toggleFieldVisibility('ridgeFields', 'advancedRidgeFields', false);
            toggleFieldVisibility('lassoFields', 'advancedLassoFields', false);
            toggleFieldVisibility('logisticFields', 'advancedLogisticFields', false);
            toggleFieldVisibility('polynomialFields', null, false);
            toggleFieldVisibility('elasticNetFields', null, false);
            toggleFieldVisibility('SVMFields', 'advancedSVMFields', false);
            toggleFieldVisibility('RFFields', 'advancedRFFields', false);
            toggleFieldVisibility('PerceptronFields', 'advancedPerceptronFields', false);
            toggleFieldVisibility('MLPFields', 'advancedMLPFields', false);
            toggleFieldVisibility('K-NearestFields', 'advancedK-NearestFields', false);
            toggleFieldVisibility('GradientBoostingFields', 'advancedGradientBoostingFields', false);
            // Additional clustering models
            toggleFieldVisibility(null, 'advancedAffinityPropagationFields', false);
            toggleFieldVisibility(null, 'advancedBisectingKmeansFields', false);
            toggleFieldVisibility(null, 'advancedHDBSCANFields', false);
            toggleFieldVisibility(null, 'advancedMeanshiftFields', false);
            toggleFieldVisibility(null, 'advancedMinibatchKmeansFields', false);
            toggleFieldVisibility(null, 'advancedOPTICSFields', false);

            // Show fields based on selection (both regular and advanced)
            if (selectedModel === 'agglo'){
                toggleFieldVisibility('AgglomerativeFields', 'advancedAgglomerativeFields', true);
            }
            else if (selectedModel === 'gmm'){
                toggleFieldVisibility('GaussianFields', 'advancedGaussianFields', true);
            }
            else if (selectedModel === 'kmeans'){
                toggleFieldVisibility('KmeansFields', 'advancedKmeansFields', true);
            }
            else if (selectedModel === 'dbscan'){
                // DBSCAN fields already handled elsewhere
            }
            else if (selectedModel === 'birch'){
                // BIRCH fields already handled elsewhere
            }
            else if (selectedModel === 'spectral'){
                // Spectral fields already handled elsewhere
            }
            else if (selectedModel === 'affinity_propagation'){
                toggleFieldVisibility(null, 'advancedAffinityPropagationFields', true);
            }
            else if (selectedModel === 'bisecting_kmeans'){
                toggleFieldVisibility(null, 'advancedBisectingKmeansFields', true);
            }
            else if (selectedModel === 'hdbscan'){
                toggleFieldVisibility(null, 'advancedHDBSCANFields', true);
            }
            else if (selectedModel === 'meanshift'){
                toggleFieldVisibility(null, 'advancedMeanshiftFields', true);
            }
            else if (selectedModel === 'minibatch_kmeans'){
                toggleFieldVisibility(null, 'advancedMinibatchKmeansFields', true);
            }
            else if (selectedModel === 'optics'){
                toggleFieldVisibility(null, 'advancedOPTICSFields', true);
            }
        });
    }
    
    const advancedClassModelsElement = document.getElementById("advancedClassModels");
    if (advancedClassModelsElement) {
        advancedClassModelsElement.addEventListener("change", function() {
            let selectedModel = this.value;
            
            // Hide all fields (both regular and advanced)
            toggleFieldVisibility('AgglomerativeFields', 'advancedAgglomerativeFields', false);
            toggleFieldVisibility('GaussianFields', 'advancedGaussianFields', false);
            toggleFieldVisibility('KmeansFields', 'advancedKmeansFields', false);
            toggleFieldVisibility('Logistic_classifierFields', 'advancedLogistic_classifierFields', false);
            toggleFieldVisibility('MLP_classifierFields', 'advancedMLP_classifierFields', false);
            toggleFieldVisibility('RF_classifierFields', 'advancedRF_classifierFields', false);
            toggleFieldVisibility('SVC_classifierFields', 'advancedSVC_classifierFields', false);
            toggleFieldVisibility('ridgeFields', 'advancedRidgeFields', false);
            toggleFieldVisibility('lassoFields', 'advancedLassoFields', false);
            toggleFieldVisibility('logisticFields', 'advancedLogisticFields', false);
            toggleFieldVisibility('polynomialFields', null, false);
            toggleFieldVisibility('elasticNetFields', null, false);
            toggleFieldVisibility('SVMFields', 'advancedSVMFields', false);
            toggleFieldVisibility('RFFields', 'advancedRFFields', false);
            toggleFieldVisibility('PerceptronFields', 'advancedPerceptronFields', false);
            toggleFieldVisibility('MLPFields', 'advancedMLPFields', false);
            toggleFieldVisibility('K-NearestFields', 'advancedK-NearestFields', false);
            toggleFieldVisibility('GradientBoostingFields', 'advancedGradientBoostingFields', false);

            // Show fields based on selection (both regular and advanced)
            if (selectedModel === 'Logistic_classifier'){
                toggleFieldVisibility('Logistic_classifierFields', 'advancedLogistic_classifierFields', true);
            }
            else if (selectedModel === 'MLP_classifier'){
                toggleFieldVisibility('MLP_classifierFields', 'advancedMLP_classifierFields', true);
            }
            else if (selectedModel === 'RF_classifier'){
                toggleFieldVisibility('RF_classifierFields', 'advancedRF_classifierFields', true);
            }
            else if (selectedModel === 'SVC_classifier'){
                toggleFieldVisibility('SVC_classifierFields', 'advancedSVC_classifierFields', true);
            }
        });
    }

