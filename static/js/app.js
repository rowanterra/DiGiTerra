/**
 * App shell: welcome, nav, popups, goToModelPreprocessing fallback. Depends on core.js and feature modules (upload, preprocess, modeling, inference).
 */
function welcomePage(){
    console.log('welcomePage function called');
    try {
        let welcomeDiv = document.getElementById("welcome")
        if (welcomeDiv) {
            welcomeDiv.classList.add("hidden")
        }

        // Show the app tabs toolbar
        if (appTabs) {
            appTabs.classList.remove('hidden');
        }

        // Use showTab to properly show the upload section
        showTab('upload')
    } catch (error) {
        console.error('Error in welcomePage:', error);
    }
}

// Also add event listener as backup in case inline onclick doesn't work
function setupStartModelingButton() {
    const startModelingButton = document.getElementById('startModelingButton');
    if (startModelingButton) {
        // Remove existing onclick to avoid double-firing, use addEventListener instead
        startModelingButton.onclick = null;
        startModelingButton.addEventListener('click', function(e) {
            e.preventDefault();
            e.stopPropagation();
            welcomePage();
        });
    }
}

// Function to reset to welcome screen (full restart when user clicks logo)
function resetToWelcomeScreen() {
    // Show welcome screen
    const welcomeDiv = document.getElementById('welcome');
    if (welcomeDiv) {
        welcomeDiv.classList.remove('hidden');
    }
    
    // Hide app tabs
    const appTabs = document.getElementById('appTabs');
    if (appTabs) {
        appTabs.classList.add('hidden');
    }
    
    // Hide all main sections (use correct id: fileuploaddiv, not fileUpload)
    const fileUploadEl = document.getElementById('fileuploaddiv');
    const documentationSection = document.getElementById('documentationSection');
    const userInputSection = document.getElementById('userInputSection');
    const predictionDiv = document.getElementById('predictionDiv');
    const processingDiv = document.getElementById('processingDiv');
    const modelPreprocessingDiv = document.getElementById('modelPreprocessingDiv');
    const modelingDiv = document.getElementById('modelingDiv');
    
    if (fileUploadEl) fileUploadEl.classList.add('hidden');
    if (documentationSection) documentationSection.classList.add('hidden');
    if (userInputSection) userInputSection.classList.add('hidden');
    if (predictionDiv) predictionDiv.classList.add('hidden');
    if (processingDiv) processingDiv.classList.add('hidden');
    if (modelPreprocessingDiv) modelPreprocessingDiv.classList.add('hidden');
    if (modelingDiv) modelingDiv.classList.add('hidden');
    
    // Restore upload section to initial state so "Start Modeling" shows the upload page again
    const uploadHeader = document.getElementById('uploadHeader');
    if (uploadHeader) uploadHeader.textContent = 'Upload Data';
    const uploadCard = document.querySelector('.upload-card');
    if (uploadCard) uploadCard.classList.remove('section-header');
    const uploadFormEl = document.getElementById('uploadForm');
    if (uploadFormEl) {
        uploadFormEl.classList.remove('hidden');
        uploadFormEl.style.display = '';
    }
    const columnSection = document.getElementById('columnsection');
    if (columnSection) columnSection.classList.add('hidden');
    const explorationOutput = document.getElementById('explorationOutput');
    if (explorationOutput) explorationOutput.innerHTML = '';
    const columnList = document.getElementById('columnList');
    if (columnList) columnList.innerHTML = '<!-- Column headers will be displayed here -->';
    const dataExploration = document.getElementById('dataExploration');
    if (dataExploration) dataExploration.innerHTML = '';
    // Clear client-side state so the app behaves like a fresh start
    uploadedFileName = '';
    
    // Scroll to top
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// Set up header logo click handler
function setupHeaderLogoClick() {
    const headerLogo = document.getElementById('headerLogo');
    if (headerLogo) {
        headerLogo.addEventListener('click', function() {
            if (confirm('Returning to the welcome screen resets the current session. Continue?')) {
                resetToWelcomeScreen();
            }
        });
        
        // Also handle keyboard navigation (Enter/Space)
        headerLogo.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                if (confirm('Returning to the welcome screen resets the current session. Continue?')) {
                    resetToWelcomeScreen();
                }
            }
        });
        
        // Add hover effect
        headerLogo.style.transition = 'opacity 0.2s';
        headerLogo.addEventListener('mouseenter', function() {
            this.style.opacity = '0.8';
        });
        headerLogo.addEventListener('mouseleave', function() {
            this.style.opacity = '1';
        });
    }
}

// Set up button when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function() {
        setupStartModelingButton();
        setupHeaderLogoClick();
        setupContinueToModelPreprocessing();
    });
} else {
    // DOM is already loaded
    setupStartModelingButton();
    setupHeaderLogoClick();
    setupContinueToModelPreprocessing();
}

function setupContinueToModelPreprocessing() {
    // Expose globally so inline onclick on the button can call it (works even if addEventListener fails)
    window.goToModelPreprocessing = function() {
        try {
            if (typeof showTab === 'function') {
                showTab('model-preprocessing');
            }
            var section = document.getElementById('userInputSection');
            var uploadDiv = document.getElementById('fileuploaddiv');
            if (section) {
                section.classList.remove('hidden');
                section.style.display = '';
            }
            if (uploadDiv) {
                uploadDiv.classList.add('hidden');
            }
            var tabs = document.querySelectorAll('.tab-button[data-tab]');
            tabs.forEach(function(b) {
                b.classList.toggle('active', b.dataset.tab === 'model-preprocessing');
            });
            window.scrollTo(0, 0);
        } catch (err) {
            console.error('goToModelPreprocessing failed:', err);
        }
    };

    var btn = document.getElementById('continueToModelPreprocessing');
    if (btn) {
        btn.addEventListener('click', function(e) {
            e.preventDefault();
            e.stopPropagation();
            window.goToModelPreprocessing();
        });
    }
}


/* Called from HTML (onclick) */
/* eslint-disable-next-line no-unused-vars */
function moveToModelPreprocess(){
    showTab('model-preprocessing')
}
    // // opens popup
// function openPopup() {
//     document.getElementById("popup").style.display = "flex";
// }

    //closes the popup for glossary
/* Called from HTML (onclick) */
/* eslint-disable-next-line no-unused-vars */
function closePopup() {
    document.getElementById("popup").style.display = "none";
}
    // opens the 'are you sure you want to restart' popup when user clicks use new dataset
/* Called from HTML (onclick) */
/* eslint-disable-next-line no-unused-vars */
function openResetPopup(){
    document.getElementById("resetPopup").style.display = "flex";
}
    // closes the 'are you sure you want to restart' popup
/* Called from HTML (onclick) */
/* eslint-disable-next-line no-unused-vars */
function closeResetPopup(){
    document.getElementById("resetPopup").style.display = "none";
}

    /// handles 'Change Columns or Output Type' Button - goes back to user input page 
/* eslint-disable-next-line no-unused-vars */
function differentColumns(){
    NumericResultDiv.classList.add("hidden")
    ClusterResultDiv.classList.add('hidden')
    ClassifierResultDiv.classList.add('hidden')
    errorDiv.classList.add("hidden")
    

    let columnDiv = document.getElementById('columnsection');
    columnDiv.classList.add('hidden');
    let fileuploaddiv = document.getElementById('fileuploaddiv');
    fileuploaddiv.classList.add('hidden');
    let userInputSection = document.getElementById('userInputSection');
    userInputSection.classList.remove('hidden');
    
    let columnSelection = document.getElementById('columnSelection');
    columnSelection.dataset.ready = 'false';
    columnSelection.style.display = 'none';
    showTab('model-preprocessing');

    //hide hyperparameters
    let lassoFields = document.getElementById("lassoFields");
    let logisticFields = document.getElementById("logisticFields");
    let polynomialFields = document.getElementById("polynomialFields");
    let elasticNetFields = document.getElementById("elasticNetFields");
    let SVMFields = document.getElementById("SVMFields");
    let RFFields = document.getElementById("RFFields");
    let PerceptronFields = document.getElementById("PerceptronFields");
    let MLPFields = document.getElementById("MLPFields");
    let KNearestFields = document.getElementById("K-NearestFields");
    let GradientBoostingFields = document.getElementById("GradientBoostingFields");
    let Logistic_classifierFields = document.getElementById('Logistic_classifierFields');
    let MLP_classifierFields = document.getElementById('MLP_classifierFields');
    let RF_classifierFields = document.getElementById('RF_classifierFields');
    let SVC_classifierFields = document.getElementById('SVC_classifierFields');
    let ridgeFields = document.getElementById("ridgeFields");
        


    ridgeFields.classList.add("hidden");
    lassoFields.classList.add("hidden");
    logisticFields.classList.add("hidden");
    polynomialFields.classList.add("hidden");
    elasticNetFields.classList.add("hidden");
    SVMFields.classList.add("hidden");
    RFFields.classList.add("hidden");
    PerceptronFields.classList.add("hidden");
    MLPFields.classList.add("hidden");
    KNearestFields.classList.add("hidden");
    GradientBoostingFields.classList.add("hidden");
    Logistic_classifierFields.classList.add("hidden");
    MLP_classifierFields.classList.add("hidden");
    RF_classifierFields.classList.add("hidden");
    SVC_classifierFields.classList.add("hidden");



}
    // when user clicks 'restart' 
/* Called from HTML (onclick) */
/* eslint-disable-next-line no-unused-vars */
function fileUploadPage(){
    location.reload();
}

// function runModelAgain(){
//     fileUpload.classList.add('hidden');
//     NumericResultDiv.classList.add('hidden');
//     ClassifierResultDiv.classList.add('hidden');
//     ClusterResultDiv.classList.add('hidden');
//     columnSelection.style.display = 'block';
//     errorDiv.classList.add('hidden');

// }

//go to prediction page
/* eslint-disable-next-line no-unused-vars */
function predictionPage(){
    if (columnSelection) columnSelection.style.display = 'none';
    const predDiv = getCachedElement('predictionDiv');
    showElement(predDiv);
    showTab('historic');
}
// Reset Inference tab to "ready for new upload" so user can upload a different dataset
function resetInferenceUI() {
    const uploadPredictDf = document.getElementById('uploadPredictDf');
    const predictionResults = document.getElementById('predictionResults');
    const predictionErrorDiv = document.getElementById('predictionErrorDiv');
    const predictFileInput = document.getElementById('predictFile');
    if (uploadPredictDf) {
        uploadPredictDf.classList.remove('hidden');
        uploadPredictDf.style.display = '';
    }
    if (predictionResults) {
        predictionResults.classList.add('hidden');
        predictionResults.innerHTML = '';
    }
    if (predictionErrorDiv) {
        predictionErrorDiv.classList.add('hidden');
        predictionErrorDiv.innerHTML = '';
    }
    if (predictFileInput) {
        predictFileInput.value = '';
    }
}

// Upload new prediction file (e.g. after "Predict Another Dataset")
/* eslint-disable-next-line no-unused-vars */
function newPredict(){
    resetInferenceUI();
}

// goes back to model from prediction page
/* Called from HTML (onclick) */
/* eslint-disable-next-line no-unused-vars */
function backToModel(){
    if (columnSelection) columnSelection.style.display = 'block';
    hideElement(predictionDiv);
    hideElement(predictionResultsDiv);
    showTab('modeling');
}

// Helper function to copy hyperparameter values from Simple to Advanced
function copyHyperparametersToAdvanced(selectedModel, _outputType) {
    // Wait a bit more for hyperparameter fields to be visible after model change
    setTimeout(() => {
        // Define mappings: simpleFieldId -> advancedFieldId
        // Note: Non-essential hyperparameters share the same IDs between Simple and Advanced
        // Only essential ones and sliders need mapping
        const fieldMappings = {};
        
        // Common patterns for different models
        if (selectedModel === 'Ridge') {
            fieldMappings['RidgeAlpha'] = 'advancedRidgeAlpha';
            fieldMappings['nonreqRidgeSlider'] = 'advancedNonreqRidgeSlider';
        } else if (selectedModel === 'Lasso') {
            fieldMappings['LassoAlpha'] = 'advancedLassoAlpha';
            fieldMappings['nonreqLassoSlider'] = 'advancedNonreqLassoSlider';
        } else if (selectedModel === 'SVM') {
            fieldMappings['C'] = 'advancedC';
            fieldMappings['kernel'] = 'advancedKernel';
            fieldMappings['nonreqSVMSlider'] = 'advancedNonreqSVMSlider';
            fieldMappings['svmGamma'] = 'advancedSvmGamma';
            fieldMappings['degree'] = 'advancedDegree';
        } else if (selectedModel === 'RF') {
            fieldMappings['RFn_estimators'] = 'advancedRFn_estimators';
            fieldMappings['nonreqRFSlider'] = 'advancedNonreqRFSlider';
        } else if (selectedModel === 'MLP') {
            fieldMappings['hidden_layer_sizes1'] = 'advancedHidden_layer_sizes1';
            fieldMappings['hidden_layer_sizes2'] = 'advancedHidden_layer_sizes2';
            fieldMappings['hidden_layer_sizes3'] = 'advancedHidden_layer_sizes3';
            fieldMappings['activation'] = 'advancedActivation';
            fieldMappings['nonreqMLPSlider'] = 'advancedNonreqMLPSlider';
        } else if (selectedModel === 'K-Nearest') {
            fieldMappings['KNearest'] = 'advancedKNearest';
            fieldMappings['nonreqKNearestSlider'] = 'advancedNonreqKNearestSlider';
        } else if (selectedModel === 'gradient_boosting') {
            fieldMappings['GBn_estimators'] = 'advancedGBn_estimators';
            fieldMappings['nonreqGBSlider'] = 'advancedNonreqGBSlider';
        } else if (selectedModel === 'Logistic_classifier') {
            fieldMappings['nonreqLogisticClassifierSlider'] = 'advancedNonreqLogisticClassifierSlider';
        } else if (selectedModel === 'MLP_classifier') {
            fieldMappings['Class_hidden_layer_sizes1'] = 'advancedClass_hidden_layer_sizes1';
            fieldMappings['Class_hidden_layer_sizes2'] = 'advancedClass_hidden_layer_sizes2';
            fieldMappings['Class_hidden_layer_sizes3'] = 'advancedClass_hidden_layer_sizes3';
            fieldMappings['Class_activation'] = 'advancedClass_activation';
            fieldMappings['nonreqMLPClassifierSlider'] = 'advancedNonreqMLPClassifierSlider';
        } else if (selectedModel === 'RF_classifier') {
            fieldMappings['Class_RFn_estimators'] = 'advancedClass_RFn_estmators'; // Note: typo in HTML
            fieldMappings['nonreqRFClassifierSlider'] = 'advancedNonreqRFClassifierSlider';
        } else if (selectedModel === 'SVC_classifier') {
            fieldMappings['Class_C'] = 'advancedClass_C';
            fieldMappings['Class_kernel'] = 'advancedClass_kernel';
            fieldMappings['nonreqSVCClassifierSlider'] = 'advancedNonreqSVCClassifierSlider';
        }
        
        // Copy all mapped fields (essential hyperparameters and sliders)
        for (const [simpleId, advancedId] of Object.entries(fieldMappings)) {
            const simpleField = document.getElementById(simpleId);
            const advancedField = document.getElementById(advancedId);
            
            if (simpleField && advancedField) {
                if (simpleField.type === 'checkbox') {
                    // Copy checkbox state
                    advancedField.checked = simpleField.checked;
                    // Trigger change event if it's a slider that shows/hides fields
                    if (simpleId.includes('Slider')) {
                        advancedField.dispatchEvent(new Event('change'));
                    }
                } else if (simpleField.tagName === 'SELECT') {
                    // Copy select value
                    advancedField.value = simpleField.value;
                } else if (simpleField.type === 'number' || simpleField.type === 'text') {
                    // Copy input value
                    advancedField.value = simpleField.value;
                }
            }
        }
        
        // Copy non-essential hyperparameters (they share the same IDs between Simple and Advanced)
        // List of common non-essential field IDs that are shared
        const sharedFieldIds = [
            'RidgeFitIntersept', 'RidgeNormalize', 'RidgeCopyX', 'RidgePositive', 'RidgeMaxIter', 'RidgeTol', 'RidgeSolver',
            'LassoFitIntersept', 'LassoPrecompute', 'LassoCopyX', 'LassoWarmStart', 'LassoSelection', 'LassoMaxIter', 'LassoTol',
            'SVMcoef0', 'SVMCacheSize', 'SVMClassWeight', 'SVMdecisionFunctionShape', 'SVMprobability', 'SVMBreakTies', 'SVMverbose', 'SVMtol',
            'RFoobScore', 'RFCriterion', 'RFmin_weight_fraction_leaf', 'RFMinImpurityDecrease', 'RFMax_depth', 'RFMin_samples_split', 'RFMin_samples_leaf',
            'MLPAlpha', 'MLPBatchSize', 'MLPValidationFraction', 'MLPLearningRate', 'MLPLearningRateInit',
            'metric', 'KNearestMetricParams',
            'GBCriterion', 'GBMax_depth', 'GBMinWeightFractionLeaf', 'GBMinImpurityDecrease', 'GBAlpha',
            'Class_LogisticDual', 'Class_LogisticFitIntercept', 'Class_LogisticSolver', 'Class_LogisticMultiClass', 'Class_LogisticWarmStart', 'Class_CLogistic', 'Class_Logistic_penalty', 'Class_LogisticTol', 'Class_Logisticintercept_scaling', 'Class_LogisticClassWeight',
            'Class_MLPAlpha', 'Class_MLPBatchSize', 'Class_MLPValidationFraction', 'Class_MLPLearningRate', 'Class_MLPLearningRateInit',
            'Class_RFoobScore', 'Class_RFCriterion', 'Class_RFmin_weight_fraction_leaf', 'Class_RFMinImpurityDecrease', 'Class_RFMax_depth', 'Class_RFMin_samples_split', 'Class_RFMin_samples_leaf',
            'Class_SVMcoef0', 'Class_SVMCacheSize', 'Class_SVMClassWeight', 'Class_SVMdecisionFunctionShape', 'Class_SVMprobability', 'Class_SVMBreakTies', 'Class_SVMverbose', 'Class_SVMtol', 'Class_SVMdegree'
        ];
        
        sharedFieldIds.forEach(fieldId => {
            const simpleField = document.getElementById(fieldId);
            const advancedField = document.getElementById(fieldId);
            
            if (simpleField && advancedField) {
                if (simpleField.type === 'checkbox') {
                    advancedField.checked = simpleField.checked;
                } else if (simpleField.tagName === 'SELECT') {
                    advancedField.value = simpleField.value;
                } else if (simpleField.type === 'number' || simpleField.type === 'text') {
                    advancedField.value = simpleField.value;
                }
            }
        });
    }, 300); // Wait a bit longer for fields to be visible
}

// Navigate to Advanced Modeling page with the currently selected model
/* eslint-disable-next-line no-unused-vars */
function navigateToAdvancedWithModel(){
    // Get output type
    const outputType = getCachedElement('outputType1');
    if (!outputType || !outputType.value) {
        const errorDiv = getCachedElement('errorDiv');
        if (errorDiv) {
            showError(errorDiv, 'Select a model before you open Advanced Modeling.');
            errorDiv.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
        return;
    }
    
    // Get the selected model from Simple Modeling page
    let selectedModel = '';
    let advancedModelSelector = null;
    
    if (outputType.value === 'Numeric') {
        const modelSelect = document.getElementById('nModels');
        if (modelSelect && modelSelect.value) {
            selectedModel = modelSelect.value;
            advancedModelSelector = 'advancedNModels';
        }
    } else if (outputType.value === 'Classifier') {
        const modelSelect = document.getElementById('classModels');
        if (modelSelect && modelSelect.value) {
            selectedModel = modelSelect.value;
            advancedModelSelector = 'advancedClassModels';
        }
    } else if (outputType.value === 'Cluster') {
        const modelSelect = document.getElementById('clModels');
        if (modelSelect && modelSelect.value) {
            selectedModel = modelSelect.value;
            advancedModelSelector = 'advancedClModels';
        }
    }
    
    if (!selectedModel) {
        const errorDiv = getCachedElement('errorDiv');
        if (errorDiv) {
            showError(errorDiv, 'Select a model before you open Advanced Modeling.');
            errorDiv.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
        return;
    }
    
    // Navigate to unified Modeling page and switch to Advanced mode
    showTab('modeling');
    
    // Switch to Advanced mode and set the model after a short delay
    setTimeout(() => {
        // Switch to Advanced mode
        const advancedModeRadio = document.getElementById('advancedMode');
        if (advancedModeRadio) {
            advancedModeRadio.checked = true;
            switchModelingMode('advanced');
        }
        
        // Update output type display to show correct model selectors
        updateOutputTypeDisplay(outputType.value);
        
        // Set the selected model
        if (advancedModelSelector) {
            const advancedSelect = document.getElementById(advancedModelSelector);
            if (advancedSelect) {
                // Map model names if needed (some models might have different names)
                let modelValue = selectedModel;
                
                // Handle model name mappings if any
                if (selectedModel === 'TerraFORMER' && outputType.value === 'Numeric') {
                    // TerraFORMER might not be available on Advanced page, use Linear as fallback
                    modelValue = 'Linear';
                }
                
                // Check if the option exists
                const optionExists = Array.from(advancedSelect.options).some(opt => opt.value === modelValue);
                if (optionExists) {
                    advancedSelect.value = modelValue;
                    // Trigger change event to show hyperparameters
                    advancedSelect.dispatchEvent(new Event('change'));
                    
                    // Copy hyperparameters from Simple to Advanced
                    copyHyperparametersToAdvanced(selectedModel, outputType.value);
                } else {
                    // If exact match doesn't exist, try to find a similar one or use first available
                    console.log(`Model ${modelValue} not found in Advanced page, using first available option`);
                }
            }
        }
    }, 100);
}

// Global variables for progress tracking (set/read by modeling.js)
let progressEventSource = null;
/* eslint-disable-next-line no-unused-vars */
let sessionId = null;
/* eslint-disable-next-line no-unused-vars */
let processResultData = null;

// Function to stop current model run (called from modeling.js stop buttons)
/* eslint-disable-next-line no-unused-vars */
function stopModelRun() {
    // Close EventSource connection
    if (progressEventSource) {
        progressEventSource.close();
        progressEventSource = null;
    }
    
    // Determine current mode
    const simpleMode = document.getElementById('simpleMode');
    const advancedMode = document.getElementById('advancedMode');
    const automlMode = document.getElementById('automlMode');
    const currentMode = simpleMode?.checked ? 'simple' : (advancedMode?.checked ? 'advanced' : (automlMode?.checked ? 'automl' : 'simple'));
    
    // Hide stop button and show appropriate loading message
    let stopButton, runButton, loadingDiv;
    if (currentMode === 'automl') {
        stopButton = document.getElementById('stopAutomlButton');
        runButton = document.getElementById('automlSubmitButton');
        loadingDiv = document.getElementById('automlLoading');
    } else if (currentMode === 'advanced') {
        stopButton = document.getElementById('stopAdvancedButton');
        runButton = getAdvancedRunSubmitButton();
        loadingDiv = getAdvancedLoadingDiv();
    } else {
        stopButton = document.getElementById('stopSimpleButton');
        runButton = getCachedElement('processButton');
        loadingDiv = getCachedElement('loading');
    }
    
    if (stopButton) stopButton.style.display = 'none';
    if (runButton) {
        runButton.disabled = false;
        if (currentMode === 'automl') {
            runButton.textContent = 'Run AutoML';
        } else if (currentMode === 'advanced') {
            runButton.textContent = 'Run Model with Advanced Options';
        } else {
            runButton.textContent = 'Run This Model';
        }
    }
    
    if (loadingDiv) {
        loadingDiv.innerHTML = `
            <p style="color: #d32f2f; font-weight: 600; margin-bottom: 8px;">Model Run Stopped</p>
            <p style="color: #666;">The model training has been cancelled. You can start a new model run.</p>
        `;
    }
}

// Handle column selection and processing

/// Section 6: running the model
    //after user selects model and hyperparameters

// Force light mode (dark mode disabled)
(function() {
    document.documentElement.classList.remove('dark-mode');
    document.body.classList.remove('dark-mode');
})();

/* ==========================================================================
   Hover explanations on the classic page (review of 2026-09-27).

   The guided page explains every control on hover of its label and on keyboard
   focus of the control. The classic page had none. The same mechanism is used
   here: the words of a label take a dotted underline, the explanation opens on
   hover or focus, stays open while the pointer is over it, and closes on Escape
   (WCAG 1.4.13). Page controls are explained by id; hyperparameter fields by the
   scikit-learn name their label carries ("Max Depth (int or None):" reads as
   max_depth), so one sentence serves every model that has the parameter.
   ========================================================================== */
(function classicHelp() {
    const HELP = {
        file: 'A table with one header row. CSV, Excel (.xlsx, .xlsm) and Parquet are read. Nothing is sent anywhere except this server.',
        corrCols: 'The columns the statistics, correlation matrix and pairplot read, as letters: "C-F" or "B,D,F". Text columns are skipped by the correlation.',
        exploreDropMissing: 'Whether rows with a blank in the explored columns are left out of the statistics.',
        exploreZeroHandle: 'What to do with zeros in the explored columns, for data where zero means not measured.',
        outputType1: 'Classification predicts a class label. Regression predicts a number. Clustering groups rows without a target. The example sets this choice. Change it if the proposal does not fit the data.',
        indicators: 'The columns the model reads, as spreadsheet letters: "B-G" or "B,C,E". Identifier columns and anything that leaks the target should be left out.',
        predictors: 'The column the model predicts, as a letter. A text column or a number with few distinct values is treated as a class. Other numbers are treated as a quantity.',
        useTransformer: 'Text columns cannot enter an estimator as they are. One-hot encoding turns each text value into its own 0/1 column. The page detects text columns itself. This setting overrides the detection.',
        transformerColumn: 'The text columns to one-hot encode, as letters.',
        imputeStrategy: 'What to do with a blank in a modeling column. The row is removed, or the blank is filled with the column mean, median or mode of the training rows.',
        zeroHandle: 'What to do with zeros in the modeling columns, for data where zero means not measured: keep them, remove the rows, or fill them.',
        scalingYesNo: 'Keeps the mix of one column the same in both halves of the split. A rare class or a site is then not left out of the test rows by chance.',
        specificVariableSelect: 'The column whose mix the split preserves.',
        quantileBins: 'A numeric column is stratified by binning it first: equal-count quantiles, or thresholds typed in the box.',
        scaler: 'Rescales the predictors before fitting. Trees do not need it. Distance and linear models often do. Standard centers to mean zero and unit variance, min-max maps to 0 to 1, and robust uses the median and quartiles.',
        seedValue: 'The random seed. The same seed and settings give the same split and the same fit. It is written to the run record.',
        trainSize: 'The fraction of rows the model is fitted on. The rest is held out as the test set. 0.80 keeps one row in five for testing.',
        testSize: 'The fraction held out for the test score. It follows the train size.',
        simpleClassModels: 'The estimator to fit, from scikit-learn. Every model in the list is trained through the same route.',
        simpleRegModels: 'The estimator to fit, from scikit-learn. Every model in the list is trained through the same route.',
        simpleClusterModels: 'The clustering method. The number of groups is a setting of the model.',
        units: 'Written beside the metrics in the result and the model card. It changes no number. "u*" is read as the micro sign.',
        sigfig: 'Decimal places shown for the metrics. The predictions file is always written at full precision.',
        predictFile: 'A table with the same predictor columns as the training file. Predictions are written with a novelty score and a note for rows outside the training range.',
    };
    const PARAM_HELP = {
        n_estimators: 'Number of trees or rounds. More is slower and usually a little better, with diminishing returns.',
        max_depth: 'Deepest a tree may grow. Smaller values fit less closely to the training rows.',
        n_neighbors: 'How many nearest rows vote for a prediction. Small values follow the training rows closely.',
        weights: 'With uniform, every neighbor has one vote. With distance, nearer neighbors count more.',
        learning_rate: 'How much each round or step changes the model. Smaller needs more rounds.',
        learning_rate_init: 'The starting step size of the optimizer.',
        c: 'Inverse regularization strength. Smaller values regularize more.',
        alpha: 'Regularization strength. Larger values regularize more.',
        l1_ratio: 'Mix of L1 and L2 penalty: 0 is ridge, 1 is lasso.',
        max_iter: 'Cap on solver iterations.',
        max_iterations: 'Cap on solver iterations.',
        tol: 'Stopping tolerance for the solver.',
        random_state: 'Seed for this estimator. The run seed is used when blank.',
        class_weight: 'Reweights classes. The balanced option counters an uneven class mix.',
        criterion: 'The split quality measure the trees use.',
        splitter: 'How each split is chosen: the best split, or a random one among the best.',
        min_samples_leaf: 'Fewest rows a leaf may contain. Larger values smooth the tree.',
        min_samples_split: 'Fewest rows a node needs before it can split.',
        max_features: 'How many predictors each split considers. Fewer predictors decorrelate the trees.',
        max_leaf_nodes: 'Cap on leaves per tree. Blank means no cap.',
        min_impurity_decrease: 'A split must reduce impurity by at least this much.',
        ccp_alpha: 'Cost-complexity pruning strength. Larger values prune more.',
        bootstrap: 'Whether each tree is fitted on a resample of the rows.',
        oob_score: 'Scores each tree on the rows its resample left out. This gives a validation estimate without a separate split.',
        kernel: 'The kernel function of the support vector machine.',
        gamma: 'Kernel width for rbf, poly and sigmoid kernels.',
        degree: 'Degree of the polynomial kernel or features.',
        solver: 'The optimization routine. The default suits most data.',
        penalty: 'The regularization type.',
        n_jobs: 'CPU cores to use. Blank means one.',
        p: 'Power of the Minkowski distance: 1 is Manhattan, 2 is Euclidean.',
        metric: 'The distance measure between rows.',
        hidden_layer_sizes: 'Neurons in each hidden layer, one box per layer. Blank boxes are unused layers.',
        activation: 'The nonlinearity between layers. The relu option suits most data.',
        batch_size: 'Rows per optimizer step. The auto option uses min(200, rows).',
        momentum: 'Momentum of the sgd solver.',
        nesterovs_momentum: 'Uses the Nesterov form of momentum with the sgd solver.',
        early_stopping: 'Holds out a fraction of the training rows and stops when its score stops improving.',
        validation_fraction: 'The fraction of training rows early stopping holds out.',
        beta_1: 'Decay rate of the first-moment estimate in adam.',
        beta_2: 'Decay rate of the second-moment estimate in adam.',
        epsilon: 'Numerical stability term of adam, or the tube width of epsilon-insensitive loss.',
        shuffle: 'Whether rows are shuffled each iteration.',
        warm_start: 'Continues from the previous fit instead of starting over.',
        verbose: 'Prints solver progress to the server log. The result is unchanged.',
        power_t: 'Exponent of the inverse-scaling learning rate schedule.',
        n_clusters: 'Number of groups to form.',
        n_init: 'How many times the clustering is started from new centers. The best run is kept.',
        init: 'How the starting centres are chosen.',
        eps: 'Neighbourhood radius of DBSCAN.',
        min_samples: 'Rows a neighbourhood needs to form a core point.',
        loss: 'The loss the model minimizes.',
        n_components: 'Number of components kept.',
        fit_intercept: 'Whether an intercept is fitted.',
        positive: 'Constrains the coefficients to be non-negative.',
        subsample: 'Fraction of rows each round is fitted on. Values below 1 add randomness.',
        damping: 'Damping factor of affinity propagation, between 0.5 and 1.',
        linkage: 'How the distance between clusters is measured when they merge.',
        threshold: 'Radius of a subcluster in Birch.',
        branching_factor: 'Maximum subclusters per node in Birch.',
        quantile: 'Quantile the estimator targets or uses for its bandwidth.',
        nu: 'Upper bound on the fraction of margin errors and support vectors.',
        cv: 'Number of cross-validation folds the estimator uses internally.',
    };
    let tipCounter = 0;
    // Chrome marks a text field :focus-visible after a mouse click too, so the
    // focus tip opens only when the focus came from the keyboard. Otherwise the
    // tip covers the buttons under the field while a person types.
    let focusFromKeyboard = false;
    document.addEventListener('keydown', (ev) => { if (ev.key === 'Tab') focusFromKeyboard = true; }, true);
    document.addEventListener('pointerdown', () => { focusFromKeyboard = false; }, true);
    function attachHelp(target, text, control) {
        if (!target || !text || target.classList.contains('tipword')) return null;
        const id = 'tip' + (++tipCounter);
        const tip = document.createElement('span');
        tip.className = 'tip';
        tip.id = id;
        tip.setAttribute('role', 'tooltip');
        tip.textContent = text;
        const wrap = document.createElement('span');
        wrap.className = 'helpwrap';
        wrap.appendChild(tip);
        target.classList.add('tipword');
        target.insertAdjacentElement('afterend', wrap);
        const open = () => {
            wrap.classList.remove('dismissed');
            wrap.classList.add('open');
        };
        const close = () => wrap.classList.remove('open');
        target.addEventListener('mouseenter', open);
        target.addEventListener('mouseleave', (ev) => { if (!wrap.contains(ev.relatedTarget)) close(); });
        wrap.addEventListener('mouseleave', (ev) => { if (ev.relatedTarget !== target && !target.contains(ev.relatedTarget)) close(); });
        if (control) {
            control.addEventListener('focusin', () => { if (focusFromKeyboard) open(); });
            control.addEventListener('focusout', (ev) => { if (!control.contains(ev.relatedTarget)) close(); });
            control.setAttribute('aria-describedby', id);
        }
        return id;
    }
    document.addEventListener('keydown', (ev) => {
        if (ev.key !== 'Escape') return;
        document.querySelectorAll('.helpwrap.open').forEach((w) => { w.classList.remove('open'); w.classList.add('dismissed'); });
    });
    function paramName(labelText) {
        let t = String(labelText).trim().replace(/:\s*$/, '');
        t = t.split(' - ')[0].split(' (')[0].trim();
        return t.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_|_$/g, '');
    }
    function init() {
        Object.keys(HELP).forEach((id) => {
            const ctl = document.getElementById(id);
            if (!ctl) return;
            // Buttons get no hover tip: an open tip would sit over the button.
            const label = document.querySelector('label[for="' + id + '"]');
            if (label) attachHelp(label, HELP[id], ctl);
        });
        // A label or heading that carries its explanation in the markup as
        // data-help shows it on hover in place of a note under the control.
        document.querySelectorAll('[data-help]').forEach((el) => {
            const ctl = el.htmlFor ? document.getElementById(el.htmlFor) : null;
            attachHelp(el, el.dataset.help, ctl);
        });
        // Hyperparameter labels read as one line: the name alone. The descriptor
        // the markup carried ("# of trees (integer >=1)") moves into the hover
        // explanation, after the sentence for the parameter when there is one.
        document.querySelectorAll('[id$="Fields"] > label, [id^="nonreq"][id$="Params"] > label, [id^="advancedNonreq"][id$="Params"] > label, [id^="advanced"][id$="Fields"] label').forEach((label) => {
            if (label.querySelector('input, select')) return;
            const full = label.textContent.trim().replace(/:\s*$/, '');
            const name = paramName(full);
            const m = full.match(/^(.*?)(?:\s+-\s+|\s+\()(.*)$/);
            const short = m ? m[1].trim() : full;
            let descriptor = m ? full.slice(short.length).trim().replace(/^-\s*/, '') : '';
            descriptor = descriptor.replace(/^\((.*)\)$/, '$1');
            if (short && short !== full) label.textContent = short;
            const sentence = PARAM_HELP[name] || PARAM_HELP[name.replace(/^class_/, '')] || '';
            const text = [sentence, descriptor ? 'Accepts: ' + descriptor + '.' : ''].filter(Boolean).join(' ');
            if (!text) return;
            const ctl = label.htmlFor ? document.getElementById(label.htmlFor) : null;
            attachHelp(label, text, ctl);
        });
    }
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
    else init();
})();
