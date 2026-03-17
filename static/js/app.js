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
    if (uploadHeader) uploadHeader.textContent = 'Upload Data from CSV';
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
            if (confirm('Are you sure you want to return to the welcome screen? This will reset your current session.')) {
                resetToWelcomeScreen();
            }
        });
        
        // Also handle keyboard navigation (Enter/Space)
        headerLogo.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                if (confirm('Are you sure you want to return to the welcome screen? This will reset your current session.')) {
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
            showError(errorDiv, 'Please select a model first before navigating to Advanced Modeling.');
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
            showError(errorDiv, 'Please select a model first before navigating to Advanced Modeling.');
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
        runButton = document.getElementById('advancedOptimizationSubmitButton');
        loadingDiv = document.getElementById('advancedLoading');
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
