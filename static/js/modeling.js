/**
 * Modeling tab: run model, process results, hyperparameters. Depends on core.js, formatDateTimeForFilename, downloadFile, etc.
 */
processForm.addEventListener('submit', async (e) => {
    e.preventDefault();

    // Determine current mode
    const simpleModeRadio = document.getElementById('simpleMode');
    const advancedModeRadio = document.getElementById('advancedMode');
    const automlModeRadio = document.getElementById('automlMode');
    const currentMode = simpleModeRadio?.checked ? 'simple' : (advancedModeRadio?.checked ? 'advanced' : (automlModeRadio?.checked ? 'automl' : 'simple'));
    
    // Disable appropriate button and show stop button
    let runButton, stopButton;
    if (currentMode === 'automl') {
        runButton = document.getElementById('automlSubmitButton');
        stopButton = document.getElementById('stopAutomlButton');
    } else if (currentMode === 'advanced') {
        runButton = document.getElementById('advancedOptimizationSubmitButton');
        stopButton = document.getElementById('stopAdvancedButton');
    } else {
        runButton = getCachedElement('processButton');
        stopButton = document.getElementById('stopSimpleButton');
    }
    
    if (runButton) runButton.disabled = true;
    if (stopButton) stopButton.style.display = 'inline-block';

    const indicators = indicatorsSelect.value
    const predictors = predictorsSelect.value
    const outputType = getCachedElement('outputType1')
    const _models = getCachedElement('models');
    const scaler = getCachedElement('scaler');
    const seedValue = getCachedElement('seedValue')?.value || '';
    const testSize = getCachedElement('testSize')?.value || '';
    // Get units and sigfig based on current mode
    let units = '';
    let sigfig = '';
    if (currentMode === 'automl') {
        units = getCachedElement('automlUnitName')?.value || '';
        sigfig = getCachedElement('automlSigfig')?.value || '';
    } else if (currentMode === 'advanced') {
        units = getCachedElement('advancedUnitName')?.value || '';
        sigfig = getCachedElement('advancedSigfig')?.value || '';
    } else {
        units = getCachedElement('unitName')?.value || '';
        sigfig = getCachedElement('sigfig')?.value || '';
    }
    const stratifyColumn = getCachedElement('specificVariableSelect')?.value || '';
    const dropMissing = getCachedElement('dropMissing')?.value || '';
    const dropZero = getCachedElement('drop0')?.value || '';
    const imputeStrategy = getCachedElement('imputeStrategy')?.value || '';
    const quantileBins = getCachedElement('quantileBins')?.value || '';
    const useTransformer = getCachedElement('useTransformer')?.value || '';
    
    // Check if we're on Advanced Modeling or AutoML mode - use advanced cross-validation fields
    // Note: currentMode is already defined above
    // Simple Modeling page should never send cross-validation (it's removed from that page)
    const isAdvancedPage = currentMode === 'advanced' || currentMode === 'automl';
    const crossValidationType = isAdvancedPage 
        ? (getCachedElement('advancedCrossValidationType')?.value || (currentMode === 'automl' ? 'KFold' : ''))
        : 'None'; // Simple Modeling page always uses 'None' for cross-validation
    const crossValidationFolds = isAdvancedPage
        ? (getCachedElement('advancedCrossValidationFolds')?.value || (currentMode === 'automl' ? '5' : ''))
        : 5; // Default value, won't be used since type is 'None'
    
    // AutoML mode: set automatic defaults
    if (currentMode === 'automl') {
        // AutoML will use these defaults - they're set in the form or will be set by backend
    }
    
    //creating stratifying dictionary to send to the backend with the number of quantiles/bins
    if (quantileBins=='None'){
        quantileBinDict = {
            quantile: 0,
            bin: 0
        }
    }
    else if (quantileBins=='quantiles'){
        const quantiles = document.getElementById('quantiles').value;
        quantileBinDict = {
            quantile: parseInt(quantiles),
            bin: 0
        }
    }
    else if (quantileBins=='Bins'){
        const bins = document.getElementById('bins').value;
        const binsLabel = document.getElementById('binsLabel').value
        quantileBinDict = {
            quantile: 0,
            bin: bins,
            binsLabel: binsLabel
        }
    }

    //ensure predictors and indicators - validate BEFORE clearing results
    let selectedOutputType = outputType.value
    if (!indicators.length || (!predictors.length && selectedOutputType !== 'Cluster')) {
        showError(NumericResultDiv, 'Please select at least one predictor and one indicator column.');
        const processButton = getCachedElement('processButton');
        if (processButton) processButton.disabled = false; // Re-enable button on validation failure
        return;
    }
    
    // Clear old results only after validation passes
    // Hide all result divs - processModelResult will show the appropriate one based on output type
    NumericResultDiv.innerHTML = '';
    ClusterResultDiv.innerHTML = '';
    ClassifierResultDiv.innerHTML = '';
    NumericResultDiv.classList.add('hidden');
    ClusterResultDiv.classList.add('hidden');
    ClassifierResultDiv.classList.add('hidden');
    
    // Show placeholder when clearing results - use mode-specific placeholder
    let resultsPlaceholder = null;
    if (currentMode === 'simple') {
        resultsPlaceholder = document.getElementById('resultsPlaceholder');
    } else if (currentMode === 'advanced') {
        resultsPlaceholder = document.getElementById('advancedResultsPlaceholder');
    } else if (currentMode === 'automl') {
        resultsPlaceholder = document.getElementById('automlResultsPlaceholder');
    }
    if (resultsPlaceholder) resultsPlaceholder.style.display = 'block';
    
    //loading graphic - will be replaced by detailed progress bars
    // Always show and reset loading when starting a new model run
    // Check current mode and use appropriate loading div (reuse currentMode from above)
    let loadingDiv = null;
    if (currentMode === 'advanced') {
        loadingDiv = document.getElementById('advancedLoading');
    } else if (currentMode === 'automl') {
        loadingDiv = document.getElementById('automlLoading');
    } else {
        loadingDiv = loading;
    }
    if (loadingDiv) {
        loadingDiv.classList.remove('hidden');
        const modeText = currentMode === 'automl' ? 'AutoML' : (currentMode === 'advanced' ? 'Advanced Modeling' : 'Modeling');
        loadingDiv.innerHTML = `
            <p style="font-size: 1.1em; font-weight: 600; margin-bottom: 12px; color: #357a53;">Initializing ${modeText}...</p>
            <p style="color: #666; margin-bottom: 16px;">Setting up model configuration and preparing data...</p>
            <div class="spinner" style="margin: 0 auto;"></div>
        `;
    }
    
    // Note: currentMode is already defined above, no need to redeclare


    //setting variables to send to the back end
    const predictorCols = getColumnIndices(predictors.toUpperCase().replace(/\s/g, ""));
    const indicatorCols = getColumnIndices(indicators.toUpperCase().replace(/\s/g, ""));

    if (stratifyColumn !== ''){
        stratifyColumnNumber = columnToIndex(stratifyColumn.toUpperCase())
        stratifyBool = true
    }
    else {
        stratifyColumnNumber = ''
        stratifyBool = false
    }

    if (stratifyBool && selectedOutputType !== 'Cluster' && predictorCols.indexOf(stratifyColumnNumber) !== -1) {
        showError(NumericResultDiv, 'Do not stratify by your target column. That would leak target information into the train/test split. Choose a different column to stratify by.', false);
        const processButton = getCachedElement('processButton');
        if (processButton) processButton.disabled = false;
        if (loadingDiv) loadingDiv.classList.add('hidden');
        return;
    }

    let transformerCols = []
    if (useTransformer === 'Yes'){
        const transformerColumnElement = document.getElementById('transformerColumn');
        if (transformerColumnElement && transformerColumnElement.value) {
            const transformerColumn = transformerColumnElement.value;
            transformerCols = getColumnIndices(transformerColumn.toUpperCase().replace(/\s/g, ""));
        }
    }


    
    //get output type to send to backend
    // Check which mode is selected - use appropriate model selectors
    let selectedModel1 = 'TerraFORMER'
    let nonreq=false
    if (currentMode === 'advanced') {
        // Use advanced model selectors - use section dropdowns directly
        if (selectedOutputType === 'Numeric'){
            model1 = document.getElementById('advancedNModels');
            selectedModel1 = model1 ? model1.value : ''
        }
        else if (selectedOutputType === 'Classifier'){
            model1 = document.getElementById('advancedClassModels');
            selectedModel1 = model1 ? model1.value : ''
        }
        else if (selectedOutputType === 'Cluster'){
            model1 = document.getElementById('advancedClModels');
            selectedModel1 = model1 ? model1.value : ''
        }
    } else if (currentMode === 'automl') {
        // Use AutoML model selectors (optional - can be empty to let AutoML choose)
        if (selectedOutputType === 'Numeric'){
            model1 = document.getElementById('automlNModels');
            selectedModel1 = model1 ? model1.value : ''
            // If no model selected in AutoML, use a sensible default (Random Forest is robust and works well with AutoML features)
            if (!selectedModel1 || selectedModel1 === '') {
                selectedModel1 = 'RF';
            }
        }
        else if (selectedOutputType === 'Classifier'){
            model1 = document.getElementById('automlClassModels');
            selectedModel1 = model1 ? model1.value : ''
            // If no model selected in AutoML, use a sensible default (Random Forest Classifier)
            if (!selectedModel1 || selectedModel1 === '') {
                selectedModel1 = 'RF_classifier';
            }
        }
        else if (selectedOutputType === 'Cluster'){
            model1 = document.getElementById('automlClModels');
            selectedModel1 = model1 ? model1.value : ''
            // If no model selected in AutoML, use a sensible default (KMeans is a good starting point)
            if (!selectedModel1 || selectedModel1 === '') {
                selectedModel1 = 'kmeans';
            }
        }
    } else {
        // Use regular (Simple) model selectors - use mode-specific dropdowns
        if (selectedOutputType === 'Numeric'){
            model1 = document.getElementById('simpleNModels');
            selectedModel1 = model1 ? model1.value : ''
        }
        else if (selectedOutputType === 'Classifier'){
            model1 = document.getElementById('simpleClassModels');
            selectedModel1 = model1 ? model1.value : ''
        }
        else if (selectedOutputType === 'Cluster'){
            model1 = document.getElementById('simpleClModels');
            selectedModel1 = model1 ? model1.value : ''
        }
    }
    
    //getting the hyperparameters based on the model 
    //each if statement checks if nonrequired hyperparameters are selected
        //if they are then adds them to the hyperparameters dictionary to send to the backend
        //if no nonreqs are changed then only send the required values to the backend
        //the backend will check if the nonreq bool is true or false
    let selectedModel = selectedModel1
    let hyperparameters = {}
    if (selectedModel === "TerraFORMER") {

    } 
    else if (selectedModel === "Linear") {
        
    } 
    else if (selectedModel === "Ridge") {
        // Get essential hyperparameters - check advanced page first, then regular
        const alpha = isAdvancedPage 
            ? document.getElementById('advancedRidgeAlpha')
            : document.getElementById('RidgeAlpha');
        if (alpha) {
            hyperparameters['alpha'] = parseFloat(alpha.value);
        }

        // Check for advanced slider if on advanced page, otherwise use regular slider
        let nonreqRidgeSlider = isAdvancedPage 
            ? document.getElementById('advancedNonreqRidgeSlider')
            : document.getElementById('nonreqRidgeSlider');
        if (nonreqRidgeSlider && nonreqRidgeSlider.checked){
            nonreq=true
            let RidgeFitIntersept = document.getElementById('RidgeFitIntersept').value
            let RidgeNormalize = document.getElementById('RidgeNormalize').value
            let RidgeCopyX = document.getElementById('RidgeCopyX').value
            let RidgePositive = document.getElementById('RidgePositive').value
            let RidgeMaxIter = document.getElementById('RidgeMaxIter').value
            let RidgeTol = document.getElementById('RidgeTol').value
            const solver = document.getElementById('RidgeSolver');
            hyperparameters['solver'] = solver.value
            //let RidgeRandomState = document.getElementById('RidgeRandomState').value
            hyperparameters['RidgeFitIntersept'] = RidgeFitIntersept
            hyperparameters['RidgeNormalize'] = RidgeNormalize
            hyperparameters['RidgeCopyX'] = RidgeCopyX
            hyperparameters['RidgePositive'] = RidgePositive
            hyperparameters['RidgeMaxIter'] = parseInt(RidgeMaxIter)
            hyperparameters['RidgeTol'] = parseFloat(RidgeTol)
            //hyperparameters['RidgeRandomState'] = parseInt(RidgeRandomState)
        }
        
    } 
    else if (selectedModel === "Lasso") {
        // Get essential hyperparameters - check both pages
        const alpha = document.getElementById('LassoAlpha');
        if (alpha) {
            hyperparameters['alpha'] = parseFloat(alpha.value);
        }

        // Check for advanced slider if on advanced page, otherwise use regular slider
        let nonreqLassoSlider = isAdvancedPage 
            ? document.getElementById('advancedNonreqLassoSlider')
            : document.getElementById('nonreqLassoSlider');
        if (nonreqLassoSlider && nonreqLassoSlider.checked){
            nonreq=true
            let LassoFitIntersept = document.getElementById('LassoFitIntersept').value
            let LassoPrecompute = document.getElementById('LassoPrecompute').value
            let LassoCopyX = document.getElementById('LassoCopyX').value
            let LassoWarmStart = document.getElementById('LassoWarmStart').value
            let LassoPositive = document.getElementById('LassoPositive').value
            let LassoTol = document.getElementById('LassoTol').value
            //let LassoRandomState = document.getElementById('LassoRandomState').value
            let LassoSelection = document.getElementById('LassoSelection').value
            const max_iter = document.getElementById('LassoMax_iter');
            hyperparameters['LassoFitIntersept'] = LassoFitIntersept
            hyperparameters['LassoPrecompute'] = LassoPrecompute
            hyperparameters['LassoCopyX'] = LassoCopyX
            hyperparameters['LassoWarmStart'] = LassoWarmStart
            hyperparameters['LassoPositive'] = LassoPositive
            hyperparameters['LassoTol'] = parseFloat(LassoTol)
            //hyperparameters['LassoRandomState'] = parseInt(LassoRandomState)
            hyperparameters['LassoSelection'] = LassoSelection
            hyperparameters['max_iter'] = parseFloat(max_iter.value)
            
        }

    } 
    else if (selectedModel === "Logistic") {
        // Check for advanced slider if on advanced page, otherwise use regular slider
        let nonreqLogisticSlider = isAdvancedPage 
            ? document.getElementById('advancedNonreqLogisticSlider')
            : document.getElementById('nonreqLogisticSlider');
        if (nonreqLogisticSlider && nonreqLogisticSlider.checked){
            nonreq=true
            let LogisticDual = document.getElementById('LogisticDual').value
            let LogisticFitIntercept = document.getElementById('LogisticFitIntercept').value
            let LogisticSolver = document.getElementById('LogisticSolver').value
            let LogisticMultiClass = document.getElementById('LogisticMultiClass').value
            let LogisticWarmStart = document.getElementById('LogisticWarmStart').value
            let LogisticTol = document.getElementById('LogisticTol').value
            let Logisticintercept_scaling = document.getElementById('Logisticintercept_scaling').value
            let LogisticClassWeight = document.getElementById('LogisticClassWeight').value
            //let LogisticRandomState = document.getElementById('LogisticRandomState').value
            let LogisticMaxIterations = document.getElementById('LogisticMaxIterations').value
            let LogisticVerbose = document.getElementById('LogisticVerbose').value
            let LogisticNJobs = document.getElementById('LogisticNJobs').value
            let Logisticl1Ratio = document.getElementById('Logisticl1Ratio').value

            const CLogistic = document.getElementById('CLogistic');
            const penatly = document.getElementById('penalty');
            hyperparameters['C'] = parseFloat(CLogistic.value);
            hyperparameters['penalty'] = penatly.value;
            
            hyperparameters['LogisticDual'] = LogisticDual
            hyperparameters['LogisticFitIntercept'] = LogisticFitIntercept
            hyperparameters['LogisticSolver'] = LogisticSolver
            hyperparameters['LogisticMultiClass'] = LogisticMultiClass
            hyperparameters['LogisticWarmStart'] = LogisticWarmStart
            hyperparameters['LogisticTol'] = parseFloat(LogisticTol)
            hyperparameters['Logisticintercept_scaling'] = parseFloat(Logisticintercept_scaling)
            hyperparameters['LogisticClassWeight'] = LogisticClassWeight
            //hyperparameters['LogisticRandomState'] = parseInt(LogisticRandomState)
            hyperparameters['LogisticMaxIterations'] = parseInt(LogisticMaxIterations)
            hyperparameters['LogisticVerbose'] = parseInt(LogisticVerbose)
            hyperparameters['LogisticNJobs'] = parseInt(LogisticNJobs)
            hyperparameters['Logisticl1Ratio'] = parseFloat(Logisticl1Ratio)
            
            
        }
    } 
    else if (selectedModel === "Polynomial") {
        const degree_specificity = document.getElementById('degree_specificity');
        hyperparameters['degree_specificity'] = parseFloat(degree_specificity.value)

    } 
    else if (selectedModel === "ElasticNet") {
        const alpha = document.getElementById('ENAlpha');
        const l1_ratio = document.getElementById('l1_ratio');
        hyperparameters['alpha'] = parseFloat(alpha.value)
        hyperparameters['l1_ratio'] = parseFloat(l1_ratio.value)

    } 
    else if (selectedModel === "SVM") {
        // Get essential hyperparameters - check advanced page first, then regular
        const C = isAdvancedPage 
            ? document.getElementById('advancedC')
            : document.getElementById('C');
        const kernel = isAdvancedPage 
            ? document.getElementById('advancedKernel')
            : document.getElementById('kernel');
        if (C) hyperparameters['C'] = parseFloat(C.value);
        if (kernel) {
            hyperparameters['kernel'] = kernel.value;
            if (kernel.value === 'poly'){
                const degree = isAdvancedPage 
                    ? document.getElementById('advancedPolyDegree')
                    : document.getElementById('polyDegree');
                if (degree) hyperparameters['degree'] = parseFloat(degree.value);
                const gamma = isAdvancedPage 
                    ? document.getElementById('advancedGamma')
                    : document.getElementById('Gamma');
                if (gamma) {
                    if (gamma.value === 'scale' || gamma.value === 'auto'){
                        hyperparameters['gamma'] = gamma.value;
                    }
                    else {
                        hyperparameters['gamma'] = parseFloat(gamma.value);
                    }
                }
            }
            else if (kernel.value === 'rbf'){
                const gamma = isAdvancedPage 
                    ? document.getElementById('advancedGamma')
                    : document.getElementById('Gamma');
                if (gamma) {
                    if (gamma.value === 'scale' || gamma.value === 'auto'){
                        hyperparameters['gamma'] = gamma.value;
                    }
                    else {
                        hyperparameters['gamma'] = parseFloat(gamma.value);
                    }
                }
            }
        }

        // Check for advanced slider if on advanced page, otherwise use regular slider
        let nonreqSVMSlider = isAdvancedPage 
            ? document.getElementById('advancedNonreqSVMSlider')
            : document.getElementById('nonreqSVMSlider');
        if (nonreqSVMSlider && nonreqSVMSlider.checked){
            nonreq=true
            let SVMshrinking = document.getElementById('SVMshrinking').value
            let SVMprobability = document.getElementById('SVMprobability').value
            let SVMBreakTies = document.getElementById('SVMBreakTies').value
            let SVMverbose = document.getElementById('SVMverbose').value
            let SVMcoef0 = document.getElementById('SVMcoef0').value
            let SVMtol = document.getElementById('SVMtol').value
            let SVMCacheSize = document.getElementById('SVMCacheSize').value
            let SVMClassWeight = document.getElementById('SVMClassWeight').value
            let SVMmaxIter = document.getElementById('SVMmaxIter').value
            let SVMdecisionFunctionShape = document.getElementById('SVMdecisionFunctionShape').value
            //let SVMrandomState = document.getElementById('SVMrandomState').value
            
            hyperparameters['SVMshrinking'] = SVMshrinking
            hyperparameters['SVMprobability'] = SVMprobability
            hyperparameters['SVMBreakTies'] = SVMBreakTies
            hyperparameters['SVMverbose'] = SVMverbose
            hyperparameters['SVMcoef0'] = parseFloat(SVMcoef0)
            hyperparameters['SVMtol'] = parseFloat(SVMtol)
            hyperparameters['SVMCacheSize'] = parseFloat(SVMCacheSize)
            hyperparameters['SVMClassWeight'] = SVMClassWeight
            hyperparameters['SVMmaxIter'] = parseInt(SVMmaxIter)
            hyperparameters['SVMdecisionFunctionShape'] = SVMdecisionFunctionShape
            //hyperparameters['SVMrandomState'] = SVMrandomState
            
        }
        
    } 
    else if (selectedModel === "RF") {
        // Get essential hyperparameters - check advanced page first, then regular
        const n_estimators = isAdvancedPage 
            ? document.getElementById('advancedRFn_estmators')
            : document.getElementById('RFn_estmators');
        if (n_estimators) {
            hyperparameters['n_estimators'] = parseFloat(n_estimators.value);
        }
        

        // Check for advanced slider if on advanced page, otherwise use regular slider
        let nonreqRFSlider = isAdvancedPage 
            ? document.getElementById('advancedNonreqRFSlider')
            : document.getElementById('nonreqRFSlider');
        if (nonreqRFSlider && nonreqRFSlider.checked){
            nonreq=true
            let RFBoostrap = document.getElementById('RFBoostrap').value
            let RFoobScore = document.getElementById('RFoobScore').value
            let RFWarmStart = document.getElementById('RFWarmStart').value
            let RFmin_weight_fraction_leaf = document.getElementById('RFmin_weight_fraction_leaf').value
            let RFMaxLeafNodes = document.getElementById('RFMaxLeafNodes').value
            let RFMinImpurityDecrease = document.getElementById('RFMinImpurityDecrease').value
            let RFNJobs = document.getElementById('RFNJobs').value
            //let RFRandomState = document.getElementById('RFRandomState').value
            let RFVerbose  = document.getElementById('RFVerbose').value

            const min_samples_split = document.getElementById('min_samples_split');
            const min_samples_leaf = document.getElementById('min_samples_leaf');
            const max_depth = document.getElementById('RFMax_depth');
            if (max_depth !== null) {
                hyperparameters['max_depth'] = parseFloat(max_depth.value)
            }
            hyperparameters['min_samples_split'] = parseFloat(min_samples_split.value)
            hyperparameters['min_samples_leaf'] = parseFloat(min_samples_leaf.value)
            
            hyperparameters['RFBoostrap'] = RFBoostrap
            hyperparameters['RFoobScore'] = RFoobScore
            hyperparameters['RFWarmStart'] = RFWarmStart
            hyperparameters['RFmin_weight_fraction_leaf'] = parseFloat(RFmin_weight_fraction_leaf)
            hyperparameters['RFMaxLeafNodes'] = parseInt(RFMaxLeafNodes)
            hyperparameters['RFMinImpurityDecrease'] = parseFloat(RFMinImpurityDecrease)
            hyperparameters['RFNJobs'] = parseInt(RFNJobs)
            //hyperparameters['RFRandomState'] = parseInt(RFRandomState)
            hyperparameters['RFVerbose'] = parseInt(RFVerbose)
            
        }
        
    } 
    else if (selectedModel === "Perceptron") {
        const max_iter = document.getElementById('PercMax_iter');
        const eta0 = document.getElementById('eta0');
        hyperparameters['max_iter'] = parseFloat(max_iter.value)
        hyperparameters['eta0'] = parseFloat(eta0.value)

        let nonreqPerceptronSlider = document.getElementById('nonreqPerceptronSlider')
        if (nonreqPerceptronSlider.checked){
            nonreq=true
            let PerceptronFitIntercept = document.getElementById('').value
            let PerceptronShuffle = document.getElementById('').value
            let PerceptronEarlyStopping = document.getElementById('').value
            let PerceptronWarmStart = document.getElementById('').value
            let PerceptronPenalty = document.getElementById('').value
            let PerceptronAlpha = document.getElementById('').value
            let PerceptronTol = document.getElementById('').value
            let PerceptronVerbose = document.getElementById('').value
            let PerceptronNJobs = document.getElementById('').value
            //let PerceptronRandomState = document.getElementById('').value
            let PerceptronValidationFraction = document.getElementById('').value
            let PerceptronNIterNoChange = document.getElementById('').value
            let PerceptronClassWeight = document.getElementById('').value
            
            hyperparameters['PerceptronFitIntercept'] = PerceptronFitIntercept
            hyperparameters['PerceptronShuffle'] = PerceptronShuffle
            hyperparameters['PerceptronEarlyStopping'] = PerceptronEarlyStopping
            hyperparameters['PerceptronWarmStart'] = PerceptronWarmStart
            hyperparameters['PerceptronPenalty'] = PerceptronPenalty
            hyperparameters['PerceptronAlpha'] = parseFloat(PerceptronAlpha)
            hyperparameters['PerceptronTol'] = parseFloat(PerceptronTol)
            hyperparameters['PerceptronVerbose'] = parseInt(PerceptronVerbose)
            hyperparameters['PerceptronNJobs'] = parseInt(PerceptronNJobs)
            //hyperparameters['PerceptronRandomState'] = parseInt(PerceptronRandomState)
            hyperparameters['PerceptronValidationFraction'] = parseFloat(PerceptronValidationFraction)
            hyperparameters['PerceptronNIterNoChange'] = parseInt(PerceptronNIterNoChange)
            hyperparameters['PerceptronClassWeight'] = PerceptronClassWeight
            
            
        }
        
    } 
    else if (selectedModel === "MLP") {
        const hidden_layer_size1 = document.getElementById('hidden_layer_sizes1');
        const hidden_layer_sizes2 = document.getElementById('hidden_layer_sizes2');
        const hidden_layer_sizes3 = document.getElementById('hidden_layer_sizes3');
        const activation = document.getElementById('activation');
        const solver = document.getElementById('MLPSolver');
        
        hyperparameters['hidden_layer_sizes1'] = hidden_layer_size1.value
        hyperparameters['hidden_layer_sizes2'] = hidden_layer_sizes2.value
        hyperparameters['hidden_layer_sizes3'] = hidden_layer_sizes3.value
        hyperparameters['activation'] = activation.value
        hyperparameters['solver'] = solver.value

        let nonreqMLPSlider = document.getElementById('nonreqMLPSlider')
        if (nonreqMLPSlider.checked){
            nonreq=true
            let MLPNesterovsMomentum = document.getElementById('MLPNesterovsMomentum').value
            let MLPEarlyStopping = document.getElementById('MLPEarlyStopping').value
            let MLPShuffle = document.getElementById('MLPShuffle').value
            let MLPVerbose = document.getElementById('MLPVerbose').value
            let MLPWarmStart = document.getElementById('MLPWarmStart').value
            let MLPBatchSize = document.getElementById('MLPBatchSize').value
            let MLPLearningRateInit = document.getElementById('MLPLearningRateInit').value
            let MLPPowerT = document.getElementById('MLPPowerT').value
            let MLPMaxIter = document.getElementById('MLPMaxIter').value
            //let MLPRandomState = document.getElementById('MLPRandomState').value
            let MLPTol = document.getElementById('MLPTol').value
            let MLPMomentum = document.getElementById('MLPMomentum').value
            let MLPValidationFraction = document.getElementById('MLPValidationFraction').value
            let MLPBeta1 = document.getElementById('MLPBeta1').value
            let MLPBeta2 = document.getElementById('MLPBeta2').value
            let MLPEpsilon = document.getElementById('MLPEpsilon').value
            const alpha = document.getElementById('MLPAlpha');
            const learning_rate = document.getElementById('MLPLearning_rate');
            
            hyperparameters['learning_rate'] = learning_rate.value
            hyperparameters['alpha'] = parseFloat(alpha.value)
            hyperparameters['MLPNesterovsMomentum'] = MLPNesterovsMomentum
            hyperparameters['MLPEarlyStopping'] = MLPEarlyStopping
            hyperparameters['MLPShuffle'] = MLPShuffle
            hyperparameters['MLPVerbose'] = MLPVerbose
            hyperparameters['MLPWarmStart'] = MLPWarmStart
            if (MLPBatchSize == 'auto'){
                hyperparameters['MLPBatchSize'] = MLPBatchSize //auto or int
            }
            else{
                hyperparameters['MLPBatchSize'] = parseInt(MLPBatchSize) //auto or int
            }
            hyperparameters['MLPLearningRateInit'] = parseFloat(MLPLearningRateInit)
            hyperparameters['MLPPowerT'] = parseFloat(MLPPowerT)
            hyperparameters['MLPMaxIter'] = parseInt(MLPMaxIter)
            //hyperparameters['MLPRandomState'] = parseInt(MLPRandomState)
            hyperparameters['MLPTol'] = parseFloat(MLPTol)
            hyperparameters['MLPMomentum'] = parseFloat(MLPMomentum)
            hyperparameters['MLPValidationFraction'] = parseFloat(MLPValidationFraction)
            hyperparameters['MLPBeta1'] = parseFloat(MLPBeta1)
            hyperparameters['MLPBeta2'] = parseFloat(MLPBeta2)
            hyperparameters['MLPEpsilon'] = parseFloat(MLPEpsilon)
            
        }
        
    } 
    else if (selectedModel === "K-Nearest") {
        // Get essential hyperparameters - check advanced page first, then regular
        const n_neighbors = isAdvancedPage 
            ? document.getElementById('advancedN_neighbors')
            : document.getElementById('n_neighbors');
        if (n_neighbors) {
            hyperparameters['n_neighbors'] = parseFloat(n_neighbors.value);
        }

        let nonreqKNearestSlider = document.getElementById('nonreqKNearestSlider')
        if (nonreqKNearestSlider.checked){
            nonreq=true
            let KNearestWeights = document.getElementById('KNearestWeights').value
            let KNearestAlgorithm = document.getElementById('KNearestAlgorithm').value
            let KNearestLeafSize = document.getElementById('KNearestLeafSize').value
            let KNearestP = document.getElementById('KNearestP').value
            let KNearestMetricParams = document.getElementById('KNearestMetricParams').value
            let KNearestNJobs = document.getElementById('KNearestNJobs').value
            const metric = document.getElementById('metric');
            
            hyperparameters['metric'] = metric.value
            hyperparameters['KNearestWeights'] = KNearestWeights
            hyperparameters['KNearestAlgorithm'] = KNearestAlgorithm
            hyperparameters['KNearestLeafSize'] = parseInt(KNearestLeafSize)
            hyperparameters['KNearestP'] = parseInt(KNearestP)
            hyperparameters['KNearestMetricParams'] = KNearestMetricParams
            hyperparameters['KNearestNJobs'] = parseInt(KNearestNJobs)
            
        }
        
    } 
    else if (selectedModel === "gradient_boosting") {
        // Get essential hyperparameters - check advanced page first, then regular
        const n_estimators = isAdvancedPage 
            ? document.getElementById('advancedGBn_estimators')
            : document.getElementById('GBn_estimators');
        const learning_rate = isAdvancedPage 
            ? document.getElementById('advancedGBlearn')
            : document.getElementById('GBlearn');
        if (n_estimators) hyperparameters['n_estimators'] = parseFloat(n_estimators.value);
        if (learning_rate) hyperparameters['learning_rate'] = parseFloat(learning_rate.value);
    

        let nonreqGBSlider = document.getElementById('nonreqGBSlider')
        if (nonreqGBSlider.checked){
            nonreq=true
            let GBLoss = document.getElementById('GBLoss').value
            let GBWarmStart = document.getElementById('GBWarmStart').value
            let GBCriterion = document.getElementById('GBCriterion').value
            let GBSubsample = document.getElementById('GBSubsample').value
            let GBMinSamplesSplit = document.getElementById('GBMinSamplesSplit').value
            let GBMinSamplesLeaf = document.getElementById('GBMinSamplesLeaf').value
            let GBMinWeightFractionLeaf = document.getElementById('GBMinWeightFractionLeaf').value
            let GBMinImpurityDecrease = document.getElementById('GBMinImpurityDecrease').value
            let GBInit = document.getElementById('GBInit').value
            //let GBRandomState = document.getElementById('GBRandomState').value
            let GBMaxFeatrues = document.getElementById('GBMaxFeatrues').value
            let GBAlpha = document.getElementById('GBAlpha').value
            let GBVerbose = document.getElementById('GBVerbose').value
            let GBMaxLeafNodes = document.getElementById('GBMaxLeafNodes').value
            const max_depth = document.getElementById('GBMax_depth');
            
            hyperparameters['max_depth'] = parseFloat(max_depth.value)
            hyperparameters['GBLoss'] = GBLoss
            hyperparameters['GBWarmStart'] = GBWarmStart
            hyperparameters['GBCriterion'] = GBCriterion
            hyperparameters['GBSubsample'] = parseFloat(GBSubsample)
            hyperparameters['GBMinSamplesSplit'] = parseFloat(GBMinSamplesSplit)
            hyperparameters['GBMinSamplesLeaf'] = parseFloat(GBMinSamplesLeaf)
            hyperparameters['GBMinWeightFractionLeaf'] = parseFloat(GBMinWeightFractionLeaf)
            hyperparameters['GBMinImpurityDecrease'] = parseFloat(GBMinImpurityDecrease)
            hyperparameters['GBInit'] = GBInit
            //hyperparameters['GBRandomState'] = parseInt(GBRandomState)
            hyperparameters['GBMaxFeatrues'] = GBMaxFeatrues //int, float, or string
            hyperparameters['GBAlpha'] = parseFloat(GBAlpha)
            hyperparameters['GBVerbose'] = parseInt(GBVerbose)
            hyperparameters['GBMaxLeafNodes'] = parseInt(GBMaxLeafNodes)
            
        }
    }

    else if (selectedModel === "Logistic_classifier"){
        let nonreqLogisticClassifierSlider = document.getElementById('nonreqLogisticClassifierSlider')
        if (nonreqLogisticClassifierSlider.checked){
            nonreq=true
            let Class_LogisticDual = document.getElementById('Class_LogisticDual').value
            let Class_LogisticFitIntercept = document.getElementById('Class_LogisticFitIntercept').value
            let Class_LogisticSolver = document.getElementById('Class_LogisticSolver').value
            let Class_LogisticMultiClass = document.getElementById('Class_LogisticMultiClass').value
            let Class_LogisticWarmStart = document.getElementById('Class_LogisticWarmStart').value
            let Class_CLogistic = document.getElementById('Class_CLogistic').value
            let Class_Logistic_penalty = document.getElementById('Class_Logistic_penalty').value
            let Class_LogisticTol = document.getElementById('Class_LogisticTol').value
            let Class_Logisticintercept_scaling = document.getElementById('Class_Logisticintercept_scaling').value
            let Class_LogisticClassWeight = document.getElementById('Class_LogisticClassWeight').value
            let Class_LogisticMaxIterations = document.getElementById('Class_LogisticMaxIterations').value
            let Class_LogisticVerbose = document.getElementById('Class_LogisticVerbose').value
            let Class_LogisticNJobs = document.getElementById('Class_LogisticNJobs').value
            let Class_Logisticl1Ratio = document.getElementById('Class_Logisticl1Ratio').value

            hyperparameters['Class_LogisticDual'] = Class_LogisticDual
            hyperparameters['Class_LogisticFitIntercept'] = Class_LogisticFitIntercept
            hyperparameters['Class_LogisticSolver'] = Class_LogisticSolver
            hyperparameters['Class_LogisticMultiClass'] = Class_LogisticMultiClass
            hyperparameters['Class_LogisticWarmStart'] = Class_LogisticWarmStart
            hyperparameters['Class_CLogistic'] = parseFloat(Class_CLogistic)
            hyperparameters['Class_Logistic_penalty'] = Class_Logistic_penalty
            hyperparameters['Class_LogisticTol'] = parseFloat(Class_LogisticTol)
            hyperparameters['Class_Logisticintercept_scaling'] = parseInt(Class_Logisticintercept_scaling)
            hyperparameters['Class_LogisticClassWeight'] = Class_LogisticClassWeight
            hyperparameters['Class_LogisticMaxIterations'] = parseInt(Class_LogisticMaxIterations)
            hyperparameters['Class_LogisticVerbose'] = parseInt(Class_LogisticVerbose)
            hyperparameters['Class_LogisticNJobs'] = parseInt(Class_LogisticNJobs)
            hyperparameters['Class_Logisticl1Ratio'] = parseFloat(Class_Logisticl1Ratio)

        }
    }
    else if (selectedModel === "MLP_classifier"){
        const hidden_layer_sizes1 = document.getElementById('Class_hidden_layer_sizes1');
        const hidden_layer_sizes2 = document.getElementById('Class_hidden_layer_sizes2');
        const hidden_layer_sizes3 = document.getElementById('Class_hidden_layer_sizes3');
        const activation = document.getElementById('Class_activation');
        const solver = document.getElementById('Class_MLPSolver');
        
        hyperparameters['hidden_layer_sizes1'] = hidden_layer_sizes1.value
        hyperparameters['hidden_layer_sizes2'] = hidden_layer_sizes2.value
        hyperparameters['hidden_layer_sizes3'] = hidden_layer_sizes3.value
        hyperparameters['activation'] = activation.value
        hyperparameters['solver'] = solver.value

        let nonreqMLPSlider = document.getElementById('nonreqMLPClassifierSlider')
        if (nonreqMLPSlider.checked){
            nonreq=true
            let MLPNesterovsMomentum = document.getElementById('Class_MLPNesterovsMomentum').value
            let MLPEarlyStopping = document.getElementById('Class_MLPEarlyStopping').value
            let MLPShuffle = document.getElementById('Class_MLPShuffle').value
            let MLPVerbose = document.getElementById('Class_MLPVerbose').value
            let MLPWarmStart = document.getElementById('Class_MLPWarmStart').value
            let MLPBatchSize = document.getElementById('Class_MLPBatchSize').value
            let MLPLearningRateInit = document.getElementById('Class_MLPLearningRateInit').value
            let MLPPowerT = document.getElementById('Class_MLPPowerT').value
            let MLPMaxIter = document.getElementById('Class_MLPMaxIter').value
            //let MLPRandomState = document.getElementById('MLPRandomState').value
            let MLPTol = document.getElementById('Class_MLPTol').value
            let MLPMomentum = document.getElementById('Class_MLPMomentum').value
            let MLPValidationFraction = document.getElementById('Class_MLPValidationFraction').value
            let MLPBeta1 = document.getElementById('Class_MLPBeta1').value
            let MLPBeta2 = document.getElementById('Class_MLPBeta2').value
            let MLPEpsilon = document.getElementById('Class_MLPEpsilon').value
            const alpha = document.getElementById('Class_MLPAlpha');
            const learning_rate = document.getElementById('Class_MLPLearning_rate');
            
            hyperparameters['learning_rate'] = learning_rate.value
            hyperparameters['alpha'] = parseFloat(alpha.value)
            hyperparameters['MLPNesterovsMomentum'] = MLPNesterovsMomentum
            hyperparameters['MLPEarlyStopping'] = MLPEarlyStopping
            hyperparameters['MLPShuffle'] = MLPShuffle
            hyperparameters['MLPVerbose'] = MLPVerbose
            hyperparameters['MLPWarmStart'] = MLPWarmStart
            if (MLPBatchSize == 'auto'){
                hyperparameters['MLPBatchSize'] = MLPBatchSize //auto or int
            }
            else{
                hyperparameters['MLPBatchSize'] = parseInt(MLPBatchSize) //auto or int
            }
            hyperparameters['MLPLearningRateInit'] = parseFloat(MLPLearningRateInit)
            hyperparameters['MLPPowerT'] = parseFloat(MLPPowerT)
            hyperparameters['MLPMaxIter'] = parseInt(MLPMaxIter)
            //hyperparameters['MLPRandomState'] = parseInt(MLPRandomState)
            hyperparameters['MLPTol'] = parseFloat(MLPTol)
            hyperparameters['MLPMomentum'] = parseFloat(MLPMomentum)
            hyperparameters['MLPValidationFraction'] = parseFloat(MLPValidationFraction)
            hyperparameters['MLPBeta1'] = parseFloat(MLPBeta1)
            hyperparameters['MLPBeta2'] = parseFloat(MLPBeta2)
            hyperparameters['MLPEpsilon'] = parseFloat(MLPEpsilon)
            
        }  
    }
    
    else if (selectedModel === "RF_classifier"){
        // Get essential hyperparameters - check advanced page first, then regular
        let n_estimators = isAdvancedPage 
            ? document.getElementById('advancedClass_RFn_estmators')
            : document.getElementById('Class_RFn_estmators');
        if (n_estimators) {
            hyperparameters['n_estimators'] = parseFloat(n_estimators.value);
        }
        

        let nonreqRFSlider = document.getElementById('nonreqRFClassifierSlider')
        if (nonreqRFSlider.checked){
            nonreq=true
            let RFBoostrap = document.getElementById('Class_RFBoostrap').value
            let RFoobScore = document.getElementById('Class_RFoobScore').value
            let RFWarmStart = document.getElementById('Class_RFWarmStart').value
            let RFmin_weight_fraction_leaf = document.getElementById('Class_RFmin_weight_fraction_leaf').value
            let RFMaxLeafNodes = document.getElementById('Class_RFMaxLeafNodes').value
            let RFMinImpurityDecrease = document.getElementById('Class_RFMinImpurityDecrease').value
            let RFNJobs = document.getElementById('Class_RFNJobs').value
            //let RFRandomState = document.getElementById('RFRandomState').value
            let RFVerbose  = document.getElementById('Class_RFVerbose').value

            const min_samples_split = document.getElementById('Class_min_samples_split');
            const min_samples_leaf = document.getElementById('Class_min_samples_leaf');
            const max_depth = document.getElementById('Class_RFMax_depth');
            if (max_depth !== null) {
                hyperparameters['max_depth'] = parseFloat(max_depth.value)
            }
            hyperparameters['min_samples_split'] = parseFloat(min_samples_split.value)
            hyperparameters['min_samples_leaf'] = parseFloat(min_samples_leaf.value)
            
            hyperparameters['RFBoostrap'] = RFBoostrap
            hyperparameters['RFoobScore'] = RFoobScore
            hyperparameters['RFWarmStart'] = RFWarmStart
            hyperparameters['RFmin_weight_fraction_leaf'] = parseFloat(RFmin_weight_fraction_leaf)
            hyperparameters['RFMaxLeafNodes'] = parseInt(RFMaxLeafNodes)
            hyperparameters['RFMinImpurityDecrease'] = parseFloat(RFMinImpurityDecrease)
            hyperparameters['RFNJobs'] = parseInt(RFNJobs)
            //hyperparameters['RFRandomState'] = parseInt(RFRandomState)
            hyperparameters['RFVerbose'] = parseInt(RFVerbose)
        }
    }
    
    else if (selectedModel === "SVC_classifier"){
        const C = document.getElementById('SVC_C');
        const kernel = document.getElementById('Class_kernel');
        hyperparameters['C'] = parseFloat(C.value)
        hyperparameters['kernel'] = kernel.value
        if (kernel.value === 'poly'){
            const degree = document.getElementById('Class_polyDegree');
            hyperparameters['degree'] = parseFloat(degree.value)
            const gamma = document.getElementById('SVCGamma');
            if (gamma.value === 'scale' || gamma.value === 'auto'){
                hyperparameters['gamma'] = gamma.value;
            }
            else {
                hyperparameters['gamma'] = parseFloat(gamma.value);
            }
        }
        else if (kernel.value === 'rbf'){
            const gamma = document.getElementById('SVCGamma');
            if (gamma.value === 'scale' || gamma.value === 'auto'){
                hyperparameters['gamma'] = gamma.value;
            }
            else {
                hyperparameters['gamma'] = parseFloat(gamma.value);
            }
        }

        let nonreqSVMSlider = document.getElementById('nonreqSVCClassifierSlider')
        if (nonreqSVMSlider.checked){
            nonreq=true
            let SVCshrinking = document.getElementById('SVCshrinking').value
            let SVCprobability = document.getElementById('SVCprobability').value
            let SVCBreakTies = document.getElementById('SVCBreakTies').value
            let SVCverbose = document.getElementById('SVCverbose').value
            let SVCcoef0 = document.getElementById('SVCcoef0').value
            let SVCtol = document.getElementById('SVCtol').value
            let SVCCacheSize = document.getElementById('SVCCacheSize').value
            let SVCClassWeight = document.getElementById('SVCClassWeight').value
            let SVCmaxIter = document.getElementById('SVCmaxIter').value
            let SVCdecisionFunctionShape = document.getElementById('SVCdecisionFunctionShape').value
            //let SVMrandomState = document.getElementById('SVMrandomState').value
            
            hyperparameters['SVCshrinking'] = SVCshrinking
            hyperparameters['SVCprobability'] = SVCprobability
            hyperparameters['SVCBreakTies'] = SVCBreakTies
            hyperparameters['SVCverbose'] = SVCverbose
            hyperparameters['SVCcoef0'] = parseFloat(SVCcoef0)
            hyperparameters['SVCtol'] = parseFloat(SVCtol)
            hyperparameters['SVCCacheSize'] = parseFloat(SVCCacheSize)
            hyperparameters['SVCClassWeight'] = SVCClassWeight.trim() === '' ? null : SVCClassWeight.trim()
            hyperparameters['SVCmaxIter'] = parseInt(SVCmaxIter)
            hyperparameters['SVCdecisionFunctionShape'] = SVCdecisionFunctionShape
            //hyperparameters['SVMrandomState'] = SVMrandomState
            
        }
    }

    else if (selectedModel === 'agglo'){
        const n_clusters = document.getElementById('Agg_n_clusters').value
        hyperparameters['n_clusters'] = parseInt(n_clusters)
        
        let nonreqAgglomerativeSlider = document.getElementById('nonreqAgglomerativeSlider')
        if (nonreqAgglomerativeSlider.checked){
            nonreq=true
            let metric = document.getElementById('Aggmetric').value
            let memory = document.getElementById('Aggmemory').value
            let connectivity = document.getElementById('Aggconnectivity').value
            let compute_full_tree = document.getElementById('aggcompute_full_tree').value
            let linkage = document.getElementById('Agglinkage').value
            let distance_threshold = document.getElementById('Aggdistance_threshold').value
            let compute_distances = document.getElementById('aggcompute_distances').value

            hyperparameters['metric']=metric
            hyperparameters['memory']=memory
            hyperparameters['connectivity']=connectivity
            hyperparameters['compute_full_tree']=compute_full_tree
            hyperparameters['linkage']=linkage
            hyperparameters['distance_threshold']=parseFloat(distance_threshold)
            hyperparameters['compute_distances']=compute_distances

        }

    }

    else if (selectedModel === 'gmm'){
        const n_components = document.getElementById('Gaun_components').value
        hyperparameters['n_components'] = parseInt(n_components)
        
        let nonreqGaussianSlider = document.getElementById('nonreqGaussianSlider')
        if (nonreqGaussianSlider.checked){
            nonreq=true
            let covariance_type = document.getElementById('Gaucovariance_type').value
            let tol = document.getElementById('GauTol').value
            let reg_covar = document.getElementById('Gaureg_covar').value
            let max_iter = document.getElementById('GauMax_iter').value
            let n_init = document.getElementById('Gaun_init').value
            let init_params = document.getElementById('Gauinit_params').value
            let weights_init = document.getElementById('Gauweights_init').value
            let means_init = document.getElementById('Gaumeans_init').value
            let precisions_init = document.getElementById('Gauprecisions_init').value
            let warm_start = document.getElementById('GauWarmStart').value
            let verbose = document.getElementById('GauVerbose').value
            let verbose_interval = document.getElementById('GauVerbose_interval').value

            hyperparameters['covariance_type']=covariance_type
            hyperparameters['tol']=parseFloat(tol)
            hyperparameters['reg_covar']=parseFloat(reg_covar)
            hyperparameters['max_iter']=parseInt(max_iter)
            hyperparameters['n_init']=parseInt(n_init)
            hyperparameters['init_params']=init_params
            hyperparameters['weights_init']=weights_init
            hyperparameters['means_init']=means_init
            hyperparameters['precisions_init']=precisions_init
            hyperparameters['warm_start']=warm_start
            hyperparameters['verbose']=parseInt(verbose)
            hyperparameters['verbose_interval']=parseInt(verbose_interval)

        }
    }

    else if (selectedModel === 'kmeans'){
        const n_clusters = document.getElementById('Kmeansn_clusters').value
        hyperparameters['n_clusters'] = parseInt(n_clusters)
        
        let nonreqKmeansSlider = document.getElementById('nonreqKmeansSlider')
        if (nonreqKmeansSlider.checked){
            nonreq=true
            let init = document.getElementById('kmeansInit').value
            let n_init = document.getElementById('kmeansn_init').value
            let max_iter = document.getElementById('kmeansmax_iter').value
            let tol = document.getElementById('kmeanstol').value
            let verbose = document.getElementById('kmeansverbose').value
            let copy_x = document.getElementById('kmeansCopyX').value
            let algorithm = document.getElementById('Kmeansalgorithm').value

            hyperparameters['init'] = init
            hyperparameters['n_init'] = n_init
            hyperparameters['max_iter'] = parseInt(max_iter)
            hyperparameters['tol']= parseFloat(tol)
            hyperparameters['verbose'] = parseInt(verbose)
            hyperparameters['copy_x'] = copy_x
            hyperparameters['algorithm'] = algorithm

        }
    }
    
    // Additional regression models hyperparameter collection
    else if (selectedModel === "AdaBoost") {
        const n_estimators = isAdvancedPage 
            ? document.getElementById('advancedAdaBoostNEstimators')
            : null;
        const learning_rate = isAdvancedPage 
            ? document.getElementById('advancedAdaBoostLearningRate')
            : null;
        if (n_estimators) hyperparameters['n_estimators'] = parseInt(n_estimators.value);
        if (learning_rate) hyperparameters['learning_rate'] = parseFloat(learning_rate.value);
        
        const nonreqSlider = isAdvancedPage 
            ? document.getElementById('advancedNonreqAdaBoostSlider')
            : null;
        if (nonreqSlider && nonreqSlider.checked) {
            nonreq = true;
            const loss = document.getElementById('AdaBoostLoss');
            if (loss) hyperparameters['loss'] = loss.value;
        }
    }
    else if (selectedModel === "Bagging") {
        const n_estimators = isAdvancedPage 
            ? document.getElementById('advancedBaggingNEstimators')
            : null;
        if (n_estimators) hyperparameters['n_estimators'] = parseInt(n_estimators.value);
        
        const nonreqSlider = isAdvancedPage 
            ? document.getElementById('advancedNonreqBaggingSlider')
            : null;
        if (nonreqSlider && nonreqSlider.checked) {
            nonreq = true;
            const maxSamples = document.getElementById('BaggingMaxSamples');
            const maxFeatures = document.getElementById('BaggingMaxFeatures');
            const bootstrap = document.getElementById('BaggingBootstrap');
            const bootstrapFeatures = document.getElementById('BaggingBootstrapFeatures');
            const oobScore = document.getElementById('BaggingOobScore');
            const warmStart = document.getElementById('BaggingWarmStart');
            const nJobs = document.getElementById('BaggingNJobs');
            const verbose = document.getElementById('BaggingVerbose');
            if (maxSamples) hyperparameters['max_samples'] = maxSamples.value;
            if (maxFeatures) hyperparameters['max_features'] = maxFeatures.value;
            if (bootstrap) hyperparameters['bootstrap'] = bootstrap.value === 'true';
            if (bootstrapFeatures) hyperparameters['bootstrap_features'] = bootstrapFeatures.value === 'true';
            if (oobScore) hyperparameters['oob_score'] = oobScore.value === 'true';
            if (warmStart) hyperparameters['warm_start'] = warmStart.value === 'true';
            if (nJobs && nJobs.value) hyperparameters['n_jobs'] = parseInt(nJobs.value);
            if (verbose) hyperparameters['verbose'] = parseInt(verbose.value);
        }
    }
    else if (selectedModel === "DecisionTree") {
        const nonreqSlider = isAdvancedPage 
            ? document.getElementById('advancedNonreqDecisionTreeSlider')
            : null;
        if (nonreqSlider && nonreqSlider.checked) {
            nonreq = true;
            const criterion = document.getElementById('DecisionTreeCriterion');
            const splitter = document.getElementById('DecisionTreeSplitter');
            const maxDepth = document.getElementById('DecisionTreeMaxDepth');
            const minSamplesSplit = document.getElementById('DecisionTreeMinSamplesSplit');
            const minSamplesLeaf = document.getElementById('DecisionTreeMinSamplesLeaf');
            const minWeightFractionLeaf = document.getElementById('DecisionTreeMinWeightFractionLeaf');
            const maxFeatures = document.getElementById('DecisionTreeMaxFeatures');
            const maxLeafNodes = document.getElementById('DecisionTreeMaxLeafNodes');
            const minImpurityDecrease = document.getElementById('DecisionTreeMinImpurityDecrease');
            const ccpAlpha = document.getElementById('DecisionTreeCcpAlpha');
            if (criterion) hyperparameters['criterion'] = criterion.value;
            if (splitter) hyperparameters['splitter'] = splitter.value;
            if (maxDepth && maxDepth.value) hyperparameters['max_depth'] = parseInt(maxDepth.value);
            if (minSamplesSplit) hyperparameters['min_samples_split'] = parseFloat(minSamplesSplit.value);
            if (minSamplesLeaf) hyperparameters['min_samples_leaf'] = parseFloat(minSamplesLeaf.value);
            if (minWeightFractionLeaf) hyperparameters['min_weight_fraction_leaf'] = parseFloat(minWeightFractionLeaf.value);
            if (maxFeatures && maxFeatures.value) hyperparameters['max_features'] = maxFeatures.value;
            if (maxLeafNodes && maxLeafNodes.value) hyperparameters['max_leaf_nodes'] = parseInt(maxLeafNodes.value);
            if (minImpurityDecrease) hyperparameters['min_impurity_decrease'] = parseFloat(minImpurityDecrease.value);
            if (ccpAlpha) hyperparameters['ccp_alpha'] = parseFloat(ccpAlpha.value);
        }
    }
    else if (selectedModel === "SGD") {
        const loss = isAdvancedPage ? document.getElementById('advancedSGDLoss') : null;
        const penalty = isAdvancedPage ? document.getElementById('advancedSGDPenalty') : null;
        const alpha = isAdvancedPage ? document.getElementById('advancedSGDAlpha') : null;
        const l1Ratio = isAdvancedPage ? document.getElementById('advancedSGDL1Ratio') : null;
        if (loss) hyperparameters['loss'] = loss.value;
        if (penalty) hyperparameters['penalty'] = penalty.value;
        if (alpha) hyperparameters['alpha'] = parseFloat(alpha.value);
        if (l1Ratio) hyperparameters['l1_ratio'] = parseFloat(l1Ratio.value);
        
        const nonreqSlider = isAdvancedPage 
            ? document.getElementById('advancedNonreqSGDSlider')
            : null;
        if (nonreqSlider && nonreqSlider.checked) {
            nonreq = true;
            const fitIntercept = document.getElementById('SGDFitIntercept');
            const maxIter = document.getElementById('SGDMaxIter');
            const tol = document.getElementById('SGDTol');
            const shuffle = document.getElementById('SGDShuffle');
            const verbose = document.getElementById('SGDVerbose');
            const epsilon = document.getElementById('SGDEpsilon');
            const learningRate = document.getElementById('SGDLearningRate');
            const eta0 = document.getElementById('SGDEta0');
            const powerT = document.getElementById('SGDPowerT');
            const earlyStopping = document.getElementById('SGDEarlyStopping');
            const validationFraction = document.getElementById('SGDValidationFraction');
            const nIterNoChange = document.getElementById('SGDNIterNoChange');
            const warmStart = document.getElementById('SGDWarmStart');
            const average = document.getElementById('SGDAverage');
            if (fitIntercept) hyperparameters['fit_intercept'] = fitIntercept.value === 'true';
            if (maxIter) hyperparameters['max_iter'] = parseInt(maxIter.value);
            if (tol) hyperparameters['tol'] = parseFloat(tol.value);
            if (shuffle) hyperparameters['shuffle'] = shuffle.value === 'true';
            if (verbose) hyperparameters['verbose'] = parseInt(verbose.value);
            if (epsilon) hyperparameters['epsilon'] = parseFloat(epsilon.value);
            if (learningRate) hyperparameters['learning_rate'] = learningRate.value;
            if (eta0) hyperparameters['eta0'] = parseFloat(eta0.value);
            if (powerT) hyperparameters['power_t'] = parseFloat(powerT.value);
            if (earlyStopping) hyperparameters['early_stopping'] = earlyStopping.value === 'true';
            if (validationFraction) hyperparameters['validation_fraction'] = parseFloat(validationFraction.value);
            if (nIterNoChange) hyperparameters['n_iter_no_change'] = parseInt(nIterNoChange.value);
            if (warmStart) hyperparameters['warm_start'] = warmStart.value === 'true';
            if (average) hyperparameters['average'] = average.value === 'true';
            if (nJobs && nJobs.value) hyperparameters['n_jobs'] = parseInt(nJobs.value);
        }
    }
    else if (selectedModel === "HistGradientBoosting") {
        const learningRate = isAdvancedPage ? document.getElementById('advancedHistGBLearningRate') : null;
        const maxIter = isAdvancedPage ? document.getElementById('advancedHistGBMaxIter') : null;
        const maxLeafNodes = isAdvancedPage ? document.getElementById('advancedHistGBMaxLeafNodes') : null;
        if (learningRate) hyperparameters['learning_rate'] = parseFloat(learningRate.value);
        if (maxIter) hyperparameters['max_iter'] = parseInt(maxIter.value);
        if (maxLeafNodes) hyperparameters['max_leaf_nodes'] = parseInt(maxLeafNodes.value);
        
        const nonreqSlider = isAdvancedPage 
            ? document.getElementById('advancedNonreqHistGBSlider')
            : null;
        if (nonreqSlider && nonreqSlider.checked) {
            nonreq = true;
            const loss = document.getElementById('HistGBLoss');
            const maxDepth = document.getElementById('HistGBMaxDepth');
            const minSamplesLeaf = document.getElementById('HistGBMinSamplesLeaf');
            const l2Regularization = document.getElementById('HistGBL2Regularization');
            const maxBins = document.getElementById('HistGBMaxBins');
            const warmStart = document.getElementById('HistGBWarmStart');
            const earlyStopping = document.getElementById('HistGBEarlyStopping');
            const scoring = document.getElementById('HistGBScoring');
            const validationFraction = document.getElementById('HistGBValidationFraction');
            const nIterNoChange = document.getElementById('HistGBNIterNoChange');
            const tol = document.getElementById('HistGBTol');
            const verbose = document.getElementById('HistGBVerbose');
            if (loss) hyperparameters['loss'] = loss.value;
            if (maxDepth && maxDepth.value) hyperparameters['max_depth'] = parseInt(maxDepth.value);
            if (minSamplesLeaf) hyperparameters['min_samples_leaf'] = parseInt(minSamplesLeaf.value);
            if (l2Regularization) hyperparameters['l2_regularization'] = parseFloat(l2Regularization.value);
            if (maxBins) hyperparameters['max_bins'] = parseInt(maxBins.value);
            if (warmStart) hyperparameters['warm_start'] = warmStart.value === 'true';
            if (earlyStopping) hyperparameters['early_stopping'] = earlyStopping.value;
            if (scoring) hyperparameters['scoring'] = scoring.value;
            if (validationFraction) hyperparameters['validation_fraction'] = parseFloat(validationFraction.value);
            if (nIterNoChange) hyperparameters['n_iter_no_change'] = parseInt(nIterNoChange.value);
            if (tol) hyperparameters['tol'] = parseFloat(tol.value);
            if (verbose) hyperparameters['verbose'] = parseInt(verbose.value);
        }
    }
    else if (selectedModel === "Huber") {
        const epsilon = isAdvancedPage ? document.getElementById('advancedHuberEpsilon') : null;
        const alpha = isAdvancedPage ? document.getElementById('advancedHuberAlpha') : null;
        if (epsilon) hyperparameters['epsilon'] = parseFloat(epsilon.value);
        if (alpha) hyperparameters['alpha'] = parseFloat(alpha.value);
        
        const nonreqSlider = isAdvancedPage 
            ? document.getElementById('advancedNonreqHuberSlider')
            : null;
        if (nonreqSlider && nonreqSlider.checked) {
            nonreq = true;
            const maxIter = document.getElementById('HuberMaxIter');
            const warmStart = document.getElementById('HuberWarmStart');
            const fitIntercept = document.getElementById('HuberFitIntercept');
            const tol = document.getElementById('HuberTol');
            if (maxIter) hyperparameters['max_iter'] = parseInt(maxIter.value);
            if (warmStart) hyperparameters['warm_start'] = warmStart.value === 'true';
            if (fitIntercept) hyperparameters['fit_intercept'] = fitIntercept.value === 'true';
            if (tol) hyperparameters['tol'] = parseFloat(tol.value);
        }
    }
    else if (selectedModel === "Quantile") {
        const quantile = isAdvancedPage ? document.getElementById('advancedQuantileQuantile') : null;
        const alpha = isAdvancedPage ? document.getElementById('advancedQuantileAlpha') : null;
        if (quantile) hyperparameters['quantile'] = parseFloat(quantile.value);
        if (alpha) hyperparameters['alpha'] = parseFloat(alpha.value);
        
        const nonreqSlider = isAdvancedPage 
            ? document.getElementById('advancedNonreqQuantileSlider')
            : null;
        if (nonreqSlider && nonreqSlider.checked) {
            nonreq = true;
            const fitIntercept = document.getElementById('QuantileFitIntercept');
            const solver = document.getElementById('QuantileSolver');
            if (fitIntercept) hyperparameters['fit_intercept'] = fitIntercept.value === 'true';
            if (solver) hyperparameters['solver'] = solver.value;
        }
    }
    else if (selectedModel === "LinearSVR") {
        const c = isAdvancedPage ? document.getElementById('advancedLinearSVRC') : null;
        const epsilon = isAdvancedPage ? document.getElementById('advancedLinearSVREpsilon') : null;
        if (c) hyperparameters['C'] = parseFloat(c.value);
        if (epsilon) hyperparameters['epsilon'] = parseFloat(epsilon.value);
        
        const nonreqSlider = isAdvancedPage 
            ? document.getElementById('advancedNonreqLinearSVRSlider')
            : null;
        if (nonreqSlider && nonreqSlider.checked) {
            nonreq = true;
            const loss = document.getElementById('LinearSVRLoss');
            const tol = document.getElementById('LinearSVRTol');
            const fitIntercept = document.getElementById('LinearSVRFitIntercept');
            const interceptScaling = document.getElementById('LinearSVRInterceptScaling');
            const dual = document.getElementById('LinearSVRDual');
            const verbose = document.getElementById('LinearSVRVerbose');
            const maxIter = document.getElementById('LinearSVRMaxIter');
            if (loss) hyperparameters['loss'] = loss.value;
            if (tol) hyperparameters['tol'] = parseFloat(tol.value);
            if (fitIntercept) hyperparameters['fit_intercept'] = fitIntercept.value === 'true';
            if (interceptScaling) hyperparameters['intercept_scaling'] = parseFloat(interceptScaling.value);
            if (dual) hyperparameters['dual'] = dual.value === 'true';
            if (verbose) hyperparameters['verbose'] = parseInt(verbose.value);
            if (maxIter) hyperparameters['max_iter'] = parseInt(maxIter.value);
        }
    }
    else if (selectedModel === "NuSVR") {
        const nu = isAdvancedPage ? document.getElementById('advancedNuSVRNu') : null;
        const c = isAdvancedPage ? document.getElementById('advancedNuSVRC') : null;
        const kernel = isAdvancedPage ? document.getElementById('advancedNuSVRKernel') : null;
        if (nu) hyperparameters['nu'] = parseFloat(nu.value);
        if (c) hyperparameters['C'] = parseFloat(c.value);
        if (kernel) hyperparameters['kernel'] = kernel.value;
        
        const nonreqSlider = isAdvancedPage 
            ? document.getElementById('advancedNonreqNuSVRSlider')
            : null;
        if (nonreqSlider && nonreqSlider.checked) {
            nonreq = true;
            const degree = document.getElementById('NuSVRDegree');
            const gamma = document.getElementById('NuSVRGamma');
            const coef0 = document.getElementById('NuSVRCoef0');
            const shrinking = document.getElementById('NuSVRShrinking');
            const tol = document.getElementById('NuSVRTol');
            const cacheSize = document.getElementById('NuSVRCacheSize');
            const verbose = document.getElementById('NuSVRVerbose');
            const maxIter = document.getElementById('NuSVRMaxIter');
            if (degree) hyperparameters['degree'] = parseInt(degree.value);
            if (gamma) hyperparameters['gamma'] = gamma.value;
            if (coef0) hyperparameters['coef0'] = parseFloat(coef0.value);
            if (shrinking) hyperparameters['shrinking'] = shrinking.value === 'true';
            if (tol) hyperparameters['tol'] = parseFloat(tol.value);
            if (cacheSize) hyperparameters['cache_size'] = parseFloat(cacheSize.value);
            if (verbose) hyperparameters['verbose'] = verbose.value === 'true';
            if (maxIter) hyperparameters['max_iter'] = parseInt(maxIter.value);
        }
    }
    else if (selectedModel === "PassiveAggressive") {
        const c = isAdvancedPage ? document.getElementById('advancedPassiveAggressiveC') : null;
        const epsilon = isAdvancedPage ? document.getElementById('advancedPassiveAggressiveEpsilon') : null;
        if (c) hyperparameters['C'] = parseFloat(c.value);
        if (epsilon) hyperparameters['epsilon'] = parseFloat(epsilon.value);
        
        const nonreqSlider = isAdvancedPage 
            ? document.getElementById('advancedNonreqPassiveAggressiveSlider')
            : null;
        if (nonreqSlider && nonreqSlider.checked) {
            nonreq = true;
            const loss = document.getElementById('PassiveAggressiveLoss');
            const fitIntercept = document.getElementById('PassiveAggressiveFitIntercept');
            const maxIter = document.getElementById('PassiveAggressiveMaxIter');
            const tol = document.getElementById('PassiveAggressiveTol');
            const shuffle = document.getElementById('PassiveAggressiveShuffle');
            const verbose = document.getElementById('PassiveAggressiveVerbose');
            const earlyStopping = document.getElementById('PassiveAggressiveEarlyStopping');
            const validationFraction = document.getElementById('PassiveAggressiveValidationFraction');
            const nIterNoChange = document.getElementById('PassiveAggressiveNIterNoChange');
            const warmStart = document.getElementById('PassiveAggressiveWarmStart');
            const average = document.getElementById('PassiveAggressiveAverage');
            const nJobs = document.getElementById('PassiveAggressiveNJobs');
            if (loss) hyperparameters['loss'] = loss.value;
            if (fitIntercept) hyperparameters['fit_intercept'] = fitIntercept.value === 'true';
            if (maxIter) hyperparameters['max_iter'] = parseInt(maxIter.value);
            if (tol) hyperparameters['tol'] = parseFloat(tol.value);
            if (shuffle) hyperparameters['shuffle'] = shuffle.value === 'true';
            if (verbose) hyperparameters['verbose'] = parseInt(verbose.value);
            if (earlyStopping) hyperparameters['early_stopping'] = earlyStopping.value === 'true';
            if (validationFraction) hyperparameters['validation_fraction'] = parseFloat(validationFraction.value);
            if (nIterNoChange) hyperparameters['n_iter_no_change'] = parseInt(nIterNoChange.value);
            if (warmStart) hyperparameters['warm_start'] = warmStart.value === 'true';
            if (average) hyperparameters['average'] = average.value === 'true';
            if (nJobs && nJobs.value) hyperparameters['n_jobs'] = parseInt(nJobs.value);
        }
    }
    else if (selectedModel === "RANSAC") {
        const maxTrials = isAdvancedPage ? document.getElementById('advancedRANSACMaxTrials') : null;
        if (maxTrials) hyperparameters['max_trials'] = parseInt(maxTrials.value);
        
        const nonreqSlider = isAdvancedPage 
            ? document.getElementById('advancedNonreqRANSACSlider')
            : null;
        if (nonreqSlider && nonreqSlider.checked) {
            nonreq = true;
            const minSamples = document.getElementById('RANSACMinSamples');
            const residualThreshold = document.getElementById('RANSACResidualThreshold');
            const stopNInliers = document.getElementById('RANSACStopNInliers');
            const stopScore = document.getElementById('RANSACStopScore');
            const stopProbability = document.getElementById('RANSACStopProbability');
            const loss = document.getElementById('RANSACLoss');
            if (minSamples && minSamples.value) hyperparameters['min_samples'] = parseFloat(minSamples.value);
            if (residualThreshold && residualThreshold.value) hyperparameters['residual_threshold'] = parseFloat(residualThreshold.value);
            if (stopNInliers && stopNInliers.value) hyperparameters['stop_n_inliers'] = parseInt(stopNInliers.value);
            if (stopScore && stopScore.value) hyperparameters['stop_score'] = parseFloat(stopScore.value);
            if (stopProbability) hyperparameters['stop_probability'] = parseFloat(stopProbability.value);
            if (loss) hyperparameters['loss'] = loss.value;
        }
    }
    else if (selectedModel === "TheilSen") {
        const nonreqSlider = isAdvancedPage 
            ? document.getElementById('advancedNonreqTheilSenSlider')
            : null;
        if (nonreqSlider && nonreqSlider.checked) {
            nonreq = true;
            const fitIntercept = document.getElementById('TheilSenFitIntercept');
            const maxSubpopulation = document.getElementById('TheilSenMaxSubpopulation');
            const nSubsamples = document.getElementById('TheilSenNSubsamples');
            const maxIter = document.getElementById('TheilSenMaxIter');
            const tol = document.getElementById('TheilSenTol');
            const nJobs = document.getElementById('TheilSenNJobs');
            const verbose = document.getElementById('TheilSenVerbose');
            if (fitIntercept) hyperparameters['fit_intercept'] = fitIntercept.value === 'true';
            if (maxSubpopulation) hyperparameters['max_subpopulation'] = parseInt(maxSubpopulation.value);
            if (nSubsamples && nSubsamples.value) hyperparameters['n_subsamples'] = parseInt(nSubsamples.value);
            if (maxIter) hyperparameters['max_iter'] = parseInt(maxIter.value);
            if (tol) hyperparameters['tol'] = parseFloat(tol.value);
            if (nJobs && nJobs.value) hyperparameters['n_jobs'] = parseInt(nJobs.value);
            if (verbose) hyperparameters['verbose'] = verbose.value === 'true';
        }
    }
    else if (selectedModel === "RadiusNeighbors") {
        const radius = isAdvancedPage ? document.getElementById('advancedRadiusNeighborsRadius') : null;
        if (radius) hyperparameters['radius'] = parseFloat(radius.value);
        
        const nonreqSlider = isAdvancedPage 
            ? document.getElementById('advancedNonreqRadiusNeighborsSlider')
            : null;
        if (nonreqSlider && nonreqSlider.checked) {
            nonreq = true;
            const weights = document.getElementById('RadiusNeighborsWeights');
            const algorithm = document.getElementById('RadiusNeighborsAlgorithm');
            const leafSize = document.getElementById('RadiusNeighborsLeafSize');
            const p = document.getElementById('RadiusNeighborsP');
            const metric = document.getElementById('RadiusNeighborsMetric');
            const nJobs = document.getElementById('RadiusNeighborsNJobs');
            if (weights) hyperparameters['weights'] = weights.value;
            if (algorithm) hyperparameters['algorithm'] = algorithm.value;
            if (leafSize) hyperparameters['leaf_size'] = parseInt(leafSize.value);
            if (p) hyperparameters['p'] = parseFloat(p.value);
            if (metric) hyperparameters['metric'] = metric.value;
            if (nJobs && nJobs.value) hyperparameters['n_jobs'] = parseInt(nJobs.value);
        }
    }
    else if (selectedModel === "OMP") {
        const nonreqSlider = isAdvancedPage 
            ? document.getElementById('advancedNonreqOMPSlider')
            : null;
        if (nonreqSlider && nonreqSlider.checked) {
            nonreq = true;
            const nNonzeroCoefs = document.getElementById('OMPNNonzeroCoefs');
            const tol = document.getElementById('OMPTol');
            const fitIntercept = document.getElementById('OMPFitIntercept');
            const precompute = document.getElementById('OMPPrecompute');
            if (nNonzeroCoefs && nNonzeroCoefs.value) hyperparameters['n_nonzero_coefs'] = parseInt(nNonzeroCoefs.value);
            if (tol && tol.value) hyperparameters['tol'] = parseFloat(tol.value);
            if (fitIntercept) hyperparameters['fit_intercept'] = fitIntercept.value === 'true';
            if (precompute) hyperparameters['precompute'] = precompute.value;
        }
    }
    else if (selectedModel === "LARS") {
        const nonreqSlider = isAdvancedPage 
            ? document.getElementById('advancedNonreqLARSSlider')
            : null;
        if (nonreqSlider && nonreqSlider.checked) {
            nonreq = true;
            const fitIntercept = document.getElementById('LARSFitIntercept');
            const verbose = document.getElementById('LARSVerbose');
            const precompute = document.getElementById('LARSPrecompute');
            const nNonzeroCoefs = document.getElementById('LARSNNonzeroCoefs');
            const eps = document.getElementById('LARSEps');
            const copyX = document.getElementById('LARSCopyX');
            const fitPath = document.getElementById('LARSFitPath');
            if (fitIntercept) hyperparameters['fit_intercept'] = fitIntercept.value === 'true';
            if (verbose) hyperparameters['verbose'] = verbose.value === 'true';
            if (precompute) hyperparameters['precompute'] = precompute.value;
            if (nNonzeroCoefs) hyperparameters['n_nonzero_coefs'] = parseInt(nNonzeroCoefs.value);
            if (eps && eps.value) hyperparameters['eps'] = parseFloat(eps.value);
            if (copyX) hyperparameters['copy_X'] = copyX.value === 'true';
            if (fitPath) hyperparameters['fit_path'] = fitPath.value === 'true';
        }
    }
    else if (selectedModel === "LARSCV") {
        const nonreqSlider = isAdvancedPage 
            ? document.getElementById('advancedNonreqLARSCVSlider')
            : null;
        if (nonreqSlider && nonreqSlider.checked) {
            nonreq = true;
            const fitIntercept = document.getElementById('LARSCVFitIntercept');
            const verbose = document.getElementById('LARSCVVerbose');
            const maxIter = document.getElementById('LARSCVMaxIter');
            const precompute = document.getElementById('LARSCVPrecompute');
            const maxNAlphas = document.getElementById('LARSCVMaxNAlphas');
            const nJobs = document.getElementById('LARSCVNJobs');
            const eps = document.getElementById('LARSCVEps');
            const copyX = document.getElementById('LARSCVCopyX');
            if (fitIntercept) hyperparameters['fit_intercept'] = fitIntercept.value === 'true';
            if (verbose) hyperparameters['verbose'] = verbose.value === 'true';
            if (maxIter) hyperparameters['max_iter'] = parseInt(maxIter.value);
            if (precompute) hyperparameters['precompute'] = precompute.value;
            if (maxNAlphas) hyperparameters['max_n_alphas'] = parseInt(maxNAlphas.value);
            if (nJobs && nJobs.value) hyperparameters['n_jobs'] = parseInt(nJobs.value);
            if (eps && eps.value) hyperparameters['eps'] = parseFloat(eps.value);
            if (copyX) hyperparameters['copy_X'] = copyX.value === 'true';
        }
    }
    else if (selectedModel === "LassoCV") {
        const nonreqSlider = isAdvancedPage 
            ? document.getElementById('advancedNonreqLassoCVSlider')
            : null;
        if (nonreqSlider && nonreqSlider.checked) {
            nonreq = true;
            const eps = document.getElementById('LassoCVEps');
            const nAlphas = document.getElementById('LassoCVNAlphas');
            const fitIntercept = document.getElementById('LassoCVFitIntercept');
            const precompute = document.getElementById('LassoCVPrecompute');
            const maxIter = document.getElementById('LassoCVMaxIter');
            const tol = document.getElementById('LassoCVTol');
            const copyX = document.getElementById('LassoCVCopyX');
            const verbose = document.getElementById('LassoCVVerbose');
            const nJobs = document.getElementById('LassoCVNJobs');
            const positive = document.getElementById('LassoCVPositive');
            const selection = document.getElementById('LassoCVSelection');
            if (eps) hyperparameters['eps'] = parseFloat(eps.value);
            if (nAlphas) hyperparameters['n_alphas'] = parseInt(nAlphas.value);
            if (fitIntercept) hyperparameters['fit_intercept'] = fitIntercept.value === 'true';
            if (precompute) hyperparameters['precompute'] = precompute.value;
            if (maxIter) hyperparameters['max_iter'] = parseInt(maxIter.value);
            if (tol) hyperparameters['tol'] = parseFloat(tol.value);
            if (copyX) hyperparameters['copy_X'] = copyX.value === 'true';
            if (verbose) hyperparameters['verbose'] = verbose.value === 'true';
            if (nJobs && nJobs.value) hyperparameters['n_jobs'] = parseInt(nJobs.value);
            if (positive) hyperparameters['positive'] = positive.value === 'true';
            if (selection) hyperparameters['selection'] = selection.value;
        }
    }
    else if (selectedModel === "ElasticNetCV") {
        const l1Ratio = isAdvancedPage ? document.getElementById('advancedElasticNetCVL1Ratio') : null;
        if (l1Ratio) hyperparameters['l1_ratio'] = parseFloat(l1Ratio.value);
        
        const nonreqSlider = isAdvancedPage 
            ? document.getElementById('advancedNonreqElasticNetCVSlider')
            : null;
        if (nonreqSlider && nonreqSlider.checked) {
            nonreq = true;
            const eps = document.getElementById('ElasticNetCVEps');
            const nAlphas = document.getElementById('ElasticNetCVNAlphas');
            const fitIntercept = document.getElementById('ElasticNetCVFitIntercept');
            const precompute = document.getElementById('ElasticNetCVPrecompute');
            const maxIter = document.getElementById('ElasticNetCVMaxIter');
            const tol = document.getElementById('ElasticNetCVTol');
            const copyX = document.getElementById('ElasticNetCVCopyX');
            const verbose = document.getElementById('ElasticNetCVVerbose');
            const nJobs = document.getElementById('ElasticNetCVNJobs');
            const positive = document.getElementById('ElasticNetCVPositive');
            const selection = document.getElementById('ElasticNetCVSelection');
            if (eps) hyperparameters['eps'] = parseFloat(eps.value);
            if (nAlphas) hyperparameters['n_alphas'] = parseInt(nAlphas.value);
            if (fitIntercept) hyperparameters['fit_intercept'] = fitIntercept.value === 'true';
            if (precompute) hyperparameters['precompute'] = precompute.value;
            if (maxIter) hyperparameters['max_iter'] = parseInt(maxIter.value);
            if (tol) hyperparameters['tol'] = parseFloat(tol.value);
            if (copyX) hyperparameters['copy_X'] = copyX.value === 'true';
            if (verbose) hyperparameters['verbose'] = parseInt(verbose.value);
            if (nJobs && nJobs.value) hyperparameters['n_jobs'] = parseInt(nJobs.value);
            if (positive) hyperparameters['positive'] = positive.value === 'true';
            if (selection) hyperparameters['selection'] = selection.value;
        }
    }
    else if (selectedModel === "RidgeCV") {
        const nonreqSlider = isAdvancedPage 
            ? document.getElementById('advancedNonreqRidgeCVSlider')
            : null;
        if (nonreqSlider && nonreqSlider.checked) {
            nonreq = true;
            const alphas = document.getElementById('RidgeCVAlphas');
            const fitIntercept = document.getElementById('RidgeCVFitIntercept');
            const scoring = document.getElementById('RidgeCVScoring');
            const gcvMode = document.getElementById('RidgeCVGCVMode');
            const storeCVResults = document.getElementById('RidgeCVStoreCVResults');
            const alphaPerTarget = document.getElementById('RidgeCVAlphaPerTarget');
            if (alphas && alphas.value) {
                // Parse comma-separated values
                try {
                    hyperparameters['alphas'] = alphas.value.split(',').map(v => parseFloat(v.trim()));
                } catch (e) {
                    hyperparameters['alphas'] = alphas.value;
                }
            }
            if (fitIntercept) hyperparameters['fit_intercept'] = fitIntercept.value === 'true';
            if (scoring && scoring.value) hyperparameters['scoring'] = scoring.value;
            if (gcvMode) hyperparameters['gcv_mode'] = gcvMode.value;
            if (storeCVResults) hyperparameters['store_cv_results'] = storeCVResults.value === 'true';
            if (alphaPerTarget) hyperparameters['alpha_per_target'] = alphaPerTarget.value === 'true';
        }
    }
    
    // Additional classification models hyperparameter collection
    else if (selectedModel === 'AdaBoost_classifier') {
        const nEstimators = isAdvancedPage ? document.getElementById('advancedAdaBoostClassifierNEstimators') : null;
        const learningRate = isAdvancedPage ? document.getElementById('advancedAdaBoostClassifierLearningRate') : null;
        if (nEstimators) hyperparameters['n_estimators'] = parseInt(nEstimators.value);
        if (learningRate) hyperparameters['learning_rate'] = parseFloat(learningRate.value);
        
        const nonreqSlider = isAdvancedPage 
            ? document.getElementById('advancedNonreqAdaBoostClassifierSlider')
            : null;
        if (nonreqSlider && nonreqSlider.checked) {
            nonreq = true;
            // No additional hyperparameters for AdaBoostClassifier
        }
    }
    else if (selectedModel === 'Bagging_classifier') {
        const nEstimators = isAdvancedPage ? document.getElementById('advancedBaggingClassifierNEstimators') : null;
        if (nEstimators) hyperparameters['n_estimators'] = parseInt(nEstimators.value);
        
        const nonreqSlider = isAdvancedPage 
            ? document.getElementById('advancedNonreqBaggingClassifierSlider')
            : null;
        if (nonreqSlider && nonreqSlider.checked) {
            nonreq = true;
            const maxSamples = document.getElementById('BaggingClassifierMaxSamples');
            const maxFeatures = document.getElementById('BaggingClassifierMaxFeatures');
            const bootstrap = document.getElementById('BaggingClassifierBootstrap');
            const bootstrapFeatures = document.getElementById('BaggingClassifierBootstrapFeatures');
            const oobScore = document.getElementById('BaggingClassifierOobScore');
            const warmStart = document.getElementById('BaggingClassifierWarmStart');
            const nJobs = document.getElementById('BaggingClassifierNJobs');
            const verbose = document.getElementById('BaggingClassifierVerbose');
            if (maxSamples) hyperparameters['max_samples'] = maxSamples.value;
            if (maxFeatures) hyperparameters['max_features'] = maxFeatures.value;
            if (bootstrap) hyperparameters['bootstrap'] = bootstrap.value === 'true';
            if (bootstrapFeatures) hyperparameters['bootstrap_features'] = bootstrapFeatures.value === 'true';
            if (oobScore) hyperparameters['oob_score'] = oobScore.value === 'true';
            if (warmStart) hyperparameters['warm_start'] = warmStart.value === 'true';
            if (nJobs && nJobs.value) hyperparameters['n_jobs'] = parseInt(nJobs.value);
            if (verbose) hyperparameters['verbose'] = parseInt(verbose.value);
        }
    }
    else if (selectedModel === 'DecisionTree_classifier') {
        const nonreqSlider = isAdvancedPage 
            ? document.getElementById('advancedNonreqDecisionTreeClassifierSlider')
            : null;
        if (nonreqSlider && nonreqSlider.checked) {
            nonreq = true;
            const criterion = document.getElementById('DecisionTreeClassifierCriterion');
            const splitter = document.getElementById('DecisionTreeClassifierSplitter');
            const maxDepth = document.getElementById('DecisionTreeClassifierMaxDepth');
            const minSamplesSplit = document.getElementById('DecisionTreeClassifierMinSamplesSplit');
            const minSamplesLeaf = document.getElementById('DecisionTreeClassifierMinSamplesLeaf');
            const minWeightFractionLeaf = document.getElementById('DecisionTreeClassifierMinWeightFractionLeaf');
            const maxFeatures = document.getElementById('DecisionTreeClassifierMaxFeatures');
            const maxLeafNodes = document.getElementById('DecisionTreeClassifierMaxLeafNodes');
            const minImpurityDecrease = document.getElementById('DecisionTreeClassifierMinImpurityDecrease');
            const classWeight = document.getElementById('DecisionTreeClassifierClassWeight');
            const ccpAlpha = document.getElementById('DecisionTreeClassifierCcpAlpha');
            if (criterion) hyperparameters['criterion'] = criterion.value;
            if (splitter) hyperparameters['splitter'] = splitter.value;
            if (maxDepth && maxDepth.value) hyperparameters['max_depth'] = parseInt(maxDepth.value);
            if (minSamplesSplit) hyperparameters['min_samples_split'] = parseFloat(minSamplesSplit.value);
            if (minSamplesLeaf) hyperparameters['min_samples_leaf'] = parseFloat(minSamplesLeaf.value);
            if (minWeightFractionLeaf) hyperparameters['min_weight_fraction_leaf'] = parseFloat(minWeightFractionLeaf.value);
            if (maxFeatures && maxFeatures.value) hyperparameters['max_features'] = maxFeatures.value;
            if (maxLeafNodes && maxLeafNodes.value) hyperparameters['max_leaf_nodes'] = parseInt(maxLeafNodes.value);
            if (minImpurityDecrease) hyperparameters['min_impurity_decrease'] = parseFloat(minImpurityDecrease.value);
            if (classWeight && classWeight.value) hyperparameters['class_weight'] = classWeight.value;
            if (ccpAlpha) hyperparameters['ccp_alpha'] = parseFloat(ccpAlpha.value);
        }
    }
    else if (selectedModel === 'GradientBoosting_classifier') {
        const nEstimators = isAdvancedPage ? document.getElementById('advancedGradientBoostingClassifierNEstimators') : null;
        const learningRate = isAdvancedPage ? document.getElementById('advancedGradientBoostingClassifierLearningRate') : null;
        if (nEstimators) hyperparameters['n_estimators'] = parseInt(nEstimators.value);
        if (learningRate) hyperparameters['learning_rate'] = parseFloat(learningRate.value);
        
        const nonreqSlider = isAdvancedPage 
            ? document.getElementById('advancedNonreqGradientBoostingClassifierSlider')
            : null;
        if (nonreqSlider && nonreqSlider.checked) {
            nonreq = true;
            const loss = document.getElementById('GradientBoostingClassifierLoss');
            const subsample = document.getElementById('GradientBoostingClassifierSubsample');
            const criterion = document.getElementById('GradientBoostingClassifierCriterion');
            const minSamplesSplit = document.getElementById('GradientBoostingClassifierMinSamplesSplit');
            const minSamplesLeaf = document.getElementById('GradientBoostingClassifierMinSamplesLeaf');
            const minWeightFractionLeaf = document.getElementById('GradientBoostingClassifierMinWeightFractionLeaf');
            const maxDepth = document.getElementById('GradientBoostingClassifierMaxDepth');
            const minImpurityDecrease = document.getElementById('GradientBoostingClassifierMinImpurityDecrease');
            const maxFeatures = document.getElementById('GradientBoostingClassifierMaxFeatures');
            const maxLeafNodes = document.getElementById('GradientBoostingClassifierMaxLeafNodes');
            const verbose = document.getElementById('GradientBoostingClassifierVerbose');
            const warmStart = document.getElementById('GradientBoostingClassifierWarmStart');
            const validationFraction = document.getElementById('GradientBoostingClassifierValidationFraction');
            const nIterNoChange = document.getElementById('GradientBoostingClassifierNIterNoChange');
            const tol = document.getElementById('GradientBoostingClassifierTol');
            if (loss) hyperparameters['loss'] = loss.value;
            if (subsample) hyperparameters['subsample'] = parseFloat(subsample.value);
            if (criterion) hyperparameters['criterion'] = criterion.value;
            if (minSamplesSplit) hyperparameters['min_samples_split'] = parseFloat(minSamplesSplit.value);
            if (minSamplesLeaf) hyperparameters['min_samples_leaf'] = parseFloat(minSamplesLeaf.value);
            if (minWeightFractionLeaf) hyperparameters['min_weight_fraction_leaf'] = parseFloat(minWeightFractionLeaf.value);
            if (maxDepth && maxDepth.value) hyperparameters['max_depth'] = parseInt(maxDepth.value);
            if (minImpurityDecrease) hyperparameters['min_impurity_decrease'] = parseFloat(minImpurityDecrease.value);
            if (maxFeatures && maxFeatures.value) hyperparameters['max_features'] = maxFeatures.value;
            if (maxLeafNodes && maxLeafNodes.value) hyperparameters['max_leaf_nodes'] = parseInt(maxLeafNodes.value);
            if (verbose) hyperparameters['verbose'] = parseInt(verbose.value);
            if (warmStart) hyperparameters['warm_start'] = warmStart.value === 'true';
            if (validationFraction) hyperparameters['validation_fraction'] = parseFloat(validationFraction.value);
            if (nIterNoChange && nIterNoChange.value) hyperparameters['n_iter_no_change'] = parseInt(nIterNoChange.value);
            if (tol) hyperparameters['tol'] = parseFloat(tol.value);
        }
    }
    else if (selectedModel === 'HistGradientBoosting_classifier') {
        const learningRate = isAdvancedPage ? document.getElementById('advancedHistGradientBoostingClassifierLearningRate') : null;
        const maxIter = isAdvancedPage ? document.getElementById('advancedHistGradientBoostingClassifierMaxIter') : null;
        const maxLeafNodes = isAdvancedPage ? document.getElementById('advancedHistGradientBoostingClassifierMaxLeafNodes') : null;
        if (learningRate) hyperparameters['learning_rate'] = parseFloat(learningRate.value);
        if (maxIter) hyperparameters['max_iter'] = parseInt(maxIter.value);
        if (maxLeafNodes) hyperparameters['max_leaf_nodes'] = parseInt(maxLeafNodes.value);
        
        const nonreqSlider = isAdvancedPage 
            ? document.getElementById('advancedNonreqHistGradientBoostingClassifierSlider')
            : null;
        if (nonreqSlider && nonreqSlider.checked) {
            nonreq = true;
            const loss = document.getElementById('HistGradientBoostingClassifierLoss');
            const maxDepth = document.getElementById('HistGradientBoostingClassifierMaxDepth');
            const minSamplesLeaf = document.getElementById('HistGradientBoostingClassifierMinSamplesLeaf');
            const l2Regularization = document.getElementById('HistGradientBoostingClassifierL2Regularization');
            const maxBins = document.getElementById('HistGradientBoostingClassifierMaxBins');
            const warmStart = document.getElementById('HistGradientBoostingClassifierWarmStart');
            const earlyStopping = document.getElementById('HistGradientBoostingClassifierEarlyStopping');
            const scoring = document.getElementById('HistGradientBoostingClassifierScoring');
            const validationFraction = document.getElementById('HistGradientBoostingClassifierValidationFraction');
            const nIterNoChange = document.getElementById('HistGradientBoostingClassifierNIterNoChange');
            const tol = document.getElementById('HistGradientBoostingClassifierTol');
            const verbose = document.getElementById('HistGradientBoostingClassifierVerbose');
            if (loss) hyperparameters['loss'] = loss.value;
            if (maxDepth && maxDepth.value) hyperparameters['max_depth'] = parseInt(maxDepth.value);
            if (minSamplesLeaf) hyperparameters['min_samples_leaf'] = parseInt(minSamplesLeaf.value);
            if (l2Regularization) hyperparameters['l2_regularization'] = parseFloat(l2Regularization.value);
            if (maxBins) hyperparameters['max_bins'] = parseInt(maxBins.value);
            if (warmStart) hyperparameters['warm_start'] = warmStart.value === 'true';
            if (earlyStopping) hyperparameters['early_stopping'] = earlyStopping.value;
            if (scoring) hyperparameters['scoring'] = scoring.value;
            if (validationFraction) hyperparameters['validation_fraction'] = parseFloat(validationFraction.value);
            if (nIterNoChange) hyperparameters['n_iter_no_change'] = parseInt(nIterNoChange.value);
            if (tol) hyperparameters['tol'] = parseFloat(tol.value);
            if (verbose) hyperparameters['verbose'] = parseInt(verbose.value);
        }
    }
    else if (selectedModel === 'KNeighbors_classifier') {
        const nNeighbors = isAdvancedPage ? document.getElementById('advancedKNeighborsClassifierNNeighbors') : null;
        if (nNeighbors) hyperparameters['n_neighbors'] = parseInt(nNeighbors.value);
        
        const nonreqSlider = isAdvancedPage 
            ? document.getElementById('advancedNonreqKNeighborsClassifierSlider')
            : null;
        if (nonreqSlider && nonreqSlider.checked) {
            nonreq = true;
            const weights = document.getElementById('KNeighborsClassifierWeights');
            const algorithm = document.getElementById('KNeighborsClassifierAlgorithm');
            const leafSize = document.getElementById('KNeighborsClassifierLeafSize');
            const p = document.getElementById('KNeighborsClassifierP');
            const metric = document.getElementById('KNeighborsClassifierMetric');
            const nJobs = document.getElementById('KNeighborsClassifierNJobs');
            if (weights) hyperparameters['weights'] = weights.value;
            if (algorithm) hyperparameters['algorithm'] = algorithm.value;
            if (leafSize) hyperparameters['leaf_size'] = parseInt(leafSize.value);
            if (p) hyperparameters['p'] = parseFloat(p.value);
            if (metric) hyperparameters['metric'] = metric.value;
            if (nJobs && nJobs.value) hyperparameters['n_jobs'] = parseInt(nJobs.value);
        }
    }
    else if (selectedModel === 'LDA_classifier') {
        const nonreqSlider = isAdvancedPage 
            ? document.getElementById('advancedNonreqLDAClassifierSlider')
            : null;
        if (nonreqSlider && nonreqSlider.checked) {
            nonreq = true;
            const solver = document.getElementById('LDAClassifierSolver');
            const shrinkage = document.getElementById('LDAClassifierShrinkage');
            const priors = document.getElementById('LDAClassifierPriors');
            const nComponents = document.getElementById('LDAClassifierNComponents');
            const storeCovariance = document.getElementById('LDAClassifierStoreCovariance');
            const tol = document.getElementById('LDAClassifierTol');
            if (solver) hyperparameters['solver'] = solver.value;
            if (shrinkage && shrinkage.value) hyperparameters['shrinkage'] = parseFloat(shrinkage.value);
            if (priors && priors.value) hyperparameters['priors'] = priors.value;
            if (nComponents && nComponents.value) hyperparameters['n_components'] = parseInt(nComponents.value);
            if (storeCovariance) hyperparameters['store_covariance'] = storeCovariance.value === 'true';
            if (tol) hyperparameters['tol'] = parseFloat(tol.value);
        }
    }
    else if (selectedModel === 'LinearSVC_classifier') {
        const c = isAdvancedPage ? document.getElementById('advancedLinearSVCC') : null;
        if (c) hyperparameters['C'] = parseFloat(c.value);
        
        const nonreqSlider = isAdvancedPage 
            ? document.getElementById('advancedNonreqLinearSVCSlider')
            : null;
        if (nonreqSlider && nonreqSlider.checked) {
            nonreq = true;
            const loss = document.getElementById('LinearSVCLoss');
            const penalty = document.getElementById('LinearSVCPenalty');
            const dual = document.getElementById('LinearSVCDual');
            const tol = document.getElementById('LinearSVCTol');
            const multiClass = document.getElementById('LinearSVCMultiClass');
            const fitIntercept = document.getElementById('LinearSVCFitIntercept');
            const interceptScaling = document.getElementById('LinearSVCInterceptScaling');
            const maxIter = document.getElementById('LinearSVCMaxIter');
            const classWeight = document.getElementById('LinearSVCClassWeight');
            const verbose = document.getElementById('LinearSVCVerbose');
            if (loss) hyperparameters['loss'] = loss.value;
            if (penalty) hyperparameters['penalty'] = penalty.value;
            if (dual) hyperparameters['dual'] = dual.value === 'true';
            if (tol) hyperparameters['tol'] = parseFloat(tol.value);
            if (multiClass) hyperparameters['multi_class'] = multiClass.value;
            if (fitIntercept) hyperparameters['fit_intercept'] = fitIntercept.value === 'true';
            if (interceptScaling) hyperparameters['intercept_scaling'] = parseFloat(interceptScaling.value);
            if (maxIter) hyperparameters['max_iter'] = parseInt(maxIter.value);
            if (classWeight && classWeight.value) hyperparameters['class_weight'] = classWeight.value;
            if (verbose) hyperparameters['verbose'] = parseInt(verbose.value);
        }
    }
    else if (selectedModel === 'NuSVC_classifier') {
        const nu = isAdvancedPage ? document.getElementById('advancedNuSVCNu') : null;
        const c = isAdvancedPage ? document.getElementById('advancedNuSVCC') : null;
        const kernel = isAdvancedPage ? document.getElementById('advancedNuSVCKernel') : null;
        if (nu) hyperparameters['nu'] = parseFloat(nu.value);
        if (c) hyperparameters['C'] = parseFloat(c.value);
        if (kernel) hyperparameters['kernel'] = kernel.value;
        
        const nonreqSlider = isAdvancedPage 
            ? document.getElementById('advancedNonreqNuSVCSlider')
            : null;
        if (nonreqSlider && nonreqSlider.checked) {
            nonreq = true;
            const degree = document.getElementById('NuSVCDegree');
            const gamma = document.getElementById('NuSVCGamma');
            const coef0 = document.getElementById('NuSVCCoef0');
            const shrinking = document.getElementById('NuSVCShrinking');
            const tol = document.getElementById('NuSVCTol');
            const cacheSize = document.getElementById('NuSVCCacheSize');
            const verbose = document.getElementById('NuSVCVerbose');
            const maxIter = document.getElementById('NuSVCMaxIter');
            const classWeight = document.getElementById('NuSVCClassWeight');
            const decisionFunctionShape = document.getElementById('NuSVCDecisionFunctionShape');
            const breakTies = document.getElementById('NuSVCBreakTies');
            if (degree) hyperparameters['degree'] = parseInt(degree.value);
            if (gamma) hyperparameters['gamma'] = gamma.value;
            if (coef0) hyperparameters['coef0'] = parseFloat(coef0.value);
            if (shrinking) hyperparameters['shrinking'] = shrinking.value === 'true';
            if (tol) hyperparameters['tol'] = parseFloat(tol.value);
            if (cacheSize) hyperparameters['cache_size'] = parseFloat(cacheSize.value);
            if (verbose) hyperparameters['verbose'] = verbose.value === 'true';
            if (maxIter) hyperparameters['max_iter'] = parseInt(maxIter.value);
            if (classWeight && classWeight.value) hyperparameters['class_weight'] = classWeight.value;
            if (decisionFunctionShape) hyperparameters['decision_function_shape'] = decisionFunctionShape.value;
            if (breakTies) hyperparameters['break_ties'] = breakTies.value === 'true';
        }
    }
    else if (selectedModel === 'PassiveAggressive_classifier') {
        const c = isAdvancedPage ? document.getElementById('advancedPassiveAggressiveClassifierC') : null;
        if (c) hyperparameters['C'] = parseFloat(c.value);
        
        const nonreqSlider = isAdvancedPage 
            ? document.getElementById('advancedNonreqPassiveAggressiveClassifierSlider')
            : null;
        if (nonreqSlider && nonreqSlider.checked) {
            nonreq = true;
            const fitIntercept = document.getElementById('PassiveAggressiveClassifierFitIntercept');
            const maxIter = document.getElementById('PassiveAggressiveClassifierMaxIter');
            const tol = document.getElementById('PassiveAggressiveClassifierTol');
            const shuffle = document.getElementById('PassiveAggressiveClassifierShuffle');
            const verbose = document.getElementById('PassiveAggressiveClassifierVerbose');
            const loss = document.getElementById('PassiveAggressiveClassifierLoss');
            const warmStart = document.getElementById('PassiveAggressiveClassifierWarmStart');
            const classWeight = document.getElementById('PassiveAggressiveClassifierClassWeight');
            const nJobs = document.getElementById('PassiveAggressiveClassifierNJobs');
            const average = document.getElementById('PassiveAggressiveClassifierAverage');
            if (fitIntercept) hyperparameters['fit_intercept'] = fitIntercept.value === 'true';
            if (maxIter) hyperparameters['max_iter'] = parseInt(maxIter.value);
            if (tol) hyperparameters['tol'] = parseFloat(tol.value);
            if (shuffle) hyperparameters['shuffle'] = shuffle.value === 'true';
            if (verbose) hyperparameters['verbose'] = parseInt(verbose.value);
            if (loss) hyperparameters['loss'] = loss.value;
            if (warmStart) hyperparameters['warm_start'] = warmStart.value === 'true';
            if (classWeight && classWeight.value) hyperparameters['class_weight'] = classWeight.value;
            if (nJobs && nJobs.value) hyperparameters['n_jobs'] = parseInt(nJobs.value);
            if (average) hyperparameters['average'] = average.value === 'true';
        }
    }
    else if (selectedModel === 'QDA_classifier') {
        const nonreqSlider = isAdvancedPage 
            ? document.getElementById('advancedNonreqQDAClassifierSlider')
            : null;
        if (nonreqSlider && nonreqSlider.checked) {
            nonreq = true;
            const priors = document.getElementById('QDAClassifierPriors');
            const regParam = document.getElementById('QDAClassifierRegParam');
            const storeCovariance = document.getElementById('QDAClassifierStoreCovariance');
            const tol = document.getElementById('QDAClassifierTol');
            if (priors && priors.value) hyperparameters['priors'] = priors.value;
            if (regParam) hyperparameters['reg_param'] = parseFloat(regParam.value);
            if (storeCovariance) hyperparameters['store_covariance'] = storeCovariance.value === 'true';
            if (tol) hyperparameters['tol'] = parseFloat(tol.value);
        }
    }
    else if (selectedModel === 'Ridge_classifier') {
        const alpha = isAdvancedPage ? document.getElementById('advancedRidgeClassifierAlpha') : null;
        if (alpha) hyperparameters['alpha'] = parseFloat(alpha.value);
        
        const nonreqSlider = isAdvancedPage 
            ? document.getElementById('advancedNonreqRidgeClassifierSlider')
            : null;
        if (nonreqSlider && nonreqSlider.checked) {
            nonreq = true;
            const fitIntercept = document.getElementById('RidgeClassifierFitIntercept');
            const copyX = document.getElementById('RidgeClassifierCopyX');
            const maxIter = document.getElementById('RidgeClassifierMaxIter');
            const tol = document.getElementById('RidgeClassifierTol');
            const classWeight = document.getElementById('RidgeClassifierClassWeight');
            const solver = document.getElementById('RidgeClassifierSolver');
            const positive = document.getElementById('RidgeClassifierPositive');
            if (fitIntercept) hyperparameters['fit_intercept'] = fitIntercept.value === 'true';
            if (copyX) hyperparameters['copy_X'] = copyX.value === 'true';
            if (maxIter && maxIter.value) hyperparameters['max_iter'] = parseInt(maxIter.value);
            if (tol) hyperparameters['tol'] = parseFloat(tol.value);
            if (classWeight && classWeight.value) hyperparameters['class_weight'] = classWeight.value;
            if (solver) hyperparameters['solver'] = solver.value;
            if (positive) hyperparameters['positive'] = positive.value === 'true';
        }
    }
    else if (selectedModel === 'BernoulliNB_classifier') {
        const nonreqSlider = isAdvancedPage 
            ? document.getElementById('advancedNonreqBernoulliNBSlider')
            : null;
        if (nonreqSlider && nonreqSlider.checked) {
            nonreq = true;
            const alpha = document.getElementById('BernoulliNBAlpha');
            const fitPrior = document.getElementById('BernoulliNBFitPrior');
            const binarize = document.getElementById('BernoulliNBBinarize');
            if (alpha) hyperparameters['alpha'] = parseFloat(alpha.value);
            if (fitPrior) hyperparameters['fit_prior'] = fitPrior.value === 'true';
            if (binarize && binarize.value) hyperparameters['binarize'] = parseFloat(binarize.value);
        }
    }
    else if (selectedModel === 'CategoricalNB_classifier') {
        const nonreqSlider = isAdvancedPage 
            ? document.getElementById('advancedNonreqCategoricalNBSlider')
            : null;
        if (nonreqSlider && nonreqSlider.checked) {
            nonreq = true;
            const alpha = document.getElementById('CategoricalNBAlpha');
            const fitPrior = document.getElementById('CategoricalNBFitPrior');
            const minCategoryCount = document.getElementById('CategoricalNBMinCategoryCount');
            if (alpha) hyperparameters['alpha'] = parseFloat(alpha.value);
            if (fitPrior) hyperparameters['fit_prior'] = fitPrior.value === 'true';
            if (minCategoryCount) hyperparameters['min_category_count'] = parseInt(minCategoryCount.value);
        }
    }
    else if (selectedModel === 'ComplementNB_classifier') {
        const nonreqSlider = isAdvancedPage 
            ? document.getElementById('advancedNonreqComplementNBSlider')
            : null;
        if (nonreqSlider && nonreqSlider.checked) {
            nonreq = true;
            const alpha = document.getElementById('ComplementNBAlpha');
            const fitPrior = document.getElementById('ComplementNBFitPrior');
            const norm = document.getElementById('ComplementNBNorm');
            if (alpha) hyperparameters['alpha'] = parseFloat(alpha.value);
            if (fitPrior) hyperparameters['fit_prior'] = fitPrior.value === 'true';
            if (norm) hyperparameters['norm'] = norm.value === 'true';
        }
    }
    else if (selectedModel === 'MultinomialNB_classifier') {
        const nonreqSlider = isAdvancedPage 
            ? document.getElementById('advancedNonreqMultinomialNBSlider')
            : null;
        if (nonreqSlider && nonreqSlider.checked) {
            nonreq = true;
            const alpha = document.getElementById('MultinomialNBAlpha');
            const fitPrior = document.getElementById('MultinomialNBFitPrior');
            const classPrior = document.getElementById('MultinomialNBClassPrior');
            if (alpha) hyperparameters['alpha'] = parseFloat(alpha.value);
            if (fitPrior) hyperparameters['fit_prior'] = fitPrior.value === 'true';
            if (classPrior && classPrior.value) hyperparameters['class_prior'] = classPrior.value;
        }
    }
    
    // Additional clustering models hyperparameter collection
    else if (selectedModel === 'affinity_propagation') {
        const nonreqSlider = isAdvancedPage 
            ? document.getElementById('advancedNonreqAffinityPropagationSlider')
            : null;
        if (nonreqSlider && nonreqSlider.checked) {
            nonreq = true;
            const damping = document.getElementById('AffinityPropagationDamping');
            const maxIter = document.getElementById('AffinityPropagationMaxIter');
            const convergenceIter = document.getElementById('AffinityPropagationConvergenceIter');
            const copy = document.getElementById('AffinityPropagationCopy');
            const preference = document.getElementById('AffinityPropagationPreference');
            const affinity = document.getElementById('AffinityPropagationAffinity');
            const verbose = document.getElementById('AffinityPropagationVerbose');
            if (damping) hyperparameters['damping'] = parseFloat(damping.value);
            if (maxIter) hyperparameters['max_iter'] = parseInt(maxIter.value);
            if (convergenceIter) hyperparameters['convergence_iter'] = parseInt(convergenceIter.value);
            if (copy) hyperparameters['copy'] = copy.value === 'true';
            if (preference && preference.value) hyperparameters['preference'] = parseFloat(preference.value);
            if (affinity) hyperparameters['affinity'] = affinity.value;
            if (verbose) hyperparameters['verbose'] = verbose.value === 'true';
        }
    }
    else if (selectedModel === 'bisecting_kmeans') {
        const nClusters = isAdvancedPage ? document.getElementById('advancedBisectingKmeansNClusters') : null;
        if (nClusters) hyperparameters['n_clusters'] = parseInt(nClusters.value);
        
        const nonreqSlider = isAdvancedPage 
            ? document.getElementById('advancedNonreqBisectingKmeansSlider')
            : null;
        if (nonreqSlider && nonreqSlider.checked) {
            nonreq = true;
            const init = document.getElementById('BisectingKmeansInit');
            const nInit = document.getElementById('BisectingKmeansNInit');
            const maxIter = document.getElementById('BisectingKmeansMaxIter');
            const verbose = document.getElementById('BisectingKmeansVerbose');
            const tol = document.getElementById('BisectingKmeansTol');
            const copyX = document.getElementById('BisectingKmeansCopyX');
            const algorithm = document.getElementById('BisectingKmeansAlgorithm');
            const bisectingStrategy = document.getElementById('BisectingKmeansBisectingStrategy');
            if (init) hyperparameters['init'] = init.value;
            if (nInit) hyperparameters['n_init'] = parseInt(nInit.value);
            if (maxIter) hyperparameters['max_iter'] = parseInt(maxIter.value);
            if (verbose) hyperparameters['verbose'] = parseInt(verbose.value);
            if (tol) hyperparameters['tol'] = parseFloat(tol.value);
            if (copyX) hyperparameters['copy_x'] = copyX.value === 'true';
            if (algorithm) hyperparameters['algorithm'] = algorithm.value;
            if (bisectingStrategy) hyperparameters['bisecting_strategy'] = bisectingStrategy.value;
        }
    }
    else if (selectedModel === 'hdbscan') {
        const minClusterSize = isAdvancedPage ? document.getElementById('advancedHDBSCANMinClusterSize') : null;
        if (minClusterSize) hyperparameters['min_cluster_size'] = parseInt(minClusterSize.value);
        
        const nonreqSlider = isAdvancedPage 
            ? document.getElementById('advancedNonreqHDBSCANSlider')
            : null;
        if (nonreqSlider && nonreqSlider.checked) {
            nonreq = true;
            const minSamples = document.getElementById('HDBSCANMinSamples');
            const clusterSelectionEpsilon = document.getElementById('HDBSCANClusterSelectionEpsilon');
            const maxClusterSize = document.getElementById('HDBSCANMaxClusterSize');
            const metric = document.getElementById('HDBSCANMetric');
            const alpha = document.getElementById('HDBSCANAlpha');
            const algorithm = document.getElementById('HDBSCANAlgorithm');
            const leafSize = document.getElementById('HDBSCANLeafSize');
            const clusterSelectionMethod = document.getElementById('HDBSCANClusterSelectionMethod');
            const allowSingleCluster = document.getElementById('HDBSCANAllowSingleCluster');
            const copy = document.getElementById('HDBSCANCopy');
            const nJobs = document.getElementById('HDBSCANNJobs');
            if (minSamples && minSamples.value) hyperparameters['min_samples'] = parseInt(minSamples.value);
            if (clusterSelectionEpsilon) hyperparameters['cluster_selection_epsilon'] = parseFloat(clusterSelectionEpsilon.value);
            if (maxClusterSize && maxClusterSize.value) hyperparameters['max_cluster_size'] = parseInt(maxClusterSize.value);
            if (metric) hyperparameters['metric'] = metric.value;
            if (alpha) hyperparameters['alpha'] = parseFloat(alpha.value);
            if (algorithm) hyperparameters['algorithm'] = algorithm.value;
            if (leafSize) hyperparameters['leaf_size'] = parseInt(leafSize.value);
            if (clusterSelectionMethod) hyperparameters['cluster_selection_method'] = clusterSelectionMethod.value;
            if (allowSingleCluster) hyperparameters['allow_single_cluster'] = allowSingleCluster.value === 'true';
            if (copy) hyperparameters['copy'] = copy.value === 'true';
            if (nJobs && nJobs.value) hyperparameters['n_jobs'] = parseInt(nJobs.value);
        }
    }
    else if (selectedModel === 'meanshift') {
        const nonreqSlider = isAdvancedPage 
            ? document.getElementById('advancedNonreqMeanshiftSlider')
            : null;
        if (nonreqSlider && nonreqSlider.checked) {
            nonreq = true;
            const bandwidth = document.getElementById('MeanshiftBandwidth');
            const seeds = document.getElementById('MeanshiftSeeds');
            const binSeeding = document.getElementById('MeanshiftBinSeeding');
            const minBinFreq = document.getElementById('MeanshiftMinBinFreq');
            const clusterAll = document.getElementById('MeanshiftClusterAll');
            const nJobs = document.getElementById('MeanshiftNJobs');
            const maxIter = document.getElementById('MeanshiftMaxIter');
            if (bandwidth && bandwidth.value) hyperparameters['bandwidth'] = parseFloat(bandwidth.value);
            if (seeds && seeds.value) hyperparameters['seeds'] = seeds.value;
            if (binSeeding) hyperparameters['bin_seeding'] = binSeeding.value === 'true';
            if (minBinFreq) hyperparameters['min_bin_freq'] = parseInt(minBinFreq.value);
            if (clusterAll) hyperparameters['cluster_all'] = clusterAll.value === 'true';
            if (nJobs && nJobs.value) hyperparameters['n_jobs'] = parseInt(nJobs.value);
            if (maxIter) hyperparameters['max_iter'] = parseInt(maxIter.value);
        }
    }
    else if (selectedModel === 'minibatch_kmeans') {
        const nClusters = isAdvancedPage ? document.getElementById('advancedMinibatchKmeansNClusters') : null;
        if (nClusters) hyperparameters['n_clusters'] = parseInt(nClusters.value);
        
        const nonreqSlider = isAdvancedPage 
            ? document.getElementById('advancedNonreqMinibatchKmeansSlider')
            : null;
        if (nonreqSlider && nonreqSlider.checked) {
            nonreq = true;
            const init = document.getElementById('MinibatchKmeansInit');
            const maxIter = document.getElementById('MinibatchKmeansMaxIter');
            const batchSize = document.getElementById('MinibatchKmeansBatchSize');
            const verbose = document.getElementById('MinibatchKmeansVerbose');
            const computeLabels = document.getElementById('MinibatchKmeansComputeLabels');
            const tol = document.getElementById('MinibatchKmeansTol');
            const maxNoImprovement = document.getElementById('MinibatchKmeansMaxNoImprovement');
            const nInit = document.getElementById('MinibatchKmeansNInit');
            const reassignmentRatio = document.getElementById('MinibatchKmeansReassignmentRatio');
            if (init) hyperparameters['init'] = init.value;
            if (maxIter) hyperparameters['max_iter'] = parseInt(maxIter.value);
            if (batchSize) hyperparameters['batch_size'] = parseInt(batchSize.value);
            if (verbose) hyperparameters['verbose'] = parseInt(verbose.value);
            if (computeLabels) hyperparameters['compute_labels'] = computeLabels.value === 'true';
            if (tol) hyperparameters['tol'] = parseFloat(tol.value);
            if (maxNoImprovement) hyperparameters['max_no_improvement'] = parseInt(maxNoImprovement.value);
            if (nInit) hyperparameters['n_init'] = parseInt(nInit.value);
            if (reassignmentRatio) hyperparameters['reassignment_ratio'] = parseFloat(reassignmentRatio.value);
        }
    }
    else if (selectedModel === 'optics') {
        const minSamples = isAdvancedPage ? document.getElementById('advancedOPTICSMinSamples') : null;
        if (minSamples) hyperparameters['min_samples'] = parseInt(minSamples.value);
        
        const nonreqSlider = isAdvancedPage 
            ? document.getElementById('advancedNonreqOPTICSSlider')
            : null;
        if (nonreqSlider && nonreqSlider.checked) {
            nonreq = true;
            const maxEps = document.getElementById('OPTICSMaxEps');
            const metric = document.getElementById('OPTICSMetric');
            const p = document.getElementById('OPTICSP');
            const clusterMethod = document.getElementById('OPTICSClusterMethod');
            const eps = document.getElementById('OPTICSEps');
            const xi = document.getElementById('OPTICSXi');
            const predecessorCorrection = document.getElementById('OPTICSPredecessorCorrection');
            const minClusterSize = document.getElementById('OPTICSMinClusterSize');
            const algorithm = document.getElementById('OPTICSAlgorithm');
            const leafSize = document.getElementById('OPTICSLeafSize');
            const nJobs = document.getElementById('OPTICSNJobs');
            if (maxEps && maxEps.value) hyperparameters['max_eps'] = parseFloat(maxEps.value);
            if (metric) hyperparameters['metric'] = metric.value;
            if (p) hyperparameters['p'] = parseFloat(p.value);
            if (clusterMethod) hyperparameters['cluster_method'] = clusterMethod.value;
            if (eps && eps.value) hyperparameters['eps'] = parseFloat(eps.value);
            if (xi) hyperparameters['xi'] = parseFloat(xi.value);
            if (predecessorCorrection) hyperparameters['predecessor_correction'] = predecessorCorrection.value === 'true';
            if (minClusterSize && minClusterSize.value) hyperparameters['min_cluster_size'] = parseInt(minClusterSize.value);
            if (algorithm) hyperparameters['algorithm'] = algorithm.value;
            if (leafSize) hyperparameters['leaf_size'] = parseInt(leafSize.value);
            if (nJobs && nJobs.value) hyperparameters['n_jobs'] = parseInt(nJobs.value);
        }
    }


    
    //getting the units if user selected
    let _unitMessageStr = ''
    let unitStr = ''
    if (units) {
        let newUnits = units.replace("u*", "\u00B5")
        _unitMessageStr = `with ${newUnits} units`
        unitStr = `${newUnits}`
    }

    // Extract hyperparameter search parameters based on current mode
    // Note: currentMode is already declared at the start of this event listener (line 3419)
    let hyperparameterSearch = 'none';
    let searchCVFolds = 5;
    let searchNIter = 50;
    let featureSelectionMethod = 'none';
    let featureSelectionK = '';
    let outlierMethod = 'none';
    let outlierAction = 'remove';
    
    if (currentMode === 'automl') {
        // AutoML mode: set automatic defaults based on intensity level
        // Rationale: Quick focuses on speed (minimal preprocessing), Long is comprehensive
        const intensitySelect = document.getElementById('automlIntensity');
        const intensity = intensitySelect ? intensitySelect.value : 'medium';
        
        if (intensity === 'quick') {
            // Quick: Skip preprocessing that may not help, focus on fast hyperparameter optimization
            featureSelectionMethod = 'none';
            featureSelectionK = '';
            outlierMethod = 'none';
            outlierAction = 'remove';
            hyperparameterSearch = 'randomized';
            searchCVFolds = 3;
            searchNIter = 20;
        } else if (intensity === 'long') {
            // Long: Comprehensive preprocessing + exhaustive search
            featureSelectionMethod = 'RFE';
            featureSelectionK = '10';
            outlierMethod = 'IsolationForest';
            outlierAction = 'remove';
            hyperparameterSearch = 'grid';
            searchCVFolds = 10;
            searchNIter = 100;
        } else {
            // Medium: Balanced preprocessing with moderate search
            featureSelectionMethod = 'RFE';
            featureSelectionK = '10';
            outlierMethod = 'IsolationForest';
            outlierAction = 'remove';
            hyperparameterSearch = 'randomized';
            searchCVFolds = 5;
            searchNIter = 50;
        }
    } else if (currentMode === 'advanced') {
        // Advanced mode: get from form
        hyperparameterSearch = getCachedElement('hyperparameterSearch')?.value || 'none';
        searchCVFolds = getCachedElement('searchCVFolds')?.value || 5;
        searchNIter = getCachedElement('searchNIter')?.value || 50;
        featureSelectionMethod = getCachedElement('featureSelectionMethod')?.value || 'none';
        featureSelectionK = getCachedElement('featureSelectionK')?.value || '';
        outlierMethod = getCachedElement('outlierMethod')?.value || 'none';
        outlierAction = getCachedElement('outlierAction')?.value || 'remove';
    }
    // Simple mode: all remain 'none' (defaults above)
    
    //sending all the data to the backend
    const requestData = {
        filename: uploadedFileName,
        indicators: indicatorCols,
        predictors: predictorCols,
        models: selectedModel,
        scaler: scaler.value,
        hyperparameters: hyperparameters,
        nonreq: nonreq,
        units: unitStr,
        sigfig: parseInt(sigfig),
        stratifyColumn: stratifyColumnNumber,
        stratifyBool: stratifyBool,
        seedValue: parseInt(seedValue),
        testSize: parseFloat(testSize),
        dropMissing: dropMissing,
        imputeStrategy: imputeStrategy,
        dropZero: dropZero,
        quantileBinDict: quantileBinDict,
        useTransformer: useTransformer,
        transformerCols: transformerCols,
        crossValidationType: crossValidationType,
        crossValidationFolds: parseInt(crossValidationFolds),
        hyperparameterSearch: hyperparameterSearch,
        searchCVFolds: parseInt(searchCVFolds),
        searchNIter: parseInt(searchNIter),
        featureSelectionMethod: featureSelectionMethod,
        featureSelectionK: featureSelectionK ? parseInt(featureSelectionK) : null,
        outlierMethod: outlierMethod,
        outlierAction: outlierAction,
        modelingMode: currentMode,
    };

    // Set up progress tracking (variables are now global)
    
    // Function to start progress tracking via SSE
    function startProgressTracking(sessionId) {
        // Determine which loading div to use based on current mode
        const simpleMode = document.getElementById('simpleMode');
        const advancedMode = document.getElementById('advancedMode');
        const automlMode = document.getElementById('automlMode');
        const currentMode = simpleMode?.checked ? 'simple' : (advancedMode?.checked ? 'advanced' : (automlMode?.checked ? 'automl' : 'simple'));
        
        let loadingDiv;
        if (currentMode === 'advanced') {
            loadingDiv = document.getElementById('advancedLoading');
        } else if (currentMode === 'automl') {
            loadingDiv = document.getElementById('automlLoading');
        } else {
            loadingDiv = getCachedElement('loading');
        }
        if (!loadingDiv) return;
        
        // Show stop button
        let stopButton;
        if (currentMode === 'automl') {
            stopButton = document.getElementById('stopAutomlButton');
        } else if (currentMode === 'advanced') {
            stopButton = document.getElementById('stopAdvancedButton');
        } else {
            stopButton = document.getElementById('stopSimpleButton');
        }
        if (stopButton) stopButton.style.display = 'inline-block';
        
        const eventSource = new EventSource(`/progress/${sessionId}`);
        progressEventSource = eventSource;
        
        eventSource.onmessage = function(event) {
            try {
                const progress = JSON.parse(event.data);
                
                // Handle result message - process the result data
                if (progress.type === 'result' && progress.data) {
                    console.log('Received result via SSE:', progress.data);
                    // Process result when training completes
                    if (processResultData) {
                        // processResultData callback will handle closing the event source
                        processResultData(progress.data);
                        return; // Exit early - callback handles cleanup
                    } else {
                        console.error('processResultData function not set!');
                        // Fallback: process directly (with empty defaults for variables)
                        // Try to get unitStr from DOM if available
                        const unitNameElement = getCachedElement('unitName');
                        let fallbackUnitStr = '';
                        if (unitNameElement && unitNameElement.value) {
                            let newUnits = unitNameElement.value.replace("u*", "\u00B5");
                            fallbackUnitStr = `${newUnits}`;
                        }
                        processModelResult(progress.data, fallbackUnitStr, progress.data.predictors || [], {});
                        // Only close event source in fallback case (callback not set)
                        eventSource.close();
                        progressEventSource = null;
                        // Determine which loading div to use based on current mode
                        const simpleMode = document.getElementById('simpleMode');
                        const advancedMode = document.getElementById('advancedMode');
                        const automlMode = document.getElementById('automlMode');
                        const currentMode = simpleMode?.checked ? 'simple' : (advancedMode?.checked ? 'advanced' : (automlMode?.checked ? 'automl' : 'simple'));
                        
                        let loadingDiv;
                        if (currentMode === 'advanced') {
                            loadingDiv = document.getElementById('advancedLoading');
                        } else if (currentMode === 'automl') {
                            loadingDiv = document.getElementById('automlLoading');
                        } else {
                            loadingDiv = getCachedElement('loading');
                        }
                        if (loadingDiv) {
                            loadingDiv.classList.add('hidden');
                            loadingDiv.innerHTML = ``;
                        }
                        
                        // Re-enable appropriate button and hide stop button
                        let stopButton;
                        if (currentMode === 'automl') {
                            const automlButton = document.getElementById('automlSubmitButton');
                            if (automlButton) {
                                automlButton.disabled = false;
                                automlButton.textContent = 'Run AutoML';
                            }
                            stopButton = document.getElementById('stopAutomlButton');
                        } else if (currentMode === 'advanced') {
                            const advancedButton = document.getElementById('advancedOptimizationSubmitButton');
                            if (advancedButton) advancedButton.disabled = false;
                            stopButton = document.getElementById('stopAdvancedButton');
                        } else {
                            const processButton = getCachedElement('processButton');
                            if (processButton) processButton.disabled = false;
                            stopButton = document.getElementById('stopSimpleButton');
                        }
                        if (stopButton) stopButton.style.display = 'none';
                        return;
                    }
                }
                
                // Handle error
                if (progress.error) {
                    console.error('Progress error:', progress.error);
                    const errorDiv = getCachedElement('errorDiv');
                    if (errorDiv) {
                        showError(errorDiv, `Error: ${progress.error}`);
                    }
                    
                    // Re-enable appropriate button on error
                    const simpleMode = document.getElementById('simpleMode');
                    const advancedMode = document.getElementById('advancedMode');
                    const automlMode = document.getElementById('automlMode');
                    const currentMode = simpleMode?.checked ? 'simple' : (advancedMode?.checked ? 'advanced' : (automlMode?.checked ? 'automl' : 'simple'));
                    
                    let stopButton;
                    if (currentMode === 'automl') {
                        const automlButton = document.getElementById('automlSubmitButton');
                        if (automlButton) {
                            automlButton.disabled = false;
                            automlButton.textContent = 'Run AutoML';
                        }
                        stopButton = document.getElementById('stopAutomlButton');
                    } else if (currentMode === 'advanced') {
                        const advancedButton = document.getElementById('advancedOptimizationSubmitButton');
                        if (advancedButton) advancedButton.disabled = false;
                        stopButton = document.getElementById('stopAdvancedButton');
                    } else {
                        const processButton = getCachedElement('processButton');
                        if (processButton) processButton.disabled = false;
                        stopButton = document.getElementById('stopSimpleButton');
                    }
                    if (stopButton) stopButton.style.display = 'none';
                    
                    // Hide loading indicator
                    if (loadingDiv) {
                        loadingDiv.classList.add('hidden');
                        loadingDiv.innerHTML = ``;
                    }
                    
                    eventSource.close();
                    progressEventSource = null;
                    return;
                }
                
                // Update progress display - use correct loading div based on current mode
                const simpleMode = document.getElementById('simpleMode');
                const advancedMode = document.getElementById('advancedMode');
                const automlMode = document.getElementById('automlMode');
                const currentMode = simpleMode?.checked ? 'simple' : (advancedMode?.checked ? 'advanced' : (automlMode?.checked ? 'automl' : 'simple'));
                
                let loadingDiv;
                if (currentMode === 'advanced') {
                    loadingDiv = document.getElementById('advancedLoading');
                } else if (currentMode === 'automl') {
                    loadingDiv = document.getElementById('automlLoading');
                } else {
                    loadingDiv = getCachedElement('loading');
                }
                if (loadingDiv) {
                    updateProgressDisplay(progress, loadingDiv);
                }
                
                // If progress is complete but no result yet, wait for it
                if (progress.overall_progress >= 100) {
                    // Result should come in next message
                }
            } catch (e) {
                console.error('Error parsing progress:', e);
            }
        };
        
        eventSource.onerror = function(error) {
            console.error('SSE error:', error);
            
            // Re-enable appropriate button on SSE error
            const simpleMode = document.getElementById('simpleMode');
            const advancedMode = document.getElementById('advancedMode');
            const automlMode = document.getElementById('automlMode');
            const currentMode = simpleMode?.checked ? 'simple' : (advancedMode?.checked ? 'advanced' : (automlMode?.checked ? 'automl' : 'simple'));
            
            let stopButton;
            if (currentMode === 'automl') {
                const automlButton = document.getElementById('automlSubmitButton');
                if (automlButton) {
                    automlButton.disabled = false;
                    automlButton.textContent = 'Run AutoML';
                }
                stopButton = document.getElementById('stopAutomlButton');
            } else if (currentMode === 'advanced') {
                const advancedButton = document.getElementById('advancedOptimizationSubmitButton');
                if (advancedButton) advancedButton.disabled = false;
                stopButton = document.getElementById('stopAdvancedButton');
            } else {
                const processButton = getCachedElement('processButton');
                if (processButton) processButton.disabled = false;
                stopButton = document.getElementById('stopSimpleButton');
            }
            if (stopButton) stopButton.style.display = 'none';
            
            // Determine which loading div to use based on CURRENT mode (not the mode when tracking started)
            // This ensures errors are displayed in the correct loading div even if user changed modes
            let currentLoadingDiv;
            if (currentMode === 'advanced') {
                currentLoadingDiv = document.getElementById('advancedLoading');
            } else if (currentMode === 'automl') {
                currentLoadingDiv = document.getElementById('automlLoading');
            } else {
                currentLoadingDiv = getCachedElement('loading');
            }
            
            // Show error message in loading div
            if (currentLoadingDiv) {
                currentLoadingDiv.innerHTML = `
                    <p style="color: #d32f2f; font-weight: 600;">Connection Error</p>
                    <p style="color: #666;">Unable to establish connection with the server. Please try again.</p>
                `;
            }
            
            eventSource.close();
            progressEventSource = null;
        };
    }
    
    // Function to update progress display
    function updateProgressDisplay(progress, container) {
        if (!container || !progress) return;
        
        const stages = progress.stages || {};
        const overallProgress = progress.overall_progress || 0;
        const elapsedTime = progress.elapsed_time || 0;
        const _estimatedTimeRemaining = progress.estimated_time_remaining || 0;
        
        // Check if this is AutoML mode for enhanced display
        const _simpleMode = document.getElementById('simpleMode');
        const _advancedMode = document.getElementById('advancedMode');
        const automlMode = document.getElementById('automlMode');
        const isAutoML = automlMode?.checked;
        
        // Format time
        const formatTime = (seconds) => {
            if (!seconds || seconds < 0) return 'Calculating...';
            const mins = Math.floor(seconds / 60);
            const secs = Math.floor(seconds % 60);
            return mins > 0 ? `${mins}m ${secs}s` : `${secs}s`;
        };
        
        // Generate HTML for stages
        let stagesHtml = '';
        for (const [stageName, stage] of Object.entries(stages)) {
            if (stage.status === 'skipped') continue;
            
            const statusClass = stage.status === 'running' ? 'running' : 
                               stage.status === 'completed' ? 'completed' : '';
            const icon = stage.status === 'running' ? '...' : 
                        stage.status === 'completed' ? '✓' : '–';
            
            stagesHtml += `
                <div class="progress-stage ${statusClass}">
                    <div class="progress-stage-header">
                        <span class="progress-stage-icon">${icon}</span>
                        <span class="progress-stage-name">${stageName.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}</span>
                        <span class="progress-stage-percent">${Math.round(stage.progress)}%</span>
                    </div>
                    <div class="progress-stage-message">${stage.message || ''}</div>
                </div>
            `;
        }
        
        // Enhanced display for AutoML - only show "Running" if not complete
        const isComplete = overallProgress >= 100;
        const autoMLWrapper = isAutoML && !isComplete ? `
            <div style="padding: 24px; background-color: #fff3cd; border: 2px solid #ffc107; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin-bottom: 20px;">
                <p style="font-size: 1.2em; font-weight: 700; margin-bottom: 0; color: #856404; display: flex; align-items: center; gap: 10px;">
                    <span style="display: inline-block; animation: spin 1s linear infinite;">...</span>
                    <span>AutoML is running...</span>
                </p>
            </div>
        ` : (isAutoML && isComplete ? `
            <div style="padding: 24px; background-color: #d4edda; border: 2px solid #28a745; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin-bottom: 20px;">
                <p style="font-size: 1.2em; font-weight: 700; margin-bottom: 0; color: #155724; display: flex; align-items: center; gap: 10px;">
                    <span>Done</span>
                    <span>AutoML complete</span>
                </p>
            </div>
        ` : '');
        
        container.innerHTML = autoMLWrapper + `
            <div class="progress-container">
                <div class="progress-overall">
                    <h3>Overall Progress</h3>
                    <div class="progress-bar-container">
                        <div class="progress-bar" style="width: ${overallProgress}%"></div>
                    </div>
                    <div class="progress-time">
                        <span>Elapsed: ${formatTime(elapsedTime)}</span>
                        <span>${Math.round(overallProgress)}%</span>
                    </div>
                </div>
                <div class="progress-stages">
                    ${stagesHtml}
                </div>
            </div>
        `;
    }
    
    let isAsyncProcessing = false;
    try {
        const response = await fetch('/process', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestData),
        });
        let data = await response.json();
        
        // Check if this is an async processing response (202 Accepted)
        isAsyncProcessing = response.status === 202 || (data && data.status === 'processing');
        
        // Start progress tracking if session_id is returned
        if (data && data.session_id) {
            sessionId = data.session_id;
            startProgressTracking(sessionId);
        }
        
        // If async processing, wait for result from SSE (don't process response data)
        if (isAsyncProcessing) {
            // Result will be processed when received via SSE
            // Store the processing function to be called from SSE handler
            // Capture variables needed by processModelResult in closure
            const capturedUnitStr = unitStr;
            const capturedPredictorCols = predictorCols;
            const capturedHyperparameters = hyperparameters;
            processResultData = function(resultData) {
                // Pass captured variables to processModelResult
                processModelResult(resultData, capturedUnitStr, capturedPredictorCols, capturedHyperparameters);
                // Close event source and clean up after processing result
                if (progressEventSource) {
                    progressEventSource.close();
                    progressEventSource = null;
                }
                
                // Re-enable appropriate button based on current mode
                const simpleMode = document.getElementById('simpleMode');
                const advancedMode = document.getElementById('advancedMode');
                const automlMode = document.getElementById('automlMode');
                const currentMode = simpleMode?.checked ? 'simple' : (advancedMode?.checked ? 'advanced' : (automlMode?.checked ? 'automl' : 'simple'));
                
                // Hide loading div and stop button when result is processed
                let loadingDiv;
                let stopButton;
                if (currentMode === 'automl') {
                    loadingDiv = document.getElementById('automlLoading');
                    const automlButton = document.getElementById('automlSubmitButton');
                    if (automlButton) {
                        automlButton.disabled = false;
                        automlButton.textContent = 'Run AutoML';
                    }
                    stopButton = document.getElementById('stopAutomlButton');
                } else if (currentMode === 'advanced') {
                    loadingDiv = document.getElementById('advancedLoading');
                    const advancedButton = document.getElementById('advancedOptimizationSubmitButton');
                    if (advancedButton) advancedButton.disabled = false;
                    stopButton = document.getElementById('stopAdvancedButton');
                } else {
                    loadingDiv = getCachedElement('loading');
                    const processButton = getCachedElement('processButton');
                    if (processButton) processButton.disabled = false;
                    stopButton = document.getElementById('stopSimpleButton');
                }
                if (stopButton) stopButton.style.display = 'none';
                
                // Hide loading div when result is complete
                if (loadingDiv) {
                    loadingDiv.classList.add('hidden');
                    loadingDiv.innerHTML = '';
                }
            };
            // Don't close event source or hide loading here - wait for result via SSE
            return; // Exit early, result will come via SSE
        }

        // Synchronous response - process immediately (backward compatibility)
        if (response.ok) {
            processModelResult(data, unitStr, predictorCols, hyperparameters);
        } else {
            showError(errorDiv, data.error ? `Error: ${data.error}` : 'Request failed. See console for details.');
        }
    } catch (error) {
        console.error('Error:', error);
        // Always clean up on error, regardless of async processing state
        // This handles cases where error occurs after isAsyncProcessing is set but before SSE is established
        if (progressEventSource) {
            progressEventSource.close();
            progressEventSource = null;
        }
        showError(errorDiv, 'Something went wrong. See console for details.');
        
        // Determine which loading div and button to use based on current mode
        const simpleMode = document.getElementById('simpleMode');
        const advancedMode = document.getElementById('advancedMode');
        const automlMode = document.getElementById('automlMode');
        const currentMode = simpleMode?.checked ? 'simple' : (advancedMode?.checked ? 'advanced' : (automlMode?.checked ? 'automl' : 'simple'));
        
        let loadingDiv;
        if (currentMode === 'advanced') {
            loadingDiv = document.getElementById('advancedLoading');
        } else if (currentMode === 'automl') {
            loadingDiv = document.getElementById('automlLoading');
        } else {
            loadingDiv = loading;
        }
        if (loadingDiv) {
            loadingDiv.classList.add('hidden');
            loadingDiv.innerHTML = ``;
        }
        
        // Re-enable appropriate button and hide stop button
        let stopButton;
        if (currentMode === 'automl') {
            const automlButton = document.getElementById('automlSubmitButton');
            if (automlButton) {
                automlButton.disabled = false;
                automlButton.textContent = 'Run AutoML';
            }
            stopButton = document.getElementById('stopAutomlButton');
        } else if (currentMode === 'advanced') {
            const advancedButton = document.getElementById('advancedOptimizationSubmitButton');
            if (advancedButton) advancedButton.disabled = false;
            stopButton = document.getElementById('stopAdvancedButton');
        } else {
            const processButton = getCachedElement('processButton');
            if (processButton) processButton.disabled = false;
            stopButton = document.getElementById('stopSimpleButton');
        }
        if (stopButton) stopButton.style.display = 'none';
    }
    finally {
        // Only close event source and clean up if NOT doing async processing
        // (For async processing, cleanup happens in processResultData callback)
        // Note: Error cases are handled in catch block above
        if (!isAsyncProcessing) {
            if (progressEventSource) {
                progressEventSource.close();
                progressEventSource = null;
            }
            // Determine which loading div to use
            const isAdvancedPage = document.getElementById('advancedOptimization') && !document.getElementById('advancedOptimization').classList.contains('hidden');
            const loadingDiv = isAdvancedPage ? document.getElementById('advancedLoading') : loading;
            if (loadingDiv) {
                loadingDiv.classList.add('hidden');
                loadingDiv.innerHTML = ``;
            }
            const processButton = getCachedElement('processButton');
            if (processButton) processButton.disabled = false;
            const stopButton = document.getElementById('stopSimpleButton');
            if (stopButton) stopButton.style.display = 'none';
        }
    }
});

// Add event listeners for stop buttons and model dropdowns (selecting a new model auto-stops current run)
document.addEventListener('DOMContentLoaded', function() {
    const stopSimpleButton = document.getElementById('stopSimpleButton');
    const stopAdvancedButton = document.getElementById('stopAdvancedButton');
    const stopAutomlButton = document.getElementById('stopAutomlButton');
    
    if (stopSimpleButton) {
        stopSimpleButton.addEventListener('click', stopModelRun);
    }
    if (stopAdvancedButton) {
        stopAdvancedButton.addEventListener('click', stopModelRun);
    }
    if (stopAutomlButton) {
        stopAutomlButton.addEventListener('click', stopModelRun);
    }

    // When user selects a different model, stop any running model so they can run the new one
    const modelSelectIds = [
        'simpleNumericModels', 'simpleClusterModels', 'simpleClassifierModels',
        'advancedNumericModels', 'advancedClusterModels', 'advancedClassifierModels',
        'automlNumericModels', 'automlClusterModels', 'automlClassifierModels',
        'headerNumericModels', 'headerClusterModels', 'headerClassifierModels',
        'headerAdvancedNumericModels', 'headerAdvancedClusterModels', 'headerAdvancedClassifierModels',
        'headerAutomlNumericModels', 'headerAutomlClusterModels', 'headerAutomlClassifierModels'
    ];
    modelSelectIds.forEach(function(id) {
        const el = document.getElementById(id);
        if (el) {
            el.addEventListener('change', function() {
                if (progressEventSource) {
                    stopModelRun();
                }
            });
        }
    });
});

// Extract result processing into a separate function
function processModelResult(data, unitStr = '', predictorCols = [], hyperparameters = {}) {
    console.log('processModelResult called with data:', data);
    try {
        const errorDiv = getCachedElement('errorDiv');
        
        // Determine current modeling mode
        const simpleMode = document.getElementById('simpleMode');
        const advancedMode = document.getElementById('advancedMode');
        const automlMode = document.getElementById('automlMode');
        
        let currentMode = 'simple'; // default
        if (simpleMode && simpleMode.checked) {
            currentMode = 'simple';
        } else if (advancedMode && advancedMode.checked) {
            currentMode = 'advanced';
        } else if (automlMode && automlMode.checked) {
            currentMode = 'automl';
        }
        
        // Select result divs based on current mode
        let NumericResultDiv, ClassifierResultDiv, ClusterResultDiv, resultsContainer, resultsPlaceholder, imageSelector;
        
        if (currentMode === 'simple') {
            NumericResultDiv = getCachedElement('NumericResultDiv');
            ClassifierResultDiv = getCachedElement('ClassifierResultDiv');
            ClusterResultDiv = getCachedElement('ClusterResultDiv');
            resultsContainer = document.getElementById('simpleModelingResults');
            resultsPlaceholder = document.getElementById('resultsPlaceholder');
            imageSelector = document.getElementById('imageSelector');
        } else if (currentMode === 'advanced') {
            NumericResultDiv = getCachedElement('AdvancedNumericResultDiv');
            ClassifierResultDiv = getCachedElement('AdvancedClassifierResultDiv');
            ClusterResultDiv = getCachedElement('AdvancedClusterResultDiv');
            resultsContainer = document.getElementById('advancedModelingResults');
            resultsPlaceholder = document.getElementById('advancedResultsPlaceholder');
            imageSelector = document.getElementById('advancedImageSelector');
        } else { // automl
            NumericResultDiv = getCachedElement('AutoMLNumericResultDiv');
            ClassifierResultDiv = getCachedElement('AutoMLClassifierResultDiv');
            ClusterResultDiv = getCachedElement('AutoMLClusterResultDiv');
            resultsContainer = document.getElementById('automlModelingResults');
            resultsPlaceholder = document.getElementById('automlResultsPlaceholder');
            imageSelector = document.getElementById('automlImageSelector');
        }
        
        if (data.error) {
            console.error('Error in result data:', data.error);
            showError(errorDiv, `Error: ${data.error}`);
            return;
        }
        
        // Use predictorCols from data if available, otherwise use passed parameter
        const actualPredictorCols = data.predictors && data.predictors.length > 0 ? data.predictors : predictorCols;
        
        // Merge form hyperparameters with model params to show all hyperparameters actually used
        // Model params include all parameters (essential and non-essential) with their actual values
        // Form hyperparameters include non-essential ones only if toggle was enabled
        // Merge them: model params take precedence (they're what was actually used), but form hyperparameters
        // fill in any gaps (like non-essential ones that weren't in model params)
        let allHyperparameters = {...hyperparameters};
        if (data.model_params && typeof data.model_params === 'object') {
            // Merge model params - these include all parameters actually used by the model
            // This ensures non-essential hyperparameters are shown even if toggle was off
            allHyperparameters = {...allHyperparameters, ...data.model_params};
        }
        
        errorDiv.innerHTML = ''
        const resultTimestamp = formatDateTimeForFilename()
        // Determine prefix based on current mode
        let modePrefix = 'simplemodeling_';
        if (currentMode === 'advanced') {
            modePrefix = 'advancedmodeling_';
        } else if (currentMode === 'automl') {
            modePrefix = 'automl_';
        }
        const performanceDownloadName = `${modePrefix}model_performance_${resultTimestamp}.xlsx`
        const visualizationsDownloadName = `${modePrefix}model_visualizations_${resultTimestamp}.pdf`
        const crossValidationDownloadName = `${modePrefix}cross_validation_${resultTimestamp}.xlsx`
        let _detailsStr = ''
        if (Object.keys(allHyperparameters).length!==0){
            _detailsStr = 'and hyperparameters: <br>'
            _detailsStr += '<table class="hyperparameterstable" border="1">';
            _detailsStr += '<tr><th>Hyperparameter</th><th>Value</th></tr>';
            _detailsStr += Object.entries(allHyperparameters)
                .map(([key, value]) => `<tr><td>${key}</td><td>${value !== null && value !== undefined ? value : 'N/A'}</td></tr>`)
                .join("");
            _detailsStr += '</table><br>';
        }
        
        
    ///output type varies by the selected model because different tables and graphics are displayed so check output type first

        // Get output type from DOM element
        const outputTypeElement = getCachedElement('outputType1');
        const selectedOutputType = outputTypeElement ? outputTypeElement.value : 'Numeric'; // Default to Numeric if not found
        
        //columnSelection.style.display = 'none';
        // Ensure results container is visible for the current mode
        if (resultsContainer) {
            resultsContainer.style.display = 'block';
            resultsContainer.style.visibility = 'visible';
            resultsContainer.classList.remove('hidden');
        }
        
        // Hide all result divs first, then show the appropriate one
        if (NumericResultDiv) NumericResultDiv.classList.add('hidden');
        if (ClusterResultDiv) ClusterResultDiv.classList.add('hidden');
        if (ClassifierResultDiv) ClassifierResultDiv.classList.add('hidden');
        
        // Show placeholder when no results are displayed
        if (resultsPlaceholder) resultsPlaceholder.style.display = 'block';
        
        // Check if advanced options were used - look for advanced visuals or advanced option data
        const allRegressionVisualsCheck = data.regression_visuals || [];
        const hasAdvancedVisuals = allRegressionVisualsCheck.some(v => v.type === 'advanced');
        const hasAdvancedOptions = data.feature_selection_info || data.outlier_info || hasAdvancedVisuals;
        
        if (selectedOutputType === 'Numeric'){
                // Ensure results container is visible
                if (resultsContainer) {
                    resultsContainer.style.display = 'block';
                    resultsContainer.style.visibility = 'visible';
                    resultsContainer.classList.remove('hidden');
                }
                // Hide placeholder and show results
                if (resultsPlaceholder) resultsPlaceholder.style.display = 'none';
                if (NumericResultDiv) NumericResultDiv.classList.remove('hidden')

                //if multiple targets then need to let users select which graphic they want to see for each target
                // Check if we're on Simple Modeling page (not Advanced Modeling)
                const isSimpleModelingPage = !document.getElementById('advancedOptimization') || document.getElementById('advancedOptimization').classList.contains('hidden');
                const crossValidationButton = isSimpleModelingPage ? '' : (data.cross_validation_file ? `
                            <a href="/download/${data.cross_validation_file}?download_name=${encodeURIComponent(crossValidationDownloadName)}" onclick="return downloadFile('${data.cross_validation_file}', '${crossValidationDownloadName}')">
                                <button type="button" class='downloadperformanceButton export-button'>Cross-Validation XLSX</button>
                            </a>
                        ` : `
                            <button type="button" class='downloadperformanceButton export-button export-button--muted' onclick="showCrossValidationUnavailable()">Cross-Validation XLSX</button>
                        `);
                if (actualPredictorCols.length > 1){
                    const allRegressionVisuals = data.regression_visuals || [
                        { label: 'Predicted vs Actual + Residuals', file: 'target_plot' },
                    ];
                    
                    // Filter visuals: baseline only for Modeling page, advanced only for Advanced Optimization page
                    const baselineVisuals = allRegressionVisuals.filter(v => v.type === 'baseline' || !v.type || v.type === 'default');
                    const advancedVisuals = allRegressionVisuals.filter(v => v.type === 'advanced');
                    
                    // Use baseline visuals for Modeling page
                    const regressionVisuals = baselineVisuals.length > 0 ? baselineVisuals : allRegressionVisuals;
                    // Remove "Baseline" and any combined-view options for Simple Modeling page
                    const regressionVisualsClean = regressionVisuals
                        .filter(v => !/combined/i.test(v.label || ''))
                        .map(v => ({
                            ...v,
                            label: v.label.replace(/\s*-\s*Baseline\s*$/i, '').trim()
                        }));
                    // Build hyperparameter table HTML using merged hyperparameters (without wrapper for Simple Modeling page)
                    const hyperparameterTableHtml = Object.keys(allHyperparameters).length > 0 ? `
                        <table class="stats-table model-stats-table">
                            <tr><th>Hyperparameter</th><th>Value</th></tr>
                            ${Object.entries(allHyperparameters).map(([key, value]) => `<tr><td>${key}</td><td>${value !== null && value !== undefined ? value : 'N/A'}</td></tr>`).join('')}
                        </table>` : '<p>No hyperparameters to display</p>';
                    
                    // Build hyperparameter table HTML with wrapper for Advanced Modeling page
                    const hyperparameterTableHtmlWithWrapper = Object.keys(allHyperparameters).length > 0 ? `
                        <div class="model-stats-table-wrapper">
                            <table class="stats-table model-stats-table">
                                <tr><th>Hyperparameter</th><th>Value</th></tr>
                                ${Object.entries(allHyperparameters).map(([key, value]) => `<tr><td>${key}</td><td>${value !== null && value !== undefined ? value : 'N/A'}</td></tr>`).join('')}
                            </table>
                        </div>` : '<p>No hyperparameters to display</p>';
                    
                    // Build cross validation table HTML
                    const cvTableHtml = data.cross_validation_summary && data.cross_validation_summary.length > 0 ? `
                        <div class="model-stats-table-wrapper">
                            <table class="stats-table model-stats-table">
                                <tr><th>Metric</th><th>Mean</th><th>Std</th></tr>
                                ${data.cross_validation_summary.map(row => `<tr><td>${row.Metric || row.metric || ''}</td><td>${row.Mean || row.mean || ''}</td><td>${row.Std || row.std || ''}</td></tr>`).join('')}
                            </table>
                        </div>` : '<p>No cross-validation data available</p>';
                    
                    // Build feature selection table HTML
                    const featureSelectionTableHtml = data.feature_selection_info ? `
                        <div class="model-stats-table-wrapper">
                            <table class="stats-table model-stats-table">
                                <tr><th>Property</th><th>Value</th></tr>
                                <tr><td>Method</td><td>${data.feature_selection_info.method || 'N/A'}</td></tr>
                                <tr><td>K Requested</td><td>${data.feature_selection_info.k_requested || 'N/A'}</td></tr>
                                <tr><td>Original Features</td><td>${data.feature_selection_info.original_count || 'N/A'}</td></tr>
                                <tr><td>Selected Features</td><td>${data.feature_selection_info.selected_count || 'N/A'}</td></tr>
                                ${data.feature_selection_info.selected_features && data.feature_selection_info.selected_features.length > 0 ? 
                                    `<tr><td colspan="2"><strong>Selected Feature Names:</strong><br>${data.feature_selection_info.selected_features.join(', ')}</td></tr>` : ''}
                            </table>
                        </div>` : '<p>No feature selection data available</p>';
                    
                    // Build outlier handling table HTML
                    const outlierHandlingTableHtml = data.outlier_info ? `
                        <div class="model-stats-table-wrapper">
                            <table class="stats-table model-stats-table">
                                <tr><th>Property</th><th>Value</th></tr>
                                <tr><td>Method</td><td>${data.outlier_info.method || 'N/A'}</td></tr>
                                <tr><td>Action</td><td>${data.outlier_info.action || 'N/A'}</td></tr>
                                <tr><td>Outliers Detected</td><td>${data.outlier_info.n_outliers || 0}</td></tr>
                                <tr><td>Original Samples</td><td>${data.outlier_info.original_samples || 'N/A'}</td></tr>
                                <tr><td>Remaining Samples</td><td>${data.outlier_info.remaining_samples || 'N/A'}</td></tr>
                            </table>
                        </div>` : '<p>No outlier handling data available</p>';
                    
                    NumericResultDiv.innerHTML = `
                    <div class="resultValues">
                        <div style="display: flex; gap: 20px; flex-wrap: wrap; align-items: flex-start;">
                            <div style="flex: 1; min-width: 300px;">
                                <h3 style="margin: 0; margin-bottom: 10px;">Performance</h3> 
                                <div class="model-stats-table-wrapper">
                                    <table class="stats-table model-stats-table performance-table">
                                        <tr><th>Value</th><th>Training</th><th>Validation</th><th class="delta-col">Δ (Train-Validation)</th></tr>
                                        <tr> <td>n</td> <td>${data.train_n != null ? data.train_n : 'N/A'}</td> <td>${data.test_n != null ? data.test_n : 'N/A'}</td> <td class="delta-col">${data.train_n != null && data.test_n != null ? (data.train_n - data.test_n) : 'N/A'}</td> </tr>
                                        <tr> <td>R²</td> <td>${data.trainscore}</td> <td>${data.valscore}</td> <td class="delta-col">${formatDelta(data.trainscore, data.valscore)}</td> </tr>
                                        <tr> <td>RMSE</td> <td>${data.trainrmse}  ${unitStr}</td> <td>${data.valrmse}  ${unitStr}</td> <td class="delta-col">${formatDelta(data.trainrmse, data.valrmse, unitStr)}</td> </tr>
                                        ${data.trainrmsestd && data.trainrmsestd !== 'N/A' && data.valrmsestd && data.valrmsestd !== 'N/A' ? `<tr> <td>RMSE σ</td> <td>${data.trainrmsestd}  ${unitStr}</td> <td>${data.valrmsestd}  ${unitStr}</td> <td class="delta-col">${formatDelta(data.trainrmsestd, data.valrmsestd, unitStr)}</td> </tr>` : ''}
                                        <tr> <td>MAE</td> <td>${data.trainmae}  ${unitStr}</td> <td>${data.valmae} ${unitStr}</td> <td class="delta-col">${formatDelta(data.trainmae, data.valmae, unitStr)}</td> </tr>
                                        ${data.trainmaestd && data.trainmaestd !== 'N/A' && data.valmaestd && data.valmaestd !== 'N/A' ? `<tr> <td>MAE σ</td> <td>${data.trainmaestd}  ${unitStr}</td> <td>${data.valmaestd} ${unitStr}</td> <td class="delta-col">${formatDelta(data.trainmaestd, data.valmaestd, unitStr)}</td> </tr>` : ''}
                                    </table>
                                </div>
                                <div class="download-buttons" style="margin-top: 12px; display: flex; gap: 12px; align-items: center;">
                                    <a href="/download/model_performance.xlsx?download_name=${encodeURIComponent(performanceDownloadName)}" onclick="return downloadFile('model_performance.xlsx', '${performanceDownloadName}')">
                                        <button type="button" class='downloadperformanceButton export-button'>Model Performance XLSX</button>
                                    </a>
                                    <a href="/download/visualizations.pdf?download_name=${encodeURIComponent(visualizationsDownloadName)}" onclick="return downloadFile('visualizations.pdf', '${visualizationsDownloadName}')">
                                        <button class="export-button" style="font-size: 0.95rem;">Visualizations PDF</button>
                                    </a>
                                    ${crossValidationButton}
                                </div>
                            </div>
                            <div style="flex: 1; min-width: 300px;">
                                ${currentMode === 'simple' ? `
                                <h3 style="margin: 0; margin-bottom: 10px;">Hyperparameters</h3>
                                <div class="model-stats-table-wrapper">
                                    ${hyperparameterTableHtml}
                                </div>
                                ` : `
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; gap: 12px;">
                                    <h3 style="margin: 0;">Additional Information</h3>
                                    <div style="display: flex; align-items: center; gap: 8px;">
                                        <label for="${currentMode === 'advanced' ? 'advancedAdditionalTableToggle' : 'automlAdditionalTableToggle'}" style="margin: 0; white-space: nowrap;">Select:</label>
                                        <select id="${currentMode === 'advanced' ? 'advancedAdditionalTableToggle' : 'automlAdditionalTableToggle'}" style="margin: 0; width: 200px; padding: 4px 8px;">
                                            <option value="hyperparameters">Hyperparameters</option>
                                            <option value="crossvalidation" ${data.cross_validation_summary && data.cross_validation_summary.length > 0 ? '' : 'disabled'}>Cross Validation</option>
                                            <option value="featureselection" ${data.feature_selection_info ? '' : 'disabled'}>Feature Selection</option>
                                            <option value="outlierhandling" ${data.outlier_info ? '' : 'disabled'}>Outlier Handling</option>
                                        </select>
                                    </div>
                                </div>
                                <div id="${currentMode === 'advanced' ? 'advancedAdditionalTableContent' : 'automlAdditionalTableContent'}">
                                    ${hyperparameterTableHtmlWithWrapper}
                                </div>
                                `}
                            </div>
                        </div>
                            <br>
                            <br>
                        </div>

                    <div class="results-header" style="margin-top: 24px; margin-bottom: 16px;">
                        <h2>Modeling Results</h2>
                        <p>Charts, tables, and downloads will appear here.</p>
                    </div>
                    <div class="result-graphic-row-train-test" style="display: flex; gap: 24px; flex-wrap: wrap; align-items: flex-start;">
                        <div class="result-graphic-box" style="flex: 1; min-width: 320px;">
                            <h4 style="margin: 0 0 8px 0; font-size: 1.1rem;">Visualization 1</h4>
                            <label for="${currentMode === 'simple' ? 'regressionVisualSelector' : currentMode === 'advanced' ? 'advancedRegressionVisualSelector' : 'automlRegressionVisualSelector'}">Select visualization</label>
                            <select id="${currentMode === 'simple' ? 'regressionVisualSelector' : currentMode === 'advanced' ? 'advancedRegressionVisualSelector' : 'automlRegressionVisualSelector'}" style="max-width: 360px;">
                                ${regressionVisualsClean
                                    .map((visual) => `<option value="${visual.file}">${visual.label}</option>`)
                                    .join('')}
                            </select>
                            <label for="${currentMode === 'simple' ? 'imageSelector' : currentMode === 'advanced' ? 'advancedImageSelector' : 'automlImageSelector'}">Select target</label>
                            <select id="${currentMode === 'simple' ? 'imageSelector' : currentMode === 'advanced' ? 'advancedImageSelector' : 'automlImageSelector'}"></select>
                            <br><br>
                            <img id="${currentMode === 'simple' ? 'targetGraphic' : currentMode === 'advanced' ? 'advancedTargetGraphic' : 'automlTargetGraphic'}" class="result-graphic" src='/user-visualizations/target_plot_1${currentMode === 'advanced' ? '_advanced' : ''}.png?t=${new Date().getTime()}' alt="Model visualization 1">
                        </div>
                        <div class="result-graphic-box" style="flex: 1; min-width: 320px;">
                            <h4 style="margin: 0 0 8px 0; font-size: 1.1rem;">Visualization 2</h4>
                            <label for="${currentMode === 'simple' ? 'regressionVisualSelector2' : currentMode === 'advanced' ? 'advancedRegressionVisualSelector2' : 'automlRegressionVisualSelector2'}">Select visualization</label>
                            <select id="${currentMode === 'simple' ? 'regressionVisualSelector2' : currentMode === 'advanced' ? 'advancedRegressionVisualSelector2' : 'automlRegressionVisualSelector2'}" style="max-width: 360px;">
                                ${regressionVisualsClean
                                    .map((visual) => `<option value="${visual.file}">${visual.label}</option>`)
                                    .join('')}
                            </select>
                            <label for="${currentMode === 'simple' ? 'imageSelector2' : currentMode === 'advanced' ? 'advancedImageSelector2' : 'automlImageSelector2'}">Select target</label>
                            <select id="${currentMode === 'simple' ? 'imageSelector2' : currentMode === 'advanced' ? 'advancedImageSelector2' : 'automlImageSelector2'}"></select>
                            <br><br>
                            <img id="${currentMode === 'simple' ? 'targetGraphic2' : currentMode === 'advanced' ? 'advancedTargetGraphic2' : 'automlTargetGraphic2'}" class="result-graphic" src='/user-visualizations/target_plot_1${currentMode === 'advanced' ? '_advanced' : ''}.png?t=${new Date().getTime()}' alt="Model visualization 2">
                        </div>
                    </div>
                    <div><br></div>
                    </div>


                    `;
                    // No table toggle needed for Simple Modeling page - hyperparameters are displayed directly
                    
                    // Set up table toggle event listeners for advanced/automl modes (multiple targets)
                    if (currentMode === 'advanced' || currentMode === 'automl') {
                        // Store HTML strings for table switching (closure to preserve access)
                        const tableHtmls = {
                            hyperparameters: hyperparameterTableHtmlWithWrapper,
                            crossvalidation: cvTableHtml,
                            featureselection: featureSelectionTableHtml,
                            outlierhandling: outlierHandlingTableHtml
                        };
                        
                        // Set up event listeners immediately after HTML is inserted
                        const toggleId = currentMode === 'advanced' ? 'advancedAdditionalTableToggle' : 'automlAdditionalTableToggle';
                        const contentId = currentMode === 'advanced' ? 'advancedAdditionalTableContent' : 'automlAdditionalTableContent';
                        
                        const tableToggle = document.getElementById(toggleId);
                        const tableContent = document.getElementById(contentId);
                        if (tableToggle && tableContent) {
                            tableToggle.addEventListener('change', function() {
                                const selectedValue = this.value;
                                if (tableHtmls[selectedValue]) {
                                    tableContent.innerHTML = tableHtmls[selectedValue];
                                }
                            });
                        }
                    }
                    
                    // Populate dropdowns and wire two side-by-side visualization panels (multiple targets)
                    if (!imageSelector) {
                        imageSelector = document.getElementById("imageSelector") || 
                                       document.getElementById("advancedImageSelector") || 
                                       document.getElementById("automlImageSelector");
                    }
                    const imageSelector2 = document.getElementById(currentMode === 'simple' ? 'imageSelector2' : currentMode === 'advanced' ? 'advancedImageSelector2' : 'automlImageSelector2');
                    const targetGraphicId = currentMode === 'simple' ? 'targetGraphic' : currentMode === 'advanced' ? 'advancedTargetGraphic' : 'automlTargetGraphic';
                    const targetGraphicId2 = currentMode === 'simple' ? 'targetGraphic2' : currentMode === 'advanced' ? 'advancedTargetGraphic2' : 'automlTargetGraphic2';
                    const visualSelectorId = currentMode === 'simple' ? 'regressionVisualSelector' : currentMode === 'advanced' ? 'advancedRegressionVisualSelector' : 'automlRegressionVisualSelector';
                    const visualSelectorId2 = currentMode === 'simple' ? 'regressionVisualSelector2' : currentMode === 'advanced' ? 'advancedRegressionVisualSelector2' : 'automlRegressionVisualSelector2';
                    let targetGraphic = document.getElementById(targetGraphicId);
                    let targetGraphic2 = document.getElementById(targetGraphicId2);
                    const regressionVisualSelector = document.getElementById(visualSelectorId);
                    const regressionVisualSelector2 = document.getElementById(visualSelectorId2);

                    data.predictors.forEach((predictor, index) => {
                        const option = document.createElement("option");
                        option.value = index + 1;
                        option.textContent = predictor.split('/').pop();
                        imageSelector.appendChild(option);
                        if (imageSelector2) {
                            const option2 = document.createElement("option");
                            option2.value = index + 1;
                            option2.textContent = predictor.split('/').pop();
                            imageSelector2.appendChild(option2);
                        }
                    });

                    const buildRegressionGraphicUrl = (selectedVisual, selectedImage) => {
                        const visualObj = regressionVisuals.find(v => v.file === selectedVisual);
                        const _visualType = visualObj ? visualObj.type : 'default';
                        const perTargetBases = ['target_plot', 'target_plot_pred_actual', 'target_plot_residuals'];
                        const base = selectedVisual.replace(/_advanced$/, '');
                        const suffix = selectedVisual.endsWith('_advanced') ? '_advanced' : '';
                        if (perTargetBases.includes(base)) {
                            return withApiRoot(`/user-visualizations/${base}_${selectedImage}${suffix}.png?t=${Date.now()}`);
                        }
                        let filename = selectedVisual;
                        if (!filename.includes('.png')) {
                            filename = filename.endsWith('_advanced') ? `${filename}.png` : `${filename}.png`;
                        }
                        return withApiRoot(`/user-visualizations/${filename}?t=${Date.now()}`);
                    };

                    const updateRegressionGraphic = (panel) => {
                        const sel = panel === 1 ? regressionVisualSelector : regressionVisualSelector2;
                        const imgSel = panel === 1 ? imageSelector : imageSelector2;
                        const img = panel === 1 ? targetGraphic : targetGraphic2;
                        if (!img) return;
                        const selectedVisual = sel ? sel.value : 'target_plot';
                        const selectedImage = imgSel ? imgSel.value : '1';
                        img.src = buildRegressionGraphicUrl(selectedVisual, selectedImage);
                    };
                    if (targetGraphic) targetGraphic.onerror = function() { console.error('Failed to load graphic:', this.src); };
                    if (targetGraphic2) targetGraphic2.onerror = function() { console.error('Failed to load graphic:', this.src); };

                    const defaultPa = regressionVisualsClean.find(v => v.label.includes('Predicted vs Actual') && !v.label.includes('combined'))?.file;
                    const defaultRes = regressionVisualsClean.find(v => v.label.includes('Test Residuals'))?.file;
                    if (defaultPa && regressionVisualSelector) regressionVisualSelector.value = defaultPa;
                    if (defaultRes && regressionVisualSelector2) regressionVisualSelector2.value = defaultRes;
                    imageSelector.addEventListener("change", () => updateRegressionGraphic(1));
                    if (regressionVisualSelector) regressionVisualSelector.addEventListener("change", () => updateRegressionGraphic(1));
                    if (imageSelector2) imageSelector2.addEventListener("change", () => updateRegressionGraphic(2));
                    if (regressionVisualSelector2) regressionVisualSelector2.addEventListener("change", () => updateRegressionGraphic(2));
                    updateRegressionGraphic(1);
                    updateRegressionGraphic(2);
                    
                    // Note: Advanced results are now shown in the Simple mode result divs above
                    // Removed separate AdvancedNumericResultDiv population since all results use Simple mode divs
                    if (false && hasAdvancedOptions && AdvancedNumericResultDiv && advancedVisuals.length > 0) {
                        // This code is disabled - all results now show in Simple mode divs
                        AdvancedNumericResultDiv.innerHTML = `
                    <div class="resultValues">
                        <div style="display: flex; gap: 20px; flex-wrap: wrap; align-items: flex-start;">
                            <div style="flex: 1; min-width: 300px;">
                                <h3 style="margin: 0; margin-bottom: 10px;">Performance</h3> 
                                <div class="model-stats-table-wrapper">
                                    <table class="stats-table model-stats-table performance-table">
                                        <tr><th>Value</th><th>Training</th><th>Validation</th><th class="delta-col">Δ (Train-Validation)</th></tr>
                                        <tr> <td>n</td> <td>${data.train_n != null ? data.train_n : 'N/A'}</td> <td>${data.test_n != null ? data.test_n : 'N/A'}</td> <td class="delta-col">${data.train_n != null && data.test_n != null ? (data.train_n - data.test_n) : 'N/A'}</td> </tr>
                                        <tr> <td>R²</td> <td>${data.trainscore}</td> <td>${data.valscore}</td> <td class="delta-col">${formatDelta(data.trainscore, data.valscore)}</td> </tr>
                                        <tr> <td>RMSE</td> <td>${data.trainrmse}  ${unitStr}</td> <td>${data.valrmse}  ${unitStr}</td> <td class="delta-col">${formatDelta(data.trainrmse, data.valrmse, unitStr)}</td> </tr>
                                        ${data.trainrmsestd && data.trainrmsestd !== 'N/A' && data.valrmsestd && data.valrmsestd !== 'N/A' ? `<tr> <td>RMSE σ</td> <td>${data.trainrmsestd}  ${unitStr}</td> <td>${data.valrmsestd}  ${unitStr}</td> <td class="delta-col">${formatDelta(data.trainrmsestd, data.valrmsestd, unitStr)}</td> </tr>` : ''}
                                        <tr> <td>MAE</td> <td>${data.trainmae}  ${unitStr}</td> <td>${data.valmae} ${unitStr}</td> <td class="delta-col">${formatDelta(data.trainmae, data.valmae, unitStr)}</td> </tr>
                                        ${data.trainmaestd && data.trainmaestd !== 'N/A' && data.valmaestd && data.valmaestd !== 'N/A' ? `<tr> <td>MAE σ</td> <td>${data.trainmaestd}  ${unitStr}</td> <td>${data.valmaestd} ${unitStr}</td> <td class="delta-col">${formatDelta(data.trainmaestd, data.valmaestd, unitStr)}</td> </tr>` : ''}
                                    </table>
                                </div>
                                <div class="download-buttons" style="margin-top: 12px; display: flex; gap: 12px; align-items: center;">
                                    <a href="/download/model_performance.xlsx?download_name=${encodeURIComponent(performanceDownloadName)}" onclick="return downloadFile('model_performance.xlsx', '${performanceDownloadName}')">
                                        <button type="button" class='downloadperformanceButton export-button'>Model Performance XLSX</button>
                                    </a>
                                    <a href="/download/visualizations.pdf?download_name=${encodeURIComponent(visualizationsDownloadName)}" onclick="return downloadFile('visualizations.pdf', '${visualizationsDownloadName}')">
                                        <button class="export-button" style="font-size: 0.95rem;">Visualizations PDF</button>
                                    </a>
                                    ${crossValidationButton}
                                </div>
                            </div>
                            <div style="flex: 1; min-width: 300px;">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                                    <h3 style="margin: 0;">Additional Information</h3>
                                    <button type="button" class="export-button" id="advancedDownloadAdditionalInfo" style="font-size: 0.9rem; padding: 6px 12px;">Download XLSX</button>
                                </div>
                                <label for="advancedAdditionalTableToggle" style="display: block; margin-bottom: 5px;">Select Table to Display:</label>
                                <select id="advancedAdditionalTableToggle" style="margin-bottom: 10px; width: 100%;">
                                    <option value="hyperparameters">Hyperparameters</option>
                                    <option value="crossvalidation" ${data.cross_validation_summary && data.cross_validation_summary.length > 0 ? '' : 'disabled'}>Cross Validation</option>
                                    <option value="featureselection" ${data.feature_selection_info ? '' : 'disabled'}>Feature Selection</option>
                                    <option value="outlierhandling" ${data.outlier_info ? '' : 'disabled'}>Outlier Handling</option>
                                </select>
                                <div id="advancedAdditionalTableContent">
                                    ${hyperparameterTableHtmlWithWrapper}
                                </div>
                            </div>
                        </div>
                            <br>
                            <br>
                        </div>

                    <div class="results-header" style="margin-top: 24px; margin-bottom: 16px;">
                        <h2>Advanced Modeling Results</h2>
                        <p>Charts, tables, and downloads will appear here.</p>
                    </div>
                    <label for="advancedRegressionVisualSelector">Select Visualization to Display</label>
                    <select id="advancedRegressionVisualSelector">
                        ${advancedVisuals
                            .map((visual) => `<option value="${visual.file}">${visual.label}</option>`)
                            .join('')}
                    </select>
                    <label for="advancedImageSelector">Select Target Graphic to Display</label>
                    <select id="advancedImageSelector"></select>
                    <br>
                    <br>
                    <img id="advancedTargetGraphic" class="result-graphic" src='/user-visualizations/target_plot_1_advanced.png?t=${new Date().getTime()}' alt="Advanced model visualization">
                        <div><br></div>
                        
                        <div><br></div>
                    </div>
                        `;
                        
                        // Set up event listeners for advanced results
                        const advancedImageSelector = document.getElementById('advancedImageSelector');
                        const advancedTargetGraphic = document.getElementById('advancedTargetGraphic');
                        const advancedVisualSelector = document.getElementById('advancedRegressionVisualSelector');
                        
                        if (advancedImageSelector && data.predictors) {
                            data.predictors.forEach((predictor, index) => {
                                const option = document.createElement("option");
                                option.value = index + 1;
                                option.textContent = predictor.split('/').pop();
                                advancedImageSelector.appendChild(option);
                            });
                        }
                        
                        if (advancedVisualSelector && advancedTargetGraphic) {
                            const updateAdvancedGraphic = () => {
                                const selectedImage = advancedImageSelector ? advancedImageSelector.value : '1';
                                const selectedVisual = advancedVisualSelector.value;
                                
                                const visualObj = advancedVisuals.find(v => v.file === selectedVisual);
                                const visualType = visualObj ? visualObj.type : 'default';
                                
                                if (selectedVisual !== 'target_plot' && selectedVisual !== 'target_plot_advanced') {
                                    let filename = selectedVisual;
                                    if (!filename.includes('.png') && !filename.includes('_advanced')) {
                                        filename = filename.endsWith('_advanced') ? `${filename}.png` : `${filename}.png`;
                                    } else if (filename.endsWith('_advanced') && !filename.includes('.png')) {
                                        filename = `${filename}.png`;
                                    }
                                    advancedTargetGraphic.src = withApiRoot(`/user-visualizations/${filename}?t=${Date.now()}`);
                                    return;
                                }
                                
                                let suffix = '';
                                if (selectedVisual === 'target_plot_advanced' || (visualType === 'advanced')) {
                                    suffix = '_advanced';
                                }
                                
                                advancedTargetGraphic.src = withApiRoot(`/user-visualizations/target_plot_${selectedImage}${suffix}.png?t=${Date.now()}`);
                            };
                            
                            if (advancedImageSelector) {
                                advancedImageSelector.addEventListener("change", updateAdvancedGraphic);
                            }
                            advancedVisualSelector.addEventListener("change", updateAdvancedGraphic);
                            updateAdvancedGraphic();
                        }
                    }
                }

                //when only one target to display 
                //there is no dropdown to select which target's graphic to display 
                //and SHAP graphic can be displayed
                else{
                    const allRegressionVisualsSingle = data.regression_visuals || [
                        { label: 'Predicted vs Actual + Residuals', file: 'target_plot' },
                    ];
                    
                    // Filter visuals: baseline only for Modeling page, advanced for Advanced Optimization page
                    const regressionVisuals = allRegressionVisualsSingle.filter(v => v.type === 'baseline' || !v.type || v.type === 'default');
                    // Remove "Baseline" and any combined-view options for Simple Modeling page
                    const regressionVisualsClean = regressionVisuals
                        .filter(v => !/combined/i.test(v.label || ''))
                        .map(v => ({
                            ...v,
                            label: v.label.replace(/\s*-\s*Baseline\s*$/i, '').trim()
                        }));
                    const advancedVisualsSingle = allRegressionVisualsSingle.filter(v => v.type === 'advanced');
                    // Build hyperparameter table HTML for single target using merged hyperparameters (without wrapper for Simple Modeling page)
                    const hyperparameterTableHtmlSingle = Object.keys(allHyperparameters).length > 0 ? `
                        <table class="stats-table model-stats-table">
                            <tr><th>Hyperparameter</th><th>Value</th></tr>
                            ${Object.entries(allHyperparameters).map(([key, value]) => `<tr><td>${key}</td><td>${value !== null && value !== undefined ? value : 'N/A'}</td></tr>`).join('')}
                        </table>` : '<p>No hyperparameters to display</p>';
                    
                    // Build hyperparameter table HTML for single target with wrapper for Advanced Modeling page
                    const hyperparameterTableHtmlSingleWithWrapper = Object.keys(allHyperparameters).length > 0 ? `
                        <div class="model-stats-table-wrapper">
                            <table class="stats-table model-stats-table">
                                <tr><th>Hyperparameter</th><th>Value</th></tr>
                                ${Object.entries(allHyperparameters).map(([key, value]) => `<tr><td>${key}</td><td>${value !== null && value !== undefined ? value : 'N/A'}</td></tr>`).join('')}
                            </table>
                        </div>` : '<p>No hyperparameters to display</p>';
                    
                    // Build cross validation table HTML for single target
                    const cvTableHtmlSingle = data.cross_validation_summary && data.cross_validation_summary.length > 0 ? `
                        <div class="model-stats-table-wrapper">
                            <table class="stats-table model-stats-table">
                                <tr><th>Metric</th><th>Mean</th><th>Std</th></tr>
                                ${data.cross_validation_summary.map(row => `<tr><td>${row.Metric || row.metric || ''}</td><td>${row.Mean || row.mean || ''}</td><td>${row.Std || row.std || ''}</td></tr>`).join('')}
                            </table>
                        </div>` : '<p>No cross-validation data available</p>';
                    
                    // Build feature selection table HTML for single target
                    const featureSelectionTableHtmlSingle = data.feature_selection_info ? `
                        <div class="model-stats-table-wrapper">
                            <table class="stats-table model-stats-table">
                                <tr><th>Property</th><th>Value</th></tr>
                                <tr><td>Method</td><td>${data.feature_selection_info.method || 'N/A'}</td></tr>
                                <tr><td>K Requested</td><td>${data.feature_selection_info.k_requested || 'N/A'}</td></tr>
                                <tr><td>Original Features</td><td>${data.feature_selection_info.original_count || 'N/A'}</td></tr>
                                <tr><td>Selected Features</td><td>${data.feature_selection_info.selected_count || 'N/A'}</td></tr>
                                ${data.feature_selection_info.selected_features && data.feature_selection_info.selected_features.length > 0 ? 
                                    `<tr><td colspan="2"><strong>Selected Feature Names:</strong><br>${data.feature_selection_info.selected_features.join(', ')}</td></tr>` : ''}
                            </table>
                        </div>` : '<p>No feature selection data available</p>';
                    
                    // Build outlier handling table HTML for single target
                    const outlierHandlingTableHtmlSingle = data.outlier_info ? `
                        <div class="model-stats-table-wrapper">
                            <table class="stats-table model-stats-table">
                                <tr><th>Property</th><th>Value</th></tr>
                                <tr><td>Method</td><td>${data.outlier_info.method || 'N/A'}</td></tr>
                                <tr><td>Action</td><td>${data.outlier_info.action || 'N/A'}</td></tr>
                                <tr><td>Outliers Detected</td><td>${data.outlier_info.n_outliers || 0}</td></tr>
                                <tr><td>Original Samples</td><td>${data.outlier_info.original_samples || 'N/A'}</td></tr>
                                <tr><td>Remaining Samples</td><td>${data.outlier_info.remaining_samples || 'N/A'}</td></tr>
                            </table>
                        </div>` : '<p>No outlier handling data available</p>';
                    
                    NumericResultDiv.innerHTML = `
                
                <div class="resultValues">
                    <div style="display: flex; gap: 20px; flex-wrap: wrap; align-items: flex-start;">
                        <div style="flex: 1; min-width: 300px;">
                            <h3 style="margin: 0; margin-bottom: 10px;">Performance</h3> 
                            <div class="model-stats-table-wrapper">
                                <table class="stats-table model-stats-table performance-table">
                                    <tr><th>Value</th><th>Training</th><th>Validation</th><th class="delta-col">Δ (Train-Validation)</th></tr>
                                    <tr> <td>n</td> <td>${data.train_n != null ? data.train_n : 'N/A'}</td> <td>${data.test_n != null ? data.test_n : 'N/A'}</td> <td class="delta-col">${data.train_n != null && data.test_n != null ? (data.train_n - data.test_n) : 'N/A'}</td> </tr>
                                    <tr> <td>R²</td> <td>${data.trainscore}</td> <td>${data.valscore}</td> <td class="delta-col">${formatDelta(data.trainscore, data.valscore)}</td> </tr>
                                    <tr> <td>RMSE</td> <td>${data.trainrmse} ${unitStr}</td> <td>${data.valrmse}  ${unitStr}</td> <td class="delta-col">${formatDelta(data.trainrmse, data.valrmse, unitStr)}</td> </tr>
                                    ${data.trainrmsestd && data.trainrmsestd !== 'N/A' && data.valrmsestd && data.valrmsestd !== 'N/A' ? `<tr> <td>RMSE σ</td> <td>${data.trainrmsestd} ${unitStr}</td> <td>${data.valrmsestd}  ${unitStr}</td> <td class="delta-col">${formatDelta(data.trainrmsestd, data.valrmsestd, unitStr)}</td> </tr>` : ''}
                                    <tr> <td>MAE</td> <td>${data.trainmae} ${unitStr}</td> <td>${data.valmae} ${unitStr}</td> <td class="delta-col">${formatDelta(data.trainmae, data.valmae, unitStr)}</td> </tr>
                                    ${data.trainmaestd && data.trainmaestd !== 'N/A' && data.valmaestd && data.valmaestd !== 'N/A' ? `<tr> <td>MAE σ</td> <td>${data.trainmaestd} ${unitStr}</td> <td>${data.valmaestd} ${unitStr}</td> <td class="delta-col">${formatDelta(data.trainmaestd, data.valmaestd, unitStr)}</td> </tr>` : ''}
                                </table>
                            </div>
                            <div class="download-buttons" style="margin-top: 12px; display: flex; gap: 12px; align-items: center;">
                                <a href="/download/model_performance.xlsx?download_name=${encodeURIComponent(performanceDownloadName)}" onclick="return downloadFile('model_performance.xlsx', '${performanceDownloadName}')">
                                    <button type="button" class='downloadperformanceButton export-button'>Model Performance XLSX</button>
                                </a>
                                <a href="/download/visualizations.pdf?download_name=${encodeURIComponent(visualizationsDownloadName)}" onclick="return downloadFile('visualizations.pdf', '${visualizationsDownloadName}')">
                                    <button class="export-button" style="font-size: 0.95rem;">Visualizations PDF</button>
                                </a>
                                ${crossValidationButton}
                            </div>
                        </div>
                        <div style="flex: 1; min-width: 300px;">
                            ${currentMode === 'simple' ? `
                            <h3 style="margin: 0; margin-bottom: 10px;">Hyperparameters</h3>
                            <div class="model-stats-table-wrapper">
                                ${hyperparameterTableHtmlSingle}
                            </div>
                            ` : `
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; gap: 12px;">
                                <h3 style="margin: 0;">Additional Information</h3>
                                <div style="display: flex; align-items: center; gap: 8px;">
                                    <label for="${currentMode === 'advanced' ? 'advancedAdditionalTableToggleSingle' : 'automlAdditionalTableToggleSingle'}" style="margin: 0; white-space: nowrap;">Select:</label>
                                    <select id="${currentMode === 'advanced' ? 'advancedAdditionalTableToggleSingle' : 'automlAdditionalTableToggleSingle'}" style="margin: 0; width: 200px; padding: 4px 8px;">
                                        <option value="hyperparameters">Hyperparameters</option>
                                        <option value="crossvalidation" ${data.cross_validation_summary && data.cross_validation_summary.length > 0 ? '' : 'disabled'}>Cross Validation</option>
                                        <option value="featureselection" ${data.feature_selection_info ? '' : 'disabled'}>Feature Selection</option>
                                        <option value="outlierhandling" ${data.outlier_info ? '' : 'disabled'}>Outlier Handling</option>
                                    </select>
                                </div>
                            </div>
                            <div id="${currentMode === 'advanced' ? 'advancedAdditionalTableContentSingle' : 'automlAdditionalTableContentSingle'}">
                                ${hyperparameterTableHtmlSingleWithWrapper}
                            </div>
                            `}
                        </div>
                    </div>
                        <br>
                        <br>
                    </div>

                    <div class="results-header" style="margin-top: 24px; margin-bottom: 16px;">
                        <h2>Modeling Results</h2>
                        <p>Charts, tables, and downloads will appear here.</p>
                    </div>
                    <div class="result-graphic-row-train-test" style="display: flex; gap: 24px; flex-wrap: wrap; align-items: flex-start;">
                        <div class="result-graphic-box" style="flex: 1; min-width: 320px;">
                            <h4 style="margin: 0 0 8px 0; font-size: 1.1rem;">Visualization 1</h4>
                            <label for="regressionVisualSelector">Select visualization</label>
                            <select id="regressionVisualSelector" style="max-width: 360px;">
                                ${regressionVisualsClean
                                    .map((visual) => `<option value="${visual.file}">${visual.label}</option>`)
                                    .join('')}
                            </select>
                            <br><br>
                            <img id="targetGraphic" class="result-graphic" src='/user-visualizations/target_plot_1.png?t=${new Date().getTime()}'>
                        </div>
                        <div class="result-graphic-box" style="flex: 1; min-width: 320px;">
                            <h4 style="margin: 0 0 8px 0; font-size: 1.1rem;">Visualization 2</h4>
                            <label for="regressionVisualSelector2">Select visualization</label>
                            <select id="regressionVisualSelector2" style="max-width: 360px;">
                                ${regressionVisualsClean
                                    .map((visual) => `<option value="${visual.file}">${visual.label}</option>`)
                                    .join('')}
                            </select>
                            <br><br>
                            <img id="targetGraphic2" class="result-graphic" src='/user-visualizations/target_plot_1.png?t=${new Date().getTime()}'>
                        </div>
                    </div>
                    <div><br></div>
                </div>


                `;
                // No table toggle needed for Simple Modeling page - hyperparameters are displayed directly
                
                // Set up table toggle event listeners for advanced/automl modes (single target)
                if (currentMode === 'advanced' || currentMode === 'automl') {
                    // Store HTML strings for table switching (closure to preserve access)
                    const tableHtmlsSingle = {
                        hyperparameters: hyperparameterTableHtmlSingleWithWrapper,
                        crossvalidation: cvTableHtmlSingle,
                        featureselection: featureSelectionTableHtmlSingle,
                        outlierhandling: outlierHandlingTableHtmlSingle
                    };
                    
                    // Set up event listeners immediately after HTML is inserted
                    const toggleId = currentMode === 'advanced' ? 'advancedAdditionalTableToggleSingle' : 'automlAdditionalTableToggleSingle';
                    const contentId = currentMode === 'advanced' ? 'advancedAdditionalTableContentSingle' : 'automlAdditionalTableContentSingle';
                    
                    const tableToggle = document.getElementById(toggleId);
                    const tableContent = document.getElementById(contentId);
                    if (tableToggle && tableContent) {
                        tableToggle.addEventListener('change', function() {
                            const selectedValue = this.value;
                            if (tableHtmlsSingle[selectedValue]) {
                                tableContent.innerHTML = tableHtmlsSingle[selectedValue];
                            }
                        });
                    }
                }
                
                const regressionVisualSelector = document.getElementById("regressionVisualSelector");
                const regressionVisualSelector2 = document.getElementById("regressionVisualSelector2");
                const targetGraphic = document.getElementById("targetGraphic");
                const targetGraphic2 = document.getElementById("targetGraphic2");
                const buildSingleTargetUrl = (selectedVisual) => {
                    const perTargetBases = ['target_plot', 'target_plot_pred_actual', 'target_plot_residuals'];
                    const base = selectedVisual.replace(/_advanced$/, '');
                    const suffix = selectedVisual.endsWith('_advanced') ? '_advanced' : '';
                    if (perTargetBases.includes(base)) {
                        return withApiRoot(`/user-visualizations/${base}_1${suffix}.png?t=${Date.now()}`);
                    }
                    let filename = selectedVisual;
                    if (!filename.includes('.png')) {
                        filename = filename.endsWith('_advanced') ? `${filename}.png` : `${filename}.png`;
                    }
                    return withApiRoot(`/user-visualizations/${filename}?t=${Date.now()}`);
                };
                const updateSingleTargetGraphic = (panel) => {
                    const sel = panel === 1 ? regressionVisualSelector : regressionVisualSelector2;
                    const img = panel === 1 ? targetGraphic : targetGraphic2;
                    if (sel && img) img.src = buildSingleTargetUrl(sel.value);
                };
                const defaultPaSingle = regressionVisualsClean.find(v => v.label.includes('Predicted vs Actual') && !v.label.includes('combined'))?.file;
                const defaultResSingle = regressionVisualsClean.find(v => v.label.includes('Test Residuals'))?.file;
                if (defaultPaSingle && regressionVisualSelector) regressionVisualSelector.value = defaultPaSingle;
                if (defaultResSingle && regressionVisualSelector2) regressionVisualSelector2.value = defaultResSingle;
                if (regressionVisualSelector) regressionVisualSelector.addEventListener("change", () => updateSingleTargetGraphic(1));
                if (regressionVisualSelector2) regressionVisualSelector2.addEventListener("change", () => updateSingleTargetGraphic(2));
                if (targetGraphic) targetGraphic.onerror = function() { console.error('Failed to load graphic:', this.src); };
                if (targetGraphic2) targetGraphic2.onerror = function() { console.error('Failed to load graphic:', this.src); };
                updateSingleTargetGraphic(1);
                updateSingleTargetGraphic(2);
                    
                    // Note: Advanced results are now shown in the Simple mode result divs above
                    // Removed separate AdvancedNumericResultDiv population since all results use Simple mode divs
                    if (false && hasAdvancedOptions && AdvancedNumericResultDiv && advancedVisualsSingle.length > 0) {
                        // This code is disabled - all results now show in Simple mode divs
                        AdvancedNumericResultDiv.innerHTML = `
                <div class="resultValues">
                    <div style="display: flex; gap: 20px; flex-wrap: wrap; align-items: flex-start;">
                        <div style="flex: 1; min-width: 300px;">
                            <h3>Performance </h3> 
                            <div class="model-stats-table-wrapper">
                                <table class="stats-table model-stats-table performance-table">
                                    <tr><th>Value</th><th>Training</th><th>Validation</th><th class="delta-col">Δ (Train-Validation)</th></tr>
                                    <tr> <td>n</td> <td>${data.train_n != null ? data.train_n : 'N/A'}</td> <td>${data.test_n != null ? data.test_n : 'N/A'}</td> <td class="delta-col">${data.train_n != null && data.test_n != null ? (data.train_n - data.test_n) : 'N/A'}</td> </tr>
                                    <tr> <td>R²</td> <td>${data.trainscore}</td> <td>${data.valscore}</td> <td class="delta-col">${formatDelta(data.trainscore, data.valscore)}</td> </tr>
                                    <tr> <td>RMSE</td> <td>${data.trainrmse} ${unitStr}</td> <td>${data.valrmse}  ${unitStr}</td> <td class="delta-col">${formatDelta(data.trainrmse, data.valrmse, unitStr)}</td> </tr>
                                    ${data.trainrmsestd && data.trainrmsestd !== 'N/A' && data.valrmsestd && data.valrmsestd !== 'N/A' ? `<tr> <td>RMSE σ</td> <td>${data.trainrmsestd} ${unitStr}</td> <td>${data.valrmsestd}  ${unitStr}</td> <td class="delta-col">${formatDelta(data.trainrmsestd, data.valrmsestd, unitStr)}</td> </tr>` : ''}
                                    <tr> <td>MAE</td> <td>${data.trainmae} ${unitStr}</td> <td>${data.valmae} ${unitStr}</td> <td class="delta-col">${formatDelta(data.trainmae, data.valmae, unitStr)}</td> </tr>
                                    ${data.trainmaestd && data.trainmaestd !== 'N/A' && data.valmaestd && data.valmaestd !== 'N/A' ? `<tr> <td>MAE σ</td> <td>${data.trainmaestd} ${unitStr}</td> <td>${data.valmaestd} ${unitStr}</td> <td class="delta-col">${formatDelta(data.trainmaestd, data.valmaestd, unitStr)}</td> </tr>` : ''}
                                </table>
                            </div>
                            <div class="download-buttons" style="margin-top: 12px; display: flex; gap: 12px; align-items: center;">
                                <a href="/download/model_performance.xlsx?download_name=${encodeURIComponent(performanceDownloadName)}" onclick="return downloadFile('model_performance.xlsx', '${performanceDownloadName}')">
                                    <button type="button" class='downloadperformanceButton export-button'>Model Performance XLSX</button>
                                </a>
                                <a href="/download/visualizations.pdf?download_name=${encodeURIComponent(visualizationsDownloadName)}" onclick="return downloadFile('visualizations.pdf', '${visualizationsDownloadName}')">
                                    <button class="export-button" style="font-size: 0.95rem;">Visualizations PDF</button>
                                </a>
                                ${crossValidationButton}
                            </div>
                        </div>
                        <div style="flex: 1; min-width: 300px;">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                                <h3 style="margin: 0;">Additional Information</h3>
                                <button type="button" class="export-button" id="advancedDownloadAdditionalInfoSingle" style="font-size: 0.9rem; padding: 6px 12px;">Download XLSX</button>
                            </div>
                            <label for="advancedAdditionalTableToggleSingle" style="display: block; margin-bottom: 5px;">Select Table to Display:</label>
                            <select id="advancedAdditionalTableToggleSingle" style="margin-bottom: 10px; width: 100%;">
                                <option value="hyperparameters">Hyperparameters</option>
                                <option value="crossvalidation" ${data.cross_validation_summary && data.cross_validation_summary.length > 0 ? '' : 'disabled'}>Cross Validation</option>
                                <option value="featureselection" ${data.feature_selection_info ? '' : 'disabled'}>Feature Selection</option>
                                <option value="outlierhandling" ${data.outlier_info ? '' : 'disabled'}>Outlier Handling</option>
                            </select>
                            <div id="advancedAdditionalTableContentSingle">
                                ${hyperparameterTableHtmlSingleWithWrapper}
                            </div>
                        </div>
                    </div>
                        <br>
                        <br>
                    </div>

                    <h3>Graphics</h3>
                    <p style="margin-top: 4px; margin-bottom: 12px; color: #666; font-size: 0.95rem;">Advanced optimization graphics will be displayed here</p>
                    <label for="advancedRegressionVisualSelectorSingle">Select Visualization to Display</label>
                    <select id="advancedRegressionVisualSelectorSingle">
                        ${advancedVisualsSingle
                            .map((visual) => `<option value="${visual.file}">${visual.label}</option>`)
                            .join('')}
                    </select>
                    <br>
                    <br>
                    <img id="advancedTargetGraphicSingle" class="result-graphic" src='/user-visualizations/target_plot_1_advanced.png?t=${new Date().getTime()}'>
                    <div><br></div>
                </div>
                        `;
                        
                        // Set up event listeners for advanced results (single target)
                        const advancedVisualSelectorSingle = document.getElementById('advancedRegressionVisualSelectorSingle');
                        const advancedTargetGraphicSingle = document.getElementById('advancedTargetGraphicSingle');
                        
                        if (advancedVisualSelectorSingle && advancedTargetGraphicSingle) {
                            const updateAdvancedGraphicSingle = () => {
                                const selectedVisual = advancedVisualSelectorSingle.value;
                                
                                const visualObj = advancedVisualsSingle.find(v => v.file === selectedVisual);
                                const visualType = visualObj ? visualObj.type : 'default';
                                
                                if (selectedVisual !== 'target_plot' && selectedVisual !== 'target_plot_advanced') {
                                    let filename = selectedVisual;
                                    if (!filename.includes('.png') && !filename.includes('_advanced')) {
                                        filename = filename.endsWith('_advanced') ? `${filename}.png` : `${filename}.png`;
                                    } else if (filename.endsWith('_advanced') && !filename.includes('.png')) {
                                        filename = `${filename}.png`;
                                    }
                                    advancedTargetGraphicSingle.src = withApiRoot(`/user-visualizations/${filename}?t=${Date.now()}`);
                                    return;
                                }
                                
                                let suffix = '';
                                if (selectedVisual === 'target_plot_advanced' || (visualType === 'advanced')) {
                                    suffix = '_advanced';
                                }
                                
                                advancedTargetGraphicSingle.src = withApiRoot(`/user-visualizations/target_plot_1${suffix}.png?t=${Date.now()}`);
                            };
                            
                            advancedVisualSelectorSingle.addEventListener("change", updateAdvancedGraphicSingle);
                            updateAdvancedGraphicSingle();
                        }
                    }
                }

                // <img src='${data.ActVpredval}?t=${new Date().getTime()}' style="width: 70%; height: auto;">
                //     <div><br></div>
            }

            //Classifier Output
            else if (selectedOutputType === 'Classifier'){
                // Ensure results container is visible (using mode-specific container already defined)
                if (resultsContainer) {
                    resultsContainer.style.display = 'block';
                    resultsContainer.style.visibility = 'visible';
                    resultsContainer.classList.remove('hidden');
                }
                // Hide placeholder and show results (using mode-specific placeholder already defined)
                if (resultsPlaceholder) resultsPlaceholder.style.display = 'none';
                if (ClassifierResultDiv) ClassifierResultDiv.classList.remove('hidden')
                
                // Build hyperparameter table HTML
                const hyperparameterTableHtml = Object.keys(allHyperparameters).length > 0 ? `
                    <table class="stats-table model-stats-table">
                        <tr><th>Hyperparameter</th><th>Value</th></tr>
                        ${Object.entries(allHyperparameters).map(([key, value]) => `<tr><td>${key}</td><td>${value !== null && value !== undefined ? value : 'N/A'}</td></tr>`).join('')}
                    </table>` : '<p>No hyperparameters to display</p>';
                
                // Build cross validation table HTML
                const cvTableHtml = data.cross_validation_summary && data.cross_validation_summary.length > 0 ? `
                    <div class="model-stats-table-wrapper">
                        <table class="stats-table model-stats-table">
                            <tr><th>Metric</th><th>Mean</th><th>Std</th></tr>
                            ${data.cross_validation_summary.map(row => `<tr><td>${row.Metric || row.metric || ''}</td><td>${row.Mean || row.mean || ''}</td><td>${row.Std || row.std || ''}</td></tr>`).join('')}
                        </table>
                    </div>` : '<p>No cross-validation data available</p>';
                
                // Build feature selection table HTML
                const featureSelectionTableHtml = data.feature_selection_info ? `
                    <div class="model-stats-table-wrapper">
                        <table class="stats-table model-stats-table">
                            <tr><th>Property</th><th>Value</th></tr>
                            <tr><td>Method</td><td>${data.feature_selection_info.method || 'N/A'}</td></tr>
                            <tr><td>K Requested</td><td>${data.feature_selection_info.k_requested || 'N/A'}</td></tr>
                            <tr><td>Original Features</td><td>${data.feature_selection_info.original_count || 'N/A'}</td></tr>
                            <tr><td>Selected Features</td><td>${data.feature_selection_info.selected_count || 'N/A'}</td></tr>
                            ${data.feature_selection_info.selected_features && data.feature_selection_info.selected_features.length > 0 ? 
                                `<tr><td colspan="2"><strong>Selected Feature Names:</strong><br>${data.feature_selection_info.selected_features.join(', ')}</td></tr>` : ''}
                        </table>
                    </div>` : '<p>No feature selection data available</p>';
                
                // Build outlier handling table HTML
                const outlierHandlingTableHtml = data.outlier_info ? `
                    <div class="model-stats-table-wrapper">
                        <table class="stats-table model-stats-table">
                            <tr><th>Property</th><th>Value</th></tr>
                            <tr><td>Method</td><td>${data.outlier_info.method || 'N/A'}</td></tr>
                            <tr><td>Action</td><td>${data.outlier_info.action || 'N/A'}</td></tr>
                            <tr><td>Outliers Detected</td><td>${data.outlier_info.n_outliers || 0}</td></tr>
                            <tr><td>Original Samples</td><td>${data.outlier_info.original_samples || 'N/A'}</td></tr>
                            <tr><td>Remaining Samples</td><td>${data.outlier_info.remaining_samples || 'N/A'}</td></tr>
                        </table>
                    </div>` : '<p>No outlier handling data available</p>';
                
                // Check if advanced options were used
                const _hasAdvancedOptions = data.feature_selection_info || data.outlier_info;
                
                ClassifierResultDiv.innerHTML = ` 
                <div class="resultValues">
                    <div style="display: flex; gap: 20px; flex-wrap: wrap; align-items: flex-start;">
                        <div style="flex: 1; min-width: 300px;">
                            <h3 style="margin: 0; margin-bottom: 10px;">Performance</h3> 
                            <div class="model-stats-table-wrapper">
                                <table class="stats-table model-stats-table performance-table">
                                    <tr><th>Value</th><th>Average</th></tr>
                                    <tr> <td>Precision (weighted)</td> <td>${data.precision}</td> </tr>
                                    <tr> <td>Recall (weighted)</td> <td>${data.recall}</td> </tr>
                                    <tr> <td>F1 Score (weighted)</td> <td>${data.f1score}</td> </tr>
                                    <tr> <td>Error Rate (1 - Recall)</td> <td>${Number.isFinite(parseFloat(data.recall)) ? (1 - parseFloat(data.recall)).toFixed(3) : ''}</td> </tr>
                                    <tr> <td>Support (weighted)</td> <td>${data.support}</td> </tr>
                                    <tr> <td>Precision (macro)</td> <td>${data.macro_precision}</td> </tr>
                                    <tr> <td>Recall (macro)</td> <td>${data.recmacro_recallall}</td> </tr>
                                    <tr> <td>F1 Score (macro)</td> <td>${data.macro_f1score}</td> </tr>
                                    <tr> <td>Support (macro)</td> <td>${data.macro_support}</td> </tr>
                                    <tr> <td>Accuracy</td> <td>${data.accuracy}</td> </tr>
                                </table>
                            </div>
                            <div class="download-buttons" style="margin-top: 12px; display: flex; gap: 12px; align-items: center;">
                                <a href="/download/model_performance.xlsx?download_name=${encodeURIComponent(performanceDownloadName)}" onclick="return downloadFile('model_performance.xlsx', '${performanceDownloadName}')">
                                    <button type="button" class='downloadperformanceButton export-button'>Model Performance XLSX</button>
                                </a>
                                <a href="/download/visualizations.pdf?download_name=${encodeURIComponent(visualizationsDownloadName)}" onclick="return downloadFile('visualizations.pdf', '${visualizationsDownloadName}')">
                                    <button class="export-button" style="font-size: 0.95rem;">Visualizations PDF</button>
                                </a>
                            </div>
                        </div>
                        <div style="flex: 1; min-width: 300px;">
                            ${currentMode === 'simple' ? `
                            <h3 style="margin: 0; margin-bottom: 10px;">Hyperparameters</h3>
                            <div class="model-stats-table-wrapper">
                                ${hyperparameterTableHtml}
                            </div>
                            ` : `
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; gap: 12px;">
                                <h3 style="margin: 0;">Additional Information</h3>
                                <div style="display: flex; align-items: center; gap: 8px;">
                                    <label for="${currentMode === 'advanced' ? 'advancedClassifierAdditionalTableToggle' : 'automlClassifierAdditionalTableToggle'}" style="margin: 0; white-space: nowrap;">Select:</label>
                                    <select id="${currentMode === 'advanced' ? 'advancedClassifierAdditionalTableToggle' : 'automlClassifierAdditionalTableToggle'}" style="margin: 0; width: 200px; padding: 4px 8px;">
                                        <option value="hyperparameters">Hyperparameters</option>
                                        <option value="crossvalidation" ${data.cross_validation_summary && data.cross_validation_summary.length > 0 ? '' : 'disabled'}>Cross Validation</option>
                                        <option value="featureselection" ${data.feature_selection_info ? '' : 'disabled'}>Feature Selection</option>
                                        <option value="outlierhandling" ${data.outlier_info ? '' : 'disabled'}>Outlier Handling</option>
                                    </select>
                                </div>
                            </div>
                            <div id="${currentMode === 'advanced' ? 'advancedClassifierAdditionalTableContent' : 'automlClassifierAdditionalTableContent'}">
                                <div class="model-stats-table-wrapper">
                                    ${hyperparameterTableHtml}
                                </div>
                            </div>
                            `}
                        </div>
                    </div>
                    <br>
                    <br>
                </div>

                <div class="results-header" style="margin-top: 24px; margin-bottom: 16px;">
                    <h2>Modeling Results</h2>
                    <p>Charts, tables, and downloads will appear here.</p>
                </div>
                <h3>Graphics</h3>
                <p style="margin-top: 4px; margin-bottom: 12px; color: #666; font-size: 0.95rem;">Modeling graphics will be displayed here</p>
                <div class="result-graphic-row-train-test" style="display: flex; gap: 24px; flex-wrap: wrap; align-items: flex-start;">
                    <div class="result-graphic-box" style="flex: 1; min-width: 320px;">
                        <h4 style="margin: 0 0 8px 0; font-size: 1.1rem;">Visualization 1</h4>
                        <label for="classifierImageSelector">Select visualization</label>
                        <select id="classifierImageSelector">
                            <option value="confusion_matrix">Confusion Matrix (Test)</option>
                            <option value="roc_curve">ROC Curve (micro)</option>
                            <option value="roc_curve_per_class">ROC Curve (per class)</option>
                            <option value="precision_recall_curve">Precision-Recall Curve</option>
                            <option value="calibration_curve">Calibration Curve</option>
                            <option value="shap_summary">SHAP Feature Importance</option>
                        </select>
                        <br><br>
                        <img id="classifierGraphic" class="result-graphic" src="${withApiRoot('/user-visualizations/confusion_matrix.png')}?t=${new Date().getTime()}" alt="Classifier visualization 1">
                    </div>
                    <div class="result-graphic-box" style="flex: 1; min-width: 320px;">
                        <h4 style="margin: 0 0 8px 0; font-size: 1.1rem;">Visualization 2</h4>
                        <label for="classifierImageSelector2">Select visualization</label>
                        <select id="classifierImageSelector2">
                            <option value="roc_curve">ROC Curve (micro)</option>
                            <option value="roc_curve_per_class">ROC Curve (per class)</option>
                            <option value="confusion_matrix">Confusion Matrix (Test)</option>
                            <option value="precision_recall_curve">Precision-Recall Curve</option>
                            <option value="calibration_curve">Calibration Curve</option>
                            <option value="shap_summary">SHAP Feature Importance</option>
                        </select>
                        <br><br>
                        <img id="classifierGraphic2" class="result-graphic" src="${withApiRoot('/user-visualizations/roc_curve.png')}?t=${new Date().getTime()}" alt="Classifier visualization 2">
                    </div>
                </div>
                <div><br></div>
                </div>
                `
                
                // Set up table toggle event listeners for advanced/automl modes
                if (currentMode === 'advanced' || currentMode === 'automl') {
                    const tableHtmls = {
                        hyperparameters: `<div class="model-stats-table-wrapper">${hyperparameterTableHtml}</div>`,
                        crossvalidation: cvTableHtml,
                        featureselection: featureSelectionTableHtml,
                        outlierhandling: outlierHandlingTableHtml
                    };
                    
                    const toggleId = currentMode === 'advanced' ? 'advancedClassifierAdditionalTableToggle' : 'automlClassifierAdditionalTableToggle';
                    const contentId = currentMode === 'advanced' ? 'advancedClassifierAdditionalTableContent' : 'automlClassifierAdditionalTableContent';
                    
                    const tableToggle = document.getElementById(toggleId);
                    const tableContent = document.getElementById(contentId);
                    if (tableToggle && tableContent) {
                        tableToggle.addEventListener('change', function() {
                            const selectedValue = this.value;
                            if (tableHtmls[selectedValue]) {
                                tableContent.innerHTML = tableHtmls[selectedValue];
                            }
                        });
                    }
                }
                
                const classifierImageSelector = document.getElementById('classifierImageSelector');
                const classifierImageSelector2 = document.getElementById('classifierImageSelector2');
                const classifierGraphic = document.getElementById('classifierGraphic');
                const classifierGraphic2 = document.getElementById('classifierGraphic2');
                const classifierGraphicUrl = (value) => withApiRoot(`/user-visualizations/${value}.png?t=${Date.now()}`);
                const updateClassifierGraphic = (panel) => {
                    const sel = panel === 1 ? classifierImageSelector : classifierImageSelector2;
                    const img = panel === 1 ? classifierGraphic : classifierGraphic2;
                    if (sel && img) img.src = classifierGraphicUrl(sel.value);
                };
                if (classifierImageSelector) classifierImageSelector.addEventListener('change', () => updateClassifierGraphic(1));
                if (classifierImageSelector2) classifierImageSelector2.addEventListener('change', () => updateClassifierGraphic(2));
                updateClassifierGraphic(1);
                updateClassifierGraphic(2);
            }

            //Cluster Output
            else if (selectedOutputType === 'Cluster'){
                // Ensure results container is visible (using mode-specific container already defined)
                if (resultsContainer) {
                    resultsContainer.style.display = 'block';
                    resultsContainer.style.visibility = 'visible';
                    resultsContainer.classList.remove('hidden');
                }
                // Hide placeholder and show results (using mode-specific placeholder already defined)
                if (resultsPlaceholder) resultsPlaceholder.style.display = 'none';
                if (ClusterResultDiv) ClusterResultDiv.classList.remove('hidden')
                
                // Build hyperparameter table HTML
                const hyperparameterTableHtml = Object.keys(allHyperparameters).length > 0 ? `
                    <table class="stats-table model-stats-table">
                        <tr><th>Hyperparameter</th><th>Value</th></tr>
                        ${Object.entries(allHyperparameters).map(([key, value]) => `<tr><td>${key}</td><td>${value !== null && value !== undefined ? value : 'N/A'}</td></tr>`).join('')}
                    </table>` : '<p>No hyperparameters to display</p>';
                
                if (ClusterResultDiv) ClusterResultDiv.innerHTML = ` 
                    <div class="resultValues">
                        <div style="display: flex; gap: 20px; flex-wrap: wrap; align-items: flex-start;">
                            <div style="flex: 1; min-width: 300px;">
                                <h3 style="margin: 0; margin-bottom: 10px;">Cluster Performance</h3> 
                                <div class="model-stats-table-wrapper">
                                    <table class="stats-table model-stats-table performance-table">
                                        <tr><th>Value</th><th>Training</th><th>Validation</th><th class="delta-col">Δ (Train-Validation)</th></tr>
                                        <tr> <td>Silhouette</td> <td>${data.train_silhouette}</td> <td>${data.test_silhouette}</td> <td class="delta-col">${formatDelta(data.train_silhouette, data.test_silhouette)}</td> </tr>
                                        <tr> <td>Calinski Harabasz</td> <td>${data.train_calinski_harabasz}</td> <td>${data.test_calinski_harabasz}</td> <td class="delta-col">${formatDelta(data.train_calinski_harabasz, data.test_calinski_harabasz)}</td> </tr>
                                        <tr> <td>Davies Bouldin</td> <td>${data.train_davies_bouldin}</td> <td>${data.test_davies_bouldin}</td> <td class="delta-col">${formatDelta(data.train_davies_bouldin, data.test_davies_bouldin)}</td> </tr>
                                    </table>
                                </div>
                                <p style="margin-top: 12px;"><strong>Best K:</strong> ${data.best_k}</p>
                                <div class="download-buttons" style="margin-top: 12px; display: flex; gap: 12px; align-items: center;">
                                    <a href="/download/model_performance.xlsx?download_name=${encodeURIComponent(performanceDownloadName)}" onclick="return downloadFile('model_performance.xlsx', '${performanceDownloadName}')">
                                        <button class='downloadperformanceButton export-button'>Model Performance XLSX</button>
                                    </a>
                                    <a href="/download/visualizations.pdf?download_name=${encodeURIComponent(visualizationsDownloadName)}" onclick="return downloadFile('visualizations.pdf', '${visualizationsDownloadName}')">
                                        <button class="export-button" style="font-size: 0.95rem;">Visualizations PDF</button>
                                    </a>
                                </div>
                            </div>
                            <div style="flex: 1; min-width: 300px;">
                                <h3 style="margin: 0; margin-bottom: 10px;">Hyperparameters</h3>
                                <div class="model-stats-table-wrapper">
                                    ${hyperparameterTableHtml}
                                </div>
                            </div>
                        </div>
                        <br>
                        <br>
                    </div>

                    <div class="results-header" style="margin-top: 24px; margin-bottom: 16px;">
                        <h2>Modeling Results</h2>
                        <p>Charts, tables, and downloads will appear here.</p>
                    </div>
                    <h3>Graphics</h3>
                    <p style="margin-top: 4px; margin-bottom: 12px; color: #666; font-size: 0.95rem;">Modeling graphics will be displayed here</p>
                    <div class="result-graphic-row-train-test" style="display: flex; gap: 24px; flex-wrap: wrap; align-items: flex-start;">
                        <div class="result-graphic-box" style="flex: 1; min-width: 320px;">
                            <h4 style="margin: 0 0 8px 0; font-size: 1.1rem;">Visualization 1</h4>
                            <label for="clusterImageSelector">Select visualization</label>
                            <select id="clusterImageSelector">
                                <option value="cluster_pca_train">PCA (Train)</option>
                                <option value="cluster_pca_test">PCA (Test)</option>
                                <option value="cluster_silhouette">Silhouette (Train)</option>
                                <option value="cluster_sizes">Cluster sizes (Train)</option>
                            </select>
                            <br><br>
                            <img id="clusterGraphic" class="result-graphic" src="${withApiRoot('/user-visualizations/cluster_pca_train.png')}?t=${new Date().getTime()}">
                        </div>
                        <div class="result-graphic-box" style="flex: 1; min-width: 320px;">
                            <h4 style="margin: 0 0 8px 0; font-size: 1.1rem;">Visualization 2</h4>
                            <label for="clusterImageSelector2">Select visualization</label>
                            <select id="clusterImageSelector2">
                                <option value="cluster_pca_train">PCA (Train)</option>
                                <option value="cluster_pca_test">PCA (Test)</option>
                                <option value="cluster_silhouette">Silhouette (Train)</option>
                                <option value="cluster_sizes">Cluster sizes (Train)</option>
                            </select>
                            <br><br>
                            <img id="clusterGraphic2" class="result-graphic" src="${withApiRoot('/user-visualizations/cluster_pca_test.png')}?t=${new Date().getTime()}">
                        </div>
                    </div>
                    <div><br></div>
                </div> `
                const clusterImageSelector = getCachedElement('clusterImageSelector');
                const clusterImageSelector2 = getCachedElement('clusterImageSelector2');
                const clusterGraphic = getCachedElement('clusterGraphic');
                const clusterGraphic2 = getCachedElement('clusterGraphic2');
                const clusterGraphicUrl = (value) => withApiRoot(`/user-visualizations/${value}.png?t=${Date.now()}`);
                const updateClusterGraphic = (panel) => {
                    const sel = panel === 1 ? clusterImageSelector : clusterImageSelector2;
                    const img = panel === 1 ? clusterGraphic : clusterGraphic2;
                    if (sel && img) img.src = clusterGraphicUrl(sel.value);
                };
                if (clusterImageSelector) clusterImageSelector.addEventListener('change', () => updateClusterGraphic(1));
                if (clusterImageSelector2) clusterImageSelector2.addEventListener('change', () => updateClusterGraphic(2));
            }
            


        //if backend failed then show error div
        else {
            showError(errorDiv, `Error: ${data.error}`);
            hideElement(NumericResultDiv);
            hideElement(ClassifierResultDiv);
            hideElement(ClusterResultDiv);
        }
    } catch (error) {
        console.error('Error processing result:', error);
        const errorDiv = getCachedElement('errorDiv');
        showError(errorDiv, 'Result processing failed. See console for details.');
    }
}

