/**
 * DiGiTerra front-end logic. Handoff note: All API calls, uploads, progress polling,
 * and result rendering live here. UI structure is in templates/index.html.
 * See HANDOFF.md for repo overview.
 */
const uploadForm = document.getElementById('uploadForm');
const corrForm = document.getElementById('corrForm');
const preprocessform = document.getElementById('preprocessform');
const file = document.getElementById('file');
const columnSelection = document.getElementById('columnSelection');
const indicatorsSelect = document.getElementById('indicators');
const predictorsSelect = document.getElementById('predictors');
const processForm = document.getElementById('processForm');
const advancedOptimizationForm = document.getElementById('advancedOptimizationForm');
const errorDiv = document.getElementById('errorDiv');
const NumericResultDiv = document.getElementById('NumericResultDiv');
const ClusterResultDiv = document.getElementById('ClusterResultDiv');
const ClassifierResultDiv = document.getElementById('ClassifierResultDiv');
const fileUpload = document.getElementById('fileuploaddiv');
const runMatrices = document.getElementById('runMatrices');
const predictionDiv = document.getElementById('predictionDiv');
const predictionForm = document.getElementById('uploadPredictDf');
const predictionResultsDiv = document.getElementById('predictionResults');
const loading = document.getElementById('loading');
const processButton = document.getElementById('processButton');
const appTabs = document.getElementById('appTabs');
const tabButtons = document.querySelectorAll('.tab-button');
const userInputSection = document.getElementById('userInputSection');
const trainSizeInput = document.getElementById('trainSize');
const testSizeInput = document.getElementById('testSize');
const backToExplorationButton = document.getElementById('backToExploration');
const backToModelPreprocessButton = document.getElementById('backToModelPreprocess');
const backToModelingFromAdvancedButton = document.getElementById('backToModelingFromAdvanced');
const documentationSection = document.getElementById('documentation');
let pywebviewReady = false;

let uploadedFileName = '';

// ============================================================================
// Utility Functions
// ============================================================================

const formatDelta = (trainValue, validationValue, unit = '') => {
    const trainNum = parseFloat(trainValue);
    const validationNum = parseFloat(validationValue);
    if (!Number.isFinite(trainNum) || !Number.isFinite(validationNum)) {
        return '';
    }
    const delta = (trainNum - validationNum).toFixed(3);
    return unit ? `${delta} ${unit}` : delta;
};

// DOM utility functions
const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => document.querySelectorAll(selector);
const $id = (id) => document.getElementById(id);

// Element visibility utilities
const showElement = (element) => {
    if (element) element.classList.remove('hidden');
};

const hideElement = (element) => {
    if (element) element.classList.add('hidden');
};

const toggleElement = (element, show) => {
    if (element) {
        if (show) {
            showElement(element);
        } else {
            hideElement(element);
        }
    }
};

// HTML escape utility to prevent XSS
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Error display utility
const showError = (element, message, useErrorClass = true) => {
    if (element) {
        showElement(element);
        const className = useErrorClass ? 'error-message' : '';
        // Sanitize message to prevent XSS - escape HTML and use textContent for safety
        const sanitizedMessage = escapeHtml(String(message));
        element.innerHTML = `<div class="${className}"><p class="error-text">${sanitizedMessage}</p></div>`;
        // Announce to screen readers (use original message for screen reader, but sanitized for display)
        announceToScreenReader(String(message), 'assertive');
        // Set ARIA attributes
        element.setAttribute('role', 'alert');
        element.setAttribute('aria-live', 'assertive');
        element.setAttribute('aria-atomic', 'true');
    }
};

// Screen reader announcement utility
function announceToScreenReader(message, priority = 'polite') {
    const announcement = document.createElement('div');
    announcement.setAttribute('role', 'status');
    announcement.setAttribute('aria-live', priority);
    announcement.setAttribute('aria-atomic', 'true');
    announcement.className = 'sr-only';
    announcement.textContent = message;
    document.body.appendChild(announcement);
    setTimeout(() => announcement.remove(), 1000);
}

// Focus management utility
function manageFocus(element) {
    if (element) {
        element.focus();
        element.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
}

// Enhance toggle switches with ARIA labels if missing
function enhanceToggleAccessibility() {
    // Find all checkboxes - no invalid selector, we check each one individually
    document.querySelectorAll('input[type="checkbox"]').forEach(toggle => {
        // Skip if already has aria-label or aria-labelledby
        if (toggle.getAttribute('aria-label') || toggle.getAttribute('aria-labelledby')) {
            return;
        }
        
        // Check if it's a toggle switch by looking at its parent structure
        // Toggle switches are typically inside a .switch label or .toggle-container
        const switchParent = toggle.closest('.switch');
        const toggleContainer = toggle.closest('.toggle-container');
        const isToggleSwitch = switchParent !== null || toggleContainer !== null;
        
        // Also include checkboxes with "Slider" or "Toggle" in their ID (like unitToggle)
        const hasToggleId = toggle.id && (toggle.id.includes('Slider') || toggle.id.includes('Toggle'));
        
        // Process if it's a toggle switch or has a toggle-like ID
        if (isToggleSwitch || hasToggleId) {
            let labelText = null;
            
            // First, try to find label in the toggle-container (most common case)
            if (toggleContainer) {
                const h3Label = toggleContainer.querySelector('h3');
                const spanLabel = toggleContainer.querySelector('span');
                if (h3Label) {
                    labelText = h3Label.textContent.trim();
                } else if (spanLabel) {
                    labelText = spanLabel.textContent.trim();
                }
            }
            
            // If not found, try the switch label's parent or siblings
            if (!labelText && switchParent) {
                const parentContainer = switchParent.parentElement;
                if (parentContainer) {
                    const h3Label = parentContainer.querySelector('h3');
                    const spanLabel = parentContainer.querySelector('span');
                    if (h3Label) {
                        labelText = h3Label.textContent.trim();
                    } else if (spanLabel) {
                        labelText = spanLabel.textContent.trim();
                    } else if (parentContainer.previousElementSibling) {
                        // Check previous sibling for label text
                        const prevSibling = parentContainer.previousElementSibling;
                        if (prevSibling.tagName === 'H3' || prevSibling.tagName === 'SPAN') {
                            labelText = prevSibling.textContent.trim();
                        }
                    }
                }
            }
            
            // If still no label found, try to find associated label element
            if (!labelText) {
                const labelElement = document.querySelector(`label[for="${toggle.id}"]`);
                if (labelElement) {
                    labelText = labelElement.textContent.trim();
                }
            }
            
            // Fallback: create descriptive label from ID
            if (!labelText && toggle.id) {
                labelText = toggle.id
                    .replace(/([A-Z])/g, ' $1')
                    .replace(/^./, str => str.toUpperCase())
                    .replace(/\b(Slider|Toggle)\b/gi, '')
                    .trim();
            }
            
            // Set the aria-label if we found or created one
            if (labelText) {
                toggle.setAttribute('aria-label', labelText);
            }
        }
    });
}

// Initialize accessibility enhancements when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', enhanceToggleAccessibility);
} else {
    enhanceToggleAccessibility();
}

// Modeling mode switching handler
function switchModelingMode(mode) {
    // Hide all mode sections
    const simpleSection = document.getElementById('simpleModelingSection');
    const advancedSection = document.getElementById('advancedModelingSection');
    const automlSection = document.getElementById('automlSection');
    const oldAdvancedContainer = document.getElementById('advancedOptimization');
    
    if (simpleSection) simpleSection.classList.add('hidden');
    if (advancedSection) advancedSection.classList.add('hidden');
    if (automlSection) automlSection.classList.add('hidden');
    
    // Show selected mode section
    if (mode === 'simple' && simpleSection) {
        simpleSection.classList.remove('hidden');
        // Hide old Advanced container
        if (oldAdvancedContainer) oldAdvancedContainer.classList.add('hidden');
    } else if (mode === 'advanced' && advancedSection) {
        advancedSection.classList.remove('hidden');
        // Clone hyperparameter fields from old container if not already present
        ensureAdvancedHyperparametersAvailable();
        // Keep old Advanced container hidden
        if (oldAdvancedContainer) oldAdvancedContainer.classList.add('hidden');
    } else if (mode === 'automl' && automlSection) {
        automlSection.classList.remove('hidden');
        // Hide old Advanced container
        if (oldAdvancedContainer) oldAdvancedContainer.classList.add('hidden');
    }
    
    // Ensure results container is always visible for the current mode
    let resultsContainer = null;
    if (mode === 'simple') {
        resultsContainer = document.getElementById('simpleModelingResults');
    } else if (mode === 'advanced') {
        resultsContainer = document.getElementById('advancedModelingResults');
    } else if (mode === 'automl') {
        resultsContainer = document.getElementById('automlModelingResults');
    }
    
    if (resultsContainer) {
        resultsContainer.style.display = 'block';
        resultsContainer.style.visibility = 'visible';
        resultsContainer.classList.remove('hidden');
    }
    
    // Update header model selection visibility based on output type
    const outputType = document.getElementById("outputType1");
    if (outputType) {
        updateOutputTypeDisplay(outputType.value);
    }
    
    // Update AutoML settings display when AutoML mode is selected
    if (mode === 'automl') {
        // Explicitly set AutoML cross-validation defaults (don't inherit from advanced mode)
        const cvTypeSelect = getCachedElement('advancedCrossValidationType');
        const cvFoldsSelect = getCachedElement('advancedCrossValidationFolds');
        if (cvTypeSelect) {
            cvTypeSelect.value = 'KFold';
        }
        if (cvFoldsSelect) {
            cvFoldsSelect.value = '5';
        }
        updateAutomlSettingsDisplay();
    }
    
    // Note: Descriptive text is now updated directly when column selection changes
    // No need to sync here since each mode has its own note element
}

// Ensure hyperparameter fields are available in the unified Advanced section
function ensureAdvancedHyperparametersAvailable() {
    // Find the hyperparameter card in unified Advanced section
    const unifiedSection = document.getElementById('advancedModelingSection');
    if (!unifiedSection) return;
    
    const hyperparamsCard = unifiedSection.querySelector('.preprocess-card h2');
    if (!hyperparamsCard || hyperparamsCard.textContent !== 'Model Hyperparameters') return;
    
    const hyperparamsContainer = hyperparamsCard.closest('.preprocess-card');
    if (!hyperparamsContainer) return;
    
    // Check if hyperparameter fields already exist (they should be copied)
    const hasRidgeFields = document.getElementById('advancedRidgeFields');
    if (hasRidgeFields && hasRidgeFields.closest('#advancedModelingSection')) {
        // Fields already exist in unified section
        return;
    }
    
    // If fields don't exist, clone them from old container
    const oldContainer = document.getElementById('advancedOptimization');
    if (!oldContainer) return;
    
    // Temporarily show old container to access its content
    const wasHidden = oldContainer.classList.contains('hidden');
    if (wasHidden) oldContainer.classList.remove('hidden');
    
    // Find all hyperparameter field divs in old container
    const oldHyperparamsCard = oldContainer.querySelector('.preprocess-card h2');
    if (!oldHyperparamsCard || oldHyperparamsCard.textContent !== 'Model Hyperparameters') {
        if (wasHidden) oldContainer.classList.add('hidden');
        return;
    }
    
    const oldHyperparamsContainer = oldHyperparamsCard.closest('.preprocess-card');
    if (!oldHyperparamsContainer) {
        if (wasHidden) oldContainer.classList.add('hidden');
        return;
    }
    
    // Clone all hyperparameter field divs (all divs with IDs containing "Fields")
    const hyperparamFields = oldHyperparamsContainer.querySelectorAll('div[id*="Fields"]');
    hyperparamFields.forEach(field => {
        // Check if it already exists in unified section
        const existing = document.getElementById(field.id);
        if (!existing || !existing.closest('#advancedModelingSection')) {
            // Clone and append to unified section's hyperparameter card
            const cloned = field.cloneNode(true);
            hyperparamsContainer.appendChild(cloned);
        }
    });
    
    // Re-hide old container if it was hidden
    if (wasHidden) oldContainer.classList.add('hidden');
}

// Initialize modeling mode selector
const modelingModeRadios = document.querySelectorAll('input[name="modelingMode"]');
modelingModeRadios.forEach(radio => {
    radio.addEventListener('change', function() {
        switchModelingMode(this.value);
    });
});

// Sync header model selectors with section model selectors
function setupHeaderModelSelectorSync() {
    // Simple mode - Numeric (using new mode-specific IDs)
    // Note: No sync needed since header dropdowns are hidden
    // The change events will be handled by existing model selection listeners
    const simpleNModels = document.getElementById('simpleNModels');
    // No recursive event dispatch needed - existing listeners handle model changes
    
    // Simple mode - Cluster
    const simpleClModels = document.getElementById('simpleClModels');
    // No recursive event dispatch needed - existing listeners handle model changes
    
    // Simple mode - Classifier
    const simpleClassModels = document.getElementById('simpleClassModels');
    // No recursive event dispatch needed - existing listeners handle model changes
    
    // Advanced mode - Numeric
    const headerAdvancedNModels = document.getElementById('headerAdvancedNModels');
    const advancedNModels = document.getElementById('advancedNModels');
    if (headerAdvancedNModels && advancedNModels) {
        headerAdvancedNModels.addEventListener('change', function() {
            advancedNModels.value = this.value;
            advancedNModels.dispatchEvent(new Event('change'));
        });
        advancedNModels.addEventListener('change', function() {
            headerAdvancedNModels.value = this.value;
        });
    }
    
    // Advanced mode - Cluster
    const headerAdvancedClModels = document.getElementById('headerAdvancedClModels');
    const advancedClModels = document.getElementById('advancedClModels');
    if (headerAdvancedClModels && advancedClModels) {
        headerAdvancedClModels.addEventListener('change', function() {
            advancedClModels.value = this.value;
            advancedClModels.dispatchEvent(new Event('change'));
        });
        advancedClModels.addEventListener('change', function() {
            headerAdvancedClModels.value = this.value;
        });
    }
    
    // Advanced mode - Classifier
    const headerAdvancedClassModels = document.getElementById('headerAdvancedClassModels');
    const advancedClassModels = document.getElementById('advancedClassModels');
    if (headerAdvancedClassModels && advancedClassModels) {
        headerAdvancedClassModels.addEventListener('change', function() {
            advancedClassModels.value = this.value;
            advancedClassModels.dispatchEvent(new Event('change'));
        });
        advancedClassModels.addEventListener('change', function() {
            headerAdvancedClassModels.value = this.value;
        });
    }
    
    // AutoML mode - Numeric
    const headerAutomlNModels = document.getElementById('headerAutomlNModels');
    const automlNModels = document.getElementById('automlNModels');
    if (headerAutomlNModels && automlNModels) {
        headerAutomlNModels.addEventListener('change', function() {
            automlNModels.value = this.value;
            automlNModels.dispatchEvent(new Event('change'));
            updateAutomlSettingsDisplay();
        });
        automlNModels.addEventListener('change', function() {
            headerAutomlNModels.value = this.value;
            updateAutomlSettingsDisplay();
        });
    }
    
    // AutoML mode - Cluster
    const headerAutomlClModels = document.getElementById('headerAutomlClModels');
    const automlClModels = document.getElementById('automlClModels');
    if (headerAutomlClModels && automlClModels) {
        headerAutomlClModels.addEventListener('change', function() {
            automlClModels.value = this.value;
            automlClModels.dispatchEvent(new Event('change'));
            updateAutomlSettingsDisplay();
        });
        automlClModels.addEventListener('change', function() {
            headerAutomlClModels.value = this.value;
            updateAutomlSettingsDisplay();
        });
    }
    
    // AutoML mode - Classifier
    const headerAutomlClassModels = document.getElementById('headerAutomlClassModels');
    const automlClassModels = document.getElementById('automlClassModels');
    if (headerAutomlClassModels && automlClassModels) {
        headerAutomlClassModels.addEventListener('change', function() {
            automlClassModels.value = this.value;
            automlClassModels.dispatchEvent(new Event('change'));
            updateAutomlSettingsDisplay();
        });
        automlClassModels.addEventListener('change', function() {
            headerAutomlClassModels.value = this.value;
            updateAutomlSettingsDisplay();
        });
    }
    
    // AutoML intensity level selector
    const automlIntensity = document.getElementById('automlIntensity');
    if (automlIntensity) {
        automlIntensity.addEventListener('change', function() {
            updateAutomlSettingsDisplay();
        });
    }
}

// Initialize header model selector sync when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', setupHeaderModelSelectorSync);
} else {
    setupHeaderModelSelectorSync();
}

// Ensure results container is visible when page loads (if on modeling tab)
function ensureResultsContainerVisible() {
    const columnSelection = document.getElementById('columnSelection');
    if (columnSelection && columnSelection.style.display !== 'none' && !columnSelection.classList.contains('hidden')) {
        // Determine which mode is active
        const simpleMode = document.getElementById('simpleMode');
        const advancedMode = document.getElementById('advancedMode');
        const automlMode = document.getElementById('automlMode');
        
        let resultsContainer = null;
        if (simpleMode && simpleMode.checked) {
            resultsContainer = document.getElementById('simpleModelingResults');
        } else if (advancedMode && advancedMode.checked) {
            resultsContainer = document.getElementById('advancedModelingResults');
        } else if (automlMode && automlMode.checked) {
            resultsContainer = document.getElementById('automlModelingResults');
        }
        
        if (resultsContainer) {
            resultsContainer.style.display = 'block';
            resultsContainer.style.visibility = 'visible';
            resultsContainer.classList.remove('hidden');
        }
    }
}

// Initialize results container visibility
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', ensureResultsContainerVisible);
} else {
    ensureResultsContainerVisible();
}

// Set initial mode (Simple is checked by default)
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        const simpleMode = document.getElementById('simpleMode');
        if (simpleMode && simpleMode.checked) {
            switchModelingMode('simple');
        }
    });
} else {
    const simpleMode = document.getElementById('simpleMode');
    if (simpleMode && simpleMode.checked) {
        switchModelingMode('simple');
    }
}

const clearError = (element) => {
    if (element) {
        element.innerHTML = '';
        hideElement(element);
        // Also clear any aria-invalid attributes from form fields when error is cleared
        const formFields = [
            'specificVariableSelect', 'quantiles', 'bins', 'binsLabel', 
            'quantileBins', 'predictors', 'modelingType', 'indicators'
        ];
        formFields.forEach(fieldId => {
            const field = document.getElementById(fieldId);
            if (field) {
                field.removeAttribute('aria-invalid');
            }
        });
    }
};

// Safe element getter with caching
const getCachedElement = (() => {
    const cache = new Map();
    return (id) => {
        if (!cache.has(id)) {
            const element = $id(id);
            if (element) cache.set(id, element);
            return element;
        }
        return cache.get(id);
    };
})();

// Safely check for pywebview API with error handling
function safeCheckPywebviewAPI() {
    try {
        if (window.pywebview?.api) {
            console.log('pywebview API detected');
            if (window.pywebview.api.save_file) {
                console.log('save_file method is available');
            } else {
                console.warn('save_file method is NOT available');
            }
            return true;
        }
        return false;
    } catch (error) {
        console.error('Error checking pywebview API:', error);
        return false;
    }
}

window.addEventListener('pywebviewready', () => {
    console.log('pywebview ready event fired');
    // Mark API as ready, but don't test it immediately to avoid crashes
    // The API will be tested when actually needed (e.g., when downloading files)
    pywebviewReady = true;
    console.log('pywebview API marked as ready');
});

// Check for API after DOM is fully loaded, with delay to ensure pywebview is ready
// Use a longer delay to avoid crashes during page load
// Only check if we're in a desktop app context (not in regular browser)
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        // Delay check to avoid interfering with page load
        setTimeout(() => {
            try {
                safeCheckPywebviewAPI();
            } catch (error) {
                console.error('Error checking pywebview API (non-fatal):', error);
            }
        }, 1000);
    });
} else {
    // DOM already loaded, check after a delay
    setTimeout(() => {
        try {
            safeCheckPywebviewAPI();
        } catch (error) {
            console.error('Error checking pywebview API (non-fatal):', error);
        }
    }, 1000);
}

document.addEventListener('click', (event) => {
    if (event.defaultPrevented) {
        return;
    }
    const link = event.target.closest('a');
    if (!link) {
        return;
    }
    const href = link.getAttribute('href');
    if (!href || !href.startsWith('/download/')) {
        return;
    }
    // Check if pywebview API is available
    try {
        if (window.pywebview?.api?.save_file) {
            event.preventDefault();
            pywebviewReady = true;
            const downloadUrl = new URL(href, window.location.origin);
            const requestedName = downloadUrl.searchParams.get('download_name');
            const sourceFilename = decodeURIComponent(downloadUrl.pathname.replace('/download/', ''));
            const downloadName = requestedName ? decodeURIComponent(requestedName) : sourceFilename;
            console.log('Intercepting download link, calling pywebview API:', sourceFilename, downloadName);
            window.pywebview.api.save_file(sourceFilename, downloadName)
            .then((success) => {
                if (success) {
                    console.log('File saved successfully via pywebview API');
                } else {
                    console.warn('File save was cancelled or failed');
                }
            })
            .catch((error) => {
                console.error('Error saving file via pywebview API:', error);
                // If API fails, navigate directly to the download URL to trigger server-side download
                // This avoids the infinite loop that would occur with link.click()
                console.log('API failed, falling back to server-side download');
                window.location.href = href;
            });
        } else {
            // Log when API is not available (for debugging)
            if (window.location.protocol === 'http:' || window.location.protocol === 'https:') {
                console.log('pywebview API not available, using regular download');
            }
        }
    } catch (error) {
        console.error('Error accessing pywebview API:', error);
        // Fallback to regular download
        window.location.href = href;
    }
});

if (backToExplorationButton) {
    backToExplorationButton.addEventListener('click', () => {
        showTab('processing');
    });
}

if (backToModelPreprocessButton) {
    backToModelPreprocessButton.addEventListener('click', () => {
        showTab('model-preprocessing');
    });
}

if (backToModelingFromAdvancedButton) {
    backToModelingFromAdvancedButton.addEventListener('click', () => {
        showTab('model-preprocessing');
    });
}

// helpers for getting the column index / letter 
function getColumnLetter(index) {
        let column = "";
        while (index >= 0) {
            column = String.fromCharCode((index % 26) + 65) + column;
            index = Math.floor(index / 26) - 1;
        }
        return column;
    }
function getColumnIndices(input) {
    const columns = [];
    input.split(',').forEach(part => {
        if (part.includes('-')) {
            const [start, end] = part.split('-').map(c => columnToIndex(c.trim()));
            for (let i = start; i <= end; i++) {
                columns.push(i);
            }
        } else {
            columns.push(columnToIndex(part.trim()));
        }
    });
    return columns;
}

function columnToIndex(column) {
    let index = 0;
    for (let i = 0; i < column.length; i++) {
        index = index * 26 + (column.charCodeAt(i) - 65 + 1);
    }
    return index - 1;
}

function setActiveTab(tabName) {
    tabButtons.forEach((button) => {
        if (button.dataset.tab === tabName) {
            button.classList.add('active');
        } else {
            button.classList.remove('active');
        }
    });
}

// Helper function to check if data exists for Data Exploration page
function checkHasDataForExploration() {
    const columnList = document.getElementById('columnList');
    // Check if columnList has actual table content (not just placeholder)
    const columnListHasContent = columnList && 
                                 columnList.innerHTML.trim() !== '' && 
                                 !columnList.innerHTML.includes('<!-- Column headers will be displayed here -->') &&
                                 (columnList.innerHTML.includes('<table') || columnList.innerHTML.includes('<tr'));
    // Note: memStorage is server-side only, not available in client-side JavaScript
    // Only check client-side accessible data: uploadedFileName and columnList content
    return uploadedFileName || columnListHasContent;
}

function showTab(tabName) {
    // Always show the app tabs toolbar when switching tabs
    if (appTabs) {
        appTabs.classList.remove('hidden');
    }
    
    // Hide welcome page if visible
    const welcomeDiv = document.getElementById('welcome');
    if (welcomeDiv && !welcomeDiv.classList.contains('hidden')) {
        welcomeDiv.classList.add('hidden');
    }
    
    // Check for data BEFORE hiding sections - need to know if we should keep fileUpload visible
    const columnDiv = document.getElementById('columnsection');
    const goingToProcessing = tabName === 'processing';
    
    const hasDataForProcessing = goingToProcessing && checkHasDataForExploration();
    
    // Hide ALL main sections first to prevent header carryover
    // BUT: Don't hide fileUpload if we're going to processing tab and data exists
    if (fileUpload && !hasDataForProcessing) {
        fileUpload.classList.add('hidden');
    }
    if (documentationSection) {
        documentationSection.classList.add('hidden');
    }
    if (userInputSection) {
        userInputSection.classList.add('hidden');
    }
    if (predictionDiv) {
        predictionDiv.classList.add('hidden');
    }
    if (columnSelection) {
        columnSelection.style.display = 'none';
    }
    const advancedOptimization = document.getElementById('advancedOptimization');
    if (advancedOptimization) {
        advancedOptimization.classList.add('hidden');
    }
    
    // Hide nested sections within fileUpload to prevent header carryover
    // BUT: Don't hide columnDiv if we're going to processing tab and data exists
    if (columnDiv && !hasDataForProcessing) {
        columnDiv.classList.add('hidden');
    }
    const redobutton = document.getElementById('redobutton');
    // Don't hide redobutton here - it will be shown/hidden based on tab and data state below

    // Show the appropriate section for the selected tab - no workflow enforcement
    if (tabName === 'documentation') {
        if (documentationSection) {
            documentationSection.classList.remove('hidden');
        }
        // Hide redobutton on documentation tab
        if (redobutton) {
            redobutton.classList.add('hidden');
        }
    } else if (tabName === 'processing') {
        // Data Exploration - show fileUpload with columnDiv if data exists
        // Re-check for data using the same helper function
        const hasData = checkHasDataForExploration();
        const columnList = document.getElementById('columnList');
        
        // Debug logging (removed memStorage check - it's server-side only)
        console.log('Data Exploration tab - hasData:', hasData, 'uploadedFileName:', uploadedFileName, 'columnList exists:', !!columnList, 'columnList has content:', columnList ? columnList.innerHTML.trim().length > 0 : false);
        
        // Always show fileUpload first (parent container) - this is critical!
        if (fileUpload) {
            fileUpload.classList.remove('hidden');
            fileUpload.style.display = ''; // Force display
        }
        
        // Always show columnDiv if data exists, regardless of previous state
        if (hasData) {
            console.log('Showing Data Exploration content - data exists');
            if (columnDiv) {
                columnDiv.classList.remove('hidden');
                columnDiv.style.display = ''; // Ensure it's visible
            }
            // Show navigation buttons when data exists
            if (redobutton) {
                redobutton.classList.remove('hidden');
            }
            // Ensure the exploration-panel is visible first (parent container)
            const explorationPanel = columnDiv ? columnDiv.querySelector('.exploration-panel') : null;
            if (explorationPanel) {
                explorationPanel.style.display = '';
                explorationPanel.classList.remove('hidden');
            }
            // Ensure columnList is visible if it has content
            if (columnList) {
                columnList.style.display = '';
                columnList.classList.remove('hidden');
            }
            // Ensure all exploration elements are visible when columnDiv is shown
            const corrForm = document.getElementById('corrForm');
            if (corrForm) {
                corrForm.classList.remove('hidden');
                corrForm.style.display = '';
            }
            const explorationOutput = document.getElementById('explorationOutput');
            const explorationResults = explorationOutput ? explorationOutput.closest('.exploration-results') : null;
            // Always show exploration-results section if data exists (even if empty, user can run exploration)
            if (explorationResults) {
                explorationResults.classList.remove('hidden');
                explorationResults.style.display = '';
            }
            // Show explorationOutput if it has content OR if data exists (so user can see the section)
            if (explorationOutput) {
                explorationOutput.classList.remove('hidden');
                explorationOutput.style.display = '';
            }
            // Also ensure dataExploration div is visible
            const dataExploration = document.getElementById('dataExploration');
            if (dataExploration) {
                dataExploration.classList.remove('hidden');
                dataExploration.style.display = '';
            }
        } else {
            console.log('Hiding Data Exploration content - no data found');
            if (columnDiv && !hasData) {
                columnDiv.classList.add('hidden');
            }
            // Hide navigation buttons when no data
            if (redobutton) {
                redobutton.classList.add('hidden');
            }
        }
    } else if (tabName === 'upload') {
        if (fileUpload) {
            fileUpload.classList.remove('hidden');
        }
        if (columnDiv) {
            columnDiv.classList.add('hidden');
        }
        // Hide redobutton on upload tab
        if (redobutton) {
            redobutton.classList.add('hidden');
        }
    } else if (tabName === 'model-preprocessing') {
        if (userInputSection) {
            userInputSection.classList.remove('hidden');
        }
        // Hide redobutton on model-preprocessing tab
        if (redobutton) {
            redobutton.classList.add('hidden');
        }
    } else if (tabName === 'modeling') {
        // Update multi-output model availability when modeling tab is shown
        if (typeof updateMultiOutputModelAvailability === 'function') {
            setTimeout(updateMultiOutputModelAvailability, 100); // Small delay to ensure DOM is ready
        }
        // Hide redobutton on modeling tab
        if (redobutton) {
            redobutton.classList.add('hidden');
        }
        if (columnSelection) {
            columnSelection.style.display = 'block';
            columnSelection.classList.remove('hidden'); // Ensure it's not hidden
            refreshModelSelections();
        }
        // Ensure results container is always visible - use setTimeout to ensure DOM is ready
        setTimeout(() => {
            // Determine which mode is active and show the corresponding results container
            const simpleMode = document.getElementById('simpleMode');
            const advancedMode = document.getElementById('advancedMode');
            const automlMode = document.getElementById('automlMode');
            
            let resultsContainer = null;
            let resultsPlaceholder = null;
            let hasResults = false;
            
            if (simpleMode && simpleMode.checked) {
                resultsContainer = document.getElementById('simpleModelingResults');
                resultsPlaceholder = document.getElementById('resultsPlaceholder');
                if (resultsPlaceholder) {
                    const numericDiv = document.getElementById('NumericResultDiv');
                    const clusterDiv = document.getElementById('ClusterResultDiv');
                    const classifierDiv = document.getElementById('ClassifierResultDiv');
                    // Add explicit null checks before calling classList.contains to prevent runtime errors
                    hasResults = (numericDiv !== null && !numericDiv.classList.contains('hidden')) ||
                                 (clusterDiv !== null && !clusterDiv.classList.contains('hidden')) ||
                                 (classifierDiv !== null && !classifierDiv.classList.contains('hidden'));
                }
            } else if (advancedMode && advancedMode.checked) {
                resultsContainer = document.getElementById('advancedModelingResults');
                resultsPlaceholder = document.getElementById('advancedResultsPlaceholder');
                if (resultsPlaceholder) {
                    const numericDiv = document.getElementById('AdvancedNumericResultDiv');
                    const clusterDiv = document.getElementById('AdvancedClusterResultDiv');
                    const classifierDiv = document.getElementById('AdvancedClassifierResultDiv');
                    // Add explicit null checks before calling classList.contains to prevent runtime errors
                    hasResults = (numericDiv !== null && !numericDiv.classList.contains('hidden')) ||
                                 (clusterDiv !== null && !clusterDiv.classList.contains('hidden')) ||
                                 (classifierDiv !== null && !classifierDiv.classList.contains('hidden'));
                }
            } else if (automlMode && automlMode.checked) {
                resultsContainer = document.getElementById('automlModelingResults');
                resultsPlaceholder = document.getElementById('automlResultsPlaceholder');
                if (resultsPlaceholder) {
                    const numericDiv = document.getElementById('AutoMLNumericResultDiv');
                    const clusterDiv = document.getElementById('AutoMLClusterResultDiv');
                    const classifierDiv = document.getElementById('AutoMLClassifierResultDiv');
                    // Add explicit null checks before calling classList.contains to prevent runtime errors
                    hasResults = (numericDiv !== null && !numericDiv.classList.contains('hidden')) ||
                                 (clusterDiv !== null && !clusterDiv.classList.contains('hidden')) ||
                                 (classifierDiv !== null && !classifierDiv.classList.contains('hidden'));
                }
            }
            
            if (resultsContainer) {
                resultsContainer.style.display = 'block';
                resultsContainer.style.visibility = 'visible';
                resultsContainer.classList.remove('hidden');
            }
            
            if (resultsPlaceholder) {
                if (!hasResults) {
                    resultsPlaceholder.style.display = 'block';
                } else {
                    resultsPlaceholder.style.display = 'none';
                }
            }
        }, 0);
    } else if (tabName === 'advanced-optimization') {
        // Redirect to unified Modeling page and switch to Advanced mode
        // First switch to modeling tab
        if (modelingSection) {
            modelingSection.classList.remove('hidden');
        }
        // Then switch to Advanced mode within the unified page
        const advancedModeRadio = document.getElementById('advancedMode');
        if (advancedModeRadio) {
            advancedModeRadio.checked = true;
            switchModelingMode('advanced');
        }
        // Hide redobutton on advanced-optimization tab
        if (redobutton) {
            redobutton.classList.add('hidden');
        }
    } else if (tabName === 'historic') {
        if (predictionDiv) {
            predictionDiv.classList.remove('hidden');
        }
        // Hide redobutton on historic tab
        if (redobutton) {
            redobutton.classList.add('hidden');
        }
    }

    setActiveTab(tabName);
    
    // DARK MODE DISABLED - Code kept for potential future use
    // Ensure dark mode is properly applied after tab switch
    // Use setTimeout to ensure DOM updates are complete
    /*
    setTimeout(() => {
        ensureDarkModeApplied();
    }, 0);
    */
}

function updateOutputTypeDisplay(outputType) {
    // Get current mode
    const simpleMode = document.getElementById('simpleMode');
    const advancedMode = document.getElementById('advancedMode');
    const automlMode = document.getElementById('automlMode');
    const currentMode = simpleMode?.checked ? 'simple' : (advancedMode?.checked ? 'advanced' : (automlMode?.checked ? 'automl' : 'simple'));
    
    // Simple model selectors (new mode-specific dropdowns)
    let numericModels = document.getElementById('simpleNumericModels');
    let clusterModels = document.getElementById("simpleClusterModels");
    let classifierModels = document.getElementById("simpleClassifierModels");
    let clusterTargetMessage = document.getElementById('clusterTargetMessage');
    
    // Advanced model selectors
    let advancedNumericModels = document.getElementById('advancedNumericModels');
    let advancedClusterModels = document.getElementById("advancedClusterModels");
    let advancedClassifierModels = document.getElementById("advancedClassifierModels");
    
    // AutoML model selectors
    let automlNumericModels = document.getElementById('automlNumericModels');
    let automlClusterModels = document.getElementById("automlClusterModels");
    let automlClassifierModels = document.getElementById("automlClassifierModels");
    
    // Header model selectors
    let headerNumericModels = document.getElementById('headerNumericModels');
    let headerClusterModels = document.getElementById('headerClusterModels');
    let headerClassifierModels = document.getElementById('headerClassifierModels');
    let headerAdvancedNumericModels = document.getElementById('headerAdvancedNumericModels');
    let headerAdvancedClusterModels = document.getElementById('headerAdvancedClusterModels');
    let headerAdvancedClassifierModels = document.getElementById('headerAdvancedClassifierModels');
    let headerAutomlNumericModels = document.getElementById('headerAutomlNumericModels');
    let headerAutomlClusterModels = document.getElementById('headerAutomlClusterModels');
    let headerAutomlClassifierModels = document.getElementById('headerAutomlClassifierModels');

    // Hide all model selectors (both section and header) based on current mode
    if (currentMode === 'simple') {
        // Hide advanced and AutoML selectors (section)
        if (advancedNumericModels) advancedNumericModels.classList.add("hidden");
        if (advancedClusterModels) advancedClusterModels.classList.add("hidden");
        if (advancedClassifierModels) advancedClassifierModels.classList.add("hidden");
        if (automlNumericModels) automlNumericModels.classList.add("hidden");
        if (automlClusterModels) automlClusterModels.classList.add("hidden");
        if (automlClassifierModels) automlClassifierModels.classList.add("hidden");
        
        // Hide advanced and AutoML selectors (header)
        if (headerAdvancedNumericModels) headerAdvancedNumericModels.classList.add("hidden");
        if (headerAdvancedClusterModels) headerAdvancedClusterModels.classList.add("hidden");
        if (headerAdvancedClassifierModels) headerAdvancedClassifierModels.classList.add("hidden");
        if (headerAutomlNumericModels) headerAutomlNumericModels.classList.add("hidden");
        if (headerAutomlClusterModels) headerAutomlClusterModels.classList.add("hidden");
        if (headerAutomlClassifierModels) headerAutomlClassifierModels.classList.add("hidden");
        
        // Show/hide simple selectors based on output type
        if (numericModels) numericModels.classList.add("hidden");
        if (clusterModels) clusterModels.classList.add("hidden");
        if (classifierModels) classifierModels.classList.add("hidden");
        if (headerNumericModels) headerNumericModels.classList.add("hidden");
        if (headerClusterModels) headerClusterModels.classList.add("hidden");
        if (headerClassifierModels) headerClassifierModels.classList.add("hidden");
    } else if (currentMode === 'advanced') {
        // Hide simple and AutoML selectors (section)
        if (numericModels) numericModels.classList.add("hidden");
        if (clusterModels) clusterModels.classList.add("hidden");
        if (classifierModels) classifierModels.classList.add("hidden");
        if (automlNumericModels) automlNumericModels.classList.add("hidden");
        if (automlClusterModels) automlClusterModels.classList.add("hidden");
        if (automlClassifierModels) automlClassifierModels.classList.add("hidden");
        
        // Hide simple and AutoML selectors (header)
        if (headerNumericModels) headerNumericModels.classList.add("hidden");
        if (headerClusterModels) headerClusterModels.classList.add("hidden");
        if (headerClassifierModels) headerClassifierModels.classList.add("hidden");
        if (headerAutomlNumericModels) headerAutomlNumericModels.classList.add("hidden");
        if (headerAutomlClusterModels) headerAutomlClusterModels.classList.add("hidden");
        if (headerAutomlClassifierModels) headerAutomlClassifierModels.classList.add("hidden");
        
        // Show/hide advanced selectors based on output type
        if (advancedNumericModels) advancedNumericModels.classList.add("hidden");
        if (advancedClusterModels) advancedClusterModels.classList.add("hidden");
        if (advancedClassifierModels) advancedClassifierModels.classList.add("hidden");
        if (headerAdvancedNumericModels) headerAdvancedNumericModels.classList.add("hidden");
        if (headerAdvancedClusterModels) headerAdvancedClusterModels.classList.add("hidden");
        if (headerAdvancedClassifierModels) headerAdvancedClassifierModels.classList.add("hidden");
    } else if (currentMode === 'automl') {
        // Hide simple and advanced selectors (section)
        if (numericModels) numericModels.classList.add("hidden");
        if (clusterModels) clusterModels.classList.add("hidden");
        if (classifierModels) classifierModels.classList.add("hidden");
        if (advancedNumericModels) advancedNumericModels.classList.add("hidden");
        if (advancedClusterModels) advancedClusterModels.classList.add("hidden");
        if (advancedClassifierModels) advancedClassifierModels.classList.add("hidden");
        
        // Hide simple and advanced selectors (header)
        if (headerNumericModels) headerNumericModels.classList.add("hidden");
        if (headerClusterModels) headerClusterModels.classList.add("hidden");
        if (headerClassifierModels) headerClassifierModels.classList.add("hidden");
        if (headerAdvancedNumericModels) headerAdvancedNumericModels.classList.add("hidden");
        if (headerAdvancedClusterModels) headerAdvancedClusterModels.classList.add("hidden");
        if (headerAdvancedClassifierModels) headerAdvancedClassifierModels.classList.add("hidden");
        
        // Show/hide AutoML selectors based on output type
        if (automlNumericModels) automlNumericModels.classList.add("hidden");
        if (automlClusterModels) automlClusterModels.classList.add("hidden");
        if (automlClassifierModels) automlClassifierModels.classList.add("hidden");
        if (headerAutomlNumericModels) headerAutomlNumericModels.classList.add("hidden");
        if (headerAutomlClusterModels) headerAutomlClusterModels.classList.add("hidden");
        if (headerAutomlClassifierModels) headerAutomlClassifierModels.classList.add("hidden");
    }

    if (outputType === 'Numeric') {
        if (currentMode === 'simple' && numericModels) numericModels.classList.remove("hidden");
        if (currentMode === 'advanced' && advancedNumericModels) advancedNumericModels.classList.remove("hidden");
        if (currentMode === 'automl' && automlNumericModels) automlNumericModels.classList.remove("hidden");
        if (clusterTargetMessage) clusterTargetMessage.classList.add('hidden');
        if (predictorsSelect) predictorsSelect.disabled = false;
    } else if (outputType === 'Cluster') {
        if (currentMode === 'simple' && clusterModels) clusterModels.classList.remove("hidden");
        if (currentMode === 'advanced' && advancedClusterModels) advancedClusterModels.classList.remove("hidden");
        if (currentMode === 'automl' && automlClusterModels) automlClusterModels.classList.remove("hidden");
        if (clusterTargetMessage) clusterTargetMessage.classList.remove('hidden');
        if (predictorsSelect) {
            predictorsSelect.value = '';
            predictorsSelect.disabled = true;
        }
    } else if (outputType === 'Classifier') {
        if (currentMode === 'simple' && classifierModels) classifierModels.classList.remove("hidden");
        if (currentMode === 'advanced' && advancedClassifierModels) advancedClassifierModels.classList.remove("hidden");
        if (currentMode === 'automl' && automlClassifierModels) automlClassifierModels.classList.remove("hidden");
        if (clusterTargetMessage) clusterTargetMessage.classList.add('hidden');
        if (predictorsSelect) predictorsSelect.disabled = false;
    }
}

function updateAutomlSettingsDisplay() {
    // Only update if we're in AutoML mode
    const simpleMode = document.getElementById('simpleMode');
    const advancedMode = document.getElementById('advancedMode');
    const automlMode = document.getElementById('automlMode');
    const currentMode = simpleMode?.checked ? 'simple' : (advancedMode?.checked ? 'advanced' : (automlMode?.checked ? 'automl' : 'simple'));
    
    if (currentMode !== 'automl') {
        return;
    }
    
    const outputType = document.getElementById("outputType1");
    if (!outputType) return;
    
    // Get model selection
    let modelName = 'Let AutoML choose';
    if (outputType.value === 'Numeric') {
        const modelSelect = document.getElementById('automlNModels');
        if (modelSelect && modelSelect.value) {
            modelName = modelSelect.options[modelSelect.selectedIndex].text;
        }
    } else if (outputType.value === 'Cluster') {
        const modelSelect = document.getElementById('automlClModels');
        if (modelSelect && modelSelect.value) {
            modelName = modelSelect.options[modelSelect.selectedIndex].text;
        }
    } else if (outputType.value === 'Classifier') {
        const modelSelect = document.getElementById('automlClassModels');
        if (modelSelect && modelSelect.value) {
            modelName = modelSelect.options[modelSelect.selectedIndex].text;
        }
    }
    
    // Get scaler
    const scalerSelect = getCachedElement('scaler');
    let scalerName = 'Standard';
    if (scalerSelect && scalerSelect.value) {
        scalerName = scalerSelect.options[scalerSelect.selectedIndex].text;
    }
    
    // Get AutoML intensity level
    const intensitySelect = document.getElementById('automlIntensity');
    const intensity = intensitySelect ? intensitySelect.value : 'medium';
    
    // AutoML settings based on intensity level
    // Rationale: AutoML should focus on core optimization (hyperparameters, model selection)
    // Feature selection and outlier detection are optional and may not always help
    let cvType, cvFolds, featureSelection, outlierHandling, hyperparameterSearch, crossValidation;
    
    if (intensity === 'quick') {
        // Quick: Minimal preprocessing, focus on fast hyperparameter search
        cvType = 'KFold';
        cvFolds = '3';
        featureSelection = 'None (let model handle features)';
        outlierHandling = 'None (preserve all data)';
        hyperparameterSearch = 'Randomized (20 iterations, 3 CV folds)';
        crossValidation = `${cvType} (${cvFolds} folds)`;
    } else if (intensity === 'long') {
        // Long: Comprehensive preprocessing + thorough search
        cvType = 'KFold';
        cvFolds = '10';
        featureSelection = 'RFE (10 features)';
        outlierHandling = 'Isolation Forest (remove)';
        hyperparameterSearch = 'Grid Search (exhaustive, 10 CV folds)';
        crossValidation = `${cvType} (${cvFolds} folds)`;
    } else {
        // Medium: Balanced approach with moderate preprocessing
        cvType = 'KFold';
        cvFolds = '5';
        featureSelection = 'RFE (10 features)';
        outlierHandling = 'Isolation Forest (remove)';
        hyperparameterSearch = 'Randomized (50 iterations, 5 CV folds)';
        crossValidation = `${cvType} (${cvFolds} folds)`;
    }
    
    // Update display elements
    const modelDisplay = document.getElementById('automlDisplayModel');
    const scalerDisplay = document.getElementById('automlDisplayScaler');
    const featureDisplay = document.getElementById('automlDisplayFeatureSelection');
    const outlierDisplay = document.getElementById('automlDisplayOutlier');
    const hyperparameterDisplay = document.getElementById('automlDisplayHyperparameter');
    const cvDisplay = document.getElementById('automlDisplayCrossValidation');
    
    if (modelDisplay) modelDisplay.textContent = modelName;
    if (scalerDisplay) scalerDisplay.textContent = scalerName;
    if (featureDisplay) featureDisplay.textContent = featureSelection;
    if (outlierDisplay) outlierDisplay.textContent = outlierHandling;
    if (hyperparameterDisplay) hyperparameterDisplay.textContent = hyperparameterSearch;
    if (cvDisplay) cvDisplay.textContent = crossValidation;
}

function refreshModelSelections() {
    const outputType = document.getElementById("outputType1");
    if (!outputType) {
        return;
    }
    updateOutputTypeDisplay(outputType.value);

    // Sync header dropdowns with section dropdowns
    const simpleMode = document.getElementById('simpleMode');
    const advancedMode = document.getElementById('advancedMode');
    const automlMode = document.getElementById('automlMode');
    const currentMode = simpleMode?.checked ? 'simple' : (advancedMode?.checked ? 'advanced' : (automlMode?.checked ? 'automl' : 'simple'));

    if (outputType.value === 'Numeric') {
        if (currentMode === 'simple') {
            const nModels = document.getElementById("simpleNModels");
            if (nModels) nModels.dispatchEvent(new Event("change"));
        } else if (currentMode === 'advanced') {
            const advancedNModels = document.getElementById("advancedNModels");
            if (advancedNModels) advancedNModels.dispatchEvent(new Event("change"));
        } else if (currentMode === 'automl') {
            const automlNModels = document.getElementById("automlNModels");
            if (automlNModels) automlNModels.dispatchEvent(new Event("change"));
        }
    } else if (outputType.value === 'Cluster') {
        if (currentMode === 'simple') {
            const clModels = document.getElementById("simpleClModels");
            if (clModels) clModels.dispatchEvent(new Event("change"));
        } else if (currentMode === 'advanced') {
            const advancedClModels = document.getElementById("advancedClModels");
            if (advancedClModels) advancedClModels.dispatchEvent(new Event("change"));
        } else if (currentMode === 'automl') {
            const automlClModels = document.getElementById("automlClModels");
            if (automlClModels) automlClModels.dispatchEvent(new Event("change"));
        }
    } else if (outputType.value === 'Classifier') {
        if (currentMode === 'simple') {
            const classModels = document.getElementById("simpleClassModels");
            if (classModels) classModels.dispatchEvent(new Event("change"));
        } else if (currentMode === 'advanced') {
            const advancedClassModels = document.getElementById("advancedClassModels");
            if (advancedClassModels) advancedClassModels.dispatchEvent(new Event("change"));
        } else if (currentMode === 'automl') {
            const automlClassModels = document.getElementById("automlClassModels");
            if (automlClassModels) automlClassModels.dispatchEvent(new Event("change"));
        }
    }

    const kernel = document.getElementById("kernel");
    if (kernel) {
        kernel.dispatchEvent(new Event("change"));
    }
    
    // Update AutoML settings display
    updateAutomlSettingsDisplay();
}

function clampSizeValue(value) {
    if (Number.isNaN(value)) {
        return 0;
    }
    return Math.min(1, Math.max(0, value));
}

function formatSizeValue(value) {
    return clampSizeValue(value).toFixed(2);
}

function syncTrainTest(changedInput) {
    if (!trainSizeInput || !testSizeInput) {
        return;
    }
    const value = clampSizeValue(parseFloat(changedInput.value));
    const otherValue = clampSizeValue(1 - value);
    if (changedInput === trainSizeInput) {
        trainSizeInput.value = formatSizeValue(value);
        testSizeInput.value = formatSizeValue(otherValue);
    } else {
        testSizeInput.value = formatSizeValue(value);
        trainSizeInput.value = formatSizeValue(otherValue);
    }
}

if (trainSizeInput && testSizeInput) {
    // Use 'change' and 'blur' events instead of 'input' to allow full typing before syncing
    trainSizeInput.addEventListener('change', () => syncTrainTest(trainSizeInput));
    trainSizeInput.addEventListener('blur', () => syncTrainTest(trainSizeInput));
    testSizeInput.addEventListener('change', () => syncTrainTest(testSizeInput));
    testSizeInput.addEventListener('blur', () => syncTrainTest(testSizeInput));
}

// Enable/disable "Continue to Modeling" button based on modeling type selection
const modelingTypeSelect = document.getElementById('modelingType');
const continueToModelingButton = document.getElementById('continueToModelingButton');
if (modelingTypeSelect && continueToModelingButton) {
    // Initially disable the button
    continueToModelingButton.disabled = true;
    
    // Enable/disable button when modeling type changes
    modelingTypeSelect.addEventListener('change', function() {
        if (this.value) {
            continueToModelingButton.disabled = false;
        } else {
            continueToModelingButton.disabled = true;
        }
    });
}

// Initialize tab button event listeners
if (tabButtons && tabButtons.length > 0) {
    tabButtons.forEach((button) => {
        button.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            if (button.dataset.tab === 'upload') {
                const columnDiv = document.getElementById('columnsection');
                const hasSessionData = Boolean(uploadedFileName)
                    || (columnSelection && columnSelection.dataset.ready === 'true')
                    || (columnDiv && !columnDiv.classList.contains('hidden'));
                if (hasSessionData) {
                    openResetPopup();
                    return;
                }
            }
            showTab(button.dataset.tab);
        });
    });
}

/// Section 1: Uploading CSV File
uploadForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const formData = new FormData(uploadForm);
    
    // Validate file is selected
    const fileInput = document.getElementById('file');
    if (!fileInput || !fileInput.files || fileInput.files.length === 0) {
        alert('Please select a CSV file to upload.');
        return;
    }
    
    // Validate file extension
    const fileName = fileInput.files[0].name;
    if (!fileName.toLowerCase().endsWith('.csv')) {
        alert('Please select a CSV file. Only .csv files are allowed.');
        return;
    }
    
    try {
        // goes to /upload route that gets column names
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData,
            
        });
        
        // Check response status before parsing JSON
        if (!response.ok) {
            let errorMessage = 'Upload failed. ';
            try {
                const errorData = await response.json();
                errorMessage += errorData.error || `Server returned ${response.status}`;
            } catch (parseError) {
                errorMessage += `Server returned ${response.status} ${response.statusText}`;
            }
            alert(errorMessage);
            console.error('Upload error:', errorMessage, response.status);
            return;
        }
        
        const data = await response.json();

        // once recieves column names
        uploadedFileName = formData.get('file').name;

        // show the divs for preprocessing and selecting targets, indicators, output type, etc
        let columnDiv = document.getElementById('columnsection');
        columnDiv.classList.remove('hidden');

        columnSelection.dataset.ready = 'false';
        columnSelection.style.display = 'none';
        setActiveTab('processing');

        // getting names of the first and last columns and their letter index
        indicatorsSelect.innerHTML = '';
        predictorsSelect.innerHTML = '';
        let lastColVal = getColumnLetter(data.numcols - 1)
        const columnList = document.getElementById('columnList');
        const uploadHeader = document.getElementById('uploadHeader');
        if (uploadHeader) {
            uploadHeader.textContent = 'Data Exploration';
            // Make the upload-card look like section-header (skinnier white block)
            const uploadCard = uploadHeader.closest('.upload-card');
            if (uploadCard) {
                uploadCard.classList.add('section-header');
                // Hide the upload form when showing Data Exploration
                const uploadForm = uploadCard.querySelector('#uploadForm');
                if (uploadForm) {
                    uploadForm.style.display = 'none';
                }
            }
        }
        const shortFileName = data.filename.length > 20
            ? `${data.filename.slice(0, 20)}…`
            : data.filename
        let warningsHtml = '';
        if (data.warnings && data.warnings.length > 0) {
            warningsHtml = `
                <div class="upload-warning" style="background-color: #fff3cd; border: 1px solid #ffc107; border-radius: 4px; padding: 12px; margin-top: 12px; margin-bottom: 12px;">
                    <strong style="color: #856404;">Warning:</strong>
                    <ul style="margin: 8px 0 0 0; padding-left: 20px; color: #856404;">
                        ${data.warnings.map(w => `<li>${w}</li>`).join('')}
                    </ul>
                </div>
            `;
        }
        
        columnList.innerHTML = `
            <table class="column-summary-table">
                <tbody>
                    <tr>
                        <th>File name</th>
                        <td>${shortFileName}</td>
                    </tr>
                    <tr>
                        <th>Total columns</th>
                        <td>${data.numcols}</td>
                    </tr>
                    <tr>
                        <th>First column</th>
                        <td>A: ${data.firstcol}</td>
                    </tr>
                    <tr>
                        <th>Last column</th>
                        <td>${lastColVal}: ${data.lastcol}</td>
                    </tr>
                    ${data.rows ? `<tr><th>Total rows</th><td>${data.rows.toLocaleString()}</td></tr>` : ''}
                    ${data.total_cells ? `<tr><th>Total cells</th><td>${data.total_cells.toLocaleString()}</td></tr>` : ''}
                </tbody>
            </table>
            ${warningsHtml}
        `

        let redobutton = document.getElementById("redobutton")
        redobutton.classList.remove("hidden")

        //hide upload form/button
        let uploadForm = document.getElementById("uploadForm")
        uploadForm.classList.add("hidden")

    } catch (error) {
        console.error('Upload error:', error);
        alert('An error occurred while uploading the file. Please check the console for details.');
    }
    
});

/// Section 2: Correlation Matrices
corrForm.addEventListener('submit', async(e) => {
    e.preventDefault();
    let corrCols = document.getElementById('corrCols').value
    if (corrCols == ''){
        corrColsIndices = 'all'
    }
    else {
        corrColsIndices = getColumnIndices(corrCols.toUpperCase().replace(/\s/g, ""));
    }

    const requestData = {
        filename: uploadedFileName,
        colsIgnore: corrColsIndices,
        dropMissing: document.getElementById('exploreDropMissing').value,
        imputeStrategy: document.getElementById('exploreImputeStrategy').value,
        dropZero: document.getElementById('exploreDrop0').value,
    };

    try {
        // sends to correlation matrices route to get the pdf and excel file generated
        const response = await fetch('/correlationMatrices', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestData),
        });
        let data = await response.json();
        if (!response.ok) {
            throw new Error(data.error || 'An error occurred.');
        }

        // show div to tell user they can download resutls
        let dataExploration = document.getElementById('dataExploration')
        const timestamp = formatDateTimeForFilename()
        const pdfDownloadName = `dataexploration_exploration_graphics_${timestamp}.pdf`
        const xlsxDownloadName = `dataexploration_exploration_stats_${timestamp}.xlsx`
        const pdfHref = `/download/correlation_matrices.pdf?download_name=${encodeURIComponent(pdfDownloadName)}`
        const xlsxHref = `/download/correlation_matrices.xlsx?download_name=${encodeURIComponent(xlsxDownloadName)}`
        dataExploration.innerHTML = `
            <a href="${pdfHref}" onclick="return downloadFile('correlation_matrices.pdf', '${pdfDownloadName}')">
                <button type="button" class="export-button">Graphics PDF</button>
            </a>
            <a href="${xlsxHref}" onclick="return downloadFile('correlation_matrices.xlsx', '${xlsxDownloadName}')">
                <button type="button" class="export-button">Stats XLSX</button>
            </a>
        `

        const explorationOutput = document.getElementById('explorationOutput')
        const statsRows = data.descriptive_stats || []
        let statsTableHtml = '<p>No descriptive stats available.</p>'
        if (statsRows.length) {
            const headers = [
                { key: "column", label: "column" },
                { key: "n", label: "n" },
                { key: "min", label: "min" },
                { key: "max", label: "max" },
                { key: "mean", label: "mean" },
                { key: "std", label: "std" },
                { key: "25", label: "25%" },
                { key: "50", label: "50%" },
                { key: "75", label: "75%" },
                { key: "100", label: "100%" },
            ]
            statsTableHtml = `
                <div class="model-stats-table-wrapper">
                    <table class="stats-table model-stats-table">
                        <thead>
                            <tr>${headers.map((header) => `<th>${header.label}</th>`).join('')}</tr>
                        </thead>
                        <tbody>
                            ${statsRows.map((row) => `
                                <tr>
                                    ${headers.map((header) => `<td>${row[header.key] ?? ''}</td>`).join('')}
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                </div>
            `
        }

        const correlationImages = data.correlation_images || {}
        const numericColumns = data.numeric_columns || []
        const pairplotImage = data.pairplot_image || ''
        const withCacheBust = (url) => url ? `${url}${url.includes('?') ? '&' : '?'}t=${Date.now()}` : ''
        const correlationOptions = [
            { key: "pearson", label: "Pearson" },
            { key: "spearman", label: "Spearman" },
            { key: "kendall", label: "Kendall" },
        ].filter((option) => correlationImages[option.key])
        const initialKey = correlationOptions[0]?.key
        const initialSrc = initialKey ? correlationImages[initialKey] : data.correlation_image
        const pairplotOptions = numericColumns
            .map((column) => `<option value="${column}">${column}</option>`)
            .join('')

        explorationOutput.innerHTML = `
            <div class="exploration-graphic-row">
                <div class="exploration-graphic">
                    <div class="matrix-heading">
                        <h3>Correlation Matrix</h3>
                        <label for="correlationMatrixSelect">Matrix type</label>
                        <select id="correlationMatrixSelect">
                            ${correlationOptions
                                .map((option) => `<option value="${option.key}">${option.label}</option>`)
                                .join('')}
                        </select>
                    </div>
                    <img id="correlationMatrixImage" src="${withCacheBust(initialSrc)}" alt="Correlation matrix heatmap">
                </div>
                <div class="exploration-graphic pairplot-graphic">
                    <div class="matrix-heading">
                        <h3>Pairplot</h3>
                        <label for="pairplotXSelect">X</label>
                        <select id="pairplotXSelect">
                            ${pairplotOptions}
                        </select>
                        <label for="pairplotYSelect">Y</label>
                        <select id="pairplotYSelect">
                            ${pairplotOptions}
                        </select>
                    </div>
                    <img id="pairplotImage" src="${withCacheBust(pairplotImage)}" alt="Pairplot preview">
                </div>
            </div>
            <div class="exploration-table">
                <h3 style="display: inline-block; margin-right: 12px; margin-bottom: 8px;">Descriptive Statistics</h3>
                ${statsRows.length > 5 ? '<span style="color: #666; font-size: 0.9rem; font-weight: normal;">Scroll to see more rows</span>' : ''}
                ${statsTableHtml}
            </div>
        `

        const matrixSelect = document.getElementById('correlationMatrixSelect')
        const matrixImage = document.getElementById('correlationMatrixImage')
        if (matrixSelect && matrixImage) {
            matrixSelect.addEventListener('change', (event) => {
                const selectedKey = event.target.value
                matrixImage.src = withCacheBust(correlationImages[selectedKey])
            })
        }

        const pairplotXSelect = document.getElementById('pairplotXSelect')
        const pairplotYSelect = document.getElementById('pairplotYSelect')
        const pairplotImageEl = document.getElementById('pairplotImage')
        if (pairplotXSelect && pairplotYSelect && pairplotImageEl && numericColumns.length >= 2) {
            // Set initial values without triggering change events
            pairplotXSelect.value = numericColumns[0]
            pairplotYSelect.value = numericColumns[1]
            
            const updatePairplot = async () => {
                // Validate that both selects have values
                if (!pairplotXSelect.value || !pairplotYSelect.value) {
                    console.warn('Pairplot: Both X and Y must be selected');
                    return;
                }
                
                // Prevent duplicate requests if one is already in progress
                if (pairplotImageEl.dataset.updating === 'true') {
                    return;
                }
                pairplotImageEl.dataset.updating = 'true';
                
                // Add a slight opacity change to show it's updating
                pairplotImageEl.style.opacity = '0.6';
                
                try {
                    // Get colsIgnore from the correlation form (same as used for correlation matrices)
                    let corrCols = document.getElementById('corrCols')?.value || '';
                    let colsIgnore = 'all';
                    if (corrCols.trim() !== '') {
                        colsIgnore = getColumnIndices(corrCols.toUpperCase().replace(/\s/g, ""));
                    }
                    
                    const response = await fetch('/pairplot', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            x: pairplotXSelect.value,
                            y: pairplotYSelect.value,
                            colsIgnore: colsIgnore,
                            dropMissing: document.getElementById('exploreDropMissing')?.value || 'none',
                            imputeStrategy: document.getElementById('exploreImputeStrategy')?.value || 'none',
                            dropZero: document.getElementById('exploreDrop0')?.value || 'none',
                        }),
                    });
                    
                    if (!response.ok) {
                        const errorData = await response.json().catch(() => ({ error: 'Unknown error' }));
                        console.error('Pairplot update failed:', errorData.error || response.statusText);
                        pairplotImageEl.style.opacity = '1';
                        pairplotImageEl.dataset.updating = 'false';
                        return;
                    }
                    
                    const pairplotData = await response.json();
                    if (pairplotData.pairplot_image) {
                        // More aggressive cache busting with timestamp and random number
                        const timestamp = Date.now();
                        const random = Math.random().toString(36).substring(7);
                        const newSrc = `${pairplotData.pairplot_image}${pairplotData.pairplot_image.includes('?') ? '&' : '?'}_=${timestamp}&r=${random}`;
                        
                        console.log('Pairplot updating:', {
                            x: pairplotXSelect.value,
                            y: pairplotYSelect.value,
                            imageUrl: pairplotData.pairplot_image,
                            newSrc: newSrc
                        });
                        
                        // Store current reference
                        const currentImg = pairplotImageEl;
                        const oldSrc = currentImg.src;
                        
                        // Clear the src completely and wait a frame
                        currentImg.src = '';
                        currentImg.style.display = 'none';
                        
                        // Use multiple frames to ensure browser processes the change
                        requestAnimationFrame(() => {
                            requestAnimationFrame(() => {
                                // Set the new src
                                currentImg.src = newSrc;
                                currentImg.style.display = '';
                                
                                // Wait for image to actually load
                                const imgLoadHandler = () => {
                                    currentImg.style.opacity = '1';
                                    currentImg.dataset.updating = 'false';
                                    currentImg.removeEventListener('load', imgLoadHandler);
                                    currentImg.removeEventListener('error', imgErrorHandler);
                                };
                                
                                const imgErrorHandler = () => {
                                    console.error('Pairplot: Failed to load image');
                                    currentImg.style.opacity = '1';
                                    currentImg.style.display = '';
                                    currentImg.dataset.updating = 'false';
                                    currentImg.removeEventListener('load', imgLoadHandler);
                                    currentImg.removeEventListener('error', imgErrorHandler);
                                };
                                
                                // Add one-time event listeners
                                currentImg.addEventListener('load', imgLoadHandler, { once: true });
                                currentImg.addEventListener('error', imgErrorHandler, { once: true });
                                
                                // Fallback timeout in case events don't fire
                                setTimeout(() => {
                                    if (currentImg.dataset.updating === 'true') {
                                        currentImg.style.opacity = '1';
                                        currentImg.style.display = '';
                                        currentImg.dataset.updating = 'false';
                                    }
                                }, 5000);
                            });
                        });
                    } else {
                        console.warn('Pairplot: No image returned from server');
                        pairplotImageEl.style.opacity = '1';
                        pairplotImageEl.dataset.updating = 'false';
                    }
                } catch (error) {
                    console.error('Pairplot update error:', error);
                    pairplotImageEl.style.opacity = '1';
                    pairplotImageEl.dataset.updating = 'false';
                }
            };
            
            // Attach event listeners
            pairplotXSelect.addEventListener('change', updatePairplot);
            pairplotYSelect.addEventListener('change', updatePairplot);
        }
    }
    catch (error) {
        showError(errorDiv, error.message || 'An error occurred.');
    }

});

function formatDateTimeForFilename(date = new Date()) {
    const pad = (value) => String(value).padStart(2, "0")
    const year = date.getFullYear()
    const month = pad(date.getMonth() + 1)
    const day = pad(date.getDate())
    const hours = pad(date.getHours())
    const minutes = pad(date.getMinutes())
    const seconds = pad(date.getSeconds())
    return `${year}${month}${day}_${hours}${minutes}${seconds}`
}

function downloadFile(filename, downloadName = filename) {
    // Check if we're in a desktop app with pywebview API
    try {
        if (window.pywebview?.api?.save_file) {
            pywebviewReady = true;
            console.log('Calling pywebview save_file API:', filename, downloadName);
            window.pywebview.api.save_file(filename, downloadName)
                .then((success) => {
                    if (success) {
                        console.log('File saved successfully via pywebview API');
                    } else {
                        console.warn('File save was cancelled or failed');
                    }
                })
                .catch((error) => {
                    console.error('Error saving file via pywebview API:', error);
                    // Fallback to regular download if API fails
                    console.log('Falling back to regular download');
                });
            return false; // Prevent default link behavior
        }
    } catch (error) {
        console.error('Error accessing pywebview API:', error);
    }
    // If not in desktop app, allow normal download
    return true;
}

function showCrossValidationUnavailable() {
    const errorDiv = getCachedElement('errorDiv');
    if (errorDiv) {
        showError(errorDiv, 'Cross-validation results are unavailable because cross-validation was not run.');
        errorDiv.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
    return false;
}

// Function to download additional information table as Excel
function downloadAdditionalInfoTable(tableData, sheetName, timestamp) {
    // Send data to backend to generate Excel file
    fetch('/downloadAdditionalInfo', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            table_data: tableData,
            sheet_name: sheetName,
        }),
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            const errorDiv = getCachedElement('errorDiv');
            if (errorDiv) {
                showError(errorDiv, 'Error generating download: ' + data.error);
                errorDiv.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }
            return;
        }
        const downloadName = `additional_info_${sheetName.toLowerCase().replace(/\s+/g, '_')}_${timestamp}.xlsx`;
        const href = `/download/${data.filename}?download_name=${encodeURIComponent(downloadName)}`;
        const link = document.createElement('a');
        link.href = href;
        link.click();
        downloadFile(data.filename, downloadName);
    })
    .catch(error => {
        console.error('Error downloading additional information:', error);
        const errorDiv = getCachedElement('errorDiv');
        if (errorDiv) {
            showError(errorDiv, 'Error downloading file. Please try again.');
            errorDiv.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
    });
}

/// Section 3: when 'Process' Button clicked
    // Ensures required input is there and gets the columns names of selected indicators and targets using the /preprocess route
preprocessform.addEventListener('submit', async (e) => {
    e.preventDefault();
    const stratErrorDiv = document.getElementById('stratErrorDiv')
    const stratifyColumn = document.getElementById('specificVariableSelect').value;
    const quantileBins = document.getElementById('quantileBins').value;
    const bins = document.getElementById('bins').value;
    const binsLabel = document.getElementById('binsLabel').value
    const quantiles = document.getElementById('quantiles').value;
    const indicators = indicatorsSelect.value
    const predictors = predictorsSelect.value
    let outputType = document.getElementById('outputType1').value

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
    
    //if required variables are filled in send to route to get message displayed with column names
    else {
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
        
        stratErrorDiv.innerHTML = ``
        let columnDiv = document.getElementById('columnsection');
        columnDiv.classList.add('hidden');
        let fileuploaddiv = document.getElementById('fileuploaddiv');
        fileuploaddiv.classList.add('hidden');
        let userInputSection = document.getElementById('userInputSection');
        userInputSection.classList.add('hidden');
        
        let columnSelection = document.getElementById('columnSelection');
        columnSelection.dataset.ready = 'true';
        columnSelection.style.display = 'block';
        
        // Always route to unified Modeling page (mode selection happens there)
        showTab('modeling');
        
        
        const predictorCols = getColumnIndices(predictors.toUpperCase().replace(/\s/g, ""));
        const indicatorCols = getColumnIndices(indicators.toUpperCase().replace(/\s/g, ""));

        stratifyColumnNumber = columnToIndex(stratifyColumn.toUpperCase())

        const requestData = {
            filename: uploadedFileName,
            indicators: indicatorCols,
            predictors: predictorCols,
            stratify: stratifyColumnNumber
        };

        try {
            const response = await fetch('/preprocess', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(requestData),
            });
            let data = await response.json();

            
            stratifyStr = ''
            const stratifyColumn = document.getElementById('specificVariableSelect').value;
            if (stratifyColumn !== ''){
                stratifyStr = 'with stratification by ' + data['stratify'] + ' value'
            }

            let predictorsColNameString = data['predictors'].join(",").substring(0,10)
            if (predictorsColNameString.length == 10){
                predictorsColNameString += '...'
            }

            let IndicatorsPredictorsSection = document.getElementById('modelingHeaderActions')
            const outputTypeLabel = outputType === 'Numeric' ? 'Regression' : outputType

            // Displaying what was selected to user 
            if (outputType=='Cluster'){
                const noteText = `<em>Columns ${escapeHtml(indicators.toUpperCase())} are selected as indicators for ${escapeHtml(outputTypeLabel)} based modeling.</em>`;
                // Update mode-specific notes
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
                const noteText = `<em>Columns ${escapeHtml(indicators.toUpperCase())} are selected as indicators to predict column(s) ${escapeHtml(predictors.toUpperCase())} (${escapeHtml(predictorsColNameString)}) by ${escapeHtml(outputTypeLabel)} based modeling.</em>`;
                // Update mode-specific notes
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
            

        }

        catch (error) {
            showError(errorDiv, 'An error occurred.');
        }
    }
    


});

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

    // Auto-detect categorical columns for transformers
    const autoDetectTransformersBtn = document.getElementById('autoDetectTransformers');
    if (autoDetectTransformersBtn) {
        autoDetectTransformersBtn.addEventListener('click', async function() {
            // Get selected indicators from text input
            const indicatorsInput = document.getElementById('indicators');
            if (!indicatorsInput || !indicatorsInput.value.trim()) {
                alert('Please enter indicator columns first (e.g., A-D).');
                return;
            }

            // Convert indicator column letters to indices
            const indicatorIndices = getColumnIndices(indicatorsInput.value.toUpperCase().replace(/\s/g, ""));

            if (indicatorIndices.length === 0) {
                alert('Could not parse indicator column indices. Please use format like "A-D" or "A,B,C".');
                return;
            }

            try {
                autoDetectTransformersBtn.disabled = true;
                autoDetectTransformersBtn.textContent = 'Detecting...';

                const response = await fetch('/auto-detect-transformers', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        indicators: indicatorIndices
                    })
                });

                const data = await response.json();

                if (response.ok && data.transformer_indices && data.transformer_indices.length > 0) {
                    // Convert indices to column letters (e.g., [0, 1, 2] -> "A-C")
                    const transformerColumnInput = document.getElementById('transformerColumn');
                    if (transformerColumnInput) {
                        // Convert indices to column letter range
                        const letters = data.transformer_indices.map(idx => getColumnLetter(idx));
                        if (letters.length === 1) {
                            transformerColumnInput.value = letters[0];
                        } else {
                            // Check if consecutive
                            const sorted = data.transformer_indices.slice().sort((a, b) => a - b);
                            const isConsecutive = sorted.every((val, i, arr) => i === 0 || val === arr[i - 1] + 1);
                            if (isConsecutive) {
                                transformerColumnInput.value = `${getColumnLetter(sorted[0])}-${getColumnLetter(sorted[sorted.length - 1])}`;
                            } else {
                                transformerColumnInput.value = letters.join(', ');
                            }
                        }
                        alert(`Auto-detected ${data.transformer_indices.length} categorical column(s): ${data.message}`);
                    }
                } else {
                    const message = data.message || data.error || 'No categorical columns detected.';
                    alert(message);
                    if (data.transformer_indices && data.transformer_indices.length === 0) {
                        // Clear the input if nothing detected
                        const transformerColumnInput = document.getElementById('transformerColumn');
                        if (transformerColumnInput) {
                            transformerColumnInput.value = '';
                        }
                    }
                }
            } catch (error) {
                console.error('Error auto-detecting transformers:', error);
                alert('Error auto-detecting categorical columns. Please check the console for details.');
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
        let svcGamma = document.getElementById("SVCGamma");

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

    //hides welcome page when user clicks 'start modeling'
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
            console.log('Start Modeling button clicked via event listener');
            welcomePage();
        });
    }
}

// Function to reset to welcome screen
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
    
    // Hide all main sections
    const fileUpload = document.getElementById('fileUpload');
    const documentationSection = document.getElementById('documentationSection');
    const userInputSection = document.getElementById('userInputSection');
    const predictionDiv = document.getElementById('predictionDiv');
    const processingDiv = document.getElementById('processingDiv');
    const modelPreprocessingDiv = document.getElementById('modelPreprocessingDiv');
    const modelingDiv = document.getElementById('modelingDiv');
    
    if (fileUpload) fileUpload.classList.add('hidden');
    if (documentationSection) documentationSection.classList.add('hidden');
    if (userInputSection) userInputSection.classList.add('hidden');
    if (predictionDiv) predictionDiv.classList.add('hidden');
    if (processingDiv) processingDiv.classList.add('hidden');
    if (modelPreprocessingDiv) modelPreprocessingDiv.classList.add('hidden');
    if (modelingDiv) modelingDiv.classList.add('hidden');
    
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
    });
} else {
    // DOM is already loaded
    setupStartModelingButton();
    setupHeaderLogoClick();
}


function moveToModelPreprocess(){
    showTab('model-preprocessing')
}
    // // opens popup
// function openPopup() {
//     document.getElementById("popup").style.display = "flex";
// }

    //closes the popup for glossary
function closePopup() {
    document.getElementById("popup").style.display = "none";
}
    // opens the 'are you sure you want to restart' popup when user clicks use new dataset
function openResetPopup(){
    document.getElementById("resetPopup").style.display = "flex";
}
    // closes the 'are you sure you want to restart' popup
function closeResetPopup(){
    document.getElementById("resetPopup").style.display = "none";
}

    /// handles 'Change Columns or Output Type' Button - goes back to user input page 
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
function predictionPage(){
    if (columnSelection) columnSelection.style.display = 'none';
    const predDiv = getCachedElement('predictionDiv');
    showElement(predDiv);
    showTab('historic');
}
//upload new prediction file
function newPredict(){
    const predictionErrorDiv = getCachedElement('predictionErrorDiv');
    const uploadPredictDf = getCachedElement('uploadPredictDf');
    const predictionResults = getCachedElement('predictionResults');

    showElement(uploadPredictDf);
    showElement(predictionErrorDiv);
    hideElement(predictionResults);
}

// goes back to model from prediction page
function backToModel(){
    if (columnSelection) columnSelection.style.display = 'block';
    hideElement(predictionDiv);
    hideElement(predictionResultsDiv);
    showTab('modeling');
}

// Helper function to copy hyperparameter values from Simple to Advanced
function copyHyperparametersToAdvanced(selectedModel, outputType) {
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

// Global variables for progress tracking
let progressEventSource = null;
let sessionId = null;
let processResultData = null;

// Function to stop current model run
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
    const models = getCachedElement('models');
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
        
        hyperparameters['hidden_layer_sizes1'] = hidden_layer_sizes1.value
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
            const gamma = document.getElementById('Gamma');
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
            hyperparameters['SVCClassWeight'] = SVCClassWeight
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
    let unitMessageStr = ''
    let unitStr = ''
    if (units) {
        let newUnits = units.replace("u*", "\u00B5")
        unitMessageStr = `with ${newUnits} units`
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
        const estimatedTimeRemaining = progress.estimated_time_remaining || 0;
        
        // Check if this is AutoML mode for enhanced display
        const simpleMode = document.getElementById('simpleMode');
        const advancedMode = document.getElementById('advancedMode');
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
            showError(errorDiv, `Error: ${data.error || 'Unknown error'}`);
        }
    } catch (error) {
        console.error('Error:', error);
        // Always clean up on error, regardless of async processing state
        // This handles cases where error occurs after isAsyncProcessing is set but before SSE is established
        if (progressEventSource) {
            progressEventSource.close();
            progressEventSource = null;
        }
        showError(errorDiv, 'An error occurred.');
        
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

// Add event listeners for stop buttons
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
        let detailsStr = ''
        if (Object.keys(allHyperparameters).length!==0){
            detailsStr = 'and hyperparameters: <br>'
            detailsStr += '<table class="hyperparameterstable" border="1">';
            detailsStr += '<tr><th>Hyperparameter</th><th>Value</th></tr>';
            detailsStr += Object.entries(allHyperparameters)
                .map(([key, value]) => `<tr><td>${key}</td><td>${value !== null && value !== undefined ? value : 'N/A'}</td></tr>`)
                .join("");
            detailsStr += '</table><br>';
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
                    // Remove "Baseline" from labels for Simple Modeling page
                    const regressionVisualsClean = regressionVisuals.map(v => ({
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
                    <label for="${currentMode === 'simple' ? 'regressionVisualSelector' : currentMode === 'advanced' ? 'advancedRegressionVisualSelector' : 'automlRegressionVisualSelector'}">Select Visualization to Display</label>
                    <select id="${currentMode === 'simple' ? 'regressionVisualSelector' : currentMode === 'advanced' ? 'advancedRegressionVisualSelector' : 'automlRegressionVisualSelector'}">
                        ${regressionVisualsClean
                            .map((visual) => `<option value="${visual.file}">${visual.label}</option>`)
                            .join('')}
                    </select>
                    <label for="${currentMode === 'simple' ? 'imageSelector' : currentMode === 'advanced' ? 'advancedImageSelector' : 'automlImageSelector'}">Select Target Graphic to Display</label>
                    <select id="${currentMode === 'simple' ? 'imageSelector' : currentMode === 'advanced' ? 'advancedImageSelector' : 'automlImageSelector'}"></select>
                    <br>
                    <br>
                    <img id="${currentMode === 'simple' ? 'targetGraphic' : currentMode === 'advanced' ? 'advancedTargetGraphic' : 'automlTargetGraphic'}" class="result-graphic" src='/user-visualizations/target_plot_1${currentMode === 'advanced' ? '_advanced' : ''}.png?t=${new Date().getTime()}' alt="Model visualization">
                        <div><br></div>
                        
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
                    
                    //populating drop down of graphics to display for multiple targets
                    // Use the mode-specific imageSelector already defined at function start
                    if (!imageSelector) {
                        // Fallback if not set (shouldn't happen, but safety check)
                        imageSelector = document.getElementById("imageSelector") || 
                                       document.getElementById("advancedImageSelector") || 
                                       document.getElementById("automlImageSelector");
                    }
                    // Use mode-specific IDs for target graphic and visual selector
                    const targetGraphicId = currentMode === 'simple' ? 'targetGraphic' : currentMode === 'advanced' ? 'advancedTargetGraphic' : 'automlTargetGraphic';
                    const visualSelectorId = currentMode === 'simple' ? 'regressionVisualSelector' : currentMode === 'advanced' ? 'advancedRegressionVisualSelector' : 'automlRegressionVisualSelector';
                    let targetGraphic = document.getElementById(targetGraphicId);
                    const regressionVisualSelector = document.getElementById(visualSelectorId);

                    data.predictors.forEach((predictor, index) => {
                    const option = document.createElement("option");
                    option.value = index + 1;
                    option.textContent = predictor.split('/').pop(); // Just show filename
                    imageSelector.appendChild(option);
                    });

                    const updateRegressionGraphic = () => {
                        const selectedImage = imageSelector.value;
                        const selectedVisual = regressionVisualSelector ? regressionVisualSelector.value : 'target_plot';
                        
                        // Find the visual object to get its type and file name
                        const visualObj = regressionVisuals.find(v => v.file === selectedVisual);
                        const visualType = visualObj ? visualObj.type : 'default';
                        
                        if (selectedVisual !== 'target_plot' && selectedVisual !== 'target_plot_advanced') {
                            // For non-target_plot visuals (like shap_summary, etc.)
                            let filename = selectedVisual;
                            // Add .png extension if not present and not already a full filename
                            if (!filename.includes('.png') && !filename.includes('_advanced')) {
                                filename = filename.endsWith('_advanced') ? `${filename}.png` : `${filename}.png`;
                            } else if (filename.endsWith('_advanced') && !filename.includes('.png')) {
                                filename = `${filename}.png`;
                            }
                            targetGraphic.src = `/user-visualizations/${filename}?t=${Date.now()}`;
                            return;
                        }
                        
                        // For target_plot visuals, construct filename with appropriate suffix
                        let suffix = '';
                        if (selectedVisual === 'target_plot_advanced' || (visualType === 'advanced')) {
                            suffix = '_advanced';
                        } else if (visualType === 'baseline') {
                            suffix = ''; // Baseline has no suffix
                        }
                        // Default (backward compatibility) also has no suffix
                        
                        targetGraphic.src = `/user-visualizations/target_plot_${selectedImage}${suffix}.png?t=${Date.now()}`;
                    };
                    // Add error handler for image loading
                    if (targetGraphic) {
                        targetGraphic.onerror = function() {
                            console.error('Failed to load graphic:', this.src);
                        };
                    }

                    imageSelector.addEventListener("change", updateRegressionGraphic);
                    if (regressionVisualSelector) {
                        regressionVisualSelector.addEventListener("change", updateRegressionGraphic);
                    }
                    // Set initial graphic based on first selected option
                    updateRegressionGraphic();
                    
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
                                    advancedTargetGraphic.src = `/user-visualizations/${filename}?t=${Date.now()}`;
                                    return;
                                }
                                
                                let suffix = '';
                                if (selectedVisual === 'target_plot_advanced' || (visualType === 'advanced')) {
                                    suffix = '_advanced';
                                }
                                
                                advancedTargetGraphic.src = `/user-visualizations/target_plot_${selectedImage}${suffix}.png?t=${Date.now()}`;
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
                    // Remove "Baseline" from labels for Simple Modeling page
                    const regressionVisualsClean = regressionVisuals.map(v => ({
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
                    <label for="regressionVisualSelector">Select Visualization to Display</label>
                    <select id="regressionVisualSelector">
                        ${regressionVisualsClean
                            .map((visual) => `<option value="${visual.file}">${visual.label}</option>`)
                            .join('')}
                    </select>
                    <br>
                    <br>
                    <img id="targetGraphic" class="result-graphic" src='/user-visualizations/target_plot_1.png?t=${new Date().getTime()}'>
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
                const targetGraphic = document.getElementById("targetGraphic");
                if (regressionVisualSelector && targetGraphic) {
                    const updateGraphic = () => {
                        const selectedVisual = regressionVisualSelector.value;
                        
                        // Find the visual object to get its type and file name
                        const visualObj = regressionVisuals.find(v => v.file === selectedVisual);
                        const visualType = visualObj ? visualObj.type : 'default';
                        
                        if (selectedVisual !== 'target_plot' && selectedVisual !== 'target_plot_advanced') {
                            // For non-target_plot visuals (like shap_summary, etc.)
                            let filename = selectedVisual;
                            // Add .png extension if not present and not already a full filename
                            if (!filename.includes('.png') && !filename.includes('_advanced')) {
                                filename = filename.endsWith('_advanced') ? `${filename}.png` : `${filename}.png`;
                            } else if (filename.endsWith('_advanced') && !filename.includes('.png')) {
                                filename = `${filename}.png`;
                            }
                            targetGraphic.src = `/user-visualizations/${filename}?t=${Date.now()}`;
                            return;
                        }
                        
                        // For target_plot visuals, construct filename with appropriate suffix
                        let suffix = '';
                        if (selectedVisual === 'target_plot_advanced' || (visualType === 'advanced')) {
                            suffix = '_advanced';
                        } else if (visualType === 'baseline') {
                            suffix = ''; // Baseline has no suffix
                        }
                        // Default (backward compatibility) also has no suffix
                        
                        targetGraphic.src = `/user-visualizations/target_plot_1${suffix}.png?t=${Date.now()}`;
                    };
                    regressionVisualSelector.addEventListener("change", updateGraphic);
                    // Add error handler for image loading
                    targetGraphic.onerror = function() {
                        console.error('Failed to load graphic:', this.src);
                    };
                    // Set initial graphic based on first selected option
                    updateGraphic();
                    
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
                                    advancedTargetGraphicSingle.src = `/user-visualizations/${filename}?t=${Date.now()}`;
                                    return;
                                }
                                
                                let suffix = '';
                                if (selectedVisual === 'target_plot_advanced' || (visualType === 'advanced')) {
                                    suffix = '_advanced';
                                }
                                
                                advancedTargetGraphicSingle.src = `/user-visualizations/target_plot_1${suffix}.png?t=${Date.now()}`;
                            };
                            
                            advancedVisualSelectorSingle.addEventListener("change", updateAdvancedGraphicSingle);
                            updateAdvancedGraphicSingle();
                        }
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
                const hasAdvancedOptions = data.feature_selection_info || data.outlier_info;
                
                ClassifierResultDiv.innerHTML = ` 
                <div class="resultValues">
                    <div style="display: flex; gap: 20px; flex-wrap: wrap; align-items: flex-start;">
                        <div style="flex: 1; min-width: 300px;">
                            <h3 style="margin: 0; margin-bottom: 10px;">Performance</h3> 
                            <div class="model-stats-table-wrapper">
                                <table class="stats-table model-stats-table performance-table">
                                    <tr><th>Value</th><th>Weighted Average</th></tr>
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
                <label for="classifierImageSelector">Select Graphic to Display</label>
                <select id="classifierImageSelector">
                    <option value="confusion_matrix">Confusion Matrix</option>
                    <option value="roc_curve">ROC Curve (micro)</option>
                    <option value="pr_curve">Precision-Recall Curve (micro)</option>
                </select>
                <br>
                <br>
                <img id="classifierGraphic" class="result-graphic" src='/user-visualizations/confusion_matrix.png?t=${new Date().getTime()}'>
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
                
                // Set up classifier image selector event listener
                const classifierImageSelector = document.getElementById('classifierImageSelector');
                const classifierGraphic = document.getElementById('classifierGraphic');
                if (classifierImageSelector && classifierGraphic) {
                    classifierImageSelector.addEventListener('change', () => {
                        const selectedImage = classifierImageSelector.value;
                        classifierGraphic.src = `/user-visualizations/${selectedImage}.png?t=${Date.now()}`;
                    });
                }
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
                    <label for="clusterImageSelector">Select Graphic to Display</label>
                    <select id="clusterImageSelector">
                        <option value="cluster_pca_train">PCA (Train)</option>
                        <option value="cluster_pca_test">PCA (Test)</option>
                    </select>
                    <br>
                    <br>
                    <img id="clusterGraphic" class="result-graphic" src='/user-visualizations/cluster_pca_train.png?t=${new Date().getTime()}'>
                    <div><br></div>
                </div> `
                const clusterImageSelector = getCachedElement('clusterImageSelector')
                const clusterGraphic = getCachedElement('clusterGraphic')
                if (clusterImageSelector && clusterGraphic) {
                    clusterImageSelector.addEventListener('change', () => {
                        const selectedImage = clusterImageSelector.value
                        clusterGraphic.src = `/user-visualizations/${selectedImage}.png?t=${Date.now()}`
                    })
                }
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
        showError(errorDiv, 'An error occurred processing the result.');
    }
}


///Section 7: prediction
predictionForm.addEventListener('submit', async (e) => {
    e.preventDefault()
    let predictionErrorDiv = document.getElementById('predictionErrorDiv')
    let uploadPredictDf = document.getElementById('uploadPredictDf')
    let predictionResults = document.getElementById('predictionResults')

    //const predictFile = document.getElementById('predictFile').files[0];
    const formData = new FormData(predictionForm);
    formData.append('indicators', 'indicators')
    const resultTimestamp = formatDateTimeForFilename() 
    const predictionDownloadName = `predictions${resultTimestamp}.csv`

    try{
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData,
        });
        let data = await response.json();
        if (response.ok === true) {
            predictionErrorDiv.classList.add('hidden')
            predictionResults.classList.remove('hidden')
            console.log(data.filename)
            // <a href="/download/predictions.csv" onclick="return downloadFile('predictions.csv')">
            predictionResults.innerHTML = `
                <h2>Prediction Results</h2>
                <p>Your results for the prediction with '<strong>${escapeHtml(data.filename || 'file')}</strong>' are ready to download.</p>
                <div class="button-group">
                    <a href="/download/predictions.csv?download_name=${encodeURIComponent(predictionDownloadName)}" onclick="return downloadFile('predictions.csv', '${predictionDownloadName}')">
                        <button class="predictionresultButton export-button">Download Results CSV</button>
                    </a>
                    <button class="secondary-button" onclick="backToModel()">Back To Model</button>
                    <button class="secondary-button" onclick="newPredict()">Predict Another Dataset</button>
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

// ============================================================================
// DARK MODE DISABLED - Code kept for potential future use
// ============================================================================
// Dark Mode Toggle
// ============================================================================

// Initialize dark mode immediately (before DOMContentLoaded) to prevent flash
// Only enable if explicitly set to 'true' (default is off)
/*
(function() {
    const darkModeSetting = localStorage.getItem('darkMode');
    // Default to light mode - only enable dark mode if explicitly set to 'true'
    if (darkModeSetting === 'true') {
        document.documentElement.classList.add('dark-mode');
        document.body.classList.add('dark-mode');
    } else {
        // Ensure dark mode is off by default (light mode)
        document.documentElement.classList.remove('dark-mode');
        document.body.classList.remove('dark-mode');
        // Explicitly set to light mode if not already set
        if (darkModeSetting === null) {
            localStorage.setItem('darkMode', 'false');
        }
    }
})();
*/
// Always ensure light mode (remove any dark mode classes)
(function() {
    document.documentElement.classList.remove('dark-mode');
    document.body.classList.remove('dark-mode');
})();

// Initialize dark mode UI elements after DOM loads
/*
function initDarkMode() {
    // Sync with localStorage to ensure consistency
    const darkModeSetting = localStorage.getItem('darkMode');
    const shouldBeDark = darkModeSetting === 'true';
    
    // Apply dark mode state
    if (shouldBeDark) {
        document.documentElement.classList.add('dark-mode');
        document.body.classList.add('dark-mode');
    } else {
        document.documentElement.classList.remove('dark-mode');
        document.body.classList.remove('dark-mode');
    }
    
    // Update icon
    const darkModeIcon = document.getElementById('darkModeIcon');
    if (darkModeIcon) {
        darkModeIcon.textContent = shouldBeDark ? 'Light' : 'Dark';
    }
}
*/

// Toggle dark mode
/*
function toggleDarkMode() {
    // Preserve the current active tab before toggling
    const activeTabButton = document.querySelector('.tab-button.active[data-tab]');
    const currentActiveTab = activeTabButton ? activeTabButton.dataset.tab : null;
    
    // Also check if welcome page is visible (no tab selected yet)
    const welcomeDiv = document.getElementById('welcome');
    const isWelcomeVisible = welcomeDiv && !welcomeDiv.classList.contains('hidden');
    
    // Get references to all main content sections to ensure they remain visible
    const mainContent = document.getElementById('main-content');
    const fileUpload = document.getElementById('fileuploaddiv');
    const userInputSection = document.getElementById('userInputSection');
    const predictionDiv = document.getElementById('predictionDiv');
    const documentationSection = document.getElementById('documentation');
    
    // Ensure all currently visible content sections maintain their visibility
    // This prevents the body background from covering content during the toggle
    if (mainContent && !mainContent.classList.contains('hidden')) {
        mainContent.style.display = '';
        mainContent.style.visibility = 'visible';
    }
    if (welcomeDiv && !welcomeDiv.classList.contains('hidden')) {
        welcomeDiv.style.display = '';
        welcomeDiv.style.visibility = 'visible';
    }
    if (fileUpload && !fileUpload.classList.contains('hidden')) {
        fileUpload.style.display = '';
        fileUpload.style.visibility = 'visible';
    }
    if (userInputSection && !userInputSection.classList.contains('hidden')) {
        userInputSection.style.display = '';
        userInputSection.style.visibility = 'visible';
    }
    if (predictionDiv && !predictionDiv.classList.contains('hidden')) {
        predictionDiv.style.display = '';
        predictionDiv.style.visibility = 'visible';
    }
    if (documentationSection && !documentationSection.classList.contains('hidden')) {
        documentationSection.style.display = '';
        documentationSection.style.visibility = 'visible';
    }
    
    // Prevent any visual glitches by ensuring we're not in the middle of other operations
    const darkModeIcon = document.getElementById('darkModeIcon');
    const isDarkMode = document.body.classList.contains('dark-mode');
    const html = document.documentElement;
    const body = document.body;
    
    // Toggle the classes synchronously
    if (isDarkMode) {
        html.classList.remove('dark-mode');
        body.classList.remove('dark-mode');
        localStorage.setItem('darkMode', 'false');
        if (darkModeIcon) {
            darkModeIcon.textContent = 'Dark';
        }
    } else {
        html.classList.add('dark-mode');
        body.classList.add('dark-mode');
        localStorage.setItem('darkMode', 'true');
        if (darkModeIcon) {
            darkModeIcon.textContent = 'Light';
        }
    }
    
    // Force a reflow to apply styles immediately
    void body.offsetHeight;
    
    // Restore the visible content section after toggling
    // Use requestAnimationFrame to ensure DOM is ready but without visible delay
    requestAnimationFrame(() => {
        if (isWelcomeVisible) {
            // If welcome was visible, ensure it stays visible
            if (welcomeDiv) {
                welcomeDiv.classList.remove('hidden');
                welcomeDiv.style.display = '';
                welcomeDiv.style.visibility = 'visible';
            }
            // Hide app tabs if welcome is showing
            if (appTabs) {
                appTabs.classList.add('hidden');
            }
        } else if (currentActiveTab) {
            // Re-show the active tab's content by calling showTab
            // This ensures all the proper sections are shown/hidden correctly
            showTab(currentActiveTab);
        } else {
            // Fallback: If no tab is active, try to determine what should be visible
            // by checking which main section is not hidden
            if (mainContent) {
                // Ensure main-content is visible
                mainContent.style.display = '';
                mainContent.style.visibility = 'visible';
                mainContent.classList.remove('hidden');
            }
        }
        
        // Force another reflow to ensure all styles are applied
        void body.offsetHeight;
    });
}
*/

// Ensure dark mode is properly applied to all elements
/*
function ensureDarkModeApplied() {
    const isDarkMode = localStorage.getItem('darkMode') === 'true';
    const html = document.documentElement;
    const body = document.body;
    
    if (isDarkMode) {
        // Ensure html and body have dark-mode class
        html.classList.add('dark-mode');
        body.classList.add('dark-mode');
    } else {
        // Ensure dark mode is removed
        html.classList.remove('dark-mode');
        body.classList.remove('dark-mode');
    }
    
    // Force a reflow to ensure styles are applied to all elements
    // Use a single reflow to avoid visual glitches
    void body.offsetHeight;
}
*/

// Set up dark mode toggle event listener
/*
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        initDarkMode();
        const darkModeToggle = document.getElementById('darkModeToggle');
        if (darkModeToggle) {
            darkModeToggle.addEventListener('click', toggleDarkMode);
        }
    });
} else {
    // DOM already loaded
    initDarkMode();
    const darkModeToggle = document.getElementById('darkModeToggle');
    if (darkModeToggle) {
        darkModeToggle.addEventListener('click', toggleDarkMode);
    }
}
*/
