/**
 * DiGiTerra front-end logic. Handoff note: All API calls, uploads, progress polling,
 * and result rendering live here. UI structure is in templates/index.html.
 * See HANDOFF.md for repo overview.
 */
const _uploadForm = document.getElementById('uploadForm');
const _corrForm = document.getElementById('corrForm');
const _preprocessform = document.getElementById('preprocessform');
const _file = document.getElementById('file');
const columnSelection = document.getElementById('columnSelection');
const _indicatorsSelect = document.getElementById('indicators');
const predictorsSelect = document.getElementById('predictors');
const _processForm = document.getElementById('processForm');
const _advancedOptimizationForm = document.getElementById('advancedOptimizationForm');
const _errorDiv = document.getElementById('errorDiv');
const _NumericResultDiv = document.getElementById('NumericResultDiv');
const _ClusterResultDiv = document.getElementById('ClusterResultDiv');
const _ClassifierResultDiv = document.getElementById('ClassifierResultDiv');
const fileUpload = document.getElementById('fileuploaddiv');
const _runMatrices = document.getElementById('runMatrices');
const predictionDiv = document.getElementById('predictionDiv');
const _predictionForm = document.getElementById('uploadPredictDf');
const _predictionResultsDiv = document.getElementById('predictionResults');
const _loading = document.getElementById('loading');
const _processButton = document.getElementById('processButton');
const appTabs = document.getElementById('appTabs');
const tabButtons = document.querySelectorAll('.tab-button');
const userInputSection = document.getElementById('userInputSection');
const trainSizeInput = document.getElementById('trainSize');
const testSizeInput = document.getElementById('testSize');
const backToExplorationButton = document.getElementById('backToExploration');
const backToModelPreprocessButton = document.getElementById('backToModelPreprocess');
const backToModelingFromAdvancedButton = document.getElementById('backToModelingFromAdvanced');
const documentationSection = document.getElementById('documentation');
/* eslint-disable-next-line no-unused-vars */
let pywebviewReady = false;
let headerResizeObserver = null;

let uploadedFileName = '';
const API_ROOT = (window.API_ROOT || '').replace(/\/+$/, '');

function withApiRoot(path) {
    if (!path) return API_ROOT || '';
    if (/^https?:\/\//i.test(path) || path.startsWith('//')) return path;
    if (!path.startsWith('/')) return `${API_ROOT}/${path}`;
    return `${API_ROOT}${path}`;
}

function rewritePrefixedAssetUrls(root = document) {
    const anchors = root.querySelectorAll('a[href^="/download/"]');
    anchors.forEach((anchor) => {
        const href = anchor.getAttribute('href');
        if (href) {
            anchor.setAttribute('href', withApiRoot(href));
        }
    });

    const images = root.querySelectorAll('img[src^="/user-visualizations/"]');
    images.forEach((image) => {
        const src = image.getAttribute('src');
        if (src) {
            image.setAttribute('src', withApiRoot(src));
        }
    });
}

const nativeFetch = window.fetch.bind(window);
window.fetch = (resource, init) => {
    if (typeof resource === 'string') {
        resource = withApiRoot(resource);
    } else if (resource instanceof Request) {
        resource = new Request(withApiRoot(resource.url), resource);
    }
    return nativeFetch(resource, init);
};

const NativeEventSource = window.EventSource;
window.EventSource = function eventSourceWithPrefix(url, config) {
    return new NativeEventSource(withApiRoot(url), config);
};
window.EventSource.prototype = NativeEventSource.prototype;

const prefixedAssetObserver = new MutationObserver((mutations) => {
    mutations.forEach((mutation) => {
        mutation.addedNodes.forEach((node) => {
            if (node.nodeType === Node.ELEMENT_NODE) {
                rewritePrefixedAssetUrls(node);
            }
        });
    });
});

document.addEventListener('DOMContentLoaded', () => {
    rewritePrefixedAssetUrls(document);
    prefixedAssetObserver.observe(document.body, { childList: true, subtree: true });
});

// ============================================================================
// Utility Functions
// ============================================================================

const _formatDelta = (trainValue, validationValue, unit = '') => {
    const trainNum = parseFloat(trainValue);
    const validationNum = parseFloat(validationValue);
    if (!Number.isFinite(trainNum) || !Number.isFinite(validationNum)) {
        return '';
    }
    const delta = (trainNum - validationNum).toFixed(3);
    return unit ? `${delta} ${unit}` : delta;
};

// DOM utility functions
const _queryOne = (selector) => document.querySelector(selector);
const _queryAll = (selector) => document.querySelectorAll(selector);
const $id = (id) => document.getElementById(id);

// Element visibility utilities
const showElement = (element) => {
    if (element) element.classList.remove('hidden');
};

const hideElement = (element) => {
    if (element) element.classList.add('hidden');
};

const _toggleElement = (element, show) => {
    if (element) {
        if (show) {
            showElement(element);
        } else {
            hideElement(element);
        }
    }
};

// Keep content offset synced with fixed header height so sections are not hidden
function updateHeaderOffset() {
    const header = document.querySelector('.header');
    if (!header) return;
    const headerHeight = Math.ceil(header.getBoundingClientRect().height);
    document.documentElement.style.setProperty('--header-offset', `${headerHeight + 16}px`);
}

function initHeaderOffsetSync() {
    updateHeaderOffset();
    window.addEventListener('resize', updateHeaderOffset);
    window.addEventListener('orientationchange', updateHeaderOffset);

    const header = document.querySelector('.header');
    if (header && 'ResizeObserver' in window) {
        headerResizeObserver = new ResizeObserver(updateHeaderOffset);
        headerResizeObserver.observe(header);
    }
}

// Firefox-safe styling for modeling mode labels
function updateModelingModeSelection() {
    const radios = document.querySelectorAll('input[type="radio"][name="modelingMode"]');
    radios.forEach((radio) => {
        const label = radio.closest('.modeling-mode-option');
        if (!label) return;
        label.classList.toggle('is-selected', radio.checked);
    });
}

function initModelingModeSelectionSync() {
    updateModelingModeSelection();
    document.addEventListener('change', (event) => {
        const target = event.target;
        if (target && target.matches('input[type="radio"][name="modelingMode"]')) {
            updateModelingModeSelection();
        }
    });
}

// HTML escape utility to prevent XSS
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Error display utility
const _showError = (element, message, useErrorClass = true) => {
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
function _manageFocus(element) {
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
    document.addEventListener('DOMContentLoaded', () => {
        enhanceToggleAccessibility();
        initHeaderOffsetSync();
        initModelingModeSelectionSync();
    });
} else {
    enhanceToggleAccessibility();
    initHeaderOffsetSync();
    initModelingModeSelectionSync();
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
    const _simpleNModels = document.getElementById('simpleNModels');
    // No recursive event dispatch needed - existing listeners handle model changes
    
    // Simple mode - Cluster
    const _simpleClModels = document.getElementById('simpleClModels');
    // No recursive event dispatch needed - existing listeners handle model changes
    
    // Simple mode - Classifier
    const _simpleClassModels = document.getElementById('simpleClassModels');
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

const _clearError = (element) => {
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
    const relativeDownloadPrefix = '/download/';
    const prefixedDownloadPrefix = `${API_ROOT}/download/`;
    if (!href || (!href.startsWith(relativeDownloadPrefix) && !href.startsWith(prefixedDownloadPrefix))) {
        return;
    }
    // Check if pywebview API is available
    try {
        if (window.pywebview?.api?.save_file) {
            event.preventDefault();
            pywebviewReady = true;
            const downloadUrl = new URL(href, window.location.origin);
            const requestedName = downloadUrl.searchParams.get('download_name');
            const sourceFilename = decodeURIComponent(downloadUrl.pathname.replace(/^.*\/download\//, ''));
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

// helpers for getting the column index / letter (reserved for future use)
/* eslint-disable-next-line no-unused-vars */
function getColumnLetter(index) {
        let column = "";
        while (index >= 0) {
            column = String.fromCharCode((index % 26) + 65) + column;
            index = Math.floor(index / 26) - 1;
        }
        return column;
    }
/* eslint-disable-next-line no-unused-vars */
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
        // Re-query in case refs were null at load (e.g. script ran before DOM ready)
        const modelPreprocessSection = document.getElementById('userInputSection');
        const uploadSection = document.getElementById('fileuploaddiv');
        if (modelPreprocessSection) {
            modelPreprocessSection.classList.remove('hidden');
            modelPreprocessSection.style.display = '';
        }
        if (uploadSection) {
            uploadSection.classList.add('hidden');
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
        // Reset Inference UI so user can upload a new dataset when returning to this tab
        resetInferenceUI();
        // Hide redobutton on historic tab
        if (redobutton) {
            redobutton.classList.add('hidden');
        }
    }

    setActiveTab(tabName);
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

// Shared helpers used by upload, modeling, inference (no bundler)
function formatDateTimeForFilename(date = new Date()) {
    const pad = (value) => String(value).padStart(2, '0');
    const year = date.getFullYear();
    const month = pad(date.getMonth() + 1);
    const day = pad(date.getDate());
    const hours = pad(date.getHours());
    const minutes = pad(date.getMinutes());
    const seconds = pad(date.getSeconds());
    return `${year}${month}${day}_${hours}${minutes}${seconds}`;
}

function downloadFile(filename, downloadName = filename) {
    try {
        if (window.pywebview?.api?.save_file) {
            pywebviewReady = true;
            window.pywebview.api.save_file(filename, downloadName)
                .then((success) => { if (success) {} else {} })
                .catch((error) => { console.error('Error saving file via pywebview API:', error); });
            return false;
        }
    } catch (error) {
        console.error('Error accessing pywebview API:', error);
    }
    return true;
}

/* eslint-disable-next-line no-unused-vars */
function showCrossValidationUnavailable() {
    const errDiv = getCachedElement('errorDiv');
    if (errDiv) {
        showError(errDiv, 'Cross-validation results are unavailable because cross-validation was not run.');
        errDiv.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
    return false;
}

/* eslint-disable-next-line no-unused-vars */
function downloadAdditionalInfoTable(tableData, sheetName, timestamp) {
    const ts = timestamp || formatDateTimeForFilename();
    fetch(withApiRoot('/downloadAdditionalInfo'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ table_data: tableData, sheet_name: sheetName }),
    })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                const errDiv = getCachedElement('errorDiv');
                if (errDiv) { showError(errDiv, 'Error generating download: ' + data.error); }
                return;
            }
            const downloadName = `additional_info_${sheetName.toLowerCase().replace(/\s+/g, '_')}_${ts}.xlsx`;
            const href = withApiRoot(`/download/${data.filename}?download_name=${encodeURIComponent(downloadName)}`);
            const link = document.createElement('a');
            link.href = href;
            link.click();
            downloadFile(data.filename, downloadName);
        })
        .catch(error => {
            console.error('Error downloading additional information:', error);
            const errDiv = getCachedElement('errorDiv');
            if (errDiv) { showError(errDiv, 'Error downloading file. Please try again.'); }
        });
}

// Expose refs and helpers for app.js and feature modules (no bundler)
window.uploadForm = document.getElementById('uploadForm');
window.corrForm = document.getElementById('corrForm');
window.preprocessform = document.getElementById('preprocessform');
window.indicatorsSelect = document.getElementById('indicators');
window.processForm = document.getElementById('processForm');
window.advancedOptimizationForm = document.getElementById('advancedOptimizationForm');
window.errorDiv = document.getElementById('errorDiv');
window.NumericResultDiv = document.getElementById('NumericResultDiv');
window.ClusterResultDiv = document.getElementById('ClusterResultDiv');
window.ClassifierResultDiv = document.getElementById('ClassifierResultDiv');
window.runMatrices = document.getElementById('runMatrices');
window.predictionForm = document.getElementById('uploadPredictDf');
window.predictionResultsDiv = document.getElementById('predictionResults');
window.processButton = document.getElementById('processButton');
window.showError = _showError;
window.manageFocus = _manageFocus;
