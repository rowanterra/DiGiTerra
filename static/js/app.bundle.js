// === init.js ===
/**
 * DiGiTerra init: theme and fallback helpers. Load early (e.g. in head) for theme; sets goToModelPreprocessing for body.
 */
(function() {
    'use strict';
    // Light mode (dark mode disabled) - prevent flash
    if (document.documentElement) {
        document.documentElement.classList.remove('dark-mode');
    }
    if (document.body) {
        document.body.classList.remove('dark-mode');
    }
})();

// Fallback so "Continue to Model Preprocessing" works even if main script fails or is cached
window.goToModelPreprocessing = window.goToModelPreprocessing || function() {
    var s = document.getElementById('userInputSection');
    var u = document.getElementById('fileuploaddiv');
    if (s) {
        s.classList.remove('hidden');
        s.style.display = '';
    }
    if (u) {
        u.classList.add('hidden');
    }
    var tabs = document.querySelectorAll('.tab-button[data-tab]');
    if (tabs && tabs.forEach) {
        tabs.forEach(function(b) {
            b.classList.toggle('active', b.dataset.tab === 'model-preprocessing');
        });
    }
    window.scrollTo(0, 0);
};


// === core.js ===
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
// Rows and columns of the uploaded file, as the server counted them. Read by the
// Data panel of the modeling page; written by upload.js.
/* eslint-disable-next-line no-unused-vars */
let uploadedShape = { rows: null, columns: null };
const API_ROOT = (window.API_ROOT || '').replace(/\/+$/, '');

// The server sets a CSRF cookie that script can read; every request the page makes
// echoes it in a header. A cross-site form can cause the cookie to be sent but
// cannot read it, so it cannot set the header. On a desktop build the server does
// not check the header, and sending it costs nothing.
function csrfToken() {
    const match = document.cookie.match(/(?:^|;\s*)digiterra_csrf=([^;]*)/);
    return match ? decodeURIComponent(match[1]) : '';
}

// fetch is wrapped once, so no call site has to remember the header.
(function addCsrfHeaderToFetch() {
    const original = window.fetch;
    window.fetch = function (input, init) {
        const options = init ? { ...init } : {};
        const method = (options.method || (typeof input === 'object' && input && input.method) || 'GET').toUpperCase();
        const url = typeof input === 'string' ? input : (input && input.url) || '';
        const sameOrigin = !/^https?:\/\//i.test(url) || url.startsWith(window.location.origin);
        if (sameOrigin && method !== 'GET' && method !== 'HEAD') {
            const token = csrfToken();
            if (token) {
                const headers = new Headers(options.headers || (typeof input === 'object' && input ? input.headers : undefined));
                headers.set('X-DiGiTerra-CSRF', token);
                options.headers = headers;
            }
        }
        return original.call(this, input, options);
    };
})();

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

/** Exposed for modeling.js / template strings (same file scope is not shared across script tags). */
window.formatDelta = _formatDelta;

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

// HTML escape utility. Escapes the five characters that matter in both text and
// attribute position, so the result is safe inside value="..." as well.
function escapeHtml(text) {
    return String(text ?? '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
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


/**
 * Resolve a hyperparameter control for the modeling mode in use.
 *
 * The Simple and Advanced sections offer the same optional parameters. The
 * Advanced copies have an "advanced" prefix so the document has unique ids;
 * this returns the Advanced node while Advanced or AutoML mode is selected and
 * the plain node otherwise.
 */
function getModeScopedElement(id) {
    const advanced = document.getElementById('advancedMode');
    const automl = document.getElementById('automlMode');
    const useAdvanced = (advanced && advanced.checked) || (automl && automl.checked);
    if (useAdvanced) {
        const scoped = document.getElementById('advanced' + id.charAt(0).toUpperCase() + id.slice(1));
        if (scoped) return scoped;
    }
    return document.getElementById(id);
}
window.getModeScopedElement = getModeScopedElement;

// The Advanced section has its own hyperparameter fields, so nothing is cloned
// at runtime any more. Kept as a no-op because switchModelingMode still calls it.
function ensureAdvancedHyperparametersAvailable() {}

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

/**
 * Unified Advanced tab uses #advancedLoading; legacy #advancedOptimization panel uses #advancedOptimizationLegacyLoading.
 */
function getAdvancedLoadingDiv() {
    const legacyPanel = document.getElementById('advancedOptimization');
    if (legacyPanel && !legacyPanel.classList.contains('hidden')) {
        return document.getElementById('advancedOptimizationLegacyLoading')
            || document.getElementById('advancedLoading');
    }
    return document.getElementById('advancedLoading');
}
window.getAdvancedLoadingDiv = getAdvancedLoadingDiv;

/**
 * Advanced run button: unified Modeling tab (#advancedModelingSection) uses #advancedOptimizationSubmitButton;
 * standalone legacy panel #advancedOptimization uses #advancedOptimizationLegacySubmitButton.
 */
function getAdvancedRunSubmitButton() {
    const legacyPanel = document.getElementById('advancedOptimization');
    if (legacyPanel && !legacyPanel.classList.contains('hidden')) {
        return document.getElementById('advancedOptimizationLegacySubmitButton')
            || document.getElementById('advancedOptimizationSubmitButton');
    }
    return document.getElementById('advancedOptimizationSubmitButton');
}
window.getAdvancedRunSubmitButton = getAdvancedRunSubmitButton;

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
// Column letters: single letters or ranges joined by a hyphen, separated by commas.
// Any other character (a colon, a digit, a space inside a letter) is refused here
// so the server never receives an index computed from the wrong character.
const COLUMN_SPEC = /^[A-Z]+(-[A-Z]+)?(,[A-Z]+(-[A-Z]+)?)*$/;

/* eslint-disable-next-line no-unused-vars */
function isValidColumnSpec(input) {
    return COLUMN_SPEC.test((input || '').toUpperCase().replace(/\s/g, ''));
}

/* eslint-disable-next-line no-unused-vars */
function getColumnIndices(input) {
    const columns = [];
    // An invalid selection yields no columns rather than an index computed from
    // the wrong character; the submit handlers check isValidColumnSpec first.
    if (!COLUMN_SPEC.test(input)) {
        return columns;
    }
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

    if (typeof window.updateDataCleaningScopeForOutputType === 'function') {
        window.updateDataCleaningScopeForOutputType(outputType);
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

// "Continue to Modeling" waits for a task type. The select is #outputType1 (an
// earlier #modelingType never existed on the page, so this ran for nothing and the
// form went to the server empty, to be refused there).
const modelingTypeSelect = document.getElementById('outputType1');
const continueToModelingButton = document.getElementById('continueToModelingButton');
if (modelingTypeSelect && continueToModelingButton) {
    const gate = () => {
        const ready = Boolean(modelingTypeSelect.value);
        continueToModelingButton.disabled = !ready;
        if (ready) continueToModelingButton.removeAttribute('title');
        else continueToModelingButton.setAttribute('title', 'Choose a modeling task type first.');
    };
    gate();
    modelingTypeSelect.addEventListener('change', gate);
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
        showError(errDiv, 'DiGiTerra has no cross-validation results, because the run did not use cross-validation.');
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
                if (errDiv) { showError(errDiv, 'DiGiTerra cannot make the download. ' + data.error); }
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
            if (errDiv) { showError(errDiv, 'The file download failed.'); }
        });
}

// Expose refs and helpers for app.js and feature modules (no bundler)
window.uploadForm = document.getElementById('uploadForm');
window.corrForm = document.getElementById('corrForm');
window.preprocessform = document.getElementById('preprocessform');
window.indicatorsSelect = document.getElementById('indicators');
window.processForm = document.getElementById('processForm');
window.advancedOptimizationForm = document.getElementById('advancedOptimizationForm');
window.advancedOptimizationLegacyForm = document.getElementById('advancedOptimizationLegacyForm');
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


/* An anchor into a closed <details> lands nowhere, because the target has no box.
   Each Contents entry opens its section first, then scrolls to it. */
function initDocContents() {
    const toc = document.querySelector('.doc-toc');
    if (!toc) return;
    toc.addEventListener('click', (e) => {
        const link = e.target.closest('a[href^="#"]');
        if (!link) return;
        const card = document.getElementById(link.getAttribute('href').slice(1));
        if (!card) return;
        e.preventDefault();
        const section = card.querySelector('details.doc-section') || card.closest('details.doc-section');
        if (section) section.open = true;
        card.scrollIntoView({ behavior: 'smooth', block: 'start' });
        // Move the reading position as well as the viewport, so a keyboard or screen
        // reader user arrives where the pointer user does.
        const summary = section ? section.querySelector('summary') : null;
        if (summary) summary.focus({ preventScroll: true });
    });
}

/* The banner's Tutorial entry opens the documentation and goes to that section. */
function initTutorialEntry() {
    const button = document.getElementById('tutorialButton');
    if (!button) return;
    button.addEventListener('click', (e) => {
        e.preventDefault();
        if (typeof showTab === 'function') showTab('documentation');
        const card = document.getElementById('doc-tutorial');
        if (!card) return;
        const section = card.querySelector('details.doc-section');
        if (section) section.open = true;
        window.setTimeout(() => card.scrollIntoView({ behavior: 'smooth', block: 'start' }), 60);
    });
}

/* The settings the tutorial tells a reader to enter, applied once for them.
   A person who uploads their own file gets none of this: the values only make sense
   for the columns penguins.csv has. Every field stays editable. */
const EXAMPLE_PREFILL = {
    outputType1: 'Classifier',
    indicators: 'B-G',
    predictors: 'A',
    imputeStrategy: 'none',
    useTransformer: 'Yes',
    transformerColumn: 'B, G',
    scalingYesNo: 'Yes',
    specificVariableSelect: 'A',
    seedValue: '33',
    corrCols: 'C-F',
    model: 'RF_classifier',
};

/* Set a value and tell the page, so a section that opens on change opens now. */
function setExampleField(id, value) {
    const el = document.getElementById(id);
    if (!el) return;
    if (el.type === 'checkbox') el.checked = Boolean(value);
    else el.value = value;
    el.dispatchEvent(new Event('input', { bubbles: true }));
    el.dispatchEvent(new Event('change', { bubbles: true }));
}

/* Fill the preprocessing form with the tutorial's run.
   The order matters: the select that reveals a section is set before the field
   inside it, or the field is written into a section that is still closed. */
function applyExamplePrefill() {
    setExampleField('outputType1', EXAMPLE_PREFILL.outputType1);
    setExampleField('indicators', EXAMPLE_PREFILL.indicators);
    setExampleField('predictors', EXAMPLE_PREFILL.predictors);
    setExampleField('corrCols', EXAMPLE_PREFILL.corrCols);

    // Remove the rows with a blank: the scope first, then the action.
    setExampleField('missScopeAll', true);
    setExampleField('imputeStrategy', EXAMPLE_PREFILL.imputeStrategy);

    // island and sex are text, so the transformer section has to open first.
    setExampleField('useTransformer', EXAMPLE_PREFILL.useTransformer);
    setExampleField('transformerColumn', EXAMPLE_PREFILL.transformerColumn);

    // Stratify on the species column, which is what the tutorial's step 5 asks for.
    setExampleField('scalingYesNo', EXAMPLE_PREFILL.scalingYesNo);
    setExampleField('specificVariableSelect', EXAMPLE_PREFILL.specificVariableSelect);

    setExampleField('seedValue', EXAMPLE_PREFILL.seedValue);
    prefillExampleModel();
}

/* Choose the tutorial's model once the Modeling view opens, so the example run is
   one click from there. The view is revealed by a class change after preprocessing,
   so the choice waits for that; it is made once, and only into an empty select, so a
   person's own choice is never overwritten. */
function prefillExampleModel() {
    const host = document.getElementById('columnSelection');
    const select = document.getElementById('simpleClassModels');
    if (!host || !select) return;
    const fill = () => {
        if (host.classList.contains('hidden')) return false;
        if (!select.value) setExampleField('simpleClassModels', EXAMPLE_PREFILL.model);
        return true;
    };
    if (fill()) return;
    const watch = new MutationObserver(() => {
        if (fill()) watch.disconnect();
    });
    watch.observe(host, { attributes: true, attributeFilter: ['class'] });
}

/* Load an example through the ordinary upload, so it meets the same parse, the same
   limits and the same data checks a chosen file meets. The file is fetched from the
   server and handed to the existing form rather than a second code path being built. */
function initTryExample() {
    const button = document.getElementById('tryExampleButton');
    if (!button) return;
    button.addEventListener('click', async () => {
        const name = 'penguins.csv';
        const input = document.getElementById('file');
        const form = document.getElementById('uploadForm');
        if (!input || !form) return;
        const original = button.textContent;
        button.disabled = true;
        button.textContent = 'Loading the example...';
        try {
            const response = await fetch(`${API_ROOT}/example-data/${name}`);
            if (!response.ok) throw new Error('status ' + response.status);
            const blob = await response.blob();
            const transfer = new DataTransfer();
            transfer.items.add(new File([blob], name, { type: 'text/csv' }));
            input.files = transfer.files;
            if (typeof showTab === 'function') showTab('upload');
            // The prefill runs on the upload's own response rather than on a timer,
            // so the fields are set once the columns the app read are on the page.
            const fill = (event) => {
                if (!event || !event.detail || event.detail.ok !== false) applyExamplePrefill();
            };
            document.addEventListener('digiterra:upload-complete', fill, { once: true });
            form.requestSubmit();
        } catch (err) {
            console.error('Example not loaded:', err);
            button.disabled = false;
            button.textContent = original;
            const box = document.getElementById('uploadErrorDiv') || document.getElementById('errorDiv');
            if (box && typeof showError === 'function') {
                showError(box, 'DiGiTerra cannot load the example file. Choose a file instead.');
            }
        }
    });
}

/* The example rows for the inference page.
   Shown only for a session that loaded penguins.csv, because the file holds that
   file's six indicator columns and nothing else would accept them. */
function penguinsIsLoaded() {
    return typeof uploadedFileName === 'string' && uploadedFileName === 'penguins.csv';
}

function updateExampleRowsVisibility() {
    const button = document.getElementById('usePenguinRowsButton');
    const note = document.getElementById('usePenguinRowsNote');
    const show = penguinsIsLoaded();
    if (button) button.classList.toggle('hidden', !show);
    if (note) note.classList.toggle('hidden', !show);
}

function initUsePenguinRows() {
    const button = document.getElementById('usePenguinRowsButton');
    if (!button) return;
    updateExampleRowsVisibility();
    // The button appears with the view, and the view is revealed by a class change.
    const host = document.getElementById('predictionDiv');
    if (host) {
        new MutationObserver(updateExampleRowsVisibility).observe(host, {
            attributes: true,
            attributeFilter: ['class'],
        });
    }
    button.addEventListener('click', async () => {
        const input = document.getElementById('predictFile');
        const form = document.getElementById('uploadPredictDf');
        if (!input || !form) return;
        const original = button.textContent;
        button.disabled = true;
        button.textContent = 'Loading the rows...';
        try {
            const response = await fetch(`${API_ROOT}/example-data/penguins_new.csv`);
            if (!response.ok) throw new Error('status ' + response.status);
            const blob = await response.blob();
            const transfer = new DataTransfer();
            transfer.items.add(new File([blob], 'penguins_new.csv', { type: 'text/csv' }));
            input.files = transfer.files;
            window.digiterraExampleRows = true;
            form.requestSubmit();
        } catch (err) {
            console.error('Example rows not loaded:', err);
            button.disabled = false;
            button.textContent = original;
        }
    });
}

document.addEventListener('DOMContentLoaded', () => {
    initDocContents();
    initTutorialEntry();
    initTryExample();
    initUsePenguinRows();
});


// === upload.js ===
/**
 * Upload and Data Exploration (correlation matrices, pairplot). Depends on core.js (refs, helpers).
 */
// Shared across the split files (they run in one script scope).
let corrColsIndices = 'all';
uploadForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const formData = new FormData(uploadForm);
    
    // Validate file is selected
    const fileInput = document.getElementById('file');
    if (!fileInput || !fileInput.files || fileInput.files.length === 0) {
        alert('No file is selected.');
        return;
    }
    
    // Validate file extension
    const fileName = fileInput.files[0].name.toLowerCase();
    const accepted = ['.csv', '.xlsx', '.xlsm', '.parquet'];
    if (!accepted.some(ext => fileName.endsWith(ext))) {
        alert('Accepted file types are .csv, .xlsx, .xlsm, and .parquet.');
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
        // The shape the server measured, kept for the Data panel of the modeling page.
        uploadedShape = { rows: data.rows || null, columns: data.numcols || null };
        // The rail names the column behind a letter, so it needs the header row.
        window.uploadedColumnNames = Array.isArray(data.columns) ? data.columns : null;

        // show the divs for preprocessing and selecting targets, indicators, output type, etc
        let columnDiv = document.getElementById('columnsection');
        columnDiv.classList.remove('hidden');

        columnSelection.dataset.ready = 'false';
        columnSelection.style.display = 'none';
        setActiveTab('processing');
        // Anything that has to run once the columns are on the page listens for this.
        document.dispatchEvent(new CustomEvent('digiterra:upload-complete', { detail: { ok: true } }));

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
                        ${data.warnings.map(w => `<li>${escapeHtml(w)}</li>`).join('')}
                    </ul>
                </div>
            `;
        }
        
        columnList.innerHTML = `
            <table class="column-summary-table">
                <tbody>
                    <tr>
                        <th>File name</th>
                        <td>${escapeHtml(shortFileName)}</td>
                    </tr>
                    <tr>
                        <th>Total columns</th>
                        <td>${data.numcols}</td>
                    </tr>
                    <tr>
                        <th>First column</th>
                        <td>A: ${escapeHtml(data.firstcol)}</td>
                    </tr>
                    <tr>
                        <th>Last column</th>
                        <td>${escapeHtml(lastColVal)}: ${escapeHtml(data.lastcol)}</td>
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
        alert('The upload failed. Open the browser console for the details.');
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
        zeroHandle: document.getElementById('exploreZeroHandle')?.value || 'drop',
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
            throw new Error(data.error || 'The upload failed.');
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
                                    ${headers.map((header) => `<td>${escapeHtml(row[header.key] ?? '')}</td>`).join('')}
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
            .map((column) => `<option value="${escapeHtml(column)}">${escapeHtml(column)}</option>`)
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
                            zeroHandle: document.getElementById('exploreZeroHandle')?.value || 'drop',
                        }),
                    });
                    
                    if (!response.ok) {
                        const errorData = await response.json().catch(() => ({ error: 'Unknown error' }));
                        console.error('Pairplot update failed:', errorData.error || response.statusText);
                        showError(errorDiv, `Pairplot update failed: ${errorData.error || response.statusText}`);
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
                        const _oldSrc = currentImg.src;
                        
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
                                    showError(errorDiv, 'The pairplot image did not load.');
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
                        showError(errorDiv, 'The server returned no pairplot image.');
                        pairplotImageEl.style.opacity = '1';
                        pairplotImageEl.dataset.updating = 'false';
                    }
                } catch (error) {
                    console.error('Pairplot update error:', error);
                    showError(errorDiv, `Pairplot update failed: ${error.message || 'request error'}`);
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
        showError(errorDiv, error.message || 'The request failed. Open the browser console for the details.');
    }

});



// === preprocess.js ===
/**
 * Preprocess tab: handlePreprocessFormSubmit, preprocess-form and exploration UI. Depends on core.js.
 */
// Shared across the split files (they run in one script scope).
let stratifyColumnNumber = '';
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
        showError(stratErrorDiv, 'Enter a stratify variable. Quantiles and bins both need one.', false);
        // Set aria-invalid on the stratify field
        const stratifyInput = document.getElementById('specificVariableSelect');
        if (stratifyInput) {
            stratifyInput.setAttribute('aria-invalid', 'true');
            manageFocus(stratifyInput);
        }
    }

    else if (quantileBins =='quantiles' && quantiles==''){
        e.preventDefault();
        showError(stratErrorDiv, 'Enter the number of quantiles.', false);
        const quantilesInput = document.getElementById('quantiles');
        if (quantilesInput) {
            quantilesInput.setAttribute('aria-invalid', 'true');
            manageFocus(quantilesInput);
        }
    }

    else if (quantileBins =='Bins' && (bins=='' || binsLabel=='')){
        e.preventDefault();
        showError(stratErrorDiv, 'Enter the bin thresholds and the bin labels.', false);
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

    // A stratify column with neither bins nor quantiles is the categorical case: the
    // split holds the proportions of the column's own values. Only a continuous
    // column needs to be cut into groups first. The server decides whether the
    // column can carry a split and says so if it cannot.
    else if (!predictors && outputType!=='Cluster'){
        e.preventDefault();
        showError(stratErrorDiv, 'Enter a target column. Classification and regression both need one.', false);
        const predictorsInput = document.getElementById('predictors');
        if (predictorsInput) {
            predictorsInput.setAttribute('aria-invalid', 'true');
            manageFocus(predictorsInput);
        }
    }
    else if (stratifyColumn.trim() !== '' && (quantileBins === 'quantiles' || quantileBins === 'Bins') && predictors.trim() !== '' && outputType !== 'Cluster' && outputType !== 'Classifier') {
        const pCols = getColumnIndices(predictors.toUpperCase().replace(/\s/g, ''));
        const stratNum = columnToIndex(stratifyColumn.trim().toUpperCase());
        if (pCols.indexOf(stratNum) !== -1) {
            e.preventDefault();
            showError(stratErrorDiv, 'Stratifying by the target column leaks target information into the train/test split, so a different stratify column is required.', false);
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
        if (typeof window.commitDataCleaningFormToHiddenSelects === 'function') {
            window.commitDataCleaningFormToHiddenSelects();
        }
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
        const badSpec = [indicators, predictors].find(v => v && !isValidColumnSpec(v));
        if (badSpec !== undefined) {
            const target = stratErrorDiv || errorDiv;
            if (target) showError(target, `Column letters are written like A-D or A-D,F,H. "${badSpec}" is not a valid column selection.`, false);
            return;
        }
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
                if (stratErrorDiv) showError(stratErrorDiv, data.error || 'DiGiTerra cannot preprocess the selected columns. Check the column letters, then try again.', false);
                else if (errorDiv) showError(errorDiv, data.error || 'DiGiTerra cannot preprocess the file.', false);
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

            const preds = data.predictors;
            let predictorsColNameString = Array.isArray(preds) ? preds.join(',').substring(0, 10) : (preds || '').substring(0, 10);
            if (predictorsColNameString.length >= 10) {
                predictorsColNameString += '...';
            }

            const IndicatorsPredictorsSection = document.getElementById('modelingHeaderActions');
            const outputTypeLabel = outputType === 'Numeric' ? 'Regression' : outputType;

            if (!IndicatorsPredictorsSection) {
                if (errorDiv) showError(errorDiv, 'DiGiTerra cannot find the modeling header section. Reload the page.');
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
                    <button class="success-button" onclick="predictionPage()">Continue to Inference on New Data</button>
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
                    <button class="success-button" onclick="predictionPage()">Continue to Inference on New Data</button>
                </div>`;
            }
        } catch (error) {
            console.error('Preprocess submit error:', error);
            const msg = error && error.message ? error.message : 'The request failed. Open the browser console for the details.';
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

    function refreshMissingActionUi(missingColSelection) {
        const imputeStrat = document.getElementById('imputeStrategy');
        if (imputeStrat) {
            imputeStrat.disabled = missingColSelection === 'none';
        }
    }

    function syncMissingCheckboxesFromSelect() {
        const sel = document.getElementById('dropMissing');
        const im = document.getElementById('missScopeInd');
        const tgt = document.getElementById('missScopeTgt');
        const all = document.getElementById('missScopeAll');
        if (!sel || !im || !tgt || !all) return;
        const v = sel.value;
        all.checked = v === 'all';
        im.checked = v === 'indicator' || v === 'indicatorAndTarget';
        tgt.checked = v === 'target' || v === 'indicatorAndTarget';
    }

    function syncMissingSelectFromCheckboxes() {
        const sel = document.getElementById('dropMissing');
        const im = document.getElementById('missScopeInd');
        const tgt = document.getElementById('missScopeTgt');
        const all = document.getElementById('missScopeAll');
        if (!sel || !im || !tgt || !all) return;
        let v = 'none';
        if (all.checked) v = 'all';
        else if (im.checked && tgt.checked) v = 'indicatorAndTarget';
        else if (im.checked) v = 'indicator';
        else if (tgt.checked) v = 'target';
        const prev = sel.value;
        sel.value = v;
        if (prev !== v) {
            sel.dispatchEvent(new Event('change', { bubbles: true }));
        } else {
            refreshMissingActionUi(v);
        }
    }

    function syncZeroCheckboxesFromSelect() {
        const sel = document.getElementById('drop0');
        const im = document.getElementById('zeroScopeInd');
        const tgt = document.getElementById('zeroScopeTgt');
        const all = document.getElementById('zeroScopeAll');
        if (!sel || !im || !tgt || !all) return;
        const v = sel.value;
        all.checked = v === 'all';
        im.checked = v === 'indicator' || v === 'indicatorAndTarget';
        tgt.checked = v === 'target' || v === 'indicatorAndTarget';
    }

    function syncZeroSelectFromCheckboxes() {
        const sel = document.getElementById('drop0');
        const zh = document.getElementById('zeroHandle');
        const im = document.getElementById('zeroScopeInd');
        const tgt = document.getElementById('zeroScopeTgt');
        const all = document.getElementById('zeroScopeAll');
        if (!sel || !im || !tgt || !all) return;
        if (zh && zh.value === 'none') {
            sel.value = 'none';
            return;
        }
        let v = 'none';
        if (all.checked) v = 'all';
        else if (im.checked && tgt.checked) v = 'indicatorAndTarget';
        else if (im.checked) v = 'indicator';
        else if (tgt.checked) v = 'target';
        if (zh && zh.value !== 'none' && v === 'none') {
            if (im && !im.disabled) im.checked = true;
            v = 'indicator';
        }
        sel.value = v;
    }

    function setZeroScopeInputsDisabled(disabled) {
        ['zeroScopeInd', 'zeroScopeTgt', 'zeroScopeAll'].forEach((id) => {
            const el = document.getElementById(id);
            if (el) el.disabled = disabled;
        });
    }

    function refreshZeroHandleUi() {
        const zh = document.getElementById('zeroHandle');
        const drop0 = document.getElementById('drop0');
        if (!zh || !drop0) return;
        if (zh.value === 'none') {
            setZeroScopeInputsDisabled(true);
            drop0.value = 'none';
            syncZeroCheckboxesFromSelect();
        } else {
            setZeroScopeInputsDisabled(false);
            if (drop0.value === 'none') {
                drop0.value = 'indicator';
                syncZeroCheckboxesFromSelect();
            }
            syncZeroSelectFromCheckboxes();
        }
    }

    function updateDataCleaningScopeForOutputType(outputType) {
        const missTgt = document.getElementById('missScopeTgt');
        const zeroTgt = document.getElementById('zeroScopeTgt');
        const missLbl = document.getElementById('missScopeTgtLabel');
        const zeroLbl = document.getElementById('zeroScopeTgtLabel');
        const isCluster = outputType === 'Cluster';
        if (missTgt) {
            if (isCluster) missTgt.checked = false;
            missTgt.disabled = isCluster;
        }
        if (zeroTgt) {
            if (isCluster) zeroTgt.checked = false;
            zeroTgt.disabled = isCluster;
        }
        if (missLbl) missLbl.style.opacity = isCluster ? '0.5' : '';
        if (zeroLbl) zeroLbl.style.opacity = isCluster ? '0.5' : '';
        syncMissingSelectFromCheckboxes();
        syncZeroSelectFromCheckboxes();
    }
    window.updateDataCleaningScopeForOutputType = updateDataCleaningScopeForOutputType;

    window.commitDataCleaningFormToHiddenSelects = function commitDataCleaningFormToHiddenSelects() {
        syncMissingSelectFromCheckboxes();
        syncZeroSelectFromCheckboxes();
    };

    function initDataCleaningScopeUi() {
        const dm = document.getElementById('dropMissing');
        if (!dm) return;
        syncMissingCheckboxesFromSelect();
        syncZeroCheckboxesFromSelect();
        refreshZeroHandleUi();
        refreshMissingActionUi(dm.value);
        const out = document.getElementById('outputType1');
        if (out && out.value) updateDataCleaningScopeForOutputType(out.value);
    }

    // Handling displaying the 'how to replace missing values' user input
    const dropMissingEl = document.getElementById('dropMissing');
    if (dropMissingEl) {
        dropMissingEl.addEventListener('change', function () {
            refreshMissingActionUi(this.value);
        });
    }

    const missScopeInd = document.getElementById('missScopeInd');
    const missScopeTgt = document.getElementById('missScopeTgt');
    const missScopeAll = document.getElementById('missScopeAll');
    if (missScopeAll) {
        missScopeAll.addEventListener('change', () => {
            if (missScopeAll.checked) {
                if (missScopeInd) missScopeInd.checked = false;
                if (missScopeTgt) missScopeTgt.checked = false;
            }
            syncMissingSelectFromCheckboxes();
        });
    }
    if (missScopeInd) {
        missScopeInd.addEventListener('change', () => {
            if (missScopeInd.checked && missScopeAll) missScopeAll.checked = false;
            syncMissingSelectFromCheckboxes();
        });
    }
    if (missScopeTgt) {
        missScopeTgt.addEventListener('change', () => {
            if (missScopeTgt.checked && missScopeAll) missScopeAll.checked = false;
            syncMissingSelectFromCheckboxes();
        });
    }

    const zeroScopeInd = document.getElementById('zeroScopeInd');
    const zeroScopeTgt = document.getElementById('zeroScopeTgt');
    const zeroScopeAll = document.getElementById('zeroScopeAll');
    if (zeroScopeAll) {
        zeroScopeAll.addEventListener('change', () => {
            if (zeroScopeAll.checked) {
                if (zeroScopeInd) zeroScopeInd.checked = false;
                if (zeroScopeTgt) zeroScopeTgt.checked = false;
            }
            syncZeroSelectFromCheckboxes();
        });
    }
    if (zeroScopeInd) {
        zeroScopeInd.addEventListener('change', () => {
            if (zeroScopeInd.checked && zeroScopeAll) zeroScopeAll.checked = false;
            syncZeroSelectFromCheckboxes();
        });
    }
    if (zeroScopeTgt) {
        zeroScopeTgt.addEventListener('change', () => {
            if (zeroScopeTgt.checked && zeroScopeAll) zeroScopeAll.checked = false;
            syncZeroSelectFromCheckboxes();
        });
    }

    const zeroHandleEl = document.getElementById('zeroHandle');
    if (zeroHandleEl) {
        zeroHandleEl.addEventListener('change', refreshZeroHandleUi);
    }

    initDataCleaningScopeUi();

    function refreshExploreCleaningUi() {
        const missSel = document.getElementById('exploreDropMissing');
        const imputeStrat = document.getElementById('exploreImputeStrategy');
        if (imputeStrat && missSel) {
            imputeStrat.disabled = missSel.value === 'none';
        }
        const z0 = document.getElementById('exploreDrop0');
        const zAct = document.getElementById('exploreZeroActionDiv');
        const zHandle = document.getElementById('exploreZeroHandle');
        if (zAct && z0) {
            if (z0.value === 'none') {
                zAct.classList.add('hidden');
                if (zHandle) zHandle.disabled = true;
            } else {
                zAct.classList.remove('hidden');
                if (zHandle) zHandle.disabled = false;
            }
        }
    }

    const exploreDropMissingEl = document.getElementById('exploreDropMissing');
    if (exploreDropMissingEl) {
        exploreDropMissingEl.addEventListener('change', refreshExploreCleaningUi);
    }
    const exploreDrop0El = document.getElementById('exploreDrop0');
    if (exploreDrop0El) {
        exploreDrop0El.addEventListener('change', refreshExploreCleaningUi);
    }
    refreshExploreCleaningUi();

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
            const outputTypeEl = document.getElementById('outputType1');
            const isCluster = outputTypeEl && outputTypeEl.value === 'Cluster';
            if (!indicatorsSelect || !indicatorsSelect.value.trim()) {
                alert('Enter the indicator columns first, for example A-D.');
                return;
            }
            if (!isCluster && (!predictorsSelect || !predictorsSelect.value.trim())) {
                alert('Enter the indicator columns and the target columns first, for example A-D and E.');
                return;
            }
            const indicatorIndices = getColumnIndices(indicatorsSelect.value.toUpperCase().replace(/\s/g, ""));
            const predictorIndices = isCluster || !predictorsSelect
                ? []
                : getColumnIndices(predictorsSelect.value.toUpperCase().replace(/\s/g, ""));
            if (indicatorIndices.length === 0) {
                alert('DiGiTerra cannot read the indicator columns. Use the form "A-D" or "A,B,C".');
                return;
            }
            if (!isCluster && predictorIndices.length === 0) {
                alert('DiGiTerra cannot read the target columns. Use the form "A-D" or "A,B,C".');
                return;
            }
            try {
                autoDetectNanZerosBtn.disabled = true;
                autoDetectNanZerosBtn.textContent = 'Detecting…';
                const response = await fetch('/auto-detect-nan-zeros', {
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
                dataCleaningAutodetectMessage.textContent = 'Autodetect failed. Open the browser console for the details.';
                if (dataCleaningMissingWrapper) dataCleaningMissingWrapper.classList.remove('autodetect-greyed');
                if (dataCleaningZerosWrapper) dataCleaningZerosWrapper.classList.remove('autodetect-greyed');
                if (dropMissingSelect) dropMissingSelect.disabled = false;
                if (drop0Select) drop0Select.disabled = false;
            } finally {
                autoDetectNanZerosBtn.disabled = false;
                autoDetectNanZerosBtn.textContent = 'Run autodetect';
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
                alert('Enter the indicator columns first, for example A-D.');
                return;
            }
            const indicatorIndices = getColumnIndices(indicatorsSelect.value.toUpperCase().replace(/\s/g, ""));
            if (indicatorIndices.length === 0) {
                alert('DiGiTerra cannot read the indicator columns. Use the form "A-D" or "A,B,C".');
                return;
            }
            try {
                autoDetectTransformersTopBtn.disabled = true;
                autoDetectTransformersTopBtn.textContent = 'Detecting...';
                const response = await fetch('/auto-detect-transformers', {
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
                alert('Enter the indicator columns first, for example A-D.');
                return;
            }
            const indicatorIndices = getColumnIndices(indicatorsInput.value.toUpperCase().replace(/\s/g, ""));
            if (indicatorIndices.length === 0) {
                alert('DiGiTerra cannot read the indicator columns. Use the form "A-D" or "A,B,C".');
                return;
            }
            try {
                autoDetectTransformersBtn.disabled = true;
                autoDetectTransformersBtn.textContent = 'Detecting...';
                const response = await fetch('/auto-detect-transformers', {
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
                if (transformerAutodetectMessage) transformerAutodetectMessage.textContent = 'DiGiTerra cannot detect the categorical columns.';
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


/* ==========================================================================
   Model Preprocessing as one column.

   The page asked a person to press two Detect buttons and then read a grid of
   controls for findings that had not been made yet. It now runs both detections
   itself, on arrival and whenever the columns change, and states what it found as
   a sentence with the action already chosen. The controls behind each sentence are
   the ones that were always there: they are moved, not rebuilt, so every id and
   every handler still works, and a "change" link reveals them.
   ========================================================================== */

/* The blocks each sentence reveals, and the settings that fold away. */
const CLEANING_BLOCKS = ['dataCleaningMissingWrapper', 'imputeDiv', 'dataCleaningZerosWrapper'];
const TRANSFORM_BLOCKS = ['transformerOptionWrapper', 'transformerYes'];
const MORE_SETTINGS_BLOCKS = ['scalingQuestion', 'scalingYes', 'preprocessScaler', 'preprocessSeed', 'preprocessSplit'];

function preprocessColumnsChosen() {
    const indicators = (document.getElementById('indicators') || {}).value || '';
    const target = (document.getElementById('predictors') || {}).value || '';
    const isCluster = (document.getElementById('outputType1') || {}).value === 'Cluster';
    return Boolean(indicators.trim()) && (isCluster || Boolean(target.trim()));
}

/** Build the findings block and the More settings disclosure, once. */
function buildPreprocessLayout() {
    const form = document.getElementById('preprocessform');
    if (!form || document.getElementById('preprocessFindings')) return;

    const findings = document.createElement('div');
    findings.id = 'preprocessFindings';
    findings.className = 'preprocess-findings';
    findings.innerHTML = '<p class="narrative-empty">Enter the indicator and target columns.</p>';

    const more = document.createElement('details');
    more.id = 'preprocessMoreSettings';
    more.className = 'preprocess-more';
    more.innerHTML = '<summary class="doc-summary"><span class="doc-chevron" aria-hidden="true"></span><h3>More settings</h3></summary>';
    const moreBody = document.createElement('div');
    moreBody.className = 'preprocess-more-body';
    more.appendChild(moreBody);

    // The cleaning and transform controls hide behind their sentences.
    const hidden = document.createElement('div');
    hidden.id = 'preprocessControls';
    hidden.className = 'preprocess-controls';

    const firstCard = form.querySelector('.preprocess-card');
    if (firstCard) firstCard.insertAdjacentElement('afterend', findings);
    else form.insertBefore(findings, form.firstChild);
    findings.insertAdjacentElement('afterend', hidden);
    hidden.insertAdjacentElement('afterend', more);

    CLEANING_BLOCKS.concat(TRANSFORM_BLOCKS).forEach((id) => {
        const el = document.getElementById(id);
        if (el) {
            el.classList.add('hidden');
            hidden.appendChild(el);
        }
    });
    MORE_SETTINGS_BLOCKS.forEach((id) => {
        const el = document.getElementById(id);
        if (el) moreBody.appendChild(el);
    });
}

/** Reveal the controls a sentence stands for. */
function revealPreprocessBlocks(ids) {
    ids.forEach((id) => {
        const el = document.getElementById(id);
        if (el) el.classList.remove('hidden');
    });
}

function preprocessSentence(text, blocks) {
    const line = document.createElement('p');
    line.className = 'preprocess-finding';
    line.appendChild(document.createTextNode(text + ' '));
    if (blocks && blocks.length) {
        const link = document.createElement('button');
        link.type = 'button';
        link.className = 'preprocess-change';
        link.textContent = 'change';
        link.addEventListener('click', () => {
            revealPreprocessBlocks(blocks);
            link.remove();
        });
        line.appendChild(link);
    }
    return line;
}

/** Run both detections and say what they found. */
async function runPreprocessDetection() {
    const host = document.getElementById('preprocessFindings');
    if (!host) return;
    if (!preprocessColumnsChosen()) {
        host.innerHTML = '<p class="narrative-empty">Enter the indicator and target columns.</p>';
        return;
    }
    const isCluster = (document.getElementById('outputType1') || {}).value === 'Cluster';
    const indicators = getColumnIndices(document.getElementById('indicators').value.toUpperCase().replace(/\s/g, ''));
    const target = isCluster ? [] : getColumnIndices(document.getElementById('predictors').value.toUpperCase().replace(/\s/g, ''));
    if (!indicators.length) {
        host.innerHTML = '<p class="narrative-empty">DiGiTerra cannot read the indicator columns. Use the form "A-D" or "A,B,C".</p>';
        return;
    }
    const body = JSON.stringify({ indicators: indicators, predictors: target });
    const post = (path) => fetch(`${API_ROOT}${path}`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: body,
    }).then((r) => (r.ok ? r.json() : null)).catch(() => null);

    const [clean, text] = await Promise.all([post('/auto-detect-nan-zeros'), post('/auto-detect-transformers')]);
    lastNumericColumns = clean && Array.isArray(clean.numeric_columns) ? clean.numeric_columns : [];
    host.innerHTML = '';

    let found = false;
    if (clean && clean.blank_rows) {
        found = true;
        const rows = clean.blank_rows === 1 ? '1 row has a blank.' : `${clean.blank_rows} rows have a blank.`;
        host.appendChild(preprocessSentence(`${rows} DiGiTerra will remove them.`, CLEANING_BLOCKS));
        setPreprocessDefault('missScopeAll', true);
        setPreprocessDefault('imputeStrategy', 'none');
    }
    if (clean && clean.zero_columns && clean.zero_columns.length) {
        found = true;
        const names = clean.zero_columns.join(' and ');
        host.appendChild(preprocessSentence(`${names} hold a zero. DiGiTerra will keep the zeros.`, CLEANING_BLOCKS));
    }
    if (text && Array.isArray(text.transformer_indices) && text.transformer_indices.length) {
        found = true;
        const names = window.uploadedColumnNames || [];
        const labels = text.transformer_indices.map((i) => names[i] || indexToColumn(i)).join(' and ');
        const verb = text.transformer_indices.length === 1 ? 'is text' : 'are text';
        host.appendChild(preprocessSentence(`${labels} ${verb}. DiGiTerra will one-hot encode them.`, TRANSFORM_BLOCKS));
        setPreprocessDefault('useTransformer', 'Yes');
        setPreprocessDefault('transformerColumn', text.transformer_indices.map(indexToColumn).join(', '));
    }
    if (!found) {
        host.appendChild(preprocessSentence('No blanks or zeros in the selected columns.', CLEANING_BLOCKS));
    }
    // Setting a default fires the change the old handlers listen for, and one of them
    // opens the transformer block. The sentence is what opens it, so it closes again.
    CLEANING_BLOCKS.concat(TRANSFORM_BLOCKS).forEach((id) => {
        const el = document.getElementById(id);
        if (el) el.classList.add('hidden');
    });
    applyStratifyDefault();
}

function setPreprocessDefault(id, value) {
    const el = document.getElementById(id);
    if (!el) return;
    if (el.type === 'checkbox') el.checked = Boolean(value);
    else el.value = value;
    el.dispatchEvent(new Event('change', { bubbles: true }));
}

function indexToColumn(index) {
    let n = Number(index) + 1;
    let out = '';
    while (n > 0) {
        const rem = (n - 1) % 26;
        out = String.fromCharCode(65 + rem) + out;
        n = Math.floor((n - 1) / 26);
    }
    return out;
}

/* Stratify defaults to Yes on the target for classification and No for regression.
   Quantiles and bins cut a continuous column into groups, so that control belongs to
   a numeric stratify column and is not drawn for a categorical one. */
function applyStratifyDefault() {
    const task = (document.getElementById('outputType1') || {}).value;
    const toggle = document.getElementById('scalingYesNo');
    const column = document.getElementById('specificVariableSelect');
    if (!toggle) return;
    if (task === 'Classifier') {
        setPreprocessDefault('scalingYesNo', 'Yes');
        if (column && !column.value.trim()) {
            setPreprocessDefault('specificVariableSelect', (document.getElementById('predictors') || {}).value || '');
        }
    } else if (task === 'Numeric') {
        setPreprocessDefault('scalingYesNo', 'No');
    }
    updateQuantileBinVisibility();
}

/* The numeric columns of the loaded file, as the last detection reported them. The
   transformer list will not do: it names the text columns among the indicators and
   leaves the target out, so a text target would read as numeric. */
let lastNumericColumns = [];

function stratifyColumnIsNumeric() {
    const column = document.getElementById('specificVariableSelect');
    if (!column || !column.value.trim() || typeof columnToIndex !== 'function') return false;
    const first = column.value.trim().toUpperCase().match(/^[A-Z]+/);
    if (!first) return false;
    return lastNumericColumns.includes(columnToIndex(first[0]));
}

function updateQuantileBinVisibility() {
    const wrapper = document.getElementById('quantileBins');
    if (!wrapper) return;
    const row = wrapper.closest('.scaling-container') || wrapper.parentElement;
    const numeric = stratifyColumnIsNumeric();
    if (row) row.classList.toggle('hidden', !numeric);
    if (!numeric) setPreprocessDefault('quantileBins', 'None');
}

function initPreprocessPage() {
    const host = document.getElementById('userInputSection');
    if (!host) return;
    buildPreprocessLayout();
    let ran = false;
    const refresh = () => {
        if (host.classList.contains('hidden')) return;
        if (!ran) {
            ran = true;
            buildPreprocessLayout();
        }
        runPreprocessDetection();
    };
    refresh();
    new MutationObserver(refresh).observe(host, { attributes: true, attributeFilter: ['class'] });
    ['indicators', 'predictors', 'outputType1'].forEach((id) => {
        const el = document.getElementById(id);
        if (el) el.addEventListener('change', runPreprocessDetection);
    });
    ['specificVariableSelect', 'scalingYesNo'].forEach((id) => {
        const el = document.getElementById(id);
        if (el) el.addEventListener('change', updateQuantileBinVisibility);
    });
}

document.addEventListener('DOMContentLoaded', initPreprocessPage);


// === modeling.js ===
/**
 * Modeling tab: run model, process results, hyperparameters. Depends on core.js, formatDateTimeForFilename, downloadFile, etc.
 */
// Shared across the split files (they run in one script scope).
let model1 = null;
let stratifyBool = false;
let quantileBinDict = {};
processForm.addEventListener('submit', async (e) => {
    e.preventDefault();

    if (typeof window.commitDataCleaningFormToHiddenSelects === 'function') {
        window.commitDataCleaningFormToHiddenSelects();
    }

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
        runButton = getAdvancedRunSubmitButton();
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
    const zeroHandle = getCachedElement('zeroHandle')?.value || 'none';
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
        showError(NumericResultDiv, 'At least one predictor column and one indicator column are required.');
        if (runButton) runButton.disabled = false;
        if (stopButton) stopButton.style.display = 'none';
        return;
    }

    // Simple / Advanced: require an explicit model (placeholder "-- Select an option --" sends empty value)
    if (currentMode === 'simple' || currentMode === 'advanced') {
        let modelPick = '';
        if (selectedOutputType === 'Numeric') {
            modelPick = (currentMode === 'simple' ? document.getElementById('simpleNModels') : document.getElementById('advancedNModels'))?.value || '';
        } else if (selectedOutputType === 'Classifier') {
            modelPick = (currentMode === 'simple' ? document.getElementById('simpleClassModels') : document.getElementById('advancedClassModels'))?.value || '';
        } else if (selectedOutputType === 'Cluster') {
            modelPick = (currentMode === 'simple' ? document.getElementById('simpleClModels') : document.getElementById('advancedClModels'))?.value || '';
        }
        if (!String(modelPick).trim()) {
            const errDiv = getCachedElement('errorDiv');
            if (errDiv) showError(errDiv, 'A model must be selected from the dropdown before the run.');
            else showError(NumericResultDiv, 'A model must be selected from the dropdown before the run.');
            if (runButton) runButton.disabled = false;
            if (stopButton) stopButton.style.display = 'none';
            return;
        }
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
        loadingDiv = getAdvancedLoadingDiv();
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

    // Classification stratifies on the class labels, which is what train_test_split's
    // own stratify argument takes, so naming the target there is correct. The guard is
    // for regression, where the target is cut into bins that would carry it into the
    // split. The route and the training path make the same exception.
    if (
        stratifyBool &&
        selectedOutputType !== 'Cluster' &&
        selectedOutputType !== 'Classifier' &&
        predictorCols.indexOf(stratifyColumnNumber) !== -1
    ) {
        showError(NumericResultDiv, 'Stratifying by the target column leaks target information into the train/test split, so a different stratify column is required.', false);
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
        // A blank model is a request for AutoML to compare its candidates by
        // cross-validation and train the one that scores best; the server does
        // that when models is blank and taskType names the task.
        if (selectedOutputType === 'Numeric'){
            model1 = document.getElementById('automlNModels');
            selectedModel1 = model1 ? model1.value : ''
        }
        else if (selectedOutputType === 'Classifier'){
            model1 = document.getElementById('automlClassModels');
            selectedModel1 = model1 ? model1.value : ''
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
            let RidgeFitIntersept = getModeScopedElement('RidgeFitIntersept').value
            let RidgeNormalize = getModeScopedElement('RidgeNormalize').value
            let RidgeCopyX = getModeScopedElement('RidgeCopyX').value
            let RidgePositive = getModeScopedElement('RidgePositive').value
            let RidgeMaxIter = getModeScopedElement('RidgeMaxIter').value
            let RidgeTol = getModeScopedElement('RidgeTol').value
            const solver = getModeScopedElement('RidgeSolver');
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
            let LassoFitIntersept = getModeScopedElement('LassoFitIntersept').value
            let LassoPrecompute = getModeScopedElement('LassoPrecompute').value
            let LassoCopyX = getModeScopedElement('LassoCopyX').value
            let LassoWarmStart = getModeScopedElement('LassoWarmStart').value
            let LassoPositive = getModeScopedElement('LassoPositive').value
            let LassoTol = getModeScopedElement('LassoTol').value
            //let LassoRandomState = document.getElementById('LassoRandomState').value
            let LassoSelection = getModeScopedElement('LassoSelection').value
            const max_iter = getModeScopedElement('LassoMax_iter');
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
            let LogisticDual = getModeScopedElement('LogisticDual').value
            let LogisticFitIntercept = getModeScopedElement('LogisticFitIntercept').value
            let LogisticSolver = getModeScopedElement('LogisticSolver').value
            let LogisticMultiClass = getModeScopedElement('LogisticMultiClass').value
            let LogisticWarmStart = getModeScopedElement('LogisticWarmStart').value
            let LogisticTol = getModeScopedElement('LogisticTol').value
            let Logisticintercept_scaling = getModeScopedElement('Logisticintercept_scaling').value
            let LogisticClassWeight = getModeScopedElement('LogisticClassWeight').value
            //let LogisticRandomState = document.getElementById('LogisticRandomState').value
            let LogisticMaxIterations = getModeScopedElement('LogisticMaxIterations').value
            let LogisticVerbose = getModeScopedElement('LogisticVerbose').value
            let LogisticNJobs = getModeScopedElement('LogisticNJobs').value
            let Logisticl1Ratio = getModeScopedElement('Logisticl1Ratio').value

            const CLogistic = getModeScopedElement('CLogistic');
            const penatly = getModeScopedElement('penalty');
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
            let SVMshrinking = getModeScopedElement('SVMshrinking').value
            let SVMprobability = getModeScopedElement('SVMprobability').value
            let SVMBreakTies = getModeScopedElement('SVMBreakTies').value
            let SVMverbose = getModeScopedElement('SVMverbose').value
            let SVMcoef0 = getModeScopedElement('SVMcoef0').value
            let SVMtol = getModeScopedElement('SVMtol').value
            let SVMCacheSize = getModeScopedElement('SVMCacheSize').value
            let SVMClassWeight = getModeScopedElement('SVMClassWeight').value
            let SVMmaxIter = getModeScopedElement('SVMmaxIter').value
            let SVMdecisionFunctionShape = getModeScopedElement('SVMdecisionFunctionShape').value
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
            let RFBoostrap = getModeScopedElement('RFBoostrap').value
            let RFoobScore = getModeScopedElement('RFoobScore').value
            let RFWarmStart = getModeScopedElement('RFWarmStart').value
            let RFmin_weight_fraction_leaf = getModeScopedElement('RFmin_weight_fraction_leaf').value
            let RFMaxLeafNodes = getModeScopedElement('RFMaxLeafNodes').value
            let RFMinImpurityDecrease = getModeScopedElement('RFMinImpurityDecrease').value
            let RFNJobs = getModeScopedElement('RFNJobs').value
            //let RFRandomState = document.getElementById('RFRandomState').value
            let RFVerbose  = getModeScopedElement('RFVerbose').value

            const min_samples_split = getModeScopedElement('min_samples_split');
            const min_samples_leaf = getModeScopedElement('min_samples_leaf');
            const max_depth = getModeScopedElement('RFMax_depth');
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

        let nonreqPerceptronSlider = getModeScopedElement('nonreqPerceptronSlider')
        if (nonreqPerceptronSlider.checked){
            nonreq=true
            let PerceptronFitIntercept = getModeScopedElement('PerceptronFitIntercept')?.value
            let PerceptronShuffle = getModeScopedElement('PerceptronShuffle')?.value
            let PerceptronEarlyStopping = getModeScopedElement('PerceptronEarlyStopping')?.value
            let PerceptronWarmStart = getModeScopedElement('PerceptronWarmStart')?.value
            let PerceptronPenalty = getModeScopedElement('PerceptronPenalty')?.value
            let PerceptronAlpha = getModeScopedElement('PerceptronAlpha')?.value
            let PerceptronTol = getModeScopedElement('PerceptronTol')?.value
            let PerceptronVerbose = getModeScopedElement('PerceptronVerbose')?.value
            let PerceptronNJobs = getModeScopedElement('PerceptronNJobs')?.value
            //let PerceptronRandomState = getModeScopedElement('PerceptronRandomState')?.value
            let PerceptronValidationFraction = getModeScopedElement('PerceptronValidationFraction')?.value
            let PerceptronNIterNoChange = getModeScopedElement('PerceptronNIterNoChange')?.value
            let PerceptronClassWeight = getModeScopedElement('PerceptronClassWeight')?.value
            
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

        let nonreqMLPSlider = getModeScopedElement('nonreqMLPSlider')
        if (nonreqMLPSlider.checked){
            nonreq=true
            let MLPNesterovsMomentum = getModeScopedElement('MLPNesterovsMomentum').value
            let MLPEarlyStopping = getModeScopedElement('MLPEarlyStopping').value
            let MLPShuffle = getModeScopedElement('MLPShuffle').value
            let MLPVerbose = getModeScopedElement('MLPVerbose').value
            let MLPWarmStart = getModeScopedElement('MLPWarmStart').value
            let MLPBatchSize = getModeScopedElement('MLPBatchSize').value
            let MLPLearningRateInit = getModeScopedElement('MLPLearningRateInit').value
            let MLPPowerT = getModeScopedElement('MLPPowerT').value
            let MLPMaxIter = getModeScopedElement('MLPMaxIter').value
            //let MLPRandomState = document.getElementById('MLPRandomState').value
            let MLPTol = getModeScopedElement('MLPTol').value
            let MLPMomentum = getModeScopedElement('MLPMomentum').value
            let MLPValidationFraction = getModeScopedElement('MLPValidationFraction').value
            let MLPBeta1 = getModeScopedElement('MLPBeta1').value
            let MLPBeta2 = getModeScopedElement('MLPBeta2').value
            let MLPEpsilon = getModeScopedElement('MLPEpsilon').value
            const alpha = getModeScopedElement('MLPAlpha');
            const learning_rate = getModeScopedElement('MLPLearning_rate');
            
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

        let nonreqKNearestSlider = getModeScopedElement('nonreqKNearestSlider')
        if (nonreqKNearestSlider.checked){
            nonreq=true
            let KNearestWeights = getModeScopedElement('KNearestWeights').value
            let KNearestAlgorithm = getModeScopedElement('KNearestAlgorithm').value
            let KNearestLeafSize = getModeScopedElement('KNearestLeafSize').value
            let KNearestP = getModeScopedElement('KNearestP').value
            let KNearestMetricParams = getModeScopedElement('KNearestMetricParams').value
            let KNearestNJobs = getModeScopedElement('KNearestNJobs').value
            const metric = getModeScopedElement('metric');
            
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
    

        let nonreqGBSlider = getModeScopedElement('nonreqGBSlider')
        if (nonreqGBSlider.checked){
            nonreq=true
            let GBLoss = getModeScopedElement('GBLoss').value
            let GBWarmStart = getModeScopedElement('GBWarmStart').value
            let GBCriterion = getModeScopedElement('GBCriterion').value
            let GBSubsample = getModeScopedElement('GBSubsample').value
            let GBMinSamplesSplit = getModeScopedElement('GBMinSamplesSplit').value
            let GBMinSamplesLeaf = getModeScopedElement('GBMinSamplesLeaf').value
            let GBMinWeightFractionLeaf = getModeScopedElement('GBMinWeightFractionLeaf').value
            let GBMinImpurityDecrease = getModeScopedElement('GBMinImpurityDecrease').value
            let GBInit = getModeScopedElement('GBInit').value
            //let GBRandomState = document.getElementById('GBRandomState').value
            let GBMaxFeatrues = getModeScopedElement('GBMaxFeatrues').value
            let GBAlpha = getModeScopedElement('GBAlpha').value
            let GBVerbose = getModeScopedElement('GBVerbose').value
            let GBMaxLeafNodes = getModeScopedElement('GBMaxLeafNodes').value
            const max_depth = getModeScopedElement('GBMax_depth');
            
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
        let nonreqLogisticClassifierSlider = getModeScopedElement('nonreqLogisticClassifierSlider')
        if (nonreqLogisticClassifierSlider.checked){
            nonreq=true
            let Class_LogisticDual = getModeScopedElement('Class_LogisticDual').value
            let Class_LogisticFitIntercept = getModeScopedElement('Class_LogisticFitIntercept').value
            let Class_LogisticSolver = getModeScopedElement('Class_LogisticSolver').value
            let Class_LogisticMultiClass = getModeScopedElement('Class_LogisticMultiClass').value
            let Class_LogisticWarmStart = getModeScopedElement('Class_LogisticWarmStart').value
            let Class_CLogistic = getModeScopedElement('Class_CLogistic').value
            let Class_Logistic_penalty = getModeScopedElement('Class_Logistic_penalty').value
            let Class_LogisticTol = getModeScopedElement('Class_LogisticTol').value
            let Class_Logisticintercept_scaling = getModeScopedElement('Class_Logisticintercept_scaling').value
            let Class_LogisticClassWeight = getModeScopedElement('Class_LogisticClassWeight').value
            let Class_LogisticMaxIterations = getModeScopedElement('Class_LogisticMaxIterations').value
            let Class_LogisticVerbose = getModeScopedElement('Class_LogisticVerbose').value
            let Class_LogisticNJobs = getModeScopedElement('Class_LogisticNJobs').value
            let Class_Logisticl1Ratio = getModeScopedElement('Class_Logisticl1Ratio').value

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

        let nonreqMLPSlider = getModeScopedElement('nonreqMLPClassifierSlider')
        if (nonreqMLPSlider.checked){
            nonreq=true
            let MLPNesterovsMomentum = getModeScopedElement('Class_MLPNesterovsMomentum').value
            let MLPEarlyStopping = getModeScopedElement('Class_MLPEarlyStopping').value
            let MLPShuffle = getModeScopedElement('Class_MLPShuffle').value
            let MLPVerbose = getModeScopedElement('Class_MLPVerbose').value
            let MLPWarmStart = getModeScopedElement('Class_MLPWarmStart').value
            let MLPBatchSize = getModeScopedElement('Class_MLPBatchSize').value
            let MLPLearningRateInit = getModeScopedElement('Class_MLPLearningRateInit').value
            let MLPPowerT = getModeScopedElement('Class_MLPPowerT').value
            let MLPMaxIter = getModeScopedElement('Class_MLPMaxIter').value
            //let MLPRandomState = document.getElementById('MLPRandomState').value
            let MLPTol = getModeScopedElement('Class_MLPTol').value
            let MLPMomentum = getModeScopedElement('Class_MLPMomentum').value
            let MLPValidationFraction = getModeScopedElement('Class_MLPValidationFraction').value
            let MLPBeta1 = getModeScopedElement('Class_MLPBeta1').value
            let MLPBeta2 = getModeScopedElement('Class_MLPBeta2').value
            let MLPEpsilon = getModeScopedElement('Class_MLPEpsilon').value
            const alpha = getModeScopedElement('Class_MLPAlpha');
            const learning_rate = getModeScopedElement('Class_MLPLearning_rate');
            
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
        

        let nonreqRFSlider = getModeScopedElement('nonreqRFClassifierSlider')
        if (nonreqRFSlider.checked){
            nonreq=true
            let RFBoostrap = getModeScopedElement('Class_RFBoostrap').value
            let RFoobScore = getModeScopedElement('Class_RFoobScore').value
            let RFWarmStart = getModeScopedElement('Class_RFWarmStart').value
            let RFmin_weight_fraction_leaf = getModeScopedElement('Class_RFmin_weight_fraction_leaf').value
            let RFMaxLeafNodes = getModeScopedElement('Class_RFMaxLeafNodes').value
            let RFMinImpurityDecrease = getModeScopedElement('Class_RFMinImpurityDecrease').value
            let RFNJobs = getModeScopedElement('Class_RFNJobs').value
            //let RFRandomState = document.getElementById('RFRandomState').value
            let RFVerbose  = getModeScopedElement('Class_RFVerbose').value

            const min_samples_split = getModeScopedElement('Class_min_samples_split');
            const min_samples_leaf = getModeScopedElement('Class_min_samples_leaf');
            const max_depth = getModeScopedElement('Class_RFMax_depth');
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

        let nonreqSVMSlider = getModeScopedElement('nonreqSVCClassifierSlider')
        if (nonreqSVMSlider.checked){
            nonreq=true
            let SVCshrinking = getModeScopedElement('SVCshrinking').value
            let SVCprobability = getModeScopedElement('SVCprobability').value
            let SVCBreakTies = getModeScopedElement('SVCBreakTies').value
            let SVCverbose = getModeScopedElement('SVCverbose').value
            let SVCcoef0 = getModeScopedElement('SVCcoef0').value
            let SVCtol = getModeScopedElement('SVCtol').value
            let SVCCacheSize = getModeScopedElement('SVCCacheSize').value
            let SVCClassWeight = getModeScopedElement('SVCClassWeight').value
            let SVCmaxIter = getModeScopedElement('SVCmaxIter').value
            let SVCdecisionFunctionShape = getModeScopedElement('SVCdecisionFunctionShape').value
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
        
        let nonreqAgglomerativeSlider = getModeScopedElement('nonreqAgglomerativeSlider')
        if (nonreqAgglomerativeSlider.checked){
            nonreq=true
            let metric = getModeScopedElement('Aggmetric').value
            let memory = getModeScopedElement('Aggmemory').value
            let connectivity = getModeScopedElement('Aggconnectivity').value
            let compute_full_tree = getModeScopedElement('aggcompute_full_tree').value
            let linkage = getModeScopedElement('Agglinkage').value
            let distance_threshold = getModeScopedElement('Aggdistance_threshold').value
            let compute_distances = getModeScopedElement('aggcompute_distances').value

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
        
        let nonreqGaussianSlider = getModeScopedElement('nonreqGaussianSlider')
        if (nonreqGaussianSlider.checked){
            nonreq=true
            let covariance_type = getModeScopedElement('Gaucovariance_type').value
            let tol = getModeScopedElement('GauTol').value
            let reg_covar = getModeScopedElement('Gaureg_covar').value
            let max_iter = getModeScopedElement('GauMax_iter').value
            let n_init = getModeScopedElement('Gaun_init').value
            let init_params = getModeScopedElement('Gauinit_params').value
            let weights_init = getModeScopedElement('Gauweights_init').value
            let means_init = getModeScopedElement('Gaumeans_init').value
            let precisions_init = getModeScopedElement('Gauprecisions_init').value
            let warm_start = getModeScopedElement('GauWarmStart').value
            let verbose = getModeScopedElement('GauVerbose').value
            let verbose_interval = getModeScopedElement('GauVerbose_interval').value

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
        
        let nonreqKmeansSlider = getModeScopedElement('nonreqKmeansSlider')
        if (nonreqKmeansSlider.checked){
            nonreq=true
            let init = getModeScopedElement('kmeansInit').value
            let n_init = getModeScopedElement('kmeansn_init').value
            let max_iter = getModeScopedElement('kmeansmax_iter').value
            let tol = getModeScopedElement('kmeanstol').value
            let verbose = getModeScopedElement('kmeansverbose').value
            let copy_x = getModeScopedElement('kmeansCopyX').value
            let algorithm = getModeScopedElement('Kmeansalgorithm').value

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
            if (dual) hyperparameters['dual'] = dual.value === 'auto' ? 'auto' : dual.value === 'true';
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
            if (dual) hyperparameters['dual'] = dual.value === 'auto' ? 'auto' : dual.value === 'true';
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
            if (nInit) hyperparameters['n_init'] = nInit.value.trim() === 'auto' ? 'auto' : parseInt(nInit.value);
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
    
    // Split method and column are Advanced-mode controls. The column is a
    // column letter, converted to a 0-based index like stratifyColumn; the server
    // resolves the name.
    let splitMethod = 'random';
    let splitColumn = '';
    let allowUnsafe = false;
    let allowUnsafeReason = '';
    if (currentMode === 'advanced') {
        const splitMethodEl = document.getElementById('advancedSplitMethod');
        const splitColumnEl = document.getElementById('advancedSplitColumn');
        splitMethod = splitMethodEl ? splitMethodEl.value : 'random';
        const splitLetter = splitColumnEl ? splitColumnEl.value.trim() : '';
        splitColumn = splitLetter !== '' ? columnToIndex(splitLetter.toUpperCase()) : '';
        const allowUnsafeEl = document.getElementById('advancedAllowUnsafe');
        allowUnsafe = allowUnsafeEl ? allowUnsafeEl.value === 'yes' : false;
        const reasonEl = document.getElementById('advancedAllowUnsafeReason');
        allowUnsafeReason = reasonEl ? reasonEl.value.trim() : '';
    }

    //sending all the data to the backend
    const requestData = {
        filename: uploadedFileName,
        indicators: indicatorCols,
        predictors: predictorCols,
        models: selectedModel,
        taskType: selectedOutputType,
        scaler: scaler.value,
        hyperparameters: hyperparameters,
        nonreq: nonreq,
        units: unitStr,
        sigfig: parseInt(sigfig),
        stratifyColumn: stratifyColumnNumber,
        splitMethod: splitMethod,
        splitColumn: splitColumn,
        allowUnsafe: allowUnsafe,
        allowUnsafeReason: allowUnsafeReason,
        stratifyBool: stratifyBool,
        seedValue: parseInt(seedValue),
        testSize: parseFloat(testSize),
        dropMissing: dropMissing,
        imputeStrategy: imputeStrategy,
        dropZero: dropZero,
        zeroHandle: zeroHandle,
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
            loadingDiv = getAdvancedLoadingDiv();
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
                        console.error('processResultData function not set');
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
                            loadingDiv = getAdvancedLoadingDiv();
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
                            const advancedButton = getAdvancedRunSubmitButton();
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
                        const advancedButton = getAdvancedRunSubmitButton();
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
                    loadingDiv = getAdvancedLoadingDiv();
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
                const advancedButton = getAdvancedRunSubmitButton();
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
                currentLoadingDiv = getAdvancedLoadingDiv();
            } else if (currentMode === 'automl') {
                currentLoadingDiv = document.getElementById('automlLoading');
            } else {
                currentLoadingDiv = getCachedElement('loading');
            }
            
            // Show error message in loading div
            if (currentLoadingDiv) {
                currentLoadingDiv.innerHTML = `
                    <p style="color: #d32f2f; font-weight: 600;">Connection Error</p>
                    <p style="color: #666;">The connection to the server failed.</p>
                `;
            }
            
            eventSource.close();
            progressEventSource = null;
        };
    }
    
    // Function to update progress display
    function updateProgressDisplay(progress, container) {
        if (typeof narrativeSetStages === 'function') narrativeSetStages(progress);
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
    if (typeof narrativeSetStep === 'function') narrativeSetStep('run');
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
                    loadingDiv = getAdvancedLoadingDiv();
                    const advancedButton = getAdvancedRunSubmitButton();
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
            showError(errorDiv, data.error ? `Error: ${data.error}` : 'The request failed. Open the browser console for the details.');
        }
    } catch (error) {
        console.error('Error:', error);
        // Always clean up on error, regardless of async processing state
        // This handles cases where error occurs after isAsyncProcessing is set but before SSE is established
        if (progressEventSource) {
            progressEventSource.close();
            progressEventSource = null;
        }
        showError(errorDiv, 'The run failed. Open the browser console for the details.');
        
        // Determine which loading div and button to use based on current mode
        const simpleMode = document.getElementById('simpleMode');
        const advancedMode = document.getElementById('advancedMode');
        const automlMode = document.getElementById('automlMode');
        const currentMode = simpleMode?.checked ? 'simple' : (advancedMode?.checked ? 'advanced' : (automlMode?.checked ? 'automl' : 'simple'));
        
        let loadingDiv;
        if (currentMode === 'advanced') {
            loadingDiv = getAdvancedLoadingDiv();
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
                    const advancedButton = getAdvancedRunSubmitButton();
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
            // Hide the correct loading region (unified advanced tab uses #advancedLoading, legacy panel uses #advancedOptimizationLegacyLoading)
            const _smFin = document.getElementById('simpleMode');
            const _amFin = document.getElementById('advancedMode');
            const _umFin = document.getElementById('automlMode');
            const cleanupMode = _smFin?.checked ? 'simple' : (_amFin?.checked ? 'advanced' : (_umFin?.checked ? 'automl' : 'simple'));
            const loadingDiv = cleanupMode === 'advanced'
                ? getAdvancedLoadingDiv()
                : (cleanupMode === 'automl' ? document.getElementById('automlLoading') : loading);
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
// Run notes shown above every result: data checks from the server, the split
// method, and the comparison against a baseline that predicts the training mean
// or the most common class on the same split.
function renderRunNotes(data) {
    const parts = [];
    const checks = data && data.data_checks;
    if (checks) {
        const blocking = Array.isArray(checks.blocking) ? checks.blocking : [];
        const warnings = Array.isArray(checks.warnings) ? checks.warnings : [];
        if (blocking.length > 0) {
            const reason = checks.override_reason ? ` Reason given: ${escapeHtml(checks.override_reason)}` : '';
            parts.push(`<p class="run-note run-note-block"><strong>Not valid for decision use. Data checks blocked this run and the override trained it anyway.</strong> ${escapeHtml(blocking.map(f => f.message).join(' '))}${reason}</p>`);
        }
        if (warnings.length > 0) {
            parts.push(`<p class="run-note run-note-warn"><strong>Data checks.</strong> ${escapeHtml(warnings.map(f => f.message).join(' '))}</p>`);
        }
        if (blocking.length === 0 && warnings.length === 0) {
            parts.push(`<p class="run-note field-note">Data checks passed on ${escapeHtml(String(checks.n_rows))} rows and ${escapeHtml(String(checks.n_features))} features.</p>`);
        }
    }
    const cmp = data && data.model_comparison;
    if (cmp && cmp.chosen) {
        const rows = Array.isArray(cmp.summary) ? cmp.summary.map(r => `${escapeHtml(String(r.model).replace('_classifier', ''))} ${Number(r.mean).toFixed(3)} ± ${Number(r.std).toFixed(3)}`).join('; ') : '';
        parts.push(`<p class="run-note field-note"><strong>AutoML chose ${escapeHtml(String(cmp.chosen).replace('_classifier', ''))}.</strong> ${escapeHtml(String(cmp.note || ''))} Mean ± SD over ${escapeHtml(String(cmp.folds))} folds and ${escapeHtml(String(cmp.seeds))} seed${cmp.seeds === 1 ? '' : 's'}: ${rows}. The comparison is drawn under Graphics.</p>`);
    }
    if (data && data.capacity_note) {
        parts.push(`<p class="run-note run-note-warn"><strong>Model not kept.</strong> ${escapeHtml(String(data.capacity_note))}</p>`);
    }
    if (data && data.split_note) {
        parts.push(`<p class="run-note field-note">${escapeHtml(String(data.split_note))}</p>`);
    }
    const base = data && data.baseline;
    if (base && base.metric) {
        const better = base.model_beats_baseline;
        const verdict = better === true ? 'The model beats the baseline.' : (better === false ? 'The model does not beat the baseline. The score adds nothing beyond the target distribution.' : '');
        parts.push(`<p class="run-note ${better === false ? 'run-note-warn' : 'field-note'}"><strong>Baseline.</strong> ${escapeHtml(base.description)} scores ${escapeHtml(String(base.baseline_value))} ${escapeHtml(base.metric)} on the validation split against ${escapeHtml(String(base.model_value))} for the model. ${verdict}</p>`);
    }
    if (data && data.model_card_url) {
        parts.push(`<p class="run-note field-note"><a href="${escapeHtml(data.model_card_url)}" target="_blank" rel="noopener">Model card</a> and <a href="${escapeHtml(data.bundle_url)}">model bundle (.zip)</a> for this run.</p>`);
    }
    return parts.length ? `<div class="run-notes">${parts.join('')}</div>` : '';
}

function processModelResult(data, unitStr = '', predictorCols = [], hyperparameters = {}) {
    if (typeof narrativeSetStep === 'function') narrativeSetStep('results');
    // The way on shows once there is a model to apply, inside the results panel.
    // Queued rather than called: the panel this appends to is revealed by the render
    // that is still running, so a direct call finds every panel hidden.
    if (typeof showInferenceForward === 'function') window.setTimeout(showInferenceForward, 0);
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
            if (errorDiv) showError(errorDiv, `Error: ${data.error}`);
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
        
        if (errorDiv) errorDiv.innerHTML = '';
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
                // True when not on the standalone legacy advanced page (#advancedOptimization); unified advanced tab is still "simple" for this value (CV button placement).
                const isLegacyAdvancedPageVisible = document.getElementById('advancedOptimization') && !document.getElementById('advancedOptimization').classList.contains('hidden');
                const isSimpleModelingPage = !isLegacyAdvancedPageVisible;
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
                    
                    // Filter visuals: baseline only for Modeling page (advanced-only entries are for the Advanced Optimization flow)
                    const baselineVisuals = allRegressionVisuals.filter(v => v.type === 'baseline' || !v.type || v.type === 'default');
                    
                    // Use baseline visuals for Modeling page
                    const regressionVisuals = baselineVisuals.length > 0 ? baselineVisuals : allRegressionVisuals;
                    // Remove "Baseline" and any combined-view options for Simple Modeling page
                    const regressionVisualsClean = regressionVisuals
                        .filter(v => !/combined/i.test(v.label || ''))
                        .map(v => ({
                            ...v,
                            label: (v.label || '').replace(/\s*-\s*Baseline\s*$/i, '').trim()
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
                                    `<tr><td colspan="2"><strong>Selected Feature Names:</strong><br>${escapeHtml(data.feature_selection_info.selected_features.join(', '))}</td></tr>` : ''}
                            </table>
                        </div>` : (Array.isArray(data.skipped_options) && data.skipped_options.length > 0 ? `<p class="field-note">${escapeHtml(data.skipped_options.map(o => `${o.option}: ${o.reason}`).join('. '))}.</p>` : '<p>No feature selection data available</p>');
                    
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
                    ${renderRunNotes(data)}
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
                                        <button type="button" class='downloadperformanceButton export-button'>Performance XLSX</button>
                                    </a>
                                    <a href="/download/visualizations.pdf?download_name=${encodeURIComponent(visualizationsDownloadName)}" onclick="return downloadFile('visualizations.pdf', '${visualizationsDownloadName}')">
                                        <button class="export-button" style="font-size: 0.95rem;">Graphics PDF</button>
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
                    // Always re-query after innerHTML: a static placeholder div may have had the same id before replace.
                    imageSelector = document.getElementById("imageSelector") ||
                        document.getElementById("advancedImageSelector") ||
                        document.getElementById("automlImageSelector");
                    const imageSelector2 = document.getElementById(currentMode === 'simple' ? 'imageSelector2' : currentMode === 'advanced' ? 'advancedImageSelector2' : 'automlImageSelector2');
                    const targetGraphicId = currentMode === 'simple' ? 'targetGraphic' : currentMode === 'advanced' ? 'advancedTargetGraphic' : 'automlTargetGraphic';
                    const targetGraphicId2 = currentMode === 'simple' ? 'targetGraphic2' : currentMode === 'advanced' ? 'advancedTargetGraphic2' : 'automlTargetGraphic2';
                    const visualSelectorId = currentMode === 'simple' ? 'regressionVisualSelector' : currentMode === 'advanced' ? 'advancedRegressionVisualSelector' : 'automlRegressionVisualSelector';
                    const visualSelectorId2 = currentMode === 'simple' ? 'regressionVisualSelector2' : currentMode === 'advanced' ? 'advancedRegressionVisualSelector2' : 'automlRegressionVisualSelector2';
                    let targetGraphic = document.getElementById(targetGraphicId);
                    let targetGraphic2 = document.getElementById(targetGraphicId2);
                    const regressionVisualSelector = document.getElementById(visualSelectorId);
                    const regressionVisualSelector2 = document.getElementById(visualSelectorId2);

                    if (imageSelector && Array.isArray(data.predictors)) {
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
                    } else {
                        console.error('Regression multi-target: target <select> not found after render (check duplicate ids).');
                    }

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
                    if (imageSelector) imageSelector.addEventListener("change", () => updateRegressionGraphic(1));
                    if (regressionVisualSelector) regressionVisualSelector.addEventListener("change", () => updateRegressionGraphic(1));
                    if (imageSelector2) imageSelector2.addEventListener("change", () => updateRegressionGraphic(2));
                    if (regressionVisualSelector2) regressionVisualSelector2.addEventListener("change", () => updateRegressionGraphic(2));
                    updateRegressionGraphic(1);
                    updateRegressionGraphic(2);
                }

                //when only one target to display 
                //there is no dropdown to select which target's graphic to display 
                //and SHAP graphic can be displayed
                else{
                    const allRegressionVisualsSingle = data.regression_visuals || [
                        { label: 'Predicted vs Actual + Residuals', file: 'target_plot' },
                    ];
                    
                    // Filter visuals: baseline only for Modeling page
                    const regressionVisuals = allRegressionVisualsSingle.filter(v => v.type === 'baseline' || !v.type || v.type === 'default');
                    // Remove "Baseline" and any combined-view options for Simple Modeling page
                    const regressionVisualsClean = regressionVisuals
                        .filter(v => !/combined/i.test(v.label || ''))
                        .map(v => ({
                            ...v,
                            label: (v.label || '').replace(/\s*-\s*Baseline\s*$/i, '').trim()
                        }));
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
                                    `<tr><td colspan="2"><strong>Selected Feature Names:</strong><br>${escapeHtml(data.feature_selection_info.selected_features.join(', '))}</td></tr>` : ''}
                            </table>
                        </div>` : (Array.isArray(data.skipped_options) && data.skipped_options.length > 0 ? `<p class="field-note">${escapeHtml(data.skipped_options.map(o => `${o.option}: ${o.reason}`).join('. '))}.</p>` : '<p>No feature selection data available</p>');
                    
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
                ${renderRunNotes(data)}
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
                                    <button type="button" class='downloadperformanceButton export-button'>Performance XLSX</button>
                                </a>
                                <a href="/download/visualizations.pdf?download_name=${encodeURIComponent(visualizationsDownloadName)}" onclick="return downloadFile('visualizations.pdf', '${visualizationsDownloadName}')">
                                    <button class="export-button" style="font-size: 0.95rem;">Graphics PDF</button>
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
                                `<tr><td colspan="2"><strong>Selected Feature Names:</strong><br>${escapeHtml(data.feature_selection_info.selected_features.join(', '))}</td></tr>` : ''}
                        </table>
                    </div>` : (Array.isArray(data.skipped_options) && data.skipped_options.length > 0 ? `<p class="field-note">${escapeHtml(data.skipped_options.map(o => `${o.option}: ${o.reason}`).join('. '))}.</p>` : '<p>No feature selection data available</p>');
                
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
                ${renderRunNotes(data)}
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
                                    <button type="button" class='downloadperformanceButton export-button'>Performance XLSX</button>
                                </a>
                                <a href="/download/visualizations.pdf?download_name=${encodeURIComponent(visualizationsDownloadName)}" onclick="return downloadFile('visualizations.pdf', '${visualizationsDownloadName}')">
                                    <button class="export-button" style="font-size: 0.95rem;">Graphics PDF</button>
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

                <h3>Graphics</h3>
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
                // The candidate comparison exists only for an AutoML run with no model named.
                if (data.model_comparison && data.model_comparison.figure) {
                    [classifierImageSelector, classifierImageSelector2].forEach((sel, i) => {
                        if (!sel) return;
                        const o = document.createElement('option');
                        o.value = 'model_comparison'; o.textContent = 'Candidate model comparison';
                        sel.insertBefore(o, sel.firstChild);
                        if (i === 1) sel.value = 'model_comparison';
                    });
                }
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
                    ${renderRunNotes(data)}
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
                                    <a href="${withApiRoot('/download/model_performance.xlsx')}?download_name=${encodeURIComponent(performanceDownloadName)}" onclick="return downloadFile('model_performance.xlsx', '${performanceDownloadName}')">
                                        <button class='downloadperformanceButton export-button'>Performance XLSX</button>
                                    </a>
                                    <a href="${withApiRoot('/download/visualizations.pdf')}?download_name=${encodeURIComponent(visualizationsDownloadName)}" onclick="return downloadFile('visualizations.pdf', '${visualizationsDownloadName}')">
                                        <button class="export-button" style="font-size: 0.95rem;">Graphics PDF</button>
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

                    <h3>Graphics</h3>
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
            if (errorDiv) showError(errorDiv, `Error: ${data.error}`);
            hideElement(NumericResultDiv);
            hideElement(ClassifierResultDiv);
            hideElement(ClusterResultDiv);
        }
    } catch (error) {
        console.error('Error processing result:', error);
        const errDiv = getCachedElement('errorDiv');
        if (errDiv) showError(errDiv, 'DiGiTerra cannot process the result. Open the browser console for the details.');
    }
}



/* Put "Continue to Inference on New Data" in the results panel of the visible mode.
   It appears only after a run has finished, because before that there is no model to
   apply and the button would lead to a page that refuses. */
function showInferenceForward() {
    const panels = ['NumericResultDiv', 'ClusterResultDiv', 'ClassifierResultDiv']
    for (const id of panels) {
        const panel = document.getElementById(id)
        if (!panel || panel.classList.contains('hidden')) continue
        if (panel.querySelector('.results-forward')) return
        const wrap = document.createElement('div')
        wrap.className = 'results-forward'
        const button = document.createElement('button')
        button.type = 'button'
        button.className = 'success-button'
        button.id = 'continueToInferenceButton'
        button.textContent = 'Continue to Inference on New Data'
        button.addEventListener('click', () => {
            if (typeof showTab === 'function') showTab('historic')
        })
        wrap.appendChild(button)
        panel.appendChild(wrap)
        return
    }
}


// === narrative.js ===
/**
 * The application as a narrative: one rail down the left carrying navigation, the
 * loaded data and the model catalog, with the views to its right.
 *
 * The rail is a sibling of the views, built once on DOMContentLoaded, so it is
 * there from a cold load rather than appearing after an upload. Its navigation is
 * the old top tab bar: the same button nodes are moved into it, so every id, every
 * listener and setActiveTab's NodeList keep working. Entries after Upload are
 * disabled until a file is loaded, and say why in their title.
 *
 * The five-step strip is not navigation. It stays inside the Modeling view and
 * reports where a run has got to.
 *
 * Everything is built from the page that is already there. The mode sections keep
 * their markup, their element ids and their listeners; nodes are moved, never
 * rebuilt, so nothing that reads a control by id is affected. The catalog is read
 * from the model <select> elements rather than from a second list, so it cannot
 * drift from the registry. Depends on core.js.
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

// The views the rail navigates, in the order a run visits them. These are the
// buttons the header used to carry; the nodes are moved, not recreated, so the
// data-tab values and the handlers core.js bound to them are untouched.
const RAIL_VIEWS = ['upload', 'processing', 'model-preprocessing', 'modeling', 'historic'];

// Everything after Upload needs a file. The reason is what the entry says when a
// person hovers a disabled one, so the rail explains itself rather than going dead.
const RAIL_NEEDS_DATA = 'Upload a file first.';

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
/** The task the run is for, from the choice the app recorded. */
function narrativeTaskKey() {
    const el = document.getElementById('outputType1');
    const value = el && el.value ? el.value : '';
    if (value === 'Classifier') return 'classification';
    if (value === 'Cluster') return 'cluster';
    if (value === 'Numeric') return 'regression';
    return '';
}

/**
 * The model select the catalog reads.
 *
 * The task comes from the recorded choice rather than from which select happens to
 * be on screen. Picking by visibility made the count follow the page: on Modeling a
 * classification select was visible and the chip read 23, and on Documentation
 * nothing was visible so the first entry won and the same session read 31.
 */
function narrativeActiveSelect() {
    const byTask = MODEL_SELECTS[narrativeCurrentMode()] || {};
    const task = narrativeTaskKey();
    if (task && byTask[task]) {
        const el = document.getElementById(byTask[task]);
        if (el) return el;
    }
    // No task chosen yet. The regression select is the empty state the rail shows.
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

/** Whether the app has a loaded file, by the same test the views use. */
function narrativeHasData() {
    try {
        if (typeof checkHasDataForExploration === 'function') return Boolean(checkHasDataForExploration());
    } catch (err) {
        console.error('Data check failed:', err);
    }
    return Boolean(typeof uploadedFileName === 'string' && uploadedFileName);
}

/** Whether a task type has been chosen, which is what makes the catalog live. */
function narrativeTaskChosen() {
    const el = document.getElementById('outputType1');
    return Boolean(el && el.value);
}

/**
 * Put the page into a rail and a view column, once, on load.
 *
 * main's children are moved into the view column rather than re-created, so the
 * views keep their ids and their listeners. The rail becomes their sibling, which
 * is what lets it exist before any of them has anything in it.
 */
function narrativeBuildShell() {
    const main = document.getElementById('main-content');
    if (!main || document.getElementById('narrativePage')) return false;

    const page = document.createElement('div');
    page.id = 'narrativePage';
    page.className = 'narrative-page is-single';

    const rail = document.createElement('aside');
    rail.id = 'narrativeRail';
    rail.className = 'narrative-rail';
    rail.setAttribute('aria-label', 'Navigation, data and models');
    rail.innerHTML = `
        <nav class="narrative-nav" id="narrativeNav" aria-label="Main navigation"></nav>
        <section class="narrative-panel" id="narrativeDataPanel">
            <div class="narrative-panel-head">
                <h3>Data</h3>
                <span class="narrative-count" id="narrativeDataCount">0</span>
            </div>
            <div class="narrative-panel-body" id="narrativeDataBody">
                <p class="narrative-empty">No data loaded.</p>
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
                <p class="narrative-empty">No models listed.</p>
            </div>
        </section>
    `;

    const views = document.createElement('div');
    views.id = 'narrativeViews';
    views.className = 'narrative-views';
    while (main.firstChild) views.appendChild(main.firstChild);

    page.appendChild(rail);
    page.appendChild(views);
    main.appendChild(page);
    return true;
}

/**
 * Move the tab bar into the rail and retire the bar itself.
 *
 * The buttons are moved, not rebuilt: core.js captured them in a NodeList at load
 * and bound a click handler to each, and setActiveTab still writes to those same
 * nodes. Documentation goes last, in its own group.
 */
function narrativeMoveNav() {
    const nav = document.getElementById('narrativeNav');
    if (!nav || nav.childElementCount) return;
    const bar = document.getElementById('appTabs');
    if (!bar) return;

    RAIL_VIEWS.forEach((tab) => {
        const btn = bar.querySelector(`.tab-button[data-tab="${tab}"]`);
        if (btn) nav.appendChild(btn);
    });
    const rest = Array.from(bar.querySelectorAll('.tab-button[data-tab]')).filter(
        (b) => b.dataset.tab !== 'documentation'
    );
    rest.forEach((b) => nav.appendChild(b));

    // Documentation sits in the header's right-hand group rather than in the bar,
    // so it is found on the page rather than inside it.
    // Documentation and Tutorial stay in the banner. They have to work with no data
    // loaded, and with no data there is no rail to put them in.
    // The bar keeps its node so core.js can still address it; empty, it draws nothing.
    bar.classList.add('is-retired');

    // The forward step lives at the foot of the rail, under the entry it leaves.
    const next = document.createElement('button');
    next.type = 'button';
    next.id = 'narrativeNext';
    next.className = 'success-button narrative-next hidden';
    next.addEventListener('click', () => {
        const source = narrativeNextSource();
        if (source && !source.disabled) source.click();
    });
    nav.appendChild(next);
}

// The in-page button that carries each view forward. The rail's forward button is
// a proxy for it: same label, same enabled state, same click. The page keeps its
// own button as the thing that holds the behaviour; the rail only repeats it.
const NEXT_SOURCE = {
    processing: 'continueToPreprocessButton',
    'model-preprocessing': 'continueToModelingButton',
    modeling: 'continueToInferenceButton',
};

function narrativeNextSource() {
    const active = document.querySelector('#narrativeNav .tab-button.active');
    const id = active ? NEXT_SOURCE[active.dataset.tab] : null;
    const source = id ? document.getElementById(id) : null;
    // A source inside a hidden panel is not on offer, so neither is the proxy.
    return source && !source.closest('.hidden') ? source : null;
}

/** Repeat the current view's forward button in the rail, or hide the proxy. */
function narrativeRefreshNext() {
    const next = document.getElementById('narrativeNext');
    if (!next) return;
    const source = narrativeNextSource();
    const show = Boolean(source);
    if (next.classList.contains('hidden') === show) next.classList.toggle('hidden', !show);
    if (!source) return;
    const label = source.textContent.trim();
    if (next.textContent !== label) next.textContent = label;
    if (next.disabled !== source.disabled) next.disabled = source.disabled;
    const title = source.getAttribute('title') || '';
    if ((next.getAttribute('title') || '') !== title) {
        if (title) next.setAttribute('title', title);
        else next.removeAttribute('title');
    }
}

/**
 * Disable what the app would refuse.
 *
 * Everything past Upload needs a file. A disabled button does not fire its click,
 * so the rail cannot reach a view the app would turn away, and the title says why.
 */
function narrativeGateNav() {
    const nav = document.getElementById('narrativeNav');
    if (!nav) return;
    const ready = narrativeHasData();
    nav.querySelectorAll('.tab-button[data-tab]').forEach((btn) => {
        const tab = btn.dataset.tab;
        const gated = tab !== 'upload' && tab !== 'documentation' && !ready;
        btn.disabled = gated;
        btn.classList.toggle('is-gated', gated);
        if (gated) {
            btn.setAttribute('title', RAIL_NEEDS_DATA);
            btn.setAttribute('aria-disabled', 'true');
        } else {
            btn.removeAttribute('title');
            btn.removeAttribute('aria-disabled');
        }
    });
}

/**
 * The run strip, inside the Modeling view.
 *
 * This is progress, not navigation: it says where a run has got to. It also takes
 * the page header, so Modeling carries one band rather than three.
 */
function narrativeBuildFlow() {
    const host = document.getElementById('columnSelection');
    if (!host || document.getElementById('narrativeFlow')) return false;

    const flow = document.createElement('nav');
    flow.id = 'narrativeFlow';
    flow.className = 'narrative-flow';
    flow.setAttribute('aria-label', 'Modeling and run progress');
    flow.innerHTML = `<ol>${RUN_STEPS.map(
        (s, i) =>
            `<li class="narrative-step" data-step="${s.key}"><span class="narrative-step-num">${i + 1}</span><span class="narrative-step-label">${s.label}</span></li>`
    ).join('')}</ol>`;

    host.insertBefore(flow, host.firstChild);
    // The page header is gone: the rail is the navigation and its active entry is the
    // title. The strip stays as one slim line of run progress at the top of the view.
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
    // The chips report a run in progress. Once every stage is done the strip's
    // Results step says the same thing, so the row goes away.
    const allDone = chips.length > 0 && Object.values(stages)
        .filter((stage) => stage.status !== 'skipped')
        .every((stage) => stage.status === 'completed');
    host.innerHTML = chips.length && !allDone ? chips.join('') : '';
    host.hidden = !chips.length || allDone;
}

/** Fill the data panel from what the page already knows about the upload. */
/** The letters a person typed, as a column range, and the name behind the first. */
function narrativeColumnLine() {
    const target = (document.getElementById('predictors') || {}).value || '';
    const indicators = (document.getElementById('indicators') || {}).value || '';
    if (!target && !indicators) return '';
    const named = (letters) => {
        const first = String(letters).trim().toUpperCase().match(/^[A-Z]+/);
        if (!first || typeof columnToIndex !== 'function') return '';
        const list = window.uploadedColumnNames;
        if (!Array.isArray(list)) return '';
        const name = list[columnToIndex(first[0])];
        return name ? ` (${name})` : '';
    };
    const parts = [];
    if (target) parts.push(`Target ${target.trim().toUpperCase()}${named(target)}`);
    if (indicators) parts.push(`indicators ${indicators.trim().toUpperCase()}`);
    return parts.join(', ');
}

function narrativeRefreshData() {
    const body = document.getElementById('narrativeDataBody');
    const count = document.getElementById('narrativeDataCount');
    if (!body) return;
    const name = typeof uploadedFileName === 'string' && uploadedFileName ? uploadedFileName : '';
    if (!name) {
        body.innerHTML = '<p class="narrative-empty">No data loaded.</p>';
        if (count) count.textContent = '0';
        return;
    }
    let shape = '';
    if (typeof uploadedShape === 'object' && uploadedShape && uploadedShape.rows) {
        shape = `${Number(uploadedShape.rows).toLocaleString()} \u00d7 ${uploadedShape.columns}`;
    }
    // The sentence the modeling page writes says the same thing at length. The panel
    // states the file, its shape, and which columns the run uses, once they are set.
    const columns = narrativeColumnLine();
    body.innerHTML = `
        <div class="narrative-item narrative-item-data">
            <span class="narrative-tile">CSV</span>
            <div>
                <strong>${escapeHtml(name)}</strong>
                <button type="button" class="narrative-replace" id="narrativeReplaceFile">Replace</button>
                ${shape ? `<p>${escapeHtml(shape)}</p>` : ''}
                ${columns ? `<p class="narrative-selection">${escapeHtml(columns)}</p>` : ''}
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
        body.innerHTML = '<p class="narrative-empty">No models listed.</p>';
        if (count) count.textContent = '0';
        return;
    }
    // Before a task type is known the catalog still shows the registry and its
    // group counts, so the rail reads as the application rather than as a gap.
    // The entries are inert: choosing one would write to a select the run does
    // not use yet.
    const live = narrativeTaskChosen();
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
                        `<button type="button" class="narrative-item narrative-app${o.value === select.value && live ? ' is-selected' : ''}${live ? '' : ' is-inert'}"${live ? '' : ' disabled title="Choose a task type in Model Preprocessing."'} data-model="${escapeHtml(o.value)}">` +
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

/**
 * Show the rail once a file is loaded, and keep it from then on.
 *
 * With no data the rail would carry an empty data panel, an inert catalog and five
 * entries that all refuse, so the welcome and upload screens take the whole page
 * instead. The shell is the same one either way: a class collapses the grid to one
 * column rather than a second layout being built.
 */
function narrativeApplyLayout() {
    const page = document.getElementById('narrativePage');
    if (!page) return;
    // narrativeGateNav reads the same answer, so the rail appears exactly when its
    // entries stop refusing.
    if (narrativeHasData()) page.classList.remove('is-single');
    else if (!page.dataset.railShown) page.classList.add('is-single');
    if (narrativeHasData()) page.dataset.railShown = 'true';
}

/* The catalog is for choosing a model and for reading which one is applied, so it
   shows on Modeling and on Inference and nowhere else. Upload, Data Exploration and
   Model Preprocessing have no use for it and it crowds the step they are on. */
const CATALOG_VIEWS = new Set(['modeling', 'historic']);

function narrativeApplyCatalogVisibility() {
    const panel = document.getElementById('narrativeAppsPanel');
    if (!panel) return;
    const active = document.querySelector('#narrativeNav .tab-button.active');
    const tab = active ? active.dataset.tab : '';
    panel.classList.toggle('hidden', !CATALOG_VIEWS.has(tab));
}

/** Bring the rail's parts up to date with whatever the app now knows. */
function narrativeSync() {
    narrativeApplyLayout();
    narrativeApplyCatalogVisibility();
    narrativeGateNav();
    narrativeRefreshNext();
    narrativeRefreshData();
    narrativeRefreshApps();
}

function narrativeWire() {
    const body = document.getElementById('narrativeAppsBody');
    if (body) {
        body.addEventListener('click', (e) => {
            const btn = e.target.closest('.narrative-app');
            if (!btn || btn.disabled) return;
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

    // The Data panel is redrawn, so the click is caught on the panel rather than
    // bound to a button that will not be the same node next time.
    const dataBody = document.getElementById('narrativeDataBody');
    if (dataBody) {
        dataBody.addEventListener('click', (e) => {
            if (!e.target.closest('#narrativeReplaceFile')) return;
            if (typeof openResetPopup === 'function') openResetPopup();
        });
    }

    ['indicators', 'predictors'].forEach((id) => {
        const el = document.getElementById(id);
        if (el) el.addEventListener('change', narrativeRefreshData);
    });

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

/** Build the shell once, then keep it in step with the app. */
function narrativeInit() {
    try {
        if (!narrativeBuildShell()) return;
        narrativeMoveNav();
        narrativeWire();
        narrativeSync();
    } catch (err) {
        // A rail that fails to build must not take the application with it.
        console.error('Narrative shell not built:', err);
    }
}

document.addEventListener('DOMContentLoaded', () => {
    narrativeInit();

    // setActiveTab writes the active class on these same nodes, on every
    // navigation and once when an upload succeeds, so watching them is how the
    // rail learns that a file has arrived without anything having to call it.
    const nav = document.getElementById('narrativeNav');
    if (nav) {
        const watch = new MutationObserver(narrativeSync);
        nav.querySelectorAll('.tab-button[data-tab]').forEach((btn) =>
            watch.observe(btn, { attributes: true, attributeFilter: ['class'] })
        );
    }

    // The forward buttons change without navigation: Continue to Modeling is
    // enabled once a task type is chosen, and Continue to Inference is created when
    // a run finishes. Watching the views for those changes keeps the proxy honest.
    const views = document.getElementById('narrativeViews');
    if (views) {
        new MutationObserver(narrativeRefreshNext).observe(views, {
            subtree: true,
            childList: true,
            attributes: true,
            attributeFilter: ['disabled', 'class'],
        });
    }

    // The Modeling view is revealed by a class change rather than by navigation.
    // The run strip is built the first time it is shown and stays with it.
    const host = document.getElementById('columnSelection');
    if (!host) return;
    const build = () => {
        if (host.classList.contains('hidden')) return;
        narrativeBuildFlow();
        narrativeSetStep('setup');
        narrativeSync();
    };
    build();
    new MutationObserver(build).observe(host, { attributes: true, attributeFilter: ['class'] });
});


// === inference.js ===
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
        const response = await fetch('/predict', {
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
                    <h4 style="margin: 0 0 8px 0;">Inference visuals (${escapeHtml(firstCol)})</h4>
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
            // Two graphics share one frame: the trained model's performance, and the
            // spread of every predictor in the training rows against the rows just
            // predicted (the check that the new data lies where the model was fitted).
            const spreadUrl = data.spread_visualization
                ? withApiRoot('/user-visualizations/' + data.spread_visualization) + '?v=' + encodeURIComponent(data.spread_visualization_version || Date.now())
                : '';
            const perfUrl = trainingViz ? (trainingPerfUrl || trainingVizUrl) : '';
            const graphicChoices = [
                spreadUrl ? { key: 'spread', label: 'Predictor spread: training rows against new rows', url: spreadUrl, alt: 'Predictor spread of training rows and new rows' } : null,
                perfUrl ? { key: 'performance', label: 'Model used (training performance)', url: perfUrl, alt: 'Training performance' } : null,
            ].filter(Boolean);
            const imgStyle = 'display: block; max-height: calc(100vh - var(--header-offset, 88px) - 360px); width: auto; max-width: 100%; object-fit: contain;';
            const modelGraphicSectionHtml = graphicChoices.length ? `
                <h3 style="margin: 0 0 8px 0;">Graphic</h3>
                <label for="inferenceGraphicSelect" class="sr-only">Select graphic</label>
                <select id="inferenceGraphicSelect" style="margin: 0 0 8px 0; max-width: 100%;">
                    ${graphicChoices.map(g => `<option value="${g.key}">${escapeHtml(g.label)}</option>`).join('')}
                </select>
                <img id="inferenceGraphic" src="${graphicChoices[0].url}" alt="${escapeHtml(graphicChoices[0].alt)}" class="inference-model-graphic-img" style="${imgStyle}">
            ` : '';
            const rightPanelHtml = (trainingSummaryTableHtml || modelGraphicSectionHtml) ? `
                <div class="inference-model-graphic" style="flex: 0 0 calc(50% - 12px); min-width: 200px; overflow: auto; align-self: flex-start;">
                    ${modelGraphicSectionHtml}
                    ${trainingSummaryTableHtml}
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
                    <button class="secondary-button" onclick="backToModel()">Back to Modeling</button>
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
            showApplicabilityLine(data.applicability_summary)
            showUncertaintyLine(data.uncertainty_summary)
            // The example rows were drawn from a known species each, so the page can
            // say how many came back the way they were drawn. Only for those rows:
            // a person's own file has no key to compare against.
            if (window.digiterraExampleRows) {
                window.digiterraExampleRows = false
                showExampleRowAgreement(preview)
            }
            const graphicSelect = document.getElementById('inferenceGraphicSelect')
            if (graphicSelect) {
                graphicSelect.addEventListener('change', () => {
                    const g = graphicChoices.find(x => x.key === graphicSelect.value)
                    const img = document.getElementById('inferenceGraphic')
                    if (g && img) { img.src = g.url; img.alt = g.alt }
                })
            }
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
                showError(errorDiv, 'DiGiTerra has no model configuration. Configure the data first.');
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

// Advanced Optimization forms (unified Modeling tab + legacy full-page panel) — same training flow as processForm
function wireAdvancedOptimizationFormSubmit(form) {
    if (!form) return;
    form.addEventListener('submit', async (e) => {
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
                showError(errorDiv, 'Select a model before you run with advanced options.');
                errorDiv.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }
            return;
        }

        if (!processForm) {
            const errorDiv = getCachedElement('errorDiv');
            if (errorDiv) {
                showError(errorDiv, 'DiGiTerra has no model configuration. Configure the data first.');
                errorDiv.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }
            return;
        }

        const submitter = e.submitter;
        const submitButton = (submitter && submitter.tagName === 'BUTTON' && submitter.getAttribute('type') === 'submit')
            ? submitter
            : form.querySelector('button[type="submit"]');
        if (submitButton) {
            submitButton.disabled = true;
            submitButton.textContent = 'Running...';
        }

        const submitEvent = new Event('submit', { bubbles: true, cancelable: true });
        processForm.dispatchEvent(submitEvent);

        setTimeout(() => {
            if (submitButton) {
                submitButton.disabled = false;
                submitButton.textContent = 'Run Model with Advanced Options';
            }
        }, 1000);
    });
}
wireAdvancedOptimizationFormSubmit(document.getElementById('advancedOptimizationForm'));
wireAdvancedOptimizationFormSubmit(document.getElementById('advancedOptimizationLegacyForm'));


// Bundle import is available where the server allows it (desktop builds). The
// form stays hidden until a probe of /import-bundle says the route accepts uploads.
(function wireBundleImport() {
    const form = document.getElementById('importBundleForm');
    if (!form) return;
    fetch('/import-bundle', { method: 'POST', body: new FormData() })
        .then(r => { if (r.status !== 403) form.classList.remove('hidden'); })
        .catch(() => {});
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const status = document.getElementById('importBundleStatus');
        status.textContent = 'Loading bundle...';
        try {
            const response = await fetch('/import-bundle', { method: 'POST', body: new FormData(form) });
            const data = await response.json();
            if (!response.ok) {
                status.textContent = data.error || 'DiGiTerra cannot load the bundle.';
                return;
            }
            const model = data.model || {};
            const features = Array.isArray(model.features) ? model.features.join(', ') : '';
            status.textContent = `Loaded ${model.name || 'model'} (${model.problem_type || ''}) trained ${data.created_at || ''}. Features: ${features}. ${data.note || ''}`;
        } catch (err) {
            status.textContent = 'DiGiTerra cannot load the bundle.';
        }
    });
})();


/* How many of the example rows came back as the species they were drawn from.

   penguins_new_key.csv holds one species per row_id, and predictions.csv keeps the
   order of the file it scored, so the two line up row for row. The count is a
   sanity check on the run rather than a score: the rows are synthetic. */
async function showExampleRowAgreement(preview) {
    try {
        const predictedKey = Object.keys(preview).find((k) => k.startsWith('Predicted_'))
        if (!predictedKey) return
        const predicted = preview[predictedKey]
        const response = await fetch(`${API_ROOT}/example-data/penguins_new_key.csv`)
        if (!response.ok) return
        const rows = (await response.text()).trim().split('\n').slice(1)
        const expected = rows.map((line) => line.split(',')[1].trim())
        if (!expected.length || expected.length !== predicted.length) return
        let agree = 0
        for (let i = 0; i < expected.length; i += 1) {
            if (String(predicted[i]).trim() === expected[i]) agree += 1
        }
        const host = document.querySelector('.inference-summary-section')
        if (!host) return
        const line = document.createElement('p')
        line.id = 'exampleRowAgreement'
        line.className = 'field-note'
        line.textContent = `${agree} of ${expected.length} rows match the species each row was drawn from.`
        host.appendChild(line)
    } catch (err) {
        console.error('Could not compare the example rows with the key:', err)
    }
}


/* One line saying how many of these predictions rest on data the model has seen.
   A model answers for any row whose columns parse, and nothing else on the page
   separates a prediction on familiar data from one on a row nothing like it. */
function showApplicabilityLine(summary) {
    if (!summary || !summary.rows) return
    const host = document.querySelector('.inference-summary-section')
    if (!host) return
    const line = document.createElement('p')
    line.id = 'applicabilitySummary'
    line.className = 'field-note'
    const outOfRange = summary.out_of_range === 1
        ? '1 row has a value outside the training range.'
        : `${summary.out_of_range} rows have a value outside the training range.`
    line.textContent = summary.novel === 0
        ? `No row scores as unlike the training data. ${outOfRange}`
        : `${summary.novel} of ${summary.rows} rows score as unlike the training data. ${outOfRange}`
    host.appendChild(line)
}


/* One line on how much to trust a single prediction: the interval's half-width for a
   regression run, or the least confident prediction for a classifier. */
function showUncertaintyLine(summary) {
    if (!summary) return
    const host = document.querySelector('.inference-summary-section')
    if (!host) return
    let text = ''
    if (summary.kind === 'interval' && summary.half_width !== null && summary.half_width !== undefined) {
        const units = summary.units ? ` ${summary.units}` : ''
        text = `${summary.coverage}% intervals are half-width ${summary.half_width.toPrecision(4)}${units}.`
    } else if (summary.kind === 'probability' && summary.lowest !== null && summary.lowest !== undefined) {
        text = `Lowest class probability among predictions is ${summary.lowest.toFixed(3)}.`
    }
    if (!text) return
    const line = document.createElement('p')
    line.id = 'uncertaintySummary'
    line.className = 'field-note'
    line.textContent = text
    host.appendChild(line)
}


// === app.js ===
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
