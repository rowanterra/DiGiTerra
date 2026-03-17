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
