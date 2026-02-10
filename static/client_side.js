<!DOCTYPE html>
<html lang="en">
<head>
    <!-- DiGiTerra UI: single-page app. Tabs = Upload, Data Exploration, Model Preprocessing, Modeling, Inference. Logic in static/client_side.js. HANDOFF.md has repo notes. -->
    <!--
    Accessibility Features:
    This application includes comprehensive accessibility support including ARIA attributes (roles, labels, live regions),
    keyboard navigation with skip links and focus management, screen reader announcements for dynamic content,
    accessible form labels and help text, and high contrast focus indicators. All interactive elements support
    keyboard navigation and provide appropriate feedback to assistive technologies. See docs/documentation.md
    for detailed accessibility documentation.
    -->
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DiGiTerra</title>
    <link rel="stylesheet" href="/static/style.css">
    <link rel="icon" type="image/png" href="/static/Terra_Axe_Logo.png">
    <script>
        // DARK MODE DISABLED - Code kept for potential future use
        // Initialize dark mode immediately to prevent flash
        // Default to light mode - only enable dark mode if explicitly set to 'true'
        /*
        (function() {
            const darkModeSetting = localStorage.getItem('darkMode');
            // Only enable dark mode if explicitly set to 'true', not if null/undefined/'false'
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
            if (document.documentElement) {
                document.documentElement.classList.remove('dark-mode');
            }
            if (document.body) {
                document.body.classList.remove('dark-mode');
            }
        })();
    </script>
</head>
<body>
    <a href="#main-content" class="skip-link">Skip to main content</a>
    <header class="header">
        <div class="left" id="headerLogo" style="cursor: pointer;" role="button" tabindex="0" aria-label="Return to welcome screen">
            <img src='/static/Terra_Axe_Logo.png' class="logo" alt="DiGiTerra logo">
            <h1 class="title">DiGiTerra</h1>
        </div>
        <nav id="appTabs" class="app-tabs hidden" role="navigation" aria-label="Main navigation">
            <button type="button" class="tab-button" data-tab="upload">Upload</button>
            <button type="button" class="tab-button" data-tab="processing">Data Exploration</button>
            <button type="button" class="tab-button" data-tab="model-preprocessing">Model Preprocessing</button>
            <button type="button" class="tab-button" data-tab="modeling">Modeling</button>
            <button type="button" class="tab-button" data-tab="historic">Inference on New Data</button>
        </nav>
        <div class="right">
            <!-- DARK MODE TOGGLE DISABLED - Code kept for potential future use -->
            <!--
            <button type="button" class="tab-button dark-mode-toggle" id="darkModeToggle" aria-label="Toggle dark mode" title="Toggle dark mode">
                <span id="darkModeIcon">Dark</span>
            </button>
            -->
            <button type="button" class="tab-button documentation-button" data-tab="documentation" id="documentationButton">Documentation</button>
        </div>
    </header>
<!-- Landing page with logos and disclaimer -->
    <main id="main-content">
    <div id="welcome" class="welcome">
        <div class="welcomesection">
            <h1 class="welcome-header">
                <span class="welcome-text">Welcome to </span>
                <br>
                <div style="display: flex; align-items: center; justify-content: center; gap: 10px; flex-wrap: nowrap;">
                    <img src="/static/Terra_Axe_Logo.png" class="logo logo-inline" alt="DiGiTerra logo">
                    <span class="welcome-text2">DiGiTerra</span>
                </div>
            </h1>
            <h4>A Machine Learning Exploration Tool</h4>
            <div style="width: 100%; max-width: 700px; margin: 0 auto;">
                <button class="welcomebutton" id='startModelingButton' aria-label="Start modeling process">Start Modeling</button>
            </div>
        </div>
        <div class="welcomesection logos">
            <img src='/static/ORISELogo.png' class="ORISE" alt="Oak Ridge Institute for Science and Education logo">
            <img src='/static/DOELogo.png' class="doe" alt="Department of Energy logo">
            <img src='/static/NETLLogo.png' class="bottomlogo" alt="National Energy Technology Laboratory logo">
        </div>
        <div class="welcomesection disclaimer">
            <div id="welcomeDisclaimerContent" style="padding: 12px; background-color: #fff3cd; border-left: 4px solid #ffc107; border-radius: 4px; margin: 0;">
                <p style="margin: 0; color: #856404; font-size: 0.9em; text-align: justify;">Disclaimer: This project was funded by the Department of Energy, National Energy Technology Laboratory an agency of the United States Government, through an appointment administered by the Oak Ridge Institute for Science and Education. Neither the United States Government nor any agency thereof, nor any of its employees, nor the support contractor, nor any of their employees, makes any warranty, express or implied, or assumes any legal liability or responsibility for the accuracy, completeness, or usefulness of any information, apparatus, product, or process disclosed, or represents that its use would not infringe privately owned rights. Reference herein to any specific commercial product, process, or service by trade name, trademark, manufacturer, or otherwise does not necessarily constitute or imply its endorsement, recommendation, or favoring by the United States Government or any agency thereof. The views and opinions of authors expressed herein do not necessarily state or reflect those of the United States Government or any agency thereof.</p>
            </div>
        </div>
    </div>
<!-- Documentation section (only visible when clicked in toolbar) -->
    <div id="documentation" class="documentation hidden">
        <div class="section-header">
            <h2>Documentation</h2>
        </div>
        
        <!-- About Section -->
        <div class="documentation-card doc-intro">
            <div class="doc-header">
                <h3>About DiGiTerra</h3>
            </div>
            <p>DiGiTerra is an interactive machine learning exploration tool designed to help researchers and analysts explore datasets, build predictive models, and generate insights through an intuitive web-based interface.</p>
            <p style="margin: 16px 0; color: #555;">DiGiTerra guides you through a repeatable pipeline:</p>
            <div class="doc-workflow-badges">
                <span class="doc-badge doc-workflow-step">1. Upload</span>
                <span class="doc-badge doc-workflow-arrow">→</span>
                <span class="doc-badge doc-workflow-step">2. Explore</span>
                <span class="doc-badge doc-workflow-arrow">→</span>
                <span class="doc-badge doc-workflow-step">3. Preprocess</span>
                <span class="doc-badge doc-workflow-arrow">→</span>
                <span class="doc-badge doc-workflow-step">4. Model</span>
                <span class="doc-badge doc-workflow-arrow">→</span>
                <span class="doc-badge doc-workflow-step">5. Review Results</span>
            </div>
            <div class="doc-warning-content" style="margin-top: 24px; padding: 16px; background-color: #fff3cd; border-left: 4px solid #ffc107; border-radius: 4px;">
                <p style="margin: 0 0 12px 0; font-weight: 600; color: #856404;">Disclaimer: This is an exploration tool, not a production system.</p>
                <p style="margin: 0 0 12px 0; color: #856404;">DiGiTerra is designed as an <strong>exploratory data analysis and modeling tool</strong> to help researchers investigate patterns and relationships in their data. <strong>We do not claim that results from this application will be reliable for decision-making purposes.</strong></p>
                <p style="margin: 0; color: #856404;"><strong>Users should consult with computational and statistics experts</strong> to verify findings, validate model assumptions, assess model performance in context, and ensure appropriate interpretation of results. This tool is intended to support, not replace, expert analysis and domain knowledge.</p>
            </div>
        </div>

        <!-- File Format Section -->
        <div class="documentation-card doc-fileformat">
            <div class="doc-header">
                <h3>File Format & Column Entry Tips</h3>
            </div>
            <div class="doc-tips-list">
                <div class="doc-tip-item">
                    <div>
                        <strong>CSV only</strong>
                        <p>DiGiTerra accepts CSV files. Lab exports, field data, assay results, or other tabular data can be exported from Excel via <em>File → Save As</em> as <code>.csv</code>.</p>
                    </div>
                </div>
                <div class="doc-tip-item">
                    <div>
                        <strong>Column letters</strong>
                        <p>When selecting columns, use spreadsheet-style letters (A, B, C...). Ranges are written as <code>A-D</code> and multiple selections can be separated with commas like <code>A-D, F, H</code>.</p>
                    </div>
                </div>
                <div class="doc-tip-item">
                    <div>
                        <strong>Targets vs indicators</strong>
                        <p>Indicators are inputs (e.g., geochemical or environmental measurements); targets are the outputs you want to predict (e.g., species, concentration, or sample class).</p>
                    </div>
                </div>
            </div>
        </div>

        <!-- Modeling Types Section -->
        <div class="documentation-card doc-modeling-types">
            <div class="doc-header">
                <h3>Modeling Types</h3>
            </div>
            <p class="doc-intro-text">DiGiTerra supports three types of machine learning tasks. Choose the type that matches your goal:</p>
            <div class="doc-modeling-types-grid">
                <div class="doc-modeling-type-card">
                    <div class="doc-approach-header">
                        <h4>Regression <span style="font-size: 0.75em; font-weight: 500; color: #666;">(33 models)</span></h4>
                    </div>
                    <p><strong>When to use:</strong> When you want to predict a continuous numeric value (e.g., concentration, temperature, yield, or other measurable outcome).</p>
                    <p><strong>What it does:</strong> Regression models learn the relationship between input features and a continuous target variable, allowing you to predict numeric outcomes for new data.</p>
                    <ul class="doc-feature-list">
                        <li>Predict numeric targets (e.g., pollutant concentration, crop yield, biomarker levels)</li>
                        <li>Evaluate with R², MSE, RMSE, MAE metrics</li>
                        <li>Visualize predictions vs actuals with scatter plots</li>
                        <li>Analyze residuals to assess model quality</li>
                    </ul>
                </div>
                <div class="doc-modeling-type-card">
                    <div class="doc-approach-header">
                        <h4>Classification <span style="font-size: 0.75em; font-weight: 500; color: #666;">(23 models)</span></h4>
                    </div>
                    <p><strong>When to use:</strong> When you want to predict a category or class label (e.g., species, sample type, or diagnostic class).</p>
                    <p><strong>What it does:</strong> Classification models learn to assign data points to predefined categories, making them ideal for categorical predictions.</p>
                    <ul class="doc-feature-list">
                        <li>Predict categorical targets (e.g., species, rock type, disease diagnosis)</li>
                        <li>Evaluate with accuracy, precision, recall, F1-score</li>
                        <li>ROC curves and confusion matrices</li>
                        <li>Support for binary and multi-class problems</li>
                    </ul>
                </div>
                <div class="doc-modeling-type-card">
                    <div class="doc-approach-header">
                        <h4>Clustering <span style="font-size: 0.75em; font-weight: 500; color: #666;">(12 models)</span></h4>
                    </div>
                    <p><strong>When to use:</strong> When you want to discover hidden patterns or group similar data points without predefined categories.</p>
                    <p><strong>What it does:</strong> Clustering models identify natural groupings in your data by finding similarities between data points, useful for exploratory analysis.</p>
                    <ul class="doc-feature-list">
                        <li>No target variable needed (unsupervised learning)</li>
                        <li>Evaluate with silhouette score, Calinski-Harabasz, Davies-Bouldin</li>
                        <li>Visualize cluster assignments with PCA plots</li>
                        <li>Discover hidden patterns in your data</li>
                    </ul>
                </div>
            </div>
            
            <!-- Types Not Supported -->
            <div class="doc-not-supported">
                <h4 class="doc-not-supported-header">Types Not Currently Supported</h4>
                <p class="doc-intro-text">DiGiTerra focuses on the three core machine learning types above. The following are not currently supported:</p>
                <ul class="doc-not-supported-list">
                    <li><strong>Time Series Forecasting:</strong> DiGiTerra does not include specialized time series models (e.g., ARIMA, LSTM for sequences). Use regression models for simple time-based predictions if needed.</li>
                    <li><strong>Reinforcement Learning:</strong> DiGiTerra does not support reinforcement learning algorithms that learn through interaction with an environment.</li>
                    <li><strong>Transformer Models (Deep Learning):</strong> DiGiTerra does not include Transformer model architectures (e.g., BERT, GPT, Vision Transformers). These are specialized deep learning models for NLP, computer vision, and other domains. <em>Note: DiGiTerra does support data transformers for preprocessing (OneHotEncoder, etc.) to encode categorical columns. This is different from Transformer model architectures.</em></li>
                    <li><strong>Other Deep Learning Architectures:</strong> While some models use neural networks (MLP), DiGiTerra does not include other deep learning architectures like CNNs or RNNs.</li>
                    <li><strong>Anomaly Detection:</strong> While some clustering models can identify outliers, DiGiTerra does not include specialized anomaly detection algorithms.</li>
                    <li><strong>Recommendation Systems:</strong> DiGiTerra does not include collaborative filtering or recommendation algorithms.</li>
                </ul>
            </div>
        </div>

        <!-- Modeling Approaches Section -->
        <div class="documentation-card doc-approaches">
            <div class="doc-header">
                <h3>Modeling Approaches</h3>
            </div>
            <p class="doc-intro-text">For Regression and Classification tasks, you can choose between three modeling approaches after configuring Model Preprocessing:</p>
            <div class="doc-approach-grid">
                <div class="doc-approach-card doc-automl">
                    <div class="doc-approach-header">
                        <h4>AutoML</h4>
                    </div>
                    <p>Fully automated modeling that lets the system optimize everything for you. Ideal for users who want the best results without manual configuration.</p>
                    <ul class="doc-feature-list">
                        <li>Automatic model selection (or specify one)</li>
                        <li>Automatic feature selection (RFE)</li>
                        <li>Automatic outlier detection (Isolation Forest)</li>
                        <li>Automatic hyperparameter optimization (Randomized Search)</li>
                        <li>Cross-validation included</li>
                    </ul>
                </div>
                <div class="doc-approach-card doc-simple">
                    <div class="doc-approach-header">
                        <h4>Simple Modeling</h4>
                    </div>
                    <p>Basic model selection and configuration. Ideal for quick experiments and straightforward modeling tasks.</p>
                    <ul class="doc-feature-list">
                        <li>Select a model</li>
                        <li>Configure essential hyperparameters</li>
                        <li>Run training with baseline evaluation</li>
                    </ul>
                </div>
                <div class="doc-approach-card doc-advanced">
                    <div class="doc-approach-header">
                        <h4>Advanced Modeling</h4>
                    </div>
                    <p>All Simple features plus advanced options for maximum control and optimization.</p>
                    <ul class="doc-feature-list">
                        <li>Feature selection</li>
                        <li>Outlier handling</li>
                        <li>Hyperparameter optimization</li>
                        <li>Cross-validation</li>
                    </ul>
                </div>
            </div>
            <p class="doc-note">All three approaches are fully independent. You can use any without configuring the others. Note: Clustering models use a simplified interface without the Simple/Advanced/AutoML distinction.</p>
        </div>

        <!-- Core Concepts Section -->
        <div class="documentation-card doc-concepts">
            <div class="doc-header">
                <h3>Core Concepts</h3>
            </div>
            <div class="doc-concepts-grid">
                <div class="doc-concept-item">
                    <strong>Indicators</strong>
                    <p>Input columns used to predict targets</p>
                </div>
                <div class="doc-concept-item">
                    <strong>Targets</strong>
                    <p>Output columns you want to predict or cluster</p>
                </div>
                <div class="doc-concept-item">
                    <strong>Transformers</strong>
                    <p>Encode non-numeric columns for modeling</p>
                </div>
                <div class="doc-concept-item">
                    <strong>Data Cleaning</strong>
                    <p>Handle missing values, zeros, and outliers to prepare data for modeling</p>
                </div>
                <div class="doc-concept-item">
                    <strong>Stratification</strong>
                    <p>Groups data for balanced splits or evaluation</p>
                </div>
            </div>
        </div>

        <!-- Advanced Options Section -->
        <div class="documentation-card doc-advanced-options">
            <div class="doc-header">
                <h3>Advanced Options</h3>
            </div>
            <p class="doc-intro-text">When using the <strong>Advanced Modeling</strong> approach, DiGiTerra provides several advanced features. <strong>Note:</strong> These features can significantly increase processing time, especially with large datasets.</p>
            
            <div class="doc-advanced-features">
                <div class="doc-feature-card">
                    <div>
                        <h4>Cross-Validation</h4>
                        <p>Multiple strategies including K-Fold, Stratified K-Fold, Repeated K-Fold, and Repeated Stratified K-Fold for robust model evaluation.</p>
                    </div>
                </div>
                <div class="doc-feature-card">
                    <div>
                        <h4>Data Preprocessing</h4>
                        <p>Robust scaling options, missing value imputation strategies (mean, median, drop), and zero-value handling.</p>
                    </div>
                </div>
                <div class="doc-feature-card">
                    <div>
                        <h4>Class Imbalance Handling</h4>
                        <p>Class weight adjustments for classification models to handle imbalanced datasets.</p>
                    </div>
                </div>
                <div class="doc-feature-card">
                    <div>
                        <h4>Model Explainability</h4>
                        <p>SHAP summaries and feature importance visualizations (model-based and permutation importance) to understand model predictions.</p>
                    </div>
                </div>
                <div class="doc-feature-card">
                    <div>
                        <h4>Calibration Curves</h4>
                        <p>For classification models, calibration curves help assess prediction probability reliability.</p>
                    </div>
                </div>
                <div class="doc-feature-card">
                    <div>
                        <h4>Ensemble Models</h4>
                        <p>Random Forest, Gradient Boosting, and Extra Trees models that combine multiple learners for improved performance.</p>
                    </div>
                </div>
            </div>

            <!-- Feature Selection Subsection -->
            <div class="doc-subsection">
                <h4 class="doc-subsection-header">
                    Feature Selection
                </h4>
                <p class="doc-subsection-intro">Feature selection helps identify and use only the most relevant features for modeling, which can improve model performance and reduce overfitting:</p>
                <div class="doc-method-grid">
                    <div class="doc-method-card">
                        <strong>Select K Best</strong>
                        <span class="doc-method-badge">Filter Method</span>
                        <p>Selects the K features with the highest statistical scores (F-test). Fast and effective for identifying features with strong relationships to the target.</p>
                    </div>
                    <div class="doc-method-card">
                        <strong>Recursive Feature Elimination (RFE)</strong>
                        <span class="doc-method-badge">Wrapper Method</span>
                        <p>Iteratively removes the least important features based on model performance. More computationally intensive but often finds better feature subsets.</p>
                    </div>
                    <div class="doc-method-card">
                        <strong>Select From Model</strong>
                        <span class="doc-method-badge">Model-based</span>
                        <p>Uses a model's built-in feature importance (e.g., Random Forest) to select features. Good balance between speed and effectiveness.</p>
                    </div>
                </div>
            </div>

            <!-- Outlier Handling Subsection -->
            <div class="doc-subsection">
                <h4 class="doc-subsection-header">
                    Outlier Handling
                </h4>
                <p class="doc-subsection-intro">Outlier detection and handling can improve model robustness by identifying and managing extreme values:</p>
                <div class="doc-method-grid">
                    <div class="doc-method-card">
                        <strong>Interquartile Range (IQR)</strong>
                        <p>Identifies outliers as values outside 1.5 × IQR from the quartiles. Simple and interpretable statistical method.</p>
                    </div>
                    <div class="doc-method-card">
                        <strong>Z-Score (3σ rule)</strong>
                        <p>Identifies outliers as values more than 3 standard deviations from the mean. Works well for normally distributed data.</p>
                    </div>
                    <div class="doc-method-card">
                        <strong>Isolation Forest</strong>
                        <p>Machine learning-based method that isolates outliers using random decision trees. Effective for complex, high-dimensional data.</p>
                    </div>
                    <div class="doc-method-card">
                        <strong>Local Outlier Factor (LOF)</strong>
                        <p>Density-based method that identifies outliers relative to their local neighborhood. Good for detecting outliers in clustered data.</p>
                    </div>
                </div>
                <div class="doc-action-note">
                    <strong>Actions:</strong> <span class="doc-action-badge">Remove</span> completely removes outliers from training data; <span class="doc-action-badge">Cap</span> limits extreme values to threshold boundaries while preserving data points.
                </div>
            </div>

            <!-- Hyperparameter Search Subsection -->
            <div class="doc-subsection">
                <h4 class="doc-subsection-header">
                    Hyperparameter Search
                </h4>
                <p class="doc-subsection-intro">Automatically finds optimal hyperparameter values instead of using manual settings:</p>
                <div class="doc-method-grid">
                    <div class="doc-method-card">
                        <strong>Grid Search</strong>
                        <span class="doc-method-badge doc-badge-slow">Slowest</span>
                        <p>Exhaustively searches all combinations of hyperparameters in a predefined grid. Most thorough but computationally expensive. Best for small parameter spaces.</p>
                    </div>
                    <div class="doc-method-card">
                        <strong>Randomized Search</strong>
                        <span class="doc-method-badge doc-badge-medium">Faster</span>
                        <p>Randomly samples hyperparameter combinations from the grid. Faster than grid search and often finds good solutions with fewer evaluations. Good for large parameter spaces.</p>
                    </div>
                    <div class="doc-method-card">
                        <strong>Bayesian Optimization</strong>
                        <span class="doc-method-badge doc-badge-fast">Most Efficient</span>
                        <p>Uses probabilistic models to intelligently select the next hyperparameters to try. Most efficient for expensive evaluations but requires additional dependencies.</p>
                    </div>
                </div>
                <div class="doc-warning-box">
                    <strong>Warning:</strong> Hyperparameter search can take hours or days depending on the model, dataset size, and number of CV folds. Grid search is the slowest; randomized search is faster but less thorough.
                </div>
            </div>
        </div>

    </div>
<!-- File upload when user clicks 'Start Modeling' -->
    <div id="fileuploaddiv" class="uploadform hidden">
        <div class="upload-card">
            <div class="RedoButton">
                <h2 id="uploadHeader">Upload Data from CSV</h2>
                <div id="redobutton" class="hidden">
                    <button class="secondary-button" onclick="openResetPopup()">Restart with New Dataset</button>
                    <button class="success-button" type="button" onclick="moveToModelPreprocess()">Move Forward to Model Preprocessing</button>
                </div>
            </div>
            <form id="uploadForm" enctype="multipart/form-data">
                <input type="file" name="file" id="file" accept=".csv" required>

                <div><br></div>
                <div id="uploadButton">
                    <button type="submit">Upload</button>
                </div>
            </form>
        </div>
<!-- The preprocessing form is below -->
        <div id="columnsection" class="columnSection hidden">
            <div class="processing-layout">
                <div class="exploration-panel">
                    <div class="exploration-card">
                        <h2>Columns in Uploaded CSV</h2>

                        <div class="column-list" id="columnList">
                            <!-- Column headers will be displayed here -->
                        </div>
                    </div>

                    <!-- Correleation Matrices Section -->
                    <form id="corrForm">
                        <div class="exploration-card">
                            <h2>Data to be Explored</h2>
                            <div class="selectcolumns">
                                <label for ="corrCols">Columns for data exploration</label>
                                <input type="text" id="corrCols" name="corrCols" placeholder="Ex: A-D, Y, Z"><br>
                                <div><br></div>  
                            </div>
                        </div>

                        <div class="exploration-card">
                            <h2>Optional Exploration Preprocessing</h2>
                            <!-- Preprocessing for missing and zero values Section  -->
                            <div class="scaling-container">
                                <label for="exploreDropMissing">Remove or replace missing values from columns?</label>
                                    <select name="exploreDropMissing" id="exploreDropMissing">
                                        <option value="none">No Columns</option>
                                        <option value="indicator">Selected Columns</option>
                                        <option value="all">All</option>
                                    </select>
                            </div>
                            <br>
                            <div class="scaling-container hidden" id="exploreImputeDiv">
                                <label for="exploreImputeStrategy">How should missing value be replaced?</label>
                                <select name="exploreImputeStrategy" id="exploreImputeStrategy">
                                    <option value="none">Drop the row</option>
                                    <option value="mean">Mean</option>
                                    <option value="median">Median</option>
                                    <!-- <option value="knn">KNN</option> -->
                                    <option value="0">0</option>
                                    <option value="0.01">0.01</option>
                                </select>
                            </div>
                            <br>
                            <div class="scaling-container">
                                <label for="exploreDrop0">Remove zeros from columns?</label>
                                    <select name="exploreDrop0" id="exploreDrop0">
                                        <option value="none">No Columns</option>
                                        <option value="indicator">Selected Columns</option>
                                        <option value="all">All</option>
                                    </select>
                            </div>
                        </div>

                        <div class="exploration-card">
                            <h2>Evaluate and Download</h2>
                            <p style="margin-bottom: 15px; color: #666; font-size: 0.9em;">Non-numeric data will be ignored in data exploration.</p>
                            <div class="corr-actions">
                                <button type="submit">Evaluate</button>
                                <div id="dataExploration" class="corr-downloads"></div>
                            </div>
                        </div>
                    </form>
                </div>

                <div class="exploration-results">
                    <div class="results-header">
                        <h2>Exploration Results</h2>
                        <p>Charts, tables, and downloads will appear here.</p>
                    </div>
                    <div id="explorationOutput"></div>
                </div>
            </div>
        </div>

    </div>   
<!-- User Selects targets, indicators, output type, transformers, stratify, etc then clicks process button -->
    <div id="userInputSection" class="userInputSection hidden">
        <div class="section-header">
            <div class="section-header-content">
                <h2>Model Preprocessing</h2>
                <button type="button" class="secondary-button" id="backToExploration">Back to Data Exploration</button>
            </div>
        </div>
        <form id="preprocessform">
            <div class="model-preprocess-layout">
                <div class="preprocess-column">
                    <div class="preprocess-card">
                        <h2>Essential Inputs</h2>
                        
                        <h3 style="margin-top: 24px; margin-bottom: 12px; padding-top: 16px; border-top: 1px solid #e5e5e5; font-size: 1.1em; font-weight: 600; color: #000000;">Model Configuration</h3>
                        <div class="selectcolumns">
                            <label for="outputType1">Modeling Task Type<span class="requiredAsterisk">*</span></label>
                            <select name="outputType1" id="outputType1" required>
                                <option value="" disabled selected>-- Select an option --</option>
                                <option value="Numeric">Regression</option>
                                <option value="Classifier">Classification</option>
                                <option value="Cluster">Cluster</option>
                            </select>
                            <div><br></div>
                            <label for ="indicators">Indicator Columns <span class="requiredAsterisk" aria-label="required">*</span></label>
                            <input type="text" id="indicators" name="indicators" placeholder="Ex: A-D" required aria-required="true" aria-describedby="indicators-help">
                            <span id="indicators-help" class="sr-only">Enter column letters separated by commas, e.g., A-C,E,G</span>
                            <div><br></div>
                            <div class="target-input-row">
                                <label for ="predictors">Target Columns <span class="requiredAsterisk" aria-label="required">*</span></label>
                                <input type="text" id="predictors" name="predictors" placeholder="Ex: A" aria-required="true" aria-describedby="predictors-help">
                                <span id="predictors-help" class="sr-only">Enter column letters for target variables</span>
                                <span id="clusterTargetMessage" class="hidden cluster-target-message">Cluster models do not have targets.</span>
                            </div>
                            <p class="field-note"><i>Columns not chosen will be ignored.</i></p>
                            <p class="field-note"><i>Non-numeric columns must be transformed below.</i></p>
                        </div>

                        <h3 style="margin-top: 24px; margin-bottom: 12px; padding-top: 16px; border-top: 1px solid #e5e5e5; font-size: 1.1em; font-weight: 600; color: #000000;">Data Cleaning</h3>
                        <!-- Preprocessing for missing and zero values Section  -->
                        <div class="scaling-container">
                            <label for="dropMissing">Handle missing values?</label>
                                <select name="dropMissing" id="dropMissing">
                                    <option value="none">No Columns</option>
                                    <option value="indicatorAndTarget">Indicator and Target</option>
                                    <option value="indicator">Indicator</option>
                                    <option value="target">Target</option>
                                    <option value="all">All</option>
                                </select>
                        </div>
                        <br>
                        <div class="scaling-container hidden" id="imputeDiv">
                            <label for="imputeStrategy">How should missing value be replaced?</label>
                            <select name="imputeStrategy" id="imputeStrategy">
                                <option value="none">Drop the row</option>
                                <option value="mean">Mean</option>
                                <option value="median">Median</option>
                                <!-- <option value="knn">KNN</option> -->
                                <option value="0">0</option>
                                <option value="0.01">0.01</option>
                            </select>
                        </div>
                        <br>
                        <div class="scaling-container">
                            <label for="drop0">Remove zeros from columns?</label>
                                <select name="drop0" id="drop0">
                                    <option value="none">No Columns</option>
                                    <option value="indicatorAndTarget">Inidicator and Target</option>
                                    <option value="indicator">Indicator</option>
                                    <option value="target">Target</option>
                                    <option value="all">All</option>
                                </select>
                        </div>

                        <h3 style="margin-top: 24px; margin-bottom: 12px; padding-top: 16px; border-top: 1px solid #e5e5e5; font-size: 1.1em; font-weight: 600; color: #000000;">Transformer</h3>
                        <!-- Select if need to transform columns section-->
                        <div class="scaling-container">
                            <label for="useTransformer">Are any columns non-numeric?</label>
                                <select name="useTransformer" id="useTransformer">
                                    <option value="No">No</option>
                                    <option value="Yes">Yes</option>
                                </select>
                        </div>
                        <br>
                        <div id="transformerYes" class="hidden">
                            <div class="scaling-container">
                                <label for="transformerColumn">Which column(s)</label>
                                <div style="display: flex; gap: 8px; align-items: center;">
                                    <input type="text" id="transformerColumn" name="transformerColumn" placeholder="Ex: A-D" style="flex: 1;">
                                    <button type="button" id="autoDetectTransformers" class="button-secondary" style="white-space: nowrap; padding: 8px 16px; font-size: 0.9em;">Auto-detect</button>
                                </div>
                                <small style="display: block; margin-top: 4px; color: #666; font-size: 0.85em;">Detects non-numeric columns and low-cardinality numeric columns from your indicators</small>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="preprocess-column">
                    <div class="preprocess-card">
                        <!-- Select other options section -->
                        <h2>Optional Modifications</h2>
                        <!-- Stratifying Section  -->
                        <h3 class="preprocess-section-title">Stratify</h3>
                        <div id="scalingQuestion">
                            <label for="scalingYesNo">Stratify the model's splitting by a specific variable?</label>
                            <select name="scalingYesNo" id="scalingYesNo">
                                <option value="No">No</option>
                                <option value="Yes">Yes</option>
                            </select>
                        </div>
                        <br>

                        <div id="scalingYes" class="hidden stratify-options"> 
                            <div class="scaling-container">
                                <label for="specificVariableSelect">Which column to stratify by?</label>
                                <input type="text" id="specificVariableSelect" name="specificVariableSelect" placeholder="Ex: A">
                            </div>
                            <div><br></div>
                            <div class="scaling-container">
                                <label for="quantileBins">Quantiles or Bins?</label>
                                <select name="quantileBins" id="quantileBins" required>
                                    <option value="None">Neither</option>
                                    <option value="quantiles">Quantiles</option>
                                    <option value="Bins">Bins</option>
                                </select>
                            </div>


                            <div id="quantileInput" class="hidden">
                                <br>
                                <label for="quantiles">Quantile Number (2-20)</label>
                                <input type="number" min=2 max=20 id="quantiles" name="quantiles" placeholder="Ex: 4">
                            </div>

                            <div id="binInput" class="hidden">
                                <br>
                                <label for="bins">Bin thresholds (comma-separated, use inf for infinity)</label>
                                <input type="text" id="bins" name="bins" placeholder="0,49,99,199,299,499,999,inf">
                                <div><br></div>
                                <label for="binsLabel">Bin labels (comma-separated, text OK)</label>
                                <input type="text" id="binsLabel" name="binsLabel" placeholder="First Bins,25-99,100-199,200-299,Over 300">
                            </div>

                        </div>

                        <h3 class="preprocess-section-title">Scaling</h3>
                        <div class="scaling-container">
                            <label for="scaler">Scaling</label>
                            <select name="scaler" id="scaler">
                                <option value="standard">Standard</option>
                                <option value="quantile">Quantile</option>
                                <option value="robust">Robust</option>
                                <option value="none">None</option>
                            </select>
                        </div>

                        <h3 class="preprocess-section-title">Seed</h3>
                        <div class="scaling-container">
                            <label for="seedValue">Select Seed</label>
                            <input type="number" id="seedValue" name="seedValue" placeholder="Ex: 33">
                            <p class="field-note"><i>Leave blank for random.</i></p>
                        </div>

                        <h3 class="preprocess-section-title">Train/Test Split</h3>
                        <div class="split-row">
                            <div class="scaling-container">
                                <label for="trainSize">Train Size</label>
                                <input type="number" value="0.80" min="0" max="1" step="0.01" id="trainSize" name="trainSize" placeholder="0.80" inputmode="decimal">
                            </div>
                            <div class="scaling-container">
                                <label for="testSize">Test Size</label>
                                <input type="number" value="0.20" min="0" max="1" step="0.01" id="testSize" name="testSize" placeholder="0.20" inputmode="decimal">
                            </div>
                        </div>
                    </div>

                </div>

                <div class="preprocess-column">
                    <div class="preprocess-card">
                        <h2>Continue to Modeling</h2>
                        <p class="field-note" style="margin-bottom: 16px;">
                            Configure your modeling settings, then select "Process" below to proceed and choose between Automatic, Simple, and Advanced Modeling.
                        </p>
                        <button class="processButton processButton--compact success-button" type="submit" id="continueToModelingButton">Process Configurations and Continue</button>
                        <br>
                        <p><span class="requiredAsterisk">*</span>Required field</p>

                        <div id="stratErrorDiv"></div>
                    </div>

                </div>
            </div>
        </form>
    </div>

    
<!-- Unified Modeling Page -->
<div class="container">
    <div class="form hidden" id="columnSelection" data-ready="false">
        <div class="section-header">
            <div class="section-header-content">
                <h2>Modeling</h2>
                <button type="button" class="secondary-button" id="backToModelPreprocess">Back to Model Preprocessing</button>
                <div id="modelingHeaderActions"></div>
            </div>
        </div>
        
        <!-- Modeling Mode Selector -->
        <div class="preprocess-card" style="margin: 20px 0; max-width: 100%;">
            <h2 style="margin-top: 0; margin-bottom: 16px;">Select Modeling Mode</h2>
            <p class="field-note" style="margin-bottom: 16px;">Choose your modeling approach. Each mode provides different levels of control and automation.</p>
            <div class="scaling-container">
                <div style="display: flex; gap: 20px; flex-wrap: wrap; align-items: stretch; width: 100%;">
                    <label style="display: flex; align-items: center; gap: 8px; cursor: pointer; padding: 12px; border: 2px solid #e0e0e0; border-radius: 8px; flex: 1; min-width: 250px; transition: all 0.2s;">
                        <input type="radio" name="modelingMode" id="automlMode" value="automl" style="margin: 0;">
                        <div>
                            <strong>AutoML</strong>
                            <p style="margin: 4px 0 0 0; font-size: 0.9em; color: #666;">Fully automated - let the system optimize everything for you</p>
                        </div>
                    </label>
                    <label style="display: flex; align-items: center; gap: 8px; cursor: pointer; padding: 12px; border: 2px solid #e0e0e0; border-radius: 8px; flex: 1; min-width: 250px; transition: all 0.2s;">
                        <input type="radio" name="modelingMode" id="simpleMode" value="simple" checked style="margin: 0;">
                        <div>
                            <strong>Simple Modeling</strong>
                            <p style="margin: 4px 0 0 0; font-size: 0.9em; color: #666;">Basic model selection and configuration</p>
                        </div>
                    </label>
                    <label style="display: flex; align-items: center; gap: 8px; cursor: pointer; padding: 12px; border: 2px solid #e0e0e0; border-radius: 8px; flex: 1.2; min-width: 280px; transition: all 0.2s;">
                        <input type="radio" name="modelingMode" id="advancedMode" value="advanced" style="margin: 0;">
                        <div>
                            <strong>Advanced Modeling</strong>
                            <p style="margin: 4px 0 0 0; font-size: 0.9em; color: #666;">Full control of model selection, hyperparameters, and advanced features</p>
                        </div>
                    </label>
                </div>
            </div>
        </div>
        
        <!-- Model Selection in Header (will be moved into each mode section) -->
        <div class="preprocess-card" style="margin: 20px 0; max-width: 600px; display: none;" id="headerModelSelection">
            <h3 style="margin-top: 0; margin-bottom: 12px;">Select Model</h3>
            <p class="field-note" style="margin-bottom: 16px;">Choose the model algorithm for your selected output type.</p>
            
            <!-- Simple Mode Model Selection -->
            <div id="headerNumericModels" class="hidden">
                <label for="headerNModels" style="display: block; margin-bottom: 8px; font-weight: 600;">Regression Model:</label>
                <select name="headerNModels" id="headerNModels" style="width: 100%; max-width: 400px; padding: 8px; border: 1px solid #ccc; border-radius: 4px;">
                    <option value="" disabled selected>-- Select an option --</option>
                    <optgroup label="Most Common">
                        <option value="Linear">Linear</option>
                        <option value="Ridge">Ridge</option>
                        <option value="Lasso">Lasso</option>
                        <option value="ElasticNet">Elastic Net</option>
                        <option value="RF">Random Forest</option>
                        <option value="gradient_boosting" data-requires-multi-output="true">Gradient Boosting</option>
                        <option value="SVM" data-requires-multi-output="true">Support Vector Machine (SVR)</option>
                        <option value="MLP">Multi-Layer Perceptron</option>
                        <option value="K-Nearest">K-Nearest Neighbors</option>
                        <option value="ExtraTrees">Extra Trees</option>
                    </optgroup>
                    <optgroup label="Additional Models (Alphabetical)">
                        <option value="AdaBoost">AdaBoost Regressor</option>
                        <option value="ARDRegression">ARD Regression</option>
                        <option value="Bagging">Bagging Regressor</option>
                        <option value="BayesianRidge">Bayesian Ridge</option>
                        <option value="DecisionTree">Decision Tree Regressor</option>
                        <option value="ElasticNetCV">Elastic Net CV</option>
                        <option value="HistGradientBoosting">Histogram Gradient Boosting</option>
                        <option value="Huber">Huber Regressor</option>
                        <option value="LARS">LARS</option>
                        <option value="LARSCV">LARS CV</option>
                        <option value="LassoCV">Lasso CV</option>
                        <option value="LassoLars">LassoLars</option>
                        <option value="LinearSVR" data-requires-multi-output="true">Linear SVR</option>
                        <option value="NuSVR" data-requires-multi-output="true">Nu-SVR</option>
                        <option value="OMP">Orthogonal Matching Pursuit</option>
                        <option value="PassiveAggressive">Passive Aggressive Regressor</option>
                        <option value="Quantile">Quantile Regressor</option>
                        <option value="RadiusNeighbors">Radius Neighbors Regressor</option>
                        <option value="RANSAC">RANSAC Regressor</option>
                        <option value="RidgeCV">Ridge CV</option>
                        <option value="SGD">SGD Regressor</option>
                        <option value="TheilSen">Theil-Sen Regressor</option>
                    </optgroup>
                </select>
            </div>
            
            <div id="headerClusterModels" class="hidden">
                <label for="headerClModels" style="display: block; margin-bottom: 8px; font-weight: 600;">Clustering Model:</label>
                <select name="headerClModels" id="headerClModels" style="width: 100%; max-width: 400px; padding: 8px; border: 1px solid #ccc; border-radius: 4px;">
                    <option value="" disabled selected>-- Select an option --</option>
                    <optgroup label="Most Common">
                        <option value="kmeans">K-Means</option>
                        <option value="dbscan">DBSCAN</option>
                        <option value="agglo">Agglomerative</option>
                        <option value="gmm">Gaussian Mixture</option>
                        <option value="spectral">Spectral Clustering</option>
                        <option value="birch">BIRCH</option>
                        <option value="affinity_propagation">Affinity Propagation</option>
                        <option value="bisecting_kmeans">Bisecting K-Means</option>
                        <option value="hdbscan">HDBSCAN</option>
                        <option value="meanshift">Mean Shift</option>
                    </optgroup>
                    <optgroup label="Additional Models (Alphabetical)">
                        <option value="minibatch_kmeans">Mini-Batch K-Means</option>
                        <option value="optics">OPTICS</option>
                    </optgroup>
                </select>
            </div>
            
            <div id="headerClassifierModels" class="hidden">
                <label for="headerClassModels" style="display: block; margin-bottom: 8px; font-weight: 600;">Classification Model:</label>
                <select name="headerClassModels" id="headerClassModels" style="width: 100%; max-width: 400px; padding: 8px; border: 1px solid #ccc; border-radius: 4px;">
                    <option value="" disabled selected>-- Select an option --</option>
                    <optgroup label="Most Common">
                        <option value="Logistic_classifier">Logistic Classifier</option>
                        <option value="RF_classifier">Random Forest Classifier</option>
                        <option value="SVC_classifier">SVC Classifier</option>
                        <option value="MLP_classifier">MLP Classifier</option>
                        <option value="ExtraTrees_classifier">Extra Trees Classifier</option>
                        <option value="SGD_classifier">SGD Classifier</option>
                        <option value="AdaBoost_classifier">AdaBoost Classifier</option>
                        <option value="GradientBoosting_classifier">Gradient Boosting Classifier</option>
                        <option value="KNeighbors_classifier">K-Neighbors Classifier</option>
                    </optgroup>
                    <optgroup label="Multi-Output Models (Multiple Targets)">
                        <option value="GaussianNB_classifier">Gaussian Naive Bayes</option>
                        <option value="BernoulliNB_classifier">Bernoulli Naive Bayes</option>
                        <option value="CategoricalNB_classifier">Categorical Naive Bayes</option>
                        <option value="ComplementNB_classifier">Complement Naive Bayes</option>
                        <option value="MultinomialNB_classifier">Multinomial Naive Bayes</option>
                    </optgroup>
                    <optgroup label="Additional Models (Alphabetical)">
                        <option value="Bagging_classifier">Bagging Classifier</option>
                        <option value="DecisionTree_classifier">Decision Tree Classifier</option>
                        <option value="HistGradientBoosting_classifier">Histogram Gradient Boosting</option>
                        <option value="LDA_classifier">Linear Discriminant Analysis</option>
                        <option value="LinearSVC_classifier">Linear SVC</option>
                        <option value="NuSVC_classifier">Nu-SVC</option>
                        <option value="PassiveAggressive_classifier">Passive Aggressive Classifier</option>
                        <option value="QDA_classifier">Quadratic Discriminant Analysis</option>
                        <option value="Ridge_classifier">Ridge Classifier</option>
                    </optgroup>
                </select>
            </div>
            
            <!-- Advanced Mode Model Selection -->
            <div id="headerAdvancedNumericModels" class="hidden">
                <label for="headerAdvancedNModels" style="display: block; margin-bottom: 8px; font-weight: 600;">Regression Model:</label>
                <select name="headerAdvancedNModels" id="headerAdvancedNModels" style="width: 100%; max-width: 400px; padding: 8px; border: 1px solid #ccc; border-radius: 4px;">
                    <option value="" disabled selected>-- Select an option --</option>
                    <optgroup label="Most Common">
                        <option value="Linear">Linear</option>
                        <option value="Ridge">Ridge</option>
                        <option value="Lasso">Lasso</option>
                        <option value="ElasticNet">Elastic Net</option>
                        <option value="RF">Random Forest</option>
                        <option value="gradient_boosting" data-requires-multi-output="true">Gradient Boosting</option>
                        <option value="SVM" data-requires-multi-output="true">Support Vector Machine (SVR)</option>
                        <option value="MLP">Multi-Layer Perceptron</option>
                        <option value="K-Nearest">K-Nearest Neighbors</option>
                        <option value="ExtraTrees">Extra Trees</option>
                    </optgroup>
                    <optgroup label="Additional Models (Alphabetical)">
                        <option value="AdaBoost">AdaBoost Regressor</option>
                        <option value="ARDRegression">ARD Regression</option>
                        <option value="Bagging">Bagging Regressor</option>
                        <option value="BayesianRidge">Bayesian Ridge</option>
                        <option value="DecisionTree">Decision Tree Regressor</option>
                        <option value="ElasticNetCV">Elastic Net CV</option>
                        <option value="HistGradientBoosting">Histogram Gradient Boosting</option>
                        <option value="Huber">Huber Regressor</option>
                        <option value="LARS">LARS</option>
                        <option value="LARSCV">LARS CV</option>
                        <option value="LassoCV">Lasso CV</option>
                        <option value="LassoLars">LassoLars</option>
                        <option value="LinearSVR" data-requires-multi-output="true">Linear SVR</option>
                        <option value="NuSVR" data-requires-multi-output="true">Nu-SVR</option>
                        <option value="OMP">Orthogonal Matching Pursuit</option>
                        <option value="PassiveAggressive">Passive Aggressive Regressor</option>
                        <option value="Quantile">Quantile Regressor</option>
                        <option value="RadiusNeighbors">Radius Neighbors Regressor</option>
                        <option value="RANSAC">RANSAC Regressor</option>
                        <option value="RidgeCV">Ridge CV</option>
                        <option value="SGD">SGD Regressor</option>
                        <option value="TheilSen">Theil-Sen Regressor</option>
                    </optgroup>
                </select>
            </div>
            
            <div id="headerAdvancedClusterModels" class="hidden">
                <label for="headerAdvancedClModels" style="display: block; margin-bottom: 8px; font-weight: 600;">Clustering Model:</label>
                <select name="headerAdvancedClModels" id="headerAdvancedClModels" style="width: 100%; max-width: 400px; padding: 8px; border: 1px solid #ccc; border-radius: 4px;">
                    <option value="" disabled selected>-- Select an option --</option>
                    <optgroup label="Most Common">
                        <option value="kmeans">K-Means</option>
                        <option value="dbscan">DBSCAN</option>
                        <option value="agglo">Agglomerative</option>
                        <option value="gmm">Gaussian Mixture</option>
                        <option value="spectral">Spectral Clustering</option>
                        <option value="birch">BIRCH</option>
                        <option value="affinity_propagation">Affinity Propagation</option>
                        <option value="bisecting_kmeans">Bisecting K-Means</option>
                        <option value="hdbscan">HDBSCAN</option>
                        <option value="meanshift">Mean Shift</option>
                    </optgroup>
                    <optgroup label="Additional Models (Alphabetical)">
                        <option value="minibatch_kmeans">Mini-Batch K-Means</option>
                        <option value="optics">OPTICS</option>
                    </optgroup>
                </select>
            </div>
            
            <div id="headerAdvancedClassifierModels" class="hidden">
                <label for="headerAdvancedClassModels" style="display: block; margin-bottom: 8px; font-weight: 600;">Classification Model:</label>
                <select name="headerAdvancedClassModels" id="headerAdvancedClassModels" style="width: 100%; max-width: 400px; padding: 8px; border: 1px solid #ccc; border-radius: 4px;">
                    <option value="" disabled selected>-- Select an option --</option>
                    <optgroup label="Most Common">
                        <option value="Logistic_classifier">Logistic Classifier</option>
                        <option value="RF_classifier">Random Forest Classifier</option>
                        <option value="SVC_classifier">SVC Classifier</option>
                        <option value="MLP_classifier">MLP Classifier</option>
                        <option value="ExtraTrees_classifier">Extra Trees Classifier</option>
                        <option value="SGD_classifier">SGD Classifier</option>
                        <option value="AdaBoost_classifier">AdaBoost Classifier</option>
                        <option value="GradientBoosting_classifier">Gradient Boosting Classifier</option>
                        <option value="KNeighbors_classifier">K-Neighbors Classifier</option>
                    </optgroup>
                    <optgroup label="Multi-Output Models (Multiple Targets)">
                        <option value="GaussianNB_classifier">Gaussian Naive Bayes</option>
                        <option value="BernoulliNB_classifier">Bernoulli Naive Bayes</option>
                        <option value="CategoricalNB_classifier">Categorical Naive Bayes</option>
                        <option value="ComplementNB_classifier">Complement Naive Bayes</option>
                        <option value="MultinomialNB_classifier">Multinomial Naive Bayes</option>
                    </optgroup>
                    <optgroup label="Additional Models (Alphabetical)">
                        <option value="Bagging_classifier">Bagging Classifier</option>
                        <option value="DecisionTree_classifier">Decision Tree Classifier</option>
                        <option value="HistGradientBoosting_classifier">Histogram Gradient Boosting</option>
                        <option value="LDA_classifier">Linear Discriminant Analysis</option>
                        <option value="LinearSVC_classifier">Linear SVC</option>
                        <option value="NuSVC_classifier">Nu-SVC</option>
                        <option value="PassiveAggressive_classifier">Passive Aggressive Classifier</option>
                        <option value="QDA_classifier">Quadratic Discriminant Analysis</option>
                        <option value="Ridge_classifier">Ridge Classifier</option>
                    </optgroup>
                </select>
            </div>
            
            <!-- AutoML Mode Model Selection -->
            <div id="headerAutomlNumericModels" class="hidden">
                <label for="headerAutomlNModels" style="display: block; margin-bottom: 8px; font-weight: 600;">Regression Model (Optional - AutoML will try multiple if not specified):</label>
                <select name="headerAutomlNModels" id="headerAutomlNModels" style="width: 100%; max-width: 400px; padding: 8px; border: 1px solid #ccc; border-radius: 4px;">
                    <option value="" selected>Let AutoML choose</option>
                    <optgroup label="Most Common">
                        <option value="Linear">Linear</option>
                        <option value="Ridge">Ridge</option>
                        <option value="Lasso">Lasso</option>
                        <option value="ElasticNet">Elastic Net</option>
                        <option value="RF">Random Forest</option>
                        <option value="gradient_boosting" data-requires-multi-output="true">Gradient Boosting</option>
                        <option value="SVM" data-requires-multi-output="true">Support Vector Machine (SVR)</option>
                        <option value="MLP">Multi-Layer Perceptron</option>
                        <option value="K-Nearest">K-Nearest Neighbors</option>
                        <option value="ExtraTrees">Extra Trees</option>
                    </optgroup>
                    <optgroup label="Additional Models (Alphabetical)">
                        <option value="AdaBoost">AdaBoost Regressor</option>
                        <option value="ARDRegression">ARD Regression</option>
                        <option value="Bagging">Bagging Regressor</option>
                        <option value="BayesianRidge">Bayesian Ridge</option>
                        <option value="DecisionTree">Decision Tree Regressor</option>
                        <option value="ElasticNetCV">Elastic Net CV</option>
                        <option value="HistGradientBoosting">Histogram Gradient Boosting</option>
                        <option value="Huber">Huber Regressor</option>
                        <option value="LARS">LARS</option>
                        <option value="LARSCV">LARS CV</option>
                        <option value="LassoCV">Lasso CV</option>
                        <option value="LassoLars">LassoLars</option>
                        <option value="LinearSVR" data-requires-multi-output="true">Linear SVR</option>
                        <option value="NuSVR" data-requires-multi-output="true">Nu-SVR</option>
                        <option value="OMP">Orthogonal Matching Pursuit</option>
                        <option value="PassiveAggressive">Passive Aggressive Regressor</option>
                        <option value="Quantile">Quantile Regressor</option>
                        <option value="RadiusNeighbors">Radius Neighbors Regressor</option>
                        <option value="RANSAC">RANSAC Regressor</option>
                        <option value="RidgeCV">Ridge CV</option>
                        <option value="SGD">SGD Regressor</option>
                        <option value="TheilSen">Theil-Sen Regressor</option>
                    </optgroup>
                </select>
            </div>
            
            <div id="headerAutomlClusterModels" class="hidden">
                <label for="headerAutomlClModels" style="display: block; margin-bottom: 8px; font-weight: 600;">Clustering Model (Optional):</label>
                <select name="headerAutomlClModels" id="headerAutomlClModels" style="width: 100%; max-width: 400px; padding: 8px; border: 1px solid #ccc; border-radius: 4px;">
                    <option value="" selected>Let AutoML choose</option>
                    <optgroup label="Most Common">
                        <option value="kmeans">K-Means</option>
                        <option value="dbscan">DBSCAN</option>
                        <option value="agglo">Agglomerative</option>
                        <option value="gmm">Gaussian Mixture</option>
                        <option value="spectral">Spectral Clustering</option>
                        <option value="birch">BIRCH</option>
                        <option value="affinity_propagation">Affinity Propagation</option>
                        <option value="bisecting_kmeans">Bisecting K-Means</option>
                        <option value="hdbscan">HDBSCAN</option>
                        <option value="meanshift">Mean Shift</option>
                    </optgroup>
                    <optgroup label="Additional Models (Alphabetical)">
                        <option value="minibatch_kmeans">Mini-Batch K-Means</option>
                        <option value="optics">OPTICS</option>
                    </optgroup>
                </select>
            </div>
            
            <div id="headerAutomlClassifierModels" class="hidden">
                <label for="headerAutomlClassModels" style="display: block; margin-bottom: 8px; font-weight: 600;">Classification Model (Optional):</label>
                <select name="headerAutomlClassModels" id="headerAutomlClassModels" style="width: 100%; max-width: 400px; padding: 8px; border: 1px solid #ccc; border-radius: 4px;">
                    <option value="" selected>Let AutoML choose</option>
                    <optgroup label="Most Common">
                        <option value="Logistic_classifier">Logistic Classifier</option>
                        <option value="RF_classifier">Random Forest Classifier</option>
                        <option value="SVC_classifier">SVC Classifier</option>
                        <option value="MLP_classifier">MLP Classifier</option>
                        <option value="ExtraTrees_classifier">Extra Trees Classifier</option>
                        <option value="SGD_classifier">SGD Classifier</option>
                        <option value="AdaBoost_classifier">AdaBoost Classifier</option>
                        <option value="GradientBoosting_classifier">Gradient Boosting Classifier</option>
                        <option value="KNeighbors_classifier">K-Neighbors Classifier</option>
                    </optgroup>
                    <optgroup label="Multi-Output Models (Multiple Targets)">
                        <option value="GaussianNB_classifier">Gaussian Naive Bayes</option>
                        <option value="BernoulliNB_classifier">Bernoulli Naive Bayes</option>
                        <option value="CategoricalNB_classifier">Categorical Naive Bayes</option>
                        <option value="ComplementNB_classifier">Complement Naive Bayes</option>
                        <option value="MultinomialNB_classifier">Multinomial Naive Bayes</option>
                    </optgroup>
                    <optgroup label="Additional Models (Alphabetical)">
                        <option value="Bagging_classifier">Bagging Classifier</option>
                        <option value="DecisionTree_classifier">Decision Tree Classifier</option>
                        <option value="HistGradientBoosting_classifier">Histogram Gradient Boosting</option>
                        <option value="LDA_classifier">Linear Discriminant Analysis</option>
                        <option value="LinearSVC_classifier">Linear SVC</option>
                        <option value="NuSVC_classifier">Nu-SVC</option>
                        <option value="PassiveAggressive_classifier">Passive Aggressive Classifier</option>
                        <option value="QDA_classifier">Quadratic Discriminant Analysis</option>
                        <option value="Ridge_classifier">Ridge Classifier</option>
                    </optgroup>
                </select>
            </div>
        </div>

        <!-- Simple Modeling Section -->
        <div id="simpleModelingSection" class="modeling-mode-section">
        <div class="model-layout">
        <form id="processForm">
    <!-- Dropdowns populated based on output type selected -->
        <div class="modelSection">  
            <!-- Descriptive text about column selection -->
            <div style="margin-bottom: 16px; padding: 12px; background-color: #f8f9fa; border-radius: 6px; border-left: 3px solid #357a53;">
                <span id="simpleModelingSelectionNote" class="modeling-selection-note" style="font-style: italic; color: #2c3e50;"></span>
            </div>
            
            <!-- Model Selection -->
            <div class="preprocess-card" style="margin-bottom: 20px;">
                <h3 style="margin-top: 0; margin-bottom: 12px;">Select Model</h3>
                <p class="field-note" style="margin-bottom: 16px;">Choose the model algorithm for your selected output type.</p>
                
                <!-- Simple Mode Model Selection -->
                <div id="simpleNumericModels" class="hidden">
                    <label for="simpleNModels" style="display: block; margin-bottom: 8px; font-weight: 600;">Regression Model:</label>
                    <select name="simpleNModels" id="simpleNModels" style="width: 100%; max-width: 400px; padding: 8px; border: 1px solid #ccc; border-radius: 4px;">
                        <option value="" disabled selected>-- Select an option --</option>
                        <optgroup label="Most Common">
                            <option value="Linear">Linear</option>
                            <option value="Ridge">Ridge</option>
                            <option value="Lasso">Lasso</option>
                            <option value="ElasticNet">Elastic Net</option>
                            <option value="RF">Random Forest</option>
                            <option value="gradient_boosting">Gradient Boosting</option>
                            <option value="SVM">Support Vector Machine (SVR)</option>
                            <option value="MLP">Multi-Layer Perceptron</option>
                            <option value="K-Nearest">K-Nearest Neighbors</option>
                            <option value="ExtraTrees">Extra Trees</option>
                        </optgroup>
                        <optgroup label="Additional Models (Alphabetical)">
                            <option value="AdaBoost">AdaBoost Regressor</option>
                            <option value="ARDRegression">ARD Regression</option>
                            <option value="Bagging">Bagging Regressor</option>
                            <option value="BayesianRidge">Bayesian Ridge</option>
                            <option value="DecisionTree">Decision Tree Regressor</option>
                            <option value="ElasticNetCV">Elastic Net CV</option>
                            <option value="HistGradientBoosting">Histogram Gradient Boosting</option>
                            <option value="Huber">Huber Regressor</option>
                            <option value="LARS">LARS</option>
                            <option value="LARSCV">LARS CV</option>
                            <option value="LassoCV">Lasso CV</option>
                            <option value="LinearSVR">Linear SVR</option>
                            <option value="NuSVR">Nu-SVR</option>
                            <option value="OMP">Orthogonal Matching Pursuit</option>
                            <option value="PassiveAggressive">Passive Aggressive Regressor</option>
                            <option value="Quantile">Quantile Regressor</option>
                            <option value="RadiusNeighbors">Radius Neighbors Regressor</option>
                            <option value="RANSAC">RANSAC Regressor</option>
                            <option value="RidgeCV">Ridge CV</option>
                            <option value="SGD">SGD Regressor</option>
                            <option value="TheilSen">Theil-Sen Regressor</option>
                        </optgroup>
                    </select>
                </div>
                
                <div id="simpleClusterModels" class="hidden">
                    <label for="simpleClModels" style="display: block; margin-bottom: 8px; font-weight: 600;">Clustering Model:</label>
                    <select name="simpleClModels" id="simpleClModels" style="width: 100%; max-width: 400px; padding: 8px; border: 1px solid #ccc; border-radius: 4px;">
                        <option value="" disabled selected>-- Select an option --</option>
                        <optgroup label="Most Common">
                            <option value="kmeans">K-Means</option>
                            <option value="dbscan">DBSCAN</option>
                            <option value="agglo">Agglomerative</option>
                            <option value="gmm">Gaussian Mixture</option>
                            <option value="spectral">Spectral Clustering</option>
                            <option value="birch">BIRCH</option>
                            <option value="affinity_propagation">Affinity Propagation</option>
                            <option value="bisecting_kmeans">Bisecting K-Means</option>
                            <option value="hdbscan">HDBSCAN</option>
                            <option value="meanshift">Mean Shift</option>
                        </optgroup>
                        <optgroup label="Additional Models (Alphabetical)">
                            <option value="minibatch_kmeans">Mini-Batch K-Means</option>
                            <option value="optics">OPTICS</option>
                        </optgroup>
                    </select>
                </div>
                
                <div id="simpleClassifierModels" class="hidden">
                    <label for="simpleClassModels" style="display: block; margin-bottom: 8px; font-weight: 600;">Classification Model:</label>
                    <select name="simpleClassModels" id="simpleClassModels" style="width: 100%; max-width: 400px; padding: 8px; border: 1px solid #ccc; border-radius: 4px;">
                        <option value="" disabled selected>-- Select an option --</option>
                        <optgroup label="Most Common">
                            <option value="Logistic_classifier">Logistic Classifier</option>
                            <option value="RF_classifier">Random Forest Classifier</option>
                            <option value="SVC_classifier">SVC Classifier</option>
                            <option value="MLP_classifier">MLP Classifier</option>
                            <option value="ExtraTrees_classifier">Extra Trees Classifier</option>
                            <option value="SGD_classifier">SGD Classifier</option>
                            <option value="AdaBoost_classifier">AdaBoost Classifier</option>
                            <option value="GradientBoosting_classifier">Gradient Boosting Classifier</option>
                            <option value="KNeighbors_classifier">K-Neighbors Classifier</option>
                        </optgroup>
                        <optgroup label="Multi-Output Models (Multiple Targets)">
                            <option value="GaussianNB_classifier">Gaussian Naive Bayes</option>
                            <option value="BernoulliNB_classifier">Bernoulli Naive Bayes</option>
                            <option value="CategoricalNB_classifier">Categorical Naive Bayes</option>
                            <option value="ComplementNB_classifier">Complement Naive Bayes</option>
                            <option value="MultinomialNB_classifier">Multinomial Naive Bayes</option>
                        </optgroup>
                        <optgroup label="Additional Models (Alphabetical)">
                            <option value="Bagging_classifier">Bagging Classifier</option>
                            <option value="DecisionTree_classifier">Decision Tree Classifier</option>
                            <option value="HistGradientBoosting_classifier">Histogram Gradient Boosting</option>
                            <option value="LDA_classifier">Linear Discriminant Analysis</option>
                            <option value="LinearSVC_classifier">Linear SVC</option>
                            <option value="NuSVC_classifier">Nu-SVC</option>
                            <option value="PassiveAggressive_classifier">Passive Aggressive Classifier</option>
                            <option value="QDA_classifier">Quadratic Discriminant Analysis</option>
                            <option value="Ridge_classifier">Ridge Classifier</option>
                        </optgroup>
                    </select>
                </div>
            </div>
            
            <div class="model-card">
            <h2 style="margin-top: 0; margin-bottom: 20px;">Model Configuration</h2>

    <!-- Hyperparamers for each Model-->

            <!-- Regression Hyperparameters -->
            <div id="ridgeFields" class="hidden">
                <h3>Ridge Model Settings</h3>
                <label for="RidgeAlpha">Alpha - regularization strength (float >=0):</label>
                <input type="number" value=1.0 placeholder=1 min=0 id="RidgeAlpha" name="alpha">
                

                <div class="nonreqHyperparams">
                    <div><br></div>
                                    <div class="toggle-container">
                                        <span><h3>Edit Non-Essential Hyperparameters?</h3></span>
                                        <label class="switch">
                                            <input type="checkbox" id="nonreqRidgeSlider" aria-label="Toggle to show or hide non-essential Ridge hyperparameters">
                                            <span class="slider" aria-hidden="true"></span>
                                        </label>
                                    </div>
                    <div id="nonreqRidgeParams" class="hidden">
                        <label for="RidgeFitIntersept">Fit Intercept:</label>
                        <select name="RidgeFitIntersept" id="RidgeFitIntersept">
                            <option value="true">True</option>
                            <option value="false">False</option>
                        </select>

                        <label for="RidgeNormalize">Normalize:</label>
                        <select name="RidgeNormalize" id="RidgeNormalize">
                            <option value="false">False</option>
                            <option value="true">True</option>
                        </select>
                        <br>
                        <label for="RidgeCopyX">Copy X:</label>
                        <select name="RidgeCopyX" id="RidgeCopyX">
                            <option value="true">True</option>
                            <option value="false">False</option>
                        </select>
                        
                        <label for="RidgePositive">Positive:</label>
                        <select name="RidgePositive" id="RidgePositive">
                            <option value="false">False</option>
                            <option value="true">True</option>
                        </select>

                        <div><br></div>
                        <label for="RidgeSolver">Solver - Optimization method:</label>
                        <select name="solver" id="RidgeSolver">
                            <option value="auto">auto</option>
                            <option value="svd">svd</option>
                            <option value="cholesky">cholesky</option>
                            <option value="lsqr">lsqr</option>
                            <option value="sparse_cg">sparse_cg</option>
                            <option value="sag">sag</option>
                            <option value="saga">saga</option>
                            <option value="lbfgs">lbfgs</option>
                        </select>
                        <div><br></div>

                        <label for="RidgeMaxIter">Max Iterations (integer >=1):</label>
                        <input type="number" step=1 min=1 id="RidgeMaxIter" name="RidgeMaxIter">
                        <div><br></div>
                        <label for="RidgeTol">Tol (float >0):</label>
                        <input type="number" value=.0001 placeholder=.0001 min="0.0000001" step="any" id="RidgeTol" name="RidgeTol">
                        <!-- <div><br></div>
                        <label for="RidgeRandomState">Random State (an integar or leave blank for None):</label>
                        <input type="number" step=1 id="RidgeRandomState" name="RidgeRandomState"> -->
                    </div>
                </div>
            </div>
            
            <div id="lassoFields" class="hidden">
                <h3>Lasso Model Settings</h3>
                <label for="LassoAlpha">Alpha - regularization strength (float >=0):</label>
                <input type="number" value=1 min=0 placeholder=1 id="LassoAlpha" name="alpha" >
                
                <div class="nonreqHyperparams">
                    <div><br></div>
                    <div class="toggle-container">
                        <span><h3>Edit Non-Essential Hyperparameters?</h3></span>
                        <label class="switch">
                            <input type="checkbox" id="nonreqLassoSlider">
                            <span class="slider"></span>
                        </label>
                    </div>
                    <div id="nonreqLassoParams" class="hidden">
                        <label for="LassoFitIntersept">Fit Intercept:</label>
                        <select name="LassoFitIntersept" id="LassoFitIntersept">
                            <option value="true">True</option>
                            <option value="false">False</option>
                        </select>

                        <label for="LassoPrecompute">Precompute:</label>
                        <select name="LassoPrecompute" id="LassoPrecompute">
                            <option value="false">False</option>
                            <option value="true">True</option>
                        </select>

                        <label for="LassoCopyX">Copy X:</label>
                        <select name="LassoCopyX" id="LassoCopyX">
                            <option value="true">True</option>
                            <option value="false">False</option>
                        </select>
                        <div><br></div>
                        <label for="LassoWarmStart">Warm Start:</label>
                        <select name="LassoWarmStart" id="LassoWarmStart">
                            <option value="false">False</option>
                            <option value="true">True</option>
                        </select>

                        <label for="LassoPositive">Positive:</label>
                        <select name="LassoPositive" id="LassoPositive">
                            <option value="false">False</option>
                            <option value="true">True</option>
                        </select>
                        <div><br></div>

                        <label for="LassoMax_iter">Max Iterations - number of iterations (integer >=1):</label>
                        <input type="number" step=1 value=1000 placeholder=1000 min=1 id="LassoMax_iter" name="max_iter">
                        <div><br></div>
                        <label for="LassoTol">Tol (float >0):</label>
                        <input type="number" value=0.0001 placeholder=.0001 min="0.0000001" step="any" id="LassoTol" name="LassoTol">
                        <!-- <div><br></div>
                        <label for="LassoRandomState">Random State (an integer or leave blank for None):</label>
                        <input type="number" step=1 id="LassoRandomState" name="LassoRandomState"> -->
                        <div><br></div>
                        <label for="LassoSelection">Selection:</label>
                        <select name="LassoSelection" id="LassoSelection">
                            <option value="cyclic">Cyclic</option>
                            <option value="random">Random</option>
                        </select>


                    </div>
                </div>
            </div>

            <div id="logisticFields" class="hidden">
                <h3>Logistic Model Settings</h3>

                <div class="nonreqHyperparams">
                    <div class="toggle-container">
                        <span><h3>Edit Non-Essential Hyperparameters?</h3></span>
                        <label class="switch">
                            <input type="checkbox" id="nonreqLogisticSlider">
                            <span class="slider"></span>
                        </label>
                    </div>

                    <div id="nonreqLogisticParams" class="hidden">
                        <label for="LogisticDual">Dual:</label>
                        <select name="LogisticDual" id="LogisticDual">
                            <option value="false">False</option>
                            <option value="true">True</option>
                        </select>
                        <label for="LogisticFitIntercept">Fit Intercept:</label>
                        <select name="LogisticFitIntercept" id="LogisticFitIntercept">
                            <option value="true">True</option>
                            <option value="false">False</option>
                        </select>
                        <label for="LogisticSolver">Solver:</label>
                        <select name="LogisticSolver" id="LogisticSolver">
                            <option value="lbfgs">lbfgs</option>
                            <option value="newton-cg">newton-cg</option>
                            <option value="liblinear">liblinear</option>
                            <option value="sag">sag</option>
                            <option value="saga">saga</option>
                        </select>
                        <div><br></div>
                        <label for="LogisticMultiClass">Multi Class:</label>
                        <select name="LogisticMultiClass" id="LogisticMultiClass">
                            <option value="auto">auto</option>
                            <option value="ovr">ovr</option>
                            <option value="multinomial">multinomial</option>
                        </select>
                        <label for="LogisticWarmStart">Warm Start:</label>
                        <select name="LogisticWarmStart" id="LogisticWarmStart">
                            <option value="false">False</option>
                            <option value="true">True</option>
                        </select>
                        <div><br></div>

                        <label for="CLogistic">C - Regularization (float >0):</label>
                        <input type="number" value=1 placeholder=1.0 min="0.0000001" step="any" id="CLogistic" name="CLogistic">
                        <div><br></div>
                        <label for="penalty">Penalty:</label>
                        <select name="penalty" id="penalty">
                            <option value="l2">l2</option>
                            <option value="l1">l1</option>
                            <option value="elasticnet">elasticnet</option>
                            <option value="None">None</option>
                        </select>
                        <div><br></div>
                        <label for="LogisticTol">Tol (float >0):</label>
                        <input type="number" value=0.0001 placeholder=.0001 min="0.0000001" step="any" id="LogisticTol" name="LogisticTol">
                        <div><br></div>
                        <label for="Logisticintercept_scaling">Intercept Scaling:</label>
                        <input type="number" value=1 placeholder=1 step="any" id="Logisticintercept_scaling" name="Logisticintercept_scaling">
                        <div><br></div>
                        <label for="LogisticClassWeight">Class Weight (dict or 'balanced'):</label>
                        <input type="text" id="LogisticClassWeight" name="LogisticClassWeight">
                        <div><br></div>
                        <!-- <label for="LogisticRandomState">Random State (an integer or leave blank for None):</label>
                        <input type="number" step=1 id="LogisticRandomState" name="LogisticRandomState">
                        <div><br></div> -->
                        <label for="LogisticMaxIterations">Max Iterations (integer >=1):</label>
                        <input type="number" step=1 min="1" value=100 placeholder="100" id="LogisticMaxIterations" name="LogisticMaxIterations">
                        <div><br></div>
                        <label for="LogisticVerbose">Verbose (int):</label>
                        <input type="number" step=1 value=0 placeholder="0" id="LogisticVerbose" name="LogisticVerbose">
                        <div><br></div>
                        <label for="LogisticNJobs">N Jobs (int):</label>
                        <input type="number" step=1 id="LogisticNJobs" name="LogisticNJobs">
                        <div><br></div>
                        <label for="Logisticl1Ratio">L1 Ratio (float [0, 1]):</label>
                        <input type="number" min="0.0000001" step="any" max=1 id="Logisticl1Ratio" name="Logisticl1Ratio">
                    </div>
                </div>
            </div>

            <div id="polynomialFields" class="hidden">
                <h3>Polynomial Model Settings</h3>
                <label for="degree_specificity">Degree Specificity (integer >=1):</label>
                <input type="number" step=1 value=2 placeholder=2 min=1 id="degree_specificity" name="degree_specificity">
            </div>

            <div id="elasticNetFields" class="hidden">
                <h3>Elastic Net Model Settings</h3>
                <label for="ENAlpha">Alpha - regularization strength (float >=0):</label>
                <input type="number" value=1 placeholder=1 min=0 id="ENAlpha" name="alpha" >
                <div><br></div>
                <label for="l1_ratio">l1 Ratio - ratio between lasso and ridge penalties <br>      (float [0.0, 1.0]):</label>
                <input type="number" value=.5 placeholder=.5 min="0.0000001" step="any" max=1 id="l1_ratio" name="l1_ratio">
            </div>

            <div id="SVMFields" class="hidden">
                <h3>SVM Model Settings</h3>
                <label for="C">C - Regularization (float >0):</label>
                <input type="number" value=1 placeholder=1 min="0.0000001" step="any" placeholder=1.0 id="C" name="C">
                <div><br></div>
                <label for="kernel">Kernel:</label>
                <select name="kernel" id="kernel">
                    <option value="rbf">rbf</option>
                    <option value="linear">linear</option>
                    <option value="poly">poly</option>
                    <option value="sigmoid">sigmoid</option>
                    <option value="precomputed">precomputed</option>
                </select>
                <div id="polykernelFields" class="hidden">
                    <div><br></div>
                    <label for="polyDegree">Degree of poly kernel function (int):</label>
                    <input type="number" step=1 value=3 placeholder=3.0 id="polyDegree" name="polyDegree">
                </div>
                <div id="svmGamma">
                    <div><br></div>
                    <label for="Gamma">Gamma - enter 'auto', 'scale', or a float:</label>
                    <input type="text"  value='scale' placeholder='scale' id="Gamma" name="Gamma">
                </div>

                <div class="nonreqHyperparams">
                    <div><br></div>
                    <div class="toggle-container">
                        <span><h3>Edit Non-Essential Hyperparameters?</h3></span>
                        <label class="switch">
                            <input type="checkbox" id="nonreqSVMSlider">
                            <span class="slider"></span>
                        </label>
                    </div>
                    <div id="nonreqSVMParams" class="hidden">
                        <label for="SVMshrinking">Shrinking:</label>
                        <select name="SVMshrinking" id="SVMshrinking">
                            <option value="true">True</option>
                            <option value="false">False</option>
                        </select>
                        <label for="SVMprobability">Probability:</label>
                        <select name="SVMprobability" id="SVMprobability">
                            <option value="false">False</option>
                            <option value="true">True</option>
                        </select>
                        <br>
                        <label for="SVMBreakTies">Break Ties:</label>
                        <select name="SVMBreakTies" id="SVMBreakTies">
                            <option value="false">False</option>
                            <option value="true">True</option>
                        </select>
                        <label for="SVMverbose">Verbose:</label>
                        <select name="SVMverbose" id="SVMverbose">
                            <option value="false">False</option>
                            <option value="true">True</option>
                        </select>
                        <div><br></div>
                        <label for="SVMcoef0">coef0 (float):</label>
                        <input type="number" step="any" value=0 placeholder="0" id="SVMcoef0" name="SVMcoef0">
                        <div><br></div>
                        <label for="SVMtol">tol(float):</label>
                        <input type="number" step="any" value=.001 placeholder=.001 id="SVMtol" name="SVMtol">
                        <div><br></div>
                        <label for="SVMCacheSize">Cache Size (float):</label>
                        <input type="number" step="any" value=200 placeholder=200 id="SVMCacheSize" name="SVMCacheSize">
                        <div><br></div>
                        <label for="SVMClassWeight">Class Weight (enter a dictionary or 'balanced')</label>
                        <input type="text" id="SVMClassWeight" name="SVMClassWeight">
                        <div><br></div>
                        <label for="SVMmaxIter">Max Iterations (int):</label>
                        <input type="number" step=1 value=-1 placeholder=-1 id="SVMmaxIter" name="SVMmaxIter">
                        <div><br></div>
                        <label for="SVMdecisionFunctionShape">Decision Function Shape:</label>
                        <select name="SVMdecisionFunctionShape" id="SVMdecisionFunctionShape">
                            <option value="ovr">ovr</option>
                            <option value="ovo">ovo</option>
                        </select>
                        <div><br></div>
                        <!-- <label for="SVMrandomState">Random State (an integer or leave blank for None):</label>
                        <input type="number" step=1 id="SVMrandomState" name="SVMrandomState">
                        <div><br></div> -->

                    </div>
                </div>

            </div>

            <div id="RFFields" class="hidden">
                <h3>Random Forest Model Settings</h3>
                <label for="RFn_estmators">N Estimators - # of trees (integer >=1):</label>
                <input type="number" step=1 value=100 placeholder="100" min=1 id="RFn_estmators" name="RFn_estmators">
                
                <div class="nonreqHyperparams">
                    <div><br></div>
                    <div class="toggle-container">
                        <span><h3>Edit Non-Essential Hyperparameters?</h3></span>
                        <label class="switch">
                            <input type="checkbox" id="nonreqRFSlider">
                            <span class="slider"></span>
                        </label>
                    </div>
                    <div id="nonreqRFParams" class="hidden">
                        <label for="RFBoostrap">Bootstrap:</label>
                        <select name="RFBoostrap" id="RFBoostrap">
                            <option value="true">True</option>
                            <option value="false">False</option>
                        </select>
                        <label for="RFoobScore">oob Score:</label>
                        <select name="RFoobScore" id="RFoobScore">
                            <option value="false">False</option>
                            <option value="true">True</option>
                        </select>
                        <label for="RFWarmStart">Warm Start:</label>
                        <select name="RFWarmStart" id="RFWarmStart">
                            <option value="false">False</option>
                            <option value="true">True</option>
                        </select>
                        <div><br></div>
                        <!-- <label for="RFCriterion">Criterion:</label>
                        <select name="RFCriterion" id="RFCriterion">
                            <option value="gini">gini</option>
                            <option value="entropy">entropy</option>
                        </select> -->
                        <label for="RFmin_weight_fraction_leaf">Min Weight Fraction Leaf (float [0.0, 0.5]):</label>
                        <input type="number" value=0 placeholder=0 min="0" max=".5" step="any" id="RFmin_weight_fraction_leaf" name="RFmin_weight_fraction_leaf">
                        <div><br></div>
                        <!-- <label for="RFMaxFeatures">Max Features:</label> -->
                        <label for="RFMaxLeafNodes">Max Leaf Nodes (an integer or leave blank for None):</label>
                        <input type="number" step="1" id="RFMaxLeafNodes" name="RFMaxLeafNodes">
                        <div><br></div>
                        <label for="RFMinImpurityDecrease">Min Impurity Decrease (float):</label>
                        <input type="number" value=0 placeholder=0 step="any" id="RFMinImpurityDecrease" name="RFMinImpurityDecrease">
                        <div><br></div>
                        <label for="RFNJobs">N Jobs (an integer or leave blank for None):</label>
                        <input type="number" step="1" id="RFNJobs" name="RFNJobs">
                        <div><br></div>
                        <!-- <label for="RFRandomState">Random State (an integer or leave blank for None):</label>
                        <input type="number" step="1" id="RFRandomState" name="RFRandomState">
                        <div><br></div> -->
                        <label for="RFVerbose">Verbose (int):</label>
                        <input type="number" value=0 placeholder=0 step="1" id="RFVerbose" name="RFVerbose">

                        <div><br></div>
                        <label for="RFMax_depth">Max Depth - Tree depth (an integer or leave blank for None):</label>
                        <input type="number" step=1 id="RFMax_depth" name="max_depth">
                        <div><br></div>
                        <label for="min_samples_split">Min Samples Split - Min samples per split (integer or float):</label>
                        <input type="number" value=2 placeholder="2" id="min_samples_split" name="min_samples_split">
                        <div><br></div>
                        <label for="min_samples_leaf">Min Samples Leaf - Min samples per leaf (integer or float):</label>
                        <input type="number" value=1 placeholder=1 id="min_samples_leaf" name="min_samples_leaf">

                    </div>
                </div>
            </div>

            <div id="PerceptronFields" class="hidden">
                <h3>Perceptron Model Settings</h3>
                <label for="PercMax_iter">Max Iter - max # of iterations (integer):</label>
                <input type="number" step=1 value=1000 placeholder=1000 id="PercMax_iter" name="max_iter">
                <div><br></div>
                <label for="eta0">eta0 - learning rate (float):</label>
                <input type="number" value=1 placeholder="1" id="eta0" name="eta0">

                <div class="nonreqHyperparams">
                    <div><br></div>
                    <div class="toggle-container">
                        <span><h3>Edit Non-Essential Hyperparameters?</h3></span>
                        <label class="switch">
                            <input type="checkbox" id="nonreqPerceptronSlider">
                            <span class="slider"></span>
                        </label>
                    </div>
                    <div id="nonreqPerceptronParams" class="hidden">
                        <label for="PerceptronFitIntercept">Fit Intercept:</label>
                        <select name="PerceptronFitIntercept" id="PerceptronFitIntercept">
                            <option value="true">True</option>
                            <option value="false">False</option>
                        </select>
                        <label for="PerceptronShuffle">Shuffle:</label>
                        <select name="PerceptronShuffle" id="PerceptronShuffle">
                            <option value="true">True</option>
                            <option value="false">False</option>
                        </select>
                        <label for="PerceptronEarlyStopping">Early Stopping:</label>
                        <select name="PerceptronEarlyStopping" id="PerceptronEarlyStopping">
                            <option value="false">False</option>
                            <option value="true">True</option>
                        </select>
                        <label for="PerceptronWarmStart">Warm Start:</label>
                        <select name="PerceptronWarmStart" id="PerceptronWarmStart">
                            <option value="false">False</option>
                            <option value="true">True</option>
                        </select>
                        <div><br></div>
                        <label for="PerceptronPenalty">Penalty:</label>
                        <select name="PerceptronPenalty" id="PerceptronPenalty">
                            <option value="None">None</option>
                            <option value="l2">l2</option>
                            <option value="l1">l1</option>
                            <option value="elasticnet">elasticnet</option>
                        </select>
                        <div><br></div>
                        <label for="PerceptronAlpha">Alpha (float):</label>
                        <input type="number" step="any" value=.0001 placeholder=.0001 id="PerceptronAlpha" name="PerceptronAlpha">
                        <div><br></div>
                        <label for="PerceptronTol">Tol (float):</label>
                        <input type="number" step="any" value=.001 placeholder=.001 id="PerceptronTol" name="PerceptronTol">
                        <div><br></div>
                        <label for="PerceptronVerbose">Verbose (int):</label>
                        <input type="number" step="1" value=0 placeholder=0 id="PerceptronVerbose" name="PerceptronVerbose">
                        <div><br></div>
                        <label for="PerceptronNJobs">N Jobs (an integer or or leave blank for None):</label>
                        <input type="number" step="1" id="PerceptronNJobs" name="PerceptronNJobs">
                        <div><br></div>
                        <!-- <label for="PerceptronRandomState">Random State (an integer or or leave blank for None):</label>
                        <input type="number" step="1" id="PerceptronRandomState" name="PerceptronRandomState">
                        <div><br></div> -->
                        <label for="PerceptronValidationFraction">Validation Fraction (float):</label>
                        <input type="number" step="any" value=.1 placeholder=.1 id="PerceptronValidationFraction" name="PerceptronValidationFraction">
                        <div><br></div>
                        <label for="PerceptronNIterNoChange">Number Iterations No Change (int):</label>
                        <input type="number" step="1" value=5 placeholder=5 id="PerceptronNIterNoChange" name="PerceptronNIterNoChange">
                        <div><br></div>
                        <label for="PerceptronClassWeight">Class Weight (enter a dictionary or 'balanced'):</label>
                        <input type="text" id="PerceptronClassWeight" name="PerceptronClassWeight">
                        <div><br></div>
                    </div>
                </div>
            </div>

            <div id="MLPFields" class="hidden">
                <h3>MLP Model Settings</h3>
                <label for="hidden_layer_sizes1">Hidden Layer Sizes - neurons in each hidden layer (ints):</label>
                <br>
                <input type="number" id="hidden_layer_sizes1" step=1 value="100" name="hidden_layer_sizes1">
                <input type="number" id="hidden_layer_sizes2" step=1 value="" name="hidden_layer_sizes2">
                <input type="number" id="hidden_layer_sizes3" step=1 value="" name="hidden_layer_sizes3">
                <div><br></div>
                <label for="activation">Activation:</label>
                <select name="activation" id="activation">
                    <option value="relu">relu</option>
                    <option value="tanh">tanh</option>
                    <option value="logistic">logistic</option>
                    <option value="identity">identity</option>
                </select>
                <div><br></div>
                <label for="MLPSolver">Solver:</label>
                <select name="MLPSolver" id="MLPSolver">
                    <option value="adam">adam</option>
                    <option value="sgd">sgd</option>
                    <option value="lbfgs">lbfgs</option>
                </select>
                

                <div class="nonreqHyperparams">
                    <div><br></div>
                    <div class="toggle-container">
                        <span><h3>Edit Non-Essential Hyperparameters?</h3></span>
                        <label class="switch">
                            <input type="checkbox" id="nonreqMLPSlider">
                            <span class="slider"></span>
                        </label>
                    </div>
                    <div id="nonreqMLPParams" class="hidden">

                    <label for="MLPNesterovsMomentum">Nesterovs Momentum:</label>
                    <select name="MLPNesterovsMomentum" id="MLPNesterovsMomentum">
                        <option value="true">True</option>
                        <option value="false">False</option>
                    </select>
                    <label for="MLPEarlyStopping">Early Stopping:</label>
                    <select name="MLPEarlyStopping" id="MLPEarlyStopping">
                        <option value="false">False</option>
                        <option value="true">True</option>
                    </select>
                    <label for="MLPShuffle">Shuffle:</label>
                    <select name="MLPShuffle" id="MLPShuffle">
                        <option value="true">True</option>
                        <option value="false">False</option>
                    </select>
                    <div><br></div>
                    <label for="MLPVerbose">Verbose:</label>
                    <select name="MLPVerbose" id="MLPVerbose">
                        <option value="false">False</option>
                        <option value="true">True</option>
                    </select>
                    <label for="MLPWarmStart">Warm Start:</label>
                    <select name="MLPWarmStart" id="MLPWarmStart">
                        <option value="false">False</option>
                        <option value="true">True</option>
                    </select>
                    <div><br></div>

                    <label for="MLPAlpha">Alpha - Regularization (float >=0):</label>
                    <input type="number" value=.0001 placeholder=.0001 step="any" min=0 id="MLPAlpha" name="alpha">
                    <div><br></div>
                    <label for="MLPLearning_rate">Learning Rate:</label>
                    <select name="MLPLearning_rate" id="MLPLearning_rate">
                        <option value="constant">constant</option>
                        <option value="invscaling">invscaling</option>
                        <option value="adaptive">adaptive</option>\
                    </select>
                    <div><br></div>
                    <label for="MLPBatchSize">Batch Size (integer >=1 or 'auto'):</label>
                    <input type="text" value=200 placeholder="200" id="MLPBatchSize" name="MLPBatchSize">
                    <div><br></div>
                    <label for="MLPLearningRateInit">Learning Rate Init (float >0):</label>
                    <input type="number" value=.001 placeholder=".001" min="0.0000001" step="any" id="MLPLearningRateInit" name="MLPLearningRateInit">
                    <div><br></div>
                    <label for="MLPPowerT">Power T (float):</label>
                    <input type="number" value=.05 placeholder=.5 step="any" id="MLPPowerT" name="MLPPowerT">
                    <div><br></div>
                    <label for="MLPMaxIter">Max Iterations (integer >=1):</label>
                    <input type="number" value=200 placeholder=200 step="1" min="1" id="MLPMaxIter" name="MLPMaxIter">
                    <div><br></div>
                    <!-- <label for="MLPRandomState">Random State (integer or leave blank for None):</label>
                    <input type="number" step="1" id="MLPRandomState" name="MLPRandomState">
                    <div><br></div> -->
                    <label for="MLPTol">Tol (float):</label>
                    <input type="number" value=.0001 placeholder=.0001 step="any" id="MLPTol" name="MLPTol">
                    <div><br></div>
                    <label for="MLPMomentum">Momentum (float [0, 1&#41; ):</label>
                    <input type="number" value=.09 placeholder=.9 min=0 max=.9999999 step="any" id="MLPMomentum" name="MLPMomentum">
                    <div><br></div>
                    <label for="MLPValidationFraction">Validation Fraction (float [0, 1&#41; ):</label>
                    <input type="number" value=.01 placeholder=.1 min=0 max=.9999999 step="any" id="MLPValidationFraction" name="MLPValidationFraction">
                    <div><br></div>
                    <label for="MLPBeta1"> Beta 1 (float [0, 1&#41; ):</label>
                    <input type="number" value=.09 placeholder=.9 min=0 max=.9999999  step="any" id="MLPBeta1" name="MLPBeta1">
                    <div><br></div>
                    <label for="MLPBeta2">Beta 2 (float [0, 1&#41; ):</label>
                    <input type="number" value=.999 placeholder=.999 min=0 max=.9999999 step="any" id="MLPBeta2" name="MLPBeta2">
                    <div><br></div>
                    <label for="MLPEpsilon">Epsilon (float >0):</label>
                    <input type="number" value=.00000001 min=.00000000001 placeholder=.00000001 step="any" id="MLPEpsilon" name="MLPEpsilon">

                    </div>
                </div>
                
            </div>

            <div id="K-NearestFields" class="hidden">
                <h3>K-Nearest Neighbors Model Settings</h3>
                <label for="n_neighbors">N Neighbors - # of neighbors (int):</label>
                <input type="number" step=1 value=5 placeholder=5 id="n_neighbors" name="n_neighbors">
                

                <div class="nonreqHyperparams">
                    <div><br></div>
                    <div class="toggle-container">
                        <span><h3>Edit Non-Essential Hyperparameters?</h3></span>
                        <label class="switch">
                            <input type="checkbox" id="nonreqKNearestSlider">
                            <span class="slider"></span>
                        </label>
                    </div>
                    <div id="nonreqKNearestParams" class="hidden">
                        
                        <label for="KNearestWeights">Weights:</label>
                        <select name="KNearestWeights" id="KNearestWeights">
                            <option value="uniform">uniform</option>
                            <option value="distance">distance</option>
                            <option value="callable">callable</option>
                        </select>
                        <label for="KNearestAlgorithm">Algorithm:</label>
                        <select name="KNearestAlgorithm" id="KNearestAlgorithm">
                            <option value="auto">auto</option>
                            <option value="ball_tree">ball_tree</option>
                            <option value="kd_tree">kd_tree</option>
                            <option value="brute">brute</option>
                        </select>
                        <div><br></div>

                        <label for="metric">Metric - (euclidean, manhattan, etc):</label>
                        <input type="text" id="metric" value='minkowski' name="metric">
                        <div><br></div>
                        <label for="KNearestLeafSize">Leaf Size (int):</label>
                        <input type="number" step="1" value=30 placeholder=30 id="KNearestLeafSize" name="KNearestLeafSize">
                        <div><br></div>
                        <label for="KNearestP">P (int):</label>
                        <input type="number" step="1" value=2 placeholder=2 id="KNearestP" name="KNearestP">
                        <div><br></div>
                        <label for="KNearestMetricParams">Metric Params (enter a dictionary or leave blank for None):</label>
                        <input type="text" id="KNearestMetricParams" name="KNearestMetricParams">
                        <div><br></div>
                        <label for="KNearestNJobs">N Jobs (an integer or leave blank for None):</label>
                        <input type="number" step="1" id="KNearestNJobs" name="KNearestNJobs">
                        <div><br></div>
                        
                    </div>
                </div>
            </div>

            <div id="GradientBoostingFields" class="hidden">
                <h3>Gradient Boosting Model Settings</h3>
                <label for="GBn_estimators">N Estimators - Trees (integer >=1):</label>
                <input type="number" step=1 value=100 placeholder=100 min=1 id="GBn_estimators" name="GBn_estimators">
                <div><br></div>
                <label for="GBlearn">Learning Rate (float >0):</label>
                <input type="number" value=.1 placeholder=.1 min="0.0000001" step="any" id="GBlearn" name="GBlearn">

                <div class="nonreqHyperparams">
                    <div><br></div>
                    <div class="toggle-container">
                        <span><h3>Edit Non-Essential Hyperparameters?</h3></span>
                        <label class="switch">
                            <input type="checkbox" id="nonreqGBSlider">
                            <span class="slider"></span>
                        </label>
                    </div>
                    <div id="nonreqGBParams" class="hidden">
                        
                        <label for="GBLoss">Loss:</label>
                        <select name="GBLoss" id="GBLoss">
                            <option value="absolute_error">absolute_error</option>
                            <option value="squared_error">squared_error</option>
                            <option value="huber">huber</option>
                            <option value="quantile">quantile</option>
                        </select>
                        <label for="GBWarmStart">Warm Start:</label>
                        <select name="GBWarmStart" id="GBWarmStart">
                            <option value="false">False</option>
                            <option value="true">True</option>
                        </select>
                        <label for="GBCriterion">Criterion:</label>
                        <select name="GBCriterion" id="GBCriterion">
                            <option value="friedman_mse">Friedman MSE</option>
                            <option value="squared_error">Squared Error</option>
                        </select>

                        <div><br></div>
                        <label for="GBMax_depth">Max Depth - Tree depth (an integer or leave blank for None):</label>
                        <input type="number" step=1 value=3 placeholder=3 id="GBMax_depth" name="GBMax_depth">
                        <div><br></div>
                        <label for="GBSubsample">Subsample (float &#40;0,1]):</label>
                        <input type="number" value=1 placeholder=1 min="0.0000001" max=1 step="any" id="GBSubsample" name="GBSubsample">
                        <div><br></div>
                        <label for="GBMinSamplesSplit">Min Samples Split (integer or float):</label>
                        <input type="number" value=2 placeholder=2 step="any" id="GBMinSamplesSplit" name="GBMinSamplesSplit">
                        <div><br></div>
                        <label for="GBMinSamplesLeaf">Min Samples Leaf (integer or float):</label>
                        <input type="number" value=1 placeholder=1 step="any" id="GBMinSamplesLeaf" name="GBMinSamplesLeaf">
                        <div><br></div>
                        <label for="GBMinWeightFractionLeaf">Min Weight Fraction Leaf (float [0.0, 0.5]):</label>
                        <input type="number" value=0 placeholder=0 step="any" min=0 max=.5 id="GBMinWeightFractionLeaf" name="GBMinWeightFractionLeaf">
                        <div><br></div>
                        <label for="GBMinImpurityDecrease">Min Impurity Decrease (float):</label>
                        <input type="number" value=0 placeholder=0 step="any" id="GBMinImpurityDecrease" name="GBMinImpurityDecrease">
                        <div><br></div>
                        <label for="GBInit">Init ('estimator' or leave blank for None):</label>
                        <input type="text" placeholder='estimator' id="GBInit" name="GBInit">
                        <div><br></div>
                        <!-- <label for="GBRandomState">Random State (an integer or leave blank for None):</label>
                        <input type="number" step="1" id="GBRandomState" name="GBRandomState">
                        <div><br></div> -->
                        <label for="GBMaxFeatrues">Max Features (int, float, or string):</label>
                        <input type="text" id="GBMaxFeatrues" name="GBMaxFeatrues">
                        <div><br></div>
                        <label for="GBAlpha">Alpha (float [0.0, 1.0]):</label>
                        <input type="number" step="any" min=0 max=1 value=.9 placeholder=.9 id="GBAlpha" name="GBAlpha">
                        <div><br></div>
                        <label for="GBVerbose">Verbose (int):</label>
                        <input type="number" step="1" value=0 placeholder=0 id="GBVerbose" name="GBVerbose">
                        <div><br></div>
                        <label for="GBMaxLeafNodes">Max Leaf Nodes (an integer or leave blank for None):</label>
                        <input type="number" step="1" id="GBMaxLeafNodes" name="GBMaxLeafNodes">
                    </div>
                </div>
            </div>

             <!-- Classifier Hyperparameters -->
            <div id="Logistic_classifierFields" class="hidden">
                <h3>Logistic Classifier Model Settings</h3>
                <div class="nonreqHyperparams">
                    <div class="toggle-container">
                        <span><h3>Edit Non-Essential Hyperparameters?</h3></span>
                        <label class="switch">
                            <input type="checkbox" id="nonreqLogisticClassifierSlider">
                            <span class="slider"></span>
                        </label>
                    </div>

                    <div id="nonreqLogisticClassifierParams" class="hidden">
                        <label for="Class_LogisticDual">Dual:</label>
                        <select name="Class_LogisticDual" id="Class_LogisticDual">
                            <option value="false">False</option>
                            <option value="true">True</option>
                        </select>
                        <label for="Class_LogisticFitIntercept">Fit Intercept:</label>
                        <select name="Class_LogisticFitIntercept" id="Class_LogisticFitIntercept">
                            <option value="true">True</option>
                            <option value="false">False</option>
                        </select>
                        <label for="Class_LogisticSolver">Solver:</label>
                        <select name="Class_LogisticSolver" id="Class_LogisticSolver">
                            <option value="lbfgs">lbfgs</option>
                            <option value="newton-cg">newton-cg</option>
                            <option value="liblinear">liblinear</option>
                            <option value="sag">sag</option>
                            <option value="saga">saga</option>
                        </select>
                        <div><br></div>
                        <label for="Class_LogisticMultiClass">Multi Class:</label>
                        <select name="Class_LogisticMultiClass" id="Class_LogisticMultiClass">
                            <option value="auto">auto</option>
                            <option value="ovr">ovr</option>
                            <option value="multinomial">multinomial</option>
                        </select>
                        <label for="Class_LogisticWarmStart">Warm Start:</label>
                        <select name="Class_LogisticWarmStart" id="Class_LogisticWarmStart">
                            <option value="false">False</option>
                            <option value="true">True</option>
                        </select>
                        <div><br></div>

                        <label for="Class_CLogistic">C - Regularization (float >0):</label>
                        <input type="number" value=1 placeholder=1.0 min="0.0000001" step="any" id="Class_CLogistic" name="Class_CLogistic">
                        <div><br></div>
                        <label for="Class_Logistic_penalty">Penalty:</label>
                        <select name="Class_Logistic_penalty" id="Class_Logistic_penalty">
                            <option value="l2">l2</option>
                            <option value="l1">l1</option>
                            <option value="elasticnet">elasticnet</option>
                            <option value="None">None</option>
                        </select>
                        <div><br></div>
                        <label for="Class_LogisticTol">Tol (float >0):</label>
                        <input type="number" value=0.0001 placeholder=.0001 min="0.0000001" step="any" id="Class_LogisticTol" name="Class_LogisticTol">
                        <div><br></div>
                        <label for="Class_Logisticintercept_scaling">Intercept Scaling:</label>
                        <input type="number" value=1 placeholder=1 step="any" id="Class_Logisticintercept_scaling" name="Class_Logisticintercept_scaling">
                        <div><br></div>
                        <label for="Class_LogisticClassWeight">Class Weight (dict or 'balanced'):</label>
                        <input type="text" id="Class_LogisticClassWeight" name="Class_LogisticClassWeight">
                        <div><br></div>
                        <!-- <label for="Class_LogisticRandomState">Random State (an integer or leave blank for None):</label>
                        <input type="number" step=1 id="Class_LogisticRandomState" name="Class_LogisticRandomState">
                        <div><br></div> -->
                        <label for="Class_LogisticMaxIterations">Max Iterations (integer >=1):</label>
                        <input type="number" step=1 min="1" value=100 placeholder="100" id="Class_LogisticMaxIterations" name="Class_LogisticMaxIterations">
                        <div><br></div>
                        <label for="Class_LogisticVerbose">Verbose (int):</label>
                        <input type="number" step=1 value=0 placeholder="0" id="Class_LogisticVerbose" name="Class_LogisticVerbose">
                        <div><br></div>
                        <label for="Class_LogisticNJobs">N Jobs (int):</label>
                        <input type="number" step=1 id="Class_LogisticNJobs" name="Class_LogisticNJobs">
                        <div><br></div>
                        <label for="Class_Logisticl1Ratio">L1 Ratio (float [0, 1]):</label>
                        <input type="number" min="0.0000001" step="any" max=1 id="Class_Logisticl1Ratio" name="Class_Logisticl1Ratio">
                    </div>
                </div>
            </div>

            <div id="MLP_classifierFields" class="hidden"> 
                <h3>MLP Classifier Model Settings</h3>
                <label for="Class_hidden_layer_sizes1">Hidden Layer Sizes - neurons in each hidden layer (ints):</label>
                <br>
                <input type="number" id="Class_hidden_layer_sizes1" step=1 value="100" name="Class_hidden_layer_sizes1">
                <input type="number" id="Class_hidden_layer_sizes2" step=1 value="" name="Class_hidden_layer_sizes2">
                <input type="number" id="Class_hidden_layer_sizes3" step=1 value="" name="Class_hidden_layer_sizes3">
                <div><br></div>

                <label for="Class_activation">Activation:</label>
                <select name="Class_activation" id="Class_activation">
                    <option value="relu">relu</option>
                    <option value="tanh">tanh</option>
                    <option value="logistic">logistic</option>
                    <option value="identity">identity</option>
                </select>
                <div><br></div>

                <label for="Class_MLPSolver">Solver:</label>
                <select name="Class_MLPSolver" id="Class_MLPSolver">
                    <option value="adam">adam</option>
                    <option value="sgd">sgd</option>
                    <option value="lbfgs">lbfgs</option>
                </select>

                <div><br></div>
                <div class="nonreqHyperparams">
                    <div class="toggle-container">
                        <span><h3>Edit Non-Essential Hyperparameters?</h3></span>
                        <label class="switch">
                            <input type="checkbox" id="nonreqMLPClassifierSlider">
                            <span class="slider"></span>
                        </label>
                    </div>

                    <div id="nonreqMLPClassifierParams" class="hidden">

                        <label for="Class_MLPNesterovsMomentum">Nesterovs Momentum:</label>
                            <select name="Class_MLPNesterovsMomentum" id="Class_MLPNesterovsMomentum">
                                <option value="true">True</option>
                                <option value="false">False</option>
                            </select>
                            <label for="Class_MLPEarlyStopping">Early Stopping:</label>
                            <select name="Class_MLPEarlyStopping" id="Class_MLPEarlyStopping">
                                <option value="false">False</option>
                                <option value="true">True</option>
                            </select>
                            <label for="Class_MLPShuffle">Shuffle:</label>
                            <select name="Class_MLPShuffle" id="Class_MLPShuffle">
                                <option value="true">True</option>
                                <option value="false">False</option>
                            </select>
                            <div><br></div>
                            <label for="Class_MLPVerbose">Verbose:</label>
                            <select name="Class_MLPVerbose" id="Class_MLPVerbose">
                                <option value="false">False</option>
                                <option value="true">True</option>
                            </select>
                            <label for="Class_MLPWarmStart">Warm Start:</label>
                            <select name="Class_MLPWarmStart" id="Class_MLPWarmStart">
                                <option value="false">False</option>
                                <option value="true">True</option>
                            </select>
                            <div><br></div>

                            <label for="Class_MLPAlpha">Alpha - Regularization (float >=0):</label>
                            <input type="number" value=.0001 placeholder=.0001 step="any" min=0 id="Class_MLPAlpha" name="Class_MLPAlpha">
                            <div><br></div>
                            <label for="Class_MLPLearning_rate">Learning Rate:</label>
                            <select name="Class_MLPLearning_rate" id="Class_MLPLearning_rate">
                                <option value="constant">constant</option>
                                <option value="invscaling">invscaling</option>
                                <option value="adaptive">adaptive</option>\
                            </select>
                            <div><br></div>
                            <label for="Class_MLPBatchSize">Batch Size (integer >=1 or 'auto'):</label>
                            <input type="text" value=200 placeholder="200" id="Class_MLPBatchSize" name="Class_MLPBatchSize">
                            <div><br></div>
                            <label for="Class_MLPLearningRateInit">Learning Rate Init (float >0):</label>
                            <input type="number" value=.001 placeholder=".001" min="0.0000001" step="any" id="Class_MLPLearningRateInit" name="Class_MLPLearningRateInit">
                            <div><br></div>
                            <label for="Class_MLPPowerT">Power T (float):</label>
                            <input type="number" value=.05 placeholder=.5 step="any" id="Class_MLPPowerT" name="Class_MLPPowerT">
                            <div><br></div>
                            <label for="Class_MLPMaxIter">Max Iterations (integer >=1):</label>
                            <input type="number" value=200 placeholder=200 step="1" min="1" id="Class_MLPMaxIter" name="Class_MLPMaxIter">
                            <div><br></div>
                            <!-- <label for="Class_MLPRandomState">Random State (integer or leave blank for None):</label>
                            <input type="number" step="1" id="Class_MLPRandomState" name="Class_MLPRandomState">
                            <div><br></div> -->
                            <label for="Class_MLPTol">Tol (float):</label>
                            <input type="number" value=.0001 placeholder=.0001 step="any" id="Class_MLPTol" name="Class_MLPTol">
                            <div><br></div>
                            <label for="Class_MLPMomentum">Momentum (float [0, 1&#41; ):</label>
                            <input type="number" value=.09 placeholder=.9 min=0 max=.9999999 step="any" id="Class_MLPMomentum" name="Class_MLPMomentum">
                            <div><br></div>
                            <label for="Class_MLPValidationFraction">Validation Fraction (float [0, 1&#41; ):</label>
                            <input type="number" value=.01 placeholder=.1 min=0 max=.9999999 step="any" id="Class_MLPValidationFraction" name="Class_MLPValidationFraction">
                            <div><br></div>
                            <label for="Class_MLPBeta1"> Beta 1 (float [0, 1&#41; ):</label>
                            <input type="number" value=.09 placeholder=.9 min=0 max=.9999999  step="any" id="Class_MLPBeta1" name="Class_MLPBeta1">
                            <div><br></div>
                            <label for="Class_MLPBeta2">Beta 2 (float [0, 1&#41; ):</label>
                            <input type="number" value=.999 placeholder=.999 min=0 max=.9999999 step="any" id="Class_MLPBeta2" name="Class_MLPBeta2">
                            <div><br></div>
                            <label for="Class_MLPEpsilon">Epsilon (float >0):</label>
                            <input type="number" value=.00000001 min=.00000000001 placeholder=.00000001 step="any" id="Class_MLPEpsilon" name="Class_MLPEpsilon">
                    </div>
                </div>
            </div>

            <div id="RF_classifierFields" class="hidden">
                <h3>Random Forest Classifier Model Settings</h3>

                <label for="Class_RFn_estmators">N Estimators - # of trees (integer >=1):</label>
                <input type="number" step=1 value=100 placeholder="100" min=1 id="Class_RFn_estmators" name="Class_RFn_estmators">

                <div class="nonreqHyperparams">
                    <div class="toggle-container">
                        <span><h3>Edit Non-Essential Hyperparameters?</h3></span>
                        <label class="switch">
                            <input type="checkbox" id="nonreqRFClassifierSlider">
                            <span class="slider"></span>
                        </label>
                    </div>

                    <div id="nonreqRFClassifierParams" class="hidden">
                        <label for="Class_RFBoostrap">Bootstrap:</label>
                        <select name="Class_RFBoostrap" id="Class_RFBoostrap">
                            <option value="true">True</option>
                            <option value="false">False</option>
                        </select>
                        <label for="Class_RFoobScore">oob Score:</label>
                        <select name="Class_RFoobScore" id="Class_RFoobScore">
                            <option value="false">False</option>
                            <option value="true">True</option>
                        </select>
                        <label for="Class_RFWarmStart">Warm Start:</label>
                        <select name="Class_RFWarmStart" id="Class_RFWarmStart">
                            <option value="false">False</option>
                            <option value="true">True</option>
                        </select>
                        <div><br></div>
                        <!-- <label for="RFCriterion">Criterion:</label>
                        <select name="RFCriterion" id="RFCriterion">
                            <option value="gini">gini</option>
                            <option value="entropy">entropy</option>
                        </select> -->
                        <label for="Class_RFmin_weight_fraction_leaf">Min Weight Fraction Leaf (float [0.0, 0.5]):</label>
                        <input type="number" value=0 placeholder=0 min="0" max=".5" step="any" id="Class_RFmin_weight_fraction_leaf" name="Class_RFmin_weight_fraction_leaf">
                        <div><br></div>
                        <!-- <label for="RFMaxFeatures">Max Features:</label> -->
                        <label for="Class_RFMaxLeafNodes">Max Leaf Nodes (an integer or leave blank for None):</label>
                        <input type="number" step="1" id="Class_RFMaxLeafNodes" name="Class_RFMaxLeafNodes">
                        <div><br></div>
                        <label for="Class_RFMinImpurityDecrease">Min Impurity Decrease (float):</label>
                        <input type="number" value=0 placeholder=0 step="any" id="Class_RFMinImpurityDecrease" name="Class_RFMinImpurityDecrease">
                        <div><br></div>
                        <label for="Class_RFNJobs">N Jobs (an integer or leave blank for None):</label>
                        <input type="number" step="1" id="Class_RFNJobs" name="Class_RFNJobs">
                        <div><br></div>
                        <!-- <label for="Class_RFRandomState">Random State (an integer or leave blank for None):</label>
                        <input type="number" step="1" id="Class_RFRandomState" name="Class_RFRandomState">
                        <div><br></div> -->
                        <label for="Class_RFVerbose">Verbose (int):</label>
                        <input type="number" value=0 placeholder=0 step="1" id="Class_RFVerbose" name="Class_RFVerbose">

                        <div><br></div>
                        <label for="Class_RFMax_depth">Max Depth - Tree depth (an integer or leave blank for None):</label>
                        <input type="number" step=1 id="Class_RFMax_depth" name="Class_RFMax_depth">
                        <div><br></div>
                        <label for="Class_min_samples_split">Min Samples Split - Min samples per split (integer or float):</label>
                        <input type="number" value=2 placeholder="2" id="Class_min_samples_split" name="Class_min_samples_split">
                        <div><br></div>
                        <label for="Class_min_samples_leaf">Min Samples Leaf - Min samples per leaf (integer or float):</label>
                        <input type="number" value=1 placeholder=1 id="Class_min_samples_leaf" name="Class_min_samples_leaf">
                    </div>
                </div>
            </div>

            <div id="SVC_classifierFields" class="hidden">
                <h3>SVC Classifier Model Settings</h3>


                <label for="SVC_C">C - Regularization (float >0):</label>
                <input type="number" value=1 placeholder=1 min="0.0000001" step="any" placeholder=1.0 id="SVC_C" name="SVC_C">
                <div><br></div>
                <label for="Class_kernel">Kernel:</label>
                <select name="Class_kernel" id="Class_kernel">
                    <option value="rbf">rbf</option>
                    <option value="linear">linear</option>
                    <option value="poly">poly</option>
                    <option value="sigmoid">sigmoid</option>
                    <option value="precomputed">precomputed</option>
                </select>
                <div id="Class_polykernelFields" class="hidden">
                    <div><br></div>
                    <label for="Class_polyDegree">Degree of poly kernel function (int):</label>
                    <input type="number" step=1 value=3 placeholder=3.0 id="Class_polyDegree" name="Class_polyDegree">
                </div>
                <div id="SVCGamma">
                    <div><br></div>
                    <label for="Gamma">Gamma - enter 'auto', 'scale', or a float:</label>
                    <input type="text"  value='scale' placeholder='scale' id="SVCGamma" name="SVCGamma">
                </div>

                <div class="nonreqHyperparams">
                    <div><br></div>
                    <div class="toggle-container">
                        <span><h3>Edit Non-Essential Hyperparameters?</h3></span>
                        <label class="switch">
                            <input type="checkbox" id="nonreqSVCClassifierSlider">
                            <span class="slider"></span>
                        </label>
                    </div>
                    <div id="nonreqSVCClassifierParams" class="hidden">

                        <label for="SVCshrinking">Shrinking:</label>
                        <select name="SVCshrinking" id="SVCshrinking">
                            <option value="true">True</option>
                            <option value="false">False</option>
                        </select>
                        <label for="SVCprobability">Probability:</label>
                        <select name="SVCprobability" id="SVCprobability">
                            <option value="false">False</option>
                            <option value="true">True</option>
                        </select>
                        <br>
                        <label for="SVCBreakTies">Break Ties:</label>
                        <select name="SVCBreakTies" id="SVCBreakTies">
                            <option value="false">False</option>
                            <option value="true">True</option>
                        </select>
                        <label for="SVCverbose">Verbose:</label>
                        <select name="SVCverbose" id="SVCverbose">
                            <option value="false">False</option>
                            <option value="true">True</option>
                        </select>
                        <div><br></div>
                        <label for="SVCcoef0">coef0 (float):</label>
                        <input type="number" step="any" value=0 placeholder="0" id="SVCcoef0" name="SVCcoef0">
                        <div><br></div>
                        <label for="SVCtol">tol(float):</label>
                        <input type="number" step="any" value=.001 placeholder=.001 id="SVCtol" name="SVCtol">
                        <div><br></div>
                        <label for="SVCCacheSize">Cache Size (float):</label>
                        <input type="number" step="any" value=200 placeholder=200 id="SVCCacheSize" name="SVCCacheSize">
                        <div><br></div>
                        <label for="SVCClassWeight">Class Weight (enter a dictionary or 'balanced')</label>
                        <input type="text" id="SVCClassWeight" name="SVCClassWeight">
                        <div><br></div>
                        <label for="SVCmaxIter">Max Iterations (int):</label>
                        <input type="number" step=1 value=-1 placeholder=-1 id="SVCmaxIter" name="SVCmaxIter">
                        <div><br></div>
                        <label for="SVCdecisionFunctionShape">Decision Function Shape:</label>
                        <select name="SVCdecisionFunctionShape" id="SVCdecisionFunctionShape">
                            <option value="ovr">ovr</option>
                            <option value="ovo">ovo</option>
                        </select>
                        <div><br></div>
                        <!-- <label for="SVCrandomState">Random State (an integer or leave blank for None):</label>
                        <input type="number" step=1 id="SVCrandomState" name="SVCrandomState">
                        <div><br></div> -->
                    </div>
                </div>
            </div>

            <!-- Cluster Hyperparameters -->
            <div id="AgglomerativeFields" class="hidden">
                <h3>Agglomerative Model Settings</h3>
                <label for="Agg_n_clusters">Number of Clusters (int or blank for None):</label>
                <input type="number" value=2 placeholder=2 step=1 id="Agg_n_clusters" name="Agg_n_clusters">
                

                <div class="nonreqHyperparams">
                    <div><br></div>
                    <div class="toggle-container">
                        <span><h3>Edit Non-Essential Hyperparameters?</h3></span>
                        <label class="switch">
                            <input type="checkbox" id="nonreqAgglomerativeSlider">
                            <span class="slider"></span>
                        </label>
                    </div>
                    <div id="nonreqAgglomerativeParams" class="hidden">

                        <label for="Agglinkage">Linkage:</label>
                        <select name="Agglinkage" id="Agglinkage">
                            <option value="ward">ward</option>
                            <option value="complete">complete</option>
                            <option value="average">average</option>
                            <option value="single">single</option>
                        </select>

                        <label for="aggcompute_distances">Compute Distances:</label>
                        <select name="aggcompute_distances" id="aggcompute_distances">
                            <option value="false">False</option>
                            <option value="true">True</option>
                        </select>
                        <div><br></div>
                        <label for="aggcompute_full_tree">Compute Full Tree:</label>
                        <select name="aggcompute_full_tree" id="aggcompute_full_tree">
                            <option value="auto">Auto</option>
                            <option value="false">False</option>
                            <option value="true">True</option>
                        </select>

                        <div><br></div>
                        <label for="Aggdistance_threshold">Distance Threshold (float or None):</label>
                        <input type="number" step=any id="Aggdistance_threshold" name="Aggdistance_threshold">
                        <div><br></div>
                        <label for="Aggmetric">Metric:</label>
                        <input type="text" id="Aggmetric" value="euclidean" placeholder="euclidean" name="Aggmetric">
                        <div><br></div>
                        <label for="Aggmemory">Memory:</label>
                        <input type="text" id="Aggmemory" name="Aggmemory">
                        <div><br></div>
                        <label for="Aggconnectivity">Connectivity</label>
                        <input type="text" id="Aggconnectivity" name="Aggconnectivity">
                    </div>
                </div>
            </div>

            <div id="GaussianFields" class="hidden">

                <h3>Gaussian Mixture Model Settings</h3>
                <label for="Gaun_components">Number of components:</label>
                <input type="number" value=1.0 placeholder=1 min=0 id="Gaun_components" name="Gaun_components">
                

                <div class="nonreqHyperparams">
                    <div><br></div>
                    <div class="toggle-container">
                        <span><h3>Edit Non-Essential Hyperparameters?</h3></span>
                        <label class="switch">
                            <input type="checkbox" id="nonreqGaussianSlider">
                            <span class="slider"></span>
                        </label>
                    </div>
                    <div id="nonreqGaussianParams" class="hidden">
                        <label for="GauWarmStart">Warm Start:</label>
                        <select name="GauWarmStart" id="GauWarmStart">
                            <option value="false">False</option>
                            <option value="true">True</option>
                        </select>
                        <label for="Gauinit_params">Init Params:</label>
                        <select name="Gauinit_params" id="Gauinit_params">
                            <option value="kmeans">kmeans</option>
                            <option value="k-means++">k-means++</option>
                            <option value="random">random</option>
                            <option value="random_from_data">random_from_data</option>
                        </select>
                        <div><br></div>
                        <label for="Gaucovariance_type">Covariance Type:</label>
                        <select name="Gaucovariance_type" id="Gaucovariance_type">
                            <option value="full">full</option>
                            <option value="tied">tied</option>
                            <option value="diag">diag</option>
                            <option value="spherical">spherical</option>
                        </select>
                        <div><br></div>
                        <label for="GauMax_iter">Max Iterations:</label>
                        <input type="number" step=1 value=100 placeholder=100 min=1 id="GauMax_iter" name="GauMax_iter">
                        <div><br></div>
                        <label for="Gaun_init">N Init:</label>
                        <input type="number" step=1 value=1 placeholder=1 min=1 id="Gaun_init" name="Gaun_init">
                        <div><br></div>
                        <label for="GauTol">Tol (float >0):</label>
                        <input type="number" value=0.001 placeholder=.001 min="0.0000001" step="any" id="GauTol" name="GauTol">
                        <div><br></div>
                        <label for="Gaureg_covar">reg_covar:</label>
                        <input type="number" value=0.000001 placeholder=.000001 min="0.0000001" step="any" id="Gaureg_covar" name="Gaureg_covar">
                        <div><br></div>
                        <label for="Gauweights_init">Weights Init:</label>
                        <input type="text" id="Gauweights_init" name="Gauweights_init">
                        <div><br></div>
                        <label for="Gaumeans_init">Means Init:</label>
                        <input type="text" id="Gaumeans_init" name="Gaumeans_init">
                        <div><br></div>
                        <label for="Gauprecisions_init">Precisions Init</label>
                        <input type="text" id="Gauprecisions_init" name="Gauprecisions_init">
                        <div><br></div>
                        <label for="GauVerbose">Verbose:</label>
                        <input type="number" step=1 value=0 placeholder=0 min=0 id="GauVerbose" name="GauVerbose">
                        <div><br></div>
                        <label for="GauVerbose_interval">Verbose Interval:</label>
                        <input type="number" step=1 value=10 placeholder=10 min=0 id="GauVerbose_interval" name="GauVerbose_interval">
                        <div><br></div>
                        
                    </div>
                </div>
            </div>

            <div id="KmeansFields" class="hidden">
                <h3>K Means Model Settings</h3>
                <label for="Kmeansn_clusters">Number of Clusters:</label>
                <input type="number" value=8 placeholder=8 step=1 id="Kmeansn_clusters" name="Kmeansn_clusters">
                

                <div class="nonreqHyperparams">
                    <div><br></div>
                    <div class="toggle-container">
                        <span><h3>Edit Non-Essential Hyperparameters?</h3></span>
                        <label class="switch">
                            <input type="checkbox" id="nonreqKmeansSlider">
                            <span class="slider"></span>
                        </label>
                    </div>
                    <div id="nonreqKmeansParams" class="hidden">

                        <label for="Kmeansalgorithm">Algorithm:</label>
                        <select name="Kmeansalgorithm" id="Kmeansalgorithm">
                            <option value="lloyd">lloyd</option>
                            <option value="elkan">elkan</option>
                            <option value="auto">auto</option>
                            <option value="full">full</option>
                        </select>

                        <label for="kmeansCopyX">Copy X:</label>
                        <select name="kmeansCopyX" id="kmeansCopyX">
                            <option value="true">True</option>
                            <option value="false">False</option>
                        </select>
                        <div><br></div>
                        <label for="kmeansmax_iter">Max Iterations:</label>
                        <input type="number" value=300 placeholder=300 step=1 id="kmeansmax_iter" name="kmeansmax_iter">
                        <div><br></div>
                        <label for="kmeansverbose">Verbose:</label>
                        <input type="number" value=0 placeholder=0 step=1 id="kmeansverbose" name="kmeansverbose">
                        <div><br></div>
                        <label for="kmeanstol">tol:</label>
                        <input type="number" value=0.0001 placeholder=0.0001 step=any id="kmeanstol" name="kmeanstol">
                        <div><br></div>
                        <label for="kmeansInit">Init:</label>
                        <input type="text" value=k-means++ placeholder="k-means++" id="kmeansInit" name="kmeansInit">
                        <div><br></div>
                        <label for="kmeansn_init">n_init (int or 'auto'):</label>
                        <input type="text" value=auto placeholder="auto" id="kmeansn_init" name="kmeansn_init">
                    </div>
                </div>
            </div>
            <div><br></div>


            
            <div class="toggle-container">
                <h3>Does Output Have Units?</h3>
                <label class="switch">
                    <input type="checkbox" id="unitToggle">
                    <span class="slider"></span>
                </label>
                <div id="units" class="hidden">
                    <label for="unitName">Enter Unit (u* for µ):</label>
                        <input type="text" id='unitName' name="unitName"> 
                    
                </div>
            </div>
            <div><br></div>
            

            <div class="scaling-container">
                <label for="sigfig">Select Number of Signifigant Figures</label>
                <input type="number" id="sigfig" name="sigfig" value=3 placeholder="3">
            </div>
            <br>

            <!-- Run My Model button -->
            <div style="display: flex; gap: 12px; align-items: center; margin-bottom: 12px;">
                <button class='processButton processButton--compact' type="submit" id="processButton">Run This Model</button>
                <button type="button" class='secondary-button' id="stopSimpleButton" style="display: none; padding: 12px 24px;">Stop Model</button>
            </div>
            <div id="loading" class="hidden" role="status" aria-live="polite" aria-atomic="true" aria-label="Loading status">
            </div>


            <div id="errorDiv"></div>
            </div>
        </div>  
        </form>
        
        <!-- Results section - right side -->
        <div class="visualSection" id="simpleModelingResults">
            <div class="model-card">
                <h2 style="margin-top: 0; margin-bottom: 20px;">Modeling Results</h2>
                <div id="NumericResultDiv" role="region" aria-label="Regression model results" class="hidden">
                    <div id="imageSelector"> </div>
                </div>
                <div id="ClusterResultDiv" role="region" aria-label="Clustering model results" class="hidden"></div>
                <div id="ClassifierResultDiv" role="region" aria-label="Classification model results" class="hidden"></div>
                <div id="resultsPlaceholder" style="padding: 40px; text-align: center; color: #666; font-style: italic;">
                    <p>Results will appear here after you run a model.</p>
                </div>
            </div>
        </div>
        </div>
        </div>
        <!-- End Simple Modeling Section -->
        
        <!-- Advanced Modeling Section -->
        <div id="advancedModelingSection" class="modeling-mode-section hidden">
            <div class="model-layout">
                <form id="advancedOptimizationForm">
                    <div class="modelSection">
                        <!-- Descriptive text about column selection -->
                        <div style="margin-bottom: 16px; padding: 12px; background-color: #f8f9fa; border-radius: 6px; border-left: 3px solid #357a53;">
                            <span id="advancedModelingSelectionNote" class="modeling-selection-note" style="font-style: italic; color: #2c3e50;"></span>
                        </div>
                        
                        <!-- Model Selection -->
                        <div class="preprocess-card" style="margin-bottom: 20px;">
                            <h3 style="margin-top: 0; margin-bottom: 12px;">Select Model</h3>
                            <p class="field-note" style="margin-bottom: 16px;">Choose the model algorithm for your selected output type.</p>
                            
                            <div id="advancedNumericModels" class="hidden">
                                <label for="advancedNModels" style="display: block; margin-bottom: 8px; font-weight: 600;">Regression Model:</label>
                                <select name="advancedNModels" id="advancedNModels" style="width: 100%; max-width: 400px; padding: 8px; border: 1px solid #ccc; border-radius: 4px;">
                                    <option value="" disabled selected>-- Select an option --</option>
                                    <optgroup label="Most Common">
                                        <option value="Linear">Linear</option>
                                        <option value="Ridge">Ridge</option>
                                        <option value="Lasso">Lasso</option>
                                        <option value="ElasticNet">Elastic Net</option>
                                        <option value="RF">Random Forest</option>
                                        <option value="gradient_boosting">Gradient Boosting</option>
                                        <option value="SVM">Support Vector Machine (SVR)</option>
                                        <option value="MLP">Multi-Layer Perceptron</option>
                                        <option value="K-Nearest">K-Nearest Neighbors</option>
                                        <option value="ExtraTrees">Extra Trees</option>
                                    </optgroup>
                                    <optgroup label="Additional Models (Alphabetical)">
                                        <option value="AdaBoost">AdaBoost Regressor</option>
                                        <option value="ARDRegression">ARD Regression</option>
                                        <option value="Bagging">Bagging Regressor</option>
                                        <option value="BayesianRidge">Bayesian Ridge</option>
                                        <option value="DecisionTree">Decision Tree Regressor</option>
                                        <option value="ElasticNetCV">Elastic Net CV</option>
                                        <option value="HistGradientBoosting">Histogram Gradient Boosting</option>
                                        <option value="Huber">Huber Regressor</option>
                                        <option value="LARS">LARS</option>
                                        <option value="LARSCV">LARS CV</option>
                                        <option value="LassoCV">Lasso CV</option>
                                        <option value="LinearSVR">Linear SVR</option>
                                        <option value="NuSVR">Nu-SVR</option>
                                        <option value="OMP">Orthogonal Matching Pursuit</option>
                                        <option value="PassiveAggressive">Passive Aggressive Regressor</option>
                                        <option value="Quantile">Quantile Regressor</option>
                                        <option value="RadiusNeighbors">Radius Neighbors Regressor</option>
                                        <option value="RANSAC">RANSAC Regressor</option>
                                        <option value="RidgeCV">Ridge CV</option>
                                        <option value="SGD">SGD Regressor</option>
                                        <option value="TheilSen">Theil-Sen Regressor</option>
                                    </optgroup>
                                </select>
                            </div>
                            
                            <div id="advancedClusterModels" class="hidden">
                                <label for="advancedClModels" style="display: block; margin-bottom: 8px; font-weight: 600;">Clustering Model:</label>
                                <select name="advancedClModels" id="advancedClModels" style="width: 100%; max-width: 400px; padding: 8px; border: 1px solid #ccc; border-radius: 4px;">
                                    <option value="" disabled selected>-- Select an option --</option>
                                    <optgroup label="Most Common">
                                        <option value="kmeans">K-Means</option>
                                        <option value="dbscan">DBSCAN</option>
                                        <option value="agglo">Agglomerative</option>
                                        <option value="gmm">Gaussian Mixture</option>
                                        <option value="spectral">Spectral Clustering</option>
                                        <option value="birch">BIRCH</option>
                                        <option value="affinity_propagation">Affinity Propagation</option>
                                        <option value="bisecting_kmeans">Bisecting K-Means</option>
                                        <option value="hdbscan">HDBSCAN</option>
                                        <option value="meanshift">Mean Shift</option>
                                    </optgroup>
                                    <optgroup label="Additional Models (Alphabetical)">
                                        <option value="minibatch_kmeans">Mini-Batch K-Means</option>
                                        <option value="optics">OPTICS</option>
                                    </optgroup>
                                </select>
                            </div>
                            
                            <div id="advancedClassifierModels" class="hidden">
                                <label for="advancedClassModels" style="display: block; margin-bottom: 8px; font-weight: 600;">Classification Model:</label>
                                <select name="advancedClassModels" id="advancedClassModels" style="width: 100%; max-width: 400px; padding: 8px; border: 1px solid #ccc; border-radius: 4px;">
                                    <option value="" disabled selected>-- Select an option --</option>
                                    <optgroup label="Most Common">
                                        <option value="Logistic_classifier">Logistic Classifier</option>
                                        <option value="RF_classifier">Random Forest Classifier</option>
                                        <option value="SVC_classifier">SVC Classifier</option>
                                        <option value="MLP_classifier">MLP Classifier</option>
                                        <option value="ExtraTrees_classifier">Extra Trees Classifier</option>
                                        <option value="SGD_classifier">SGD Classifier</option>
                                        <option value="AdaBoost_classifier">AdaBoost Classifier</option>
                                        <option value="GradientBoosting_classifier">Gradient Boosting Classifier</option>
                                        <option value="KNeighbors_classifier">K-Neighbors Classifier</option>
                                    </optgroup>
                                    <optgroup label="Multi-Output Models (Multiple Targets)">
                                        <option value="GaussianNB_classifier">Gaussian Naive Bayes</option>
                                        <option value="BernoulliNB_classifier">Bernoulli Naive Bayes</option>
                                        <option value="CategoricalNB_classifier">Categorical Naive Bayes</option>
                                        <option value="ComplementNB_classifier">Complement Naive Bayes</option>
                                        <option value="MultinomialNB_classifier">Multinomial Naive Bayes</option>
                                    </optgroup>
                                    <optgroup label="Additional Models (Alphabetical)">
                                        <option value="Bagging_classifier">Bagging Classifier</option>
                                        <option value="DecisionTree_classifier">Decision Tree Classifier</option>
                                        <option value="HistGradientBoosting_classifier">Histogram Gradient Boosting</option>
                                        <option value="LDA_classifier">Linear Discriminant Analysis</option>
                                        <option value="LinearSVC_classifier">Linear SVC</option>
                                        <option value="NuSVC_classifier">Nu-SVC</option>
                                        <option value="PassiveAggressive_classifier">Passive Aggressive Classifier</option>
                                        <option value="QDA_classifier">Quadratic Discriminant Analysis</option>
                                        <option value="Ridge_classifier">Ridge Classifier</option>
                                    </optgroup>
                                </select>
                            </div>
                        </div>
                        
                        <div class="model-card">
                            <h2 style="margin-top: 0; margin-bottom: 20px;">Advanced Model Configuration</h2>
                            
                            <!-- Model Hyperparameters Section -->
                            <div style="margin-bottom: 30px;">
                                <h3 style="margin-top: 0; margin-bottom: 12px; font-size: 1.2em; font-weight: 600;">Model Hyperparameters</h3>
                                <p class="field-note" style="margin-bottom: 16px;">Configure all model hyperparameters. Essential hyperparameters appear first, followed by non-essential options. These sections will appear based on the model selected above.</p>
                                
                                <!-- Note: Hyperparameter fields are duplicated from the old advancedOptimization container -->
                                <!-- They are also present there for backward compatibility, but these are the active ones -->
                                <!-- The hyperparameter fields start below and are copied from the old Advanced page -->
                                
                            <!-- Ridge Hyperparameters -->
                            <div id="advancedRidgeFields" class="hidden">
                                <h3>Ridge Model Settings</h3>
                                <p class="field-note" style="margin-bottom: 12px; margin-top: 0;">Ridge regression adds L2 regularization to prevent overfitting. Higher alpha values increase regularization.</p>
                                <label for="advancedRidgeAlpha">Alpha - regularization strength (float >=0):</label>
                                <input type="number" value=1.0 placeholder=1 min=0 id="advancedRidgeAlpha" name="advancedRidgeAlpha">
                                <div class="nonreqHyperparams">
                                    <div><br></div>
                                    <div class="toggle-container">
                                        <span><h3>Edit Non-Essential Hyperparameters?</h3></span>
                                        <label class="switch">
                                            <input type="checkbox" id="advancedNonreqRidgeSlider">
                                            <span class="slider"></span>
                                        </label>
                                    </div>
                                    <div id="advancedNonreqRidgeParams" class="hidden">
                                        <label for="RidgeFitIntersept">Fit Intercept:</label>
                                        <select name="RidgeFitIntersept" id="RidgeFitIntersept">
                                            <option value="true">True</option>
                                            <option value="false">False</option>
                                        </select>
                                        <label for="RidgeNormalize">Normalize:</label>
                                        <select name="RidgeNormalize" id="RidgeNormalize">
                                            <option value="false">False</option>
                                            <option value="true">True</option>
                                        </select>
                                        <br>
                                        <label for="RidgeCopyX">Copy X:</label>
                                        <select name="RidgeCopyX" id="RidgeCopyX">
                                            <option value="true">True</option>
                                            <option value="false">False</option>
                                        </select>
                                        <label for="RidgePositive">Positive:</label>
                                        <select name="RidgePositive" id="RidgePositive">
                                            <option value="false">False</option>
                                            <option value="true">True</option>
                                        </select>
                                        <div><br></div>
                                        <label for="RidgeSolver">Solver - Optimization method:</label>
                                        <select name="solver" id="RidgeSolver">
                                            <option value="auto">auto</option>
                                            <option value="svd">svd</option>
                                            <option value="cholesky">cholesky</option>
                                            <option value="lsqr">lsqr</option>
                                            <option value="sparse_cg">sparse_cg</option>
                                            <option value="sag">sag</option>
                                            <option value="saga">saga</option>
                                            <option value="lbfgs">lbfgs</option>
                                        </select>
                                        <div><br></div>
                                        <label for="RidgeMaxIter">Max Iterations (integer >=1):</label>
                                        <input type="number" step=1 min=1 id="RidgeMaxIter" name="RidgeMaxIter">
                                        <div><br></div>
                                        <label for="RidgeTol">Tol (float >0):</label>
                                        <input type="number" value=.0001 placeholder=.0001 min="0.0000001" step="any" id="RidgeTol" name="RidgeTol">
                                    </div>
                                </div>
                            </div>
                            
                            <!-- Lasso Hyperparameters -->
                            <div id="advancedLassoFields" class="hidden">
                                <h3>Lasso Model Settings</h3>
                                <p class="field-note" style="margin-bottom: 12px; margin-top: 0;">Lasso regression uses L1 regularization, which can set coefficients to zero for feature selection.</p>
                                <label for="advancedLassoAlpha">Alpha - regularization strength (float >=0):</label>
                                <input type="number" value=1 min=0 placeholder=1 id="advancedLassoAlpha" name="advancedLassoAlpha">
                                <div class="nonreqHyperparams">
                                    <div><br></div>
                                    <div class="toggle-container">
                                        <span><h3>Edit Non-Essential Hyperparameters?</h3></span>
                                        <label class="switch">
                                            <input type="checkbox" id="advancedNonreqLassoSlider">
                                            <span class="slider"></span>
                                        </label>
                                    </div>
                                    <div id="advancedNonreqLassoParams" class="hidden">
                                        <label for="LassoFitIntersept">Fit Intercept:</label>
                                        <select name="LassoFitIntersept" id="LassoFitIntersept">
                                            <option value="true">True</option>
                                            <option value="false">False</option>
                                        </select>
                                        <label for="LassoPrecompute">Precompute:</label>
                                        <select name="LassoPrecompute" id="LassoPrecompute">
                                            <option value="false">False</option>
                                            <option value="true">True</option>
                                        </select>
                                        <label for="LassoCopyX">Copy X:</label>
                                        <select name="LassoCopyX" id="LassoCopyX">
                                            <option value="true">True</option>
                                            <option value="false">False</option>
                                        </select>
                                        <div><br></div>
                                        <label for="LassoWarmStart">Warm Start:</label>
                                        <select name="LassoWarmStart" id="LassoWarmStart">
                                            <option value="false">False</option>
                                            <option value="true">True</option>
                                        </select>
                                        <label for="LassoPositive">Positive:</label>
                                        <select name="LassoPositive" id="LassoPositive">
                                            <option value="false">False</option>
                                            <option value="true">True</option>
                                        </select>
                                        <div><br></div>
                                        <label for="LassoMax_iter">Max Iterations - number of iterations (integer >=1):</label>
                                        <input type="number" step=1 value=1000 placeholder=1000 min=1 id="LassoMax_iter" name="max_iter">
                                        <div><br></div>
                                        <label for="LassoTol">Tol (float >0):</label>
                                        <input type="number" value=0.0001 placeholder=.0001 min="0.0000001" step="any" id="LassoTol" name="LassoTol">
                                        <div><br></div>
                                        <label for="LassoSelection">Selection:</label>
                                        <select name="LassoSelection" id="LassoSelection">
                                            <option value="cyclic">Cyclic</option>
                                            <option value="random">Random</option>
                                        </select>
                                    </div>
                                </div>
                            </div>
                            
                            <!-- Logistic Non-Essential Hyperparameters -->
                            <div id="advancedLogisticFields" class="hidden">
                                <div class="nonreqHyperparams">
                                    <div class="toggle-container">
                                        <span><h3>Edit Non-Essential Hyperparameters?</h3></span>
                                        <label class="switch">
                                            <input type="checkbox" id="advancedNonreqLogisticSlider">
                                            <span class="slider"></span>
                                        </label>
                                    </div>
                                    <div id="advancedNonreqLogisticParams" class="hidden">
                                        <label for="LogisticDual">Dual:</label>
                                        <select name="LogisticDual" id="LogisticDual">
                                            <option value="false">False</option>
                                            <option value="true">True</option>
                                        </select>
                                        <label for="LogisticFitIntercept">Fit Intercept:</label>
                                        <select name="LogisticFitIntercept" id="LogisticFitIntercept">
                                            <option value="true">True</option>
                                            <option value="false">False</option>
                                        </select>
                                        <label for="LogisticSolver">Solver:</label>
                                        <select name="LogisticSolver" id="LogisticSolver">
                                            <option value="lbfgs">lbfgs</option>
                                            <option value="newton-cg">newton-cg</option>
                                            <option value="liblinear">liblinear</option>
                                            <option value="sag">sag</option>
                                            <option value="saga">saga</option>
                                        </select>
                                        <div><br></div>
                                        <label for="LogisticMultiClass">Multi Class:</label>
                                        <select name="LogisticMultiClass" id="LogisticMultiClass">
                                            <option value="auto">auto</option>
                                            <option value="ovr">ovr</option>
                                            <option value="multinomial">multinomial</option>
                                        </select>
                                        <label for="LogisticWarmStart">Warm Start:</label>
                                        <select name="LogisticWarmStart" id="LogisticWarmStart">
                                            <option value="false">False</option>
                                            <option value="true">True</option>
                                        </select>
                                        <div><br></div>
                                        <label for="CLogistic">C - Regularization (float >0):</label>
                                        <input type="number" value=1 placeholder=1.0 min="0.0000001" step="any" id="CLogistic" name="CLogistic">
                                        <div><br></div>
                                        <label for="penalty">Penalty:</label>
                                        <select name="penalty" id="penalty">
                                            <option value="l2">l2</option>
                                            <option value="l1">l1</option>
                                            <option value="elasticnet">elasticnet</option>
                                            <option value="None">None</option>
                                        </select>
                                        <div><br></div>
                                        <label for="LogisticTol">Tol (float >0):</label>
                                        <input type="number" value=0.0001 placeholder=.0001 min="0.0000001" step="any" id="LogisticTol" name="LogisticTol">
                                        <div><br></div>
                                        <label for="Logisticintercept_scaling">Intercept Scaling:</label>
                                        <input type="number" value=1 placeholder=1 step="any" id="Logisticintercept_scaling" name="Logisticintercept_scaling">
                                        <div><br></div>
                                        <label for="LogisticClassWeight">Class Weight (dict or 'balanced'):</label>
                                        <input type="text" id="LogisticClassWeight" name="LogisticClassWeight">
                                        <div><br></div>
                                        <label for="LogisticMaxIterations">Max Iterations (integer >=1):</label>
                                        <input type="number" step=1 min="1" value=100 placeholder="100" id="LogisticMaxIterations" name="LogisticMaxIterations">
                                        <div><br></div>
                                        <label for="LogisticVerbose">Verbose (int):</label>
                                        <input type="number" step=1 value=0 placeholder="0" id="LogisticVerbose" name="LogisticVerbose">
                                        <div><br></div>
                                        <label for="LogisticNJobs">N Jobs (int):</label>
                                        <input type="number" step=1 id="LogisticNJobs" name="LogisticNJobs">
                                        <div><br></div>
                                        <label for="Logisticl1Ratio">L1 Ratio (float [0, 1]):</label>
                                        <input type="number" min="0.0000001" step="any" max=1 id="Logisticl1Ratio" name="Logisticl1Ratio">
                                    </div>
                                </div>
                            </div>
                            
                            <!-- SVM Hyperparameters -->
                            <div id="advancedSVMFields" class="hidden">
                                <h3>SVM Model Settings</h3>
                                <label for="advancedC">C - Regularization (float >0):</label>
                                <input type="number" value=1 placeholder=1 min="0.0000001" step="any" id="advancedC" name="advancedC">
                                <div><br></div>
                                <label for="advancedKernel">Kernel:</label>
                                <select name="advancedKernel" id="advancedKernel">
                                    <option value="rbf">rbf</option>
                                    <option value="linear">linear</option>
                                    <option value="poly">poly</option>
                                    <option value="sigmoid">sigmoid</option>
                                    <option value="precomputed">precomputed</option>
                                </select>
                                <div id="advancedPolykernelFields" class="hidden">
                                    <div><br></div>
                                    <label for="advancedPolyDegree">Degree of poly kernel function (int):</label>
                                    <input type="number" step=1 value=3 placeholder=3.0 id="advancedPolyDegree" name="advancedPolyDegree">
                                </div>
                                <div id="advancedSvmGamma">
                                    <div><br></div>
                                    <label for="advancedGamma">Gamma - enter 'auto', 'scale', or a float:</label>
                                    <input type="text" value='scale' placeholder='scale' id="advancedGamma" name="advancedGamma">
                                </div>
                                <div class="nonreqHyperparams">
                                    <div><br></div>
                                    <div class="toggle-container">
                                        <span><h3>Edit Non-Essential Hyperparameters?</h3></span>
                                        <label class="switch">
                                            <input type="checkbox" id="advancedNonreqSVMSlider">
                                            <span class="slider"></span>
                                        </label>
                                    </div>
                                    <div id="advancedNonreqSVMParams" class="hidden">
                                        <label for="SVMshrinking">Shrinking:</label>
                                        <select name="SVMshrinking" id="SVMshrinking">
                                            <option value="true">True</option>
                                            <option value="false">False</option>
                                        </select>
                                        <label for="SVMprobability">Probability:</label>
                                        <select name="SVMprobability" id="SVMprobability">
                                            <option value="false">False</option>
                                            <option value="true">True</option>
                                        </select>
                                        <br>
                                        <label for="SVMBreakTies">Break Ties:</label>
                                        <select name="SVMBreakTies" id="SVMBreakTies">
                                            <option value="false">False</option>
                                            <option value="true">True</option>
                                        </select>
                                        <label for="SVMverbose">Verbose:</label>
                                        <select name="SVMverbose" id="SVMverbose">
                                            <option value="false">False</option>
                                            <option value="true">True</option>
                                        </select>
                                        <div><br></div>
                                        <label for="SVMcoef0">coef0 (float):</label>
                                        <input type="number" step="any" value=0 placeholder="0" id="SVMcoef0" name="SVMcoef0">
                                        <div><br></div>
                                        <label for="SVMtol">tol(float):</label>
                                        <input type="number" step="any" value=.001 placeholder=.001 id="SVMtol" name="SVMtol">
                                        <div><br></div>
                                        <label for="SVMCacheSize">Cache Size (float):</label>
                                        <input type="number" step="any" value=200 placeholder=200 id="SVMCacheSize" name="SVMCacheSize">
                                        <div><br></div>
                                        <label for="SVMClassWeight">Class Weight (enter a dictionary or 'balanced')</label>
                                        <input type="text" id="SVMClassWeight" name="SVMClassWeight">
                                        <div><br></div>
                                        <label for="SVMmaxIter">Max Iterations (int):</label>
                                        <input type="number" step=1 value=-1 placeholder=-1 id="SVMmaxIter" name="SVMmaxIter">
                                        <div><br></div>
                                        <label for="SVMdecisionFunctionShape">Decision Function Shape:</label>
                                        <select name="SVMdecisionFunctionShape" id="SVMdecisionFunctionShape">
                                            <option value="ovr">ovr</option>
                                            <option value="ovo">ovo</option>
                                        </select>
                                    </div>
                                </div>
                            </div>
                            
                            <!-- Random Forest Hyperparameters -->
                            <div id="advancedRFFields" class="hidden">
                                <h3>Random Forest Model Settings</h3>
                                <p class="field-note" style="margin-bottom: 12px; margin-top: 0;">Random Forest combines multiple decision trees. More trees generally improve performance but increase computation time.</p>
                                <label for="advancedRFn_estmators">N Estimators - # of trees (integer >=1):</label>
                                <input type="number" step=1 value=100 placeholder="100" min=1 id="advancedRFn_estmators" name="advancedRFn_estmators">
                                <div class="nonreqHyperparams">
                                    <div><br></div>
                                    <div class="toggle-container">
                                        <span><h3>Edit Non-Essential Hyperparameters?</h3></span>
                                        <label class="switch">
                                            <input type="checkbox" id="advancedNonreqRFSlider">
                                            <span class="slider"></span>
                                        </label>
                                    </div>
                                    <div id="advancedNonreqRFParams" class="hidden">
                                        <label for="RFBoostrap">Bootstrap:</label>
                                        <select name="RFBoostrap" id="RFBoostrap">
                                            <option value="true">True</option>
                                            <option value="false">False</option>
                                        </select>
                                        <label for="RFoobScore">oob Score:</label>
                                        <select name="RFoobScore" id="RFoobScore">
                                            <option value="false">False</option>
                                            <option value="true">True</option>
                                        </select>
                                        <label for="RFWarmStart">Warm Start:</label>
                                        <select name="RFWarmStart" id="RFWarmStart">
                                            <option value="false">False</option>
                                            <option value="true">True</option>
                                        </select>
                                        <div><br></div>
                                        <label for="RFmin_weight_fraction_leaf">Min Weight Fraction Leaf (float [0.0, 0.5]):</label>
                                        <input type="number" value=0 placeholder=0 min="0" max=".5" step="any" id="RFmin_weight_fraction_leaf" name="RFmin_weight_fraction_leaf">
                                        <div><br></div>
                                        <label for="RFMaxLeafNodes">Max Leaf Nodes (an integer or leave blank for None):</label>
                                        <input type="number" step="1" id="RFMaxLeafNodes" name="RFMaxLeafNodes">
                                        <div><br></div>
                                        <label for="RFMinImpurityDecrease">Min Impurity Decrease (float):</label>
                                        <input type="number" value=0 placeholder=0 step="any" id="RFMinImpurityDecrease" name="RFMinImpurityDecrease">
                                        <div><br></div>
                                        <label for="RFNJobs">N Jobs (an integer or leave blank for None):</label>
                                        <input type="number" step="1" id="RFNJobs" name="RFNJobs">
                                        <div><br></div>
                                        <label for="RFVerbose">Verbose (int):</label>
                                        <input type="number" value=0 placeholder=0 step="1" id="RFVerbose" name="RFVerbose">
                                        <div><br></div>
                                        <label for="RFMax_depth">Max Depth - Tree depth (an integer or leave blank for None):</label>
                                        <input type="number" step=1 id="RFMax_depth" name="max_depth">
                                        <div><br></div>
                                        <label for="min_samples_split">Min Samples Split - Min samples per split (integer or float):</label>
                                        <input type="number" value=2 placeholder="2" id="min_samples_split" name="min_samples_split">
                                        <div><br></div>
                                        <label for="min_samples_leaf">Min Samples Leaf - Min samples per leaf (integer or float):</label>
                                        <input type="number" value=1 placeholder=1 id="min_samples_leaf" name="min_samples_leaf">
                                    </div>
                                </div>
                            </div>
                            
                            <!-- Perceptron Non-Essential Hyperparameters -->
                            <div id="advancedPerceptronFields" class="hidden">
                                <div class="nonreqHyperparams">
                                    <div><br></div>
                                    <div class="toggle-container">
                                        <span><h3>Edit Non-Essential Hyperparameters?</h3></span>
                                        <label class="switch">
                                            <input type="checkbox" id="advancedNonreqPerceptronSlider">
                                            <span class="slider"></span>
                                        </label>
                                    </div>
                                    <div id="advancedNonreqPerceptronParams" class="hidden">
                                        <label for="PerceptronFitIntercept">Fit Intercept:</label>
                                        <select name="PerceptronFitIntercept" id="PerceptronFitIntercept">
                                            <option value="true">True</option>
                                            <option value="false">False</option>
                                        </select>
                                        <label for="PerceptronShuffle">Shuffle:</label>
                                        <select name="PerceptronShuffle" id="PerceptronShuffle">
                                            <option value="true">True</option>
                                            <option value="false">False</option>
                                        </select>
                                        <label for="PerceptronEarlyStopping">Early Stopping:</label>
                                        <select name="PerceptronEarlyStopping" id="PerceptronEarlyStopping">
                                            <option value="false">False</option>
                                            <option value="true">True</option>
                                        </select>
                                        <label for="PerceptronWarmStart">Warm Start:</label>
                                        <select name="PerceptronWarmStart" id="PerceptronWarmStart">
                                            <option value="false">False</option>
                                            <option value="true">True</option>
                                        </select>
                                        <div><br></div>
                                        <label for="PerceptronPenalty">Penalty:</label>
                                        <select name="PerceptronPenalty" id="PerceptronPenalty">
                                            <option value="None">None</option>
                                            <option value="l2">l2</option>
                                            <option value="l1">l1</option>
                                            <option value="elasticnet">elasticnet</option>
                                        </select>
                                        <div><br></div>
                                        <label for="PerceptronAlpha">Alpha (float):</label>
                                        <input type="number" step="any" value=.0001 placeholder=.0001 id="PerceptronAlpha" name="PerceptronAlpha">
                                        <div><br></div>
                                        <label for="PerceptronTol">Tol (float):</label>
                                        <input type="number" step="any" value=.001 placeholder=.001 id="PerceptronTol" name="PerceptronTol">
                                        <div><br></div>
                                        <label for="PerceptronVerbose">Verbose (int):</label>
                                        <input type="number" step="1" value=0 placeholder=0 id="PerceptronVerbose" name="PerceptronVerbose">
                                        <div><br></div>
                                        <label for="PerceptronNJobs">N Jobs (an integer or or leave blank for None):</label>
                                        <input type="number" step="1" id="PerceptronNJobs" name="PerceptronNJobs">
                                        <div><br></div>
                                        <label for="PerceptronValidationFraction">Validation Fraction (float):</label>
                                        <input type="number" step="any" value=.1 placeholder=.1 id="PerceptronValidationFraction" name="PerceptronValidationFraction">
                                        <div><br></div>
                                        <label for="PerceptronNIterNoChange">Number Iterations No Change (int):</label>
                                        <input type="number" step="1" value=5 placeholder=5 id="PerceptronNIterNoChange" name="PerceptronNIterNoChange">
                                        <div><br></div>
                                        <label for="PerceptronClassWeight">Class Weight (enter a dictionary or 'balanced'):</label>
                                        <input type="text" id="PerceptronClassWeight" name="PerceptronClassWeight">
                                    </div>
                                </div>
                            </div>
                            
                            <!-- MLP Non-Essential Hyperparameters -->
                            <div id="advancedMLPFields" class="hidden">
                                <h3>MLP Model Settings</h3>
                                <p class="field-note" style="margin-bottom: 12px; margin-top: 0;">Multi-Layer Perceptron is a neural network. Configure hidden layers and learning parameters for optimal performance.</p>
                                <div class="nonreqHyperparams">
                                    <div><br></div>
                                    <div class="toggle-container">
                                        <span><h3>Edit Non-Essential Hyperparameters?</h3></span>
                                        <label class="switch">
                                            <input type="checkbox" id="advancedNonreqMLPSlider">
                                            <span class="slider"></span>
                                        </label>
                                    </div>
                                    <div id="advancedNonreqMLPParams" class="hidden">
                                        <label for="MLPNesterovsMomentum">Nesterovs Momentum:</label>
                                        <select name="MLPNesterovsMomentum" id="MLPNesterovsMomentum">
                                            <option value="true">True</option>
                                            <option value="false">False</option>
                                        </select>
                                        <label for="MLPEarlyStopping">Early Stopping:</label>
                                        <select name="MLPEarlyStopping" id="MLPEarlyStopping">
                                            <option value="false">False</option>
                                            <option value="true">True</option>
                                        </select>
                                        <label for="MLPShuffle">Shuffle:</label>
                                        <select name="MLPShuffle" id="MLPShuffle">
                                            <option value="true">True</option>
                                            <option value="false">False</option>
                                        </select>
                                        <div><br></div>
                                        <label for="MLPVerbose">Verbose:</label>
                                        <select name="MLPVerbose" id="MLPVerbose">
                                            <option value="false">False</option>
                                            <option value="true">True</option>
                                        </select>
                                        <label for="MLPWarmStart">Warm Start:</label>
                                        <select name="MLPWarmStart" id="MLPWarmStart">
                                            <option value="false">False</option>
                                            <option value="true">True</option>
                                        </select>
                                        <div><br></div>
                                        <label for="MLPAlpha">Alpha - Regularization (float >=0):</label>
                                        <input type="number" value=.0001 placeholder=.0001 step="any" min=0 id="MLPAlpha" name="alpha">
                                        <div><br></div>
                                        <label for="MLPLearning_rate">Learning Rate:</label>
                                        <select name="MLPLearning_rate" id="MLPLearning_rate">
                                            <option value="constant">constant</option>
                                            <option value="invscaling">invscaling</option>
                                            <option value="adaptive">adaptive</option>
                                        </select>
                                        <div><br></div>
                                        <label for="MLPBatchSize">Batch Size (integer >=1 or 'auto'):</label>
                                        <input type="text" value=200 placeholder="200" id="MLPBatchSize" name="MLPBatchSize">
                                        <div><br></div>
                                        <label for="MLPLearningRateInit">Learning Rate Init (float >0):</label>
                                        <input type="number" value=.001 placeholder=".001" min="0.0000001" step="any" id="MLPLearningRateInit" name="MLPLearningRateInit">
                                        <div><br></div>
                                        <label for="MLPPowerT">Power T (float):</label>
                                        <input type="number" value=.05 placeholder=.5 step="any" id="MLPPowerT" name="MLPPowerT">
                                        <div><br></div>
                                        <label for="MLPMaxIter">Max Iterations (integer >=1):</label>
                                        <input type="number" value=200 placeholder=200 step="1" min="1" id="MLPMaxIter" name="MLPMaxIter">
                                        <div><br></div>
                                        <label for="MLPTol">Tol (float):</label>
                                        <input type="number" value=.0001 placeholder=.0001 step="any" id="MLPTol" name="MLPTol">
                                        <div><br></div>
                                        <label for="MLPMomentum">Momentum (float [0, 1) ):</label>
                                        <input type="number" value=.09 placeholder=.9 min=0 max=.9999999 step="any" id="MLPMomentum" name="MLPMomentum">
                                        <div><br></div>
                                        <label for="MLPValidationFraction">Validation Fraction (float [0, 1) ):</label>
                                        <input type="number" value=.01 placeholder=.1 min=0 max=.9999999 step="any" id="MLPValidationFraction" name="MLPValidationFraction">
                                        <div><br></div>
                                        <label for="MLPBeta1"> Beta 1 (float [0, 1) ):</label>
                                        <input type="number" value=.09 placeholder=.9 min=0 max=.9999999  step="any" id="MLPBeta1" name="MLPBeta1">
                                        <div><br></div>
                                        <label for="MLPBeta2">Beta 2 (float [0, 1) ):</label>
                                        <input type="number" value=.999 placeholder=.999 min=0 max=.9999999 step="any" id="MLPBeta2" name="MLPBeta2">
                                        <div><br></div>
                                        <label for="MLPEpsilon">Epsilon (float >0):</label>
                                        <input type="number" value=.00000001 min=.00000000001 placeholder=.00000001 step="any" id="MLPEpsilon" name="MLPEpsilon">
                                    </div>
                                </div>
                            </div>
                            
                            <!-- K-Nearest Neighbors Hyperparameters -->
                            <div id="advancedK-NearestFields" class="hidden">
                                <h3>K-Nearest Neighbors Model Settings</h3>
                                <label for="advancedN_neighbors">N Neighbors - # of neighbors (int):</label>
                                <input type="number" step=1 value=5 placeholder=5 id="advancedN_neighbors" name="advancedN_neighbors">
                                <div class="nonreqHyperparams">
                                    <div><br></div>
                                    <div class="toggle-container">
                                        <span><h3>Edit Non-Essential Hyperparameters?</h3></span>
                                        <label class="switch">
                                            <input type="checkbox" id="advancedNonreqKNearestSlider">
                                            <span class="slider"></span>
                                        </label>
                                    </div>
                                    <div id="advancedNonreqKNearestParams" class="hidden">
                                        <label for="KNearestWeights">Weights:</label>
                                        <select name="KNearestWeights" id="KNearestWeights">
                                            <option value="uniform">uniform</option>
                                            <option value="distance">distance</option>
                                            <option value="callable">callable</option>
                                        </select>
                                        <label for="KNearestAlgorithm">Algorithm:</label>
                                        <select name="KNearestAlgorithm" id="KNearestAlgorithm">
                                            <option value="auto">auto</option>
                                            <option value="ball_tree">ball_tree</option>
                                            <option value="kd_tree">kd_tree</option>
                                            <option value="brute">brute</option>
                                        </select>
                                        <div><br></div>
                                        <label for="metric">Metric - (euclidean, manhattan, etc):</label>
                                        <input type="text" id="metric" value='minkowski' name="metric">
                                        <div><br></div>
                                        <label for="KNearestLeafSize">Leaf Size (int):</label>
                                        <input type="number" step="1" value=30 placeholder=30 id="KNearestLeafSize" name="KNearestLeafSize">
                                        <div><br></div>
                                        <label for="KNearestP">P (int):</label>
                                        <input type="number" step="1" value=2 placeholder=2 id="KNearestP" name="KNearestP">
                                        <div><br></div>
                                        <label for="KNearestMetricParams">Metric Params (enter a dictionary or leave blank for None):</label>
                                        <input type="text" id="KNearestMetricParams" name="KNearestMetricParams">
                                        <div><br></div>
                                        <label for="KNearestNJobs">N Jobs (an integer or leave blank for None):</label>
                                        <input type="number" step="1" id="KNearestNJobs" name="KNearestNJobs">
                                    </div>
                                </div>
                            </div>
                            
                            <!-- Gradient Boosting Hyperparameters -->
                            <div id="advancedGradientBoostingFields" class="hidden">
                                <h3>Gradient Boosting Model Settings</h3>
                                <p class="field-note" style="margin-bottom: 12px; margin-top: 0;">Gradient Boosting builds trees sequentially, each correcting previous errors. Often achieves high performance.</p>
                                <label for="advancedGBn_estimators">N Estimators - Trees (integer >=1):</label>
                                <input type="number" step=1 value=100 placeholder=100 min=1 id="advancedGBn_estimators" name="advancedGBn_estimators">
                                <div><br></div>
                                <label for="advancedGBlearn">Learning Rate (float >0):</label>
                                <input type="number" value=.1 placeholder=.1 min="0.0000001" step="any" id="advancedGBlearn" name="advancedGBlearn">
                                <div class="nonreqHyperparams">
                                    <div><br></div>
                                    <div class="toggle-container">
                                        <span><h3>Edit Non-Essential Hyperparameters?</h3></span>
                                        <label class="switch">
                                            <input type="checkbox" id="advancedNonreqGBSlider">
                                            <span class="slider"></span>
                                        </label>
                                    </div>
                                    <div id="advancedNonreqGBParams" class="hidden">
                                        <label for="GBLoss">Loss:</label>
                                        <select name="GBLoss" id="GBLoss">
                                            <option value="absolute_error">absolute_error</option>
                                            <option value="squared_error">squared_error</option>
                                            <option value="huber">huber</option>
                                            <option value="quantile">quantile</option>
                                        </select>
                                        <label for="GBWarmStart">Warm Start:</label>
                                        <select name="GBWarmStart" id="GBWarmStart">
                                            <option value="false">False</option>
                                            <option value="true">True</option>
                                        </select>
                                        <label for="GBCriterion">Criterion:</label>
                                        <select name="GBCriterion" id="GBCriterion">
                                            <option value="friedman_mse">Friedman MSE</option>
                                            <option value="squared_error">Squared Error</option>
                                        </select>
                                        <div><br></div>
                                        <label for="GBMax_depth">Max Depth - Tree depth (an integer or leave blank for None):</label>
                                        <input type="number" step=1 value=3 placeholder=3 id="GBMax_depth" name="GBMax_depth">
                                        <div><br></div>
                                        <label for="GBSubsample">Subsample (float (0,1]):</label>
                                        <input type="number" value=1 placeholder=1 min="0.0000001" max=1 step="any" id="GBSubsample" name="GBSubsample">
                                        <div><br></div>
                                        <label for="GBMinSamplesSplit">Min Samples Split (integer or float):</label>
                                        <input type="number" value=2 placeholder=2 step="any" id="GBMinSamplesSplit" name="GBMinSamplesSplit">
                                        <div><br></div>
                                        <label for="GBMinSamplesLeaf">Min Samples Leaf (integer or float):</label>
                                        <input type="number" value=1 placeholder=1 step="any" id="GBMinSamplesLeaf" name="GBMinSamplesLeaf">
                                        <div><br></div>
                                        <label for="GBMinWeightFractionLeaf">Min Weight Fraction Leaf (float [0.0, 0.5]):</label>
                                        <input type="number" value=0 placeholder=0 step="any" min=0 max=.5 id="GBMinWeightFractionLeaf" name="GBMinWeightFractionLeaf">
                                        <div><br></div>
                                        <label for="GBMinImpurityDecrease">Min Impurity Decrease (float):</label>
                                        <input type="number" value=0 placeholder=0 step="any" id="GBMinImpurityDecrease" name="GBMinImpurityDecrease">
                                        <div><br></div>
                                        <label for="GBInit">Init ('estimator' or leave blank for None):</label>
                                        <input type="text" placeholder='estimator' id="GBInit" name="GBInit">
                                        <div><br></div>
                                        <label for="GBMaxFeatrues">Max Features (int, float, or string):</label>
                                        <input type="text" id="GBMaxFeatrues" name="GBMaxFeatrues">
                                        <div><br></div>
                                        <label for="GBAlpha">Alpha (float [0.0, 1.0]):</label>
                                        <input type="number" step="any" min=0 max=1 value=.9 placeholder=.9 id="GBAlpha" name="GBAlpha">
                                        <div><br></div>
                                        <label for="GBVerbose">Verbose (int):</label>
                                        <input type="number" step="1" value=0 placeholder=0 id="GBVerbose" name="GBVerbose">
                                        <div><br></div>
                                        <label for="GBMaxLeafNodes">Max Leaf Nodes (an integer or leave blank for None):</label>
                                        <input type="number" step="1" id="GBMaxLeafNodes" name="GBMaxLeafNodes">
                                    </div>
                                </div>
                            </div>
                            
                            <!-- Classifier Non-Essential Hyperparameters -->
                            <div id="advancedLogistic_classifierFields" class="hidden">
                                <h3>Logistic Classifier Model Settings</h3>
                                <p class="field-note" style="margin-bottom: 12px; margin-top: 0;">Logistic regression for classification. Uses a sigmoid function to predict class probabilities.</p>
                                <div class="nonreqHyperparams">
                                    <div class="toggle-container">
                                        <span><h3>Edit Non-Essential Hyperparameters?</h3></span>
                                        <label class="switch">
                                            <input type="checkbox" id="advancedNonreqLogisticClassifierSlider">
                                            <span class="slider"></span>
                                        </label>
                                    </div>
                                    <div id="advancedNonreqLogisticClassifierParams" class="hidden">
                                        <label for="Class_LogisticDual">Dual:</label>
                                        <select name="Class_LogisticDual" id="Class_LogisticDual">
                                            <option value="false">False</option>
                                            <option value="true">True</option>
                                        </select>
                                        <label for="Class_LogisticFitIntercept">Fit Intercept:</label>
                                        <select name="Class_LogisticFitIntercept" id="Class_LogisticFitIntercept">
                                            <option value="true">True</option>
                                            <option value="false">False</option>
                                        </select>
                                        <label for="Class_LogisticSolver">Solver:</label>
                                        <select name="Class_LogisticSolver" id="Class_LogisticSolver">
                                            <option value="lbfgs">lbfgs</option>
                                            <option value="newton-cg">newton-cg</option>
                                            <option value="liblinear">liblinear</option>
                                            <option value="sag">sag</option>
                                            <option value="saga">saga</option>
                                        </select>
                                        <div><br></div>
                                        <label for="Class_LogisticMultiClass">Multi Class:</label>
                                        <select name="Class_LogisticMultiClass" id="Class_LogisticMultiClass">
                                            <option value="auto">auto</option>
                                            <option value="ovr">ovr</option>
                                            <option value="multinomial">multinomial</option>
                                        </select>
                                        <label for="Class_LogisticWarmStart">Warm Start:</label>
                                        <select name="Class_LogisticWarmStart" id="Class_LogisticWarmStart">
                                            <option value="false">False</option>
                                            <option value="true">True</option>
                                        </select>
                                        <div><br></div>
                                        <label for="Class_CLogistic">C - Regularization (float >0):</label>
                                        <input type="number" value=1 placeholder=1.0 min="0.0000001" step="any" id="Class_CLogistic" name="Class_CLogistic">
                                        <div><br></div>
                                        <label for="Class_Logistic_penalty">Penalty:</label>
                                        <select name="Class_Logistic_penalty" id="Class_Logistic_penalty">
                                            <option value="l2">l2</option>
                                            <option value="l1">l1</option>
                                            <option value="elasticnet">elasticnet</option>
                                            <option value="None">None</option>
                                        </select>
                                        <div><br></div>
                                        <label for="Class_LogisticTol">Tol (float >0):</label>
                                        <input type="number" value=0.0001 placeholder=.0001 min="0.0000001" step="any" id="Class_LogisticTol" name="Class_LogisticTol">
                                        <div><br></div>
                                        <label for="Class_Logisticintercept_scaling">Intercept Scaling:</label>
                                        <input type="number" value=1 placeholder=1 step="any" id="Class_Logisticintercept_scaling" name="Class_Logisticintercept_scaling">
                                        <div><br></div>
                                        <label for="Class_LogisticClassWeight">Class Weight (dict or 'balanced'):</label>
                                        <input type="text" id="Class_LogisticClassWeight" name="Class_LogisticClassWeight">
                                        <div><br></div>
                                        <label for="Class_LogisticMaxIterations">Max Iterations (integer >=1):</label>
                                        <input type="number" step=1 min="1" value=100 placeholder="100" id="Class_LogisticMaxIterations" name="Class_LogisticMaxIterations">
                                        <div><br></div>
                                        <label for="Class_LogisticVerbose">Verbose (int):</label>
                                        <input type="number" step=1 value=0 placeholder="0" id="Class_LogisticVerbose" name="Class_LogisticVerbose">
                                        <div><br></div>
                                        <label for="Class_LogisticNJobs">N Jobs (int):</label>
                                        <input type="number" step=1 id="Class_LogisticNJobs" name="Class_LogisticNJobs">
                                        <div><br></div>
                                        <label for="Class_Logisticl1Ratio">L1 Ratio (float [0, 1]):</label>
                                        <input type="number" min="0.0000001" step="any" max=1 id="Class_Logisticl1Ratio" name="Class_Logisticl1Ratio">
                                    </div>
                                </div>
                            </div>
                            
                            <div id="advancedMLP_classifierFields" class="hidden">
                                <h3>MLP Classifier Model Settings</h3>
                                <p class="field-note" style="margin-bottom: 12px; margin-top: 0;">Neural network classifier. Configure hidden layers and learning parameters for classification tasks.</p>
                                <div class="nonreqHyperparams">
                                    <div class="toggle-container">
                                        <span><h3>Edit Non-Essential Hyperparameters?</h3></span>
                                        <label class="switch">
                                            <input type="checkbox" id="advancedNonreqMLPClassifierSlider">
                                            <span class="slider"></span>
                                        </label>
                                    </div>
                                    <div id="advancedNonreqMLPClassifierParams" class="hidden">
                                        <label for="Class_MLPNesterovsMomentum">Nesterovs Momentum:</label>
                                        <select name="Class_MLPNesterovsMomentum" id="Class_MLPNesterovsMomentum">
                                            <option value="true">True</option>
                                            <option value="false">False</option>
                                        </select>
                                        <label for="Class_MLPEarlyStopping">Early Stopping:</label>
                                        <select name="Class_MLPEarlyStopping" id="Class_MLPEarlyStopping">
                                            <option value="false">False</option>
                                            <option value="true">True</option>
                                        </select>
                                        <label for="Class_MLPShuffle">Shuffle:</label>
                                        <select name="Class_MLPShuffle" id="Class_MLPShuffle">
                                            <option value="true">True</option>
                                            <option value="false">False</option>
                                        </select>
                                        <div><br></div>
                                        <label for="Class_MLPVerbose">Verbose:</label>
                                        <select name="Class_MLPVerbose" id="Class_MLPVerbose">
                                            <option value="false">False</option>
                                            <option value="true">True</option>
                                        </select>
                                        <label for="Class_MLPWarmStart">Warm Start:</label>
                                        <select name="Class_MLPWarmStart" id="Class_MLPWarmStart">
                                            <option value="false">False</option>
                                            <option value="true">True</option>
                                        </select>
                                        <div><br></div>
                                        <label for="Class_MLPAlpha">Alpha - Regularization (float >=0):</label>
                                        <input type="number" value=.0001 placeholder=.0001 step="any" min=0 id="Class_MLPAlpha" name="Class_MLPAlpha">
                                        <div><br></div>
                                        <label for="Class_MLPLearning_rate">Learning Rate:</label>
                                        <select name="Class_MLPLearning_rate" id="Class_MLPLearning_rate">
                                            <option value="constant">constant</option>
                                            <option value="invscaling">invscaling</option>
                                            <option value="adaptive">adaptive</option>
                                        </select>
                                        <div><br></div>
                                        <label for="Class_MLPBatchSize">Batch Size (integer >=1 or 'auto'):</label>
                                        <input type="text" value=200 placeholder="200" id="Class_MLPBatchSize" name="Class_MLPBatchSize">
                                        <div><br></div>
                                        <label for="Class_MLPLearningRateInit">Learning Rate Init (float >0):</label>
                                        <input type="number" value=.001 placeholder=".001" min="0.0000001" step="any" id="Class_MLPLearningRateInit" name="Class_MLPLearningRateInit">
                                        <div><br></div>
                                        <label for="Class_MLPPowerT">Power T (float):</label>
                                        <input type="number" value=.05 placeholder=.5 step="any" id="Class_MLPPowerT" name="Class_MLPPowerT">
                                        <div><br></div>
                                        <label for="Class_MLPMaxIter">Max Iterations (integer >=1):</label>
                                        <input type="number" value=200 placeholder=200 step="1" min="1" id="Class_MLPMaxIter" name="Class_MLPMaxIter">
                                        <div><br></div>
                                        <label for="Class_MLPTol">Tol (float):</label>
                                        <input type="number" value=.0001 placeholder=.0001 step="any" id="Class_MLPTol" name="Class_MLPTol">
                                        <div><br></div>
                                        <label for="Class_MLPMomentum">Momentum (float [0, 1) ):</label>
                                        <input type="number" value=.09 placeholder=.9 min=0 max=.9999999 step="any" id="Class_MLPMomentum" name="Class_MLPMomentum">
                                        <div><br></div>
                                        <label for="Class_MLPValidationFraction">Validation Fraction (float [0, 1) ):</label>
                                        <input type="number" value=.01 placeholder=.1 min=0 max=.9999999 step="any" id="Class_MLPValidationFraction" name="Class_MLPValidationFraction">
                                        <div><br></div>
                                        <label for="Class_MLPBeta1"> Beta 1 (float [0, 1) ):</label>
                                        <input type="number" value=.09 placeholder=.9 min=0 max=.9999999  step="any" id="Class_MLPBeta1" name="Class_MLPBeta1">
                                        <div><br></div>
                                        <label for="Class_MLPBeta2">Beta 2 (float [0, 1) ):</label>
                                        <input type="number" value=.999 placeholder=.999 min=0 max=.9999999 step="any" id="Class_MLPBeta2" name="Class_MLPBeta2">
                                        <div><br></div>
                                        <label for="Class_MLPEpsilon">Epsilon (float >0):</label>
                                        <input type="number" value=.00000001 min=.00000000001 placeholder=.00000001 step="any" id="Class_MLPEpsilon" name="Class_MLPEpsilon">
                                    </div>
                                </div>
                            </div>
                            
                            <div id="advancedRF_classifierFields" class="hidden">
                                <h3>Random Forest Classifier Model Settings</h3>
                                <p class="field-note" style="margin-bottom: 12px; margin-top: 0;">Ensemble classifier using multiple decision trees. More trees generally improve accuracy but increase computation time.</p>
                                <label for="advancedClass_RFn_estmators">N Estimators - # of trees (integer >=1):</label>
                                <input type="number" step=1 value=100 placeholder="100" min=1 id="advancedClass_RFn_estmators" name="advancedClass_RFn_estmators">
                                <div class="nonreqHyperparams">
                                    <div class="toggle-container">
                                        <span><h3>Edit Non-Essential Hyperparameters?</h3></span>
                                        <label class="switch">
                                            <input type="checkbox" id="advancedNonreqRFClassifierSlider">
                                            <span class="slider"></span>
                                        </label>
                                    </div>
                                    <div id="advancedNonreqRFClassifierParams" class="hidden">
                                        <label for="Class_RFBoostrap">Bootstrap:</label>
                                        <select name="Class_RFBoostrap" id="Class_RFBoostrap">
                                            <option value="true">True</option>
                                            <option value="false">False</option>
                                        </select>
                                        <label for="Class_RFoobScore">oob Score:</label>
                                        <select name="Class_RFoobScore" id="Class_RFoobScore">
                                            <option value="false">False</option>
                                            <option value="true">True</option>
                                        </select>
                                        <label for="Class_RFWarmStart">Warm Start:</label>
                                        <select name="Class_RFWarmStart" id="Class_RFWarmStart">
                                            <option value="false">False</option>
                                            <option value="true">True</option>
                                        </select>
                                        <div><br></div>
                                        <label for="Class_RFmin_weight_fraction_leaf">Min Weight Fraction Leaf (float [0.0, 0.5]):</label>
                                        <input type="number" value=0 placeholder=0 min="0" max=".5" step="any" id="Class_RFmin_weight_fraction_leaf" name="Class_RFmin_weight_fraction_leaf">
                                        <div><br></div>
                                        <label for="Class_RFMaxLeafNodes">Max Leaf Nodes (an integer or leave blank for None):</label>
                                        <input type="number" step="1" id="Class_RFMaxLeafNodes" name="Class_RFMaxLeafNodes">
                                        <div><br></div>
                                        <label for="Class_RFMinImpurityDecrease">Min Impurity Decrease (float):</label>
                                        <input type="number" value=0 placeholder=0 step="any" id="Class_RFMinImpurityDecrease" name="Class_RFMinImpurityDecrease">
                                        <div><br></div>
                                        <label for="Class_RFNJobs">N Jobs (an integer or leave blank for None):</label>
                                        <input type="number" step="1" id="Class_RFNJobs" name="Class_RFNJobs">
                                        <div><br></div>
                                        <label for="Class_RFVerbose">Verbose (int):</label>
                                        <input type="number" value=0 placeholder=0 step="1" id="Class_RFVerbose" name="Class_RFVerbose">
                                        <div><br></div>
                                        <label for="Class_RFMax_depth">Max Depth - Tree depth (an integer or leave blank for None):</label>
                                        <input type="number" step=1 id="Class_RFMax_depth" name="Class_RFMax_depth">
                                        <div><br></div>
                                        <label for="Class_min_samples_split">Min Samples Split - Min samples per split (integer or float):</label>
                                        <input type="number" value=2 placeholder="2" id="Class_min_samples_split" name="Class_min_samples_split">
                                        <div><br></div>
                                        <label for="Class_min_samples_leaf">Min Samples Leaf - Min samples per leaf (integer or float):</label>
                                        <input type="number" value=1 placeholder=1 id="Class_min_samples_leaf" name="Class_min_samples_leaf">
                                    </div>
                                </div>
                            </div>
                            
                            <div id="advancedSVC_classifierFields" class="hidden">
                                <h3>SVC Classifier Model Settings</h3>
                                <p class="field-note" style="margin-bottom: 12px; margin-top: 0;">Support Vector Classifier finds optimal decision boundaries between classes. Effective for complex classification problems.</p>
                                <div class="nonreqHyperparams">
                                    <div><br></div>
                                    <div class="toggle-container">
                                        <span><h3>Edit Non-Essential Hyperparameters?</h3></span>
                                        <label class="switch">
                                            <input type="checkbox" id="advancedNonreqSVCClassifierSlider">
                                            <span class="slider"></span>
                                        </label>
                                    </div>
                                    <div id="advancedNonreqSVCClassifierParams" class="hidden">
                                        <label for="SVCshrinking">Shrinking:</label>
                                        <select name="SVCshrinking" id="SVCshrinking">
                                            <option value="true">True</option>
                                            <option value="false">False</option>
                                        </select>
                                        <label for="SVCprobability">Probability:</label>
                                        <select name="SVCprobability" id="SVCprobability">
                                            <option value="false">False</option>
                                            <option value="true">True</option>
                                        </select>
                                        <br>
                                        <label for="SVCBreakTies">Break Ties:</label>
                                        <select name="SVCBreakTies" id="SVCBreakTies">
                                            <option value="false">False</option>
                                            <option value="true">True</option>
                                        </select>
                                        <label for="SVCverbose">Verbose:</label>
                                        <select name="SVCverbose" id="SVCverbose">
                                            <option value="false">False</option>
                                            <option value="true">True</option>
                                        </select>
                                        <div><br></div>
                                        <label for="SVCcoef0">coef0 (float):</label>
                                        <input type="number" step="any" value=0 placeholder="0" id="SVCcoef0" name="SVCcoef0">
                                        <div><br></div>
                                        <label for="SVCtol">tol(float):</label>
                                        <input type="number" step="any" value=.001 placeholder=.001 id="SVCtol" name="SVCtol">
                                        <div><br></div>
                                        <label for="SVCCacheSize">Cache Size (float):</label>
                                        <input type="number" step="any" value=200 placeholder=200 id="SVCCacheSize" name="SVCCacheSize">
                                        <div><br></div>
                                        <label for="SVCClassWeight">Class Weight (enter a dictionary or 'balanced')</label>
                                        <input type="text" id="SVCClassWeight" name="SVCClassWeight">
                                        <div><br></div>
                                        <label for="SVCmaxIter">Max Iterations (int):</label>
                                        <input type="number" step=1 value=-1 placeholder=-1 id="SVCmaxIter" name="SVCmaxIter">
                                        <div><br></div>
                                        <label for="SVCdecisionFunctionShape">Decision Function Shape:</label>
                                        <select name="SVCdecisionFunctionShape" id="SVCdecisionFunctionShape">
                                            <option value="ovr">ovr</option>
                                            <option value="ovo">ovo</option>
                                        </select>
                                    </div>
                                </div>
                            </div>
                            
                            <!-- Cluster Non-Essential Hyperparameters -->
                            <div id="advancedAgglomerativeFields" class="hidden">
                                <div class="nonreqHyperparams">
                                    <div><br></div>
                                    <div class="toggle-container">
                                        <span><h3>Edit Non-Essential Hyperparameters?</h3></span>
                                        <label class="switch">
                                            <input type="checkbox" id="advancedNonreqAgglomerativeSlider">
                                            <span class="slider"></span>
                                        </label>
                                    </div>
                                    <div id="advancedNonreqAgglomerativeParams" class="hidden">
                                        <label for="Agglinkage">Linkage:</label>
                                        <select name="Agglinkage" id="Agglinkage">
                                            <option value="ward">ward</option>
                                            <option value="complete">complete</option>
                                            <option value="average">average</option>
                                            <option value="single">single</option>
                                        </select>
                                        <label for="aggcompute_distances">Compute Distances:</label>
                                        <select name="aggcompute_distances" id="aggcompute_distances">
                                            <option value="false">False</option>
                                            <option value="true">True</option>
                                        </select>
                                        <div><br></div>
                                        <label for="aggcompute_full_tree">Compute Full Tree:</label>
                                        <select name="aggcompute_full_tree" id="aggcompute_full_tree">
                                            <option value="auto">Auto</option>
                                            <option value="false">False</option>
                                            <option value="true">True</option>
                                        </select>
                                        <div><br></div>
                                        <label for="Aggdistance_threshold">Distance Threshold (float or None):</label>
                                        <input type="number" step=any id="Aggdistance_threshold" name="Aggdistance_threshold">
                                        <div><br></div>
                                        <label for="Aggmetric">Metric:</label>
                                        <input type="text" id="Aggmetric" value="euclidean" placeholder="euclidean" name="Aggmetric">
                                        <div><br></div>
                                        <label for="Aggmemory">Memory:</label>
                                        <input type="text" id="Aggmemory" name="Aggmemory">
                                        <div><br></div>
                                        <label for="Aggconnectivity">Connectivity</label>
                                        <input type="text" id="Aggconnectivity" name="Aggconnectivity">
                                    </div>
                                </div>
                            </div>
                            
                            <div id="advancedGaussianFields" class="hidden">
                                <div class="nonreqHyperparams">
                                    <div><br></div>
                                    <div class="toggle-container">
                                        <span><h3>Edit Non-Essential Hyperparameters?</h3></span>
                                        <label class="switch">
                                            <input type="checkbox" id="advancedNonreqGaussianSlider">
                                            <span class="slider"></span>
                                        </label>
                                    </div>
                                    <div id="advancedNonreqGaussianParams" class="hidden">
                                        <label for="GauWarmStart">Warm Start:</label>
                                        <select name="GauWarmStart" id="GauWarmStart">
                                            <option value="false">False</option>
                                            <option value="true">True</option>
                                        </select>
                                        <label for="Gauinit_params">Init Params:</label>
                                        <select name="Gauinit_params" id="Gauinit_params">
                                            <option value="kmeans">kmeans</option>
                                            <option value="k-means++">k-means++</option>
                                            <option value="random">random</option>
                                            <option value="random_from_data">random_from_data</option>
                                        </select>
                                        <div><br></div>
                                        <label for="Gaucovariance_type">Covariance Type:</label>
                                        <select name="Gaucovariance_type" id="Gaucovariance_type">
                                            <option value="full">full</option>
                                            <option value="tied">tied</option>
                                            <option value="diag">diag</option>
                                            <option value="spherical">spherical</option>
                                        </select>
                                        <div><br></div>
                                        <label for="GauMax_iter">Max Iterations:</label>
                                        <input type="number" step=1 value=100 placeholder=100 min=1 id="GauMax_iter" name="GauMax_iter">
                                        <div><br></div>
                                        <label for="Gaun_init">N Init:</label>
                                        <input type="number" step=1 value=1 placeholder=1 min=1 id="Gaun_init" name="Gaun_init">
                                        <div><br></div>
                                        <label for="GauTol">Tol (float >0):</label>
                                        <input type="number" value=0.001 placeholder=.001 min="0.0000001" step="any" id="GauTol" name="GauTol">
                                        <div><br></div>
                                        <label for="Gaureg_covar">reg_covar:</label>
                                        <input type="number" value=0.000001 placeholder=.000001 min="0.0000001" step="any" id="Gaureg_covar" name="Gaureg_covar">
                                        <div><br></div>
                                        <label for="Gauweights_init">Weights Init:</label>
                                        <input type="text" id="Gauweights_init" name="Gauweights_init">
                                        <div><br></div>
                                        <label for="Gaumeans_init">Means Init:</label>
                                        <input type="text" id="Gaumeans_init" name="Gaumeans_init">
                                        <div><br></div>
                                        <label for="Gauprecisions_init">Precisions Init</label>
                                        <input type="text" id="Gauprecisions_init" name="Gauprecisions_init">
                                        <div><br></div>
                                        <label for="GauVerbose">Verbose:</label>
                                        <input type="number" step=1 value=0 placeholder=0 min=0 id="GauVerbose" name="GauVerbose">
                                        <div><br></div>
                                        <label for="GauVerbose_interval">Verbose Interval:</label>
                                        <input type="number" step=1 value=10 placeholder=10 min=0 id="GauVerbose_interval" name="GauVerbose_interval">
                                    </div>
                                </div>
                            </div>
                            
                            <div id="advancedKmeansFields" class="hidden">
                                <div class="nonreqHyperparams">
                                    <div><br></div>
                                    <div class="toggle-container">
                                        <span><h3>Edit Non-Essential Hyperparameters?</h3></span>
                                        <label class="switch">
                                            <input type="checkbox" id="advancedNonreqKmeansSlider">
                                            <span class="slider"></span>
                                        </label>
                                    </div>
                                    <div id="advancedNonreqKmeansParams" class="hidden">
                                        <label for="Kmeansalgorithm">Algorithm:</label>
                                        <select name="Kmeansalgorithm" id="Kmeansalgorithm">
                                            <option value="lloyd">lloyd</option>
                                            <option value="elkan">elkan</option>
                                            <option value="auto">auto</option>
                                            <option value="full">full</option>
                                        </select>
                                        <label for="kmeansCopyX">Copy X:</label>
                                        <select name="kmeansCopyX" id="kmeansCopyX">
                                            <option value="true">True</option>
                                            <option value="false">False</option>
                                        </select>
                                        <div><br></div>
                                        <label for="kmeansmax_iter">Max Iterations:</label>
                                        <input type="number" value=300 placeholder=300 step=1 id="kmeansmax_iter" name="kmeansmax_iter">
                                        <div><br></div>
                                        <label for="kmeansverbose">Verbose:</label>
                                        <input type="number" value=0 placeholder=0 step=1 id="kmeansverbose" name="kmeansverbose">
                                        <div><br></div>
                                        <label for="kmeanstol">tol:</label>
                                        <input type="number" value=0.0001 placeholder=0.0001 step=any id="kmeanstol" name="kmeanstol">
                                        <div><br></div>
                                        <label for="kmeansInit">Init:</label>
                                        <input type="text" value=k-means++ placeholder="k-means++" id="kmeansInit" name="kmeansInit">
                                        <div><br></div>
                                        <label for="kmeansn_init">n_init (int or 'auto'):</label>
                                        <input type="text" value=auto placeholder="auto" id="kmeansn_init" name="kmeansn_init">
                                    </div>
                                </div>
                            </div>
                            </div>
                            
                            <!-- Advanced Options Section -->
                            <div style="margin-top: 30px; padding-top: 20px; border-top: 1px solid #e5e5e5;">
                                <h3 style="margin-top: 0; margin-bottom: 16px; font-size: 1.2em; font-weight: 600;">Advanced Options</h3>
                                
                                <h4 style="margin-top: 24px; margin-bottom: 12px; padding-top: 16px; border-top: 1px solid #e5e5e5; font-size: 1.1em; font-weight: 600; color: #000000;">Feature Selection</h4>
                                <p class="field-note" style="margin-bottom: 12px; margin-top: 0;">Automatically select the most important features to improve model performance and reduce overfitting.</p>
                                <div class="scaling-container">
                                    <label for="featureSelectionMethod">Feature Selection Method</label>
                                    <select name="featureSelectionMethod" id="featureSelectionMethod">
                                        <option value="none">None</option>
                                        <option value="SelectKBest">Select K Best (Filter)</option>
                                        <option value="RFE">Recursive Feature Elimination (Wrapper)</option>
                                        <option value="SelectFromModel">Select From Model (Model-based)</option>
                                    </select>
                                </div>
                                <div class="scaling-container hidden" id="featureSelectionParams">
                                    <label for="featureSelectionK">Number of Features (K)</label>
                                    <input type="number" min="1" id="featureSelectionK" name="featureSelectionK" placeholder="10">
                                </div>

                                <h4 style="margin-top: 24px; margin-bottom: 12px; padding-top: 16px; border-top: 1px solid #e5e5e5; font-size: 1.1em; font-weight: 600; color: #000000;">Outlier Handling</h4>
                                <p class="field-note" style="margin-bottom: 12px; margin-top: 0;">Detect and handle outliers that may negatively impact model performance.</p>
                                <div class="scaling-container">
                                    <label for="outlierMethod">Outlier Detection Method</label>
                                    <select name="outlierMethod" id="outlierMethod">
                                        <option value="none">None</option>
                                        <option value="IQR">Interquartile Range (IQR)</option>
                                        <option value="IsolationForest">Isolation Forest</option>
                                        <option value="LocalOutlierFactor">Local Outlier Factor</option>
                                        <option value="ZScore">Z-Score (3σ rule)</option>
                                    </select>
                                </div>
                                <div class="scaling-container hidden" id="outlierActionDiv">
                                    <label for="outlierAction">Action</label>
                                    <select name="outlierAction" id="outlierAction">
                                        <option value="remove">Remove outliers</option>
                                        <option value="cap">Cap at threshold</option>
                                    </select>
                                </div>

                                <h4 style="margin-top: 24px; margin-bottom: 12px; padding-top: 16px; border-top: 1px solid #e5e5e5; font-size: 1.1em; font-weight: 600; color: #000000;">Hyperparameter Search</h4>
                                <p class="field-note" style="margin-bottom: 12px; margin-top: 0;">Automatically find optimal hyperparameter values. This can significantly increase processing time.</p>
                                <div class="scaling-container">
                                    <label for="hyperparameterSearch">Enable Hyperparameter Search</label>
                                    <select name="hyperparameterSearch" id="hyperparameterSearch">
                                        <option value="none">None (use manual settings)</option>
                                        <option value="grid">Grid Search</option>
                                        <option value="randomized">Randomized Search</option>
                                        <option value="bayesian">Bayesian Optimization</option>
                                    </select>
                                </div>
                                <div class="scaling-container hidden" id="hyperparameterSearchParams">
                                    <label for="searchCVFolds">CV Folds for Search</label>
                                    <input type="number" min="2" max="10" value="5" id="searchCVFolds" name="searchCVFolds">
                                    <label for="searchNIter">Number of Iterations (Randomized/Bayesian)</label>
                                    <input type="number" min="10" max="100" value="50" id="searchNIter" name="searchNIter">
                                    <p class="field-note">Parameter grids are automatically generated based on selected model</p>
                                </div>

                                <h4 style="margin-top: 24px; margin-bottom: 12px; padding-top: 16px; border-top: 1px solid #e5e5e5; font-size: 1.1em; font-weight: 600; color: #000000;">Cross-Validation</h4>
                                <p class="field-note" style="margin-bottom: 12px; margin-top: 0;">Evaluate model performance across multiple data splits for more robust performance estimates.</p>
                                <div class="scaling-container">
                                    <label for="advancedCrossValidationType">Cross-Validation Type</label>
                                    <select name="advancedCrossValidationType" id="advancedCrossValidationType">
                                        <option value="None">None</option>
                                        <option value="KFold">K-Fold</option>
                                        <option value="StratifiedKFold">Stratified K-Fold</option>
                                        <option value="RepeatedKFold">Repeated K-Fold</option>
                                        <option value="RepeatedStratifiedKFold">Repeated Stratified K-Fold</option>
                                        <option value="ShuffleSplit">Shuffle Split</option>
                                        <option value="StratifiedShuffleSplit">Stratified Shuffle Split</option>
                                    </select>
                                </div>
                                <div class="scaling-container">
                                    <label for="advancedCrossValidationFolds">Number of Folds (2-100)</label>
                                    <input type="number" value="5" min="2" max="100" id="advancedCrossValidationFolds" name="advancedCrossValidationFolds" placeholder="2-100">
                                </div>
                            </div>
                            
                            <!-- Units and Significant Figures Section -->
                            <div style="margin-top: 30px; padding-top: 20px; border-top: 1px solid #e5e5e5;">
                                <h3 style="margin-top: 0; margin-bottom: 16px; font-size: 1.2em; font-weight: 600;">Output Formatting</h3>
                                
                                <div class="toggle-container" style="margin-bottom: 16px;">
                                    <h4 style="margin: 0; font-size: 1.1em; font-weight: 600;">Does Output Have Units?</h4>
                                    <label class="switch">
                                        <input type="checkbox" id="advancedUnitToggle">
                                        <span class="slider"></span>
                                    </label>
                                    <div id="advancedUnits" class="hidden" style="margin-top: 12px;">
                                        <label for="advancedUnitName">Enter Unit (u* for µ):</label>
                                        <input type="text" id="advancedUnitName" name="advancedUnitName" style="max-width: 200px;">
                                    </div>
                                </div>
                                
                                <div class="scaling-container">
                                    <label for="advancedSigfig">Select Number of Significant Figures</label>
                                    <input type="number" id="advancedSigfig" name="advancedSigfig" value="3" placeholder="3" min="1" max="10" style="max-width: 100px;">
                                </div>
                            </div>
                            
                            <!-- Run Button Section -->
                            <div style="margin-top: 30px; padding-top: 20px; border-top: 1px solid #e5e5e5;">
                                <div style="display: flex; gap: 12px; align-items: center; margin-bottom: 12px;">
                                    <button type="submit" class='processButton processButton--compact' id="advancedOptimizationSubmitButton">Run Model with Advanced Options</button>
                                    <button type="button" class='secondary-button' id="stopAdvancedButton" style="display: none; padding: 12px 24px;">Stop Model</button>
                                </div>
                                <p class="field-note" style="color: #856404; margin-top: 12px; margin-bottom: 0;">
                                    <strong>Warning:</strong> Advanced options such as hyperparameter search, feature selection, and extensive cross-validation can take multiple hours to complete, especially with large datasets.
                                </p>
                            </div>
                        </div>
                    </div>
                </form>
                
                <!-- Results section - right side -->
                <div class="visualSection" id="advancedModelingResults">
                    <div class="model-card">
                        <h2 style="margin-top: 0; margin-bottom: 20px;">Modeling Results</h2>
                        <div id="AdvancedNumericResultDiv" role="region" aria-label="Advanced regression model results" class="hidden">
                            <div id="advancedImageSelector"></div>
                        </div>
                        <div id="AdvancedClusterResultDiv" role="region" aria-label="Advanced clustering model results" class="hidden"></div>
                        <div id="AdvancedClassifierResultDiv" role="region" aria-label="Advanced classification model results" class="hidden"></div>
                        <div id="advancedResultsPlaceholder" style="padding: 40px; text-align: center; color: #666; font-style: italic;">
                            <p>Results will appear here after you run a model.</p>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Loading div outside model-layout to preserve grid structure -->
            <div id="advancedLoading" class="hidden" style="margin-top: 20px; max-width: 35%;" role="status" aria-live="polite" aria-atomic="true" aria-label="Advanced modeling loading status"></div>
        </div>
        
        <!-- AutoML Section -->
        <div id="automlSection" class="modeling-mode-section hidden">
            <div class="model-layout">
                <form id="automlForm">
                    <div class="modelSection">
                        <!-- Descriptive text about column selection -->
                        <div style="margin-bottom: 16px; padding: 12px; background-color: #f8f9fa; border-radius: 6px; border-left: 3px solid #357a53;">
                            <span id="automlModelingSelectionNote" class="modeling-selection-note" style="font-style: italic; color: #2c3e50;"></span>
                        </div>
                        
                        <!-- Model Selection -->
                        <div class="preprocess-card" style="margin-bottom: 20px;">
                            <h3 style="margin-top: 0; margin-bottom: 12px;">Select Model (Optional)</h3>
                            <p class="field-note" style="margin-bottom: 16px;">Choose a model algorithm, or let AutoML choose automatically.</p>
                            
                            <div id="automlNumericModels" class="hidden">
                                <label for="automlNModels" style="display: block; margin-bottom: 8px; font-weight: 600;">Regression Model (Optional - AutoML will try multiple if not specified):</label>
                                <select name="automlNModels" id="automlNModels" style="width: 100%; max-width: 400px; padding: 8px; border: 1px solid #ccc; border-radius: 4px;">
                                    <option value="" selected>Let AutoML choose</option>
                                    <optgroup label="Most Common">
                                        <option value="Linear">Linear</option>
                                        <option value="Ridge">Ridge</option>
                                        <option value="Lasso">Lasso</option>
                                        <option value="ElasticNet">Elastic Net</option>
                                        <option value="RF">Random Forest</option>
                                        <option value="gradient_boosting">Gradient Boosting</option>
                                        <option value="SVM">Support Vector Machine (SVR)</option>
                                        <option value="MLP">Multi-Layer Perceptron</option>
                                        <option value="K-Nearest">K-Nearest Neighbors</option>
                                        <option value="ExtraTrees">Extra Trees</option>
                                    </optgroup>
                                    <optgroup label="Additional Models (Alphabetical)">
                                        <option value="AdaBoost">AdaBoost Regressor</option>
                                        <option value="ARDRegression">ARD Regression</option>
                                        <option value="Bagging">Bagging Regressor</option>
                                        <option value="BayesianRidge">Bayesian Ridge</option>
                                        <option value="DecisionTree">Decision Tree Regressor</option>
                                        <option value="ElasticNetCV">Elastic Net CV</option>
                                        <option value="HistGradientBoosting">Histogram Gradient Boosting</option>
                                        <option value="Huber">Huber Regressor</option>
                                        <option value="LARS">LARS</option>
                                        <option value="LARSCV">LARS CV</option>
                                        <option value="LassoCV">Lasso CV</option>
                                        <option value="LinearSVR">Linear SVR</option>
                                        <option value="NuSVR">Nu-SVR</option>
                                        <option value="OMP">Orthogonal Matching Pursuit</option>
                                        <option value="PassiveAggressive">Passive Aggressive Regressor</option>
                                        <option value="Quantile">Quantile Regressor</option>
                                        <option value="RadiusNeighbors">Radius Neighbors Regressor</option>
                                        <option value="RANSAC">RANSAC Regressor</option>
                                        <option value="RidgeCV">Ridge CV</option>
                                        <option value="SGD">SGD Regressor</option>
                                        <option value="TheilSen">Theil-Sen Regressor</option>
                                    </optgroup>
                                </select>
                            </div>
                            
                            <div id="automlClusterModels" class="hidden">
                                <label for="automlClModels" style="display: block; margin-bottom: 8px; font-weight: 600;">Clustering Model (Optional):</label>
                                <select name="automlClModels" id="automlClModels" style="width: 100%; max-width: 400px; padding: 8px; border: 1px solid #ccc; border-radius: 4px;">
                                    <option value="" selected>Let AutoML choose</option>
                                    <optgroup label="Most Common">
                                        <option value="kmeans">K-Means</option>
                                        <option value="dbscan">DBSCAN</option>
                                        <option value="agglo">Agglomerative</option>
                                        <option value="gmm">Gaussian Mixture</option>
                                        <option value="spectral">Spectral Clustering</option>
                                        <option value="birch">BIRCH</option>
                                        <option value="affinity_propagation">Affinity Propagation</option>
                                        <option value="bisecting_kmeans">Bisecting K-Means</option>
                                        <option value="hdbscan">HDBSCAN</option>
                                        <option value="meanshift">Mean Shift</option>
                                    </optgroup>
                                    <optgroup label="Additional Models (Alphabetical)">
                                        <option value="minibatch_kmeans">Mini-Batch K-Means</option>
                                        <option value="optics">OPTICS</option>
                                    </optgroup>
                                </select>
                            </div>
                            
                            <div id="automlClassifierModels" class="hidden">
                                <label for="automlClassModels" style="display: block; margin-bottom: 8px; font-weight: 600;">Classification Model (Optional):</label>
                                <select name="automlClassModels" id="automlClassModels" style="width: 100%; max-width: 400px; padding: 8px; border: 1px solid #ccc; border-radius: 4px;">
                                    <option value="" selected>Let AutoML choose</option>
                                    <optgroup label="Most Common">
                                        <option value="Logistic_classifier">Logistic Classifier</option>
                                        <option value="RF_classifier">Random Forest Classifier</option>
                                        <option value="SVC_classifier">SVC Classifier</option>
                                        <option value="MLP_classifier">MLP Classifier</option>
                                        <option value="ExtraTrees_classifier">Extra Trees Classifier</option>
                                        <option value="SGD_classifier">SGD Classifier</option>
                                        <option value="AdaBoost_classifier">AdaBoost Classifier</option>
                                        <option value="GradientBoosting_classifier">Gradient Boosting Classifier</option>
                                        <option value="KNeighbors_classifier">K-Neighbors Classifier</option>
                                    </optgroup>
                                    <optgroup label="Multi-Output Models (Multiple Targets)">
                                        <option value="GaussianNB_classifier">Gaussian Naive Bayes</option>
                                        <option value="BernoulliNB_classifier">Bernoulli Naive Bayes</option>
                                        <option value="CategoricalNB_classifier">Categorical Naive Bayes</option>
                                        <option value="ComplementNB_classifier">Complement Naive Bayes</option>
                                        <option value="MultinomialNB_classifier">Multinomial Naive Bayes</option>
                                    </optgroup>
                                    <optgroup label="Additional Models (Alphabetical)">
                                        <option value="Bagging_classifier">Bagging Classifier</option>
                                        <option value="DecisionTree_classifier">Decision Tree Classifier</option>
                                        <option value="HistGradientBoosting_classifier">Histogram Gradient Boosting</option>
                                        <option value="LDA_classifier">Linear Discriminant Analysis</option>
                                        <option value="LinearSVC_classifier">Linear SVC</option>
                                        <option value="NuSVC_classifier">Nu-SVC</option>
                                        <option value="PassiveAggressive_classifier">Passive Aggressive Classifier</option>
                                        <option value="QDA_classifier">Quadratic Discriminant Analysis</option>
                                        <option value="Ridge_classifier">Ridge Classifier</option>
                                    </optgroup>
                                </select>
                            </div>
                        </div>
                        
                        <div class="preprocess-card">
                            <h2>AutoML Configuration</h2>
                            <p class="field-note" style="margin-bottom: 16px;">AutoML will automatically select the best model, optimize hyperparameters, handle feature selection and outliers, and perform cross-validation.</p>
                            
                            <div class="scaling-container" style="margin-bottom: 20px;">
                                <label for="automlIntensity" style="display: block; margin-bottom: 8px; font-weight: 600;">AutoML Intensity Level:</label>
                                <select name="automlIntensity" id="automlIntensity" style="width: 100%; max-width: 400px; padding: 8px; border: 1px solid #ccc; border-radius: 4px;">
                                    <option value="quick">Quick (Fast, minimal preprocessing)</option>
                                    <option value="medium" selected>Medium (Balanced, moderate preprocessing)</option>
                                    <option value="long">Long (Thorough, comprehensive preprocessing)</option>
                                </select>
                                <p class="field-note" style="margin-top: 8px; font-size: 0.9em;">
                                    <strong>Quick:</strong> Focuses on hyperparameter optimization only (no feature selection or outlier detection). Fastest option.<br>
                                    <strong>Medium:</strong> Includes feature selection and outlier detection with moderate hyperparameter search. Good balance of speed and thoroughness.<br>
                                    <strong>Long:</strong> Comprehensive preprocessing (feature selection, outlier detection) with exhaustive grid search. Slowest but most thorough.
                                </p>
                            </div>
                            
                            <div class="preprocess-card" style="margin-top: 20px;">
                                <h2>AutoML Settings Summary</h2>
                                <p class="field-note" style="margin-bottom: 16px;">AutoML will use the following configuration:</p>
                                <div id="automlSettingsDisplay" style="padding: 16px; background-color: #f0f7ff; border: 2px solid #357a53; border-radius: 8px; margin: 12px 0;">
                                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; font-size: 0.95rem;">
                                        <div><strong>Model:</strong> <span id="automlDisplayModel" style="color: #2c3e50;">-</span></div>
                                        <div><strong>Scaler:</strong> <span id="automlDisplayScaler" style="color: #2c3e50;">-</span></div>
                                        <div><strong>Feature Selection:</strong> <span id="automlDisplayFeatureSelection" style="color: #2c3e50;">RFE (10 features)</span></div>
                                        <div><strong>Outlier Handling:</strong> <span id="automlDisplayOutlier" style="color: #2c3e50;">Isolation Forest (remove)</span></div>
                                        <div><strong>Hyperparameter Search:</strong> <span id="automlDisplayHyperparameter" style="color: #2c3e50;">Randomized (50 iterations, 5 CV folds)</span></div>
                                        <div><strong>Cross-Validation:</strong> <span id="automlDisplayCrossValidation" style="color: #2c3e50;">KFold (5 folds)</span></div>
                                    </div>
                                </div>
                            </div>
                            
                            <!-- Units and Significant Figures Section -->
                            <div style="margin-top: 30px; padding-top: 20px; border-top: 1px solid #e5e5e5;">
                                <h3 style="margin-top: 0; margin-bottom: 16px; font-size: 1.2em; font-weight: 600;">Output Formatting</h3>
                                
                                <div class="toggle-container" style="margin-bottom: 16px;">
                                    <h4 style="margin: 0; font-size: 1.1em; font-weight: 600;">Does Output Have Units?</h4>
                                    <label class="switch">
                                        <input type="checkbox" id="automlUnitToggle">
                                        <span class="slider"></span>
                                    </label>
                                    <div id="automlUnits" class="hidden" style="margin-top: 12px;">
                                        <label for="automlUnitName">Enter Unit (u* for µ):</label>
                                        <input type="text" id="automlUnitName" name="automlUnitName" style="max-width: 200px;">
                                    </div>
                                </div>
                                
                                <div class="scaling-container">
                                    <label for="automlSigfig">Select Number of Significant Figures</label>
                                    <input type="number" id="automlSigfig" name="automlSigfig" value="3" placeholder="3" min="1" max="10" style="max-width: 100px;">
                                </div>
                            </div>
                            
                            <div style="margin-top: 20px; padding: 16px; background-color: #f8f9fa; border-radius: 8px; border: 1px solid #e0e0e0;">
                                <div style="display: flex; gap: 12px; align-items: center; margin-bottom: 12px;">
                                    <button type="submit" class='processButton processButton--compact success-button' id="automlSubmitButton">Run AutoML</button>
                                    <button type="button" class='secondary-button' id="stopAutomlButton" style="display: none; padding: 12px 24px;">Stop AutoML</button>
                                </div>
                                <p class="field-note" style="color: #856404; margin-top: 12px; margin-bottom: 0;">
                                    <strong>Warning:</strong> AutoML performs extensive optimization and may take several hours to complete, especially with large datasets.
                                </p>
                            </div>
                        </div>
                    </div>
                </form>
                
                <!-- Results section - right side -->
                <div class="visualSection" id="automlModelingResults">
                    <div class="model-card">
                        <h2 style="margin-top: 0; margin-bottom: 20px;">Modeling Results</h2>
                        <div id="AutoMLNumericResultDiv" role="region" aria-label="AutoML regression model results" class="hidden">
                            <div id="automlImageSelector"></div>
                        </div>
                        <div id="AutoMLClusterResultDiv" role="region" aria-label="AutoML clustering model results" class="hidden"></div>
                        <div id="AutoMLClassifierResultDiv" role="region" aria-label="AutoML classification model results" class="hidden"></div>
                        <div id="automlResultsPlaceholder" style="padding: 40px; text-align: center; color: #666; font-style: italic;">
                            <p>Results will appear here after you run AutoML.</p>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Loading div outside model-layout to preserve grid structure -->
            <div id="automlLoading" class="hidden" style="margin-top: 20px; padding: 20px; background-color: #f8f9fa; border-radius: 8px; border: 1px solid #e0e0e0; max-width: 100%;" role="status" aria-live="polite" aria-atomic="true" aria-label="AutoML loading status"></div>
        </div>
    </div>
    
    

    <div id="popup" class="popup-container">
        <div class="popup-box">
            <h2>Glossary</h2>
            <p>RMSE is Root Mean Square Error. It measures the differences between true or predicted values</p>
            <p>MAE is Mean Absolute Error. Absolute error is the magnitude of the difference between the prediction and the observation's value. MAE is the average absolute error values for a set of predictions and observations. It measures the magnitude of errors for the group.</p>
            <p>A MAE or RMSE of 0 is a perfect prediction so lower scores are better but vary depending on the values in the dataset </p>
            <button class="close-btn" onclick="closePopup()">×</button>
        </div>
    </div>

    <div id="resetPopup" class="popup-container">
        <div class="popup-box">
            <h2>Are you sure you want to restart?</h2>
            <p>Restarting will lose all of your data</p>
            <button onclick="closeResetPopup()">Cancel</button> 
            &nbsp; &nbsp;
            <button class='orangebutton' onclick="fileUploadPage()">Restart</button> 
            
        </div>
    </div>

<!-- Advanced Modeling Page -->
    <div id="advancedOptimization" class="container hidden">
        <div class="form">
            <div class="section-header">
                <div class="section-header-content">
                    <h2>Advanced Modeling</h2>
                    <button type="button" class="secondary-button" id="backToModelingFromAdvanced">Back to Model Preprocessing</button>
                </div>
            </div>
            <div class="topSection">
            </div>

            <form id="advancedOptimizationForm">
                <div class="model-layout">
                    <div class="modelSection">
                        <div class="preprocess-card" style="margin-bottom: 20px;">
                            <h2>AutoML Mode</h2>
                            <p class="field-note" style="margin-bottom: 16px;">Enable AutoML Mode to automatically configure all advanced options. The system will select optimal settings for feature selection, outlier handling, hyperparameter tuning, and cross-validation.</p>
                            <div class="scaling-container">
                                <div class="toggle-container" style="display: flex; align-items: center; gap: 12px;">
                                    <label for="automlMode" style="font-weight: 600; margin: 0;">Enable AutoML Mode</label>
                                    <label class="switch">
                                        <input type="checkbox" id="automlMode" name="automlMode" aria-label="Enable AutoML Mode to automatically configure all advanced options">
                                        <span class="slider" aria-hidden="true"></span>
                                    </label>
                                </div>
                            </div>
                            <div id="automlInfo" class="hidden" style="margin-top: 12px; padding: 12px; background-color: #e7f3ff; border-left: 4px solid #357a53; border-radius: 4px;">
                                <p style="margin: 0; color: #2c3e50; font-size: 0.95rem;"><strong>AutoML will automatically configure:</strong></p>
                                <ul style="margin: 8px 0 0 0; padding-left: 20px; color: #2c3e50; font-size: 0.95rem;">
                                    <li>Feature selection with optimal method and feature count</li>
                                    <li>Outlier detection and handling</li>
                                    <li>Hyperparameter optimization</li>
                                    <li>Cross-validation for robust evaluation</li>
                                </ul>
                            </div>
                        </div>
                        <div class="preprocess-card">
                            <h2>Model Selection</h2>
                            <p class="field-note" style="margin-bottom: 16px;">Select your model type and specific model. Hyperparameter options will appear below based on your selection.</p>
                            
                            <div id="advancedNumericModels" class="hidden">
                                <label for="advancedNModels"><h3>Select Model</h3></label>
                                <select name="advancedNModels" id="advancedNModels">
                                    <option value="" disabled selected>-- Select an option --</option>
                                    <optgroup label="Most Common">
                                        <option value="Linear">Linear</option>
                                        <option value="Ridge">Ridge</option>
                                        <option value="Lasso">Lasso</option>
                                        <option value="ElasticNet">Elastic Net</option>
                                        <option value="RF">Random Forest</option>
                                        <option value="gradient_boosting">Gradient Boosting</option>
                                        <option value="SVM">Support Vector Machine (SVR)</option>
                                        <option value="MLP">Multi-Layer Perceptron</option>
                                        <option value="K-Nearest">K-Nearest Neighbors</option>
                                        <option value="ExtraTrees">Extra Trees</option>
                                    </optgroup>
                                    <optgroup label="Additional Models (Alphabetical)">
                                        <option value="AdaBoost">AdaBoost Regressor</option>
                                        <option value="ARDRegression">ARD Regression</option>
                                        <option value="Bagging">Bagging Regressor</option>
                                        <option value="BayesianRidge">Bayesian Ridge</option>
                                        <option value="DecisionTree">Decision Tree Regressor</option>
                                        <option value="ElasticNetCV">Elastic Net CV</option>
                                        <option value="HistGradientBoosting">Histogram Gradient Boosting</option>
                                        <option value="Huber">Huber Regressor</option>
                                        <option value="LARS">LARS</option>
                                        <option value="LARSCV">LARS CV</option>
                                        <option value="LassoCV">Lasso CV</option>
                                        <option value="LinearSVR">Linear SVR</option>
                                        <option value="NuSVR">Nu-SVR</option>
                                        <option value="OMP">Orthogonal Matching Pursuit</option>
                                        <option value="PassiveAggressive">Passive Aggressive Regressor</option>
                                        <option value="Quantile">Quantile Regressor</option>
                                        <option value="RadiusNeighbors">Radius Neighbors Regressor</option>
                                        <option value="RANSAC">RANSAC Regressor</option>
                                        <option value="RidgeCV">Ridge CV</option>
                                        <option value="SGD">SGD Regressor</option>
                                        <option value="TheilSen">Theil-Sen Regressor</option>
                                    </optgroup>
                                </select>
                            </div>

                            <div id="advancedClusterModels" class="hidden">
                                <label for="advancedClModels"><h3>Select Model</h3></label>
                                <select name="advancedClModels" id="advancedClModels">
                                    <option value="" disabled selected>-- Select an option --</option>
                                    <optgroup label="Most Common">
                                        <option value="kmeans">K-Means</option>
                                        <option value="dbscan">DBSCAN</option>
                                        <option value="agglo">Agglomerative</option>
                                        <option value="gmm">Gaussian Mixture</option>
                                        <option value="spectral">Spectral Clustering</option>
                                        <option value="birch">BIRCH</option>
                                        <option value="affinity_propagation">Affinity Propagation</option>
                                        <option value="bisecting_kmeans">Bisecting K-Means</option>
                                        <option value="hdbscan">HDBSCAN</option>
                                        <option value="meanshift">Mean Shift</option>
                                    </optgroup>
                                    <optgroup label="Additional Models (Alphabetical)">
                                        <option value="minibatch_kmeans">Mini-Batch K-Means</option>
                                        <option value="optics">OPTICS</option>
                                    </optgroup>
                                </select>
                            </div>

                            <div id="advancedClassifierModels" class="hidden">
                                <label for="advancedClassModels"><h3>Select Model</h3></label>
                                <select name="advancedClassModels" id="advancedClassModels">
                                    <option value="" disabled selected>-- Select an option --</option>
                                    <optgroup label="Most Common">
                                        <option value="Logistic_classifier">Logistic Classifier</option>
                                        <option value="RF_classifier">Random Forest Classifier</option>
                                        <option value="SVC_classifier">SVC Classifier</option>
                                        <option value="MLP_classifier">MLP Classifier</option>
                                        <option value="ExtraTrees_classifier">Extra Trees Classifier</option>
                                        <option value="SGD_classifier">SGD Classifier</option>
                                        <option value="AdaBoost_classifier">AdaBoost Classifier</option>
                                        <option value="GradientBoosting_classifier">Gradient Boosting Classifier</option>
                                        <option value="KNeighbors_classifier">K-Neighbors Classifier</option>
                                    </optgroup>
                                    <optgroup label="Multi-Output Models (Multiple Targets)">
                                        <option value="GaussianNB_classifier">Gaussian Naive Bayes</option>
                                        <option value="BernoulliNB_classifier">Bernoulli Naive Bayes</option>
                                        <option value="CategoricalNB_classifier">Categorical Naive Bayes</option>
                                        <option value="ComplementNB_classifier">Complement Naive Bayes</option>
                                        <option value="MultinomialNB_classifier">Multinomial Naive Bayes</option>
                                    </optgroup>
                                    <optgroup label="Additional Models (Alphabetical)">
                                        <option value="Bagging_classifier">Bagging Classifier</option>
                                        <option value="DecisionTree_classifier">Decision Tree Classifier</option>
                                        <option value="HistGradientBoosting_classifier">Histogram Gradient Boosting</option>
                                        <option value="LDA_classifier">Linear Discriminant Analysis</option>
                                        <option value="LinearSVC_classifier">Linear SVC</option>
                                        <option value="NuSVC_classifier">Nu-SVC</option>
                                        <option value="PassiveAggressive_classifier">Passive Aggressive Classifier</option>
                                        <option value="QDA_classifier">Quadratic Discriminant Analysis</option>
                                        <option value="Ridge_classifier">Ridge Classifier</option>
                                    </optgroup>
                                </select>
                            </div>
                        </div>

                        <div class="preprocess-card">
                            <h2>Model Hyperparameters</h2>
                            <p class="field-note">Configure all model hyperparameters. Essential hyperparameters appear first, followed by non-essential options. These sections will appear based on the model selected above.</p>
                            
                            <!-- Ridge Hyperparameters -->
                            <div id="advancedRidgeFields" class="hidden">
                                <h3>Ridge Model Settings</h3>
                                <p class="field-note" style="margin-bottom: 12px; margin-top: 0;">Ridge regression adds L2 regularization to prevent overfitting. Higher alpha values increase regularization.</p>
                                <label for="advancedRidgeAlpha">Alpha - regularization strength (float >=0):</label>
                                <input type="number" value=1.0 placeholder=1 min=0 id="advancedRidgeAlpha" name="advancedRidgeAlpha">
                                <div class="nonreqHyperparams">
                                    <div><br></div>
                                    <div class="toggle-container">
                                        <span><h3>Edit Non-Essential Hyperparameters?</h3></span>
                                        <label class="switch">
                                            <input type="checkbox" id="advancedNonreqRidgeSlider">
                                            <span class="slider"></span>
                                        </label>
                                    </div>
                                    <div id="advancedNonreqRidgeParams" class="hidden">
                                        <label for="RidgeFitIntersept">Fit Intercept:</label>
                                        <select name="RidgeFitIntersept" id="RidgeFitIntersept">
                                            <option value="true">True</option>
                                            <option value="false">False</option>
                                        </select>
                                        <label for="RidgeNormalize">Normalize:</label>
                                        <select name="RidgeNormalize" id="RidgeNormalize">
                                            <option value="false">False</option>
                                            <option value="true">True</option>
                                        </select>
                                        <br>
                                        <label for="RidgeCopyX">Copy X:</label>
                                        <select name="RidgeCopyX" id="RidgeCopyX">
                                            <option value="true">True</option>
                                            <option value="false">False</option>
                                        </select>
                                        <label for="RidgePositive">Positive:</label>
                                        <select name="RidgePositive" id="RidgePositive">
                                            <option value="false">False</option>
                                            <option value="true">True</option>
                                        </select>
                                        <div><br></div>
                                        <label for="RidgeSolver">Solver - Optimization method:</label>
                                        <select name="solver" id="RidgeSolver">
                                            <option value="auto">auto</option>
                                            <option value="svd">svd</option>
                                            <option value="cholesky">cholesky</option>
                                            <option value="lsqr">lsqr</option>
                                            <option value="sparse_cg">sparse_cg</option>
                                            <option value="sag">sag</option>
                                            <option value="saga">saga</option>
                                            <option value="lbfgs">lbfgs</option>
                                        </select>
                                        <div><br></div>
                                        <label for="RidgeMaxIter">Max Iterations (integer >=1):</label>
                                        <input type="number" step=1 min=1 id="RidgeMaxIter" name="RidgeMaxIter">
                                        <div><br></div>
                                        <label for="RidgeTol">Tol (float >0):</label>
                                        <input type="number" value=.0001 placeholder=.0001 min="0.0000001" step="any" id="RidgeTol" name="RidgeTol">
                                    </div>
                                </div>
                            </div>
                            
                            <!-- Lasso Hyperparameters -->
                            <div id="advancedLassoFields" class="hidden">
                                <h3>Lasso Model Settings</h3>
                                <p class="field-note" style="margin-bottom: 12px; margin-top: 0;">Lasso regression uses L1 regularization, which can set coefficients to zero for feature selection.</p>
                                <label for="advancedLassoAlpha">Alpha - regularization strength (float >=0):</label>
                                <input type="number" value=1 min=0 placeholder=1 id="advancedLassoAlpha" name="advancedLassoAlpha">
                                <div class="nonreqHyperparams">
                                    <div><br></div>
                                    <div class="toggle-container">
                                        <span><h3>Edit Non-Essential Hyperparameters?</h3></span>
                                        <label class="switch">
                                            <input type="checkbox" id="advancedNonreqLassoSlider">
                                            <span class="slider"></span>
                                        </label>
                                    </div>
                                    <div id="advancedNonreqLassoParams" class="hidden">
                                        <label for="LassoFitIntersept">Fit Intercept:</label>
                                        <select name="LassoFitIntersept" id="LassoFitIntersept">
                                            <option value="true">True</option>
                                            <option value="false">False</option>
                                        </select>
                                        <label for="LassoPrecompute">Precompute:</label>
                                        <select name="LassoPrecompute" id="LassoPrecompute">
                                            <option value="false">False</option>
                                            <option value="true">True</option>
                                        </select>
                                        <label for="LassoCopyX">Copy X:</label>
                                        <select name="LassoCopyX" id="LassoCopyX">
                                            <option value="true">True</option>
                                            <option value="false">False</option>
                                        </select>
                                        <div><br></div>
                                        <label for="LassoWarmStart">Warm Start:</label>
                                        <select name="LassoWarmStart" id="LassoWarmStart">
                                            <option value="false">False</option>
                                            <option value="true">True</option>
                                        </select>
                                        <label for="LassoPositive">Positive:</label>
                                        <select name="LassoPositive" id="LassoPositive">
                                            <option value="false">False</option>
                                            <option value="true">True</option>
                                        </select>
                                        <div><br></div>
                                        <label for="LassoMax_iter">Max Iterations - number of iterations (integer >=1):</label>
                                        <input type="number" step=1 value=1000 placeholder=1000 min=1 id="LassoMax_iter" name="max_iter">
                                        <div><br></div>
                                        <label for="LassoTol">Tol (float >0):</label>
                                        <input type="number" value=0.0001 placeholder=.0001 min="0.0000001" step="any" id="LassoTol" name="LassoTol">
                                        <div><br></div>
                                        <label for="LassoSelection">Selection:</label>
                                        <select name="LassoSelection" id="LassoSelection">
                                            <option value="cyclic">Cyclic</option>
                                            <option value="random">Random</option>
                                        </select>
                                    </div>
                                </div>
                            </div>
                            
                            <!-- Logistic Non-Essential Hyperparameters -->
                            <div id="advancedLogisticFields" class="hidden">
                                <div class="nonreqHyperparams">
                                    <div class="toggle-container">
                                        <span><h3>Edit Non-Essential Hyperparameters?</h3></span>
                                        <label class="switch">
                                            <input type="checkbox" id="advancedNonreqLogisticSlider">
                                            <span class="slider"></span>
                                        </label>
                                    </div>
                                    <div id="advancedNonreqLogisticParams" class="hidden">
                                        <label for="LogisticDual">Dual:</label>
                                        <select name="LogisticDual" id="LogisticDual">
                                            <option value="false">False</option>
                                            <option value="true">True</option>
                                        </select>
                                        <label for="LogisticFitIntercept">Fit Intercept:</label>
                                        <select name="LogisticFitIntercept" id="LogisticFitIntercept">
                                            <option value="true">True</option>
                                            <option value="false">False</option>
                                        </select>
                                        <label for="LogisticSolver">Solver:</label>
                                        <select name="LogisticSolver" id="LogisticSolver">
                                            <option value="lbfgs">lbfgs</option>
                                            <option value="newton-cg">newton-cg</option>
                                            <option value="liblinear">liblinear</option>
                                            <option value="sag">sag</option>
                                            <option value="saga">saga</option>
                                        </select>
                                        <div><br></div>
                                        <label for="LogisticMultiClass">Multi Class:</label>
                                        <select name="LogisticMultiClass" id="LogisticMultiClass">
                                            <option value="auto">auto</option>
                                            <option value="ovr">ovr</option>
                                            <option value="multinomial">multinomial</option>
                                        </select>
                                        <label for="LogisticWarmStart">Warm Start:</label>
                                        <select name="LogisticWarmStart" id="LogisticWarmStart">
                                            <option value="false">False</option>
                                            <option value="true">True</option>
                                        </select>
                                        <div><br></div>
                                        <label for="CLogistic">C - Regularization (float >0):</label>
                                        <input type="number" value=1 placeholder=1.0 min="0.0000001" step="any" id="CLogistic" name="CLogistic">
                                        <div><br></div>
                                        <label for="penalty">Penalty:</label>
                                        <select name="penalty" id="penalty">
                                            <option value="l2">l2</option>
                                            <option value="l1">l1</option>
                                            <option value="elasticnet">elasticnet</option>
                                            <option value="None">None</option>
                                        </select>
                                        <div><br></div>
                                        <label for="LogisticTol">Tol (float >0):</label>
                                        <input type="number" value=0.0001 placeholder=.0001 min="0.0000001" step="any" id="LogisticTol" name="LogisticTol">
                                        <div><br></div>
                                        <label for="Logisticintercept_scaling">Intercept Scaling:</label>
                                        <input type="number" value=1 placeholder=1 step="any" id="Logisticintercept_scaling" name="Logisticintercept_scaling">
                                        <div><br></div>
                                        <label for="LogisticClassWeight">Class Weight (dict or 'balanced'):</label>
                                        <input type="text" id="LogisticClassWeight" name="LogisticClassWeight">
                                        <div><br></div>
                                        <label for="LogisticMaxIterations">Max Iterations (integer >=1):</label>
                                        <input type="number" step=1 min="1" value=100 placeholder="100" id="LogisticMaxIterations" name="LogisticMaxIterations">
                                        <div><br></div>
                                        <label for="LogisticVerbose">Verbose (int):</label>
                                        <input type="number" step=1 value=0 placeholder="0" id="LogisticVerbose" name="LogisticVerbose">
                                        <div><br></div>
                                        <label for="LogisticNJobs">N Jobs (int):</label>
                                        <input type="number" step=1 id="LogisticNJobs" name="LogisticNJobs">
                                        <div><br></div>
                                        <label for="Logisticl1Ratio">L1 Ratio (float [0, 1]):</label>
                                        <input type="number" min="0.0000001" step="any" max=1 id="Logisticl1Ratio" name="Logisticl1Ratio">
                                    </div>
                                </div>
                            </div>
                            
                            <!-- SVM Hyperparameters -->
                            <div id="advancedSVMFields" class="hidden">
                                <h3>SVM Model Settings</h3>
                                <label for="advancedC">C - Regularization (float >0):</label>
                                <input type="number" value=1 placeholder=1 min="0.0000001" step="any" id="advancedC" name="advancedC">
                                <div><br></div>
                                <label for="advancedKernel">Kernel:</label>
                                <select name="advancedKernel" id="advancedKernel">
                                    <option value="rbf">rbf</option>
                                    <option value="linear">linear</option>
                                    <option value="poly">poly</option>
                                    <option value="sigmoid">sigmoid</option>
                                    <option value="precomputed">precomputed</option>
                                </select>
                                <div id="advancedPolykernelFields" class="hidden">
                                    <div><br></div>
                                    <label for="advancedPolyDegree">Degree of poly kernel function (int):</label>
                                    <input type="number" step=1 value=3 placeholder=3.0 id="advancedPolyDegree" name="advancedPolyDegree">
                                </div>
                                <div id="advancedSvmGamma">
                                    <div><br></div>
                                    <label for="advancedGamma">Gamma - enter 'auto', 'scale', or a float:</label>
                                    <input type="text" value='scale' placeholder='scale' id="advancedGamma" name="advancedGamma">
                                </div>
                                <div class="nonreqHyperparams">
                                    <div><br></div>
                                    <div class="toggle-container">
                                        <span><h3>Edit Non-Essential Hyperparameters?</h3></span>
                                        <label class="switch">
                                            <input type="checkbox" id="advancedNonreqSVMSlider">
                                            <span class="slider"></span>
                                        </label>
                                    </div>
                                    <div id="advancedNonreqSVMParams" class="hidden">
                                        <label for="SVMshrinking">Shrinking:</label>
                                        <select name="SVMshrinking" id="SVMshrinking">
                                            <option value="true">True</option>
                                            <option value="false">False</option>
                                        </select>
                                        <label for="SVMprobability">Probability:</label>
                                        <select name="SVMprobability" id="SVMprobability">
                                            <option value="false">False</option>
                                            <option value="true">True</option>
                                        </select>
                                        <br>
                                        <label for="SVMBreakTies">Break Ties:</label>
                                        <select name="SVMBreakTies" id="SVMBreakTies">
                                            <option value="false">False</option>
                                            <option value="true">True</option>
                                        </select>
                                        <label for="SVMverbose">Verbose:</label>
                                        <select name="SVMverbose" id="SVMverbose">
                                            <option value="false">False</option>
                                            <option value="true">True</option>
                                        </select>
                                        <div><br></div>
                                        <label for="SVMcoef0">coef0 (float):</label>
                                        <input type="number" step="any" value=0 placeholder="0" id="SVMcoef0" name="SVMcoef0">
                                        <div><br></div>
                                        <label for="SVMtol">tol(float):</label>
                                        <input type="number" step="any" value=.001 placeholder=.001 id="SVMtol" name="SVMtol">
                                        <div><br></div>
                                        <label for="SVMCacheSize">Cache Size (float):</label>
                                        <input type="number" step="any" value=200 placeholder=200 id="SVMCacheSize" name="SVMCacheSize">
                                        <div><br></div>
                                        <label for="SVMClassWeight">Class Weight (enter a dictionary or 'balanced')</label>
                                        <input type="text" id="SVMClassWeight" name="SVMClassWeight">
                                        <div><br></div>
                                        <label for="SVMmaxIter">Max Iterations (int):</label>
                                        <input type="number" step=1 value=-1 placeholder=-1 id="SVMmaxIter" name="SVMmaxIter">
                                        <div><br></div>
                                        <label for="SVMdecisionFunctionShape">Decision Function Shape:</label>
                                        <select name="SVMdecisionFunctionShape" id="SVMdecisionFunctionShape">
                                            <option value="ovr">ovr</option>
                                            <option value="ovo">ovo</option>
                                        </select>
                                    </div>
                                </div>
                            </div>
                            
                            <!-- Random Forest Hyperparameters -->
                            <div id="advancedRFFields" class="hidden">
                                <h3>Random Forest Model Settings</h3>
                                <p class="field-note" style="margin-bottom: 12px; margin-top: 0;">Random Forest combines multiple decision trees. More trees generally improve performance but increase computation time.</p>
                                <label for="advancedRFn_estmators">N Estimators - # of trees (integer >=1):</label>
                                <input type="number" step=1 value=100 placeholder="100" min=1 id="advancedRFn_estmators" name="advancedRFn_estmators">
                                <div class="nonreqHyperparams">
                                    <div><br></div>
                                    <div class="toggle-container">
                                        <span><h3>Edit Non-Essential Hyperparameters?</h3></span>
                                        <label class="switch">
                                            <input type="checkbox" id="advancedNonreqRFSlider">
                                            <span class="slider"></span>
                                        </label>
                                    </div>
                                    <div id="advancedNonreqRFParams" class="hidden">
                                        <label for="RFBoostrap">Bootstrap:</label>
                                        <select name="RFBoostrap" id="RFBoostrap">
                                            <option value="true">True</option>
                                            <option value="false">False</option>
                                        </select>
                                        <label for="RFoobScore">oob Score:</label>
                                        <select name="RFoobScore" id="RFoobScore">
                                            <option value="false">False</option>
                                            <option value="true">True</option>
                                        </select>
                                        <label for="RFWarmStart">Warm Start:</label>
                                        <select name="RFWarmStart" id="RFWarmStart">
                                            <option value="false">False</option>
                                            <option value="true">True</option>
                                        </select>
                                        <div><br></div>
                                        <label for="RFmin_weight_fraction_leaf">Min Weight Fraction Leaf (float [0.0, 0.5]):</label>
                                        <input type="number" value=0 placeholder=0 min="0" max=".5" step="any" id="RFmin_weight_fraction_leaf" name="RFmin_weight_fraction_leaf">
                                        <div><br></div>
                                        <label for="RFMaxLeafNodes">Max Leaf Nodes (an integer or leave blank for None):</label>
                                        <input type="number" step="1" id="RFMaxLeafNodes" name="RFMaxLeafNodes">
                                        <div><br></div>
                                        <label for="RFMinImpurityDecrease">Min Impurity Decrease (float):</label>
                                        <input type="number" value=0 placeholder=0 step="any" id="RFMinImpurityDecrease" name="RFMinImpurityDecrease">
                                        <div><br></div>
                                        <label for="RFNJobs">N Jobs (an integer or leave blank for None):</label>
                                        <input type="number" step="1" id="RFNJobs" name="RFNJobs">
                                        <div><br></div>
                                        <label for="RFVerbose">Verbose (int):</label>
                                        <input type="number" value=0 placeholder=0 step="1" id="RFVerbose" name="RFVerbose">
                                        <div><br></div>
                                        <label for="RFMax_depth">Max Depth - Tree depth (an integer or leave blank for None):</label>
                                        <input type="number" step=1 id="RFMax_depth" name="max_depth">
                                        <div><br></div>
                                        <label for="min_samples_split">Min Samples Split - Min samples per split (integer or float):</label>
                                        <input type="number" value=2 placeholder="2" id="min_samples_split" name="min_samples_split">
                                        <div><br></div>
                                        <label for="min_samples_leaf">Min Samples Leaf - Min samples per leaf (integer or float):</label>
                                        <input type="number" value=1 placeholder=1 id="min_samples_leaf" name="min_samples_leaf">
                                    </div>
                                </div>
                            </div>
                            
                            <!-- Perceptron Non-Essential Hyperparameters -->
                            <div id="advancedPerceptronFields" class="hidden">
                                <div class="nonreqHyperparams">
                                    <div><br></div>
                                    <div class="toggle-container">
                                        <span><h3>Edit Non-Essential Hyperparameters?</h3></span>
                                        <label class="switch">
                                            <input type="checkbox" id="advancedNonreqPerceptronSlider">
                                            <span class="slider"></span>
                                        </label>
                                    </div>
                                    <div id="advancedNonreqPerceptronParams" class="hidden">
                                        <label for="PerceptronFitIntercept">Fit Intercept:</label>
                                        <select name="PerceptronFitIntercept" id="PerceptronFitIntercept">
                                            <option value="true">True</option>
                                            <option value="false">False</option>
                                        </select>
                                        <label for="PerceptronShuffle">Shuffle:</label>
                                        <select name="PerceptronShuffle" id="PerceptronShuffle">
                                            <option value="true">True</option>
                                            <option value="false">False</option>
                                        </select>
                                        <label for="PerceptronEarlyStopping">Early Stopping:</label>
                                        <select name="PerceptronEarlyStopping" id="PerceptronEarlyStopping">
                                            <option value="false">False</option>
                                            <option value="true">True</option>
                                        </select>
                                        <label for="PerceptronWarmStart">Warm Start:</label>
                                        <select name="PerceptronWarmStart" id="PerceptronWarmStart">
                                            <option value="false">False</option>
                                            <option value="true">True</option>
                                        </select>
                                        <div><br></div>
                                        <label for="PerceptronPenalty">Penalty:</label>
                                        <select name="PerceptronPenalty" id="PerceptronPenalty">
                                            <option value="None">None</option>
                                            <option value="l2">l2</option>
                                            <option value="l1">l1</option>
                                            <option value="elasticnet">elasticnet</option>
                                        </select>
                                        <div><br></div>
                                        <label for="PerceptronAlpha">Alpha (float):</label>
                                        <input type="number" step="any" value=.0001 placeholder=.0001 id="PerceptronAlpha" name="PerceptronAlpha">
                                        <div><br></div>
                                        <label for="PerceptronTol">Tol (float):</label>
                                        <input type="number" step="any" value=.001 placeholder=.001 id="PerceptronTol" name="PerceptronTol">
                                        <div><br></div>
                                        <label for="PerceptronVerbose">Verbose (int):</label>
                                        <input type="number" step="1" value=0 placeholder=0 id="PerceptronVerbose" name="PerceptronVerbose">
                                        <div><br></div>
                                        <label for="PerceptronNJobs">N Jobs (an integer or or leave blank for None):</label>
                                        <input type="number" step="1" id="PerceptronNJobs" name="PerceptronNJobs">
                                        <div><br></div>
                                        <label for="PerceptronValidationFraction">Validation Fraction (float):</label>
                                        <input type="number" step="any" value=.1 placeholder=.1 id="PerceptronValidationFraction" name="PerceptronValidationFraction">
                                        <div><br></div>
                                        <label for="PerceptronNIterNoChange">Number Iterations No Change (int):</label>
                                        <input type="number" step="1" value=5 placeholder=5 id="PerceptronNIterNoChange" name="PerceptronNIterNoChange">
                                        <div><br></div>
                                        <label for="PerceptronClassWeight">Class Weight (enter a dictionary or 'balanced'):</label>
                                        <input type="text" id="PerceptronClassWeight" name="PerceptronClassWeight">
                                    </div>
                                </div>
                            </div>
                            
                            <!-- MLP Non-Essential Hyperparameters -->
                            <div id="advancedMLPFields" class="hidden">
                                <h3>MLP Model Settings</h3>
                                <p class="field-note" style="margin-bottom: 12px; margin-top: 0;">Multi-Layer Perceptron is a neural network. Configure hidden layers and learning parameters for optimal performance.</p>
                                <div class="nonreqHyperparams">
                                    <div><br></div>
                                    <div class="toggle-container">
                                        <span><h3>Edit Non-Essential Hyperparameters?</h3></span>
                                        <label class="switch">
                                            <input type="checkbox" id="advancedNonreqMLPSlider">
                                            <span class="slider"></span>
                                        </label>
                                    </div>
                                    <div id="advancedNonreqMLPParams" class="hidden">
                                        <label for="MLPNesterovsMomentum">Nesterovs Momentum:</label>
                                        <select name="MLPNesterovsMomentum" id="MLPNesterovsMomentum">
                                            <option value="true">True</option>
                                            <option value="false">False</option>
                                        </select>
                                        <label for="MLPEarlyStopping">Early Stopping:</label>
                                        <select name="MLPEarlyStopping" id="MLPEarlyStopping">
                                            <option value="false">False</option>
                                            <option value="true">True</option>
                                        </select>
                                        <label for="MLPShuffle">Shuffle:</label>
                                        <select name="MLPShuffle" id="MLPShuffle">
                                            <option value="true">True</option>
                                            <option value="false">False</option>
                                        </select>
                                        <div><br></div>
                                        <label for="MLPVerbose">Verbose:</label>
                                        <select name="MLPVerbose" id="MLPVerbose">
                                            <option value="false">False</option>
                                            <option value="true">True</option>
                                        </select>
                                        <label for="MLPWarmStart">Warm Start:</label>
                                        <select name="MLPWarmStart" id="MLPWarmStart">
                                            <option value="false">False</option>
                                            <option value="true">True</option>
                                        </select>
                                        <div><br></div>
                                        <label for="MLPAlpha">Alpha - Regularization (float >=0):</label>
                                        <input type="number" value=.0001 placeholder=.0001 step="any" min=0 id="MLPAlpha" name="alpha">
                                        <div><br></div>
                                        <label for="MLPLearning_rate">Learning Rate:</label>
                                        <select name="MLPLearning_rate" id="MLPLearning_rate">
                                            <option value="constant">constant</option>
                                            <option value="invscaling">invscaling</option>
                                            <option value="adaptive">adaptive</option>
                                        </select>
                                        <div><br></div>
                                        <label for="MLPBatchSize">Batch Size (integer >=1 or 'auto'):</label>
                                        <input type="text" value=200 placeholder="200" id="MLPBatchSize" name="MLPBatchSize">
                                        <div><br></div>
                                        <label for="MLPLearningRateInit">Learning Rate Init (float >0):</label>
                                        <input type="number" value=.001 placeholder=".001" min="0.0000001" step="any" id="MLPLearningRateInit" name="MLPLearningRateInit">
                                        <div><br></div>
                                        <label for="MLPPowerT">Power T (float):</label>
                                        <input type="number" value=.05 placeholder=.5 step="any" id="MLPPowerT" name="MLPPowerT">
                                        <div><br></div>
                                        <label for="MLPMaxIter">Max Iterations (integer >=1):</label>
                                        <input type="number" value=200 placeholder=200 step="1" min="1" id="MLPMaxIter" name="MLPMaxIter">
                                        <div><br></div>
                                        <label for="MLPTol">Tol (float):</label>
                                        <input type="number" value=.0001 placeholder=.0001 step="any" id="MLPTol" name="MLPTol">
                                        <div><br></div>
                                        <label for="MLPMomentum">Momentum (float [0, 1) ):</label>
                                        <input type="number" value=.09 placeholder=.9 min=0 max=.9999999 step="any" id="MLPMomentum" name="MLPMomentum">
                                        <div><br></div>
                                        <label for="MLPValidationFraction">Validation Fraction (float [0, 1) ):</label>
                                        <input type="number" value=.01 placeholder=.1 min=0 max=.9999999 step="any" id="MLPValidationFraction" name="MLPValidationFraction">
                                        <div><br></div>
                                        <label for="MLPBeta1"> Beta 1 (float [0, 1) ):</label>
                                        <input type="number" value=.09 placeholder=.9 min=0 max=.9999999  step="any" id="MLPBeta1" name="MLPBeta1">
                                        <div><br></div>
                                        <label for="MLPBeta2">Beta 2 (float [0, 1) ):</label>
                                        <input type="number" value=.999 placeholder=.999 min=0 max=.9999999 step="any" id="MLPBeta2" name="MLPBeta2">
                                        <div><br></div>
                                        <label for="MLPEpsilon">Epsilon (float >0):</label>
                                        <input type="number" value=.00000001 min=.00000000001 placeholder=.00000001 step="any" id="MLPEpsilon" name="MLPEpsilon">
                                    </div>
                                </div>
                            </div>
                            
                            <!-- K-Nearest Neighbors Hyperparameters -->
                            <div id="advancedK-NearestFields" class="hidden">
                                <h3>K-Nearest Neighbors Model Settings</h3>
                                <label for="advancedN_neighbors">N Neighbors - # of neighbors (int):</label>
                                <input type="number" step=1 value=5 placeholder=5 id="advancedN_neighbors" name="advancedN_neighbors">
                                <div class="nonreqHyperparams">
                                    <div><br></div>
                                    <div class="toggle-container">
                                        <span><h3>Edit Non-Essential Hyperparameters?</h3></span>
                                        <label class="switch">
                                            <input type="checkbox" id="advancedNonreqKNearestSlider">
                                            <span class="slider"></span>
                                        </label>
                                    </div>
                                    <div id="advancedNonreqKNearestParams" class="hidden">
                                        <label for="KNearestWeights">Weights:</label>
                                        <select name="KNearestWeights" id="KNearestWeights">
                                            <option value="uniform">uniform</option>
                                            <option value="distance">distance</option>
                                            <option value="callable">callable</option>
                                        </select>
                                        <label for="KNearestAlgorithm">Algorithm:</label>
                                        <select name="KNearestAlgorithm" id="KNearestAlgorithm">
                                            <option value="auto">auto</option>
                                            <option value="ball_tree">ball_tree</option>
                                            <option value="kd_tree">kd_tree</option>
                                            <option value="brute">brute</option>
                                        </select>
                                        <div><br></div>
                                        <label for="metric">Metric - (euclidean, manhattan, etc):</label>
                                        <input type="text" id="metric" value='minkowski' name="metric">
                                        <div><br></div>
                                        <label for="KNearestLeafSize">Leaf Size (int):</label>
                                        <input type="number" step="1" value=30 placeholder=30 id="KNearestLeafSize" name="KNearestLeafSize">
                                        <div><br></div>
                                        <label for="KNearestP">P (int):</label>
                                        <input type="number" step="1" value=2 placeholder=2 id="KNearestP" name="KNearestP">
                                        <div><br></div>
                                        <label for="KNearestMetricParams">Metric Params (enter a dictionary or leave blank for None):</label>
                                        <input type="text" id="KNearestMetricParams" name="KNearestMetricParams">
                                        <div><br></div>
                                        <label for="KNearestNJobs">N Jobs (an integer or leave blank for None):</label>
                                        <input type="number" step="1" id="KNearestNJobs" name="KNearestNJobs">
                                    </div>
                                </div>
                            </div>
                            
                            <!-- Gradient Boosting Hyperparameters -->
                            <div id="advancedGradientBoostingFields" class="hidden">
                                <h3>Gradient Boosting Model Settings</h3>
                                <p class="field-note" style="margin-bottom: 12px; margin-top: 0;">Gradient Boosting builds trees sequentially, each correcting previous errors. Often achieves high performance.</p>
                                <label for="advancedGBn_estimators">N Estimators - Trees (integer >=1):</label>
                                <input type="number" step=1 value=100 placeholder=100 min=1 id="advancedGBn_estimators" name="advancedGBn_estimators">
                                <div><br></div>
                                <label for="advancedGBlearn">Learning Rate (float >0):</label>
                                <input type="number" value=.1 placeholder=.1 min="0.0000001" step="any" id="advancedGBlearn" name="advancedGBlearn">
                                <div class="nonreqHyperparams">
                                    <div><br></div>
                                    <div class="toggle-container">
                                        <span><h3>Edit Non-Essential Hyperparameters?</h3></span>
                                        <label class="switch">
                                            <input type="checkbox" id="advancedNonreqGBSlider">
                                            <span class="slider"></span>
                                        </label>
                                    </div>
                                    <div id="advancedNonreqGBParams" class="hidden">
                                        <label for="GBLoss">Loss:</label>
                                        <select name="GBLoss" id="GBLoss">
                                            <option value="absolute_error">absolute_error</option>
                                            <option value="squared_error">squared_error</option>
                                            <option value="huber">huber</option>
                                            <option value="quantile">quantile</option>
                                        </select>
                                        <label for="GBWarmStart">Warm Start:</label>
                                        <select name="GBWarmStart" id="GBWarmStart">
                                            <option value="false">False</option>
                                            <option value="true">True</option>
                                        </select>
                                        <label for="GBCriterion">Criterion:</label>
                                        <select name="GBCriterion" id="GBCriterion">
                                            <option value="friedman_mse">Friedman MSE</option>
                                            <option value="squared_error">Squared Error</option>
                                        </select>
                                        <div><br></div>
                                        <label for="GBMax_depth">Max Depth - Tree depth (an integer or leave blank for None):</label>
                                        <input type="number" step=1 value=3 placeholder=3 id="GBMax_depth" name="GBMax_depth">
                                        <div><br></div>
                                        <label for="GBSubsample">Subsample (float (0,1]):</label>
                                        <input type="number" value=1 placeholder=1 min="0.0000001" max=1 step="any" id="GBSubsample" name="GBSubsample">
                                        <div><br></div>
                                        <label for="GBMinSamplesSplit">Min Samples Split (integer or float):</label>
                                        <input type="number" value=2 placeholder=2 step="any" id="GBMinSamplesSplit" name="GBMinSamplesSplit">
                                        <div><br></div>
                                        <label for="GBMinSamplesLeaf">Min Samples Leaf (integer or float):</label>
                                        <input type="number" value=1 placeholder=1 step="any" id="GBMinSamplesLeaf" name="GBMinSamplesLeaf">
                                        <div><br></div>
                                        <label for="GBMinWeightFractionLeaf">Min Weight Fraction Leaf (float [0.0, 0.5]):</label>
                                        <input type="number" value=0 placeholder=0 step="any" min=0 max=.5 id="GBMinWeightFractionLeaf" name="GBMinWeightFractionLeaf">
                                        <div><br></div>
                                        <label for="GBMinImpurityDecrease">Min Impurity Decrease (float):</label>
                                        <input type="number" value=0 placeholder=0 step="any" id="GBMinImpurityDecrease" name="GBMinImpurityDecrease">
                                        <div><br></div>
                                        <label for="GBInit">Init ('estimator' or leave blank for None):</label>
                                        <input type="text" placeholder='estimator' id="GBInit" name="GBInit">
                                        <div><br></div>
                                        <label for="GBMaxFeatrues">Max Features (int, float, or string):</label>
                                        <input type="text" id="GBMaxFeatrues" name="GBMaxFeatrues">
                                        <div><br></div>
                                        <label for="GBAlpha">Alpha (float [0.0, 1.0]):</label>
                                        <input type="number" step="any" min=0 max=1 value=.9 placeholder=.9 id="GBAlpha" name="GBAlpha">
                                        <div><br></div>
                                        <label for="GBVerbose">Verbose (int):</label>
                                        <input type="number" step="1" value=0 placeholder=0 id="GBVerbose" name="GBVerbose">
                                        <div><br></div>
                                        <label for="GBMaxLeafNodes">Max Leaf Nodes (an integer or leave blank for None):</label>
                                        <input type="number" step="1" id="GBMaxLeafNodes" name="GBMaxLeafNodes">
                                    </div>
                                </div>
                            </div>
                            
                            <!-- Classifier Non-Essential Hyperparameters -->
                            <div id="advancedLogistic_classifierFields" class="hidden">
                                <h3>Logistic Classifier Model Settings</h3>
                                <p class="field-note" style="margin-bottom: 12px; margin-top: 0;">Logistic regression for classification. Uses a sigmoid function to predict class probabilities.</p>
                                <div class="nonreqHyperparams">
                                    <div class="toggle-container">
                                        <span><h3>Edit Non-Essential Hyperparameters?</h3></span>
                                        <label class="switch">
                                            <input type="checkbox" id="advancedNonreqLogisticClassifierSlider">
                                            <span class="slider"></span>
                                        </label>
                                    </div>
                                    <div id="advancedNonreqLogisticClassifierParams" class="hidden">
                                        <label for="Class_LogisticDual">Dual:</label>
                                        <select name="Class_LogisticDual" id="Class_LogisticDual">
                                            <option value="false">False</option>
                                            <option value="true">True</option>
                                        </select>
                                        <label for="Class_LogisticFitIntercept">Fit Intercept:</label>
                                        <select name="Class_LogisticFitIntercept" id="Class_LogisticFitIntercept">
                                            <option value="true">True</option>
                                            <option value="false">False</option>
                                        </select>
                                        <label for="Class_LogisticSolver">Solver:</label>
                                        <select name="Class_LogisticSolver" id="Class_LogisticSolver">
                                            <option value="lbfgs">lbfgs</option>
                                            <option value="newton-cg">newton-cg</option>
                                            <option value="liblinear">liblinear</option>
                                            <option value="sag">sag</option>
                                            <option value="saga">saga</option>
                                        </select>
                                        <div><br></div>
                                        <label for="Class_LogisticMultiClass">Multi Class:</label>
                                        <select name="Class_LogisticMultiClass" id="Class_LogisticMultiClass">
                                            <option value="auto">auto</option>
                                            <option value="ovr">ovr</option>
                                            <option value="multinomial">multinomial</option>
                                        </select>
                                        <label for="Class_LogisticWarmStart">Warm Start:</label>
                                        <select name="Class_LogisticWarmStart" id="Class_LogisticWarmStart">
                                            <option value="false">False</option>
                                            <option value="true">True</option>
                                        </select>
                                        <div><br></div>
                                        <label for="Class_CLogistic">C - Regularization (float >0):</label>
                                        <input type="number" value=1 placeholder=1.0 min="0.0000001" step="any" id="Class_CLogistic" name="Class_CLogistic">
                                        <div><br></div>
                                        <label for="Class_Logistic_penalty">Penalty:</label>
                                        <select name="Class_Logistic_penalty" id="Class_Logistic_penalty">
                                            <option value="l2">l2</option>
                                            <option value="l1">l1</option>
                                            <option value="elasticnet">elasticnet</option>
                                            <option value="None">None</option>
                                        </select>
                                        <div><br></div>
                                        <label for="Class_LogisticTol">Tol (float >0):</label>
                                        <input type="number" value=0.0001 placeholder=.0001 min="0.0000001" step="any" id="Class_LogisticTol" name="Class_LogisticTol">
                                        <div><br></div>
                                        <label for="Class_Logisticintercept_scaling">Intercept Scaling:</label>
                                        <input type="number" value=1 placeholder=1 step="any" id="Class_Logisticintercept_scaling" name="Class_Logisticintercept_scaling">
                                        <div><br></div>
                                        <label for="Class_LogisticClassWeight">Class Weight (dict or 'balanced'):</label>
                                        <input type="text" id="Class_LogisticClassWeight" name="Class_LogisticClassWeight">
                                        <div><br></div>
                                        <label for="Class_LogisticMaxIterations">Max Iterations (integer >=1):</label>
                                        <input type="number" step=1 min="1" value=100 placeholder="100" id="Class_LogisticMaxIterations" name="Class_LogisticMaxIterations">
                                        <div><br></div>
                                        <label for="Class_LogisticVerbose">Verbose (int):</label>
                                        <input type="number" step=1 value=0 placeholder="0" id="Class_LogisticVerbose" name="Class_LogisticVerbose">
                                        <div><br></div>
                                        <label for="Class_LogisticNJobs">N Jobs (int):</label>
                                        <input type="number" step=1 id="Class_LogisticNJobs" name="Class_LogisticNJobs">
                                        <div><br></div>
                                        <label for="Class_Logisticl1Ratio">L1 Ratio (float [0, 1]):</label>
                                        <input type="number" min="0.0000001" step="any" max=1 id="Class_Logisticl1Ratio" name="Class_Logisticl1Ratio">
                                    </div>
                                </div>
                            </div>
                            
                            <div id="advancedMLP_classifierFields" class="hidden">
                                <h3>MLP Classifier Model Settings</h3>
                                <p class="field-note" style="margin-bottom: 12px; margin-top: 0;">Neural network classifier. Configure hidden layers and learning parameters for classification tasks.</p>
                                <div class="nonreqHyperparams">
                                    <div class="toggle-container">
                                        <span><h3>Edit Non-Essential Hyperparameters?</h3></span>
                                        <label class="switch">
                                            <input type="checkbox" id="advancedNonreqMLPClassifierSlider">
                                            <span class="slider"></span>
                                        </label>
                                    </div>
                                    <div id="advancedNonreqMLPClassifierParams" class="hidden">
                                        <label for="Class_MLPNesterovsMomentum">Nesterovs Momentum:</label>
                                        <select name="Class_MLPNesterovsMomentum" id="Class_MLPNesterovsMomentum">
                                            <option value="true">True</option>
                                            <option value="false">False</option>
                                        </select>
                                        <label for="Class_MLPEarlyStopping">Early Stopping:</label>
                                        <select name="Class_MLPEarlyStopping" id="Class_MLPEarlyStopping">
                                            <option value="false">False</option>
                                            <option value="true">True</option>
                                        </select>
                                        <label for="Class_MLPShuffle">Shuffle:</label>
                                        <select name="Class_MLPShuffle" id="Class_MLPShuffle">
                                            <option value="true">True</option>
                                            <option value="false">False</option>
                                        </select>
                                        <div><br></div>
                                        <label for="Class_MLPVerbose">Verbose:</label>
                                        <select name="Class_MLPVerbose" id="Class_MLPVerbose">
                                            <option value="false">False</option>
                                            <option value="true">True</option>
                                        </select>
                                        <label for="Class_MLPWarmStart">Warm Start:</label>
                                        <select name="Class_MLPWarmStart" id="Class_MLPWarmStart">
                                            <option value="false">False</option>
                                            <option value="true">True</option>
                                        </select>
                                        <div><br></div>
                                        <label for="Class_MLPAlpha">Alpha - Regularization (float >=0):</label>
                                        <input type="number" value=.0001 placeholder=.0001 step="any" min=0 id="Class_MLPAlpha" name="Class_MLPAlpha">
                                        <div><br></div>
                                        <label for="Class_MLPLearning_rate">Learning Rate:</label>
                                        <select name="Class_MLPLearning_rate" id="Class_MLPLearning_rate">
                                            <option value="constant">constant</option>
                                            <option value="invscaling">invscaling</option>
                                            <option value="adaptive">adaptive</option>
                                        </select>
                                        <div><br></div>
                                        <label for="Class_MLPBatchSize">Batch Size (integer >=1 or 'auto'):</label>
                                        <input type="text" value=200 placeholder="200" id="Class_MLPBatchSize" name="Class_MLPBatchSize">
                                        <div><br></div>
                                        <label for="Class_MLPLearningRateInit">Learning Rate Init (float >0):</label>
                                        <input type="number" value=.001 placeholder=".001" min="0.0000001" step="any" id="Class_MLPLearningRateInit" name="Class_MLPLearningRateInit">
                                        <div><br></div>
                                        <label for="Class_MLPPowerT">Power T (float):</label>
                                        <input type="number" value=.05 placeholder=.5 step="any" id="Class_MLPPowerT" name="Class_MLPPowerT">
                                        <div><br></div>
                                        <label for="Class_MLPMaxIter">Max Iterations (integer >=1):</label>
                                        <input type="number" value=200 placeholder=200 step="1" min="1" id="Class_MLPMaxIter" name="Class_MLPMaxIter">
                                        <div><br></div>
                                        <label for="Class_MLPTol">Tol (float):</label>
                                        <input type="number" value=.0001 placeholder=.0001 step="any" id="Class_MLPTol" name="Class_MLPTol">
                                        <div><br></div>
                                        <label for="Class_MLPMomentum">Momentum (float [0, 1) ):</label>
                                        <input type="number" value=.09 placeholder=.9 min=0 max=.9999999 step="any" id="Class_MLPMomentum" name="Class_MLPMomentum">
                                        <div><br></div>
                                        <label for="Class_MLPValidationFraction">Validation Fraction (float [0, 1) ):</label>
                                        <input type="number" value=.01 placeholder=.1 min=0 max=.9999999 step="any" id="Class_MLPValidationFraction" name="Class_MLPValidationFraction">
                                        <div><br></div>
                                        <label for="Class_MLPBeta1"> Beta 1 (float [0, 1) ):</label>
                                        <input type="number" value=.09 placeholder=.9 min=0 max=.9999999  step="any" id="Class_MLPBeta1" name="Class_MLPBeta1">
                                        <div><br></div>
                                        <label for="Class_MLPBeta2">Beta 2 (float [0, 1) ):</label>
                                        <input type="number" value=.999 placeholder=.999 min=0 max=.9999999 step="any" id="Class_MLPBeta2" name="Class_MLPBeta2">
                                        <div><br></div>
                                        <label for="Class_MLPEpsilon">Epsilon (float >0):</label>
                                        <input type="number" value=.00000001 min=.00000000001 placeholder=.00000001 step="any" id="Class_MLPEpsilon" name="Class_MLPEpsilon">
                                    </div>
                                </div>
                            </div>
                            
                            <div id="advancedRF_classifierFields" class="hidden">
                                <h3>Random Forest Classifier Model Settings</h3>
                                <p class="field-note" style="margin-bottom: 12px; margin-top: 0;">Ensemble classifier using multiple decision trees. More trees generally improve accuracy but increase computation time.</p>
                                <label for="advancedClass_RFn_estmators">N Estimators - # of trees (integer >=1):</label>
                                <input type="number" step=1 value=100 placeholder="100" min=1 id="advancedClass_RFn_estmators" name="advancedClass_RFn_estmators">
                                <div class="nonreqHyperparams">
                                    <div class="toggle-container">
                                        <span><h3>Edit Non-Essential Hyperparameters?</h3></span>
                                        <label class="switch">
                                            <input type="checkbox" id="advancedNonreqRFClassifierSlider">
                                            <span class="slider"></span>
                                        </label>
                                    </div>
                                    <div id="advancedNonreqRFClassifierParams" class="hidden">
                                        <label for="Class_RFBoostrap">Bootstrap:</label>
                                        <select name="Class_RFBoostrap" id="Class_RFBoostrap">
                                            <option value="true">True</option>
                                            <option value="false">False</option>
                                        </select>
                                        <label for="Class_RFoobScore">oob Score:</label>
                                        <select name="Class_RFoobScore" id="Class_RFoobScore">
                                            <option value="false">False</option>
                                            <option value="true">True</option>
                                        </select>
                                        <label for="Class_RFWarmStart">Warm Start:</label>
                                        <select name="Class_RFWarmStart" id="Class_RFWarmStart">
                                            <option value="false">False</option>
                                            <option value="true">True</option>
                                        </select>
                                        <div><br></div>
                                        <label for="Class_RFmin_weight_fraction_leaf">Min Weight Fraction Leaf (float [0.0, 0.5]):</label>
                                        <input type="number" value=0 placeholder=0 min="0" max=".5" step="any" id="Class_RFmin_weight_fraction_leaf" name="Class_RFmin_weight_fraction_leaf">
                                        <div><br></div>
                                        <label for="Class_RFMaxLeafNodes">Max Leaf Nodes (an integer or leave blank for None):</label>
                                        <input type="number" step="1" id="Class_RFMaxLeafNodes" name="Class_RFMaxLeafNodes">
                                        <div><br></div>
                                        <label for="Class_RFMinImpurityDecrease">Min Impurity Decrease (float):</label>
                                        <input type="number" value=0 placeholder=0 step="any" id="Class_RFMinImpurityDecrease" name="Class_RFMinImpurityDecrease">
                                        <div><br></div>
                                        <label for="Class_RFNJobs">N Jobs (an integer or leave blank for None):</label>
                                        <input type="number" step="1" id="Class_RFNJobs" name="Class_RFNJobs">
                                        <div><br></div>
                                        <label for="Class_RFVerbose">Verbose (int):</label>
                                        <input type="number" value=0 placeholder=0 step="1" id="Class_RFVerbose" name="Class_RFVerbose">
                                        <div><br></div>
                                        <label for="Class_RFMax_depth">Max Depth - Tree depth (an integer or leave blank for None):</label>
                                        <input type="number" step=1 id="Class_RFMax_depth" name="Class_RFMax_depth">
                                        <div><br></div>
                                        <label for="Class_min_samples_split">Min Samples Split - Min samples per split (integer or float):</label>
                                        <input type="number" value=2 placeholder="2" id="Class_min_samples_split" name="Class_min_samples_split">
                                        <div><br></div>
                                        <label for="Class_min_samples_leaf">Min Samples Leaf - Min samples per leaf (integer or float):</label>
                                        <input type="number" value=1 placeholder=1 id="Class_min_samples_leaf" name="Class_min_samples_leaf">
                                    </div>
                                </div>
                            </div>
                            
                            <div id="advancedSVC_classifierFields" class="hidden">
                                <h3>SVC Classifier Model Settings</h3>
                                <p class="field-note" style="margin-bottom: 12px; margin-top: 0;">Support Vector Classifier finds optimal decision boundaries between classes. Effective for complex classification problems.</p>
                                <div class="nonreqHyperparams">
                                    <div><br></div>
                                    <div class="toggle-container">
                                        <span><h3>Edit Non-Essential Hyperparameters?</h3></span>
                                        <label class="switch">
                                            <input type="checkbox" id="advancedNonreqSVCClassifierSlider">
                                            <span class="slider"></span>
                                        </label>
                                    </div>
                                    <div id="advancedNonreqSVCClassifierParams" class="hidden">
                                        <label for="SVCshrinking">Shrinking:</label>
                                        <select name="SVCshrinking" id="SVCshrinking">
                                            <option value="true">True</option>
                                            <option value="false">False</option>
                                        </select>
                                        <label for="SVCprobability">Probability:</label>
                                        <select name="SVCprobability" id="SVCprobability">
                                            <option value="false">False</option>
                                            <option value="true">True</option>
                                        </select>
                                        <br>
                                        <label for="SVCBreakTies">Break Ties:</label>
                                        <select name="SVCBreakTies" id="SVCBreakTies">
                                            <option value="false">False</option>
                                            <option value="true">True</option>
                                        </select>
                                        <label for="SVCverbose">Verbose:</label>
                                        <select name="SVCverbose" id="SVCverbose">
                                            <option value="false">False</option>
                                            <option value="true">True</option>
                                        </select>
                                        <div><br></div>
                                        <label for="SVCcoef0">coef0 (float):</label>
                                        <input type="number" step="any" value=0 placeholder="0" id="SVCcoef0" name="SVCcoef0">
                                        <div><br></div>
                                        <label for="SVCtol">tol(float):</label>
                                        <input type="number" step="any" value=.001 placeholder=.001 id="SVCtol" name="SVCtol">
                                        <div><br></div>
                                        <label for="SVCCacheSize">Cache Size (float):</label>
                                        <input type="number" step="any" value=200 placeholder=200 id="SVCCacheSize" name="SVCCacheSize">
                                        <div><br></div>
                                        <label for="SVCClassWeight">Class Weight (enter a dictionary or 'balanced')</label>
                                        <input type="text" id="SVCClassWeight" name="SVCClassWeight">
                                        <div><br></div>
                                        <label for="SVCmaxIter">Max Iterations (int):</label>
                                        <input type="number" step=1 value=-1 placeholder=-1 id="SVCmaxIter" name="SVCmaxIter">
                                        <div><br></div>
                                        <label for="SVCdecisionFunctionShape">Decision Function Shape:</label>
                                        <select name="SVCdecisionFunctionShape" id="SVCdecisionFunctionShape">
                                            <option value="ovr">ovr</option>
                                            <option value="ovo">ovo</option>
                                        </select>
                                    </div>
                                </div>
                            </div>
                            
                            <!-- Cluster Non-Essential Hyperparameters -->
                            <div id="advancedAgglomerativeFields" class="hidden">
                                <div class="nonreqHyperparams">
                                    <div><br></div>
                                    <div class="toggle-container">
                                        <span><h3>Edit Non-Essential Hyperparameters?</h3></span>
                                        <label class="switch">
                                            <input type="checkbox" id="advancedNonreqAgglomerativeSlider">
                                            <span class="slider"></span>
                                        </label>
                                    </div>
                                    <div id="advancedNonreqAgglomerativeParams" class="hidden">
                                        <label for="Agglinkage">Linkage:</label>
                                        <select name="Agglinkage" id="Agglinkage">
                                            <option value="ward">ward</option>
                                            <option value="complete">complete</option>
                                            <option value="average">average</option>
                                            <option value="single">single</option>
                                        </select>
                                        <label for="aggcompute_distances">Compute Distances:</label>
                                        <select name="aggcompute_distances" id="aggcompute_distances">
                                            <option value="false">False</option>
                                            <option value="true">True</option>
                                        </select>
                                        <div><br></div>
                                        <label for="aggcompute_full_tree">Compute Full Tree:</label>
                                        <select name="aggcompute_full_tree" id="aggcompute_full_tree">
                                            <option value="auto">Auto</option>
                                            <option value="false">False</option>
                                            <option value="true">True</option>
                                        </select>
                                        <div><br></div>
                                        <label for="Aggdistance_threshold">Distance Threshold (float or None):</label>
                                        <input type="number" step=any id="Aggdistance_threshold" name="Aggdistance_threshold">
                                        <div><br></div>
                                        <label for="Aggmetric">Metric:</label>
                                        <input type="text" id="Aggmetric" value="euclidean" placeholder="euclidean" name="Aggmetric">
                                        <div><br></div>
                                        <label for="Aggmemory">Memory:</label>
                                        <input type="text" id="Aggmemory" name="Aggmemory">
                                        <div><br></div>
                                        <label for="Aggconnectivity">Connectivity</label>
                                        <input type="text" id="Aggconnectivity" name="Aggconnectivity">
                                    </div>
                                </div>
                            </div>
                            
                            <div id="GaussianFields" class="hidden">
                                <div class="nonreqHyperparams">
                                    <div><br></div>
                                    <div class="toggle-container">
                                        <span><h3>Edit Non-Essential Hyperparameters?</h3></span>
                                        <label class="switch">
                                            <input type="checkbox" id="nonreqGaussianSlider">
                                            <span class="slider"></span>
                                        </label>
                                    </div>
                                    <div id="nonreqGaussianParams" class="hidden">
                                        <label for="GauWarmStart">Warm Start:</label>
                                        <select name="GauWarmStart" id="GauWarmStart">
                                            <option value="false">False</option>
                                            <option value="true">True</option>
                                        </select>
                                        <label for="Gauinit_params">Init Params:</label>
                                        <select name="Gauinit_params" id="Gauinit_params">
                                            <option value="kmeans">kmeans</option>
                                            <option value="k-means++">k-means++</option>
                                            <option value="random">random</option>
                                            <option value="random_from_data">random_from_data</option>
                                        </select>
                                        <div><br></div>
                                        <label for="Gaucovariance_type">Covariance Type:</label>
                                        <select name="Gaucovariance_type" id="Gaucovariance_type">
                                            <option value="full">full</option>
                                            <option value="tied">tied</option>
                                            <option value="diag">diag</option>
                                            <option value="spherical">spherical</option>
                                        </select>
                                        <div><br></div>
                                        <label for="GauMax_iter">Max Iterations:</label>
                                        <input type="number" step=1 value=100 placeholder=100 min=1 id="GauMax_iter" name="GauMax_iter">
                                        <div><br></div>
                                        <label for="Gaun_init">N Init:</label>
                                        <input type="number" step=1 value=1 placeholder=1 min=1 id="Gaun_init" name="Gaun_init">
                                        <div><br></div>
                                        <label for="GauTol">Tol (float >0):</label>
                                        <input type="number" value=0.001 placeholder=.001 min="0.0000001" step="any" id="GauTol" name="GauTol">
                                        <div><br></div>
                                        <label for="Gaureg_covar">reg_covar:</label>
                                        <input type="number" value=0.000001 placeholder=.000001 min="0.0000001" step="any" id="Gaureg_covar" name="Gaureg_covar">
                                        <div><br></div>
                                        <label for="Gauweights_init">Weights Init:</label>
                                        <input type="text" id="Gauweights_init" name="Gauweights_init">
                                        <div><br></div>
                                        <label for="Gaumeans_init">Means Init:</label>
                                        <input type="text" id="Gaumeans_init" name="Gaumeans_init">
                                        <div><br></div>
                                        <label for="Gauprecisions_init">Precisions Init</label>
                                        <input type="text" id="Gauprecisions_init" name="Gauprecisions_init">
                                        <div><br></div>
                                        <label for="GauVerbose">Verbose:</label>
                                        <input type="number" step=1 value=0 placeholder=0 min=0 id="GauVerbose" name="GauVerbose">
                                        <div><br></div>
                                        <label for="GauVerbose_interval">Verbose Interval:</label>
                                        <input type="number" step=1 value=10 placeholder=10 min=0 id="GauVerbose_interval" name="GauVerbose_interval">
                                    </div>
                                </div>
                            </div>
                            
                            <div id="advancedKmeansFields" class="hidden">
                                <div class="nonreqHyperparams">
                                    <div><br></div>
                                    <div class="toggle-container">
                                        <span><h3>Edit Non-Essential Hyperparameters?</h3></span>
                                        <label class="switch">
                                            <input type="checkbox" id="advancedNonreqKmeansSlider">
                                            <span class="slider"></span>
                                        </label>
                                    </div>
                                    <div id="advancedNonreqKmeansParams" class="hidden">
                                        <label for="Kmeansalgorithm">Algorithm:</label>
                                        <select name="Kmeansalgorithm" id="Kmeansalgorithm">
                                            <option value="lloyd">lloyd</option>
                                            <option value="elkan">elkan</option>
                                            <option value="auto">auto</option>
                                            <option value="full">full</option>
                                        </select>
                                        <label for="kmeansCopyX">Copy X:</label>
                                        <select name="kmeansCopyX" id="kmeansCopyX">
                                            <option value="true">True</option>
                                            <option value="false">False</option>
                                        </select>
                                        <div><br></div>
                                        <label for="kmeansmax_iter">Max Iterations:</label>
                                        <input type="number" value=300 placeholder=300 step=1 id="kmeansmax_iter" name="kmeansmax_iter">
                                        <div><br></div>
                                        <label for="kmeansverbose">Verbose:</label>
                                        <input type="number" value=0 placeholder=0 step=1 id="kmeansverbose" name="kmeansverbose">
                                        <div><br></div>
                                        <label for="kmeanstol">tol:</label>
                                        <input type="number" value=0.0001 placeholder=0.0001 step=any id="kmeanstol" name="kmeanstol">
                                        <div><br></div>
                                        <label for="kmeansInit">Init:</label>
                                        <input type="text" value=k-means++ placeholder="k-means++" id="kmeansInit" name="kmeansInit">
                                        <div><br></div>
                                        <label for="kmeansn_init">n_init (int or 'auto'):</label>
                                        <input type="text" value=auto placeholder="auto" id="kmeansn_init" name="kmeansn_init">
                                    </div>
                                </div>
                            </div>
                            
                            <!-- Additional Regression Model Hyperparameters -->
                            <!-- AdaBoost Regressor -->
                            <div id="advancedAdaBoostFields" class="hidden">
                                <h3>AdaBoost Regressor Settings</h3>
                                <p class="field-note" style="margin-bottom: 12px; margin-top: 0;">AdaBoost combines multiple weak learners sequentially, with each focusing on previous errors.</p>
                                <label for="advancedAdaBoostNEstimators">N Estimators - Number of estimators (integer >=1):</label>
                                <input type="number" step=1 value=50 placeholder=50 min=1 id="advancedAdaBoostNEstimators" name="advancedAdaBoostNEstimators">
                                <div><br></div>
                                <label for="advancedAdaBoostLearningRate">Learning Rate (float >0):</label>
                                <input type="number" value=1.0 placeholder=1.0 min="0.0000001" step="any" id="advancedAdaBoostLearningRate" name="advancedAdaBoostLearningRate">
                                <div class="nonreqHyperparams">
                                    <div><br></div>
                                    <div class="toggle-container">
                                        <span><h3>Edit Non-Essential Hyperparameters?</h3></span>
                                        <label class="switch">
                                            <input type="checkbox" id="advancedNonreqAdaBoostSlider">
                                            <span class="slider"></span>
                                        </label>
                                    </div>
                                    <div id="advancedNonreqAdaBoostParams" class="hidden">
                                        <label for="AdaBoostLoss">Loss Function:</label>
                                        <select name="AdaBoostLoss" id="AdaBoostLoss">
                                            <option value="linear">Linear</option>
                                            <option value="square">Square</option>
                                            <option value="exponential">Exponential</option>
                                        </select>
                                    </div>
                                </div>
                            </div>
                            
                            <!-- Bagging Regressor -->
                            <div id="advancedBaggingFields" class="hidden">
                                <h3>Bagging Regressor Settings</h3>
                                <p class="field-note" style="margin-bottom: 12px; margin-top: 0;">Bagging trains multiple estimators on random subsets of data and averages predictions.</p>
                                <label for="advancedBaggingNEstimators">N Estimators - Number of base estimators (integer >=1):</label>
                                <input type="number" step=1 value=10 placeholder=10 min=1 id="advancedBaggingNEstimators" name="advancedBaggingNEstimators">
                                <div class="nonreqHyperparams">
                                    <div><br></div>
                                    <div class="toggle-container">
                                        <span><h3>Edit Non-Essential Hyperparameters?</h3></span>
                                        <label class="switch">
                                            <input type="checkbox" id="advancedNonreqBaggingSlider">
                                            <span class="slider"></span>
                                        </label>
                                    </div>
                                    <div id="advancedNonreqBaggingParams" class="hidden">
                                        <label for="BaggingMaxSamples">Max Samples (float (0,1] or int):</label>
                                        <input type="number" value=1.0 placeholder=1.0 min="0.0000001" max=1 step="any" id="BaggingMaxSamples" name="BaggingMaxSamples">
                                        <div><br></div>
                                        <label for="BaggingMaxFeatures">Max Features (float (0,1] or int):</label>
                                        <input type="number" value=1.0 placeholder=1.0 min="0.0000001" max=1 step="any" id="BaggingMaxFeatures" name="BaggingMaxFeatures">
                                        <div><br></div>
                                        <label for="BaggingBootstrap">Bootstrap:</label>
                                        <select name="BaggingBootstrap" id="BaggingBootstrap">
                                            <option value="true">True</option>
                                            <option value="false">False</option>
                                        </select>
                                        <label for="BaggingBootstrapFeatures">Bootstrap Features:</label>
                                        <select name="BaggingBootstrapFeatures" id="BaggingBootstrapFeatures">
                                            <option value="false">False</option>
                                            <option value="true">True</option>
                                        </select>
                                        <div><br></div>
                                        <label for="BaggingOobScore">OOB Score:</label>
                                        <select name="BaggingOobScore" id="BaggingOobScore">
                                            <option value="false">False</option>
                                            <option value="true">True</option>
                                        </select>
                                        <label for="BaggingWarmStart">Warm Start:</label>
                                        <select name="BaggingWarmStart" id="BaggingWarmStart">
                                            <option value="false">False</option>
                                            <option value="true">True</option>
                                        </select>
                                        <div><br></div>
                                        <label for="BaggingNJobs">N Jobs (int or None):</label>
                                        <input type="number" step=1 id="BaggingNJobs" name="BaggingNJobs">
                                        <div><br></div>
                                        <label for="BaggingVerbose">Verbose (int):</label>
                                        <input type="number" step=1 value=0 placeholder=0 id="BaggingVerbose" name="BaggingVerbose">
                                    </div>
                                </div>
                            </div>
                            
                            <!-- Decision Tree Regressor -->
                            <div id="advancedDecisionTreeFields" class="hidden">
                                <h3>Decision Tree Regressor Settings</h3>
                                <p class="field-note" style="margin-bottom: 12px; margin-top: 0;">Decision trees split data recursively based on feature values to make predictions.</p>
                                <div class="nonreqHyperparams">
                                    <div class="toggle-container">
                                        <span><h3>Edit Hyperparameters?</h3></span>
                                        <label class="switch">
                                            <input type="checkbox" id="advancedNonreqDecisionTreeSlider">
                                            <span class="slider"></span>
                                        </label>
                                    </div>
                                    <div id="advancedNonreqDecisionTreeParams" class="hidden">
                                        <label for="DecisionTreeCriterion">Criterion:</label>
                                        <select name="DecisionTreeCriterion" id="DecisionTreeCriterion">
                                            <option value="squared_error">Squared Error</option>
                                            <option value="friedman_mse">Friedman MSE</option>
                                            <option value="absolute_error">Absolute Error</option>
                                            <option value="poisson">Poisson</option>
                                        </select>
                                        <label for="DecisionTreeSplitter">Splitter:</label>
                                        <select name="DecisionTreeSplitter" id="DecisionTreeSplitter">
                                            <option value="best">Best</option>
                                            <option value="random">Random</option>
                                        </select>
                                        <div><br></div>
                                        <label for="DecisionTreeMaxDepth">Max Depth (int or None):</label>
                                        <input type="number" step=1 id="DecisionTreeMaxDepth" name="DecisionTreeMaxDepth">
                                        <div><br></div>
                                        <label for="DecisionTreeMinSamplesSplit">Min Samples Split (int or float):</label>
                                        <input type="number" value=2 placeholder=2 step="any" min=1 id="DecisionTreeMinSamplesSplit" name="DecisionTreeMinSamplesSplit">
                                        <div><br></div>
                                        <label for="DecisionTreeMinSamplesLeaf">Min Samples Leaf (int or float):</label>
                                        <input type="number" value=1 placeholder=1 step="any" min=1 id="DecisionTreeMinSamplesLeaf" name="DecisionTreeMinSamplesLeaf">
                                        <div><br></div>
                                        <label for="DecisionTreeMinWeightFractionLeaf">Min Weight Fraction Leaf (float [0.0, 0.5]):</label>
                                        <input type="number" value=0 placeholder=0 step="any" min=0 max=.5 id="DecisionTreeMinWeightFractionLeaf" name="DecisionTreeMinWeightFractionLeaf">
                                        <div><br></div>
                                        <label for="DecisionTreeMaxFeatures">Max Features (int, float, or string):</label>
                                        <input type="text" id="DecisionTreeMaxFeatures" name="DecisionTreeMaxFeatures">
                                        <div><br></div>
                                        <label for="DecisionTreeMaxLeafNodes">Max Leaf Nodes (int or None):</label>
                                        <input type="number" step=1 id="DecisionTreeMaxLeafNodes" name="DecisionTreeMaxLeafNodes">
                                        <div><br></div>
                                        <label for="DecisionTreeMinImpurityDecrease">Min Impurity Decrease (float >=0):</label>
                                        <input type="number" value=0 placeholder=0 step="any" min=0 id="DecisionTreeMinImpurityDecrease" name="DecisionTreeMinImpurityDecrease">
                                        <div><br></div>
                                        <label for="DecisionTreeCcpAlpha">CCP Alpha (float >=0):</label>
                                        <input type="number" value=0 placeholder=0 step="any" min=0 id="DecisionTreeCcpAlpha" name="DecisionTreeCcpAlpha">
                                    </div>
                                </div>
                            </div>
                            
                            <!-- SGD Regressor -->
                            <div id="advancedSGDFields" class="hidden">
                                <h3>SGD Regressor Settings</h3>
                                <p class="field-note" style="margin-bottom: 12px; margin-top: 0;">Stochastic Gradient Descent for linear regression. Fast and scalable for large datasets.</p>
                                <label for="advancedSGDLoss">Loss Function:</label>
                                <select name="advancedSGDLoss" id="advancedSGDLoss">
                                    <option value="squared_error">Squared Error</option>
                                    <option value="huber">Huber</option>
                                    <option value="epsilon_insensitive">Epsilon Insensitive</option>
                                    <option value="squared_epsilon_insensitive">Squared Epsilon Insensitive</option>
                                </select>
                                <div><br></div>
                                <label for="advancedSGDPenalty">Penalty:</label>
                                <select name="advancedSGDPenalty" id="advancedSGDPenalty">
                                    <option value="l2">L2</option>
                                    <option value="l1">L1</option>
                                    <option value="elasticnet">Elastic Net</option>
                                    <option value="None">None</option>
                                </select>
                                <div><br></div>
                                <label for="advancedSGDAlpha">Alpha - Regularization strength (float >0):</label>
                                <input type="number" value=0.0001 placeholder=0.0001 min="0.0000001" step="any" id="advancedSGDAlpha" name="advancedSGDAlpha">
                                <div><br></div>
                                <label for="advancedSGDL1Ratio">L1 Ratio - Elastic Net mixing (float [0,1]):</label>
                                <input type="number" value=0.15 placeholder=0.15 min=0 max=1 step="any" id="advancedSGDL1Ratio" name="advancedSGDL1Ratio">
                                <div class="nonreqHyperparams">
                                    <div><br></div>
                                    <div class="toggle-container">
                                        <span><h3>Edit Non-Essential Hyperparameters?</h3></span>
                                        <label class="switch">
                                            <input type="checkbox" id="advancedNonreqSGDSlider">
                                            <span class="slider"></span>
                                        </label>
                                    </div>
                                    <div id="advancedNonreqSGDParams" class="hidden">
                                        <label for="SGDFitIntercept">Fit Intercept:</label>
                                        <select name="SGDFitIntercept" id="SGDFitIntercept">
                                            <option value="true">True</option>
                                            <option value="false">False</option>
                                        </select>
                                        <div><br></div>
                                        <label for="SGDMaxIter">Max Iterations (int >=1):</label>
                                        <input type="number" step=1 value=1000 placeholder=1000 min=1 id="SGDMaxIter" name="SGDMaxIter">
                                        <div><br></div>
                                        <label for="SGDTol">Tol (float >0):</label>
                                        <input type="number" value=0.001 placeholder=0.001 min="0.0000001" step="any" id="SGDTol" name="SGDTol">
                                        <div><br></div>
                                        <label for="SGDShuffle">Shuffle:</label>
                                        <select name="SGDShuffle" id="SGDShuffle">
                                            <option value="true">True</option>
                                            <option value="false">False</option>
                                        </select>
                                        <div><br></div>
                                        <label for="SGDVerbose">Verbose (int):</label>
                                        <input type="number" step=1 value=0 placeholder=0 id="SGDVerbose" name="SGDVerbose">
                                        <div><br></div>
                                        <label for="SGDEpsilon">Epsilon (float >0):</label>
                                        <input type="number" value=0.1 placeholder=0.1 min="0.0000001" step="any" id="SGDEpsilon" name="SGDEpsilon">
                                        <div><br></div>
                                        <label for="SGDLearningRate">Learning Rate Schedule:</label>
                                        <select name="SGDLearningRate" id="SGDLearningRate">
                                            <option value="invscaling">Inverse Scaling</option>
                                            <option value="constant">Constant</option>
                                            <option value="optimal">Optimal</option>
                                            <option value="adaptive">Adaptive</option>
                                        </select>
                                        <div><br></div>
                                        <label for="SGDEta0">Eta0 - Initial learning rate (float >0):</label>
                                        <input type="number" value=0.01 placeholder=0.01 min="0.0000001" step="any" id="SGDEta0" name="SGDEta0">
                                        <div><br></div>
                                        <label for="SGDPowerT">Power T (float):</label>
                                        <input type="number" value=0.25 placeholder=0.25 step="any" id="SGDPowerT" name="SGDPowerT">
                                        <div><br></div>
                                        <label for="SGDEarlyStopping">Early Stopping:</label>
                                        <select name="SGDEarlyStopping" id="SGDEarlyStopping">
                                            <option value="false">False</option>
                                            <option value="true">True</option>
                                        </select>
                                        <div><br></div>
                                        <label for="SGDValidationFraction">Validation Fraction (float (0,1)):</label>
                                        <input type="number" value=0.1 placeholder=0.1 min="0.0000001" max=0.999999 step="any" id="SGDValidationFraction" name="SGDValidationFraction">
                                        <div><br></div>
                                        <label for="SGDNIterNoChange">N Iter No Change (int >=1):</label>
                                        <input type="number" step=1 value=5 placeholder=5 min=1 id="SGDNIterNoChange" name="SGDNIterNoChange">
                                        <div><br></div>
                                        <label for="SGDWarmStart">Warm Start:</label>
                                        <select name="SGDWarmStart" id="SGDWarmStart">
                                            <option value="false">False</option>
                                            <option value="true">True</option>
                                        </select>
                                        <div><br></div>
                                        <label for="SGDAverage">Average:</label>
                                        <select name="SGDAverage" id="SGDAverage">
                                            <option value="false">False</option>
                                            <option value="true">True</option>
                                        </select>
                                        <div><br></div>
                                    </div>
                                </div>
                            </div>
                            
                            <!-- Histogram Gradient Boosting Regressor -->
                            <div id="advancedHistGradientBoostingFields" class="hidden">
                                <h3>Histogram Gradient Boosting Regressor Settings</h3>
                                <p class="field-note" style="margin-bottom: 12px; margin-top: 0;">Fast gradient boosting using histogram-based tree building. More efficient than standard gradient boosting.</p>
                                <label for="advancedHistGBLearningRate">Learning Rate (float >0):</label>
                                <input type="number" value=0.1 placeholder=0.1 min="0.0000001" step="any" id="advancedHistGBLearningRate" name="advancedHistGBLearningRate">
                                <div><br></div>
                                <label for="advancedHistGBMaxIter">Max Iterations (int >=1):</label>
                                <input type="number" step=1 value=100 placeholder=100 min=1 id="advancedHistGBMaxIter" name="advancedHistGBMaxIter">
                                <div><br></div>
                                <label for="advancedHistGBMaxLeafNodes">Max Leaf Nodes (int >=2):</label>
                                <input type="number" step=1 value=31 placeholder=31 min=2 id="advancedHistGBMaxLeafNodes" name="advancedHistGBMaxLeafNodes">
                                <div class="nonreqHyperparams">
                                    <div><br></div>
                                    <div class="toggle-container">
                                        <span><h3>Edit Non-Essential Hyperparameters?</h3></span>
                                        <label class="switch">
                                            <input type="checkbox" id="advancedNonreqHistGBSlider">
                                            <span class="slider"></span>
                                        </label>
                                    </div>
                                    <div id="advancedNonreqHistGBParams" class="hidden">
                                        <label for="HistGBLoss">Loss Function:</label>
                                        <select name="HistGBLoss" id="HistGBLoss">
                                            <option value="squared_error">Squared Error</option>
                                            <option value="absolute_error">Absolute Error</option>
                                            <option value="poisson">Poisson</option>
                                            <option value="quantile">Quantile</option>
                                        </select>
                                        <div><br></div>
                                        <label for="HistGBMaxDepth">Max Depth (int or None):</label>
                                        <input type="number" step=1 id="HistGBMaxDepth" name="HistGBMaxDepth">
                                        <div><br></div>
                                        <label for="HistGBMinSamplesLeaf">Min Samples Leaf (int >=1):</label>
                                        <input type="number" step=1 value=20 placeholder=20 min=1 id="HistGBMinSamplesLeaf" name="HistGBMinSamplesLeaf">
                                        <div><br></div>
                                        <label for="HistGBL2Regularization">L2 Regularization (float >=0):</label>
                                        <input type="number" value=0 placeholder=0 step="any" min=0 id="HistGBL2Regularization" name="HistGBL2Regularization">
                                        <div><br></div>
                                        <label for="HistGBMaxBins">Max Bins (int >=2):</label>
                                        <input type="number" step=1 value=255 placeholder=255 min=2 id="HistGBMaxBins" name="HistGBMaxBins">
                                        <div><br></div>
                                        <label for="HistGBWarmStart">Warm Start:</label>
                                        <select name="HistGBWarmStart" id="HistGBWarmStart">
                                            <option value="false">False</option>
                                            <option value="true">True</option>
                                        </select>
                                        <div><br></div>
                                        <label for="HistGBEarlyStopping">Early Stopping:</label>
                                        <select name="HistGBEarlyStopping" id="HistGBEarlyStopping">
                                            <option value="auto">Auto</option>
                                            <option value="false">False</option>
                                            <option value="true">True</option>
                                        </select>
                                        <div><br></div>
                                        <label for="HistGBScoring">Scoring:</label>
                                        <input type="text" value="loss" placeholder="loss" id="HistGBScoring" name="HistGBScoring">
                                        <div><br></div>
                                        <label for="HistGBValidationFraction">Validation Fraction (float (0,1)):</label>
                                        <input type="number" value=0.1 placeholder=0.1 min="0.0000001" max=0.999999 step="any" id="HistGBValidationFraction" name="HistGBValidationFraction">
                                        <div><br></div>
                                        <label for="HistGBNIterNoChange">N Iter No Change (int >=1):</label>
                                        <input type="number" step=1 value=10 placeholder=10 min=1 id="HistGBNIterNoChange" name="HistGBNIterNoChange">
                                        <div><br></div>
                                        <label for="HistGBTol">Tol (float >0):</label>
                                        <input type="number" value=0.0000001 placeholder=0.0000001 min="0.0000001" step="any" id="HistGBTol" name="HistGBTol">
                                        <div><br></div>
                                        <label for="HistGBVerbose">Verbose (int):</label>
                                        <input type="number" step=1 value=0 placeholder=0 id="HistGBVerbose" name="HistGBVerbose">
                                    </div>
                                </div>
                            </div>
                            
                            <!-- Huber Regressor -->
                            <div id="advancedHuberFields" class="hidden">
                                <h3>Huber Regressor Settings</h3>
                                <p class="field-note" style="margin-bottom: 12px; margin-top: 0;">Robust regression that is less sensitive to outliers than standard linear regression.</p>
                                <label for="advancedHuberEpsilon">Epsilon - Outlier threshold (float >1.0):</label>
                                <input type="number" value=1.35 placeholder=1.35 min="1.0000001" step="any" id="advancedHuberEpsilon" name="advancedHuberEpsilon">
                                <div><br></div>
                                <label for="advancedHuberAlpha">Alpha - Regularization strength (float >=0):</label>
                                <input type="number" value=0.0001 placeholder=0.0001 min=0 step="any" id="advancedHuberAlpha" name="advancedHuberAlpha">
                                <div class="nonreqHyperparams">
                                    <div><br></div>
                                    <div class="toggle-container">
                                        <span><h3>Edit Non-Essential Hyperparameters?</h3></span>
                                        <label class="switch">
                                            <input type="checkbox" id="advancedNonreqHuberSlider">
                                            <span class="slider"></span>
                                        </label>
                                    </div>
                                    <div id="advancedNonreqHuberParams" class="hidden">
                                        <label for="HuberMaxIter">Max Iterations (int >=1):</label>
                                        <input type="number" step=1 value=100 placeholder=100 min=1 id="HuberMaxIter" name="HuberMaxIter">
                                        <div><br></div>
                                        <label for="HuberWarmStart">Warm Start:</label>
                                        <select name="HuberWarmStart" id="HuberWarmStart">
                                            <option value="false">False</option>
                                            <option value="true">True</option>
                                        </select>
                                        <div><br></div>
                                        <label for="HuberFitIntercept">Fit Intercept:</label>
                                        <select name="HuberFitIntercept" id="HuberFitIntercept">
                                            <option value="true">True</option>
                                            <option value="false">False</option>
                                        </select>
                                        <div><br></div>
                                        <label for="HuberTol">Tol (float >0):</label>
                                        <input type="number" value=0.00001 placeholder=0.00001 min="0.0000001" step="any" id="HuberTol" name="HuberTol">
                                    </div>
                                </div>
                            </div>
                            
                            <!-- Quantile Regressor -->
                            <div id="advancedQuantileFields" class="hidden">
                                <h3>Quantile Regressor Settings</h3>
                                <p class="field-note" style="margin-bottom: 12px; margin-top: 0;">Predicts conditional quantiles rather than the mean. Useful for understanding prediction uncertainty.</p>
                                <label for="advancedQuantileQuantile">Quantile (float (0,1)):</label>
                                <input type="number" value=0.5 placeholder=0.5 min="0.0000001" max=0.999999 step="any" id="advancedQuantileQuantile" name="advancedQuantileQuantile">
                                <div><br></div>
                                <label for="advancedQuantileAlpha">Alpha - Regularization strength (float >=0):</label>
                                <input type="number" value=1.0 placeholder=1.0 min=0 step="any" id="advancedQuantileAlpha" name="advancedQuantileAlpha">
                                <div class="nonreqHyperparams">
                                    <div><br></div>
                                    <div class="toggle-container">
                                        <span><h3>Edit Non-Essential Hyperparameters?</h3></span>
                                        <label class="switch">
                                            <input type="checkbox" id="advancedNonreqQuantileSlider">
                                            <span class="slider"></span>
                                        </label>
                                    </div>
                                    <div id="advancedNonreqQuantileParams" class="hidden">
                                        <label for="QuantileFitIntercept">Fit Intercept:</label>
                                        <select name="QuantileFitIntercept" id="QuantileFitIntercept">
                                            <option value="true">True</option>
                                            <option value="false">False</option>
                                        </select>
                                        <div><br></div>
                                        <label for="QuantileSolver">Solver:</label>
                                        <select name="QuantileSolver" id="QuantileSolver">
                                            <option value="highs">Highs</option>
                                            <option value="highs-ipm">Highs IPM</option>
                                            <option value="highs-ds">Highs DS</option>
                                            <option value="interior-point">Interior Point</option>
                                            <option value="revised simplex">Revised Simplex</option>
                                        </select>
                                    </div>
                                </div>
                            </div>
                            
                            <!-- Linear SVR -->
                            <div id="advancedLinearSVRFields" class="hidden">
                                <h3>Linear SVR Settings</h3>
                                <p class="field-note" style="margin-bottom: 12px; margin-top: 0;">Linear Support Vector Regression. Fast for large datasets.</p>
                                <label for="advancedLinearSVRC">C - Regularization parameter (float >0):</label>
                                <input type="number" value=1.0 placeholder=1.0 min="0.0000001" step="any" id="advancedLinearSVRC" name="advancedLinearSVRC">
                                <div><br></div>
                                <label for="advancedLinearSVREpsilon">Epsilon - Margin tolerance (float >=0):</label>
                                <input type="number" value=0.0 placeholder=0.0 min=0 step="any" id="advancedLinearSVREpsilon" name="advancedLinearSVREpsilon">
                                <div class="nonreqHyperparams">
                                    <div><br></div>
                                    <div class="toggle-container">
                                        <span><h3>Edit Non-Essential Hyperparameters?</h3></span>
                                        <label class="switch">
                                            <input type="checkbox" id="advancedNonreqLinearSVRSlider">
                                            <span class="slider"></span>
                                        </label>
                                    </div>
                                    <div id="advancedNonreqLinearSVRParams" class="hidden">
                                        <label for="LinearSVRLoss">Loss Function:</label>
                                        <select name="LinearSVRLoss" id="LinearSVRLoss">
                                            <option value="epsilon_insensitive">Epsilon Insensitive</option>
                                            <option value="squared_epsilon_insensitive">Squared Epsilon Insensitive</option>
                                        </select>
                                        <div><br></div>
                                        <label for="LinearSVRTol">Tol (float >0):</label>
                                        <input type="number" value=0.0001 placeholder=0.0001 min="0.0000001" step="any" id="LinearSVRTol" name="LinearSVRTol">
                                        <div><br></div>
                                        <label for="LinearSVRFitIntercept">Fit Intercept:</label>
                                        <select name="LinearSVRFitIntercept" id="LinearSVRFitIntercept">
                                            <option value="true">True</option>
                                            <option value="false">False</option>
                                        </select>
                                        <div><br></div>
                                        <label for="LinearSVRInterceptScaling">Intercept Scaling (float >0):</label>
                                        <input type="number" value=1.0 placeholder=1.0 min="0.0000001" step="any" id="LinearSVRInterceptScaling" name="LinearSVRInterceptScaling">
                                        <div><br></div>
                                        <label for="LinearSVRDual">Dual:</label>
                                        <select name="LinearSVRDual" id="LinearSVRDual">
                                            <option value="true">True</option>
                                            <option value="false">False</option>
                                        </select>
                                        <div><br></div>
                                        <label for="LinearSVRVerbose">Verbose (int):</label>
                                        <input type="number" step=1 value=0 placeholder=0 id="LinearSVRVerbose" name="LinearSVRVerbose">
                                        <div><br></div>
                                        <label for="LinearSVRMaxIter">Max Iterations (int >=1):</label>
                                        <input type="number" step=1 value=1000 placeholder=1000 min=1 id="LinearSVRMaxIter" name="LinearSVRMaxIter">
                                    </div>
                                </div>
                            </div>
                            
                            <!-- Nu-SVR -->
                            <div id="advancedNuSVRFields" class="hidden">
                                <h3>Nu-SVR Settings</h3>
                                <p class="field-note" style="margin-bottom: 12px; margin-top: 0;">Nu-Support Vector Regression with kernel functions.</p>
                                <label for="advancedNuSVRNu">Nu - Upper bound on fraction of margin errors (float (0,1)):</label>
                                <input type="number" value=0.5 placeholder=0.5 min="0.0000001" max=0.999999 step="any" id="advancedNuSVRNu" name="advancedNuSVRNu">
                                <div><br></div>
                                <label for="advancedNuSVRC">C - Regularization parameter (float >0):</label>
                                <input type="number" value=1.0 placeholder=1.0 min="0.0000001" step="any" id="advancedNuSVRC" name="advancedNuSVRC">
                                <div><br></div>
                                <label for="advancedNuSVRKernel">Kernel:</label>
                                <select name="advancedNuSVRKernel" id="advancedNuSVRKernel">
                                    <option value="rbf">RBF</option>
                                    <option value="linear">Linear</option>
                                    <option value="poly">Polynomial</option>
                                    <option value="sigmoid">Sigmoid</option>
                                </select>
                                <div class="nonreqHyperparams">
                                    <div><br></div>
                                    <div class="toggle-container">
                                        <span><h3>Edit Non-Essential Hyperparameters?</h3></span>
                                        <label class="switch">
                                            <input type="checkbox" id="advancedNonreqNuSVRSlider">
                                            <span class="slider"></span>
                                        </label>
                                    </div>
                                    <div id="advancedNonreqNuSVRParams" class="hidden">
                                        <label for="NuSVRDegree">Degree (for poly kernel, int >=1):</label>
                                        <input type="number" step=1 value=3 placeholder=3 min=1 id="NuSVRDegree" name="NuSVRDegree">
                                        <div><br></div>
                                        <label for="NuSVRGamma">Gamma (float or 'scale' or 'auto'):</label>
                                        <input type="text" value="scale" placeholder="scale" id="NuSVRGamma" name="NuSVRGamma">
                                        <div><br></div>
                                        <label for="NuSVRCoef0">Coef0 (for poly/sigmoid kernels, float):</label>
                                        <input type="number" value=0.0 placeholder=0.0 step="any" id="NuSVRCoef0" name="NuSVRCoef0">
                                        <div><br></div>
                                        <label for="NuSVRShrinking">Shrinking:</label>
                                        <select name="NuSVRShrinking" id="NuSVRShrinking">
                                            <option value="true">True</option>
                                            <option value="false">False</option>
                                        </select>
                                        <div><br></div>
                                        <label for="NuSVRTol">Tol (float >0):</label>
                                        <input type="number" value=0.001 placeholder=0.001 min="0.0000001" step="any" id="NuSVRTol" name="NuSVRTol">
                                        <div><br></div>
                                        <label for="NuSVRCacheSize">Cache Size (float):</label>
                                        <input type="number" step="any" value=200 placeholder=200 id="NuSVRCacheSize" name="NuSVRCacheSize">
                                        <div><br></div>
                                        <label for="NuSVRVerbose">Verbose:</label>
                                        <select name="NuSVRVerbose" id="NuSVRVerbose">
                                            <option value="false">False</option>
                                            <option value="true">True</option>
                                        </select>
                                        <div><br></div>
                                        <label for="NuSVRMaxIter">Max Iterations (int, -1 for no limit):</label>
                                        <input type="number" step=1 value=-1 placeholder=-1 id="NuSVRMaxIter" name="NuSVRMaxIter">
                                    </div>
                                </div>
                            </div>
                            
                            <!-- Passive Aggressive Regressor -->
                            <div id="advancedPassiveAggressiveFields" class="hidden">
                                <h3>Passive Aggressive Regressor Settings</h3>
                                <p class="field-note" style="margin-bottom: 12px; margin-top: 0;">Online learning algorithm that updates model only when prediction error exceeds threshold.</p>
                                <label for="advancedPassiveAggressiveC">C - Regularization strength (float >0):</label>
                                <input type="number" value=1.0 placeholder=1.0 min="0.0000001" step="any" id="advancedPassiveAggressiveC" name="advancedPassiveAggressiveC">
                                <div><br></div>
                                <label for="advancedPassiveAggressiveEpsilon">Epsilon - Margin tolerance (float >0):</label>
                                <input type="number" value=0.1 placeholder=0.1 min="0.0000001" step="any" id="advancedPassiveAggressiveEpsilon" name="advancedPassiveAggressiveEpsilon">
                                <div class="nonreqHyperparams">
                                    <div><br></div>
                                    <div class="toggle-container">
                                        <span><h3>Edit Non-Essential Hyperparameters?</h3></span>
                                        <label class="switch">
                                            <input type="checkbox" id="advancedNonreqPassiveAggressiveSlider">
                                            <span class="slider"></span>
                                        </label>
                                    </div>
                                    <div id="advancedNonreqPassiveAggressiveParams" class="hidden">
                                        <label for="PassiveAggressiveLoss">Loss Function:</label>
                                        <select name="PassiveAggressiveLoss" id="PassiveAggressiveLoss">
                                            <option value="epsilon_insensitive">Epsilon Insensitive</option>
                                            <option value="squared_epsilon_insensitive">Squared Epsilon Insensitive</option>
                                        </select>
                                        <div><br></div>
                                        <label for="PassiveAggressiveFitIntercept">Fit Intercept:</label>
                                        <select name="PassiveAggressiveFitIntercept" id="PassiveAggressiveFitIntercept">
                                            <option value="true">True</option>
                                            <option value="false">False</option>
                                        </select>
                                        <div><br></div>
                                        <label for="PassiveAggressiveMaxIter">Max Iterations (int >=1):</label>
                                        <input type="number" step=1 value=1000 placeholder=1000 min=1 id="PassiveAggressiveMaxIter" name="PassiveAggressiveMaxIter">
                                        <div><br></div>
                                        <label for="PassiveAggressiveTol">Tol (float >0):</label>
                                        <input type="number" value=0.001 placeholder=0.001 min="0.0000001" step="any" id="PassiveAggressiveTol" name="PassiveAggressiveTol">
                                        <div><br></div>
                                        <label for="PassiveAggressiveShuffle">Shuffle:</label>
                                        <select name="PassiveAggressiveShuffle" id="PassiveAggressiveShuffle">
                                            <option value="true">True</option>
                                            <option value="false">False</option>
                                        </select>
                                        <div><br></div>
                                        <label for="PassiveAggressiveVerbose">Verbose (int):</label>
                                        <input type="number" step=1 value=0 placeholder=0 id="PassiveAggressiveVerbose" name="PassiveAggressiveVerbose">
                                        <div><br></div>
                                        <label for="PassiveAggressiveEarlyStopping">Early Stopping:</label>
                                        <select name="PassiveAggressiveEarlyStopping" id="PassiveAggressiveEarlyStopping">
                                            <option value="false">False</option>
                                            <option value="true">True</option>
                                        </select>
                                        <div><br></div>
                                        <label for="PassiveAggressiveValidationFraction">Validation Fraction (float (0,1)):</label>
                                        <input type="number" value=0.1 placeholder=0.1 min="0.0000001" max=0.999999 step="any" id="PassiveAggressiveValidationFraction" name="PassiveAggressiveValidationFraction">
                                        <div><br></div>
                                        <label for="PassiveAggressiveNIterNoChange">N Iter No Change (int >=1):</label>
                                        <input type="number" step=1 value=5 placeholder=5 min=1 id="PassiveAggressiveNIterNoChange" name="PassiveAggressiveNIterNoChange">
                                        <div><br></div>
                                        <label for="PassiveAggressiveWarmStart">Warm Start:</label>
                                        <select name="PassiveAggressiveWarmStart" id="PassiveAggressiveWarmStart">
                                            <option value="false">False</option>
                                            <option value="true">True</option>
                                        </select>
                                        <div><br></div>
                                        <label for="PassiveAggressiveAverage">Average:</label>
                                        <select name="PassiveAggressiveAverage" id="PassiveAggressiveAverage">
                                            <option value="false">False</option>
                                            <option value="true">True</option>
                                        </select>
                                        <div><br></div>
                                        <label for="PassiveAggressiveNJobs">N Jobs (int or None):</label>
                                        <input type="number" step=1 id="PassiveAggressiveNJobs" name="PassiveAggressiveNJobs">
                                    </div>
                                </div>
                            </div>
                            
                            <!-- RANSAC Regressor -->
                            <div id="advancedRANSACFields" class="hidden">
                                <h3>RANSAC Regressor Settings</h3>
                                <p class="field-note" style="margin-bottom: 12px; margin-top: 0;">Robust regression using RANSAC algorithm. Handles outliers well.</p>
                                <label for="advancedRANSACMaxTrials">Max Trials - Maximum iterations (int >=1):</label>
                                <input type="number" step=1 value=100 placeholder=100 min=1 id="advancedRANSACMaxTrials" name="advancedRANSACMaxTrials">
                                <div class="nonreqHyperparams">
                                    <div><br></div>
                                    <div class="toggle-container">
                                        <span><h3>Edit Non-Essential Hyperparameters?</h3></span>
                                        <label class="switch">
                                            <input type="checkbox" id="advancedNonreqRANSACSlider">
                                            <span class="slider"></span>
                                        </label>
                                    </div>
                                    <div id="advancedNonreqRANSACParams" class="hidden">
                                        <label for="RANSACMinSamples">Min Samples (int or float):</label>
                                        <input type="number" step="any" id="RANSACMinSamples" name="RANSACMinSamples">
                                        <div><br></div>
                                        <label for="RANSACResidualThreshold">Residual Threshold (float):</label>
                                        <input type="number" step="any" id="RANSACResidualThreshold" name="RANSACResidualThreshold">
                                        <div><br></div>
                                        <label for="RANSACStopNInliers">Stop N Inliers (int or inf):</label>
                                        <input type="number" step=1 id="RANSACStopNInliers" name="RANSACStopNInliers">
                                        <div><br></div>
                                        <label for="RANSACStopScore">Stop Score (float or inf):</label>
                                        <input type="number" step="any" id="RANSACStopScore" name="RANSACStopScore">
                                        <div><br></div>
                                        <label for="RANSACStopProbability">Stop Probability (float (0,1)):</label>
                                        <input type="number" value=0.99 placeholder=0.99 min="0.0000001" max=0.999999 step="any" id="RANSACStopProbability" name="RANSACStopProbability">
                                        <div><br></div>
                                        <label for="RANSACLoss">Loss Function:</label>
                                        <select name="RANSACLoss" id="RANSACLoss">
                                            <option value="absolute_error">Absolute Error</option>
                                            <option value="squared_error">Squared Error</option>
                                        </select>
                                    </div>
                                </div>
                            </div>
                            
                            <!-- Theil-Sen Regressor -->
                            <div id="advancedTheilSenFields" class="hidden">
                                <h3>Theil-Sen Regressor Settings</h3>
                                <p class="field-note" style="margin-bottom: 12px; margin-top: 0;">Robust regression using Theil-Sen estimator. Very robust to outliers.</p>
                                <div class="nonreqHyperparams">
                                    <div class="toggle-container">
                                        <span><h3>Edit Hyperparameters?</h3></span>
                                        <label class="switch">
                                            <input type="checkbox" id="advancedNonreqTheilSenSlider">
                                            <span class="slider"></span>
                                        </label>
                                    </div>
                                    <div id="advancedNonreqTheilSenParams" class="hidden">
                                        <label for="TheilSenFitIntercept">Fit Intercept:</label>
                                        <select name="TheilSenFitIntercept" id="TheilSenFitIntercept">
                                            <option value="true">True</option>
                                            <option value="false">False</option>
                                        </select>
                                        <div><br></div>
                                        <label for="TheilSenMaxSubpopulation">Max Subpopulation (int >=1):</label>
                                        <input type="number" step=1 value=10000 placeholder=10000 min=1 id="TheilSenMaxSubpopulation" name="TheilSenMaxSubpopulation">
                                        <div><br></div>
                                        <label for="TheilSenNSubsamples">N Subsamples (int or None):</label>
                                        <input type="number" step=1 id="TheilSenNSubsamples" name="TheilSenNSubsamples">
                                        <div><br></div>
                                        <label for="TheilSenMaxIter">Max Iterations (int >=1):</label>
                                        <input type="number" step=1 value=300 placeholder=300 min=1 id="TheilSenMaxIter" name="TheilSenMaxIter">
                                        <div><br></div>
                                        <label for="TheilSenTol">Tol (float >0):</label>
                                        <input type="number" value=0.001 placeholder=0.001 min="0.0000001" step="any" id="TheilSenTol" name="TheilSenTol">
                                        <div><br></div>
                                        <label for="TheilSenNJobs">N Jobs (int or None):</label>
                                        <input type="number" step=1 id="TheilSenNJobs" name="TheilSenNJobs">
                                        <div><br></div>
                                        <label for="TheilSenVerbose">Verbose:</label>
                                        <select name="TheilSenVerbose" id="TheilSenVerbose">
                                            <option value="false">False</option>
                                            <option value="true">True</option>
                                        </select>
                                    </div>
                                </div>
                            </div>
                            
                            <!-- Radius Neighbors Regressor -->
                            <div id="advancedRadiusNeighborsFields" class="hidden">
                                <h3>Radius Neighbors Regressor Settings</h3>
                                <p class="field-note" style="margin-bottom: 12px; margin-top: 0;">Predicts using neighbors within a fixed radius. Useful for sparse datasets.</p>
                                <label for="advancedRadiusNeighborsRadius">Radius - Neighbor search radius (float >0):</label>
                                <input type="number" value=1.0 placeholder=1.0 min="0.0000001" step="any" id="advancedRadiusNeighborsRadius" name="advancedRadiusNeighborsRadius">
                                <div class="nonreqHyperparams">
                                    <div><br></div>
                                    <div class="toggle-container">
                                        <span><h3>Edit Non-Essential Hyperparameters?</h3></span>
                                        <label class="switch">
                                            <input type="checkbox" id="advancedNonreqRadiusNeighborsSlider">
                                            <span class="slider"></span>
                                        </label>
                                    </div>
                                    <div id="advancedNonreqRadiusNeighborsParams" class="hidden">
                                        <label for="RadiusNeighborsWeights">Weights:</label>
                                        <select name="RadiusNeighborsWeights" id="RadiusNeighborsWeights">
                                            <option value="uniform">Uniform</option>
                                            <option value="distance">Distance</option>
                                        </select>
                                        <div><br></div>
                                        <label for="RadiusNeighborsAlgorithm">Algorithm:</label>
                                        <select name="RadiusNeighborsAlgorithm" id="RadiusNeighborsAlgorithm">
                                            <option value="auto">Auto</option>
                                            <option value="ball_tree">Ball Tree</option>
                                            <option value="kd_tree">KD Tree</option>
                                            <option value="brute">Brute Force</option>
                                        </select>
                                        <div><br></div>
                                        <label for="RadiusNeighborsLeafSize">Leaf Size (int >=1):</label>
                                        <input type="number" step=1 value=30 placeholder=30 min=1 id="RadiusNeighborsLeafSize" name="RadiusNeighborsLeafSize">
                                        <div><br></div>
                                        <label for="RadiusNeighborsP">P - Minkowski metric power (float >=1):</label>
                                        <input type="number" value=2 placeholder=2 min=1 step="any" id="RadiusNeighborsP" name="RadiusNeighborsP">
                                        <div><br></div>
                                        <label for="RadiusNeighborsMetric">Metric:</label>
                                        <input type="text" value="minkowski" placeholder="minkowski" id="RadiusNeighborsMetric" name="RadiusNeighborsMetric">
                                        <div><br></div>
                                        <label for="RadiusNeighborsNJobs">N Jobs (int or None):</label>
                                        <input type="number" step=1 id="RadiusNeighborsNJobs" name="RadiusNeighborsNJobs">
                                    </div>
                                </div>
                            </div>
                            
                            <!-- Orthogonal Matching Pursuit -->
                            <div id="advancedOMPFields" class="hidden">
                                <h3>Orthogonal Matching Pursuit Settings</h3>
                                <p class="field-note" style="margin-bottom: 12px; margin-top: 0;">Sparse regression using orthogonal matching pursuit algorithm.</p>
                                <div class="nonreqHyperparams">
                                    <div class="toggle-container">
                                        <span><h3>Edit Hyperparameters?</h3></span>
                                        <label class="switch">
                                            <input type="checkbox" id="advancedNonreqOMPSlider">
                                            <span class="slider"></span>
                                        </label>
                                    </div>
                                    <div id="advancedNonreqOMPParams" class="hidden">
                                        <label for="OMPNNonzeroCoefs">N Nonzero Coefs (int or None):</label>
                                        <input type="number" step=1 id="OMPNNonzeroCoefs" name="OMPNNonzeroCoefs">
                                        <div><br></div>
                                        <label for="OMPTol">Tol (float or None):</label>
                                        <input type="number" step="any" id="OMPTol" name="OMPTol">
                                        <div><br></div>
                                        <label for="OMPFitIntercept">Fit Intercept:</label>
                                        <select name="OMPFitIntercept" id="OMPFitIntercept">
                                            <option value="true">True</option>
                                            <option value="false">False</option>
                                        </select>
                                        <div><br></div>
                                        <label for="OMPPrecompute">Precompute:</label>
                                        <select name="OMPPrecompute" id="OMPPrecompute">
                                            <option value="auto">Auto</option>
                                            <option value="true">True</option>
                                            <option value="false">False</option>
                                        </select>
                                    </div>
                                </div>
                            </div>
                            
                            <!-- LARS -->
                            <div id="advancedLARSFields" class="hidden">
                                <h3>LARS Settings</h3>
                                <p class="field-note" style="margin-bottom: 12px; margin-top: 0;">Least Angle Regression. Good for high-dimensional data.</p>
                                <div class="nonreqHyperparams">
                                    <div class="toggle-container">
                                        <span><h3>Edit Hyperparameters?</h3></span>
                                        <label class="switch">
                                            <input type="checkbox" id="advancedNonreqLARSSlider">
                                            <span class="slider"></span>
                                        </label>
                                    </div>
                                    <div id="advancedNonreqLARSParams" class="hidden">
                                        <label for="LARSFitIntercept">Fit Intercept:</label>
                                        <select name="LARSFitIntercept" id="LARSFitIntercept">
                                            <option value="true">True</option>
                                            <option value="false">False</option>
                                        </select>
                                        <div><br></div>
                                        <label for="LARSVerbose">Verbose:</label>
                                        <select name="LARSVerbose" id="LARSVerbose">
                                            <option value="false">False</option>
                                            <option value="true">True</option>
                                        </select>
                                        <div><br></div>
                                        <label for="LARSPrecompute">Precompute:</label>
                                        <select name="LARSPrecompute" id="LARSPrecompute">
                                            <option value="auto">Auto</option>
                                            <option value="true">True</option>
                                            <option value="false">False</option>
                                        </select>
                                        <div><br></div>
                                        <label for="LARSNNonzeroCoefs">N Nonzero Coefs (int >=1):</label>
                                        <input type="number" step=1 value=500 placeholder=500 min=1 id="LARSNNonzeroCoefs" name="LARSNNonzeroCoefs">
                                        <div><br></div>
                                        <label for="LARSEps">Eps (float >0):</label>
                                        <input type="number" step="any" id="LARSEps" name="LARSEps">
                                        <div><br></div>
                                        <label for="LARSCopyX">Copy X:</label>
                                        <select name="LARSCopyX" id="LARSCopyX">
                                            <option value="true">True</option>
                                            <option value="false">False</option>
                                        </select>
                                        <div><br></div>
                                        <label for="LARSFitPath">Fit Path:</label>
                                        <select name="LARSFitPath" id="LARSFitPath">
                                            <option value="true">True</option>
                                            <option value="false">False</option>
                                        </select>
                                    </div>
                                </div>
                            </div>
                            
                            <!-- LARS CV -->
                            <div id="advancedLARSCVFields" class="hidden">
                                <h3>LARS CV Settings</h3>
                                <p class="field-note" style="margin-bottom: 12px; margin-top: 0;">LARS with cross-validation for automatic parameter selection.</p>
                                <div class="nonreqHyperparams">
                                    <div class="toggle-container">
                                        <span><h3>Edit Hyperparameters?</h3></span>
                                        <label class="switch">
                                            <input type="checkbox" id="advancedNonreqLARSCVSlider">
                                            <span class="slider"></span>
                                        </label>
                                    </div>
                                    <div id="advancedNonreqLARSCVParams" class="hidden">
                                        <label for="LARSCVFitIntercept">Fit Intercept:</label>
                                        <select name="LARSCVFitIntercept" id="LARSCVFitIntercept">
                                            <option value="true">True</option>
                                            <option value="false">False</option>
                                        </select>
                                        <div><br></div>
                                        <label for="LARSCVVerbose">Verbose:</label>
                                        <select name="LARSCVVerbose" id="LARSCVVerbose">
                                            <option value="false">False</option>
                                            <option value="true">True</option>
                                        </select>
                                        <div><br></div>
                                        <label for="LARSCVMaxIter">Max Iterations (int >=1):</label>
                                        <input type="number" step=1 value=500 placeholder=500 min=1 id="LARSCVMaxIter" name="LARSCVMaxIter">
                                        <div><br></div>
                                        <label for="LARSCVPrecompute">Precompute:</label>
                                        <select name="LARSCVPrecompute" id="LARSCVPrecompute">
                                            <option value="auto">Auto</option>
                                            <option value="true">True</option>
                                            <option value="false">False</option>
                                        </select>
                                        <div><br></div>
                                        <label for="LARSCVMaxNAlphas">Max N Alphas (int >=1):</label>
                                        <input type="number" step=1 value=1000 placeholder=1000 min=1 id="LARSCVMaxNAlphas" name="LARSCVMaxNAlphas">
                                        <div><br></div>
                                        <label for="LARSCVNJobs">N Jobs (int or None):</label>
                                        <input type="number" step=1 id="LARSCVNJobs" name="LARSCVNJobs">
                                        <div><br></div>
                                        <label for="LARSCVEps">Eps (float >0):</label>
                                        <input type="number" step="any" id="LARSCVEps" name="LARSCVEps">
                                        <div><br></div>
                                        <label for="LARSCVCopyX">Copy X:</label>
                                        <select name="LARSCVCopyX" id="LARSCVCopyX">
                                            <option value="true">True</option>
                                            <option value="false">False</option>
                                        </select>
                                    </div>
                                </div>
                            </div>
                            
                            <!-- Lasso CV -->
                            <div id="advancedLassoCVFields" class="hidden">
                                <h3>Lasso CV Settings</h3>
                                <p class="field-note" style="margin-bottom: 12px; margin-top: 0;">Lasso with cross-validation for automatic alpha selection.</p>
                                <div class="nonreqHyperparams">
                                    <div class="toggle-container">
                                        <span><h3>Edit Hyperparameters?</h3></span>
                                        <label class="switch">
                                            <input type="checkbox" id="advancedNonreqLassoCVSlider">
                                            <span class="slider"></span>
                                        </label>
                                    </div>
                                    <div id="advancedNonreqLassoCVParams" class="hidden">
                                        <label for="LassoCVEps">Eps (float >0):</label>
                                        <input type="number" value=0.001 placeholder=0.001 min="0.0000001" step="any" id="LassoCVEps" name="LassoCVEps">
                                        <div><br></div>
                                        <label for="LassoCVNAlphas">N Alphas (int >=1):</label>
                                        <input type="number" step=1 value=100 placeholder=100 min=1 id="LassoCVNAlphas" name="LassoCVNAlphas">
                                        <div><br></div>
                                        <label for="LassoCVFitIntercept">Fit Intercept:</label>
                                        <select name="LassoCVFitIntercept" id="LassoCVFitIntercept">
                                            <option value="true">True</option>
                                            <option value="false">False</option>
                                        </select>
                                        <div><br></div>
                                        <label for="LassoCVPrecompute">Precompute:</label>
                                        <select name="LassoCVPrecompute" id="LassoCVPrecompute">
                                            <option value="auto">Auto</option>
                                            <option value="true">True</option>
                                            <option value="false">False</option>
                                        </select>
                                        <div><br></div>
                                        <label for="LassoCVMaxIter">Max Iterations (int >=1):</label>
                                        <input type="number" step=1 value=1000 placeholder=1000 min=1 id="LassoCVMaxIter" name="LassoCVMaxIter">
                                        <div><br></div>
                                        <label for="LassoCVTol">Tol (float >0):</label>
                                        <input type="number" value=0.0001 placeholder=0.0001 min="0.0000001" step="any" id="LassoCVTol" name="LassoCVTol">
                                        <div><br></div>
                                        <label for="LassoCVCopyX">Copy X:</label>
                                        <select name="LassoCVCopyX" id="LassoCVCopyX">
                                            <option value="true">True</option>
                                            <option value="false">False</option>
                                        </select>
                                        <div><br></div>
                                        <label for="LassoCVVerbose">Verbose:</label>
                                        <select name="LassoCVVerbose" id="LassoCVVerbose">
                                            <option value="false">False</option>
                                            <option value="true">True</option>
                                        </select>
                                        <div><br></div>
                                        <label for="LassoCVNJobs">N Jobs (int or None):</label>
                                        <input type="number" step=1 id="LassoCVNJobs" name="LassoCVNJobs">
                                        <div><br></div>
                                        <label for="LassoCVPositive">Positive:</label>
                                        <select name="LassoCVPositive" id="LassoCVPositive">
                                            <option value="false">False</option>
                                            <option value="true">True</option>
                                        </select>
                                        <div><br></div>
                                        <label for="LassoCVSelection">Selection:</label>
                                        <select name="LassoCVSelection" id="LassoCVSelection">
                                            <option value="cyclic">Cyclic</option>
                                            <option value="random">Random</option>
                                        </select>
                                    </div>
                                </div>
                            </div>
                            
                            <!-- ElasticNet CV -->
                            <div id="advancedElasticNetCVFields" class="hidden">
                                <h3>ElasticNet CV Settings</h3>
                                <p class="field-note" style="margin-bottom: 12px; margin-top: 0;">ElasticNet with cross-validation for automatic parameter selection.</p>
                                <label for="advancedElasticNetCVL1Ratio">L1 Ratio - Mixing parameter (float [0,1]):</label>
                                <input type="number" value=0.5 placeholder=0.5 min=0 max=1 step="any" id="advancedElasticNetCVL1Ratio" name="advancedElasticNetCVL1Ratio">
                                <div class="nonreqHyperparams">
                                    <div><br></div>
                                    <div class="toggle-container">
                                        <span><h3>Edit Non-Essential Hyperparameters?</h3></span>
                                        <label class="switch">
                                            <input type="checkbox" id="advancedNonreqElasticNetCVSlider">
                                            <span class="slider"></span>
                                        </label>
                                    </div>
                                    <div id="advancedNonreqElasticNetCVParams" class="hidden">
                                        <label for="ElasticNetCVEps">Eps (float >0):</label>
                                        <input type="number" value=0.001 placeholder=0.001 min="0.0000001" step="any" id="ElasticNetCVEps" name="ElasticNetCVEps">
                                        <div><br></div>
                                        <label for="ElasticNetCVNAlphas">N Alphas (int >=1):</label>
                                        <input type="number" step=1 value=100 placeholder=100 min=1 id="ElasticNetCVNAlphas" name="ElasticNetCVNAlphas">
                                        <div><br></div>
                                        <label for="ElasticNetCVFitIntercept">Fit Intercept:</label>
                                        <select name="ElasticNetCVFitIntercept" id="ElasticNetCVFitIntercept">
                                            <option value="true">True</option>
                                            <option value="false">False</option>
                                        </select>
                                        <div><br></div>
                                        <label for="ElasticNetCVPrecompute">Precompute:</label>
                                        <select name="ElasticNetCVPrecompute" id="ElasticNetCVPrecompute">
                                            <option value="auto">Auto</option>
                                            <option value="true">True</option>
                                            <option value="false">False</option>
                                        </select>
                                        <div><br></div>
                                        <label for="ElasticNetCVMaxIter">Max Iterations (int >=1):</label>
                                        <input type="number" step=1 value=1000 placeholder=1000 min=1 id="ElasticNetCVMaxIter" name="ElasticNetCVMaxIter">
                                        <div><br></div>
                                        <label for="ElasticNetCVTol">Tol (float >0):</label>
                                        <input type="number" value=0.0001 placeholder=0.0001 min="0.0000001" step="any" id="ElasticNetCVTol" name="ElasticNetCVTol">
                                        <div><br></div>
                                        <label for="ElasticNetCVCopyX">Copy X:</label>
                                        <select name="ElasticNetCVCopyX" id="ElasticNetCVCopyX">
                                            <option value="true">True</option>
                                            <option value="false">False</option>
                                        </select>
                                        <div><br></div>
                                        <label for="ElasticNetCVVerbose">Verbose (int):</label>
                                        <input type="number" step=1 value=0 placeholder=0 id="ElasticNetCVVerbose" name="ElasticNetCVVerbose">
                                        <div><br></div>
                                        <label for="ElasticNetCVNJobs">N Jobs (int or None):</label>
                                        <input type="number" step=1 id="ElasticNetCVNJobs" name="ElasticNetCVNJobs">
                                        <div><br></div>
                                        <label for="ElasticNetCVPositive">Positive:</label>
                                        <select name="ElasticNetCVPositive" id="ElasticNetCVPositive">
                                            <option value="false">False</option>
                                            <option value="true">True</option>
                                        </select>
                                        <div><br></div>
                                        <label for="ElasticNetCVSelection">Selection:</label>
                                        <select name="ElasticNetCVSelection" id="ElasticNetCVSelection">
                                            <option value="cyclic">Cyclic</option>
                                            <option value="random">Random</option>
                                        </select>
                                    </div>
                                </div>
                            </div>
                            
                            <!-- Ridge CV -->
                            <div id="advancedRidgeCVFields" class="hidden">
                                <h3>Ridge CV Settings</h3>
                                <p class="field-note" style="margin-bottom: 12px; margin-top: 0;">Ridge with cross-validation for automatic alpha selection.</p>
                                <div class="nonreqHyperparams">
                                    <div class="toggle-container">
                                        <span><h3>Edit Hyperparameters?</h3></span>
                                        <label class="switch">
                                            <input type="checkbox" id="advancedNonreqRidgeCVSlider">
                                            <span class="slider"></span>
                                        </label>
                                    </div>
                                    <div id="advancedNonreqRidgeCVParams" class="hidden">
                                        <label for="RidgeCVAlphas">Alphas (comma-separated floats or tuple):</label>
                                        <input type="text" value="0.1, 1.0, 10.0" placeholder="0.1, 1.0, 10.0" id="RidgeCVAlphas" name="RidgeCVAlphas">
                                        <div><br></div>
                                        <label for="RidgeCVFitIntercept">Fit Intercept:</label>
                                        <select name="RidgeCVFitIntercept" id="RidgeCVFitIntercept">
                                            <option value="true">True</option>
                                            <option value="false">False</option>
                                        </select>
                                        <div><br></div>
                                        <label for="RidgeCVScoring">Scoring (string or None):</label>
                                        <input type="text" id="RidgeCVScoring" name="RidgeCVScoring">
                                        <div><br></div>
                                        <label for="RidgeCVGCVMode">GCV Mode:</label>
                                        <select name="RidgeCVGCVMode" id="RidgeCVGCVMode">
                                            <option value="auto">Auto</option>
                                            <option value="svd">SVD</option>
                                            <option value="eigen">Eigen</option>
                                        </select>
                                        <div><br></div>
                                        <label for="RidgeCVStoreCVResults">Store CV Results:</label>
                                        <select name="RidgeCVStoreCVResults" id="RidgeCVStoreCVResults">
                                            <option value="false">False</option>
                                            <option value="true">True</option>
                                        </select>
                                        <div><br></div>
                                        <label for="RidgeCVAlphaPerTarget">Alpha Per Target:</label>
                                        <select name="RidgeCVAlphaPerTarget" id="RidgeCVAlphaPerTarget">
                                            <option value="false">False</option>
                                            <option value="true">True</option>
                                        </select>
                                    </div>
                                </div>
                            </div>
                            
                            <!-- Additional Classification Model Hyperparameters -->
                            <!-- AdaBoost Classifier -->
                            <div id="advancedAdaBoost_classifierFields" class="hidden">
                                <h3>AdaBoost Classifier Settings</h3>
                                <p class="field-note" style="margin-bottom: 12px; margin-top: 0;">AdaBoost combines multiple weak classifiers sequentially, with each focusing on previous errors.</p>
                                <label for="advancedAdaBoostClassifierNEstimators">N Estimators - Number of estimators (integer >=1):</label>
                                <input type="number" step=1 value=50 placeholder=50 min=1 id="advancedAdaBoostClassifierNEstimators" name="advancedAdaBoostClassifierNEstimators">
                                <div><br></div>
                                <label for="advancedAdaBoostClassifierLearningRate">Learning Rate (float >0):</label>
                                <input type="number" value=1.0 placeholder=1.0 min="0.0000001" step="any" id="advancedAdaBoostClassifierLearningRate" name="advancedAdaBoostClassifierLearningRate">
                                <div class="nonreqHyperparams">
                                    <div><br></div>
                                    <div class="toggle-container">
                                        <span><h3>Edit Non-Essential Hyperparameters?</h3></span>
                                        <label class="switch">
                                            <input type="checkbox" id="advancedNonreqAdaBoostClassifierSlider">
                                            <span class="slider"></span>
                                        </label>
                                    </div>
                                    <div id="advancedNonreqAdaBoostClassifierParams" class="hidden">
                                        <!-- No additional hyperparameters for AdaBoostClassifier -->
                                    </div>
                                </div>
                            </div>
                            
                            <!-- Bagging Classifier -->
                            <div id="advancedBagging_classifierFields" class="hidden">
                                <h3>Bagging Classifier Settings</h3>
                                <p class="field-note" style="margin-bottom: 12px; margin-top: 0;">Bagging trains multiple classifiers on random subsets of data and votes on predictions.</p>
                                <label for="advancedBaggingClassifierNEstimators">N Estimators - Number of base estimators (integer >=1):</label>
                                <input type="number" step=1 value=10 placeholder=10 min=1 id="advancedBaggingClassifierNEstimators" name="advancedBaggingClassifierNEstimators">
                                <div class="nonreqHyperparams">
                                    <div><br></div>
                                    <div class="toggle-container">
                                        <span><h3>Edit Non-Essential Hyperparameters?</h3></span>
                                        <label class="switch">
                                            <input type="checkbox" id="advancedNonreqBaggingClassifierSlider">
                                            <span class="slider"></span>
                                        </label>
                                    </div>
                                    <div id="advancedNonreqBaggingClassifierParams" class="hidden">
                                        <label for="BaggingClassifierMaxSamples">Max Samples (float (0,1] or int):</label>
                                        <input type="number" value=1.0 placeholder=1.0 min="0.0000001" max=1 step="any" id="BaggingClassifierMaxSamples" name="BaggingClassifierMaxSamples">
                                        <div><br></div>
                                        <label for="BaggingClassifierMaxFeatures">Max Features (float (0,1] or int):</label>
                                        <input type="number" value=1.0 placeholder=1.0 min="0.0000001" max=1 step="any" id="BaggingClassifierMaxFeatures" name="BaggingClassifierMaxFeatures">
                                        <div><br></div>
                                        <label for="BaggingClassifierBootstrap">Bootstrap:</label>
                                        <select name="BaggingClassifierBootstrap" id="BaggingClassifierBootstrap">
                                            <option value="true">True</option>
                                            <option value="false">False</option>
                                        </select>
                                        <label for="BaggingClassifierBootstrapFeatures">Bootstrap Features:</label>
                                        <select name="BaggingClassifierBootstrapFeatures" id="BaggingClassifierBootstrapFeatures">
                                            <option value="false">False</option>
                                            <option value="true">True</option>
                                        </select>
                                        <div><br></div>
                                        <label for="BaggingClassifierOobScore">OOB Score:</label>
                                        <select name="BaggingClassifierOobScore" id="BaggingClassifierOobScore">
                                            <option value="false">False</option>
                                            <option value="true">True</option>
                                        </select>
                                        <label for="BaggingClassifierWarmStart">Warm Start:</label>
                                        <select name="BaggingClassifierWarmStart" id="BaggingClassifierWarmStart">
                                            <option value="false">False</option>
                                            <option value="true">True</option>
                                        </select>
                                        <div><br></div>
                                        <label for="BaggingClassifierNJobs">N Jobs (int or None):</label>
                                        <input type="number" step=1 id="BaggingClassifierNJobs" name="BaggingClassifierNJobs">
                                        <div><br></div>
                                        <label for="BaggingClassifierVerbose">Verbose (int):</label>
                                        <input type="number" step=1 value=0 placeholder=0 id="BaggingClassifierVerbose" name="BaggingClassifierVerbose">
                                    </div>
                                </div>
                            </div>
                            
                            <!-- Decision Tree Classifier -->
                            <div id="advancedDecisionTree_classifierFields" class="hidden">
                                <h3>Decision Tree Classifier Settings</h3>
                                <p class="field-note" style="margin-bottom: 12px; margin-top: 0;">Decision trees split data recursively based on feature values to make class predictions.</p>
                                <div class="nonreqHyperparams">
                                    <div class="toggle-container">
                                        <span><h3>Edit Hyperparameters?</h3></span>
                                        <label class="switch">
                                            <input type="checkbox" id="advancedNonreqDecisionTreeClassifierSlider">
                                            <span class="slider"></span>
                                        </label>
                                    </div>
                                    <div id="advancedNonreqDecisionTreeClassifierParams" class="hidden">
                                        <label for="DecisionTreeClassifierCriterion">Criterion:</label>
                                        <select name="DecisionTreeClassifierCriterion" id="DecisionTreeClassifierCriterion">
                                            <option value="gini">Gini</option>
                                            <option value="entropy">Entropy</option>
                                            <option value="log_loss">Log Loss</option>
                                        </select>
                                        <label for="DecisionTreeClassifierSplitter">Splitter:</label>
                                        <select name="DecisionTreeClassifierSplitter" id="DecisionTreeClassifierSplitter">
                                            <option value="best">Best</option>
                                            <option value="random">Random</option>
                                        </select>
                                        <div><br></div>
                                        <label for="DecisionTreeClassifierMaxDepth">Max Depth (int or None):</label>
                                        <input type="number" step=1 id="DecisionTreeClassifierMaxDepth" name="DecisionTreeClassifierMaxDepth">
                                        <div><br></div>
                                        <label for="DecisionTreeClassifierMinSamplesSplit">Min Samples Split (int or float):</label>
                                        <input type="number" value=2 placeholder=2 step="any" min=1 id="DecisionTreeClassifierMinSamplesSplit" name="DecisionTreeClassifierMinSamplesSplit">
                                        <div><br></div>
                                        <label for="DecisionTreeClassifierMinSamplesLeaf">Min Samples Leaf (int or float):</label>
                                        <input type="number" value=1 placeholder=1 step="any" min=1 id="DecisionTreeClassifierMinSamplesLeaf" name="DecisionTreeClassifierMinSamplesLeaf">
                                        <div><br></div>
                                        <label for="DecisionTreeClassifierMinWeightFractionLeaf">Min Weight Fraction Leaf (float [0.0, 0.5]):</label>
                                        <input type="number" value=0 placeholder=0 step="any" min=0 max=.5 id="DecisionTreeClassifierMinWeightFractionLeaf" name="DecisionTreeClassifierMinWeightFractionLeaf">
                                        <div><br></div>
                                        <label for="DecisionTreeClassifierMaxFeatures">Max Features (int, float, or string):</label>
                                        <input type="text" id="DecisionTreeClassifierMaxFeatures" name="DecisionTreeClassifierMaxFeatures">
                                        <div><br></div>
                                        <label for="DecisionTreeClassifierMaxLeafNodes">Max Leaf Nodes (int or None):</label>
                                        <input type="number" step=1 id="DecisionTreeClassifierMaxLeafNodes" name="DecisionTreeClassifierMaxLeafNodes">
                                        <div><br></div>
                                        <label for="DecisionTreeClassifierMinImpurityDecrease">Min Impurity Decrease (float >=0):</label>
                                        <input type="number" value=0 placeholder=0 step="any" min=0 id="DecisionTreeClassifierMinImpurityDecrease" name="DecisionTreeClassifierMinImpurityDecrease">
                                        <div><br></div>
                                        <label for="DecisionTreeClassifierClassWeight">Class Weight (dict, 'balanced', or None):</label>
                                        <input type="text" id="DecisionTreeClassifierClassWeight" name="DecisionTreeClassifierClassWeight">
                                        <div><br></div>
                                        <label for="DecisionTreeClassifierCcpAlpha">CCP Alpha (float >=0):</label>
                                        <input type="number" value=0 placeholder=0 step="any" min=0 id="DecisionTreeClassifierCcpAlpha" name="DecisionTreeClassifierCcpAlpha">
                                    </div>
                                </div>
                            </div>
                            
                            <!-- Gradient Boosting Classifier -->
                            <div id="advancedGradientBoosting_classifierFields" class="hidden">
                                <h3>Gradient Boosting Classifier Settings</h3>
                                <p class="field-note" style="margin-bottom: 12px; margin-top: 0;">Gradient Boosting builds trees sequentially, each correcting previous errors. Often achieves high performance.</p>
                                <label for="advancedGradientBoostingClassifierNEstimators">N Estimators - Trees (integer >=1):</label>
                                <input type="number" step=1 value=100 placeholder=100 min=1 id="advancedGradientBoostingClassifierNEstimators" name="advancedGradientBoostingClassifierNEstimators">
                                <div><br></div>
                                <label for="advancedGradientBoostingClassifierLearningRate">Learning Rate (float >0):</label>
                                <input type="number" value=0.1 placeholder=0.1 min="0.0000001" step="any" id="advancedGradientBoostingClassifierLearningRate" name="advancedGradientBoostingClassifierLearningRate">
                                <div class="nonreqHyperparams">
                                    <div><br></div>
                                    <div class="toggle-container">
                                        <span><h3>Edit Non-Essential Hyperparameters?</h3></span>
                                        <label class="switch">
                                            <input type="checkbox" id="advancedNonreqGradientBoostingClassifierSlider">
                                            <span class="slider"></span>
                                        </label>
                                    </div>
                                    <div id="advancedNonreqGradientBoostingClassifierParams" class="hidden">
                                        <label for="GradientBoostingClassifierLoss">Loss Function:</label>
                                        <select name="GradientBoostingClassifierLoss" id="GradientBoostingClassifierLoss">
                                            <option value="log_loss">Log Loss</option>
                                            <option value="exponential">Exponential</option>
                                        </select>
                                        <div><br></div>
                                        <label for="GradientBoostingClassifierSubsample">Subsample (float (0,1]):</label>
                                        <input type="number" value=1.0 placeholder=1.0 min="0.0000001" max=1 step="any" id="GradientBoostingClassifierSubsample" name="GradientBoostingClassifierSubsample">
                                        <div><br></div>
                                        <label for="GradientBoostingClassifierCriterion">Criterion:</label>
                                        <select name="GradientBoostingClassifierCriterion" id="GradientBoostingClassifierCriterion">
                                            <option value="friedman_mse">Friedman MSE</option>
                                            <option value="squared_error">Squared Error</option>
                                        </select>
                                        <div><br></div>
                                        <label for="GradientBoostingClassifierMinSamplesSplit">Min Samples Split (int or float):</label>
                                        <input type="number" value=2 placeholder=2 step="any" min=1 id="GradientBoostingClassifierMinSamplesSplit" name="GradientBoostingClassifierMinSamplesSplit">
                                        <div><br></div>
                                        <label for="GradientBoostingClassifierMinSamplesLeaf">Min Samples Leaf (int or float):</label>
                                        <input type="number" value=1 placeholder=1 step="any" min=1 id="GradientBoostingClassifierMinSamplesLeaf" name="GradientBoostingClassifierMinSamplesLeaf">
                                        <div><br></div>
                                        <label for="GradientBoostingClassifierMinWeightFractionLeaf">Min Weight Fraction Leaf (float [0.0, 0.5]):</label>
                                        <input type="number" value=0 placeholder=0 step="any" min=0 max=.5 id="GradientBoostingClassifierMinWeightFractionLeaf" name="GradientBoostingClassifierMinWeightFractionLeaf">
                                        <div><br></div>
                                        <label for="GradientBoostingClassifierMaxDepth">Max Depth (int or None):</label>
                                        <input type="number" step=1 value=3 placeholder=3 id="GradientBoostingClassifierMaxDepth" name="GradientBoostingClassifierMaxDepth">
                                        <div><br></div>
                                        <label for="GradientBoostingClassifierMinImpurityDecrease">Min Impurity Decrease (float >=0):</label>
                                        <input type="number" value=0 placeholder=0 step="any" min=0 id="GradientBoostingClassifierMinImpurityDecrease" name="GradientBoostingClassifierMinImpurityDecrease">
                                        <div><br></div>
                                        <label for="GradientBoostingClassifierMaxFeatures">Max Features (int, float, or string):</label>
                                        <input type="text" id="GradientBoostingClassifierMaxFeatures" name="GradientBoostingClassifierMaxFeatures">
                                        <div><br></div>
                                        <label for="GradientBoostingClassifierMaxLeafNodes">Max Leaf Nodes (int or None):</label>
                                        <input type="number" step=1 id="GradientBoostingClassifierMaxLeafNodes" name="GradientBoostingClassifierMaxLeafNodes">
                                        <div><br></div>
                                        <label for="GradientBoostingClassifierVerbose">Verbose (int):</label>
                                        <input type="number" step=1 value=0 placeholder=0 id="GradientBoostingClassifierVerbose" name="GradientBoostingClassifierVerbose">
                                        <div><br></div>
                                        <label for="GradientBoostingClassifierWarmStart">Warm Start:</label>
                                        <select name="GradientBoostingClassifierWarmStart" id="GradientBoostingClassifierWarmStart">
                                            <option value="false">False</option>
                                            <option value="true">True</option>
                                        </select>
                                        <div><br></div>
                                        <label for="GradientBoostingClassifierValidationFraction">Validation Fraction (float (0,1)):</label>
                                        <input type="number" value=0.1 placeholder=0.1 min="0.0000001" max=0.999999 step="any" id="GradientBoostingClassifierValidationFraction" name="GradientBoostingClassifierValidationFraction">
                                        <div><br></div>
                                        <label for="GradientBoostingClassifierNIterNoChange">N Iter No Change (int or None):</label>
                                        <input type="number" step=1 id="GradientBoostingClassifierNIterNoChange" name="GradientBoostingClassifierNIterNoChange">
                                        <div><br></div>
                                        <label for="GradientBoostingClassifierTol">Tol (float >0):</label>
                                        <input type="number" value=0.0001 placeholder=0.0001 min="0.0000001" step="any" id="GradientBoostingClassifierTol" name="GradientBoostingClassifierTol">
                                    </div>
                                </div>
                            </div>
                            
                            <!-- Histogram Gradient Boosting Classifier -->
                            <div id="advancedHistGradientBoosting_classifierFields" class="hidden">
                                <h3>Histogram Gradient Boosting Classifier Settings</h3>
                                <p class="field-note" style="margin-bottom: 12px; margin-top: 0;">Fast gradient boosting using histogram-based tree building. More efficient than standard gradient boosting.</p>
                                <label for="advancedHistGradientBoostingClassifierLearningRate">Learning Rate (float >0):</label>
                                <input type="number" value=0.1 placeholder=0.1 min="0.0000001" step="any" id="advancedHistGradientBoostingClassifierLearningRate" name="advancedHistGradientBoostingClassifierLearningRate">
                                <div><br></div>
                                <label for="advancedHistGradientBoostingClassifierMaxIter">Max Iterations (int >=1):</label>
                                <input type="number" step=1 value=100 placeholder=100 min=1 id="advancedHistGradientBoostingClassifierMaxIter" name="advancedHistGradientBoostingClassifierMaxIter">
                                <div><br></div>
                                <label for="advancedHistGradientBoostingClassifierMaxLeafNodes">Max Leaf Nodes (int >=2):</label>
                                <input type="number" step=1 value=31 placeholder=31 min=2 id="advancedHistGradientBoostingClassifierMaxLeafNodes" name="advancedHistGradientBoostingClassifierMaxLeafNodes">
                                <div class="nonreqHyperparams">
                                    <div><br></div>
                                    <div class="toggle-container">
                                        <span><h3>Edit Non-Essential Hyperparameters?</h3></span>
                                        <label class="switch">
                                            <input type="checkbox" id="advancedNonreqHistGradientBoostingClassifierSlider">
                                            <span class="slider"></span>
                                        </label>
                                    </div>
                                    <div id="advancedNonreqHistGradientBoostingClassifierParams" class="hidden">
                                        <label for="HistGradientBoostingClassifierLoss">Loss Function:</label>
                                        <select name="HistGradientBoostingClassifierLoss" id="HistGradientBoostingClassifierLoss">
                                            <option value="log_loss">Log Loss</option>
                                            <option value="auto">Auto</option>
                                        </select>
                                        <div><br></div>
                                        <label for="HistGradientBoostingClassifierMaxDepth">Max Depth (int or None):</label>
                                        <input type="number" step=1 id="HistGradientBoostingClassifierMaxDepth" name="HistGradientBoostingClassifierMaxDepth">
                                        <div><br></div>
                                        <label for="HistGradientBoostingClassifierMinSamplesLeaf">Min Samples Leaf (int >=1):</label>
                                        <input type="number" step=1 value=20 placeholder=20 min=1 id="HistGradientBoostingClassifierMinSamplesLeaf" name="HistGradientBoostingClassifierMinSamplesLeaf">
                                        <div><br></div>
                                        <label for="HistGradientBoostingClassifierL2Regularization">L2 Regularization (float >=0):</label>
                                        <input type="number" value=0 placeholder=0 step="any" min=0 id="HistGradientBoostingClassifierL2Regularization" name="HistGradientBoostingClassifierL2Regularization">
                                        <div><br></div>
                                        <label for="HistGradientBoostingClassifierMaxBins">Max Bins (int >=2):</label>
                                        <input type="number" step=1 value=255 placeholder=255 min=2 id="HistGradientBoostingClassifierMaxBins" name="HistGradientBoostingClassifierMaxBins">
                                        <div><br></div>
                                        <label for="HistGradientBoostingClassifierWarmStart">Warm Start:</label>
                                        <select name="HistGradientBoostingClassifierWarmStart" id="HistGradientBoostingClassifierWarmStart">
                                            <option value="false">False</option>
                                            <option value="true">True</option>
                                        </select>
                                        <div><br></div>
                                        <label for="HistGradientBoostingClassifierEarlyStopping">Early Stopping:</label>
                                        <select name="HistGradientBoostingClassifierEarlyStopping" id="HistGradientBoostingClassifierEarlyStopping">
                                            <option value="auto">Auto</option>
                                            <option value="false">False</option>
                                            <option value="true">True</option>
                                        </select>
                                        <div><br></div>
                                        <label for="HistGradientBoostingClassifierScoring">Scoring:</label>
                                        <input type="text" value="loss" placeholder="loss" id="HistGradientBoostingClassifierScoring" name="HistGradientBoostingClassifierScoring">
                                        <div><br></div>
                                        <label for="HistGradientBoostingClassifierValidationFraction">Validation Fraction (float (0,1)):</label>
                                        <input type="number" value=0.1 placeholder=0.1 min="0.0000001" max=0.999999 step="any" id="HistGradientBoostingClassifierValidationFraction" name="HistGradientBoostingClassifierValidationFraction">
                                        <div><br></div>
                                        <label for="HistGradientBoostingClassifierNIterNoChange">N Iter No Change (int >=1):</label>
                                        <input type="number" step=1 value=10 placeholder=10 min=1 id="HistGradientBoostingClassifierNIterNoChange" name="HistGradientBoostingClassifierNIterNoChange">
                                        <div><br></div>
                                        <label for="HistGradientBoostingClassifierTol">Tol (float >0):</label>
                                        <input type="number" value=0.0000001 placeholder=0.0000001 min="0.0000001" step="any" id="HistGradientBoostingClassifierTol" name="HistGradientBoostingClassifierTol">
                                        <div><br></div>
                                        <label for="HistGradientBoostingClassifierVerbose">Verbose (int):</label>
                                        <input type="number" step=1 value=0 placeholder=0 id="HistGradientBoostingClassifierVerbose" name="HistGradientBoostingClassifierVerbose">
                                    </div>
                                </div>
                            </div>
                            
                            <!-- K-Neighbors Classifier -->
                            <div id="advancedKNeighbors_classifierFields" class="hidden">
                                <h3>K-Neighbors Classifier Settings</h3>
                                <p class="field-note" style="margin-bottom: 12px; margin-top: 0;">Classifies based on k nearest neighbors. Simple and effective for many problems.</p>
                                <label for="advancedKNeighborsClassifierNNeighbors">N Neighbors - Number of neighbors (int >=1):</label>
                                <input type="number" step=1 value=5 placeholder=5 min=1 id="advancedKNeighborsClassifierNNeighbors" name="advancedKNeighborsClassifierNNeighbors">
                                <div class="nonreqHyperparams">
                                    <div><br></div>
                                    <div class="toggle-container">
                                        <span><h3>Edit Non-Essential Hyperparameters?</h3></span>
                                        <label class="switch">
                                            <input type="checkbox" id="advancedNonreqKNeighborsClassifierSlider">
                                            <span class="slider"></span>
                                        </label>
                                    </div>
                                    <div id="advancedNonreqKNeighborsClassifierParams" class="hidden">
                                        <label for="KNeighborsClassifierWeights">Weights:</label>
                                        <select name="KNeighborsClassifierWeights" id="KNeighborsClassifierWeights">
                                            <option value="uniform">Uniform</option>
                                            <option value="distance">Distance</option>
                                        </select>
                                        <div><br></div>
                                        <label for="KNeighborsClassifierAlgorithm">Algorithm:</label>
                                        <select name="KNeighborsClassifierAlgorithm" id="KNeighborsClassifierAlgorithm">
                                            <option value="auto">Auto</option>
                                            <option value="ball_tree">Ball Tree</option>
                                            <option value="kd_tree">KD Tree</option>
                                            <option value="brute">Brute Force</option>
                                        </select>
                                        <div><br></div>
                                        <label for="KNeighborsClassifierLeafSize">Leaf Size (int >=1):</label>
                                        <input type="number" step=1 value=30 placeholder=30 min=1 id="KNeighborsClassifierLeafSize" name="KNeighborsClassifierLeafSize">
                                        <div><br></div>
                                        <label for="KNeighborsClassifierP">P - Minkowski metric power (float >=1):</label>
                                        <input type="number" value=2 placeholder=2 min=1 step="any" id="KNeighborsClassifierP" name="KNeighborsClassifierP">
                                        <div><br></div>
                                        <label for="KNeighborsClassifierMetric">Metric:</label>
                                        <input type="text" value="minkowski" placeholder="minkowski" id="KNeighborsClassifierMetric" name="KNeighborsClassifierMetric">
                                        <div><br></div>
                                        <label for="KNeighborsClassifierNJobs">N Jobs (int or None):</label>
                                        <input type="number" step=1 id="KNeighborsClassifierNJobs" name="KNeighborsClassifierNJobs">
                                    </div>
                                </div>
                            </div>
                            
                            <!-- Linear Discriminant Analysis -->
                            <div id="advancedLDA_classifierFields" class="hidden">
                                <h3>Linear Discriminant Analysis Settings</h3>
                                <p class="field-note" style="margin-bottom: 12px; margin-top: 0;">Linear classifier that finds optimal linear combination of features for class separation.</p>
                                <div class="nonreqHyperparams">
                                    <div class="toggle-container">
                                        <span><h3>Edit Hyperparameters?</h3></span>
                                        <label class="switch">
                                            <input type="checkbox" id="advancedNonreqLDAClassifierSlider">
                                            <span class="slider"></span>
                                        </label>
                                    </div>
                                    <div id="advancedNonreqLDAClassifierParams" class="hidden">
                                        <label for="LDAClassifierSolver">Solver:</label>
                                        <select name="LDAClassifierSolver" id="LDAClassifierSolver">
                                            <option value="svd">SVD</option>
                                            <option value="lsqr">LSQR</option>
                                            <option value="eigen">Eigen</option>
                                        </select>
                                        <div><br></div>
                                        <label for="LDAClassifierShrinkage">Shrinkage (float [0,1] or None):</label>
                                        <input type="number" step="any" id="LDAClassifierShrinkage" name="LDAClassifierShrinkage">
                                        <div><br></div>
                                        <label for="LDAClassifierPriors">Priors (array-like or None):</label>
                                        <input type="text" id="LDAClassifierPriors" name="LDAClassifierPriors">
                                        <div><br></div>
                                        <label for="LDAClassifierNComponents">N Components (int or None):</label>
                                        <input type="number" step=1 id="LDAClassifierNComponents" name="LDAClassifierNComponents">
                                        <div><br></div>
                                        <label for="LDAClassifierStoreCovariance">Store Covariance:</label>
                                        <select name="LDAClassifierStoreCovariance" id="LDAClassifierStoreCovariance">
                                            <option value="false">False</option>
                                            <option value="true">True</option>
                                        </select>
                                        <div><br></div>
                                        <label for="LDAClassifierTol">Tol (float >0):</label>
                                        <input type="number" value=0.0001 placeholder=0.0001 min="0.0000001" step="any" id="LDAClassifierTol" name="LDAClassifierTol">
                                    </div>
                                </div>
                            </div>
                            
                            <!-- Linear SVC -->
                            <div id="advancedLinearSVC_classifierFields" class="hidden">
                                <h3>Linear SVC Settings</h3>
                                <p class="field-note" style="margin-bottom: 12px; margin-top: 0;">Linear Support Vector Classification. Fast for large datasets.</p>
                                <label for="advancedLinearSVCC">C - Regularization parameter (float >0):</label>
                                <input type="number" value=1.0 placeholder=1.0 min="0.0000001" step="any" id="advancedLinearSVCC" name="advancedLinearSVCC">
                                <div class="nonreqHyperparams">
                                    <div><br></div>
                                    <div class="toggle-container">
                                        <span><h3>Edit Non-Essential Hyperparameters?</h3></span>
                                        <label class="switch">
                                            <input type="checkbox" id="advancedNonreqLinearSVCSlider">
                                            <span class="slider"></span>
                                        </label>
                                    </div>
                                    <div id="advancedNonreqLinearSVCParams" class="hidden">
                                        <label for="LinearSVCLoss">Loss Function:</label>
                                        <select name="LinearSVCLoss" id="LinearSVCLoss">
                                            <option value="squared_hinge">Squared Hinge</option>
                                            <option value="hinge">Hinge</option>
                                        </select>
                                        <div><br></div>
                                        <label for="LinearSVCPenalty">Penalty:</label>
                                        <select name="LinearSVCPenalty" id="LinearSVCPenalty">
                                            <option value="l2">L2</option>
                                            <option value="l1">L1</option>
                                        </select>
                                        <div><br></div>
                                        <label for="LinearSVCDual">Dual:</label>
                                        <select name="LinearSVCDual" id="LinearSVCDual">
                                            <option value="true">True</option>
                                            <option value="false">False</option>
                                        </select>
                                        <div><br></div>
                                        <label for="LinearSVCTol">Tol (float >0):</label>
                                        <input type="number" value=0.0001 placeholder=0.0001 min="0.0000001" step="any" id="LinearSVCTol" name="LinearSVCTol">
                                        <div><br></div>
                                        <label for="LinearSVCMultiClass">Multi Class:</label>
                                        <select name="LinearSVCMultiClass" id="LinearSVCMultiClass">
                                            <option value="ovr">OVR</option>
                                            <option value="crammer_singer">Crammer Singer</option>
                                        </select>
                                        <div><br></div>
                                        <label for="LinearSVCFitIntercept">Fit Intercept:</label>
                                        <select name="LinearSVCFitIntercept" id="LinearSVCFitIntercept">
                                            <option value="true">True</option>
                                            <option value="false">False</option>
                                        </select>
                                        <div><br></div>
                                        <label for="LinearSVCInterceptScaling">Intercept Scaling (float >0):</label>
                                        <input type="number" value=1.0 placeholder=1.0 min="0.0000001" step="any" id="LinearSVCInterceptScaling" name="LinearSVCInterceptScaling">
                                        <div><br></div>
                                        <label for="LinearSVCMaxIter">Max Iterations (int >=1):</label>
                                        <input type="number" step=1 value=1000 placeholder=1000 min=1 id="LinearSVCMaxIter" name="LinearSVCMaxIter">
                                        <div><br></div>
                                        <label for="LinearSVCClassWeight">Class Weight (dict, 'balanced', or None):</label>
                                        <input type="text" id="LinearSVCClassWeight" name="LinearSVCClassWeight">
                                        <div><br></div>
                                        <label for="LinearSVCVerbose">Verbose (int):</label>
                                        <input type="number" step=1 value=0 placeholder=0 id="LinearSVCVerbose" name="LinearSVCVerbose">
                                    </div>
                                </div>
                            </div>
                            
                            <!-- Nu-SVC -->
                            <div id="advancedNuSVC_classifierFields" class="hidden">
                                <h3>Nu-SVC Settings</h3>
                                <p class="field-note" style="margin-bottom: 12px; margin-top: 0;">Nu-Support Vector Classification with kernel functions.</p>
                                <label for="advancedNuSVCNu">Nu - Upper bound on fraction of margin errors (float (0,1)):</label>
                                <input type="number" value=0.5 placeholder=0.5 min="0.0000001" max=0.999999 step="any" id="advancedNuSVCNu" name="advancedNuSVCNu">
                                <div><br></div>
                                <label for="advancedNuSVCC">C - Regularization parameter (float >0):</label>
                                <input type="number" value=1.0 placeholder=1.0 min="0.0000001" step="any" id="advancedNuSVCC" name="advancedNuSVCC">
                                <div><br></div>
                                <label for="advancedNuSVCKernel">Kernel:</label>
                                <select name="advancedNuSVCKernel" id="advancedNuSVCKernel">
                                    <option value="rbf">RBF</option>
                                    <option value="linear">Linear</option>
                                    <option value="poly">Polynomial</option>
                                    <option value="sigmoid">Sigmoid</option>
                                </select>
                                <div class="nonreqHyperparams">
                                    <div><br></div>
                                    <div class="toggle-container">
                                        <span><h3>Edit Non-Essential Hyperparameters?</h3></span>
                                        <label class="switch">
                                            <input type="checkbox" id="advancedNonreqNuSVCSlider">
                                            <span class="slider"></span>
                                        </label>
                                    </div>
                                    <div id="advancedNonreqNuSVCParams" class="hidden">
                                        <label for="NuSVCDegree">Degree (for poly kernel, int >=1):</label>
                                        <input type="number" step=1 value=3 placeholder=3 min=1 id="NuSVCDegree" name="NuSVCDegree">
                                        <div><br></div>
                                        <label for="NuSVCGamma">Gamma (float or 'scale' or 'auto'):</label>
                                        <input type="text" value="scale" placeholder="scale" id="NuSVCGamma" name="NuSVCGamma">
                                        <div><br></div>
                                        <label for="NuSVCCoef0">Coef0 (for poly/sigmoid kernels, float):</label>
                                        <input type="number" value=0.0 placeholder=0.0 step="any" id="NuSVCCoef0" name="NuSVCCoef0">
                                        <div><br></div>
                                        <label for="NuSVCShrinking">Shrinking:</label>
                                        <select name="NuSVCShrinking" id="NuSVCShrinking">
                                            <option value="true">True</option>
                                            <option value="false">False</option>
                                        </select>
                                        <div><br></div>
                                        <label for="NuSVCTol">Tol (float >0):</label>
                                        <input type="number" value=0.001 placeholder=0.001 min="0.0000001" step="any" id="NuSVCTol" name="NuSVCTol">
                                        <div><br></div>
                                        <label for="NuSVCCacheSize">Cache Size (float):</label>
                                        <input type="number" step="any" value=200 placeholder=200 id="NuSVCCacheSize" name="NuSVCCacheSize">
                                        <div><br></div>
                                        <label for="NuSVCVerbose">Verbose:</label>
                                        <select name="NuSVCVerbose" id="NuSVCVerbose">
                                            <option value="false">False</option>
                                            <option value="true">True</option>
                                        </select>
                                        <div><br></div>
                                        <label for="NuSVCMaxIter">Max Iterations (int, -1 for no limit):</label>
                                        <input type="number" step=1 value=-1 placeholder=-1 id="NuSVCMaxIter" name="NuSVCMaxIter">
                                        <div><br></div>
                                        <label for="NuSVCClassWeight">Class Weight (dict, 'balanced', or None):</label>
                                        <input type="text" id="NuSVCClassWeight" name="NuSVCClassWeight">
                                        <div><br></div>
                                        <label for="NuSVCDecisionFunctionShape">Decision Function Shape:</label>
                                        <select name="NuSVCDecisionFunctionShape" id="NuSVCDecisionFunctionShape">
                                            <option value="ovr">OVR</option>
                                            <option value="ovo">OVO</option>
                                        </select>
                                        <div><br></div>
                                        <label for="NuSVCBreakTies">Break Ties:</label>
                                        <select name="NuSVCBreakTies" id="NuSVCBreakTies">
                                            <option value="false">False</option>
                                            <option value="true">True</option>
                                        </select>
                                    </div>
                                </div>
                            </div>
                            
                            <!-- Passive Aggressive Classifier -->
                            <div id="advancedPassiveAggressive_classifierFields" class="hidden">
                                <h3>Passive Aggressive Classifier Settings</h3>
                                <p class="field-note" style="margin-bottom: 12px; margin-top: 0;">Online learning algorithm that updates model only when prediction error exceeds threshold.</p>
                                <label for="advancedPassiveAggressiveClassifierC">C - Regularization strength (float >0):</label>
                                <input type="number" value=1.0 placeholder=1.0 min="0.0000001" step="any" id="advancedPassiveAggressiveClassifierC" name="advancedPassiveAggressiveClassifierC">
                                <div class="nonreqHyperparams">
                                    <div><br></div>
                                    <div class="toggle-container">
                                        <span><h3>Edit Non-Essential Hyperparameters?</h3></span>
                                        <label class="switch">
                                            <input type="checkbox" id="advancedNonreqPassiveAggressiveClassifierSlider">
                                            <span class="slider"></span>
                                        </label>
                                    </div>
                                    <div id="advancedNonreqPassiveAggressiveClassifierParams" class="hidden">
                                        <label for="PassiveAggressiveClassifierFitIntercept">Fit Intercept:</label>
                                        <select name="PassiveAggressiveClassifierFitIntercept" id="PassiveAggressiveClassifierFitIntercept">
                                            <option value="true">True</option>
                                            <option value="false">False</option>
                                        </select>
                                        <div><br></div>
                                        <label for="PassiveAggressiveClassifierMaxIter">Max Iterations (int >=1):</label>
                                        <input type="number" step=1 value=1000 placeholder=1000 min=1 id="PassiveAggressiveClassifierMaxIter" name="PassiveAggressiveClassifierMaxIter">
                                        <div><br></div>
                                        <label for="PassiveAggressiveClassifierTol">Tol (float >0):</label>
                                        <input type="number" value=0.001 placeholder=0.001 min="0.0000001" step="any" id="PassiveAggressiveClassifierTol" name="PassiveAggressiveClassifierTol">
                                        <div><br></div>
                                        <label for="PassiveAggressiveClassifierShuffle">Shuffle:</label>
                                        <select name="PassiveAggressiveClassifierShuffle" id="PassiveAggressiveClassifierShuffle">
                                            <option value="true">True</option>
                                            <option value="false">False</option>
                                        </select>
                                        <div><br></div>
                                        <label for="PassiveAggressiveClassifierVerbose">Verbose (int):</label>
                                        <input type="number" step=1 value=0 placeholder=0 id="PassiveAggressiveClassifierVerbose" name="PassiveAggressiveClassifierVerbose">
                                        <div><br></div>
                                        <label for="PassiveAggressiveClassifierLoss">Loss Function:</label>
                                        <select name="PassiveAggressiveClassifierLoss" id="PassiveAggressiveClassifierLoss">
                                            <option value="hinge">Hinge</option>
                                            <option value="squared_hinge">Squared Hinge</option>
                                        </select>
                                        <div><br></div>
                                        <label for="PassiveAggressiveClassifierWarmStart">Warm Start:</label>
                                        <select name="PassiveAggressiveClassifierWarmStart" id="PassiveAggressiveClassifierWarmStart">
                                            <option value="false">False</option>
                                            <option value="true">True</option>
                                        </select>
                                        <div><br></div>
                                        <label for="PassiveAggressiveClassifierClassWeight">Class Weight (dict, 'balanced', or None):</label>
                                        <input type="text" id="PassiveAggressiveClassifierClassWeight" name="PassiveAggressiveClassifierClassWeight">
                                        <div><br></div>
                                        <label for="PassiveAggressiveClassifierNJobs">N Jobs (int or None):</label>
                                        <input type="number" step=1 id="PassiveAggressiveClassifierNJobs" name="PassiveAggressiveClassifierNJobs">
                                        <div><br></div>
                                        <label for="PassiveAggressiveClassifierAverage">Average:</label>
                                        <select name="PassiveAggressiveClassifierAverage" id="PassiveAggressiveClassifierAverage">
                                            <option value="false">False</option>
                                            <option value="true">True</option>
                                        </select>
                                    </div>
                                </div>
                            </div>
                            
                            <!-- Quadratic Discriminant Analysis -->
                            <div id="advancedQDA_classifierFields" class="hidden">
                                <h3>Quadratic Discriminant Analysis Settings</h3>
                                <p class="field-note" style="margin-bottom: 12px; margin-top: 0;">Quadratic classifier that models each class with its own covariance matrix.</p>
                                <div class="nonreqHyperparams">
                                    <div class="toggle-container">
                                        <span><h3>Edit Hyperparameters?</h3></span>
                                        <label class="switch">
                                            <input type="checkbox" id="advancedNonreqQDAClassifierSlider">
                                            <span class="slider"></span>
                                        </label>
                                    </div>
                                    <div id="advancedNonreqQDAClassifierParams" class="hidden">
                                        <label for="QDAClassifierPriors">Priors (array-like or None):</label>
                                        <input type="text" id="QDAClassifierPriors" name="QDAClassifierPriors">
                                        <div><br></div>
                                        <label for="QDAClassifierRegParam">Reg Param - Regularization (float [0,1]):</label>
                                        <input type="number" value=0.0 placeholder=0.0 min=0 max=1 step="any" id="QDAClassifierRegParam" name="QDAClassifierRegParam">
                                        <div><br></div>
                                        <label for="QDAClassifierStoreCovariance">Store Covariance:</label>
                                        <select name="QDAClassifierStoreCovariance" id="QDAClassifierStoreCovariance">
                                            <option value="false">False</option>
                                            <option value="true">True</option>
                                        </select>
                                        <div><br></div>
                                        <label for="QDAClassifierTol">Tol (float >0):</label>
                                        <input type="number" value=0.0001 placeholder=0.0001 min="0.0000001" step="any" id="QDAClassifierTol" name="QDAClassifierTol">
                                    </div>
                                </div>
                            </div>
                            
                            <!-- Ridge Classifier -->
                            <div id="advancedRidge_classifierFields" class="hidden">
                                <h3>Ridge Classifier Settings</h3>
                                <p class="field-note" style="margin-bottom: 12px; margin-top: 0;">Ridge regression adapted for classification. Uses L2 regularization.</p>
                                <label for="advancedRidgeClassifierAlpha">Alpha - Regularization strength (float >=0):</label>
                                <input type="number" value=1.0 placeholder=1.0 min=0 step="any" id="advancedRidgeClassifierAlpha" name="advancedRidgeClassifierAlpha">
                                <div class="nonreqHyperparams">
                                    <div><br></div>
                                    <div class="toggle-container">
                                        <span><h3>Edit Non-Essential Hyperparameters?</h3></span>
                                        <label class="switch">
                                            <input type="checkbox" id="advancedNonreqRidgeClassifierSlider">
                                            <span class="slider"></span>
                                        </label>
                                    </div>
                                    <div id="advancedNonreqRidgeClassifierParams" class="hidden">
                                        <label for="RidgeClassifierFitIntercept">Fit Intercept:</label>
                                        <select name="RidgeClassifierFitIntercept" id="RidgeClassifierFitIntercept">
                                            <option value="true">True</option>
                                            <option value="false">False</option>
                                        </select>
                                        <div><br></div>
                                        <label for="RidgeClassifierCopyX">Copy X:</label>
                                        <select name="RidgeClassifierCopyX" id="RidgeClassifierCopyX">
                                            <option value="true">True</option>
                                            <option value="false">False</option>
                                        </select>
                                        <div><br></div>
                                        <label for="RidgeClassifierMaxIter">Max Iterations (int or None):</label>
                                        <input type="number" step=1 id="RidgeClassifierMaxIter" name="RidgeClassifierMaxIter">
                                        <div><br></div>
                                        <label for="RidgeClassifierTol">Tol (float >0):</label>
                                        <input type="number" value=0.0001 placeholder=0.0001 min="0.0000001" step="any" id="RidgeClassifierTol" name="RidgeClassifierTol">
                                        <div><br></div>
                                        <label for="RidgeClassifierClassWeight">Class Weight (dict, 'balanced', or None):</label>
                                        <input type="text" id="RidgeClassifierClassWeight" name="RidgeClassifierClassWeight">
                                        <div><br></div>
                                        <label for="RidgeClassifierSolver">Solver:</label>
                                        <select name="RidgeClassifierSolver" id="RidgeClassifierSolver">
                                            <option value="auto">Auto</option>
                                            <option value="svd">SVD</option>
                                            <option value="cholesky">Cholesky</option>
                                            <option value="lsqr">LSQR</option>
                                            <option value="sparse_cg">Sparse CG</option>
                                            <option value="sag">SAG</option>
                                            <option value="saga">SAGA</option>
                                            <option value="lbfgs">LBFGS</option>
                                        </select>
                                        <div><br></div>
                                        <label for="RidgeClassifierPositive">Positive:</label>
                                        <select name="RidgeClassifierPositive" id="RidgeClassifierPositive">
                                            <option value="false">False</option>
                                            <option value="true">True</option>
                                        </select>
                                    </div>
                                </div>
                            </div>
                            
                            <!-- Bernoulli Naive Bayes -->
                            <div id="advancedBernoulliNB_classifierFields" class="hidden">
                                <h3>Bernoulli Naive Bayes Settings</h3>
                                <p class="field-note" style="margin-bottom: 12px; margin-top: 0;">Naive Bayes for binary/boolean features. Assumes features are independent.</p>
                                <div class="nonreqHyperparams">
                                    <div class="toggle-container">
                                        <span><h3>Edit Hyperparameters?</h3></span>
                                        <label class="switch">
                                            <input type="checkbox" id="advancedNonreqBernoulliNBSlider">
                                            <span class="slider"></span>
                                        </label>
                                    </div>
                                    <div id="advancedNonreqBernoulliNBParams" class="hidden">
                                        <label for="BernoulliNBAlpha">Alpha - Smoothing parameter (float >=0):</label>
                                        <input type="number" value=1.0 placeholder=1.0 min=0 step="any" id="BernoulliNBAlpha" name="BernoulliNBAlpha">
                                        <div><br></div>
                                        <label for="BernoulliNBFitPrior">Fit Prior:</label>
                                        <select name="BernoulliNBFitPrior" id="BernoulliNBFitPrior">
                                            <option value="true">True</option>
                                            <option value="false">False</option>
                                        </select>
                                        <div><br></div>
                                        <label for="BernoulliNBBinarize">Binarize (float or None):</label>
                                        <input type="number" step="any" id="BernoulliNBBinarize" name="BernoulliNBBinarize">
                                    </div>
                                </div>
                            </div>
                            
                            <!-- Categorical Naive Bayes -->
                            <div id="advancedCategoricalNB_classifierFields" class="hidden">
                                <h3>Categorical Naive Bayes Settings</h3>
                                <p class="field-note" style="margin-bottom: 12px; margin-top: 0;">Naive Bayes for categorical features. Each feature is assumed to follow a categorical distribution.</p>
                                <div class="nonreqHyperparams">
                                    <div class="toggle-container">
                                        <span><h3>Edit Hyperparameters?</h3></span>
                                        <label class="switch">
                                            <input type="checkbox" id="advancedNonreqCategoricalNBSlider">
                                            <span class="slider"></span>
                                        </label>
                                    </div>
                                    <div id="advancedNonreqCategoricalNBParams" class="hidden">
                                        <label for="CategoricalNBAlpha">Alpha - Smoothing parameter (float >=0):</label>
                                        <input type="number" value=1.0 placeholder=1.0 min=0 step="any" id="CategoricalNBAlpha" name="CategoricalNBAlpha">
                                        <div><br></div>
                                        <label for="CategoricalNBFitPrior">Fit Prior:</label>
                                        <select name="CategoricalNBFitPrior" id="CategoricalNBFitPrior">
                                            <option value="true">True</option>
                                            <option value="false">False</option>
                                        </select>
                                        <div><br></div>
                                        <label for="CategoricalNBMinCategoryCount">Min Category Count (int >=1):</label>
                                        <input type="number" step=1 value=1 placeholder=1 min=1 id="CategoricalNBMinCategoryCount" name="CategoricalNBMinCategoryCount">
                                    </div>
                                </div>
                            </div>
                            
                            <!-- Complement Naive Bayes -->
                            <div id="advancedComplementNB_classifierFields" class="hidden">
                                <h3>Complement Naive Bayes Settings</h3>
                                <p class="field-note" style="margin-bottom: 12px; margin-top: 0;">Adaptation of Multinomial Naive Bayes that works well with imbalanced datasets.</p>
                                <div class="nonreqHyperparams">
                                    <div class="toggle-container">
                                        <span><h3>Edit Hyperparameters?</h3></span>
                                        <label class="switch">
                                            <input type="checkbox" id="advancedNonreqComplementNBSlider">
                                            <span class="slider"></span>
                                        </label>
                                    </div>
                                    <div id="advancedNonreqComplementNBParams" class="hidden">
                                        <label for="ComplementNBAlpha">Alpha - Smoothing parameter (float >=0):</label>
                                        <input type="number" value=1.0 placeholder=1.0 min=0 step="any" id="ComplementNBAlpha" name="ComplementNBAlpha">
                                        <div><br></div>
                                        <label for="ComplementNBFitPrior">Fit Prior:</label>
                                        <select name="ComplementNBFitPrior" id="ComplementNBFitPrior">
                                            <option value="true">True</option>
                                            <option value="false">False</option>
                                        </select>
                                        <div><br></div>
                                        <label for="ComplementNBNorm">Norm:</label>
                                        <select name="ComplementNBNorm" id="ComplementNBNorm">
                                            <option value="false">False</option>
                                            <option value="true">True</option>
                                        </select>
                                    </div>
                                </div>
                            </div>
                            
                            <!-- Multinomial Naive Bayes -->
                            <div id="advancedMultinomialNB_classifierFields" class="hidden">
                                <h3>Multinomial Naive Bayes Settings</h3>
                                <p class="field-note" style="margin-bottom: 12px; margin-top: 0;">Naive Bayes for multinomial data. Good for text classification and count data.</p>
                                <div class="nonreqHyperparams">
                                    <div class="toggle-container">
                                        <span><h3>Edit Hyperparameters?</h3></span>
                                        <label class="switch">
                                            <input type="checkbox" id="advancedNonreqMultinomialNBSlider">
                                            <span class="slider"></span>
                                        </label>
                                    </div>
                                    <div id="advancedNonreqMultinomialNBParams" class="hidden">
                                        <label for="MultinomialNBAlpha">Alpha - Smoothing parameter (float >=0):</label>
                                        <input type="number" value=1.0 placeholder=1.0 min=0 step="any" id="MultinomialNBAlpha" name="MultinomialNBAlpha">
                                        <div><br></div>
                                        <label for="MultinomialNBFitPrior">Fit Prior:</label>
                                        <select name="MultinomialNBFitPrior" id="MultinomialNBFitPrior">
                                            <option value="true">True</option>
                                            <option value="false">False</option>
                                        </select>
                                        <div><br></div>
                                        <label for="MultinomialNBClassPrior">Class Prior (array-like or None):</label>
                                        <input type="text" id="MultinomialNBClassPrior" name="MultinomialNBClassPrior">
                                    </div>
                                </div>
                            </div>
                            
                            <!-- Additional Clustering Model Hyperparameters -->
                            <!-- Affinity Propagation -->
                            <div id="advancedAffinityPropagationFields" class="hidden">
                                <h3>Affinity Propagation Settings</h3>
                                <p class="field-note" style="margin-bottom: 12px; margin-top: 0;">Clustering by passing messages between data points. Automatically determines number of clusters.</p>
                                <div class="nonreqHyperparams">
                                    <div class="toggle-container">
                                        <span><h3>Edit Hyperparameters?</h3></span>
                                        <label class="switch">
                                            <input type="checkbox" id="advancedNonreqAffinityPropagationSlider">
                                            <span class="slider"></span>
                                        </label>
                                    </div>
                                    <div id="advancedNonreqAffinityPropagationParams" class="hidden">
                                        <label for="AffinityPropagationDamping">Damping (float [0.5, 1.0]):</label>
                                        <input type="number" value=0.5 placeholder=0.5 min=0.5 max=1.0 step="any" id="AffinityPropagationDamping" name="AffinityPropagationDamping">
                                        <div><br></div>
                                        <label for="AffinityPropagationMaxIter">Max Iterations (int >=1):</label>
                                        <input type="number" step=1 value=200 placeholder=200 min=1 id="AffinityPropagationMaxIter" name="AffinityPropagationMaxIter">
                                        <div><br></div>
                                        <label for="AffinityPropagationConvergenceIter">Convergence Iter (int >=1):</label>
                                        <input type="number" step=1 value=15 placeholder=15 min=1 id="AffinityPropagationConvergenceIter" name="AffinityPropagationConvergenceIter">
                                        <div><br></div>
                                        <label for="AffinityPropagationCopy">Copy:</label>
                                        <select name="AffinityPropagationCopy" id="AffinityPropagationCopy">
                                            <option value="true">True</option>
                                            <option value="false">False</option>
                                        </select>
                                        <div><br></div>
                                        <label for="AffinityPropagationPreference">Preference (float or None):</label>
                                        <input type="number" step="any" id="AffinityPropagationPreference" name="AffinityPropagationPreference">
                                        <div><br></div>
                                        <label for="AffinityPropagationAffinity">Affinity:</label>
                                        <select name="AffinityPropagationAffinity" id="AffinityPropagationAffinity">
                                            <option value="euclidean">Euclidean</option>
                                            <option value="precomputed">Precomputed</option>
                                        </select>
                                        <div><br></div>
                                        <label for="AffinityPropagationVerbose">Verbose:</label>
                                        <select name="AffinityPropagationVerbose" id="AffinityPropagationVerbose">
                                            <option value="false">False</option>
                                            <option value="true">True</option>
                                        </select>
                                    </div>
                                </div>
                            </div>
                            
                            <!-- Bisecting K-Means -->
                            <div id="advancedBisectingKmeansFields" class="hidden">
                                <h3>Bisecting K-Means Settings</h3>
                                <p class="field-note" style="margin-bottom: 12px; margin-top: 0;">Hierarchical variant of K-Means that recursively splits clusters.</p>
                                <label for="advancedBisectingKmeansNClusters">N Clusters - Number of clusters (int >=2):</label>
                                <input type="number" step=1 value=8 placeholder=8 min=2 id="advancedBisectingKmeansNClusters" name="advancedBisectingKmeansNClusters">
                                <div class="nonreqHyperparams">
                                    <div><br></div>
                                    <div class="toggle-container">
                                        <span><h3>Edit Non-Essential Hyperparameters?</h3></span>
                                        <label class="switch">
                                            <input type="checkbox" id="advancedNonreqBisectingKmeansSlider">
                                            <span class="slider"></span>
                                        </label>
                                    </div>
                                    <div id="advancedNonreqBisectingKmeansParams" class="hidden">
                                        <label for="BisectingKmeansInit">Init:</label>
                                        <select name="BisectingKmeansInit" id="BisectingKmeansInit">
                                            <option value="k-means++">K-Means++</option>
                                            <option value="random">Random</option>
                                        </select>
                                        <div><br></div>
                                        <label for="BisectingKmeansNInit">N Init (int >=1):</label>
                                        <input type="number" step=1 value=1 placeholder=1 min=1 id="BisectingKmeansNInit" name="BisectingKmeansNInit">
                                        <div><br></div>
                                        <label for="BisectingKmeansMaxIter">Max Iterations (int >=1):</label>
                                        <input type="number" step=1 value=300 placeholder=300 min=1 id="BisectingKmeansMaxIter" name="BisectingKmeansMaxIter">
                                        <div><br></div>
                                        <label for="BisectingKmeansVerbose">Verbose (int):</label>
                                        <input type="number" step=1 value=0 placeholder=0 id="BisectingKmeansVerbose" name="BisectingKmeansVerbose">
                                        <div><br></div>
                                        <label for="BisectingKmeansTol">Tol (float >0):</label>
                                        <input type="number" value=0.0001 placeholder=0.0001 min="0.0000001" step="any" id="BisectingKmeansTol" name="BisectingKmeansTol">
                                        <div><br></div>
                                        <label for="BisectingKmeansCopyX">Copy X:</label>
                                        <select name="BisectingKmeansCopyX" id="BisectingKmeansCopyX">
                                            <option value="true">True</option>
                                            <option value="false">False</option>
                                        </select>
                                        <div><br></div>
                                        <label for="BisectingKmeansAlgorithm">Algorithm:</label>
                                        <select name="BisectingKmeansAlgorithm" id="BisectingKmeansAlgorithm">
                                            <option value="lloyd">Lloyd</option>
                                            <option value="elkan">Elkan</option>
                                        </select>
                                        <div><br></div>
                                        <label for="BisectingKmeansBisectingStrategy">Bisecting Strategy:</label>
                                        <select name="BisectingKmeansBisectingStrategy" id="BisectingKmeansBisectingStrategy">
                                            <option value="biggest_inertia">Biggest Inertia</option>
                                            <option value="largest_cluster">Largest Cluster</option>
                                        </select>
                                    </div>
                                </div>
                            </div>
                            
                            <!-- HDBSCAN -->
                            <div id="advancedHDBSCANFields" class="hidden">
                                <h3>HDBSCAN Settings</h3>
                                <p class="field-note" style="margin-bottom: 12px; margin-top: 0;">Hierarchical density-based clustering. Handles clusters of varying densities and noise points.</p>
                                <label for="advancedHDBSCANMinClusterSize">Min Cluster Size (int >=2):</label>
                                <input type="number" step=1 value=5 placeholder=5 min=2 id="advancedHDBSCANMinClusterSize" name="advancedHDBSCANMinClusterSize">
                                <div class="nonreqHyperparams">
                                    <div><br></div>
                                    <div class="toggle-container">
                                        <span><h3>Edit Non-Essential Hyperparameters?</h3></span>
                                        <label class="switch">
                                            <input type="checkbox" id="advancedNonreqHDBSCANSlider">
                                            <span class="slider"></span>
                                        </label>
                                    </div>
                                    <div id="advancedNonreqHDBSCANParams" class="hidden">
                                        <label for="HDBSCANMinSamples">Min Samples (int or None):</label>
                                        <input type="number" step=1 id="HDBSCANMinSamples" name="HDBSCANMinSamples">
                                        <div><br></div>
                                        <label for="HDBSCANClusterSelectionEpsilon">Cluster Selection Epsilon (float >=0):</label>
                                        <input type="number" value=0.0 placeholder=0.0 min=0 step="any" id="HDBSCANClusterSelectionEpsilon" name="HDBSCANClusterSelectionEpsilon">
                                        <div><br></div>
                                        <label for="HDBSCANMaxClusterSize">Max Cluster Size (int or None):</label>
                                        <input type="number" step=1 id="HDBSCANMaxClusterSize" name="HDBSCANMaxClusterSize">
                                        <div><br></div>
                                        <label for="HDBSCANMetric">Metric:</label>
                                        <input type="text" value="euclidean" placeholder="euclidean" id="HDBSCANMetric" name="HDBSCANMetric">
                                        <div><br></div>
                                        <label for="HDBSCANAlpha">Alpha (float >0):</label>
                                        <input type="number" value=1.0 placeholder=1.0 min="0.0000001" step="any" id="HDBSCANAlpha" name="HDBSCANAlpha">
                                        <div><br></div>
                                        <label for="HDBSCANAlgorithm">Algorithm:</label>
                                        <select name="HDBSCANAlgorithm" id="HDBSCANAlgorithm">
                                            <option value="auto">Auto</option>
                                            <option value="generic">Generic</option>
                                            <option value="prims_mst">Prims MST</option>
                                            <option value="boruvka_mst">Boruvka MST</option>
                                        </select>
                                        <div><br></div>
                                        <label for="HDBSCANLeafSize">Leaf Size (int >=1):</label>
                                        <input type="number" step=1 value=40 placeholder=40 min=1 id="HDBSCANLeafSize" name="HDBSCANLeafSize">
                                        <div><br></div>
                                        <label for="HDBSCANClusterSelectionMethod">Cluster Selection Method:</label>
                                        <select name="HDBSCANClusterSelectionMethod" id="HDBSCANClusterSelectionMethod">
                                            <option value="eom">EOM</option>
                                            <option value="leaf">Leaf</option>
                                        </select>
                                        <div><br></div>
                                        <label for="HDBSCANAllowSingleCluster">Allow Single Cluster:</label>
                                        <select name="HDBSCANAllowSingleCluster" id="HDBSCANAllowSingleCluster">
                                            <option value="false">False</option>
                                            <option value="true">True</option>
                                        </select>
                                        <div><br></div>
                                        <label for="HDBSCANCopy">Copy:</label>
                                        <select name="HDBSCANCopy" id="HDBSCANCopy">
                                            <option value="false">False</option>
                                            <option value="true">True</option>
                                        </select>
                                        <div><br></div>
                                        <label for="HDBSCANNJobs">N Jobs (int or None):</label>
                                        <input type="number" step=1 id="HDBSCANNJobs" name="HDBSCANNJobs">
                                    </div>
                                </div>
                            </div>
                            
                            <!-- Mean Shift -->
                            <div id="advancedMeanshiftFields" class="hidden">
                                <h3>Mean Shift Settings</h3>
                                <p class="field-note" style="margin-bottom: 12px; margin-top: 0;">Centroid-based clustering that finds modes in data density. Automatically determines number of clusters.</p>
                                <div class="nonreqHyperparams">
                                    <div class="toggle-container">
                                        <span><h3>Edit Hyperparameters?</h3></span>
                                        <label class="switch">
                                            <input type="checkbox" id="advancedNonreqMeanshiftSlider">
                                            <span class="slider"></span>
                                        </label>
                                    </div>
                                    <div id="advancedNonreqMeanshiftParams" class="hidden">
                                        <label for="MeanshiftBandwidth">Bandwidth (float or None):</label>
                                        <input type="number" step="any" id="MeanshiftBandwidth" name="MeanshiftBandwidth">
                                        <div><br></div>
                                        <label for="MeanshiftSeeds">Seeds (array-like or None):</label>
                                        <input type="text" id="MeanshiftSeeds" name="MeanshiftSeeds">
                                        <div><br></div>
                                        <label for="MeanshiftBinSeeding">Bin Seeding:</label>
                                        <select name="MeanshiftBinSeeding" id="MeanshiftBinSeeding">
                                            <option value="false">False</option>
                                            <option value="true">True</option>
                                        </select>
                                        <div><br></div>
                                        <label for="MeanshiftMinBinFreq">Min Bin Freq (int >=1):</label>
                                        <input type="number" step=1 value=1 placeholder=1 min=1 id="MeanshiftMinBinFreq" name="MeanshiftMinBinFreq">
                                        <div><br></div>
                                        <label for="MeanshiftClusterAll">Cluster All:</label>
                                        <select name="MeanshiftClusterAll" id="MeanshiftClusterAll">
                                            <option value="true">True</option>
                                            <option value="false">False</option>
                                        </select>
                                        <div><br></div>
                                        <label for="MeanshiftNJobs">N Jobs (int or None):</label>
                                        <input type="number" step=1 id="MeanshiftNJobs" name="MeanshiftNJobs">
                                        <div><br></div>
                                        <label for="MeanshiftMaxIter">Max Iterations (int >=1):</label>
                                        <input type="number" step=1 value=300 placeholder=300 min=1 id="MeanshiftMaxIter" name="MeanshiftMaxIter">
                                    </div>
                                </div>
                            </div>
                            
                            <!-- Mini-Batch K-Means -->
                            <div id="advancedMinibatchKmeansFields" class="hidden">
                                <h3>Mini-Batch K-Means Settings</h3>
                                <p class="field-note" style="margin-bottom: 12px; margin-top: 0;">Faster variant of K-Means using mini-batches. Good for large datasets.</p>
                                <label for="advancedMinibatchKmeansNClusters">N Clusters - Number of clusters (int >=2):</label>
                                <input type="number" step=1 value=8 placeholder=8 min=2 id="advancedMinibatchKmeansNClusters" name="advancedMinibatchKmeansNClusters">
                                <div class="nonreqHyperparams">
                                    <div><br></div>
                                    <div class="toggle-container">
                                        <span><h3>Edit Non-Essential Hyperparameters?</h3></span>
                                        <label class="switch">
                                            <input type="checkbox" id="advancedNonreqMinibatchKmeansSlider">
                                            <span class="slider"></span>
                                        </label>
                                    </div>
                                    <div id="advancedNonreqMinibatchKmeansParams" class="hidden">
                                        <label for="MinibatchKmeansInit">Init:</label>
                                        <select name="MinibatchKmeansInit" id="MinibatchKmeansInit">
                                            <option value="k-means++">K-Means++</option>
                                            <option value="random">Random</option>
                                        </select>
                                        <div><br></div>
                                        <label for="MinibatchKmeansMaxIter">Max Iterations (int >=1):</label>
                                        <input type="number" step=1 value=100 placeholder=100 min=1 id="MinibatchKmeansMaxIter" name="MinibatchKmeansMaxIter">
                                        <div><br></div>
                                        <label for="MinibatchKmeansBatchSize">Batch Size (int >=1):</label>
                                        <input type="number" step=1 value=1024 placeholder=1024 min=1 id="MinibatchKmeansBatchSize" name="MinibatchKmeansBatchSize">
                                        <div><br></div>
                                        <label for="MinibatchKmeansVerbose">Verbose (int):</label>
                                        <input type="number" step=1 value=0 placeholder=0 id="MinibatchKmeansVerbose" name="MinibatchKmeansVerbose">
                                        <div><br></div>
                                        <label for="MinibatchKmeansComputeLabels">Compute Labels:</label>
                                        <select name="MinibatchKmeansComputeLabels" id="MinibatchKmeansComputeLabels">
                                            <option value="true">True</option>
                                            <option value="false">False</option>
                                        </select>
                                        <div><br></div>
                                        <label for="MinibatchKmeansTol">Tol (float >=0):</label>
                                        <input type="number" value=0.0 placeholder=0.0 min=0 step="any" id="MinibatchKmeansTol" name="MinibatchKmeansTol">
                                        <div><br></div>
                                        <label for="MinibatchKmeansMaxNoImprovement">Max No Improvement (int >=1):</label>
                                        <input type="number" step=1 value=10 placeholder=10 min=1 id="MinibatchKmeansMaxNoImprovement" name="MinibatchKmeansMaxNoImprovement">
                                        <div><br></div>
                                        <label for="MinibatchKmeansNInit">N Init (int >=1):</label>
                                        <input type="number" step=1 value=3 placeholder=3 min=1 id="MinibatchKmeansNInit" name="MinibatchKmeansNInit">
                                        <div><br></div>
                                        <label for="MinibatchKmeansReassignmentRatio">Reassignment Ratio (float [0,1]):</label>
                                        <input type="number" value=0.01 placeholder=0.01 min=0 max=1 step="any" id="MinibatchKmeansReassignmentRatio" name="MinibatchKmeansReassignmentRatio">
                                    </div>
                                </div>
                            </div>
                            
                            <!-- OPTICS -->
                            <div id="advancedOPTICSFields" class="hidden">
                                <h3>OPTICS Settings</h3>
                                <p class="field-note" style="margin-bottom: 12px; margin-top: 0;">Ordering points to identify clustering structure. Similar to DBSCAN but handles varying densities better.</p>
                                <label for="advancedOPTICSMinSamples">Min Samples (int >=1):</label>
                                <input type="number" step=1 value=5 placeholder=5 min=1 id="advancedOPTICSMinSamples" name="advancedOPTICSMinSamples">
                                <div class="nonreqHyperparams">
                                    <div><br></div>
                                    <div class="toggle-container">
                                        <span><h3>Edit Non-Essential Hyperparameters?</h3></span>
                                        <label class="switch">
                                            <input type="checkbox" id="advancedNonreqOPTICSSlider">
                                            <span class="slider"></span>
                                        </label>
                                    </div>
                                    <div id="advancedNonreqOPTICSParams" class="hidden">
                                        <label for="OPTICSMaxEps">Max Eps (float >0 or inf):</label>
                                        <input type="number" step="any" id="OPTICSMaxEps" name="OPTICSMaxEps">
                                        <div><br></div>
                                        <label for="OPTICSMetric">Metric:</label>
                                        <input type="text" value="minkowski" placeholder="minkowski" id="OPTICSMetric" name="OPTICSMetric">
                                        <div><br></div>
                                        <label for="OPTICSP">P - Minkowski metric power (float >=1):</label>
                                        <input type="number" value=2 placeholder=2 min=1 step="any" id="OPTICSP" name="OPTICSP">
                                        <div><br></div>
                                        <label for="OPTICSClusterMethod">Cluster Method:</label>
                                        <select name="OPTICSClusterMethod" id="OPTICSClusterMethod">
                                            <option value="xi">Xi</option>
                                            <option value="dbscan">DBSCAN</option>
                                        </select>
                                        <div><br></div>
                                        <label for="OPTICSEps">Eps (float or None):</label>
                                        <input type="number" step="any" id="OPTICSEps" name="OPTICSEps">
                                        <div><br></div>
                                        <label for="OPTICSXi">Xi (float (0,1)):</label>
                                        <input type="number" value=0.05 placeholder=0.05 min="0.0000001" max=0.999999 step="any" id="OPTICSXi" name="OPTICSXi">
                                        <div><br></div>
                                        <label for="OPTICSPredecessorCorrection">Predecessor Correction:</label>
                                        <select name="OPTICSPredecessorCorrection" id="OPTICSPredecessorCorrection">
                                            <option value="true">True</option>
                                            <option value="false">False</option>
                                        </select>
                                        <div><br></div>
                                        <label for="OPTICSMinClusterSize">Min Cluster Size (int or None):</label>
                                        <input type="number" step=1 id="OPTICSMinClusterSize" name="OPTICSMinClusterSize">
                                        <div><br></div>
                                        <label for="OPTICSAlgorithm">Algorithm:</label>
                                        <select name="OPTICSAlgorithm" id="OPTICSAlgorithm">
                                            <option value="auto">Auto</option>
                                            <option value="ball_tree">Ball Tree</option>
                                            <option value="kd_tree">KD Tree</option>
                                            <option value="brute">Brute Force</option>
                                        </select>
                                        <div><br></div>
                                        <label for="OPTICSLeafSize">Leaf Size (int >=1):</label>
                                        <input type="number" step=1 value=30 placeholder=30 min=1 id="OPTICSLeafSize" name="OPTICSLeafSize">
                                        <div><br></div>
                                        <label for="OPTICSNJobs">N Jobs (int or None):</label>
                                        <input type="number" step=1 id="OPTICSNJobs" name="OPTICSNJobs">
                                    </div>
                                </div>
                            </div>
                        </div>

                        <div class="preprocess-card">
                            <h2>Advanced Options</h2>
                            
                            <h3 style="margin-top: 24px; margin-bottom: 12px; padding-top: 16px; border-top: 1px solid #e5e5e5; font-size: 1.1em; font-weight: 600; color: #000000;">Feature Selection</h3>
                            <p class="field-note" style="margin-bottom: 12px; margin-top: 0;">Automatically select the most important features to improve model performance and reduce overfitting.</p>
                            <div class="scaling-container">
                                <label for="featureSelectionMethod">Feature Selection Method</label>
                                <select name="featureSelectionMethod" id="featureSelectionMethod">
                                    <option value="none">None</option>
                                    <option value="SelectKBest">Select K Best (Filter)</option>
                                    <option value="RFE">Recursive Feature Elimination (Wrapper)</option>
                                    <option value="SelectFromModel">Select From Model (Model-based)</option>
                                </select>
                            </div>
                            <div class="scaling-container hidden" id="featureSelectionParams">
                                <label for="featureSelectionK">Number of Features (K)</label>
                                <input type="number" min="1" id="featureSelectionK" name="featureSelectionK" placeholder="10">
                            </div>

                            <h3 style="margin-top: 24px; margin-bottom: 12px; padding-top: 16px; border-top: 1px solid #e5e5e5; font-size: 1.1em; font-weight: 600; color: #000000;">Outlier Handling</h3>
                            <p class="field-note" style="margin-bottom: 12px; margin-top: 0;">Detect and handle outliers that may negatively impact model performance.</p>
                            <div class="scaling-container">
                                <label for="outlierMethod">Outlier Detection Method</label>
                                <select name="outlierMethod" id="outlierMethod">
                                    <option value="none">None</option>
                                    <option value="IQR">Interquartile Range (IQR)</option>
                                    <option value="IsolationForest">Isolation Forest</option>
                                    <option value="LocalOutlierFactor">Local Outlier Factor</option>
                                    <option value="ZScore">Z-Score (3σ rule)</option>
                                </select>
                            </div>
                            <div class="scaling-container hidden" id="outlierActionDiv">
                                <label for="outlierAction">Action</label>
                                <select name="outlierAction" id="outlierAction">
                                    <option value="remove">Remove outliers</option>
                                    <option value="cap">Cap at threshold</option>
                                </select>
                            </div>

                            <h3 style="margin-top: 24px; margin-bottom: 12px; padding-top: 16px; border-top: 1px solid #e5e5e5; font-size: 1.1em; font-weight: 600; color: #000000;">Hyperparameter Search</h3>
                            <p class="field-note" style="margin-bottom: 12px; margin-top: 0;">Automatically find optimal hyperparameter values. This can significantly increase processing time.</p>
                            <div class="scaling-container">
                                <label for="hyperparameterSearch">Enable Hyperparameter Search</label>
                                <select name="hyperparameterSearch" id="hyperparameterSearch">
                                    <option value="none">None (use manual settings)</option>
                                    <option value="grid">Grid Search</option>
                                    <option value="randomized">Randomized Search</option>
                                    <option value="bayesian">Bayesian Optimization</option>
                                </select>
                            </div>
                            <div class="scaling-container hidden" id="hyperparameterSearchParams">
                                <label for="searchCVFolds">CV Folds for Search</label>
                                <input type="number" min="2" max="10" value="5" id="searchCVFolds" name="searchCVFolds">
                                <label for="searchNIter">Number of Iterations (Randomized/Bayesian)</label>
                                <input type="number" min="10" max="100" value="50" id="searchNIter" name="searchNIter">
                                <p class="field-note">Parameter grids are automatically generated based on selected model</p>
                            </div>

                            <h3 style="margin-top: 24px; margin-bottom: 12px; padding-top: 16px; border-top: 1px solid #e5e5e5; font-size: 1.1em; font-weight: 600; color: #000000;">Cross-Validation</h3>
                            <p class="field-note" style="margin-bottom: 12px; margin-top: 0;">Evaluate model performance across multiple data splits for more robust performance estimates.</p>
                            <div class="scaling-container">
                                <label for="advancedCrossValidationType">Cross-Validation Type</label>
                                <select name="advancedCrossValidationType" id="advancedCrossValidationType">
                                    <option value="None">None</option>
                                    <option value="KFold">K-Fold</option>
                                    <option value="StratifiedKFold">Stratified K-Fold</option>
                                    <option value="RepeatedKFold">Repeated K-Fold</option>
                                    <option value="RepeatedStratifiedKFold">Repeated Stratified K-Fold</option>
                                    <option value="ShuffleSplit">Shuffle Split</option>
                                    <option value="StratifiedShuffleSplit">Stratified Shuffle Split</option>
                                </select>
                            </div>
                            <div class="scaling-container">
                                <label for="advancedCrossValidationFolds">Number of Folds (2-100)</label>
                                <input type="number" value="5" min="2" max="100" id="advancedCrossValidationFolds" name="advancedCrossValidationFolds" placeholder="2-100">
                            </div>
                        </div>

                        <div style="margin-top: 20px; padding: 16px; background-color: #f8f9fa; border-radius: 8px; border: 1px solid #e0e0e0;">
                            <p class="field-note" style="margin-bottom: 12px;">
                                <strong>Note:</strong> Select your model above and configure all hyperparameters here. Then configure advanced options (feature selection, outlier handling, hyperparameter search, cross-validation). Click the button below to run the model with these advanced settings.
                            </p>
                            <button type="submit" class='processButton processButton--compact' id="advancedOptimizationSubmitButton">Run Model with Advanced Options</button>
                            <p class="field-note" style="color: #856404; margin-top: 12px; margin-bottom: 0;">
                                <strong>Warning:</strong> Advanced options such as hyperparameter search, feature selection, and extensive cross-validation can take multiple hours to complete, especially with large datasets.
                            </p>
                        </div>
                    </div>

                    <!-- Results display area on the right side -->
                    <div class="visualSection">
                        <div class="model-card">
                            <div id="AdvancedNumericResultDiv" role="region" aria-label="Advanced regression model results">
                                <div id="advancedImageSelector"></div>
                            </div>
                            <div id="AdvancedClusterResultDiv" role="region" aria-label="Advanced clustering model results"></div>
                            <div id="AdvancedClassifierResultDiv" role="region" aria-label="Advanced classification model results"></div>
                        </div>
                    </div>
                </div>
            </form>
            
            <!-- Loading bars positioned below graphics, aligned with the left column (button area) -->
            <div id="advancedLoading" class="hidden" style="margin-top: 20px; max-width: 35%;" role="status" aria-live="polite" aria-atomic="true" aria-label="Advanced modeling loading status">
            </div>
        </div>
    </div>

    <div id="predictionDiv" class="hidden">
        <div class="section-header">
            <div class="section-header-content">
                <h2>Inference on New Data</h2>
                <button class="secondary-button" onclick="backToModel()">Back To Model</button>
            </div>
        </div>
        <form id="uploadPredictDf" enctype="multipart/form-data">
            <h3>Upload New Data CSV File to Predict With</h3>
            <input type="file" name="predictFile" id="predictFile" accept=".csv" required>
            <div id="uploadPredictDFButton">
                <button type="submit">Upload</button>
            </div>
        </form>

        <div id="predictionResults" class="hidden">
        </div>
        <div id="predictionErrorDiv" class="hidden"></div>
    </div>


</div>


    </main>
    <script src='/static/client_side.js'></script> 
</body>
</html>
