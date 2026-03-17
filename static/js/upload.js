/**
 * Upload and Data Exploration (correlation matrices, pairplot). Depends on core.js (refs, helpers).
 */
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
        showError(errorDiv, error.message || 'Request failed. See console for details.');
    }

});

