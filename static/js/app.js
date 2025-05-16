// static/js/app.js
document.addEventListener('DOMContentLoaded', function () {
    // DOM elements - Forms and UI
    const analysisForm = document.getElementById('analysisForm');
    const loadingSpinner = document.getElementById('loadingSpinner');
    const errorMessageDiv = document.getElementById('errorMessage');
    const lookbackSlider = document.getElementById('lookback');
    const numClustersSlider = document.getElementById('numClusters');
    const themeToggle = document.getElementById('checkbox');
    const lookbackBtns = document.querySelectorAll('.btn-group .btn[data-hour]');
    const incrementClustersBtn = document.getElementById('incrementClusters');
    const decrementClustersBtn = document.getElementById('decrementClusters');
    const downloadClusterBtn = document.getElementById('downloadClusterBtn');
    const downloadForecastBtn = document.getElementById('downloadForecastBtn');

    // DOM elements - Plot containers
    const clusterPlot = document.getElementById('clusterPlot');
    const forecastPlot = document.getElementById('forecastPlot');

    // DOM elements - Results display
    const silhouetteScoreEl = document.getElementById('silhouetteScore');
    const clusterFeaturesUsedEl = document.getElementById('clusterFeaturesUsed');
    const clusterCentersEl = document.getElementById('clusterCenters');
    
    const forecastMetricsContainer = document.getElementById('forecastMetricsContainer');
    const forecastModelUsedEl = document.getElementById('forecastModelUsed');
    const forecastMAEl = document.getElementById('forecastMAE');
    const forecastRMSEl = document.getElementById('forecastRMSE');
    const forecastR2l = document.getElementById('forecastR2');
    const forecastMAPEl = document.getElementById('forecastMAPE');
    const forecastFeaturesUsedEl = document.getElementById('forecastFeaturesUsed');

    // Theme Management
    function setTheme(isDark) {
        if (isDark) {
            document.body.classList.remove('light-mode');
            // Set Plotly dark theme config for future chart updates
            window.plotlyConfig = {
                colorway: ['#00b4d8', '#06d6a0', '#ffd166', '#ef476f', '#c77dff', '#118ab2'],
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                font: { color: '#e5e5e5' },
                grid: { color: 'rgba(255,255,255,0.1)' }
            };
        } else {
            document.body.classList.add('light-mode');
            // Set Plotly light theme config
            window.plotlyConfig = {
                colorway: ['#0096c7', '#06b582', '#e9c46a', '#dd2e44', '#9d4edd', '#0a7999'],
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                font: { color: '#212529' },
                grid: { color: 'rgba(0,0,0,0.1)' }
            };
        }
        
        // Re-initialize plots with the new theme
        if (window.clusterData) {
            updateClusterPlot(window.clusterData);
        }
        if (window.forecastTraces) {
            updateForecastPlot(window.forecastTraces);
        }
    }
    
    // Initialize theme based on user preference or default to dark
    if (localStorage.getItem('theme') === 'light') {
        themeToggle.checked = true;
        setTheme(false);
    } else {
        setTheme(true);
    }
    
    themeToggle.addEventListener('change', function() {
        if (this.checked) {
            localStorage.setItem('theme', 'light');
            setTheme(false);
        } else {
            localStorage.setItem('theme', 'dark');
            setTheme(true);
        }
    });

    // Quick set buttons for lookback hours
    lookbackBtns.forEach(btn => {
        btn.addEventListener('click', function() {
            const hours = parseInt(this.dataset.hour);
            lookbackSlider.value = hours;
            document.getElementById('lookbackValue').textContent = hours;
        });
    });
    
    // Increment/decrement buttons for clusters
    incrementClustersBtn.addEventListener('click', function() {
        let current = parseInt(numClustersSlider.value);
        if (current < parseInt(numClustersSlider.max)) {
            numClustersSlider.value = current + 1;
            document.getElementById('numClustersValue').textContent = current + 1;
        }
    });
    
    decrementClustersBtn.addEventListener('click', function() {
        let current = parseInt(numClustersSlider.value);
        if (current > parseInt(numClustersSlider.min)) {
            numClustersSlider.value = current - 1;
            document.getElementById('numClustersValue').textContent = current - 1;
        }
    });
    
    // Download buttons for charts
    downloadClusterBtn.addEventListener('click', function() {
        if (clusterPlot && clusterPlot._fullLayout) {
            Plotly.downloadImage(clusterPlot, {
                format: 'png',
                filename: 'cluster_visualization',
                height: 800,
                width: 1200
            });
        }
    });
    
    downloadForecastBtn.addEventListener('click', function() {
        if (forecastPlot && forecastPlot._fullLayout) {
            Plotly.downloadImage(forecastPlot, {
                format: 'png',
                filename: 'demand_forecast',
                height: 800,
                width: 1200
            });
        }
    });

    // Initialize plots with empty state
    const defaultLayout = {
        margin: { t: 30, b: 50, l: 60, r: 30 },
        autosize: true,
        hovermode: 'closest',
        ...window.plotlyConfig
    };
    
    Plotly.newPlot(clusterPlot, [], {
        ...defaultLayout,
        title: { text: 'Cluster Analysis (PCA)', font: { size: 16 } },
        xaxis: { title: 'Principal Component 1' },
        yaxis: { title: 'Principal Component 2' }
    });
    
    Plotly.newPlot(forecastPlot, [], {
        ...defaultLayout,
        title: { text: 'Demand Forecast', font: { size: 16 } },
        xaxis: { title: 'Date' },
        yaxis: { title: 'Demand' }
    });

    // Form submission
    analysisForm.addEventListener('submit', async function (event) {
        event.preventDefault();
        
        // Show loading, hide errors
        loadingSpinner.classList.remove('d-none');
        errorMessageDiv.classList.add('d-none');
        errorMessageDiv.querySelector('.error-text').textContent = '';
        
        // Reset results display
        silhouetteScoreEl.textContent = '-';
        clusterFeaturesUsedEl.textContent = '-';
        clusterCentersEl.textContent = '-';
        forecastMetricsContainer.classList.add('d-none');
        
        // Clear charts
        Plotly.react(clusterPlot, [], {
            ...defaultLayout,
            title: { text: 'Cluster Analysis (PCA)', font: { size: 16 } },
            xaxis: { title: 'Principal Component 1' },
            yaxis: { title: 'Principal Component 2' }
        });
        
        Plotly.react(forecastPlot, [], {
            ...defaultLayout,
            title: { text: 'Demand Forecast', font: { size: 16 } },
            xaxis: { title: 'Date' },
            yaxis: { title: 'Demand' }
        });

        // Gather form data
        const formData = new FormData(analysisForm);
        const data = {
            city: formData.get('city'),
            start_date: formData.get('startDate'),
            end_date: formData.get('endDate'),
            lookback: parseInt(formData.get('lookback')),
            k: parseInt(formData.get('numClusters'))
        };

        const selectedModels = Array.from(document.querySelectorAll('.model-checkbox:checked')).map(cb => cb.value);

        try {
            // --- Call Cluster API ---
            const clusterResponse = await fetch('/api/cluster', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    city: data.city, 
                    start_date: data.start_date, 
                    end_date: data.end_date, 
                    k: data.k 
                })
            });

            if (!clusterResponse.ok) {
                const errorResult = await clusterResponse.json();
                throw new Error(`Cluster Analysis: ${errorResult.error || clusterResponse.statusText}`);
            }
            
            const clusterResult = await clusterResponse.json();
            window.clusterData = clusterResult; // Store for theme changes
            updateClusterPlot(clusterResult);

            // --- Call Forecast API for each selected model ---
            let forecastTraces = [];
            let actualDemandTrace = null;
            let firstModel = true;

            for (const model of selectedModels) {
                const forecastPayload = {
                    city: data.city,
                    start_date: data.start_date,
                    end_date: data.end_date,
                    lookback: data.lookback,
                    model: model
                };

                const forecastResponse = await fetch('/api/forecast', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(forecastPayload)
                });

                if (!forecastResponse.ok) {
                    const errorResult = await forecastResponse.json();
                    console.warn(`Forecast API error for model ${model}: ${errorResult.error || forecastResponse.statusText}`);
                    appendErrorMessage(`Forecast failed for ${model}: ${errorResult.error || forecastResponse.statusText}`);
                    continue; // Skip to next model
                }
                
                const forecastResult = await forecastResponse.json();
                
                // Add actual demand trace only once
                if (firstModel && forecastResult.actual && forecastResult.dates) {
                    actualDemandTrace = {
                        x: forecastResult.dates.map(d => new Date(d)),
                        y: forecastResult.actual,
                        mode: 'lines',
                        name: 'Actual Demand',
                        line: { 
                            color: window.plotlyConfig ? window.plotlyConfig.colorway[0] : '#1f77b4',
                            width: 3
                        }
                    };
                    forecastTraces.push(actualDemandTrace);
                }

                // Add the forecast trace with model name
                const modelColorIndex = firstModel ? 1 : forecastTraces.length % (window.plotlyConfig ? window.plotlyConfig.colorway.length : 6);
                forecastTraces.push({
                    x: forecastResult.dates.map(d => new Date(d)),
                    y: forecastResult.predicted,
                    mode: 'lines',
                    name: `Predicted (${model})`,
                    line: { 
                        color: window.plotlyConfig ? window.plotlyConfig.colorway[modelColorIndex] : undefined,
                        dash: model === 'ARIMA' ? 'dash' : 'solid' // Different line style for ARIMA
                    }
                });

                // Update metrics for the first successfully fetched model or a specific one
                if (firstModel || selectedModels.length === 1) { 
                    updateForecastMetrics(forecastResult);
                }
                firstModel = false;
            }

            // Save for theme changes
            window.forecastTraces = forecastTraces;
            
            if (forecastTraces.length > 0) {
                updateForecastPlot(forecastTraces);
            } else if (selectedModels.length > 0) {
                // No forecast data could be plotted, but models were selected
                Plotly.react(forecastPlot, [], {
                    ...defaultLayout,
                    title: { text: 'Demand Forecast - No data to display', font: { size: 16 } },
                    xaxis: { title: 'Date' },
                    yaxis: { title: 'Demand' }
                });
                appendErrorMessage("Could not retrieve forecast data for any selected model.");
            }

        } catch (error) {
            console.error('Error during analysis:', error);
            errorMessageDiv.querySelector('.error-text').textContent = error.message;
            errorMessageDiv.classList.remove('d-none');
        } finally {
            loadingSpinner.classList.add('d-none');
        }
    });

    function appendErrorMessage(message) {
        errorMessageDiv.classList.remove('d-none');
        const errorTextEl = errorMessageDiv.querySelector('.error-text');
        if (errorTextEl.textContent) {
            errorTextEl.textContent += '\n' + message;
        } else {
            errorTextEl.textContent = message;
        }
    }

    function updateClusterPlot(data) {
        if (!data.clusters || data.clusters.length === 0) {
            Plotly.react(clusterPlot, [], {
                ...defaultLayout,
                title: { text: 'Cluster Analysis - No data', font: { size: 16 } },
                xaxis: { title: 'Principal Component 1' },
                yaxis: { title: 'Principal Component 2' }
            });
            
            silhouetteScoreEl.textContent = data.silhouette_score !== undefined ? parseFloat(data.silhouette_score).toFixed(4) : 'N/A';
            clusterFeaturesUsedEl.textContent = data.features_used ? data.features_used.join(', ') : 'N/A';
            clusterCentersEl.textContent = 'No cluster data to display centers.';
            appendErrorMessage("No data points returned for clustering.");
            return;
        }

        const traces = [];
        const uniqueLabels = [...new Set(data.clusters.map(item => item.label))].sort((a,b) => a-b);
        const colorway = window.plotlyConfig ? window.plotlyConfig.colorway : ['#00b4d8', '#06d6a0', '#ffd166', '#ef476f', '#c77dff', '#118ab2'];

        uniqueLabels.forEach((label, idx) => {
            const points = data.clusters.filter(p => p.label === label);
            traces.push({
                x: points.map(p => p.x),
                y: points.map(p => p.y),
                mode: 'markers',
                type: 'scatter',
                name: `Cluster ${label}`,
                marker: { 
                    size: 8,
                    color: colorway[idx % colorway.length],
                    line: { width: 1, color: 'rgba(255, 255, 255, 0.3)' }
                },
                hovertemplate: 'PC1: %{x:.2f}<br>PC2: %{y:.2f}<br>Cluster: %{text}<extra></extra>',
                text: Array(points.length).fill(`Cluster ${label}`)
            });
        });

        const layout = {
            ...defaultLayout,
            title: { 
                text: 'Cluster Visualization (PCA)',
                font: { size: 16 }
            },
            xaxis: { 
                title: 'Principal Component 1',
                gridcolor: window.plotlyConfig ? window.plotlyConfig.grid.color : undefined,
                zerolinecolor: window.plotlyConfig ? window.plotlyConfig.grid.color : undefined
            },
            yaxis: { 
                title: 'Principal Component 2',
                gridcolor: window.plotlyConfig ? window.plotlyConfig.grid.color : undefined,
                zerolinecolor: window.plotlyConfig ? window.plotlyConfig.grid.color : undefined
            },
            showlegend: true,
            legend: { orientation: 'h', y: -0.2 }
        };
        
        Plotly.react(clusterPlot, traces, layout);

        // Update metrics display
        silhouetteScoreEl.textContent = parseFloat(data.silhouette_score).toFixed(4);
        clusterFeaturesUsedEl.textContent = data.features_used ? data.features_used.join(', ') : 'N/A';
        
        if (data.cluster_centers) {
            clusterCentersEl.textContent = JSON.stringify(data.cluster_centers, null, 2);
        } else {
            clusterCentersEl.textContent = 'N/A';
        }
    }

    function updateForecastPlot(traces) {
        const layout = {
            ...defaultLayout,
            title: { 
                text: 'Demand Forecast',
                font: { size: 16 }
            },
            xaxis: { 
                title: 'Date',
                type: 'date',
                gridcolor: window.plotlyConfig ? window.plotlyConfig.grid.color : undefined
            },
            yaxis: { 
                title: 'Demand',
                gridcolor: window.plotlyConfig ? window.plotlyConfig.grid.color : undefined
            },
            hovermode: 'x unified', // unified hover for better comparison
            showlegend: true,
            legend: { orientation: 'h', y: -0.2 }
        };
        
        Plotly.react(forecastPlot, traces, layout);
    }

    function updateForecastMetrics(data) {
        forecastMetricsContainer.classList.remove('d-none');
        forecastModelUsedEl.textContent = data.model_used || 'N/A';
        
        // Display raw metrics
        forecastMAEl.textContent = data.metrics && data.metrics.mae !== undefined ? 
            parseFloat(data.metrics.mae).toFixed(2) : 'N/A';
        forecastRMSEl.textContent = data.metrics && data.metrics.rmse !== undefined ? 
            parseFloat(data.metrics.rmse).toFixed(2) : 'N/A';
        forecastR2l.textContent = data.metrics && data.metrics.r2 !== undefined ? 
            parseFloat(data.metrics.r2).toFixed(4) : 'N/A';
        forecastMAPEl.textContent = data.metrics && data.metrics.mape !== undefined && data.metrics.mape !== 'N/A' ? 
            parseFloat(data.metrics.mape).toFixed(2) + '%' : 'N/A';
        
        // Add normalized metrics and performance score
        const metricsContainer = document.getElementById('forecastMetricsContainer');
        
        // Clear any existing performance metrics (in case of multiple runs)
        const existingPerformanceRow = document.getElementById('performanceMetricsRow');
        if (existingPerformanceRow) {
            existingPerformanceRow.remove();
        }
        
        // Create a new row for normalized metrics and performance score
        if (data.normalized_metrics) {
            const performanceRow = document.createElement('div');
            performanceRow.id = 'performanceMetricsRow';
            performanceRow.className = 'row mt-3';
            
            // Create normalized metrics display
            const normalizedHTML = `
                <div class="col-12 mb-2">
                    <h6>Performance Score</h6>
                </div>
                <div class="col-md-4 mb-2">
                    <div class="progress" style="height: 20px;" title="Overall model performance score">
                        <div class="progress-bar ${getScoreColorClass(data.performance.score)}" 
                             role="progressbar" 
                             style="width: ${data.performance.score * 100}%;" 
                             aria-valuenow="${data.performance.score * 100}" 
                             aria-valuemin="0" 
                             aria-valuemax="100">
                            ${(data.performance.score * 100).toFixed(0)}%
                        </div>
                    </div>
                </div>
                <div class="col-md-4 mb-2">
                    <span class="badge ${getScoreColorClass(data.performance.score)}">${data.performance.grade}</span>
                </div>
                <div class="col-md-4 mb-2">
                    <div class="d-flex flex-column">
                        <small class="text-muted">Normalized Metrics (0-1 scale, higher is better):</small>
                        <small>MAE: ${(data.normalized_metrics.mae * 100).toFixed(0)}%</small>
                        <small>RMSE: ${(data.normalized_metrics.rmse * 100).toFixed(0)}%</small>
                        <small>R²: ${(data.normalized_metrics.r2 * 100).toFixed(0)}%</small>
                        <small>MAPE: ${(data.normalized_metrics.mape * 100).toFixed(0)}%</small>
                    </div>
                </div>
            `;
            
            performanceRow.innerHTML = normalizedHTML;
            metricsContainer.appendChild(performanceRow);
        }
        
        forecastFeaturesUsedEl.textContent = data.features_used ? data.features_used.join(', ') : 'N/A';
    }

    // Helper function to get appropriate color class based on score
    function getScoreColorClass(score) {
        if (score >= 0.9) return 'bg-success';
        if (score >= 0.75) return 'bg-info';
        if (score >= 0.6) return 'bg-primary';
        if (score >= 0.4) return 'bg-warning';
        return 'bg-danger';
    }

    // Update slider value display dynamically
    lookbackSlider.addEventListener('input', function() {
        document.getElementById('lookbackValue').textContent = this.value;
    });
    
    numClustersSlider.addEventListener('input', function() {
        document.getElementById('numClustersValue').textContent = this.value;
    });
}); 