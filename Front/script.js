async function checkPlagiarism() {
    const text = document.getElementById('textInput').value;
    const fileInput = document.getElementById('fileInput').files[0];
    const uploadStatus = document.getElementById('uploadStatus');
    const scoreP = document.getElementById('score');
    const analysisP = document.getElementById('analysis');
    const perplexityP = document.getElementById('perplexity');
    const burstinessP = document.getElementById('burstiness');

    try {
        uploadStatus.style.display = 'block';
        
        if (!text && !fileInput) {
            throw new Error('Please enter text or upload a file');
        }

        let dataToSend = { text: text };

        if (fileInput) {
            const formData = new FormData();
            formData.append('file', fileInput);
            
            try {
                uploadStatus.innerHTML = 'Uploading file...';
                uploadStatus.style.color = '#007bff';
                
                const response = await fetchWithRetry('http://localhost:5000/upload', {
                    method: 'POST',
                    body: formData,
                    mode: 'cors'
                });
                
                const data = await response.json();
                
                if (!response.ok) {
                    throw new Error(data.error || `HTTP ${response.status}: Upload failed`);
                }
                
                if (!data.text || data.text.trim() === '') {
                    throw new Error('No text could be extracted from file');
                }
                
                console.log(`Extracted ${data.text.length} characters from file`);
                dataToSend.text = data.text;  // Use the extracted text
                uploadStatus.innerHTML = `<strong>File uploaded:</strong> ${data.filename} (${data.characters || data.text.length} characters)`;
                uploadStatus.style.color = '#28a745';
            } catch (uploadError) {
                console.error('Upload error:', uploadError);
                uploadStatus.innerHTML = `Upload failed: ${uploadError.message}`;
                uploadStatus.style.color = '#dc3545';
                throw new Error(`Upload failed: ${uploadError.message}`);
            }
        }

        // Only proceed if we have text to analyze
        if (!dataToSend.text || !dataToSend.text.trim()) {
            throw new Error('No text available for analysis');
        }

        // Show loading states
        perplexityP.textContent = 'Calculating...';
        burstinessP.textContent = 'Calculating...';
        analysisP.textContent = 'Analyzing...';
        scoreP.style.display = 'none';

        try {
            // Run both analyses in parallel with retry logic
            const [plagiarismResult, streamlitResult] = await Promise.all([
                fetchAnalysis('http://localhost:5000/check', dataToSend),
                fetchAnalysis('http://localhost:5000/streamlit-analysis', dataToSend)
            ]);

            // Combine results
            const combinedResults = {
                ...plagiarismResult,
                streamlit: streamlitResult
            };

            // Update UI with results
            updateResults(combinedResults);

        } catch (error) {
            console.error('Analysis error:', error);
            analysisP.textContent = `Error: ${error.message}`;
            perplexityP.textContent = 'Analysis failed';
            burstinessP.textContent = 'Analysis failed';
        }
    } catch (error) {
        console.error('Error:', error);
        analysisP.textContent = error.message;
        perplexityP.textContent = 'Analysis failed';
        burstinessP.textContent = 'Analysis failed';
        uploadStatus.textContent = `Error: ${error.message}`;
        uploadStatus.style.color = '#dc3545';
    }
}

// Helper function to fetch with retry logic
async function fetchWithRetry(url, options, maxRetries = 3) {
    for (let i = 0; i < maxRetries; i++) {
        try {
            const response = await fetch(url, {
                ...options,
                headers: {
                    'Accept': 'application/json',
                    ...options.headers
                }
            });
            
            if (response.ok || response.status < 500) {
                return response;
            }
            
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        } catch (error) {
            console.warn(`Attempt ${i + 1} failed:`, error.message);
            
            if (i === maxRetries - 1) {
                throw error;
            }
            
            // Wait before retry (exponential backoff)
            await new Promise(resolve => setTimeout(resolve, Math.pow(2, i) * 1000));
        }
    }
}

// Helper function for analysis requests
async function fetchAnalysis(url, data) {
    const response = await fetchWithRetry(url, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        },
        body: JSON.stringify(data),
        mode: 'cors'
    });

    if (!response.ok) {
        throw new Error(`Analysis request failed: HTTP ${response.status}`);
    }

    return await response.json();
}

async function runStreamlitAnalysis(text) {
    try {
        const response = await fetch('http://localhost:5000/streamlit-analysis', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            credentials: 'include',
            body: JSON.stringify({ text: text })
        });

        if (!response.ok) {
            throw new Error('Streamlit analysis failed');
        }

        const result = await response.json();
        
        // Update the plot if it exists
        const plotContainer = document.getElementById('wordFrequencyPlot');
        if (plotContainer) {
            // Create frequency plot
            const trace = {
                x: result.word_frequency.words,
                y: result.word_frequency.counts,
                type: 'bar'
            };
            Plotly.newPlot(plotContainer, [trace]);
        }

        return result;
    } catch (error) {
        console.error('Streamlit analysis error:', error);
        throw error;
    }
}

function createAnalysisChart(data) {
    const ctx = document.getElementById('analysisChart').getContext('2d');
    return new Chart(ctx, {
        type: 'radar',
        data: {
            labels: ['Originality', 'Coherence', 'Complexity', 'Variability', 'Style Consistency'],
            datasets: [{
                label: 'Text Analysis',
                data: [
                    100 - (data.perplexity / 100),
                    data.burstiness * 100,
                    data.complexity || 50,
                    data.variability || 60,
                    data.styleConsistency || 75
                ],
                backgroundColor: 'rgba(54, 162, 235, 0.2)',
                borderColor: 'rgba(54, 162, 235, 1)',
                borderWidth: 2
            }]
        },
        options: {
            scales: {
                r: {
                    beginAtZero: true,
                    max: 100
                }
            }
        }
    });
}

function generateSuggestions(result) {
    const suggestions = document.getElementById('suggestions');
    if (!suggestions) return;
    
    suggestions.innerHTML = '';
    
    const addSuggestion = (text, type) => {
        const div = document.createElement('div');
        div.className = `suggestion ${type}`;
        div.textContent = text;
        suggestions.appendChild(div);
    };

    if (result.perplexity < 60.0) {
        addSuggestion('Consider revising for more natural language variation', 'warning');
    }
    
    if (result.burstiness < 0.1) {
        addSuggestion('Try using more diverse vocabulary', 'warning');
    }
    
    // Add more context-specific suggestions
    if (result.style_consistency && result.style_consistency < 0.7) {
        addSuggestion('Writing style appears inconsistent', 'info');
    }
}

// Update existing updateResults function
async function updateResults(result) {
    const elements = {
        score: document.getElementById('score'),
        analysis: document.getElementById('analysis'),
        perplexity: document.getElementById('perplexity'),
        burstiness: document.getElementById('burstiness')
    };

    // Update metrics with detailed analysis
    elements.perplexity.textContent = `Perplexity: ${result.perplexity.toFixed(2)} (${result.analysis?.perplexity_analysis || 'Analysis complete'})`;
    elements.burstiness.textContent = `Burstiness: ${result.burstiness.toFixed(4)} (${result.analysis?.burstiness_analysis || 'Analysis complete'})`;
    elements.analysis.textContent = result.analysis?.overall || (result.is_ai_generated ? 'Likely AI-generated' : 'Likely human-written');
    
    // Color coding based on AI detection
    const color = result.is_ai_generated ? '#dc3545' : '#28a745';
    elements.analysis.style.color = color;
    
    // Create analysis chart if element exists
    const chartElement = document.getElementById('analysisChart');
    if (chartElement) {
        createAnalysisChart(result);
    }
    
    // Generate suggestions
    generateSuggestions(result);
    
    // Update word frequency plot if streamlit data is available
    if (result.streamlit && result.streamlit.word_frequency) {
        const plotContainer = document.getElementById('wordFrequencyPlot');
        if (plotContainer) {
            const trace = {
                x: result.streamlit.word_frequency.words,
                y: result.streamlit.word_frequency.counts,
                type: 'bar',
                marker: {
                    color: '#007bff'
                }
            };
            Plotly.newPlot(plotContainer, [trace], {
                title: 'Word Frequency Analysis',
                xaxis: { title: 'Words' },
                yaxis: { title: 'Frequency' }
            });
        }
    }
}

function animateValue(element, start, end, duration, suffix = '') {
    const range = end - start;
    const increment = end > start ? 1 : -1;
    const stepTime = Math.abs(Math.floor(duration / range));
    let current = start;
    
    const timer = setInterval(() => {
        current += increment;
        element.textContent = current + suffix;
        if (current === end) {
            clearInterval(timer);
        }
    }, stepTime);
}

// Add file input change handler
document.addEventListener('DOMContentLoaded', function() {
    const fileInput = document.getElementById('fileInput');
    if (fileInput) {
        fileInput.addEventListener('change', function(e) {
            const uploadStatus = document.getElementById('uploadStatus');
            if (this.files.length > 0) {
                uploadStatus.textContent = `Selected file: ${this.files[0].name}`;
                uploadStatus.style.display = 'block';
                uploadStatus.style.color = '#007bff';
            }
        });
    }
    
    // Restore file name on page load
    const currentFile = localStorage.getItem('currentFile');
    if (currentFile) {
        const uploadStatus = document.getElementById('uploadStatus');
        if (uploadStatus) {
            uploadStatus.innerHTML = `<strong>Current File:</strong> ${currentFile}`;
            uploadStatus.style.display = 'block';
        }
    }
});