const ui = {
    textInput: document.getElementById('textInput'),
    fileInput: document.getElementById('fileInput'),
    uploadStatus: document.getElementById('uploadStatus'),
    charCounter: document.getElementById('charCounter'),
    analyzeBtn: document.getElementById('analyzeBtn'),
    loader: document.getElementById('loader'),
    results: document.getElementById('results'),
    errorCard: document.getElementById('errorCard'),
    errorMessage: document.getElementById('errorMessage'),
    retryBtn: document.getElementById('retryBtn'),
    modelStatus: document.getElementById('modelStatus'),
    riskBadge: document.getElementById('riskBadge'),
    aiProbability: document.getElementById('aiProbability'),
    humanProbability: document.getElementById('humanProbability'),
    confidence: document.getElementById('confidenceScore'),
    verdictText: document.getElementById('verdictText'),
    verdictReasons: document.getElementById('verdictReasons'),
    perplexity: document.getElementById('perplexity'),
    burstiness: document.getElementById('burstiness'),
    tokenCount: document.getElementById('tokenCount'),
    modelUsed: document.getElementById('modelUsed'),
    inferenceTime: document.getElementById('inferenceTime'),
    entropy: document.getElementById('entropy'),
    forensicList: document.getElementById('forensicList'),
    copyMetrics: document.getElementById('copyMetrics'),
    monospaceToggle: document.getElementById('monospaceToggle'),
    splineContainer: document.getElementById('splineContainer'),
    freqChart: document.getElementById('freqChart'),
    gaugeChart: document.getElementById('gaugeChart'),
    radarChart: document.getElementById('radarChart'),
    suggestionList: document.getElementById('suggestionList'),
    heatmapContainer: document.getElementById('heatmapContainer'),
    heatmapToggle: document.getElementById('heatmapToggle'),
    heatmapCard: document.getElementById('heatmapCard')
};

let splineLoaded = false;
let lastAnalyzedText = '';

function setLoading(isLoading) {
    ui.analyzeBtn.disabled = isLoading;
    ui.loader.classList.toggle('hidden', !isLoading);
    ui.loader.setAttribute('aria-busy', isLoading.toString());
    ui.modelStatus.querySelector('.status-text').textContent = isLoading ? 'Model Loading' : 'Model Active';
    const dot = ui.modelStatus.querySelector('.status-dot');
    dot.classList.toggle('active', !isLoading);
    if (isLoading) ensureSplineLoaded();
}

function ensureSplineLoaded() {
    if (splineLoaded || !ui.splineContainer) return;
    const iframe = document.createElement('iframe');
    iframe.src = 'https://my.spline.design/dunes-e5udKuZ4pYQhrEugeAVGiDkE/';
    iframe.loading = 'lazy';
    ui.splineContainer.appendChild(iframe);
    splineLoaded = true;
}

function showError(message) {
    ui.errorMessage.textContent = message;
    ui.errorCard.classList.remove('hidden');
}

function clearError() {
    ui.errorCard.classList.add('hidden');
}

function updateCharCount() {
    const length = ui.textInput.value.length;
    ui.charCounter.textContent = `${length} character${length === 1 ? '' : 's'}`;
}

async function checkPlagiarism() {
    clearError();
    const text = ui.textInput.value;
    const file = ui.fileInput.files[0];

    try {
        if (!text && !file) {
            throw new Error('Please enter text or upload a file.');
        }

        let payload = { text };

        if (file) {
            const formData = new FormData();
            formData.append('file', file);
            ui.uploadStatus.textContent = 'Uploading file…';

            const response = await fetchWithRetry('http://localhost:5000/upload', {
                method: 'POST',
                body: formData,
                mode: 'cors'
            });

            const data = await response.json();
            if (!response.ok) {
                throw new Error(data.error || `HTTP ${response.status}: Upload failed`);
            }

            if (!data.text || !data.text.trim()) {
                throw new Error('No text could be extracted from the file.');
            }

            payload.text = data.text;
            ui.textInput.value = data.text;
            updateCharCount();
            ui.uploadStatus.textContent = `File: ${data.filename} (${data.characters || data.text.length} chars)`;
            ui.uploadStatus.style.color = '#1a1a1a';
        }

        if (!payload.text || !payload.text.trim()) {
            throw new Error('No text available for analysis.');
        }

        setLoading(true);

        const [plagiarismResult, streamlitResult] = await Promise.all([
            fetchAnalysis('http://localhost:5000/check', payload),
            fetchAnalysis('http://localhost:5000/streamlit-analysis', payload)
        ]);

        const combined = { ...plagiarismResult, streamlit: streamlitResult };
        lastAnalyzedText = payload.text;
        updateResults(combined);
    } catch (error) {
        console.error('Analysis error:', error);
        showError(error.message);
    } finally {
        setLoading(false);
    }
}

async function fetchWithRetry(url, options, maxRetries = 3) {
    for (let attempt = 0; attempt < maxRetries; attempt++) {
        try {
            const response = await fetch(url, {
                ...options,
                headers: {
                    Accept: 'application/json',
                    ...(options.headers || {})
                }
            });

            if (response.ok || response.status < 500) {
                return response;
            }
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        } catch (err) {
            if (attempt === maxRetries - 1) throw err;
            await new Promise(resolve => setTimeout(resolve, Math.pow(2, attempt) * 500));
        }
    }
}

async function fetchAnalysis(url, data) {
    const response = await fetchWithRetry(url, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            Accept: 'application/json'
        },
        body: JSON.stringify(data),
        mode: 'cors'
    });

    if (!response.ok) {
        throw new Error(`Analysis request failed: HTTP ${response.status}`);
    }

    return await response.json();
}

function clamp(val, min, max) {
    return Math.min(Math.max(val, min), max);
}

function computeProbabilities(result) {
    console.log('[DEBUG] /check result:', JSON.stringify(result, null, 2));

    // 1. Try backend signal score (0-1)
    const signalScore = result.analysis?.signals?.score;
    // 2. Try explicit AI probability fields
    const rawAi = result.ai_probability ?? result.ai_probability_percent ?? result.ai_likelihood;

    let aiProbability;

    if (typeof signalScore === 'number' && !isNaN(signalScore)) {
        // Backend gives a signal score 0-1. Scale to percentage.
        aiProbability = clamp(Math.round(signalScore * 100), 0, 100);
    } else if (typeof rawAi === 'number') {
        aiProbability = clamp(Math.round(rawAi > 1 ? rawAi : rawAi * 100), 0, 100);
    } else {
        // Fallback: compute from perplexity + burstiness directly
        aiProbability = computeAiProbFromMetrics(result);
    }

    // If backend explicitly says AI-generated but our score is low, boost it
    if (result.is_ai_generated === true && aiProbability < 60) {
        aiProbability = Math.max(aiProbability, 70);
    }

    const humanProbability = clamp(100 - aiProbability, 0, 100);

    // Confidence: backend sends 0-1 float inside analysis
    let confidenceRaw = result.analysis?.confidence;
    if (typeof confidenceRaw !== 'number' || isNaN(confidenceRaw)) {
        confidenceRaw = result.confidence ?? result.confidence_score ?? 0.5;
    }
    // Normalize: if <= 1, treat as fraction
    if (confidenceRaw >= 0 && confidenceRaw <= 1) confidenceRaw = confidenceRaw * 100;
    const confidence = clamp(Math.round(confidenceRaw), 0, 100);

    console.log('[DEBUG] computed:', { signalScore, aiProbability, humanProbability, confidence });
    return { aiProbability, humanProbability, confidence };
}

function computeAiProbFromMetrics(result) {
    // Heuristic fallback when backend signals aren't available
    const perplexity = result.perplexity;
    const burstiness = result.burstiness;
    let score = 0;

    if (typeof perplexity === 'number') {
        // Low perplexity = more AI-like
        if (perplexity < 30) score += 35;
        else if (perplexity < 50) score += 25;
        else if (perplexity < 80) score += 15;
        else score += 5;
    }

    if (typeof burstiness === 'number') {
        const absBurst = Math.abs(burstiness);
        // Very low absolute burstiness = more AI-like
        if (absBurst < 0.05) score += 35;
        else if (absBurst < 0.1) score += 25;
        else if (absBurst < 0.2) score += 15;
        else score += 5;
    }

    // Cap at 95 since this is a heuristic
    return clamp(score, 0, 95);
}

function setRiskBadge(prob, reasons = []) {
    let text = 'Low Risk';
    let bg = '#d4f5ef';
    const inconclusive = reasons.some(r => r.toLowerCase().includes('too short'));

    if (inconclusive) { text = 'Inconclusive'; bg = '#f1e0a3'; }
    else if (prob >= 70) { text = 'High Risk'; bg = '#f6b8a4'; }
    else if (prob >= 40) { text = 'Medium Risk'; bg = '#f1e0a3'; }
    ui.riskBadge.textContent = text;
    ui.riskBadge.style.background = bg;
}

function renderForensics(result) {
    ui.forensicList.innerHTML = '';
    const { aiProbability, confidence } = computeProbabilities(result);
    const base = value => clamp(Math.round(value), 0, 100);

    // Style consistency: backend puts it at top level, not inside analysis
    const styleConsistency = result.style_consistency ?? result.analysis?.style_consistency;
    const sentenceConsistency = typeof styleConsistency === 'number'
        ? base(styleConsistency * 100)
        : base(confidence);

    // Burstiness: handle negative values properly
    // Map burstiness range (-1 to 1) to repetition (0-100)
    // More negative or closer to 0 = more repetitive = higher intensity
    const burstVal = typeof result.burstiness === 'number' ? result.burstiness : 0;
    const repetitionIntensity = base((1 - Math.min(Math.abs(burstVal), 1)) * 100);

    // Language predictability: scale perplexity (0-200+) to 0-100, invert
    const perpVal = typeof result.perplexity === 'number' ? result.perplexity : 60;
    const languagePredictability = base(Math.max(0, 100 - (perpVal / 1.5)));

    const bars = result.paragraph_probabilities || result.streamlit?.paragraph_probabilities;
    const derived = [
        { label: 'Paragraph AI Probability', value: aiProbability },
        { label: 'Sentence Consistency', value: sentenceConsistency },
        { label: 'Repetition Intensity', value: repetitionIntensity },
        { label: 'Language Predictability', value: languagePredictability }
    ];

    const items = Array.isArray(bars) && bars.length
        ? bars.map((v, i) => ({ label: `Paragraph ${i + 1}`, value: base(v) })).concat(derived.slice(1))
        : derived;

    items.forEach(item => {
        const row = document.createElement('div');
        row.className = 'forensic-item';

        const header = document.createElement('div');
        header.className = 'forensic-item-header';
        header.innerHTML = `<span>${item.label}</span><span>${item.value}%</span>`;

        const track = document.createElement('div');
        track.className = 'bar-track';
        const fill = document.createElement('div');
        fill.className = 'bar-fill';
        fill.style.width = `${item.value}%`;
        fill.style.background = item.value > 65 ? 'var(--danger)' : item.value > 40 ? 'var(--warning)' : 'var(--success)';
        track.appendChild(fill);

        row.appendChild(header);
        row.appendChild(track);
        ui.forensicList.appendChild(row);
    });
}

function formatNumber(value, decimals = 2) {
    return typeof value === 'number' ? value.toFixed(decimals) : '—';
}

function updateResults(result) {
    ui.results.classList.remove('hidden');

    const { aiProbability, humanProbability, confidence } = computeProbabilities(result);
    animateValue(ui.aiProbability, 0, aiProbability, 600, '%');
    animateValue(ui.humanProbability, 0, humanProbability, 600, '%');
    animateValue(ui.confidence, 0, confidence, 600, '%');

    const reasons = Array.isArray(result.analysis?.reasons) ? result.analysis.reasons : [];
    setRiskBadge(aiProbability, reasons);
    ui.verdictText.textContent = result.analysis?.overall || (result.is_ai_generated ? 'Likely AI-generated' : 'Likely human-written');

    if (ui.verdictReasons) {
        ui.verdictReasons.innerHTML = '';
        reasons.forEach(r => {
            const li = document.createElement('li');
            li.textContent = r;
            ui.verdictReasons.appendChild(li);
        });
    }

    ui.perplexity.textContent = formatNumber(result.perplexity);
    ui.burstiness.textContent = formatNumber(result.burstiness, 4);
    ui.tokenCount.textContent = result.token_count || result.tokens || '—';
    ui.modelUsed.textContent = result.model_name || result.model || '—';
    ui.inferenceTime.textContent = result.inference_time ? `${formatNumber(result.inference_time, 3)}s` : '—';
    ui.entropy.textContent = result.entropy ? formatNumber(result.entropy, 3) : '—';

    renderForensics(result);
    renderGaugeChart(aiProbability);
    renderRadarChart(result);
    renderFrequencyChart(result.streamlit?.word_frequency);
    renderSuggestions(result.streamlit?.word_frequency);
    renderHeatmap(lastAnalyzedText);
    clearError();
}

function animateValue(element, start, end, duration, suffix = '') {
    const range = end - start;
    if (range === 0) {
        element.textContent = end + suffix;
        return;
    }
    const stepTime = Math.max(Math.floor(duration / Math.abs(range)), 10);
    let current = start;
    const timer = setInterval(() => {
        current += Math.sign(range);
        element.textContent = current + suffix;
        if (current === end) clearInterval(timer);
    }, stepTime);
}

function copyMetricsToClipboard() {
    const text = `Perplexity: ${ui.perplexity.textContent}\nBurstiness: ${ui.burstiness.textContent}\nTokens: ${ui.tokenCount.textContent}\nModel: ${ui.modelUsed.textContent}\nInference Time: ${ui.inferenceTime.textContent}\nEntropy: ${ui.entropy.textContent}`;
    navigator.clipboard?.writeText(text).catch(() => {});
}

/* ─── Plotly shared config ─── */
const PLOTLY_THEME = {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { family: "'Space Grotesk', 'Segoe UI', sans-serif", color: '#1a1a1a', size: 13 },
    margin: { t: 30, r: 20, b: 60, l: 50 },
};

const PLOTLY_CONFIG = {
    displayModeBar: true,
    modeBarButtonsToAdd: ['lasso2d', 'select2d'],
    modeBarButtonsToRemove: ['sendDataToCloud'],
    displaylogo: false,
    responsive: true,
    scrollZoom: true,
    toImageButtonOptions: {
        format: 'png',
        filename: 'ai-plagiarism-chart',
        height: 600,
        width: 900,
        scale: 2
    }
};

/* Color palette matching the retro theme */
const COLORS = {
    bar: '#f4a261',
    barHover: '#e76f51',
    accent: '#2a9d8f',
    danger: '#e76f51',
    warning: '#e9c46a',
    success: '#2a9d8f',
    gridline: 'rgba(184, 166, 125, 0.4)',
    radarFill: 'rgba(244, 162, 97, 0.3)',
    radarLine: '#e76f51',
};

/* ─── AI Gauge Chart ─── */
function renderGaugeChart(aiProbability) {
    if (!ui.gaugeChart) return;

    const gaugeColor = aiProbability >= 70 ? COLORS.danger
        : aiProbability >= 40 ? COLORS.warning
        : COLORS.success;

    const data = [{
        type: 'indicator',
        mode: 'gauge+number+delta',
        value: aiProbability,
        number: {
            suffix: '%',
            font: { size: 48, family: "'JetBrains Mono', monospace", color: '#1a1a1a' }
        },
        gauge: {
            axis: {
                range: [0, 100],
                tickwidth: 2,
                tickcolor: '#1a1a1a',
                tickfont: { size: 11, family: "'Space Grotesk', sans-serif" },
                dtick: 20,
            },
            bar: { color: gaugeColor, thickness: 0.75 },
            bgcolor: '#e8d8b7',
            borderwidth: 3,
            bordercolor: '#1a1a1a',
            steps: [
                { range: [0, 30], color: 'rgba(42, 157, 143, 0.15)' },
                { range: [30, 60], color: 'rgba(233, 196, 106, 0.2)' },
                { range: [60, 100], color: 'rgba(231, 111, 81, 0.15)' },
            ],
            threshold: {
                line: { color: '#1a1a1a', width: 3 },
                thickness: 0.8,
                value: aiProbability,
            }
        }
    }];

    const layout = {
        ...PLOTLY_THEME,
        height: 280,
        margin: { t: 20, r: 30, b: 10, l: 30 },
        annotations: [{
            text: aiProbability >= 70 ? 'HIGH RISK' : aiProbability >= 40 ? 'MEDIUM RISK' : 'LOW RISK',
            x: 0.5, y: -0.05,
            showarrow: false,
            font: { size: 14, color: gaugeColor, family: "'Space Grotesk', sans-serif", weight: 700 }
        }]
    };

    Plotly.newPlot(ui.gaugeChart, data, layout, { ...PLOTLY_CONFIG, displayModeBar: false });
}

/* ─── Signal Radar Chart ─── */
function renderRadarChart(result) {
    if (!ui.radarChart) return;

    const signals = result.analysis?.signals || {};
    const contributions = signals.contributions || {};

    const categories = [
        'Burstiness', 'Sent. Variability', 'Vocab Diversity',
        'Repetition', 'Perplexity', 'Punct. Pattern', 'Bigram Repeat'
    ];
    const keys = [
        'low_burstiness', 'low_sent_var', 'low_ttr',
        'high_repetition', 'low_perplexity', 'low_punct_ratio', 'repeat_bigrams'
    ];

    const values = keys.map(k => clamp(Math.round((contributions[k] || 0) * 100), 0, 100));
    // Close the polygon
    const radarValues = [...values, values[0]];
    const radarCats = [...categories, categories[0]];

    const data = [{
        type: 'scatterpolar',
        r: radarValues,
        theta: radarCats,
        fill: 'toself',
        fillcolor: COLORS.radarFill,
        line: { color: COLORS.radarLine, width: 3 },
        marker: { size: 7, color: COLORS.radarLine, symbol: 'diamond' },
        name: 'AI Signals',
        hovertemplate: '<b>%{theta}</b><br>Signal: %{r}%<extra></extra>',
    }];

    const layout = {
        ...PLOTLY_THEME,
        height: 340,
        margin: { t: 40, r: 60, b: 40, l: 60 },
        polar: {
            bgcolor: 'rgba(232, 216, 183, 0.3)',
            radialaxis: {
                visible: true,
                range: [0, 100],
                ticksuffix: '%',
                tickfont: { size: 10 },
                gridcolor: COLORS.gridline,
                linecolor: '#1a1a1a',
            },
            angularaxis: {
                tickfont: { size: 11, family: "'Space Grotesk', sans-serif" },
                gridcolor: COLORS.gridline,
                linecolor: '#1a1a1a',
                linewidth: 2,
            }
        },
        showlegend: false,
    };

    Plotly.newPlot(ui.radarChart, data, layout, PLOTLY_CONFIG);
}

/* ─── Word Frequency Bar Chart (Plotly) ─── */
function renderFrequencyChart(freqData) {
    if (!ui.freqChart) return;

    const hasData = freqData && Array.isArray(freqData.words) && Array.isArray(freqData.counts)
        && freqData.words.length && freqData.counts.length;
    if (!hasData) {
        ui.freqChart.innerHTML = '<p style="text-align:center;padding:40px;color:#3a2d20;">No frequency data available</p>';
        return;
    }

    // Generate gradient colors for bars
    const maxCount = Math.max(...freqData.counts);
    const barColors = freqData.counts.map(c => {
        const ratio = c / maxCount;
        if (ratio > 0.7) return COLORS.danger;
        if (ratio > 0.4) return COLORS.warning;
        return COLORS.accent;
    });

    const data = [{
        type: 'bar',
        x: freqData.words,
        y: freqData.counts,
        marker: {
            color: barColors,
            line: { color: '#1a1a1a', width: 2 },
            cornerradius: 6,
        },
        hovertemplate: '<b>%{x}</b><br>Count: %{y}<extra></extra>',
        hoverlabel: {
            bgcolor: '#f0dfc0',
            bordercolor: '#1a1a1a',
            font: { family: "'Space Grotesk', sans-serif", color: '#1a1a1a', size: 14 }
        },
        textposition: 'outside',
        text: freqData.counts.map(String),
        textfont: { family: "'JetBrains Mono', monospace", size: 12, color: '#1a1a1a' },
    }];

    const layout = {
        ...PLOTLY_THEME,
        height: 380,
        xaxis: {
            tickangle: -35,
            tickfont: { size: 12, family: "'Space Grotesk', sans-serif" },
            gridcolor: 'transparent',
            linecolor: '#1a1a1a',
            linewidth: 2,
        },
        yaxis: {
            title: { text: 'Count', font: { size: 13 } },
            gridcolor: COLORS.gridline,
            linecolor: '#1a1a1a',
            linewidth: 2,
            zeroline: false,
        },
        dragmode: 'zoom',
        selectdirection: 'h',
        bargap: 0.25,
    };

    Plotly.newPlot(ui.freqChart, data, layout, PLOTLY_CONFIG);
}

function renderSuggestions(freqData) {
    if (!ui.suggestionList) return;
    ui.suggestionList.innerHTML = '';

    const hasData = freqData && Array.isArray(freqData.words) && Array.isArray(freqData.counts) && freqData.words.length;
    if (!hasData) {
        ui.suggestionList.textContent = 'No repetition detected.';
        return;
    }

    const wordCounts = freqData.words.map((w, i) => ({ word: w, count: freqData.counts[i] || 0 }));
    wordCounts.sort((a, b) => b.count - a.count);

    const synonyms = {
        good: ['great', 'strong', 'solid'],
        important: ['crucial', 'key', 'vital'],
        improve: ['enhance', 'refine', 'boost'],
        use: ['leverage', 'apply', 'employ'],
        make: ['create', 'build', 'craft'],
        help: ['assist', 'support', 'facilitate'],
        work: ['contribute', 'execute', 'deliver'],
        team: ['group', 'crew', 'squad'],
        project: ['initiative', 'engagement', 'assignment'],
        data: ['dataset', 'information', 'metrics']
    };

    const items = wordCounts.slice(0, 6).filter(item => item.count > 1);
    if (!items.length) {
        ui.suggestionList.textContent = 'Repetition is minimal. Nice variety!';
        return;
    }

    items.forEach(({ word, count }) => {
        const lower = word.toLowerCase();
        const options = synonyms[lower];
        const alt = options ? options.join(', ') : 'try a synonym to add variety';
        const div = document.createElement('div');
        div.className = 'suggestion-item';
        div.innerHTML = `<strong>${word}</strong><span>Used ${count}× — consider: ${alt}</span>`;
        ui.suggestionList.appendChild(div);
    });
}

/* ─── Sentence AI Heatmap ─── */
function heatmapColor(prob) {
    // Green (human) → Yellow (uncertain) → Red (AI)
    // 0.0 = #2a9d8f, 0.5 = #e9c46a, 1.0 = #e76f51
    const stops = [
        [0.0,  42, 157, 143],   // --success green
        [0.25, 137, 194, 123],  // light green
        [0.5,  233, 196, 106],  // --warning yellow
        [0.75, 244, 162,  97],  // --accent orange
        [1.0,  231, 111,  81],  // --danger red
    ];
    const p = Math.max(0, Math.min(1, prob));
    let lo = stops[0], hi = stops[stops.length - 1];
    for (let i = 0; i < stops.length - 1; i++) {
        if (p >= stops[i][0] && p <= stops[i + 1][0]) {
            lo = stops[i];
            hi = stops[i + 1];
            break;
        }
    }
    const t = (hi[0] - lo[0]) > 0 ? (p - lo[0]) / (hi[0] - lo[0]) : 0;
    const r = Math.round(lo[1] + t * (hi[1] - lo[1]));
    const g = Math.round(lo[2] + t * (hi[2] - lo[2]));
    const b = Math.round(lo[3] + t * (hi[3] - lo[3]));
    return `rgba(${r}, ${g}, ${b}, 0.35)`;
}

async function renderHeatmap(text) {
    if (!ui.heatmapContainer) return;
    ui.heatmapContainer.innerHTML = '<p class="heatmap-placeholder">Analyzing sentences…</p>';

    if (!text || text.trim().length < 20) {
        ui.heatmapContainer.innerHTML = '<p class="heatmap-placeholder">Text too short for sentence analysis.</p>';
        return;
    }

    try {
        const response = await fetchWithRetry('http://localhost:5000/sentence-analysis', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
            body: JSON.stringify({ text }),
            mode: 'cors'
        });

        const data = await response.json();
        if (!response.ok || !data.sentences || !data.sentences.length) {
            ui.heatmapContainer.innerHTML = '<p class="heatmap-placeholder">Could not analyze sentences.</p>';
            return;
        }

        ui.heatmapContainer.innerHTML = '';
        data.sentences.forEach((s, idx) => {
            const span = document.createElement('span');
            span.className = 'heatmap-span';
            span.style.backgroundColor = heatmapColor(s.probability);
            span.textContent = s.text + ' ';

            // Tooltip
            const tip = document.createElement('span');
            tip.className = 'heatmap-tooltip';
            const pct = Math.round(s.probability * 100);
            const pplText = s.perplexity != null ? ` · PPL ${s.perplexity}` : '';
            tip.textContent = `AI: ${pct}%${pplText}`;
            span.appendChild(tip);

            ui.heatmapContainer.appendChild(span);
        });
    } catch (err) {
        console.error('Heatmap error:', err);
        ui.heatmapContainer.innerHTML = '<p class="heatmap-placeholder">Heatmap unavailable.</p>';
    }
}

document.addEventListener('DOMContentLoaded', () => {
    updateCharCount();

    ui.textInput.addEventListener('input', updateCharCount);
    ui.analyzeBtn.addEventListener('click', checkPlagiarism);
    ui.retryBtn.addEventListener('click', checkPlagiarism);
    ui.copyMetrics.addEventListener('click', copyMetricsToClipboard);

    // Heatmap toggle
    if (ui.heatmapToggle && ui.heatmapContainer) {
        ui.heatmapToggle.addEventListener('click', () => {
            const isActive = ui.heatmapToggle.getAttribute('aria-pressed') === 'true';
            ui.heatmapToggle.setAttribute('aria-pressed', String(!isActive));
            ui.heatmapContainer.classList.toggle('collapsed', isActive);
            ui.heatmapToggle.querySelector('.toggle-icon').textContent = isActive ? '○' : '◉';
        });
    }

    ui.fileInput.addEventListener('change', e => {
        const file = e.target.files[0];
        if (file) {
            ui.uploadStatus.textContent = `Selected: ${file.name}`;
        }
    });

    ui.monospaceToggle.addEventListener('change', e => {
        ui.textInput.classList.toggle('monospace-input', e.target.checked);
    });
});