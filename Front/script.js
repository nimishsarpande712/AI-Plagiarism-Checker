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
    suggestionList: document.getElementById('suggestionList')
};

let splineLoaded = false;
let freqChartInstance = null;
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
    renderFrequencyChart(result.streamlit?.word_frequency);
    renderSuggestions(result.streamlit?.word_frequency);
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

function renderFrequencyChart(freqData) {
    if (!ui.freqChart) return;
    ui.freqChart.innerHTML = '';

    const hasData = freqData && Array.isArray(freqData.words) && Array.isArray(freqData.counts) && freqData.words.length && freqData.counts.length;
    if (!hasData) {
        ui.freqChart.textContent = 'No frequency data available';
        if (freqChartInstance) {
            freqChartInstance.destroy();
            freqChartInstance = null;
        }
        return;
    }

    if (freqChartInstance) {
        freqChartInstance.destroy();
        freqChartInstance = null;
    }

    freqChartInstance = c3.generate({
        bindto: ui.freqChart,
        data: {
            columns: [ ['Frequency', ...freqData.counts] ],
            type: 'bar'
        },
        axis: {
            x: {
                type: 'category',
                categories: freqData.words,
                tick: { rotate: 45, multiline: false }
            },
            y: { label: 'Count' }
        },
        bar: { width: { ratio: 0.6 } },
        padding: { right: 20 }
    });
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

document.addEventListener('DOMContentLoaded', () => {
    updateCharCount();

    ui.textInput.addEventListener('input', updateCharCount);
    ui.analyzeBtn.addEventListener('click', checkPlagiarism);
    ui.retryBtn.addEventListener('click', checkPlagiarism);
    ui.copyMetrics.addEventListener('click', copyMetricsToClipboard);

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