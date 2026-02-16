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
    heatmapCard: document.getElementById('heatmapCard'),
    voiceSummary: document.getElementById('voiceSummary'),
    voiceRatioBar: document.getElementById('voiceRatioBar'),
    voiceAssessment: document.getElementById('voiceAssessment'),
    voiceFlagged: document.getElementById('voiceFlagged'),
};

let splineLoaded = false;
let lastAnalyzedText = '';
let lastAnalysisId = null;

// Session management - generate a persistent session ID
function getSessionId() {
    let sid = localStorage.getItem('plagiarism_session_id');
    if (!sid) {
        sid = 'sess_' + crypto.randomUUID();
        localStorage.setItem('plagiarism_session_id', sid);
    }
    return sid;
}

const SESSION_ID = getSessionId();

// ─── Auth State ───
let currentUser = null;
let authToken = null;

function loadAuthState() {
    const saved = localStorage.getItem('plagiarism_auth');
    if (saved) {
        try {
            const parsed = JSON.parse(saved);
            authToken = parsed.access_token || null;
            currentUser = parsed.user || null;
            if (authToken) {
                // Verify token is still valid
                verifyAuth();
            }
        } catch (e) {
            clearAuthState();
        }
    }
    updateAuthUI();
}

function saveAuthState(data) {
    authToken = data.access_token;
    currentUser = data.user;
    localStorage.setItem('plagiarism_auth', JSON.stringify({
        access_token: data.access_token,
        refresh_token: data.refresh_token,
        user: data.user,
    }));
    updateAuthUI();
}

function clearAuthState() {
    authToken = null;
    currentUser = null;
    localStorage.removeItem('plagiarism_auth');
    updateAuthUI();
}

function updateAuthUI() {
    const loginBtn = document.getElementById('loginBtn');
    const avatarBtn = document.getElementById('userAvatarBtn');
    const initial = document.getElementById('userInitial');
    const menuName = document.getElementById('userMenuName');
    const menuEmail = document.getElementById('userMenuEmail');

    if (currentUser && authToken) {
        // Logged in
        if (loginBtn) loginBtn.style.display = 'none';
        if (avatarBtn) {
            avatarBtn.style.display = 'flex';
            const name = currentUser.full_name || currentUser.email || '?';
            if (initial) initial.textContent = name.charAt(0).toUpperCase();
        }
        if (menuName) menuName.textContent = currentUser.full_name || 'User';
        if (menuEmail) menuEmail.textContent = currentUser.email || '';
    } else {
        // Not logged in
        if (loginBtn) loginBtn.style.display = 'flex';
        if (avatarBtn) avatarBtn.style.display = 'none';
        // Close menu if open
        const menu = document.getElementById('userMenu');
        if (menu) menu.style.display = 'none';
    }
}

async function verifyAuth() {
    if (!authToken) return;
    try {
        const resp = await fetch('http://localhost:5000/auth/me', {
            headers: { 'Authorization': `Bearer ${authToken}` }
        });
        if (!resp.ok) {
            clearAuthState();
        } else {
            const data = await resp.json();
            if (data.user) {
                currentUser = data.user;
                updateAuthUI();
            }
        }
    } catch (e) {
        console.warn('Auth verification failed:', e);
    }
}

function getAuthHeaders() {
    const headers = { 'X-Session-Id': SESSION_ID };
    if (authToken) {
        headers['Authorization'] = `Bearer ${authToken}`;
    }
    return headers;
}

// ─── Auth Modal ───
function openAuthModal(tab = 'login') {
    const modal = document.getElementById('authModal');
    if (modal) {
        modal.style.display = 'flex';
        switchAuthTab(tab);
        // Clear errors
        const loginErr = document.getElementById('loginError');
        const signupErr = document.getElementById('signupError');
        const signupOk = document.getElementById('signupSuccess');
        if (loginErr) loginErr.style.display = 'none';
        if (signupErr) signupErr.style.display = 'none';
        if (signupOk) signupOk.style.display = 'none';
    }
}

function closeAuthModal() {
    const modal = document.getElementById('authModal');
    if (modal) modal.style.display = 'none';
}

function switchAuthTab(tab) {
    const tabLogin = document.getElementById('tabLogin');
    const tabSignup = document.getElementById('tabSignup');
    const formLogin = document.getElementById('loginForm');
    const formSignup = document.getElementById('signupForm');

    if (tab === 'login') {
        tabLogin.classList.add('active');
        tabSignup.classList.remove('active');
        formLogin.style.display = 'block';
        formSignup.style.display = 'none';
    } else {
        tabLogin.classList.remove('active');
        tabSignup.classList.add('active');
        formLogin.style.display = 'none';
        formSignup.style.display = 'block';
    }
}

function toggleUserMenu() {
    const menu = document.getElementById('userMenu');
    if (menu) {
        menu.style.display = menu.style.display === 'none' ? 'block' : 'none';
    }
}

// Close user menu when clicking outside
document.addEventListener('click', (e) => {
    const menu = document.getElementById('userMenu');
    const avatarBtn = document.getElementById('userAvatarBtn');
    if (menu && menu.style.display !== 'none' && !menu.contains(e.target) && !avatarBtn.contains(e.target)) {
        menu.style.display = 'none';
    }
});

async function handleLogin(e) {
    e.preventDefault();
    const email = document.getElementById('loginEmail').value.trim();
    const password = document.getElementById('loginPassword').value;
    const errorEl = document.getElementById('loginError');
    const btn = document.getElementById('loginSubmitBtn');

    errorEl.style.display = 'none';
    btn.disabled = true;
    btn.textContent = 'Signing in...';

    try {
        const resp = await fetch('http://localhost:5000/auth/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email, password }),
        });
        const data = await resp.json();

        if (!resp.ok || data.error) {
            errorEl.textContent = data.error || 'Login failed';
            errorEl.style.display = 'block';
            return;
        }

        saveAuthState(data);
        closeAuthModal();
        loadHistory(); // Refresh with user-specific history
    } catch (err) {
        errorEl.textContent = 'Network error. Please try again.';
        errorEl.style.display = 'block';
    } finally {
        btn.disabled = false;
        btn.textContent = 'Sign In';
    }
}

async function handleSignup(e) {
    e.preventDefault();
    const fullName = document.getElementById('signupName').value.trim();
    const email = document.getElementById('signupEmail').value.trim();
    const password = document.getElementById('signupPassword').value;
    const errorEl = document.getElementById('signupError');
    const successEl = document.getElementById('signupSuccess');
    const btn = document.getElementById('signupSubmitBtn');

    errorEl.style.display = 'none';
    successEl.style.display = 'none';
    btn.disabled = true;
    btn.textContent = 'Creating account...';

    try {
        const resp = await fetch('http://localhost:5000/auth/signup', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email, password, full_name: fullName }),
        });
        const data = await resp.json();

        if (!resp.ok || data.error) {
            errorEl.textContent = data.error || 'Signup failed';
            errorEl.style.display = 'block';
            return;
        }

        // If session came back (email confirm disabled), auto-login
        if (data.access_token) {
            saveAuthState(data);
            closeAuthModal();
            loadHistory();
        } else {
            // Email confirmation required
            successEl.textContent = 'Account created! Check your email to confirm, then sign in.';
            successEl.style.display = 'block';
        }
    } catch (err) {
        errorEl.textContent = 'Network error. Please try again.';
        errorEl.style.display = 'block';
    } finally {
        btn.disabled = false;
        btn.textContent = 'Create Account';
    }
}

async function handleLogout() {
    try {
        await fetch('http://localhost:5000/auth/logout', {
            method: 'POST',
            headers: getAuthHeaders(),
        });
    } catch (e) { /* ignore */ }
    clearAuthState();
    loadHistory();
}

// Handle Supabase email confirmation redirect (tokens arrive in URL hash)
function handleAuthCallback() {
    const hash = window.location.hash;
    if (!hash || !hash.includes('access_token')) return;

    // Parse hash params: #access_token=...&refresh_token=...&...
    const params = new URLSearchParams(hash.substring(1));
    const accessToken = params.get('access_token');
    const refreshToken = params.get('refresh_token');

    if (accessToken) {
        // Save tokens and verify the session
        authToken = accessToken;
        saveAuthState({
            access_token: accessToken,
            refresh_token: refreshToken,
            user: null,
        });
        verifyAuth().then(() => loadHistory());

        // Clean up the URL
        history.replaceState(null, '', window.location.pathname);
    }
}

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
        lastAnalysisId = plagiarismResult.analysis_id || null;
        updateResults(combined);
        // Refresh history after successful analysis
        loadHistory();
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
            Accept: 'application/json',
            ...getAuthHeaders()
        },
        body: JSON.stringify({ ...data, session_id: SESSION_ID }),
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

    // Passive voice ratio as a forensic signal
    const voiceData = result.voice_analysis;
    const passivePercent = voiceData && typeof voiceData.passive_ratio === 'number'
        ? base(voiceData.passive_ratio * 100)
        : null;

    const bars = result.paragraph_probabilities || result.streamlit?.paragraph_probabilities;
    const derived = [
        { label: 'Paragraph AI Probability', value: aiProbability },
        { label: 'Sentence Consistency', value: sentenceConsistency },
        { label: 'Repetition Intensity', value: repetitionIntensity },
        { label: 'Language Predictability', value: languagePredictability }
    ];

    if (passivePercent !== null) {
        derived.push({ label: 'Passive Voice Ratio', value: passivePercent });
    }

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

function renderVoiceAnalysis(voice) {
    if (!ui.voiceSummary) return;

    // Reset
    ui.voiceSummary.innerHTML = '';
    ui.voiceRatioBar.innerHTML = '';
    ui.voiceAssessment.innerHTML = '';
    ui.voiceFlagged.innerHTML = '';

    if (!voice || typeof voice.passive_ratio !== 'number') {
        ui.voiceSummary.innerHTML = '<p class="voice-placeholder">Voice analysis unavailable.</p>';
        return;
    }

    const passivePct = Math.round(voice.passive_ratio * 100);
    const activePct = Math.round(voice.active_ratio * 100);

    // Summary stats
    ui.voiceSummary.innerHTML = `
        <div class="voice-stats">
            <div class="voice-stat">
                <span class="voice-stat-value voice-passive">${passivePct}%</span>
                <span class="voice-stat-label">Passive</span>
            </div>
            <div class="voice-stat">
                <span class="voice-stat-value voice-active">${activePct}%</span>
                <span class="voice-stat-label">Active</span>
            </div>
            <div class="voice-stat">
                <span class="voice-stat-value">${voice.total_sentences}</span>
                <span class="voice-stat-label">Sentences</span>
            </div>
            <div class="voice-stat">
                <span class="voice-stat-value voice-passive">${voice.passive_count}</span>
                <span class="voice-stat-label">Passive Hits</span>
            </div>
        </div>
    `;

    // Ratio bar (stacked horizontal bar)
    const passiveColor = passivePct >= 50 ? 'var(--danger)' : passivePct >= 30 ? 'var(--warning)' : 'var(--success)';
    ui.voiceRatioBar.innerHTML = `
        <div class="voice-bar-track">
            <div class="voice-bar-segment voice-bar-active" style="width: ${activePct}%; background: var(--success);"></div>
            <div class="voice-bar-segment voice-bar-passive" style="width: ${passivePct}%; background: ${passiveColor};"></div>
        </div>
        <div class="voice-bar-labels">
            <span style="color: var(--success);">● Active (${voice.active_count})</span>
            <span style="color: ${passiveColor};">● Passive (${voice.passive_count})</span>
        </div>
    `;

    // Assessment text
    ui.voiceAssessment.innerHTML = `<p class="voice-assessment-text">${voice.assessment}</p>`;

    // Flagged passive sentences (collapsible)
    if (voice.flagged_sections && voice.flagged_sections.length > 0) {
        const toggleId = 'voiceFlaggedToggle';
        let html = `
            <button class="voice-flagged-toggle" id="${toggleId}" type="button" aria-expanded="false">
                <span class="toggle-arrow">▸</span> Show ${voice.flagged_sections.length} passive sentence${voice.flagged_sections.length > 1 ? 's' : ''}
            </button>
            <div class="voice-flagged-list" id="voiceFlaggedList" style="display: none;">
        `;
        voice.flagged_sections.forEach((item, i) => {
            html += `
                <div class="voice-flagged-item">
                    <span class="voice-flagged-index">${item.index + 1}</span>
                    <span class="voice-flagged-text">${escapeHtml(item.text)}</span>
                </div>
            `;
        });
        html += '</div>';
        ui.voiceFlagged.innerHTML = html;

        // Toggle handler
        document.getElementById(toggleId).addEventListener('click', function () {
            const list = document.getElementById('voiceFlaggedList');
            const expanded = this.getAttribute('aria-expanded') === 'true';
            this.setAttribute('aria-expanded', !expanded);
            this.querySelector('.toggle-arrow').textContent = expanded ? '▸' : '▾';
            list.style.display = expanded ? 'none' : 'block';
            this.childNodes[1].textContent = expanded
                ? ` Show ${voice.flagged_sections.length} passive sentence${voice.flagged_sections.length > 1 ? 's' : ''}`
                : ` Hide ${voice.flagged_sections.length} passive sentence${voice.flagged_sections.length > 1 ? 's' : ''}`;
        });
    }
}

function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
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
    renderVoiceAnalysis(result.voice_analysis);
    renderGaugeChart(aiProbability);
    renderRadarChart(result);
    renderFrequencyChart(result.streamlit?.word_frequency);
    renderSuggestions(result.streamlit?.word_frequency);
    renderHeatmap(lastAnalyzedText);
    renderReport(result, aiProbability, humanProbability, confidence, reasons);
    clearError();

    // Show model learning status
    updateModelLearningUI(result.model_learning);
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

/* Color palette matching the Gitingest theme */
const COLORS = {
    bar: '#f4845f',
    barHover: '#e76f51',
    accent: '#2dd4a8',
    danger: '#e76f51',
    warning: '#e9c46a',
    success: '#2dd4a8',
    gridline: 'rgba(184, 166, 125, 0.3)',
    radarFill: 'rgba(244, 132, 95, 0.25)',
    radarLine: '#f4845f',
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
                { range: [0, 30], color: 'rgba(45, 212, 168, 0.15)' },
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
            bgcolor: 'rgba(252, 236, 211, 0.3)',
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

/* ─── Analysis Report ─── */
let lastReportData = null;

function renderReport(result, aiProbability, humanProbability, confidence, reasons) {
    const container = document.getElementById('reportContent');
    if (!container) return;

    const riskLevel = aiProbability >= 70 ? 'high' : (aiProbability >= 40 ? 'medium' : 'low');
    const riskText = aiProbability >= 70 ? 'HIGH RISK — Likely AI-Generated'
        : (aiProbability >= 40 ? 'MEDIUM RISK — Partially AI-influenced'
        : 'LOW RISK — Likely Human-Written');

    const timestamp = new Date().toLocaleString('en-US', {
        dateStyle: 'long', timeStyle: 'short'
    });

    // Store for download
    lastReportData = { result, aiProbability, humanProbability, confidence, reasons, riskLevel, riskText, timestamp };

    container.innerHTML = `
        <div class="report-section">
            <h3><span class="report-icon">⚖</span> Verdict</h3>
            <div class="report-verdict-box ${riskLevel}">
                ${riskText} — AI Probability: ${aiProbability}%
            </div>
            ${reasons.length ? `<ul class="report-reasons">${reasons.map(r => `<li>${r}</li>`).join('')}</ul>` : ''}
        </div>

        <div class="report-section">
            <h3><span class="report-icon">📊</span> Detection Metrics</h3>
            <div class="report-metric-grid">
                <div class="report-metric"><span class="label">AI Probability</span><span class="value">${aiProbability}%</span></div>
                <div class="report-metric"><span class="label">Human Probability</span><span class="value">${humanProbability}%</span></div>
                <div class="report-metric"><span class="label">Confidence</span><span class="value">${confidence}%</span></div>
                <div class="report-metric"><span class="label">Perplexity</span><span class="value">${formatNumber(result.perplexity)}</span></div>
                <div class="report-metric"><span class="label">Burstiness</span><span class="value">${formatNumber(result.burstiness, 4)}</span></div>
                <div class="report-metric"><span class="label">Entropy</span><span class="value">${result.entropy ? formatNumber(result.entropy, 3) : '—'}</span></div>
                <div class="report-metric"><span class="label">Token Count</span><span class="value">${result.token_count || '—'}</span></div>
                <div class="report-metric"><span class="label">Model</span><span class="value">${result.model_name || '—'}</span></div>
                <div class="report-metric"><span class="label">Inference Time</span><span class="value">${result.inference_time ? formatNumber(result.inference_time, 3) + 's' : '—'}</span></div>
            </div>
        </div>

        <div class="report-section">
            <h3><span class="report-icon">🔬</span> Signal Analysis</h3>
            <div class="report-metric-grid">
                ${Object.entries(result.analysis?.signals?.contributions || {}).map(([key, value]) => {
                    const label = key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
                    const pct = Math.round(value * 100);
                    return `<div class="report-metric"><span class="label">${label}</span><span class="value">${pct}%</span></div>`;
                }).join('')}
            </div>
        </div>

        <div class="report-section">
            <h3><span class="report-icon">📝</span> Text Statistics</h3>
            <div class="report-metric-grid">
                <div class="report-metric"><span class="label">Characters</span><span class="value">${lastAnalyzedText.length.toLocaleString()}</span></div>
                <div class="report-metric"><span class="label">Words</span><span class="value">${lastAnalyzedText.split(/\s+/).filter(Boolean).length.toLocaleString()}</span></div>
                <div class="report-metric"><span class="label">Style Consistency</span><span class="value">${result.style_consistency != null ? Math.round(result.style_consistency * 100) + '%' : '—'}</span></div>
                <div class="report-metric"><span class="label">Complexity</span><span class="value">${result.complexity != null ? Math.round(result.complexity * 100) + '%' : '—'}</span></div>
                <div class="report-metric"><span class="label">Variability</span><span class="value">${result.variability != null ? Math.round(result.variability * 100) + '%' : '—'}</span></div>
                <div class="report-metric"><span class="label">Readability</span><span class="value">${result.readability != null ? Math.round(result.readability * 100) + '%' : '—'}</span></div>
                <div class="report-metric"><span class="label">Passive Voice</span><span class="value">${result.voice_analysis ? Math.round(result.voice_analysis.passive_ratio * 100) + '%' : '—'}</span></div>
                <div class="report-metric"><span class="label">Active Voice</span><span class="value">${result.voice_analysis ? Math.round(result.voice_analysis.active_ratio * 100) + '%' : '—'}</span></div>
            </div>
        </div>

        <p class="report-timestamp">Report generated: ${timestamp}</p>
    `;
}

function downloadReport() {
    if (!lastReportData) {
        alert('Run an analysis first to generate a report.');
        return;
    }

    const { result, aiProbability, humanProbability, confidence, reasons, riskLevel, riskText, timestamp } = lastReportData;

    const inputPreview = lastAnalyzedText.length > 500
        ? lastAnalyzedText.substring(0, 500) + '...'
        : lastAnalyzedText;

    const signalRows = Object.entries(result.analysis?.signals?.contributions || {}).map(([key, value]) => {
        const label = key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
        const pct = Math.round(value * 100);
        const bar = '█'.repeat(Math.round(pct / 5)) + '░'.repeat(20 - Math.round(pct / 5));
        return `<tr><td style="padding:6px 12px;border:1px solid #ddd;font-weight:500;">${label}</td><td style="padding:6px 12px;border:1px solid #ddd;font-family:monospace;">${bar} ${pct}%</td></tr>`;
    }).join('');

    const riskColors = { low: '#2dd4a8', medium: '#e9c46a', high: '#e76f51' };

    const html = `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>AI Plagiarism Analysis Report</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Arial, sans-serif; background: #fef9ef; color: #1a1a1a; padding: 40px; max-width: 900px; margin: 0 auto; }
        h1 { font-size: 1.8rem; margin-bottom: 4px; }
        h2 { font-size: 1.1rem; color: #f4845f; margin: 24px 0 10px; border-bottom: 2px solid #1a1a1a; padding-bottom: 6px; }
        .header { text-align: center; margin-bottom: 30px; border-bottom: 3px solid #1a1a1a; padding-bottom: 20px; }
        .header p { color: #7a7067; font-size: 0.9rem; }
        .verdict-box { padding: 16px 20px; border-radius: 10px; border: 2px solid #1a1a1a; font-weight: 700; font-size: 1.1rem; margin: 12px 0; background: ${riskColors[riskLevel]}33; border-left: 6px solid ${riskColors[riskLevel]}; }
        table { width: 100%; border-collapse: collapse; margin: 10px 0; }
        td { padding: 8px 12px; border: 1px solid #ddd; font-size: 0.9rem; }
        td:first-child { font-weight: 600; background: #fcecd3; width: 40%; }
        .reasons { margin: 8px 0; padding-left: 20px; }
        .reasons li { padding: 3px 0; font-size: 0.88rem; color: #555; }
        .text-preview { background: #f5f0e3; border: 1px solid #ddd; border-radius: 8px; padding: 14px; font-size: 0.85rem; max-height: 200px; overflow: hidden; color: #444; margin: 8px 0; white-space: pre-wrap; word-break: break-word; }
        .footer { text-align: center; margin-top: 30px; padding-top: 16px; border-top: 2px solid #ddd; font-size: 0.78rem; color: #999; }
        @media print { body { padding: 20px; } }
    </style>
</head>
<body>
    <div class="header">
        <h1>🔍 AI Plagiarism Analysis Report</h1>
        <p>Generated: ${timestamp}</p>
    </div>

    <h2>Verdict</h2>
    <div class="verdict-box">${riskText} — AI Probability: ${aiProbability}%</div>
    ${reasons.length ? `<ul class="reasons">${reasons.map(r => `<li>${r}</li>`).join('')}</ul>` : ''}

    <h2>Detection Metrics</h2>
    <table>
        <tr><td>AI Probability</td><td>${aiProbability}%</td></tr>
        <tr><td>Human Probability</td><td>${humanProbability}%</td></tr>
        <tr><td>Confidence Score</td><td>${confidence}%</td></tr>
        <tr><td>Perplexity</td><td>${formatNumber(result.perplexity)}</td></tr>
        <tr><td>Burstiness</td><td>${formatNumber(result.burstiness, 4)}</td></tr>
        <tr><td>Entropy</td><td>${result.entropy ? formatNumber(result.entropy, 3) : '—'}</td></tr>
        <tr><td>Token Count</td><td>${result.token_count || '—'}</td></tr>
        <tr><td>Model Used</td><td>${result.model_name || '—'}</td></tr>
        <tr><td>Inference Time</td><td>${result.inference_time ? formatNumber(result.inference_time, 3) + 's' : '—'}</td></tr>
    </table>

    <h2>Signal Analysis</h2>
    <table>${signalRows}</table>

    <h2>Additional Metrics</h2>
    <table>
        <tr><td>Style Consistency</td><td>${result.style_consistency != null ? Math.round(result.style_consistency * 100) + '%' : '—'}</td></tr>
        <tr><td>Complexity</td><td>${result.complexity != null ? Math.round(result.complexity * 100) + '%' : '—'}</td></tr>
        <tr><td>Variability</td><td>${result.variability != null ? Math.round(result.variability * 100) + '%' : '—'}</td></tr>
        <tr><td>Readability</td><td>${result.readability != null ? Math.round(result.readability * 100) + '%' : '—'}</td></tr>
    </table>

    <h2>Analyzed Text (Preview)</h2>
    <div class="text-preview">${inputPreview.replace(/</g, '&lt;').replace(/>/g, '&gt;')}</div>

    <div class="footer">
        <p>AI Plagiarism Engine · Perplexity + Burstiness + Entropy Analysis</p>
        <p>Characters: ${lastAnalyzedText.length.toLocaleString()} · Words: ${lastAnalyzedText.split(/\\s+/).filter(Boolean).length.toLocaleString()}</p>
    </div>
</body>
</html>`;

    const blob = new Blob([html], { type: 'text/html' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `AI-Plagiarism-Report-${new Date().toISOString().slice(0, 10)}.html`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
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
        // Use AbortController for timeout on large texts
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 60000); // 60s timeout

        const response = await fetch('http://localhost:5000/sentence-analysis', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
            body: JSON.stringify({ text }),
            mode: 'cors',
            signal: controller.signal
        });

        clearTimeout(timeoutId);

        const data = await response.json();
        if (!response.ok || !data.sentences || !data.sentences.length) {
            ui.heatmapContainer.innerHTML = '<p class="heatmap-placeholder">Could not analyze sentences.</p>';
            return;
        }

        ui.heatmapContainer.innerHTML = '';

        // Show sentence count info if applicable
        if (data.total_sentences && data.total_sentences > data.analyzed) {
            const info = document.createElement('p');
            info.className = 'heatmap-info';
            info.textContent = `Showing ${data.analyzed} of ${data.total_sentences} sentences`;
            info.style.cssText = 'font-size:0.78rem;color:var(--muted);margin-bottom:8px;font-weight:600;';
            ui.heatmapContainer.appendChild(info);
        }

        data.sentences.forEach((s, idx) => {
            // Skip info notes
            if (s.is_note) {
                const note = document.createElement('span');
                note.className = 'heatmap-note';
                note.textContent = s.text;
                note.style.cssText = 'color:var(--muted);font-style:italic;font-size:0.85rem;';
                ui.heatmapContainer.appendChild(note);
                return;
            }

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
        if (err.name === 'AbortError') {
            ui.heatmapContainer.innerHTML = '<p class="heatmap-placeholder">Heatmap analysis timed out for this text. Try with shorter text.</p>';
        } else {
            ui.heatmapContainer.innerHTML = '<p class="heatmap-placeholder">Heatmap unavailable.</p>';
        }
    }
}

function updateModelLearningUI(learningData) {
    // Remove existing indicator if any
    const existing = document.getElementById('trainingIndicator');
    if (existing) existing.remove();

    if (!learningData || !learningData.total_analyses) return;

    const indicator = document.createElement('div');
    indicator.id = 'trainingIndicator';
    indicator.className = 'training-indicator';
    
    const n = learningData.total_analyses;
    const improved = learningData.model_improved;
    
    if (improved) {
        indicator.innerHTML = `<span>🧠</span> Model adapted (${n} samples)`;
        indicator.style.borderColor = 'var(--success)';
        indicator.style.background = 'rgba(45, 212, 168, 0.15)';
    } else {
        indicator.innerHTML = `<span>📈</span> Training: ${n}/20 samples`;
        indicator.style.borderColor = 'var(--warning)';
        indicator.style.background = 'rgba(233, 196, 106, 0.15)';
        indicator.style.color = '#b8860b';
    }

    // Insert into nav-right, before the settings button
    const navRight = document.querySelector('.nav-right');
    const settingsBtn = navRight?.querySelector('.icon-button');
    if (navRight && settingsBtn) {
        navRight.insertBefore(indicator, settingsBtn);
    } else if (navRight) {
        navRight.appendChild(indicator);
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

    // Share button
    const shareBtn = document.getElementById('shareBtn');
    if (shareBtn) {
        shareBtn.addEventListener('click', () => shareReport(lastAnalysisId));
    }

    // Download report button
    const downloadBtn = document.getElementById('downloadReportBtn');
    if (downloadBtn) {
        downloadBtn.addEventListener('click', downloadReport);
    }

    // Load analysis history on page load
    loadHistory();

    // Load auth state on page load
    loadAuthState();

    // Handle Supabase email confirmation redirect (tokens in URL hash)
    handleAuthCallback();
});

// ─── History Panel ───
async function loadHistory() {
    try {
        const response = await fetch(`http://localhost:5000/history?session_id=${SESSION_ID}&limit=10`, {
            headers: getAuthHeaders()
        });
        const data = await response.json();
        renderHistory(data.history || []);
    } catch (err) {
        console.warn('Could not load history:', err);
    }
}

function renderHistory(items) {
    const container = document.getElementById('historyList');
    if (!container) return;

    if (!items.length) {
        container.innerHTML = '<p style="color:var(--muted);text-align:center;padding:16px;">No previous scans yet.</p>';
        return;
    }

    container.innerHTML = items.map(item => {
        const date = new Date(item.created_at).toLocaleDateString('en-US', {
            month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit'
        });
        const risk = item.risk_level || 'low';
        const riskColors = { low: '#d4f5ef', medium: '#f1e0a3', high: '#f6b8a4', inconclusive: '#e0e0e0' };
        const prob = item.ai_probability != null ? Math.round(item.ai_probability * 100) + '%' : '—';

        return `
            <div class="history-item" onclick="viewAnalysis('${item.id}')" style="
                background: var(--bg, #faf4e8);
                border: 2px solid var(--border, #1a1a1a);
                border-radius: 12px;
                padding: 10px 14px;
                cursor: pointer;
                margin-bottom: 8px;
                transition: transform 0.15s ease;
            " onmouseover="this.style.transform='translateY(-2px)'" onmouseout="this.style.transform='none'">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <span style="font-weight:600;font-size:0.9rem;">${date}</span>
                    <span style="
                        padding:3px 10px;
                        border:2px solid var(--border, #1a1a1a);
                        border-radius:20px;
                        font-size:0.75rem;
                        font-weight:700;
                        background:${riskColors[risk] || riskColors.low};
                    ">${risk.toUpperCase()}</span>
                </div>
                <div style="display:flex;gap:16px;margin-top:6px;font-size:0.82rem;color:var(--muted, #6b5c42);">
                    <span>AI: ${prob}</span>
                    <span>PPL: ${item.perplexity ? item.perplexity.toFixed(1) : '—'}</span>
                    <span>${item.input_source || 'paste'}</span>
                </div>
            </div>
        `;
    }).join('');
}

async function viewAnalysis(analysisId) {
    try {
        const response = await fetch(`http://localhost:5000/history/${analysisId}`);
        const data = await response.json();
        if (data && data.input_text) {
            ui.textInput.value = data.input_text;
            updateCharCount();
        }
    } catch (err) {
        console.warn('Could not load analysis:', err);
    }
}

// Share report
async function shareReport(analysisId) {
    if (!analysisId) {
        alert('Run an analysis first before sharing.');
        return;
    }
    try {
        const response = await fetch('http://localhost:5000/report', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ analysis_id: analysisId }),
        });
        const data = await response.json();
        if (data.share_token) {
            const url = `${window.location.origin}/report/${data.share_token}`;
            await navigator.clipboard?.writeText(url);
            alert(`Report link copied to clipboard!\n${url}`);
        }
    } catch (err) {
        console.error('Could not create report:', err);
    }
}