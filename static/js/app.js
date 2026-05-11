/* ═══════════════════════════════════════════════════════════════════════
   PROSTHETIC HAND CONTROL — CLIENT APP
   ═══════════════════════════════════════════════════════════════════════ */

// ─── State ──────────────────────────────────────────────────────────
let movements = [];
let currentFilter = 'all';
let systemReady = false;
let socket = null;

// ─── DOM References ─────────────────────────────────────────────────
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

// ─── Initialize ─────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    initSocket();
    loadMovements();
    checkStatus();
    initFilterTabs();
});

// ═══════════════════════════════════════════════════════════════════════
// SOCKET.IO CONNECTION
// ═══════════════════════════════════════════════════════════════════════
function initSocket() {
    try {
        socket = io();

        socket.on('connect', () => {
            console.log('🔌 Socket connected');
        });

        socket.on('status_update', (data) => {
            updatePipeline(data.state, data.message);
            updateOverlay(data.state, data.message);
        });

        socket.on('inference_result', (data) => {
            showResults(data);
        });

        socket.on('servo_update', (data) => {
            updateServoGauges(data.angles);
        });
    } catch (e) {
        console.warn('Socket.IO not available, using polling');
    }
}

// ═══════════════════════════════════════════════════════════════════════
// DATA LOADING
// ═══════════════════════════════════════════════════════════════════════
async function loadMovements() {
    try {
        const res = await fetch('/api/movements');
        movements = await res.json();
        renderMovementCards();
    } catch (e) {
        console.error('Failed to load movements:', e);
    }
}

async function checkStatus() {
    try {
        const res = await fetch('/api/status');
        const status = await res.json();

        const badge = $('#system-status-badge');
        const modeBadge = $('#servo-mode-badge');

        if (status.initialized) {
            badge.className = 'status-badge ready';
            badge.querySelector('.status-text').textContent = 'System Ready';
            systemReady = true;
        } else {
            badge.className = 'status-badge error';
            badge.querySelector('.status-text').textContent = 'Not Initialized';
        }

        if (status.servo_mode === 'hardware') {
            modeBadge.className = 'servo-mode-badge hardware';
            modeBadge.querySelector('span').textContent = 'HW';
        } else {
            modeBadge.className = 'servo-mode-badge';
            modeBadge.querySelector('span').textContent = 'SIM';
        }
    } catch (e) {
        const badge = $('#system-status-badge');
        badge.className = 'status-badge error';
        badge.querySelector('.status-text').textContent = 'Connection Error';
    }
}

// ═══════════════════════════════════════════════════════════════════════
// RENDER MOVEMENT CARDS
// ═══════════════════════════════════════════════════════════════════════
function renderMovementCards() {
    const grid = $('#movements-grid');
    grid.innerHTML = '';

    const exerciseNames = { 1: 'Finger', 2: 'Hand', 3: 'Grasp' };

    movements.forEach((mov, i) => {
        // Filter check
        if (currentFilter !== 'all' && mov.exercise !== parseInt(currentFilter)) {
            return;
        }

        const card = document.createElement('div');
        card.className = 'movement-card';
        card.style.animationDelay = `${i * 0.03}s`;
        card.dataset.index = mov.encoded;
        card.id = `movement-card-${mov.encoded}`;

        const exClass = `ex${mov.exercise}`;
        const exName = exerciseNames[mov.exercise] || `E${mov.exercise}`;

        let imageContent;
        if (mov.image) {
            imageContent = `<img src="${mov.image}" alt="${mov.name}" loading="lazy">`;
        } else {
            imageContent = `
                <div class="card-image-placeholder">
                    <span class="placeholder-icon">✋</span>
                    <span>Label ${mov.original}</span>
                </div>`;
        }

        card.innerHTML = `
            <div class="card-number">${mov.encoded + 1}</div>
            <div class="card-exercise ${exClass}">${exName}</div>
            <div class="card-image">${imageContent}</div>
            <div class="card-body">
                <div class="card-title">${mov.name}</div>
                <div class="card-meta">
                    <span class="card-label">Label ${mov.original}</span>
                    <span class="card-action">Execute →</span>
                </div>
            </div>
        `;

        card.addEventListener('click', () => executeMovement(mov.encoded));
        grid.appendChild(card);
    });
}

// ═══════════════════════════════════════════════════════════════════════
// FILTER TABS
// ═══════════════════════════════════════════════════════════════════════
function initFilterTabs() {
    $$('.filter-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            $$('.filter-tab').forEach(t => t.classList.remove('active'));
            tab.classList.add('active');

            currentFilter = tab.dataset.filter;

            // Update description
            $$('.filter-desc').forEach(d => d.classList.remove('active'));
            const desc = $(`.filter-desc[data-for="${currentFilter}"]`);
            if (desc) desc.classList.add('active');

            renderMovementCards();
        });
    });
}

// ═══════════════════════════════════════════════════════════════════════
// EXECUTE MOVEMENT
// ═══════════════════════════════════════════════════════════════════════
async function executeMovement(index) {
    if (!systemReady) {
        alert('System is not ready yet. Please wait for initialization.');
        return;
    }

    // Show pipeline and overlay
    $('#pipeline-section').classList.remove('hidden');
    $('#execution-overlay').classList.remove('hidden');
    $('#overlay-title').textContent = `Executing: ${movements[index].name}`;
    $('#overlay-message').textContent = 'Starting...';

    // Disable all cards
    $$('.movement-card').forEach(c => c.classList.add('disabled'));

    try {
        const res = await fetch(`/api/execute/${index}`, { method: 'POST' });
        const data = await res.json();

        if (data.error) {
            alert(`Error: ${data.error}`);
            resetUI();
        }
        // The rest of the flow is handled by socket events
    } catch (e) {
        console.error('Execution failed:', e);
        alert('Failed to execute movement. Check the console.');
        resetUI();
    }
}

// ═══════════════════════════════════════════════════════════════════════
// PIPELINE VISUALIZATION
// ═══════════════════════════════════════════════════════════════════════
const PIPELINE_ORDER = ['sampling', 'preprocessing', 'inferring', 'moving', 'holding', 'resting'];

function updatePipeline(state, message) {
    const stateIndex = PIPELINE_ORDER.indexOf(state);

    $$('.pipeline-step').forEach((step, i) => {
        step.classList.remove('active', 'done');
        if (i < stateIndex) step.classList.add('done');
        else if (i === stateIndex) step.classList.add('active');
    });

    $$('.pipeline-connector').forEach((conn, i) => {
        conn.classList.toggle('done', i < stateIndex);
    });

    $('#pipeline-message').textContent = message;
}

function updateOverlay(state, message) {
    const overlay = $('#execution-overlay');
    const overlayMsg = $('#overlay-message');

    overlayMsg.textContent = message;

    if (state === 'ready' || state === 'error') {
        setTimeout(() => {
            overlay.classList.add('hidden');
            resetUI();
        }, 600);
    }
}

function resetUI() {
    $('#execution-overlay').classList.add('hidden');
    $$('.movement-card').forEach(c => c.classList.remove('disabled'));
}

// ═══════════════════════════════════════════════════════════════════════
// RESULTS DISPLAY
// ═══════════════════════════════════════════════════════════════════════
function showResults(data) {
    const resultsSection = $('#results-section');
    resultsSection.classList.remove('hidden');

    // Inference results
    $('#result-selected').textContent = data.selected.name;
    $('#result-predicted').textContent = data.prediction.name;

    const conf = (data.prediction.confidence * 100).toFixed(1);
    const confBar = $('#confidence-bar');
    confBar.style.setProperty('--confidence', `${conf}%`);
    $('#confidence-value').textContent = `${conf}%`;

    $('#result-time').textContent = `${data.prediction.inference_time_ms.toFixed(1)} ms`;

    const matchEl = $('#result-match');
    if (data.prediction.correct) {
        matchEl.textContent = '✅ Correct';
        matchEl.className = 'result-value correct';
    } else {
        matchEl.textContent = '❌ Mismatch';
        matchEl.className = 'result-value incorrect';
    }

    // Sample badge
    $('#sample-badge').textContent = `Sample ${data.sample_index}/${data.total_samples}`;

    // Draw EMG
    drawEMG(data.emg_data);

    // Update servo gauges
    const predictedMovement = movements[data.prediction.index];
    if (predictedMovement) {
        updateServoGauges(predictedMovement.angles);
    }
}

// ═══════════════════════════════════════════════════════════════════════
// SERVO GAUGES
// ═══════════════════════════════════════════════════════════════════════
function updateServoGauges(angles) {
    const fingers = ['thumb', 'index', 'middle', 'ring', 'little'];
    fingers.forEach(name => {
        const gauge = $(`#gauge-${name}`);
        if (!gauge) return;
        const angle = angles[name] || 0;
        const pct = (angle / 180) * 100;
        gauge.querySelector('.gauge-fill').style.setProperty('--fill', `${pct}%`);
        gauge.querySelector('.gauge-value').textContent = `${angle}°`;
    });
}

// ═══════════════════════════════════════════════════════════════════════
// EMG VISUALIZATION
// ═══════════════════════════════════════════════════════════════════════
function drawEMG(emgData) {
    const canvas = $('#emg-canvas');
    if (!canvas || !emgData || emgData.length === 0) return;

    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;

    canvas.width = canvas.offsetWidth * dpr;
    canvas.height = canvas.offsetHeight * dpr;
    ctx.scale(dpr, dpr);

    const W = canvas.offsetWidth;
    const H = canvas.offsetHeight;

    ctx.clearRect(0, 0, W, H);

    // Background grid
    ctx.strokeStyle = 'rgba(255,255,255,0.04)';
    ctx.lineWidth = 1;
    for (let y = 0; y < H; y += 20) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(W, y);
        ctx.stroke();
    }
    for (let x = 0; x < W; x += 20) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, H);
        ctx.stroke();
    }

    const nSamples = emgData.length;
    const nChannels = emgData[0].length;
    const channelsToShow = Math.min(nChannels, 10);

    // Find global min/max for scaling
    let globalMin = Infinity, globalMax = -Infinity;
    for (let t = 0; t < nSamples; t++) {
        for (let ch = 0; ch < channelsToShow; ch++) {
            const v = emgData[t][ch];
            if (v < globalMin) globalMin = v;
            if (v > globalMax) globalMax = v;
        }
    }
    const range = globalMax - globalMin || 1;

    // Color palette for channels
    const colors = [
        '#6C5CE7', '#00CEC9', '#FD79A8', '#FDCB6E', '#00B894',
        '#E17055', '#0984E3', '#A29BFE', '#81ECEC', '#FAB1A0'
    ];

    // Draw each channel
    for (let ch = 0; ch < channelsToShow; ch++) {
        ctx.beginPath();
        ctx.strokeStyle = colors[ch % colors.length];
        ctx.lineWidth = 1.5;
        ctx.globalAlpha = 0.7;

        for (let t = 0; t < nSamples; t++) {
            const x = (t / (nSamples - 1)) * W;
            const normalized = (emgData[t][ch] - globalMin) / range;
            const y = H - (normalized * H * 0.85 + H * 0.075);

            if (t === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();
    }

    ctx.globalAlpha = 1.0;

    // Labels
    ctx.fillStyle = 'rgba(255,255,255,0.25)';
    ctx.font = '10px Inter, sans-serif';
    ctx.fillText(`${nSamples} samples × ${channelsToShow} channels`, 8, H - 6);
}
