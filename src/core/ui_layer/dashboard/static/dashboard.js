/**
 * AI Network Optimizer - Dashboard Logic
 * Handles WebSocket communication and REST API calls for intents.
 */

// --- API Functionality ---

async function sendIntent() {
    const input = document.getElementById('userIntent');
    const text = input.value.trim();
    
    if (!text) return;

    // Visual feedback
    const btn = document.getElementById('sendIntentBtn');
    const originalText = btn.innerText;
    btn.innerText = "Sending...";
    btn.disabled = true;

    try {
        const response = await fetch('/api/intent', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ text: text })
        });

        const data = await response.json();
        console.log("Intent Sent:", data);
        
        if (data.status === "ok") {
            input.value = ""; // Clear input on success
        } else {
            alert("Error: " + (data.msg || "Unknown error"));
        }

    } catch (error) {
        console.error("Error sending intent:", error);
        alert("Failed to send intent. Is the backend running?");
    } finally {
        btn.innerText = originalText;
        btn.disabled = false;
    }
}


// --- WebSocket Client ---

class DashboardClient {
    constructor() {
        this.ws = null;
        this.reconnectInterval = 3000;
        this.maxReconnectAttempts = 10;
        this.reconnectAttempts = 0;

        // Data State
        this.intents = new Map();
        this.deviations = [];
        this.commands = [];
        this.maxHistoryItems = 50;
        
        // NEW: Tracker for fixing the ns-3 cumulative throughput bug


        // UI References
        this.statusDot = document.getElementById('statusDot');
        this.statusText = document.getElementById('statusText');
        this.intentsList = document.getElementById('intentsList');
        this.deviationsList = document.getElementById('deviationsList');
        this.commandsList = document.getElementById('commandsList');
        this.intentCount = document.getElementById('intentCount');
        this.deviationCount = document.getElementById('deviationCount');
        this.commandCount = document.getElementById('commandCount');
        this.gnbLatency = document.getElementById('gnbLatency');
        this.cellId = document.getElementById('cellId');
        this.ueCountBadge = document.getElementById('ueCountBadge');
        this.ueMetricsList = document.getElementById('ueMetricsList');
        this.gnbState = { latency: "--", cellId: "--" };
        this.ueState = new Map(); // Key: ueId, Value: ueData

        this.connect();
    }

    connect() {
        // Automatically determine WS protocol and host based on current page
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;

        try {
            console.log(`Connecting to WebSocket at ${wsUrl}...`);
            this.ws = new WebSocket(wsUrl);

            this.ws.onopen = () => this.onConnect();
            this.ws.onmessage = (event) => this.onMessage(event);
            this.ws.onclose = () => this.onDisconnect();
            this.ws.onerror = (error) => this.onError(error);
        } catch (error) {
            console.error('WebSocket connection setup error:', error);
            this.scheduleReconnect();
        }
    }

    onConnect() {
        console.log('Connected to AI Dashboard');
        this.reconnectAttempts = 0;
        this.updateConnectionStatus(true);
    }

    onDisconnect() {
        console.log('Disconnected from AI Dashboard');
        this.updateConnectionStatus(false);
        this.scheduleReconnect();
    }

    onError(error) {
        console.error('WebSocket error:', error);
    }

    onMessage(event) {
        try {
            const message = JSON.parse(event.data);

            switch (message.type) {
                case 'connected':
                    console.log('Backend Handshake:', message.message);
                    break;
                case 'intent':
                    this.handleIntent(message.data);
                    break;
                case 'deviation':
                    this.handleDeviation(message.data);
                    break;
                case 'command':
                    this.handleCommand(message.data);
                    break;
                case 'kpi':
                    this.handleKPI(message.data);
                    break;
                default:
                    console.warn('Unknown message type:', message.type);
            }
        } catch (error) {
            console.error('Error parsing message:', error);
        }
    }

    scheduleReconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            console.log(`Reconnecting in ${this.reconnectInterval}ms (attempt ${this.reconnectAttempts})`);
            setTimeout(() => this.connect(), this.reconnectInterval);
        }
    }

    updateConnectionStatus(connected) {
        if (connected) {
            this.statusDot.classList.add('connected');
            this.statusText.textContent = 'Connected';
        } else {
            this.statusDot.classList.remove('connected');
            this.statusText.textContent = 'Disconnected';
        }
    }

    // --- Handlers ---

    handleIntent(data) {
        // This receives the message from DashboardServer
        if (!data || Object.keys(data).length === 0) {
            this.intents.clear();
        } else {
            const intentId = data.intent_id || 'unknown';
            this.intents.set(intentId, data);
        }
        this.renderIntents();
    }

    renderIntents() {
        this.intentCount.textContent = this.intents.size;

        if (this.intents.size === 0) {
            this.intentsList.innerHTML = '<div class="empty-state">No active intents</div>';
            return;
        }

        const html = Array.from(this.intents.values()).map(intent => {
            const procedureSteps = (intent.procedure || []).map(step => {
                if (typeof step === 'object' && step !== null) {
                    const action = this.escapeHtml(step.action || '');
                    const rationale = step.rationale ? `<div class="step-rationale">${this.escapeHtml(step.rationale)}</div>` : '';
                    return `<li><strong>${action}</strong>${rationale}</li>`;
                }
                return `<li>${this.escapeHtml(step)}</li>`;
            }).join('');

            const procedure = procedureSteps.length > 0
                ? `<ol class="procedure-steps">${procedureSteps}</ol>`
                : `<p class="no-data">Decomposing requirements...</p>`;

            return `
                <div class="intent-card hierarchical">
                    <div class="intent-description-header">
                        <strong>Intent:</strong> ${this.escapeHtml(intent.intent_name || 'Optimizing Network Performance')}
                    </div>

                    <div class="intent-label-small">Resulting Procedure</div>
                    <div class="intent-procedure-container">
                        ${procedure}
                    </div>
                    
                    </div>
            `;
        }).join('');

        this.intentsList.innerHTML = html;
    }

    handleDeviation(data) {
        this.deviations.unshift({
            ...data,
            timestamp: new Date().toISOString()
        });

        if (this.deviations.length > this.maxHistoryItems) {
            this.deviations = this.deviations.slice(0, this.maxHistoryItems);
        }

        this.renderDeviations();
    }

    handleCommand(data) {
        this.commands.unshift({
            ...data,
            timestamp: new Date().toISOString()
        });

        if (this.commands.length > this.maxHistoryItems) {
            this.commands = this.commands.slice(0, this.maxHistoryItems);
        }

        this.renderCommands();
    }

    handleKPI(data) {
        const { cell, ues } = data;

        // Update gNB metrics
        if (cell && cell.DRB_PdcpSduDelayDl !== undefined) {
            this.gnbLatency.textContent = cell.DRB_PdcpSduDelayDl.toFixed(2);
        }
        if (cell && cell.cell_id) {
            this.cellId.textContent = cell.cell_id;
        }

        // Update UE metrics
        if (ues && ues.length > 0) {
            this.ueCountBadge.textContent = ues.length;
            this.renderUEMetrics(ues);
        } else {
            this.ueCountBadge.textContent = '0';
            this.ueMetricsList.innerHTML = '<div class="empty-state">No UE data</div>';
        }
    }

    // --- Rendering ---

    renderUEMetrics(ues) {
        const html = ues.map(ue => {
            const ueId = ue.ue_id || 'Unknown';
            const latency = ue.UE_DRB_PdcpSduDelayDl_UEID;
            
            const currentRawThp = ue.UE_DRB_UEThpDl_UEID || 0;
            const trueThpMbps = currentRawThp / 1e6;

            return `
                <div class="ue-card">
                    <div class="ue-id">📱 UE: ${this.escapeHtml(ueId)}</div>
                    ${latency !== undefined ? `
                        <div class="ue-metric-row">
                            <span class="label">Latency:</span>
                            <span class="value">${latency.toFixed(2)} ms</span>
                        </div>
                    ` : ''}
                    <div class="ue-metric-row">
                        <span class="label">Throughput:</span>
                        <span class="value">${trueThpMbps.toFixed(2)} Mbps</span>
                    </div>
                </div>
            `;
        }).join('');

        this.ueMetricsList.innerHTML = html;
    }

    renderDeviations() {
        this.deviationCount.textContent = this.deviations.length;

        if (this.deviations.length === 0) {
            this.deviationsList.innerHTML = '<div class="empty-state">No deviations detected</div>';
            return;
        }

        const html = this.deviations.slice(0, 20).map(dev => `
            <div class="deviation-item">
                <div class="deviation-metric">${this.escapeHtml(dev.metric || 'Unknown')}</div>
                <div class="deviation-value">Val: ${dev.value !== undefined ? dev.value.toFixed(2) : 'N/A'}</div>
                <div class="deviation-time">${this.formatTime(dev.timestamp)}</div>
            </div>
        `).join('');

        this.deviationsList.innerHTML = html;
    }

    renderCommands() {
        this.commandCount.textContent = this.commands.length;

        if (this.commands.length === 0) {
            this.commandsList.innerHTML = '<div class="empty-state">No commands sent</div>';
            return;
        }

        const html = this.commands.slice(0, 20).map(cmd => `
            <div class="command-item">
                <div class="command-name">${this.escapeHtml(cmd.command || 'Unknown')}</div>
                <div class="command-params">${this.formatParams(cmd.params)}</div>
                <div class="command-time">${this.formatTime(cmd.timestamp)}</div>
            </div>
        `).join('');

        this.commandsList.innerHTML = html;
    }

    // --- Helpers ---

    formatParams(params) {
        if (!params) return 'N/A';
        return JSON.stringify(params);
    }

    formatTime(timestamp) {
        if (!timestamp) return '';
        const date = new Date(timestamp);
        return date.toLocaleTimeString();
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// --- Initialization ---

document.addEventListener('DOMContentLoaded', () => {
    // 1. Initialize WebSocket Client
    new DashboardClient();

    // 2. Attach Event Listener to Button
    const btn = document.getElementById('sendIntentBtn');
    if(btn) {
        btn.addEventListener('click', sendIntent);
    }

    // 3. Allow "Enter" key in input field
    const input = document.getElementById('userIntent');
    if(input) {
        input.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') sendIntent();
        });
    }
});