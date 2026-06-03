/**
 * Main application orchestrator.
 */

class App {
    constructor() {
        // Upload
        this.uploadSection   = document.getElementById('uploadSection');
        this.progressSection = document.getElementById('progressSection');
        this.uploadBtn       = document.getElementById('uploadBtn');
        this.progressFill    = document.getElementById('progressFill');
        this.progressText    = document.getElementById('progressText');
        this.progressLabel   = document.getElementById('progressLabel');
        this.progressStatus  = document.getElementById('progressStatus');
        this.incidentType    = document.getElementById('incidentType');

        // Results
        this.resultsSection  = document.getElementById('resultsSection');
        this.verdictBadge    = document.getElementById('verdictBadge');
        this.verdictText     = document.getElementById('verdictText');
        this.confidenceText  = document.getElementById('confidenceText');

        // Media
        this.freezeImg       = document.getElementById('freezeImg');
        this.resultVideo     = document.getElementById('resultVideo');
        this.videoSource     = document.getElementById('videoSource');
        this.diagramImg      = document.getElementById('diagramImg');

        // Download links
        this.downloadVideoBtn  = document.getElementById('downloadVideoBtn');
        this.downloadFreezeBtn = document.getElementById('downloadFreezeBtn');

        // Other
        this.errorSection   = document.getElementById('errorSection');
        this.errorText      = document.getElementById('errorText');
        this.retryBtn       = document.getElementById('retryBtn');
        this.newUploadBtn   = document.getElementById('newUploadBtn');
        this.historySection = document.getElementById('historySection');
        this.jobsList       = document.getElementById('jobsList');

        this.currentJobId = null;

        this._setupTabs();
        this._setupListeners();
        this._loadHistory();
    }

    _setupListeners() {
        this.uploadBtn.addEventListener('click',   () => this._handleUpload());
        this.newUploadBtn.addEventListener('click', () => this._resetUI());
        this.retryBtn.addEventListener('click',    () => this._resetUI());
    }

    _setupTabs() {
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const tab = btn.dataset.tab;
                document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
                document.querySelectorAll('.tab-content').forEach(c => c.style.display = 'none');
                btn.classList.add('active');
                document.getElementById(`tab-${tab}`).style.display = 'block';

                // Lazy-load video only when its tab is opened
                if (tab === 'video' && this.videoSource.src !== window.location.href) {
                    this.resultVideo.load();
                }
            });
        });
    }

    async _handleUpload() {
        const file = uploader.getSelectedFile();
        if (!file) {
            uploader.showError('Please choose a video file first');
            return;
        }

        this.uploadBtn.disabled = true;
        this.progressSection.style.display = 'block';
        this.resultsSection.style.display  = 'none';
        this.errorSection.style.display    = 'none';
        this._setProgress(5, 'Uploading...');

        try {
            const resp = await api.uploadVideo(file, this.incidentType.value);
            this.currentJobId = resp.job_id;
            this._setProgress(15, 'Queued for processing');

            poller.start(this.currentJobId, {
                onProgress: s => this._onProgress(s),
                onComplete: s => this._onComplete(s),
                onError:    e => console.warn('Poll error:', e),
            });
        } catch (e) {
            this._showError(`Upload failed: ${e.message}`);
        }
    }

    _onProgress(status) {
        const p = Math.max(15, Math.min(95, status.progress || 0));
        const labels = {
            uploading:  'Uploading to Kaggle...',
            processing: `Running on Kaggle T4 GPU (${status.progress || 0}%)`,
        };
        this._setProgress(p, labels[status.status] || 'Processing...');
        if (this.progressStatus) {
            this.progressStatus.textContent = status.status;
        }
    }

    _onComplete(status) {
        if (status.status === 'failed') {
            this._showError(status.error || 'Processing failed on Kaggle kernel');
            return;
        }

        this._setProgress(100, 'Complete');
        this.progressSection.style.display = 'none';
        this.resultsSection.style.display  = 'block';

        this._displayVerdict(status);
        this._loadOutputs(status.job_id);
        this._loadHistory();
    }

    _displayVerdict(status) {
        const verdict    = status.verdict || 'UNCERTAIN';
        const confidence = status.confidence || 0;

        this.verdictText.textContent     = verdict.replace('_', ' ');
        this.confidenceText.textContent  = `${(confidence * 100).toFixed(0)}% confidence`;

        this.verdictBadge.className = 'verdict-badge';
        if (['OFFSIDE', 'NO-GOAL', 'NO_GOAL'].includes(verdict)) {
            this.verdictBadge.classList.add('offside');
        } else {
            this.verdictBadge.classList.add('onside');   // ONSIDE or GOAL
        }
    }

    _loadOutputs(jobId) {
        const base = `/download/${jobId}`;

        // Freeze frame — show immediately in active tab
        const freezeUrl = `${base}?format=freeze`;
        this.freezeImg.src = freezeUrl;
        this.freezeImg.onerror = () => {
            document.getElementById('tab-freeze').innerHTML =
                '<p style="color:#888;padding:20px;">Freeze frame not available</p>';
        };

        // Diagram
        this.diagramImg.src = `${base}?format=diagram`;
        this.diagramImg.onerror = () => {
            document.getElementById('tab-diagram').innerHTML =
                '<p style="color:#888;padding:20px;">Diagram not available</p>';
        };

        // Video — set src but don't load until tab is opened
        const videoUrl = `${base}?format=video`;
        this.videoSource.src = videoUrl;

        // Download buttons
        this.downloadVideoBtn.href  = videoUrl;
        this.downloadFreezeBtn.href = freezeUrl;
    }

    _setProgress(pct, label = '') {
        this.progressFill.style.width   = `${pct}%`;
        this.progressText.textContent   = `${pct}%`;
        if (label && this.progressLabel) this.progressLabel.textContent = label;
    }

    _showError(msg) {
        this.errorSection.style.display  = 'block';
        this.resultsSection.style.display = 'none';
        this.errorText.textContent = msg;
        this.uploadBtn.disabled = false;
    }

    _resetUI() {
        uploader.reset();
        poller.stop();
        this.uploadBtn.disabled        = false;
        this.progressSection.style.display = 'none';
        this.resultsSection.style.display  = 'none';
        this.errorSection.style.display    = 'none';
        this._setProgress(0, 'Processing on Kaggle T4 GPU...');
        this.currentJobId = null;
    }

    async _loadHistory() {
        try {
            const resp = await api.listJobs(8);
            const jobs  = (resp.jobs || []).filter(j => j.status === 'completed' || j.status === 'failed');
            if (!jobs.length) { this.historySection.style.display = 'none'; return; }

            this.historySection.style.display = 'block';
            this.jobsList.innerHTML = jobs.map(j => {
                const icon    = j.status === 'completed' ? '✅' : '❌';
                const verdict = j.verdict || '—';
                const conf    = j.confidence != null ? `${(j.confidence * 100).toFixed(0)}%` : '';
                const date    = new Date(j.created_at).toLocaleString();
                return `
                    <div class="job-item" onclick="app._reloadJob('${j.job_id}')">
                        <div>${icon} <strong>${verdict}</strong> ${conf}</div>
                        <div class="job-id">${j.job_id.slice(0,12)}</div>
                        <div style="font-size:0.8rem;color:#888;">${date}</div>
                    </div>`;
            }).join('');
        } catch (e) {
            console.warn('History load failed:', e);
        }
    }

    _reloadJob(jobId) {
        // Re-display results for a past completed job
        api.getStatus(jobId).then(status => {
            if (status.status !== 'completed') return;
            this.currentJobId = jobId;
            this.progressSection.style.display = 'none';
            this.resultsSection.style.display  = 'block';
            this.errorSection.style.display    = 'none';
            this._displayVerdict(status);
            this._loadOutputs(jobId);
        });
    }
}

document.addEventListener('DOMContentLoaded', () => { window.app = new App(); });
