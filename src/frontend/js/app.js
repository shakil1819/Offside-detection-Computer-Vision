/**
 * Main application orchestrator.
 */

class App {
    constructor() {
        // DOM Elements
        this.uploadSection = document.getElementById('uploadSection');
        this.progressSection = document.getElementById('progressSection');
        this.resultsSection = document.getElementById('resultsSection');
        this.errorSection = document.getElementById('errorSection');
        this.historySection = document.getElementById('historySection');

        this.uploadBtn = document.getElementById('uploadBtn');
        this.progressFill = document.getElementById('progressFill');
        this.progressText = document.getElementById('progressText');
        this.verdictText = document.getElementById('verdictText');
        this.verdictBadge = document.getElementById('verdictBadge');
        this.confidenceText = document.getElementById('confidenceText');
        this.resultVideo = document.getElementById('resultVideo');
        this.videoSource = document.getElementById('videoSource');
        this.downloadBtn = document.getElementById('downloadBtn');
        this.newUploadBtn = document.getElementById('newUploadBtn');
        this.errorText = document.getElementById('errorText');
        this.retryBtn = document.getElementById('retryBtn');
        this.jobsList = document.getElementById('jobsList');

        this.currentJobId = null;

        this.setupEventListeners();
        this.checkHealth();
    }

    setupEventListeners() {
        this.uploadBtn.addEventListener('click', () => this.handleUpload());
        this.downloadBtn.addEventListener('click', () => this.handleDownload());
        this.newUploadBtn.addEventListener('click', () => this.resetUI());
        this.retryBtn.addEventListener('click', () => this.resetUI());
    }

    async checkHealth() {
        try {
            const health = await api.getHealth();
            console.log('Backend health:', health);
        } catch (error) {
            console.warn('Backend not available:', error);
        }
    }

    async handleUpload() {
        const file = uploader.getSelectedFile();
        if (!file) {
            this.showError('Please select a video file');
            return;
        }

        try {
            // Disable upload button
            this.uploadBtn.disabled = true;

            // Show progress section
            this.uploadSection.style.display = 'block';
            this.progressSection.style.display = 'block';
            this.resultsSection.style.display = 'none';
            this.errorSection.style.display = 'none';

            this.setProgress(5);

            // Upload file
            const response = await api.uploadVideo(file);
            this.currentJobId = response.job_id;

            this.setProgress(20);

            // Start polling
            poller.start(this.currentJobId, {
                onProgress: (status) => this.handleProgress(status),
                onComplete: (status) => this.handleComplete(status),
                onError: (error) => this.handlePollError(error),
            });

        } catch (error) {
            this.showError(`Upload failed: ${error.message}`);
            this.uploadBtn.disabled = false;
        }
    }

    handleProgress(status) {
        console.log('Job progress:', status);

        // Update progress bar
        const progress = Math.max(20, Math.min(95, status.progress || 0));
        this.setProgress(progress);

        // Update status text
        let statusText = 'Processing';
        if (status.status === 'uploading') {
            statusText = 'Uploading to Kaggle';
        } else if (status.status === 'processing') {
            statusText = `Processing (${status.progress || 0}%)`;
        }
        document.querySelector('.progress-section h3').textContent = statusText + '...';
    }

    handleComplete(status) {
        console.log('Job complete:', status);

        if (status.status === 'failed') {
            this.showError(status.error || 'Processing failed');
            this.uploadBtn.disabled = false;
            return;
        }

        this.setProgress(100);

        // Hide progress, show results
        this.progressSection.style.display = 'none';
        this.resultsSection.style.display = 'block';

        // Display verdict
        this.displayVerdict(status);

        // Load video
        this.loadResultVideo(status.job_id);

        // Store job ID
        this.currentJobId = status.job_id;

        // Reload history
        this.loadJobHistory();
    }

    handlePollError(error) {
        console.error('Poll error:', error);
        // Continue polling even on error
    }

    displayVerdict(status) {
        const verdict = status.verdict || 'UNCERTAIN';
        const confidence = status.confidence || 0;

        // Set verdict text
        this.verdictText.textContent = verdict;
        this.confidenceText.textContent = `${(confidence * 100).toFixed(0)}% confidence`;

        // Color code badge
        this.verdictBadge.className = 'verdict-badge';
        if (verdict.includes('OFFSIDE') || verdict.includes('NO-GOAL')) {
            this.verdictBadge.classList.add('offside');
        } else if (verdict.includes('ONSIDE') || verdict.includes('GOAL')) {
            this.verdictBadge.classList.add('onside');
        } else {
            this.verdictBadge.classList.add('uncertain');
        }
    }

    loadResultVideo(jobId) {
        // Construct video URL
        const videoUrl = `/download/${jobId}?format=video`;
        this.videoSource.src = videoUrl;
        this.resultVideo.load();
    }

    handleDownload() {
        if (!this.currentJobId) {
            this.showError('No job selected');
            return;
        }

        api.downloadResults(this.currentJobId, 'video');
    }

    setProgress(percent) {
        const fill = this.progressFill;
        fill.style.width = `${percent}%`;

        const text = this.progressText;
        text.textContent = `${percent}%`;
    }

    showError(message) {
        this.errorSection.style.display = 'block';
        this.errorText.textContent = message;
        this.uploadBtn.disabled = false;
    }

    resetUI() {
        uploader.reset();
        this.uploadSection.style.display = 'block';
        this.progressSection.style.display = 'none';
        this.resultsSection.style.display = 'none';
        this.errorSection.style.display = 'none';
        this.uploadBtn.disabled = false;
        this.setProgress(0);
        this.currentJobId = null;

        poller.stop();
    }

    async loadJobHistory() {
        try {
            const response = await api.listJobs(10);
            const jobs = response.jobs || [];

            if (jobs.length === 0) {
                this.historySection.style.display = 'none';
                return;
            }

            this.historySection.style.display = 'block';
            this.jobsList.innerHTML = '';

            jobs.forEach(job => {
                const jobEl = this.createJobElement(job);
                this.jobsList.appendChild(jobEl);
            });

        } catch (error) {
            console.error('Failed to load history:', error);
        }
    }

    createJobElement(job) {
        const div = document.createElement('div');
        div.className = 'job-item';

        const dateStr = new Date(job.created_at).toLocaleString();
        const verdict = job.verdict || '-';
        const confidence = job.confidence ? `${(job.confidence * 100).toFixed(0)}%` : '-';

        div.innerHTML = `
            <div>
                <div class="job-id">${job.job_id.substring(0, 12)}</div>
                <div style="font-size: 0.9rem; color: #6b7280;">${dateStr}</div>
            </div>
            <div>
                <div>${job.video_filename}</div>
                <div><span class="job-status ${job.status}">${job.status}</span></div>
            </div>
            <div class="job-verdict">
                <div style="font-size: 1.1rem; font-weight: 700;">${verdict}</div>
                <div style="font-size: 0.85rem; color: #6b7280;">${confidence}</div>
            </div>
        `;

        return div;
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.app = new App();
});
