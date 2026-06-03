/**
 * Job status polling with exponential backoff.
 */

class StatusPoller {
    constructor(initialIntervalMs = 2000, maxIntervalMs = 30000) {
        this.initialInterval = initialIntervalMs;
        this.maxInterval = maxIntervalMs;
        this.currentInterval = initialIntervalMs;
        this.pollTimer = null;
        this.callbacks = {
            onProgress: null,
            onComplete: null,
            onError: null,
        };
    }

    async start(jobId, callbacks = {}) {
        this.jobId = jobId;
        this.callbacks = { ...this.callbacks, ...callbacks };
        this.currentInterval = this.initialInterval;

        // Poll immediately
        await this.poll();
    }

    async poll() {
        try {
            const status = await api.getStatus(this.jobId);

            // Call progress callback
            if (this.callbacks.onProgress) {
                this.callbacks.onProgress(status);
            }

            // Check if completed
            if (status.status === 'completed' || status.status === 'failed') {
                if (this.callbacks.onComplete) {
                    this.callbacks.onComplete(status);
                }
                this.stop();
                return;
            }

            // Reset interval on progress
            if (status.status === 'processing') {
                this.currentInterval = this.initialInterval;
            }

            // Schedule next poll
            this.pollTimer = setTimeout(() => this.poll(), this.currentInterval);

        } catch (error) {
            console.error('Poll error:', error);

            // Exponential backoff
            this.currentInterval = Math.min(
                this.currentInterval * 1.5,
                this.maxInterval,
            );

            // Call error callback
            if (this.callbacks.onError) {
                this.callbacks.onError(error);
            }

            // Retry
            if (this.currentInterval <= this.maxInterval) {
                this.pollTimer = setTimeout(() => this.poll(), this.currentInterval);
            }
        }
    }

    stop() {
        if (this.pollTimer) {
            clearTimeout(this.pollTimer);
            this.pollTimer = null;
        }
    }
}

// Create global poller instance
const poller = new StatusPoller();
