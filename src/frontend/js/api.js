/**
 * API Client for Atlético Intelligence backend.
 */

class APIClient {
    constructor(baseUrl = '') {
        this.baseUrl = baseUrl || (window.location.origin);
        this.timeout = 30000;
    }

    async request(endpoint, options = {}) {
        const url = `${this.baseUrl}${endpoint}`;
        const fetchOptions = {
            timeout: this.timeout,
            ...options,
        };

        try {
            const response = await fetch(url, fetchOptions);

            if (!response.ok) {
                const error = await response.json().catch(() => ({}));
                throw new Error(error.detail || `HTTP ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error(`API Error: ${endpoint}`, error);
            throw error;
        }
    }

    async uploadVideo(file) {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch(`${this.baseUrl}/upload`, {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            const error = await response.json().catch(() => ({}));
            throw new Error(error.detail || `Upload failed: ${response.status}`);
        }

        return await response.json();
    }

    async getStatus(jobId) {
        return this.request(`/status/${jobId}`);
    }

    async downloadResults(jobId, format = 'video') {
        const url = `${this.baseUrl}/download/${jobId}?format=${format}`;
        window.location.href = url;
    }

    async listJobs(limit = 20) {
        return this.request(`/jobs?limit=${limit}`);
    }

    async deleteJob(jobId) {
        return this.request(`/jobs/${jobId}`, { method: 'DELETE' });
    }

    async getHealth() {
        return this.request('/health');
    }
}

// Create global API client instance
const api = new APIClient();
