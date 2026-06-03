/**
 * Video file uploader with drag-and-drop support.
 */

class Uploader {
    constructor() {
        this.videoInput = document.getElementById('videoInput');
        this.uploadBtn = document.getElementById('uploadBtn');
        this.fileNameDisplay = document.getElementById('fileName');
        this.uploadBox = document.querySelector('.upload-box');
        this.selectedFile = null;

        this.setupEventListeners();
    }

    setupEventListeners() {
        // Button click
        this.uploadBtn.addEventListener('click', () => {
            this.videoInput.click();
        });

        // File input change
        this.videoInput.addEventListener('change', (e) => {
            this.handleFileSelected(e.target.files[0]);
        });

        // Drag and drop
        this.uploadBox.addEventListener('dragover', (e) => {
            e.preventDefault();
            this.uploadBox.classList.add('dragover');
        });

        this.uploadBox.addEventListener('dragleave', () => {
            this.uploadBox.classList.remove('dragover');
        });

        this.uploadBox.addEventListener('drop', (e) => {
            e.preventDefault();
            this.uploadBox.classList.remove('dragover');

            const files = e.dataTransfer.files;
            if (files.length > 0) {
                this.handleFileSelected(files[0]);
            }
        });
    }

    handleFileSelected(file) {
        if (!file) return;

        // Validate file
        const error = this.validateFile(file);
        if (error) {
            this.showError(error);
            this.videoInput.value = '';
            return;
        }

        this.selectedFile = file;
        this.updateDisplay();
    }

    validateFile(file) {
        // Check MIME type
        const allowedMimes = ['video/mp4', 'video/avi', 'video/quicktime', 'video/x-matroska'];
        if (!allowedMimes.some(mime => file.type.startsWith('video'))) {
            return `Invalid file type: ${file.type}`;
        }

        // Check size (500 MB)
        const maxSizeBytes = 500 * 1024 * 1024;
        if (file.size > maxSizeBytes) {
            const sizeMB = (file.size / (1024 * 1024)).toFixed(1);
            return `File too large: ${sizeMB}MB > 500MB`;
        }

        return null;
    }

    updateDisplay() {
        if (this.selectedFile) {
            const sizeMB = (this.selectedFile.size / (1024 * 1024)).toFixed(1);
            this.fileNameDisplay.textContent = `✓ ${this.selectedFile.name} (${sizeMB}MB)`;
            this.fileNameDisplay.style.color = '#10b981';
            this.uploadBtn.textContent = '🚀 Upload & Process';
        } else {
            this.fileNameDisplay.textContent = '';
            this.uploadBtn.textContent = '📹 Choose Video';
        }
    }

    showError(message) {
        this.fileNameDisplay.textContent = `❌ ${message}`;
        this.fileNameDisplay.style.color = '#ef4444';
        setTimeout(() => {
            this.fileNameDisplay.textContent = '';
        }, 5000);
    }

    getSelectedFile() {
        return this.selectedFile;
    }

    reset() {
        this.selectedFile = null;
        this.videoInput.value = '';
        this.fileNameDisplay.textContent = '';
        this.fileNameDisplay.style.color = '';
        this.updateDisplay();
    }
}

// Create global uploader instance
const uploader = new Uploader();
