/**
 * =============================================================================
 * EndeeSync - Frontend Application
 * =============================================================================
 * 
 * Production-ready vanilla JavaScript for the EndeeSync dashboard.
 * Handles API communication, UI state management, and user interactions.
 * 
 * Features:
 * - Async/await API client with error handling
 * - Loading states and skeleton placeholders
 * - Toast notifications with auto-dismiss
 * - Form validation and submission
 * - XSS protection via HTML escaping
 * 
 * @version 1.0.0
 * =============================================================================
 */

'use strict';

// =============================================================================
// Configuration
// =============================================================================

const CONFIG = Object.freeze({
    API_BASE: '',
    HEALTH_CHECK_INTERVAL: 30000,
    TOAST_DURATION: 4000,
    DEFAULT_TOP_K: 5,
    DEFAULT_SEARCH_LIMIT: 10,
    DEFAULT_SUMMARIZE_LENGTH: 500,
});

// =============================================================================
// API Client
// =============================================================================

/**
 * HTTP client for API communication.
 * Handles JSON serialization, error responses, and network failures.
 */
const ApiClient = {
    /**
     * Make an API request with automatic error handling.
     * @param {string} endpoint - API endpoint path
     * @param {Object} options - Fetch options
     * @returns {Promise<Object|null>} Response data or null for 204
     * @throws {Error} On network or API errors
     */
    async request(endpoint, options = {}) {
        const url = `${CONFIG.API_BASE}${endpoint}`;

        const config = {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers,
            },
            ...options,
        };

        try {
            const response = await fetch(url, config);

            if (!response.ok) {
                const error = await response.json().catch(() => ({
                    detail: `HTTP ${response.status}: ${response.statusText}`
                }));
                throw new Error(error.detail || error.message || 'Request failed');
            }

            if (response.status === 204) {
                return null;
            }

            return await response.json();
        } catch (error) {
            if (error.name === 'TypeError' && error.message.includes('fetch')) {
                throw new Error('Network error: Unable to connect to server');
            }
            throw error;
        }
    },

    // Health check
    async checkHealth() {
        return this.request('/health');
    },

    // Ingest endpoints
    async ingest(text, source, tags) {
        return this.request('/api/v1/ingest', {
            method: 'POST',
            body: JSON.stringify({
                text,
                source: source || null,
                tags: tags || [],
                metadata: {},
            }),
        });
    },

    async listDocuments(limit = 100) {
        return this.request(`/api/v1/ingest?limit=${limit}`);
    },

    async deleteDocument(documentId) {
        return this.request(`/api/v1/ingest/${documentId}`, {
            method: 'DELETE',
        });
    },

    // Query endpoint
    async query(question, topK = CONFIG.DEFAULT_TOP_K, tags = null) {
        const payload = {
            question,
            top_k: topK,
            include_sources: true,
        };

        if (tags?.length > 0) {
            payload.filters = { tags };
        }

        return this.request('/api/v1/query', {
            method: 'POST',
            body: JSON.stringify(payload),
        });
    },

    // Search endpoint
    async search(query, topK = CONFIG.DEFAULT_SEARCH_LIMIT, tags = null) {
        const payload = {
            query,
            top_k: topK,
            threshold: 0.0,
        };

        if (tags?.length > 0) {
            payload.filters = { tags };
        }

        return this.request('/api/v1/search', {
            method: 'POST',
            body: JSON.stringify(payload),
        });
    },

    // Summarize endpoint
    async summarize(topic, topK = 10, maxLength = CONFIG.DEFAULT_SUMMARIZE_LENGTH, tags = null) {
        const payload = {
            topic,
            top_k: topK,
            max_length: maxLength,
        };

        if (tags?.length > 0) {
            payload.filters = { tags };
        }

        return this.request('/api/v1/summarize', {
            method: 'POST',
            body: JSON.stringify(payload),
        });
    },
};

// =============================================================================
// UI Utilities
// =============================================================================

const UI = {
    /**
     * Show a toast notification.
     * @param {string} message - Notification message
     * @param {'success'|'error'|'info'|'warning'} type - Notification type
     */
    showToast(message, type = 'info') {
        const container = document.getElementById('toast-container');
        if (!container) return;

        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.setAttribute('role', 'alert');

        const icons = {
            success: '✓',
            error: '✕',
            info: 'ℹ',
            warning: '⚠'
        };

        toast.innerHTML = `<span aria-hidden="true">${icons[type] || icons.info}</span> ${this.escapeHtml(message)}`;
        container.appendChild(toast);

        setTimeout(() => {
            toast.style.opacity = '0';
            toast.style.transform = 'translateX(100%) scale(0.9)';
            toast.style.transition = 'all 0.3s ease';
            setTimeout(() => toast.remove(), 300);
        }, CONFIG.TOAST_DURATION);
    },

    /**
     * Set loading state on a button.
     * @param {HTMLButtonElement} button - Target button
     * @param {boolean} loading - Loading state
     */
    setButtonLoading(button, loading) {
        const text = button.querySelector('.btn-text');
        const spinner = button.querySelector('.btn-spinner');

        if (loading) {
            button.disabled = true;
            button.style.minWidth = `${button.offsetWidth}px`;
            text?.classList.add('hidden');
            spinner?.classList.remove('hidden');
        } else {
            button.disabled = false;
            button.style.minWidth = '';
            text?.classList.remove('hidden');
            spinner?.classList.add('hidden');
        }
    },

    /**
     * Show skeleton loading placeholder.
     * @param {HTMLElement} container - Target container
     * @param {number} lines - Number of skeleton lines
     */
    showSkeleton(container, lines = 3) {
        container.innerHTML = Array(lines)
            .fill('<div class="skeleton skeleton-text"></div>')
            .join('') + '<div class="skeleton skeleton-text" style="width: 60%"></div>';
    },

    /**
     * Escape HTML to prevent XSS attacks.
     * @param {string} text - Raw text
     * @returns {string} Escaped HTML
     */
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text || '';
        return div.innerHTML;
    },

    /**
     * Parse comma-separated tags string.
     * @param {string} input - Comma-separated tags
     * @returns {string[]} Array of trimmed tags
     */
    parseTags(input) {
        if (!input?.trim()) return [];
        return input.split(',').map(t => t.trim()).filter(Boolean);
    },

    /**
     * Format milliseconds to human-readable duration.
     * @param {number} ms - Milliseconds
     * @returns {string} Formatted duration
     */
    formatDuration(ms) {
        const value = Math.round(ms);
        return value < 1000 ? `${value}ms` : `${(ms / 1000).toFixed(1)}s`;
    },

    /**
     * Get CSS class for score visualization.
     * @param {number} score - Score between 0 and 1
     * @returns {'high'|'medium'|'low'} Score class
     */
    getScoreClass(score) {
        if (score >= 0.7) return 'high';
        if (score >= 0.4) return 'medium';
        return 'low';
    },

    /**
     * Truncate text with ellipsis.
     * @param {string} text - Input text
     * @param {number} maxLength - Maximum length
     * @returns {string} Truncated text
     */
    truncate(text, maxLength = 200) {
        if (!text || text.length <= maxLength) return text || '';
        return text.substring(0, maxLength).trim() + '…';
    },
};

// =============================================================================
// Navigation Controller
// =============================================================================

const Navigation = {
    init() {
        const navButtons = document.querySelectorAll('.nav-btn');
        const sections = document.querySelectorAll('.section');

        navButtons.forEach(btn => {
            btn.addEventListener('click', () => {
                const sectionId = btn.dataset.section;

                // Update nav buttons
                navButtons.forEach(b => {
                    b.classList.remove('active');
                    b.removeAttribute('aria-current');
                });
                btn.classList.add('active');
                btn.setAttribute('aria-current', 'page');

                // Update sections
                sections.forEach(s => s.classList.remove('active'));
                const targetSection = document.getElementById(`section-${sectionId}`);
                targetSection?.classList.add('active');
            });
        });
    },
};

// =============================================================================
// Health Status Controller
// =============================================================================

const HealthStatus = {
    init() {
        this.check();
        setInterval(() => this.check(), CONFIG.HEALTH_CHECK_INTERVAL);
    },

    async check() {
        const dot = document.querySelector('.status-dot');
        const text = document.querySelector('.status-text');

        if (!dot || !text) return;

        try {
            await ApiClient.checkHealth();
            dot.classList.add('online');
            dot.classList.remove('error');
            text.textContent = 'Connected';
        } catch {
            dot.classList.add('error');
            dot.classList.remove('online');
            text.textContent = 'Offline';
        }
    },
};

// =============================================================================
// Query Section Controller
// =============================================================================

const QuerySection = {
    init() {
        const form = document.getElementById('query-form');
        if (!form) return;

        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            await this.handleSubmit(form);
        });
    },

    async handleSubmit(form) {
        const question = document.getElementById('query-input').value.trim();
        const tags = UI.parseTags(document.getElementById('query-tags').value);
        const topK = parseInt(document.getElementById('query-topk').value, 10) || CONFIG.DEFAULT_TOP_K;

        if (!question) return;

        const submitBtn = form.querySelector('button[type="submit"]');
        const resultsDiv = document.getElementById('query-results');
        const answerDiv = document.getElementById('query-answer');
        const sourcesDiv = document.getElementById('query-sources');
        const metricsDiv = document.getElementById('query-metrics');

        UI.setButtonLoading(submitBtn, true);
        resultsDiv.classList.add('hidden');

        try {
            const data = await ApiClient.query(question, topK, tags.length > 0 ? tags : null);

            // Display answer
            answerDiv.textContent = data.answer;

            // Display metrics
            if (data.timings) {
                metricsDiv.innerHTML = `
                    <span class="metric">⚡ ${UI.formatDuration(data.timings.total_request_ms || 0)}</span>
                    <span class="metric">🔍 ${UI.formatDuration(data.timings.retrieval_ms || 0)}</span>
                    <span class="metric">🤖 ${UI.formatDuration(data.timings.llm_generation_ms || 0)}</span>
                `;
            }

            // Display sources
            sourcesDiv.innerHTML = '';
            if (data.sources?.length > 0) {
                sourcesDiv.innerHTML = data.sources.map((source, i) => `
                    <div class="source-card">
                        <div class="source-header">
                            <strong>Source ${i + 1}</strong>
                            <span class="source-score ${UI.getScoreClass(source.score)}">${Math.round(source.score * 100)}%</span>
                        </div>
                        <div class="source-text">${UI.escapeHtml(UI.truncate(source.text, 300))}</div>
                        <div class="source-meta">
                            ${source.source ? `📄 ${UI.escapeHtml(source.source)}` : ''}
                            ${source.tags?.length > 0 ? ` · 🏷️ ${source.tags.map(t => UI.escapeHtml(t)).join(', ')}` : ''}
                        </div>
                    </div>
                `).join('');
            }

            resultsDiv.classList.remove('hidden');
        } catch (error) {
            UI.showToast(`Query failed: ${error.message}`, 'error');
        } finally {
            UI.setButtonLoading(submitBtn, false);
        }
    },
};

// =============================================================================
// Notes Section Controller
// =============================================================================

const NotesSection = {
    init() {
        this.initNoteForm();
        this.initFileUpload();
        this.initRefresh();
        this.loadNotes();
    },

    initNoteForm() {
        const form = document.getElementById('note-form');
        if (!form) return;

        form.addEventListener('submit', async (e) => {
            e.preventDefault();

            const content = document.getElementById('note-content').value.trim();
            const title = document.getElementById('note-title').value.trim();
            const tags = UI.parseTags(document.getElementById('note-tags').value);

            if (!content) return;

            const submitBtn = form.querySelector('button[type="submit"]');
            UI.setButtonLoading(submitBtn, true);

            try {
                const data = await ApiClient.ingest(content, title, tags);
                UI.showToast(`Note saved! ${data.chunk_count} chunks created.`, 'success');
                form.reset();
                this.loadNotes();
            } catch (error) {
                UI.showToast(`Failed to save: ${error.message}`, 'error');
            } finally {
                UI.setButtonLoading(submitBtn, false);
            }
        });
    },

    initFileUpload() {
        const fileDrop = document.getElementById('file-drop');
        const fileInput = document.getElementById('file-input');
        const fileName = document.getElementById('file-name');
        const uploadBtn = document.getElementById('upload-btn');
        const form = document.getElementById('upload-form');

        if (!fileDrop || !fileInput) return;

        let selectedFile = null;

        // Click to upload
        fileDrop.addEventListener('click', () => fileInput.click());
        fileDrop.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                fileInput.click();
            }
        });

        // Drag and drop
        fileDrop.addEventListener('dragover', (e) => {
            e.preventDefault();
            fileDrop.classList.add('dragover');
        });

        fileDrop.addEventListener('dragleave', () => {
            fileDrop.classList.remove('dragover');
        });

        fileDrop.addEventListener('drop', (e) => {
            e.preventDefault();
            fileDrop.classList.remove('dragover');
            if (e.dataTransfer.files.length > 0) {
                handleFile(e.dataTransfer.files[0]);
            }
        });

        fileInput.addEventListener('change', () => {
            if (fileInput.files.length > 0) {
                handleFile(fileInput.files[0]);
            }
        });

        function handleFile(file) {
            const ext = file.name.split('.').pop().toLowerCase();
            if (!['txt', 'md'].includes(ext)) {
                UI.showToast('Only .txt and .md files are supported', 'error');
                return;
            }
            selectedFile = file;
            fileName.textContent = file.name;
            uploadBtn.disabled = false;
        }

        form?.addEventListener('submit', async (e) => {
            e.preventDefault();
            if (!selectedFile) return;

            const tags = UI.parseTags(document.getElementById('upload-tags').value);
            UI.setButtonLoading(uploadBtn, true);

            try {
                const content = await selectedFile.text();
                const data = await ApiClient.ingest(content, selectedFile.name, tags);
                UI.showToast(`Uploaded! ${data.chunk_count} chunks created.`, 'success');

                selectedFile = null;
                fileName.textContent = '';
                uploadBtn.disabled = true;
                form.reset();
                NotesSection.loadNotes();
            } catch (error) {
                UI.showToast(`Upload failed: ${error.message}`, 'error');
            } finally {
                UI.setButtonLoading(uploadBtn, false);
            }
        });
    },

    initRefresh() {
        const btn = document.getElementById('refresh-notes');
        btn?.addEventListener('click', () => this.loadNotes());
    },

    async loadNotes() {
        const notesList = document.getElementById('notes-list');
        if (!notesList) return;

        UI.showSkeleton(notesList, 4);

        try {
            const data = await ApiClient.listDocuments();
            const documents = data.documents || [];

            if (documents.length === 0) {
                notesList.innerHTML = '<div class="loading-placeholder">No notes yet. Add your first note above!</div>';
                return;
            }

            notesList.innerHTML = documents.map(doc => `
                <div class="note-item" data-id="${UI.escapeHtml(doc.document_id)}" role="listitem">
                    <div class="note-info">
                        <div class="note-title">${UI.escapeHtml(doc.source || 'Untitled')}</div>
                        <div class="note-meta">
                            ${doc.chunk_count} chunks
                            ${doc.tags?.length > 0 ? ` · ${doc.tags.map(t => UI.escapeHtml(t)).join(', ')}` : ''}
                        </div>
                    </div>
                    <button class="note-delete" title="Delete note" aria-label="Delete ${UI.escapeHtml(doc.source || 'note')}">🗑️</button>
                </div>
            `).join('');

            // Attach delete handlers
            notesList.querySelectorAll('.note-delete').forEach(btn => {
                btn.addEventListener('click', async (e) => {
                    e.stopPropagation();
                    const item = btn.closest('.note-item');
                    const docId = item.dataset.id;

                    try {
                        await ApiClient.deleteDocument(docId);
                        item.remove();
                        UI.showToast('Note deleted', 'success');

                        // Check if list is empty
                        if (notesList.children.length === 0) {
                            notesList.innerHTML = '<div class="loading-placeholder">No notes yet. Add your first note above!</div>';
                        }
                    } catch (error) {
                        UI.showToast(`Delete failed: ${error.message}`, 'error');
                    }
                });
            });
        } catch (error) {
            notesList.innerHTML = `<div class="loading-placeholder">Error: ${UI.escapeHtml(error.message)}</div>`;
        }
    },
};

// =============================================================================
// Search Section Controller
// =============================================================================

const SearchSection = {
    init() {
        const form = document.getElementById('search-form');
        if (!form) return;

        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            await this.handleSubmit(form);
        });
    },

    async handleSubmit(form) {
        const query = document.getElementById('search-input').value.trim();
        const tags = UI.parseTags(document.getElementById('search-tags').value);
        const limit = parseInt(document.getElementById('search-limit').value, 10) || CONFIG.DEFAULT_SEARCH_LIMIT;

        if (!query) return;

        const submitBtn = form.querySelector('button[type="submit"]');
        const resultsDiv = document.getElementById('search-results');
        const itemsDiv = document.getElementById('search-items');
        const metricsDiv = document.getElementById('search-metrics');

        UI.setButtonLoading(submitBtn, true);
        resultsDiv.classList.add('hidden');

        try {
            const data = await ApiClient.search(query, limit, tags.length > 0 ? tags : null);

            if (data.timings) {
                metricsDiv.innerHTML = `
                    <span class="metric">${data.total || 0} results</span>
                    <span class="metric">⚡ ${UI.formatDuration(data.timings.total_request_ms || 0)}</span>
                `;
            }

            if (data.results?.length > 0) {
                itemsDiv.innerHTML = data.results.map(item => `
                    <div class="search-item">
                        <div class="search-item-header">
                            <strong>${UI.escapeHtml(item.source || 'Untitled')}</strong>
                            <span class="source-score ${UI.getScoreClass(item.score)}">${Math.round(item.score * 100)}%</span>
                        </div>
                        <div class="search-item-text">${UI.escapeHtml(UI.truncate(item.text, 250))}</div>
                        <div class="source-meta">
                            ${item.tags?.length > 0 ? `🏷️ ${item.tags.map(t => UI.escapeHtml(t)).join(', ')}` : ''}
                        </div>
                    </div>
                `).join('');
            } else {
                itemsDiv.innerHTML = '<div class="loading-placeholder">No results found</div>';
            }

            resultsDiv.classList.remove('hidden');
        } catch (error) {
            UI.showToast(`Search failed: ${error.message}`, 'error');
        } finally {
            UI.setButtonLoading(submitBtn, false);
        }
    },
};

// =============================================================================
// Summarize Section Controller
// =============================================================================

const SummarizeSection = {
    init() {
        const form = document.getElementById('summarize-form');
        if (!form) return;

        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            await this.handleSubmit(form);
        });
    },

    async handleSubmit(form) {
        const topic = document.getElementById('summarize-topic').value.trim();
        const tags = UI.parseTags(document.getElementById('summarize-tags').value);
        const topK = parseInt(document.getElementById('summarize-topk').value, 10) || 10;
        const maxLength = parseInt(document.getElementById('summarize-length').value, 10) || CONFIG.DEFAULT_SUMMARIZE_LENGTH;

        if (!topic) return;

        const submitBtn = form.querySelector('button[type="submit"]');
        const resultsDiv = document.getElementById('summarize-results');
        const contentDiv = document.getElementById('summarize-content');
        const metricsDiv = document.getElementById('summarize-metrics');
        const statsDiv = document.getElementById('summarize-stats');

        UI.setButtonLoading(submitBtn, true);
        resultsDiv.classList.add('hidden');

        try {
            const data = await ApiClient.summarize(topic, topK, maxLength, tags.length > 0 ? tags : null);

            contentDiv.textContent = data.summary;

            if (data.timings) {
                metricsDiv.innerHTML = `
                    <span class="metric">⚡ ${UI.formatDuration(data.timings.total_request_ms || 0)}</span>
                    <span class="metric">🔍 ${UI.formatDuration(data.timings.retrieval_ms || 0)}</span>
                    <span class="metric">🤖 ${UI.formatDuration(data.timings.llm_generation_ms || 0)}</span>
                `;
            }

            statsDiv.innerHTML = `
                <span class="metric">📄 ${data.chunk_count || 0} chunks used</span>
                <span class="metric">🏷️ ${UI.escapeHtml(data.topic)}</span>
            `;

            resultsDiv.classList.remove('hidden');
        } catch (error) {
            UI.showToast(`Summarize failed: ${error.message}`, 'error');
        } finally {
            UI.setButtonLoading(submitBtn, false);
        }
    },
};

// =============================================================================
// Application Entry Point
// =============================================================================

document.addEventListener('DOMContentLoaded', () => {
    // Initialize all modules
    Navigation.init();
    HealthStatus.init();
    QuerySection.init();
    NotesSection.init();
    SearchSection.init();
    SummarizeSection.init();

    console.log('🧠 EndeeSync initialized');
});
