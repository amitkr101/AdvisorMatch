const API_BASE_URL = 'http://localhost:8000';

const searchInput = document.getElementById('searchInput');
const searchButton = document.getElementById('searchButton');
const loadingSpinner = document.getElementById('loadingSpinner');
const errorMessage = document.getElementById('errorMessage');
const resultsSection = document.getElementById('resultsSection');
const resultsInfo = document.getElementById('resultsInfo');
const resultsContainer = document.getElementById('resultsContainer');

searchButton.addEventListener('click', performSearch);
searchInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') performSearch();
});

async function performSearch() {
    const query = searchInput.value.trim();
    if (!query) return;

    showLoading();
    hideError();
    hideResults();

    try {
        const response = await fetch(`${API_BASE_URL}/api/bm25/search`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: query, top_k: 20 })
        });

        if (!response.ok) throw new Error(`API error: ${response.status}`);

        const data = await response.json();
        displayResults(data);

    } catch (error) {
        showError(`Search failed: ${error.message}`);
    } finally {
        hideLoading();
    }
}

function displayResults(data) {
    if (!data.results || data.results.length === 0) {
        showError('No matches found.');
        return;
    }

    resultsInfo.textContent = `Found ${data.total_results} advisors for "${data.query}" (${data.search_time_ms.toFixed(0)}ms)`;
    resultsContainer.innerHTML = '';

    data.results.forEach(prof => {
        const card = document.createElement('div');
        card.className = 'professor-card';

        // Generate publications HTML
        const pubsHtml = prof.top_publications.map(pub => `
            <div class="publication-item">
                <div class="publication-title">
                    ${pub.url ? `<a href="${pub.url}" target="_blank" style="color: inherit; text-decoration: none; border-bottom: 1px dotted #666;">${pub.title}</a>` : pub.title}
                </div>
                <div class="publication-meta">
                    <span class="similarity-score">BM25: ${pub.similarity.toFixed(2)}</span>
                    <span>•</span>
                    <span>${pub.year || 'N/A'}</span>
                    <span>•</span>
                    <span>${pub.citations || 0} citations</span>
                </div>
            </div>
        `).join('');

        card.innerHTML = `
            <div class="professor-header">
                <div class="professor-header-content">
                    <img src="${prof.image_url || 'https://via.placeholder.com/120'}" alt="${prof.name}" class="professor-image" onerror="this.src='https://via.placeholder.com/120'">
                    <div class="professor-info">
                        <h3>${prof.name}</h3>
                        <div class="professor-meta">${prof.department} • ${prof.college}</div>
                        <div class="professor-details">
                            <div class="detail-row">
                                <span class="detail-label">Research Interests:</span>
                                <span class="detail-value">${truncateText(prof.interests, 100)}</span>
                            </div>
                            <div class="detail-row">
                                <span class="detail-label">Matching Papers:</span>
                                <span class="detail-value">${prof.num_matching_papers}</span>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="gauge-container">
                    <div style="text-align: center;">
                        <div style="font-size: 1.5rem; font-weight: bold; color: #500000;">${prof.final_score.toFixed(1)}</div>
                        <div style="font-size: 0.8rem; color: #666;">Avg BM25</div>
                    </div>
                </div>
            </div>

            <div class="publications-section">
                <h4>Top Matching Publications (Lexical Match)</h4>
                ${pubsHtml}
            </div>
        `;
        resultsContainer.appendChild(card);
    });

    showResults();
}

function truncateText(text, maxLength) {
    if (!text || text.length <= maxLength) return text;
    return text.substring(0, maxLength) + '...';
}

function showLoading() { loadingSpinner.classList.remove('hidden'); }
function hideLoading() { loadingSpinner.classList.add('hidden'); }
function showError(msg) { errorMessage.textContent = msg; errorMessage.classList.remove('hidden'); }
function hideError() { errorMessage.classList.add('hidden'); }
function showResults() { resultsSection.classList.remove('hidden'); }
function hideResults() { resultsSection.classList.add('hidden'); }
