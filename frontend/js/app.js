// AdvisorMatch Frontend Application

const API_BASE_URL = 'http://localhost:8000';

// DOM Elements
const searchInput = document.getElementById('searchInput');
const searchButton = document.getElementById('searchButton');
const topKInput = document.getElementById('topK');
const loadingSpinner = document.getElementById('loadingSpinner');
const errorMessage = document.getElementById('errorMessage');
const resultsSection = document.getElementById('resultsSection');
const resultsInfo = document.getElementById('resultsInfo');
const resultsContainer = document.getElementById('resultsContainer');

// Event Listeners
searchButton.addEventListener('click', performSearch);
searchInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') performSearch();
});

// Perform Search
async function performSearch() {
    const query = searchInput.value.trim();

    if (!query) {
        showError('Please enter a research query');
        return;
    }

    // Show loading, hide results and errors
    showLoading();
    hideError();
    hideResults();

    try {
        const topK = parseInt(topKInput.value) || 10;

        const response = await fetch(`${API_BASE_URL}/api/search`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query: query,
                top_k: topK,
                include_publications: true
            })
        });

        if (!response.ok) {
            throw new Error(`API error: ${response.status}`);
        }

        const data = await response.json();
        displayResults(data);

    } catch (error) {
        console.error('Search error:', error);
        showError(`Search failed: ${error.message}. Make sure the API server is running at ${API_BASE_URL}`);
    } finally {
        hideLoading();
    }
}

// Display Results
function displayResults(data) {
    if (!data.results || data.results.length === 0) {
        showError('No advisors found for your query. Try different keywords.');
        return;
    }

    // Update results info
    resultsInfo.textContent = `Found ${data.total_results} advisor${data.total_results !== 1 ? 's' : ''} for "${data.query}" (${data.search_time_ms.toFixed(0)}ms)`;

    // Clear previous results
    resultsContainer.innerHTML = '';

    // Create professor cards
    data.results.forEach((professor, index) => {
        const card = createProfessorCard(professor, index + 1);
        resultsContainer.appendChild(card);
    });

    // Show results section
    showResults();
}

// Create Professor Card
function createProfessorCard(professor, rank) {
    const card = document.createElement('div');
    card.className = 'professor-card';

    // Score percentage for display
    const scorePercent = (professor.final_score * 100).toFixed(1);

    // Determine gauge color
    let gaugeColor = '#dc3545'; // Red
    if (professor.final_score > 0.5) {
        gaugeColor = '#28a745'; // Green
    } else if (professor.final_score >= 0.3) {
        gaugeColor = '#ffc107'; // Amber
    }

    // SVG Gauge Calculation
    // Semi-circle arc length = PI * R
    // R = 50 (viewBox 0 0 120 60, center 60,60)
    // Circumference = 157
    const radius = 50;
    const circumference = Math.PI * radius;
    const dashArray = circumference;
    const dashOffset = circumference * (1 - professor.final_score);

    card.innerHTML = `
        <div class="professor-header">
            <div class="professor-header-content">
                ${professor.image_url ? `
                    <img src="${professor.image_url}" alt="${professor.name}" class="professor-image">
                ` : ''}
                <div class="professor-info">
                    <h3 class="professor-name-link" onclick="assistantModal.open(${professor.professor_id}, '${professor.name.replace(/'/g, "\\'")}', '${professor.department}')">
                        ${rank}. ${professor.name} <span class="ai-badge">AI Assistant</span>
                    </h3>
                    <div class="professor-meta">
                        ${professor.department} • ${professor.college}
                    </div>
                </div>
            </div>
            
            <div class="gauge-container">
                <svg viewBox="0 0 120 60" class="gauge-svg">
                    <!-- Background Arc -->
                    <path d="M 10 60 A 50 50 0 0 1 110 60" fill="none" stroke="#eee" stroke-width="15" />
                    <!-- Fill Arc -->
                    <path d="M 10 60 A 50 50 0 0 1 110 60" fill="none" stroke="${gaugeColor}" stroke-width="15" 
                          stroke-dasharray="${dashArray}" stroke-dashoffset="${dashOffset}" />
                </svg>
                <div class="gauge-text">${scorePercent}%</div>
                <div class="gauge-label">Match Score</div>
            </div>
        </div>
        
        <div class="professor-details">
            ${professor.interests ? `
                <div class="detail-row">
                    <span class="detail-label">Research Interests:</span>
                    <span class="detail-value">${truncateText(professor.interests, 200)}</span>
                </div>
            ` : ''}
            
            <div class="detail-row">
                <span class="detail-label">Matching Papers:</span>
                <span class="detail-value">${professor.num_matching_papers}</span>
            </div>
            
            ${professor.url ? `
                <div class="detail-row">
                    <span class="detail-label">Profile:</span>
                    <span class="detail-value">
                        <a href="${professor.url}" target="_blank">View Faculty Page →</a>
                    </span>
                </div>
            ` : ''}
        </div>
        
        <div class="score-breakdown">
            <div class="score-item">
                <div class="score-item-label">Paper Similarity</div>
                <div class="score-item-value">${(professor.avg_similarity * 100).toFixed(1)}%</div>
            </div>
            <div class="score-item">
                <div class="score-item-label">Recency</div>
                <div class="score-item-value">${(professor.recency_weight * 100).toFixed(1)}%</div>
            </div>
            <div class="score-item">
                <div class="score-item-label">Activity</div>
                <div class="score-item-value">+${(professor.activity_bonus * 100).toFixed(1)}%</div>
            </div>
            <div class="score-item">
                <div class="score-item-label">Citations</div>
                <div class="score-item-value">${(professor.citation_impact * 100).toFixed(1)}%</div>
            </div>
        </div>
        
        ${professor.top_publications && professor.top_publications.length > 0 ? `
            <div class="publications-section">
                <h4>Top Matching Publications</h4>
                ${professor.top_publications.map(pub => createPublicationHTML(pub)).join('')}
            </div>
        ` : ''}
    `;

    return card;
}

// Create Publication HTML
function createPublicationHTML(publication) {
    // Determine border color class
    let borderClass = 'pub-border-red';
    if (publication.similarity > 0.5) {
        borderClass = 'pub-border-green';
    } else if (publication.similarity >= 0.3) {
        borderClass = 'pub-border-amber';
    }

    return `
        <div class="publication-item ${borderClass}">
            <div class="publication-title">
                ${publication.url ?
            `<a href="${publication.url}" target="_blank" class="publication-link">${publication.title}</a>` :
            publication.title}
            </div>
            <div class="publication-meta">
                <span>${publication.year || 'N/A'}</span>
                <span>•</span>
                <span>${publication.citations || 0} citations</span>
                ${publication.venue ? `
                    <span>•</span>
                    <span>${publication.venue}</span>
                ` : ''}
                <span>•</span>
                <span class="similarity-score">Paper Similarity: ${(publication.similarity * 100).toFixed(1)}%</span>
            </div>
        </div>
    `;
}

// Utility Functions
function showLoading() {
    loadingSpinner.classList.remove('hidden');
}

function hideLoading() {
    loadingSpinner.classList.add('hidden');
}

function showError(message) {
    errorMessage.textContent = message;
    errorMessage.classList.remove('hidden');
}

function hideError() {
    errorMessage.classList.add('hidden');
}

function showResults() {
    resultsSection.classList.remove('hidden');
}

function hideResults() {
    resultsSection.classList.add('hidden');
}

function truncateText(text, maxLength) {
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength) + '...';
}

// Check API Health on Load
async function checkAPIHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        if (response.ok) {
            console.log('✓ API server is running');
        }
    } catch (error) {
        console.warn('⚠ API server not reachable. Make sure to start it with: cd app && python3 api.py');
    }
}

// Initialize
checkAPIHealth();
