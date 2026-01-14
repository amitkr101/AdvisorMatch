
// Assistant Modal Logic

class AssistantModal {
    constructor() {
        this.modal = document.getElementById('profModal');
        this.currentProfId = null;
        this.currentProfName = null;
        this.selectedAngle = null;
        this.chatHistory = [];

        this.setupEventListeners();
    }

    setupEventListeners() {
        // Close Button
        document.getElementById('closeModal').addEventListener('click', () => this.close());

        // Tabs
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', (e) => this.switchTab(e.target));
        });

        // Find Angle Button
        document.getElementById('findAngleBtn').addEventListener('click', () => this.findAngle());

        // Generate Steps Button
        document.getElementById('generateStepsBtn').addEventListener('click', () => this.generateSteps());

        // Resume Upload
        const fileInput = document.getElementById('resumeUpload');
        if (fileInput) {
            fileInput.addEventListener('change', (e) => this.handleFileUpload(e));
        }

        // Chat Button
        document.getElementById('sendChatBtn').addEventListener('click', () => this.sendChatMessage());
        document.getElementById('chatInput').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.sendChatMessage();
        });

        // Export PDF
        document.getElementById('exportChatBtn').addEventListener('click', () => this.exportToPDF());
    }

    async handleFileUpload(event) {
        const file = event.target.files[0];
        if (!file) return;

        if (file.type !== 'application/pdf') {
            alert('Please upload a PDF file.');
            return;
        }

        const statusSpan = document.getElementById('parsingStatus');
        const textArea = document.getElementById('angleResumeInput');

        statusSpan.textContent = "Parsing PDF...";
        statusSpan.style.color = "#666";

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('http://localhost:8000/api/parse-resume', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) throw new Error("Parsing failed");

            const data = await response.json();

            // Append parsed text to existing text (or replace if empty)
            const currentText = textArea.value.trim();
            textArea.value = currentText ? currentText + "\n\n" + data.text : data.text;

            statusSpan.textContent = "Resume parsed successfully!";
            statusSpan.style.color = "green";

        } catch (err) {
            console.error(err);
            statusSpan.textContent = "Failed to parse PDF.";
            statusSpan.style.color = "red";
        }
    }

    open(profId, profName, dept) {
        this.currentProfId = profId;
        this.currentProfName = profName;
        this.selectedAngle = null;

        // Reset UI
        document.getElementById('modalProfName').textContent = profName;
        document.getElementById('modalProfDept').textContent = dept;
        document.getElementById('understandContent').innerHTML = '';
        document.getElementById('angleResults').innerHTML = '';
        document.getElementById('stepsResult').innerHTML = '';
        document.getElementById('nextStepsContent').classList.add('hidden');
        document.getElementById('nextStepsPlaceholder').classList.remove('hidden');

        // Reset Inputs
        document.getElementById('angleInterestInput').value = '';
        document.getElementById('angleResumeInput').value = '';

        // Show Modal
        this.modal.classList.remove('hidden');

        // Reset Chat
        this.chatHistory = [];
        document.getElementById('chatHistory').innerHTML = '';
        document.getElementById('chatInterface').classList.add('hidden');
        document.getElementById('chatProfName').textContent = profName;

        // Load Defaults
        this.switchTab(document.querySelector('[data-tab="tab-understand"]'));
        this.loadUnderstand();
    }

    close() {
        this.modal.classList.add('hidden');
    }

    switchTab(targetBtn) {
        // Remove active class from all
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

        // Add to target
        targetBtn.classList.add('active');
        const tabId = targetBtn.getAttribute('data-tab');
        document.getElementById(tabId).classList.add('active');
    }

    async loadUnderstand() {
        const contentDiv = document.getElementById('understandContent');
        const loading = document.getElementById('understandLoading');

        // Check if already loaded to avoid re-fetch on tab switch
        if (contentDiv.innerHTML.trim() !== '') return;

        loading.classList.remove('hidden');

        try {
            const response = await fetch(`http://localhost:8000/api/assistant/understand/${this.currentProfId}`);
            if (!response.ok) throw new Error("API Limit or Error");
            const data = await response.json();

            // Render Themes
            const themeTags = data.themes.map(t => `<span class="theme-tag">${t}</span>`).join('');

            contentDiv.innerHTML = `
                <div class="analysis-card">
                    <h3>Current Focus</h3>
                    <p>${data.summary}</p>
                    <div class="tags-container">${themeTags}</div>
                </div>
                <div class="analysis-card">
                    <h3>Evolution of Research</h3>
                    <p>${data.trajectory}</p>
                </div>
            `;

            // Reveal Chat after analysis is done
            document.getElementById('chatInterface').classList.remove('hidden');

        } catch (err) {
            contentDiv.innerHTML = `<p class="error-text">Failed to load analysis. ${err.message}</p>`;
        } finally {
            loading.classList.add('hidden');
        }
    }

    async sendChatMessage() {
        const input = document.getElementById('chatInput');
        const historyDiv = document.getElementById('chatHistory');
        const text = input.value.trim();

        if (!text) return;

        // 1. Add User Message
        this.addMessageToUI('user', text);
        this.chatHistory.push({ role: 'user', content: text });
        input.value = '';

        // 2. Add Loading State
        const loaderId = 'loader-' + Date.now();
        const loaderHtml = `
            <div id="${loaderId}" class="chat-message assistant">
                <div class="message-loader">
                    <div class="dot"></div>
                    <div class="dot"></div>
                    <div class="dot"></div>
                </div>
            </div>
        `;
        historyDiv.insertAdjacentHTML('beforeend', loaderHtml);
        historyDiv.scrollTop = historyDiv.scrollHeight;

        try {
            // 3. Call API
            const response = await fetch('http://localhost:8000/api/assistant/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    professor_id: this.currentProfId,
                    history: this.chatHistory
                })
            });

            if (!response.ok) throw new Error("Chat failed");

            const data = await response.json();

            // 4. Remove Loader and Add Assistant Message
            document.getElementById(loaderId).remove();
            this.addMessageToUI('assistant', data.answer);
            this.chatHistory.push({ role: 'assistant', content: data.answer });

        } catch (err) {
            document.getElementById(loaderId).remove();
            this.addMessageToUI('assistant', "I'm sorry, I encountered an error. Please try again.");
        }
    }

    addMessageToUI(role, content) {
        const historyDiv = document.getElementById('chatHistory');
        const msgDiv = document.createElement('div');
        msgDiv.className = `chat-message ${role}`;

        // 1. Convert markdown bold **text** to <strong>text</strong>
        let formatted = content.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

        // 2. Convert line breaks
        formatted = formatted.replace(/\n/g, '<br>');

        msgDiv.innerHTML = formatted;

        historyDiv.appendChild(msgDiv);
        historyDiv.scrollTop = historyDiv.scrollHeight;
    }

    async exportToPDF() {
        if (this.chatHistory.length === 0) {
            alert("No chat history to export.");
            return;
        }

        const { jsPDF } = window.jspdf;
        const doc = new jsPDF();
        let y = 20;

        // Title
        doc.setFontSize(18);
        doc.text(`Research Consultation: Prof. ${this.currentProfName}`, 15, y);
        y += 10;

        doc.setFontSize(10);
        doc.setTextColor(100);
        doc.text(`Generated by AdvisorMatch AI Assist`, 15, y);
        y += 15;

        doc.setFontSize(12);
        doc.setTextColor(0);

        this.chatHistory.forEach(msg => {
            const role = msg.role === 'user' ? 'Student' : 'Assistant';
            const prefix = `${role}: `;

            // Handle long text wrapping
            const text = msg.content.replace(/\*\*/g, ''); // Remove markdown for PDF text
            const lines = doc.splitTextToSize(text, 170);

            // Check for page overflow
            if (y + (lines.length * 7) > 280) {
                doc.addPage();
                y = 20;
            }

            doc.setFont("helvetica", "bold");
            doc.text(prefix, 15, y);

            doc.setFont("helvetica", "normal");
            doc.text(lines, 15 + doc.getTextWidth(prefix), y);

            y += (lines.length * 7) + 5;
        });

        doc.save(`${this.currentProfName.replace(/\s+/g, '_')}_Research_Chat.pdf`);
    }


    async findAngle() {
        const interest = document.getElementById('angleInterestInput').value;
        const resume = document.getElementById('angleResumeInput').value;
        const resultDiv = document.getElementById('angleResults');
        const loading = document.getElementById('angleLoading');

        if (interest.length < 5) {
            alert("Please enter a valid interest.");
            return;
        }

        loading.classList.remove('hidden');
        resultDiv.innerHTML = '';

        try {
            const response = await fetch('http://localhost:8000/api/assistant/find-angle', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    professor_id: this.currentProfId,
                    student_interest: interest,
                    resume_text: resume
                })
            });

            const data = await response.json();

            resultDiv.innerHTML = data.angles.map((angle, index) => `
                <div class="angle-card" onclick="assistantModal.selectAngle('${angle.title.replace(/'/g, "\\'")}')">
                    <h4>${index + 1}. ${angle.title}</h4>
                    <p><strong>Why:</strong> ${angle.logic}</p>
                    <p style="font-size: 0.9em; color: #666; margin-top: 10px;">
                        <strong>Prep:</strong> ${angle.background_needed}
                    </p>
                    <div style="margin-top: 10px; font-size: 0.8em; color: #2563eb; font-weight: bold;">
                        Click to Select &rarr;
                    </div>
                </div>
            `).join('');
        } catch (err) {
            resultDiv.innerHTML = `<p class="error-text">Failed. ${err.message}</p>`;
        } finally {
            loading.classList.add('hidden');
        }
    }

    selectAngle(title) {
        this.selectedAngle = title;

        // Unlock Tab 3
        document.getElementById('nextStepsPlaceholder').classList.add('hidden');
        document.getElementById('nextStepsContent').classList.remove('hidden');

        // Switch to Tab 3
        this.switchTab(document.querySelector('[data-tab="tab-next"]'));

        // Auto trigger generation? Or let user click. User click is better.
    }

    async generateSteps() {
        const level = document.getElementById('studentLevelSelect').value;
        const resultDiv = document.getElementById('stepsResult');
        const loading = document.getElementById('stepsLoading');

        loading.classList.remove('hidden');

        try {
            const response = await fetch('http://localhost:8000/api/assistant/next-steps', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    professor_id: this.currentProfId,
                    selected_angle: this.selectedAngle,
                    student_level: level
                })
            });

            const data = await response.json();

            const checklistHtml = data.checklist.map(item => `
                <div class="checklist-item">
                    <input type="checkbox" class="checklist-checkbox">
                    <span>${item}</span>
                </div>
            `).join('');

            resultDiv.innerHTML = `
                <div style="margin-top: 20px;">
                    <h3>Checklist for ${this.selectedAngle}</h3>
                    <div style="border: 1px solid #eee; border-radius: 8px; overflow: hidden;">
                        ${checklistHtml}
                    </div>
                    
                    <div class="analysis-card" style="margin-top: 20px; background: #fff8f1; border-color: #ffecd9;">
                        <h3 style="color: #9a3412; border-color: #ffecd9;">Outreach Tips</h3>
                        <p style="color: #7c2d12;">${data.outreach_tips}</p>
                    </div>
                </div>
            `;
        } catch (err) {
            resultDiv.innerHTML = `<p class="error-text">Failed. ${err.message}</p>`;
        } finally {
            loading.classList.add('hidden');
        }
    }
}

// Global Instance
const assistantModal = new AssistantModal();
