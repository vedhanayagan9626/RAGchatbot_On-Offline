const chatBox = document.getElementById('chat-box');
const userInput = document.getElementById('user-input');
const pdfUpload = document.getElementById('pdf-upload');
const toast = document.getElementById('toast');
const modelSelect = document.getElementById('model-select');

// Show toast notification
function showToast(message) {
    toast.textContent = message;
    toast.classList.add('show');
    setTimeout(() => {
        toast.classList.remove('show');
    }, 3000);
}

// Append message to UI
function appendMessage(sender, text) {
    const isUser = sender === 'user';
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${isUser ? 'user-message' : 'ai-message'}`;
    
    if (!isUser) {
        const label = document.createElement('div');
        label.className = 'message-label';
        label.textContent = 'AI assistant';
        msgDiv.appendChild(label);
    }
    
    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    bubble.textContent = text;
    msgDiv.appendChild(bubble);
    
    chatBox.appendChild(msgDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
}

// Append loading indicator
function showLoading() {
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'loading-dots';
    loadingDiv.id = 'loading-indicator';
    loadingDiv.innerHTML = '<div class="dot"></div><div class="dot"></div><div class="dot"></div>';
    chatBox.appendChild(loadingDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
}

// Remove loading indicator
function removeLoading() {
    const loadingDiv = document.getElementById('loading-indicator');
    if (loadingDiv) loadingDiv.remove();
}

// Handle PDF Upload
pdfUpload.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    showToast('Uploading and parsing PDF...');
    
    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/upload-pdf', {
            method: 'POST',
            body: formData
        });
        
        if (response.ok) {
            showToast('Knowledge base successfully updated!');
            appendMessage('ai', `I have successfully read '${file.name}'. You can now ask me questions about it!`);
        } else {
            const data = await response.json();
            showToast('Upload failed: ' + data.detail);
        }
    } catch (err) {
        showToast('Error uploading file.');
    }
    
    pdfUpload.value = ''; // reset
});

// Handle User Message Submission
userInput.addEventListener('keypress', async (e) => {
    if (e.key === 'Enter' && userInput.value.trim() !== '') {
        const query = userInput.value.trim();
        userInput.value = '';
        
        const selectedModel = modelSelect.value; // e.g. "online-groq", "offline-ollama"
        let endpoint = '/chat/online';
        let requestBody = { query: query };
        
        if (selectedModel === 'offline-ollama') {
            endpoint = '/chat/offline';
        } else {
            // It's an online model, set the provider
            const provider = selectedModel.split('-')[1]; // "groq" or "gpt"
            requestBody.provider = provider;
        }

        appendMessage('user', query);
        showLoading();
        
        try {
            const response = await fetch(endpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestBody)
            });
            
            removeLoading();
            
            if (response.ok) {
                const data = await response.json();
                appendMessage('ai', data.answer);
            } else {
                const errorData = await response.json();
                appendMessage('ai', 'Error: ' + errorData.detail);
            }
        } catch (err) {
            removeLoading();
            appendMessage('ai', 'Network error occurred while fetching the response.');
        }
    }
});
