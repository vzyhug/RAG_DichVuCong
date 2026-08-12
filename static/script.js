// DOM Elements
const chatForm = document.getElementById('chat-form');
const messageInput = document.getElementById('message-input');
const messagesWrapper = document.getElementById('messages-wrapper');
const welcomeScreen = document.getElementById('welcome-screen');
const typingIndicator = document.getElementById('typing-indicator');
const chatContainer = document.getElementById('chat-container');
const themeToggle = document.getElementById('theme-toggle');
const sendBtn = document.getElementById('send-btn');
const modalOverlay = document.getElementById('context-modal');
const closeModalBtn = document.getElementById('close-modal');
const modalBody = document.getElementById('modal-body');

// State
let isWaitingForResponse = false;
let globalContexts = [];

// Initialize marked
marked.setOptions({
    breaks: true,
    gfm: true
});

// Auto-resize textarea
messageInput.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = (this.scrollHeight < 150 ? this.scrollHeight : 150) + 'px';
    
    if(this.value.trim().length > 0) {
        sendBtn.disabled = false;
    } else {
        sendBtn.disabled = true;
    }
});

messageInput.addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        if(!isWaitingForResponse && this.value.trim().length > 0) {
            chatForm.dispatchEvent(new Event('submit'));
        }
    }
});

// Theme Toggle
function initTheme() {
    const savedTheme = localStorage.getItem('theme') || 'light';
    document.documentElement.setAttribute('data-theme', savedTheme);
    updateThemeIcon(savedTheme);
}

function toggleTheme() {
    const currentTheme = document.documentElement.getAttribute('data-theme');
    const newTheme = currentTheme === 'light' ? 'dark' : 'light';
    
    document.documentElement.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);
    updateThemeIcon(newTheme);
}

function updateThemeIcon(theme) {
    const icon = themeToggle.querySelector('i');
    if (theme === 'dark') {
        icon.className = 'fa-solid fa-sun';
        themeToggle.innerHTML = '<i class="fa-solid fa-sun"></i> Chế độ sáng';
    } else {
        icon.className = 'fa-solid fa-moon';
        themeToggle.innerHTML = '<i class="fa-solid fa-moon"></i> Chế độ tối';
    }
}

themeToggle.addEventListener('click', toggleTheme);

// Helper to set input from suggestions
function setInput(text) {
    messageInput.value = text;
    messageInput.dispatchEvent(new Event('input'));
    messageInput.focus();
}

// Add user message to UI
function addUserMessage(text) {
    if(welcomeScreen && !welcomeScreen.classList.contains('hidden')) {
        welcomeScreen.classList.add('hidden');
    }

    const messageDiv = document.createElement('div');
    messageDiv.className = 'message user';
    messageDiv.innerHTML = `
        <div class="avatar"><i class="fa-solid fa-user"></i></div>
        <div class="message-content">
            ${text.replace(/\n/g, '<br>')}
        </div>
    `;
    messagesWrapper.appendChild(messageDiv);
    scrollToBottom();
}

function scrollToBottom() {
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

// Modal handling
function showContext(index) {
    const ctx = globalContexts[index];
    if(!ctx) return;
    
    const title = ctx.metadata?.filename || 'Tài liệu trích dẫn';
    const text = ctx.text || 'Không có nội dung.';
    
    modalBody.innerHTML = `
        <h4 style="margin-bottom: 10px; color: var(--accent-color);">${title}</h4>
        <div style="background: var(--bg-tertiary); padding: 15px; border-radius: 8px; font-family: monospace; white-space: pre-wrap; font-size: 0.85rem;">
            ${text}
        </div>
    `;
    
    modalOverlay.classList.add('active');
}

closeModalBtn.addEventListener('click', () => {
    modalOverlay.classList.remove('active');
});

modalOverlay.addEventListener('click', (e) => {
    if(e.target === modalOverlay) {
        modalOverlay.classList.remove('active');
    }
});

// Form submission with Streaming (SSE)
chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const query = messageInput.value.trim();
    if (!query || isWaitingForResponse) return;
    
    addUserMessage(query);
    
    messageInput.value = '';
    messageInput.style.height = 'auto';
    sendBtn.disabled = true;
    
    isWaitingForResponse = true;
    typingIndicator.classList.add('hidden'); // We stream directly
    scrollToBottom();
    
    // Create bot message container for streaming
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message bot';
    messageDiv.innerHTML = `
        <div class="avatar"><i class="fa-solid fa-robot"></i></div>
        <div class="message-content">
            <div class="text-content">Đang suy nghĩ...</div>
            <div class="source-references"></div>
        </div>
    `;
    messagesWrapper.appendChild(messageDiv);
    scrollToBottom();

    const textContentDiv = messageDiv.querySelector('.text-content');
    const sourceReferencesDiv = messageDiv.querySelector('.source-references');
    let fullText = '';

    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ query: query })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n\n');
            buffer = lines.pop();

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const dataStr = line.substring(6).trim();
                    if (dataStr === '[DONE]') continue;
                    try {
                        const parsed = JSON.parse(dataStr);
                        if (parsed.contexts || parsed.emergency !== undefined) {
                            if (parsed.emergency) {
                                messageDiv.classList.add('emergency');
                            }
                            if (parsed.contexts && parsed.contexts.length > 0) {
                                globalContexts = parsed.contexts;
                                let contextHtml = '';
                                const uniqueSources = new Map();
                                parsed.contexts.forEach((ctx, idx) => {
                                    const name = ctx.metadata?.filename || `Nguồn ${idx+1}`;
                                    if(!uniqueSources.has(name)) {
                                        uniqueSources.set(name, idx);
                                    }
                                });
                                uniqueSources.forEach((idx, name) => {
                                    contextHtml += `<span class="source-badge" onclick="showContext(${idx})"><i class="fa-solid fa-file-lines"></i> ${name}</span>`;
                                });
                                sourceReferencesDiv.innerHTML = contextHtml;
                            }
                        }
                        if (parsed.delta) {
                            if (fullText === '' && textContentDiv.textContent === 'Đang suy nghĩ...') {
                                fullText = '';
                            }
                            fullText += parsed.delta;
                            textContentDiv.innerHTML = marked.parse(fullText);
                            scrollToBottom();
                        }
                    } catch (err) {
                        console.error('Parse error:', err, dataStr);
                    }
                }
            }
        }

        if (!fullText.trim()) {
            fullText = "Xin lỗi, hiện tại tôi chưa tìm thấy thông tin chi tiết hoặc câu trả lời chưa sẵn sàng. Anh/chị vui lòng thử lại hoặc liên hệ trực tiếp cơ quan công an.";
            textContentDiv.innerHTML = marked.parse(fullText);
        }
        
    } catch (error) {
        console.error('Error:', error);
        let errorDetails = error.message ? error.message : String(error);
        textContentDiv.innerHTML = marked.parse(`Lỗi hệ thống: ${errorDetails}. Vui lòng thử lại sau.`);
    } finally {
        isWaitingForResponse = false;
    }
});

// Initialize new chat button
document.getElementById('new-chat').addEventListener('click', () => {
    messagesWrapper.innerHTML = '';
    welcomeScreen.classList.remove('hidden');
    messageInput.value = '';
    messageInput.focus();
});

// Init
initTheme();
sendBtn.disabled = true;
