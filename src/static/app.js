// State
let sessionId = "web-" + Math.random().toString(36).substring(2, 10);
let isGenerating = false;

// DOM Elements
const chatHistory = document.getElementById("chat-history");
const chatInput = document.getElementById("chat-input");
const sendBtn = document.getElementById("send-btn");
const newChatBtn = document.getElementById("btn-new-chat");
const apiKeyInput = document.getElementById("api-key-input");
const uploadZone = document.getElementById("upload-zone");
const fileInput = document.getElementById("file-input");
const uploadProgress = document.getElementById("upload-progress");
const documentList = document.getElementById("document-list");

// Markdown setup
marked.setOptions({ breaks: true });

// --- Chat Logic ---

function getAuthHeader() {
    return { "Authorization": `Bearer ${apiKeyInput.value.trim()}` };
}

function scrollToBottom() {
    chatHistory.scrollTo({ top: chatHistory.scrollHeight, behavior: 'smooth' });
}

function appendUserMessage(text) {
    const tpl = document.getElementById("tpl-user-msg").content.cloneNode(true);
    tpl.querySelector(".content").textContent = text;
    chatHistory.appendChild(tpl);
    scrollToBottom();
}

function createAssistantMessage() {
    const tpl = document.getElementById("tpl-assistant-msg").content.cloneNode(true);
    const msgEl = tpl.querySelector(".message");
    chatHistory.appendChild(tpl);
    scrollToBottom();
    return msgEl;
}

async function handleSend() {
    if (isGenerating) return;
    const text = chatInput.value.trim();
    if (!text) return;

    // Reset input
    chatInput.value = "";
    chatInput.style.height = "auto";
    
    appendUserMessage(text);
    const msgEl = createAssistantMessage();
    
    const contentEl = msgEl.querySelector(".content");
    const statusEl = msgEl.querySelector(".status-indicator");
    const metaEl = msgEl.querySelector(".meta-footer");
    const badgeEl = msgEl.querySelector(".verification-badge");
    const latencyEl = msgEl.querySelector(".latency-metrics");
    const sourcesContainer = msgEl.querySelector(".sources-container");
    const sourcesList = msgEl.querySelector(".sources-list");

    isGenerating = true;
    sendBtn.disabled = true;

    // Cycle waiting messages to keep UI feeling alive during slow local inference
    let loadingTimer = setInterval(() => {
        if (statusEl.style.display === "none") {
            clearInterval(loadingTimer);
            return;
        }
        const msgs = ["Reading documents...", "Analyzing context...", "Synthesizing answer...", "Cross-referencing...", "Almost there..."];
        if (statusEl.textContent.includes("passages") || statusEl.textContent.includes("...")) {
             statusEl.innerHTML = msgs[Math.floor(Math.random() * msgs.length)];
        }
    }, 3500);

    try {
        const response = await fetch("/query/stream", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                ...getAuthHeader()
            },
            body: JSON.stringify({ query: text, session_id: sessionId })
        });

        if (!response.ok) throw new Error("API Error: " + response.statusText);

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let rawText = "";

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            
            const chunk = decoder.decode(value);
            const events = chunk.split("\n\n");
            
            for (const ev of events) {
                if (!ev.trim()) continue;
                
                const lines = ev.split("\n");
                const eventType = lines[0].replace("event: ", "");
                const dataRaw = lines[1].replace("data: ", "");
                
                if (eventType === "error") {
                    contentEl.innerHTML = `<span style="color:var(--danger)">Error: ${JSON.parse(dataRaw).detail}</span>`;
                    break;
                }

                const data = JSON.parse(dataRaw);

                if (eventType === "progress") {
                    if (data.stage === "planning") statusEl.textContent = "Planning search queries...";
                    if (data.stage === "retrieving") statusEl.textContent = `Found ${data.chunks_found} relevant passages...`;
                }
                else if (eventType === "token") {
                    clearInterval(loadingTimer);
                    statusEl.style.display = "none";
                    rawText += data.token;
                    contentEl.innerHTML = marked.parse(rawText);
                    scrollToBottom();
                }
                else if (eventType === "done") {
                    clearInterval(loadingTimer);
                    // Verification
                    metaEl.style.display = "flex";
                    if (data.verified) {
                        badgeEl.className = "verification-badge pass";
                        badgeEl.innerHTML = `✅ Verified <span class="verification-reason">- ${data.verification_reason}</span>`;
                    } else {
                        badgeEl.className = "verification-badge fail";
                        badgeEl.innerHTML = `⚠️ Failed Verification <span class="verification-reason">- ${data.verification_reason}</span>`;
                    }
                    
                    // Sources
                    if (data.sources && data.sources.length > 0) {
                        sourcesContainer.style.display = "block";
                        data.sources.forEach(s => {
                            const sc = document.createElement("div");
                            sc.className = "source-card";
                            const page = s.page ? `(Page ${s.page})` : '';
                            sc.innerHTML = `
                                <div class="source-header">
                                    <span>[Doc ${s.doc_id}] ${s.source} ${page}</span>
                                    <span class="source-score">Relevance: ${s.score}</span>
                                </div>
                            `;
                            sourcesList.appendChild(sc);
                        });
                    }
                    
                    // Latency
                    if (data.stage_latencies_ms) {
                        latencyEl.innerHTML = `⏱️ ${data.stage_latencies_ms.total}ms total`;
                    }
                    
                    scrollToBottom();
                }
            }
        }
    } catch (e) {
        clearInterval(loadingTimer);
        msgEl.querySelector(".content").innerHTML = `<span style="color:var(--danger)">Connection failed. Check API key and server.</span>`;
    } finally {
        clearInterval(loadingTimer);
        isGenerating = false;
        sendBtn.disabled = false;
        chatInput.focus();
    }
}

// Events
sendBtn.addEventListener("click", handleSend);
chatInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleSend();
    }
});
chatInput.addEventListener("input", function() {
    this.style.height = "auto";
    this.style.height = (this.scrollHeight < 200 ? this.scrollHeight : 200) + "px";
});

newChatBtn.addEventListener("click", () => {
    sessionId = "web-" + Math.random().toString(36).substring(2, 10);
    chatHistory.innerHTML = `
        <div class="message assistant-message">
            <div class="avatar">🤖</div>
            <div class="content">Memory cleared. How can I help you next?</div>
        </div>
    `;
    fetch(`/sessions/${sessionId}`, { method: "DELETE", headers: getAuthHeader() }).catch(()=>{});
});

// --- Document Management ---

async function refreshDocuments() {
    try {
        const res = await fetch("/documents", { headers: getAuthHeader() });
        const docs = await res.json();
        documentList.innerHTML = "";
        
        docs.forEach(d => {
            const el = document.createElement("div");
            el.className = "doc-item";
            el.innerHTML = `
                <div class="doc-info">
                    <div class="doc-name">${d.filename}</div>
                    <div class="doc-meta">${d.num_chunks} chunks embedded</div>
                </div>
                <button class="btn-delete" onclick="deleteDocument('${d.doc_id}')" title="Delete">🗑️</button>
            `;
            documentList.appendChild(el);
        });
    } catch (e) {
        console.error("Failed to load documents", e);
    }
}

async function deleteDocument(docId) {
    if(!confirm("Delete this document from the knowledge base?")) return;
    await fetch(`/documents/${docId}`, { method: "DELETE", headers: getAuthHeader() });
    refreshDocuments();
}

async function uploadFile(file) {
    const formData = new FormData();
    formData.append("file", file);
    
    uploadProgress.style.display = "block";
    
    try {
        const res = await fetch("/ingest", {
            method: "POST",
            headers: getAuthHeader(), // FormData boundary is set automatically
            body: formData
        });
        if (!res.ok) {
            const err = await res.json();
            alert("Upload failed: " + err.detail);
        }
    } catch(e) {
        alert("Upload error.");
    } finally {
        uploadProgress.style.display = "none";
        refreshDocuments();
    }
}

// Drag & Drop
uploadZone.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", (e) => {
    if(e.target.files.length > 0) uploadFile(e.target.files[0]);
});

uploadZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadZone.style.borderColor = "var(--accent)";
});
uploadZone.addEventListener("dragleave", (e) => {
    e.preventDefault();
    uploadZone.style.borderColor = "var(--border)";
});
uploadZone.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadZone.style.borderColor = "var(--border)";
    if(e.dataTransfer.files.length > 0) uploadFile(e.dataTransfer.files[0]);
});

// Init
refreshDocuments();
