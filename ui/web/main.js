async function callChatStream(query, onToken, onComplete, onError, onRetrieval) {
  const r = await fetch("/api/v1/chat/stream", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      query,
      include_sources: true,
    }),
  });

  if (!r.ok) {
    const data = await r.json().catch(() => ({}));
    const msg = (data && (data.detail || data.error)) || `HTTP ${r.status}`;
    onError(new Error(msg));
    return;
  }

  const reader = r.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        if (line.startsWith("data: ")) {
          const dataStr = line.slice(6).trim();
          if (dataStr === "[DONE]") {
            onComplete();
            return;
          }
          try {
            const data = JSON.parse(dataStr);
            if (data.type === "token" && data.content) {
              onToken(data.content);
            } else if (data.type === "retrieval") {
              if (typeof onRetrieval === "function") onRetrieval(data);
            } else if (data.type === "done") {
              onComplete(data);
            } else if (data.type === "error") {
              onError(new Error(data.error));
              return;
            }
          } catch (e) {
            console.error("Failed to parse SSE data:", e, dataStr);
          }
        }
      }
    }
  } catch (e) {
    onError(e);
  }
}

async function uploadDoc({ file, doc_id, title, uri, source, lang, tags }) {
  const fd = new FormData();
  fd.append("file", file);
  fd.append("doc_id", doc_id);
  if (title) fd.append("title", title);
  if (uri) fd.append("uri", uri);
  if (source) fd.append("source", source);
  if (lang) fd.append("lang", lang);
  if (tags) fd.append("tags", tags);
  fd.append("refresh", "false");

  const r = await fetch("/api/v1/documents/upload", { method: "POST", body: fd });
  const data = await r.json();
  if (!r.ok) {
    const msg = (data && (data.detail || data.error)) || `HTTP ${r.status}`;
    throw new Error(msg);
  }
  return data;
}

async function listDocuments() {
  const limit = docsState.limit;
  const offset = docsState.offset;
  const r = await fetch(`/api/v1/documents?limit=${encodeURIComponent(limit)}&offset=${encodeURIComponent(offset)}`);
  const data = await r.json();
  if (!r.ok) {
    throw new Error(data.error || `HTTP ${r.status}`);
  }
  return data;
}

async function deleteDoc(doc_id) {
  const r = await fetch(`/api/v1/documents/${encodeURIComponent(doc_id)}`, { method: "DELETE" });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) {
    const msg = (data && (data.detail || data.error)) || `HTTP ${r.status}`;
    throw new Error(msg);
  }
  return data;
}

async function deleteAllDocs() {
  const r = await fetch(`/api/v1/documents?confirm=true`, { method: "DELETE" });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) {
    const msg = (data && (data.detail || data.error)) || `HTTP ${r.status}`;
    throw new Error(msg);
  }
  return data;
}

const docsState = {
  items: [],
  total: 0,
  limit: 100,
  offset: 0,
};

function updateDocsMeta() {
  const el = document.getElementById("docs_meta");
  if (!el) return;
  const shown = docsState.items.length;
  const total = docsState.total || 0;
  if (!total) {
    el.textContent = shown ? `Показано: ${shown}` : "";
  } else {
    el.textContent = `Показано: ${shown} из ${total}`;
  }

  const btn = document.getElementById("load_more_docs");
  if (btn) {
    btn.disabled = !total || shown >= total;
  }
}

// UI helpers
function randId() {
  return `doc-${Date.now()}-${Math.floor(Math.random() * 1e6)}`;
}

function setBusy(busy, text) {
  const askBtn = document.getElementById("ask_stream");
  if (askBtn) askBtn.disabled = busy;
  const spinner = document.getElementById("ask_stream_spinner");
  if (spinner) spinner.style.display = busy ? "inline-block" : "none";

  const statusEl = document.getElementById("status");
  if (text) {
    statusEl.textContent = text;
    statusEl.className = "composer-status" + (text.includes("Ошибка") ? " error" : text.includes("OK") ? " success" : "");
  } else {
    statusEl.textContent = "";
    statusEl.className = "composer-status";
  }
}

function setUploadBusy(busy, text) {
  const uploadBtn = document.getElementById("upload");
  if (uploadBtn) uploadBtn.disabled = busy;
  const statusEl = document.getElementById("upload_status");
  if (text) {
    statusEl.textContent = text;
    statusEl.className = "status" + (text.includes("Ошибка") ? " error" : text.includes("OK") ? " success" : "");
  } else {
    statusEl.textContent = "";
    statusEl.className = "status";
  }
}

function fmtSources(sources) {
  if (!sources || !sources.length) return null;
  return sources.map(s => ({
    ref: s.ref,
    title: s.title || s.doc_id,
    uri: s.uri,
    doc_id: s.doc_id,
  }));
}

function sourcesHtml(sources) {
  if (!sources || !sources.length) return "";
  return sources
    .map(
      s => `
      <div class="source-item">
        <div class="source-title">${escapeHtml(s.ref ? `[${s.ref}] ` : "")}${escapeHtml(s.title || s.doc_id)}</div>
        ${s.uri ? `<div class="source-uri">${escapeHtml(s.uri)}</div>` : ""}
        <div style="font-size: 11px; color: var(--text-secondary); margin-top: 4px;">doc_id: ${escapeHtml(s.doc_id)}</div>
      </div>
    `
    )
    .join("");
}

function retrievalHtml(context) {
  if (!context || !context.length) return "";
  return context
    .map(c => {
      const src = c.source || {};
      const title = (src.title || src.doc_id || c.doc_id || "doc").toString();
      const ref = src.ref ? `[${src.ref}] ` : "";
      const score = typeof c.score === "number" ? c.score.toFixed(4) : "";
      const uri = src.uri ? `<div class="source-uri">${escapeHtml(src.uri)}</div>` : "";
      const text = c.text ? escapeHtml(c.text) : "";
      return `
        <div class="source-item">
          <div class="source-title">${escapeHtml(ref)}${escapeHtml(title)}${score ? ` <span style="color: var(--text-secondary); font-size: 11px;">(score=${escapeHtml(score)})</span>` : ""}</div>
          ${uri}
          <pre style="margin-top: 10px; max-height: 240px;">${text || "(empty)"}</pre>
        </div>
      `;
    })
    .join("");
}

function renderDocuments(docs) {
  const container = document.getElementById("doc_list");
  if (!docs || !docs.length) {
    container.innerHTML = '<div style="color: var(--text-secondary); text-align: center; padding: 40px;">Документы не найдены</div>';
    return;
  }

  container.innerHTML = docs
    .map(
      doc => `
    <div class="doc-item">
      <div class="doc-header">
        <div style="flex: 1;">
          <div class="doc-title">${escapeHtml(doc.title || doc.doc_id || "Без названия")}</div>
          <div class="doc-id">ID: ${escapeHtml(doc.doc_id)}</div>
        </div>
        <div class="badges">
          ${doc.stored !== false ? '<span class="badge stored">Сохранен</span>' : ""}
          ${doc.indexed ? '<span class="badge indexed">Проиндексирован</span>' : '<span class="badge not-indexed">Не проиндексирован</span>'}
        </div>
      </div>
      <div class="doc-meta">
        ${doc.source ? `<span>Источник: ${escapeHtml(doc.source)}</span>` : ""}
        ${doc.lang ? `<span style="margin-left: 12px;">Язык: ${escapeHtml(doc.lang)}</span>` : ""}
        ${doc.size ? `<span style="margin-left: 12px;">Размер: ${formatBytes(doc.size)}</span>` : ""}
        ${doc.stored_at ? `<span style="margin-left: 12px;">Загружен: ${new Date(doc.stored_at).toLocaleString("ru")}</span>` : ""}
      </div>
      ${doc.tags && doc.tags.length ? `<div class="doc-meta">Теги: ${doc.tags.map(t => escapeHtml(t)).join(", ")}</div>` : ""}
      <div class="doc-meta" style="margin-top: 10px;">
        <button class="btn btn-danger" data-action="delete-doc" data-doc-id="${escapeHtml(doc.doc_id)}">Удалить</button>
      </div>
    </div>
  `
    )
    .join("");

  // Wire delete buttons (event delegation)
  container.querySelectorAll('[data-action="delete-doc"]').forEach(btn => {
    btn.addEventListener("click", async e => {
      const docId = e.currentTarget.getAttribute("data-doc-id");
      if (!docId) return;
      const ok = confirm(`Удалить документ ${docId}?\n\nЭто удалит файл из хранилища и удалит чанки из индекса.`);
      if (!ok) return;
      try {
        e.currentTarget.disabled = true;
        await deleteDoc(docId);
        await loadDocuments();
      } catch (err) {
        e.currentTarget.disabled = false;
        alert(`Ошибка удаления: ${err.message}`);
      }
    });
  });
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

function formatBytes(bytes) {
  if (!bytes) return "0 B";
  const k = 1024;
  const sizes = ["B", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return Math.round(bytes / Math.pow(k, i) * 100) / 100 + " " + sizes[i];
}

function getMessagesEl() {
  return document.getElementById("messages");
}

function scrollMessagesToBottom() {
  const el = getMessagesEl();
  if (!el) return;
  el.scrollTop = el.scrollHeight;
}

function addMessage({ role, text }) {
  const messages = getMessagesEl();
  if (!messages) return null;

  const wrapper = document.createElement("div");
  wrapper.className = `msg ${role === "user" ? "user" : "assistant"}`;

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  const bubbleText = document.createElement("div");
  bubbleText.className = "bubble-text";
  bubbleText.textContent = text || "";
  bubble.appendChild(bubbleText);

  wrapper.appendChild(bubble);
  messages.appendChild(wrapper);
  scrollMessagesToBottom();

  return { wrapper, bubble, bubbleText };
}

function setAssistantExtras(msgWrapper, { sources, context }) {
  if (!msgWrapper) return;
  const bubble = msgWrapper.querySelector(".bubble");
  if (!bubble) return;

  // Remove previous extras if re-rendering
  const prev = bubble.querySelector(".bubble-meta");
  if (prev) prev.remove();

  const meta = document.createElement("div");
  meta.className = "bubble-meta";

  if (sources && sources.length) {
    const d = document.createElement("details");
    d.open = false;
    d.innerHTML = `<summary>Источники (${sources.length})</summary>${sourcesHtml(fmtSources(sources) || [])}`;
    meta.appendChild(d);
  }

  if (context && context.length) {
    const d = document.createElement("details");
    d.open = false;
    d.innerHTML = `<summary>Контекст (retrieval) (${context.length})</summary>${retrievalHtml(context)}`;
    meta.appendChild(d);
  }

  if (meta.childNodes.length) {
    bubble.appendChild(meta);
    scrollMessagesToBottom();
  }
}

// Event handlers
const uploadBtn = document.getElementById("upload");
if (uploadBtn) uploadBtn.addEventListener("click", async () => {
  const fileEl = document.getElementById("file");
  const file = fileEl.files && fileEl.files[0];
  if (!file) {
    alert("Выберите файл");
    return;
  }

  const doc_id = document.getElementById("doc_id").value.trim() || randId();
  const title = document.getElementById("title").value.trim();
  const uri = document.getElementById("uri").value.trim();
  const source = document.getElementById("source").value.trim();
  const lang = document.getElementById("lang").value.trim();
  const tags = document.getElementById("tags").value.trim();

  setUploadBusy(true, "Загрузка…");
  try {
    const data = await uploadDoc({ file, doc_id, title, uri, source, lang, tags });
    setUploadBusy(false, "OK");
    // Refresh document list
    loadDocuments();
  } catch (e) {
    setUploadBusy(false, "Ошибка");
    alert(`Ошибка загрузки: ${e.message}`);
  }
});

async function askStream() {
  const qEl = document.getElementById("q");
  const q = qEl ? qEl.value.trim() : "";
  if (!q) return;

  // Add user message + assistant placeholder
  addMessage({ role: "user", text: q });
  const assistant = addMessage({ role: "assistant", text: "" });
  let latestContext = null;

  // Clear composer early (chat-like)
  if (qEl) {
    qEl.value = "";
    qEl.focus();
  }

  setBusy(true, "Запрос…");
  let answerText = "";

  try {
    await callChatStream(
      q,
      token => {
        answerText += token;
        if (assistant && assistant.bubbleText) {
          assistant.bubbleText.textContent = answerText;
          scrollMessagesToBottom();
        }
      },
      data => {
        // done
        if (assistant && assistant.bubbleText) assistant.bubbleText.textContent = answerText;
        if (assistant && assistant.wrapper && data) {
          setAssistantExtras(assistant.wrapper, { sources: data.sources, context: data.context || latestContext });
        }
        const flags = [];
        if (data && data.partial) flags.push("partial");
        if (data && data.degraded && data.degraded.length) flags.push(`degraded=${data.degraded.join(",")}`);
        setBusy(false, flags.length ? flags.join(" · ") : "OK");
      },
      error => {
        if (assistant && assistant.bubbleText) assistant.bubbleText.textContent = `Ошибка: ${error.message}`;
        setBusy(false, "Ошибка");
      },
      data => {
        // retrieval
        if (!assistant || !assistant.wrapper) return;
        if (data && data.context) {
          latestContext = data.context;
          // Show retrieval as soon as it arrives (helps when done event doesn't include context)
          setAssistantExtras(assistant.wrapper, { sources: null, context: latestContext });
        }
      }
    );
  } catch (e) {
    if (assistant && assistant.bubbleText) assistant.bubbleText.textContent = `Ошибка: ${e.message}`;
    setBusy(false, "Ошибка");
  }
}

const askBtn = document.getElementById("ask_stream");
if (askBtn) askBtn.addEventListener("click", askStream);

// Composer keybinds:
// - Enter: send
// - Shift+Enter: newline
// - Ctrl/Meta+Enter: send
const qEl = document.getElementById("q");
if (qEl) qEl.addEventListener("keydown", e => {
  if (e.isComposing) return;
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
    e.preventDefault();
    askStream();
    return;
  }
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    askStream();
  }
});

async function loadDocuments() {
  const container = document.getElementById("doc_list");
  try {
    const data = await listDocuments();
    const docs = data.documents || [];
    docsState.total = data.total || 0;
    if (docsState.offset === 0) {
      docsState.items = docs;
    } else {
      // append next page; backend sorts by stored_at desc, so this is stable for browsing
      docsState.items = docsState.items.concat(docs);
    }
    docsState.offset += docs.length;
    renderDocuments(docsState.items);
    updateDocsMeta();
  } catch (e) {
    container.innerHTML = `<div style="color: var(--error); text-align: center; padding: 40px;">Ошибка загрузки: ${escapeHtml(e.message)}</div>`;
  }
}

const refreshBtn = document.getElementById("refresh_docs");
if (refreshBtn) refreshBtn.addEventListener("click", () => {
  docsState.offset = 0;
  loadDocuments();
});

const deleteAllBtn = document.getElementById("delete_all_docs");
if (deleteAllBtn) deleteAllBtn.addEventListener("click", async () => {
  const ok = confirm(
    "Удалить ВСЕ документы?\n\n" +
      "- удалит файлы из document-storage\n" +
      "- удалит чанки из retrieval индекса\n\n" +
      "Операция необратима."
  );
  if (!ok) return;

  deleteAllBtn.disabled = true;
  try {
    const res = await deleteAllDocs();
    // Reset pagination and reload
    docsState.offset = 0;
    docsState.items = [];
    await loadDocuments();
    const partial = res && res.partial;
    const deleted = (res && res.deleted) || 0;
    alert(partial ? `Удалено: ${deleted}. Есть деградации/ошибки — см. ответ API.` : `Удалено: ${deleted}.`);
  } catch (e) {
    alert(`Ошибка удаления всех документов: ${e.message}`);
  } finally {
    deleteAllBtn.disabled = false;
  }
});

const loadMoreBtn = document.getElementById("load_more_docs");
if (loadMoreBtn) loadMoreBtn.addEventListener("click", () => {
  loadDocuments();
});

function setActiveTab(tab) {
  const all = document.querySelectorAll(".tab-content");
  all.forEach(el => el.classList.toggle("active", el.getAttribute("data-tab") === tab));

  const tabs = document.querySelectorAll(".tab");
  tabs.forEach(btn => btn.classList.toggle("active", btn.getAttribute("data-tab") === tab));

  // Focus composer when switching to chat
  if (tab === "chat") {
    const q = document.getElementById("q");
    if (q) q.focus();
  }
}

function initTabs() {
  document.querySelectorAll(".tab").forEach(btn => {
    btn.addEventListener("click", () => {
      const tab = btn.getAttribute("data-tab");
      if (!tab) return;
      window.location.hash = tab === "files" ? "#files" : "#chat";
      setActiveTab(tab);
    });
  });

  const hash = (window.location.hash || "").replace("#", "");
  setActiveTab(hash === "files" ? "files" : "chat");

  window.addEventListener("hashchange", () => {
    const h = (window.location.hash || "").replace("#", "");
    setActiveTab(h === "files" ? "files" : "chat");
  });
}

// Init
initTabs();
const docIdEl = document.getElementById("doc_id");
if (docIdEl) docIdEl.value = randId();
docsState.offset = 0;
loadDocuments();

// Auto-refresh document list every 30 seconds
setInterval(() => {
  // Don't disrupt pagination browsing: only refresh the first page
  if (docsState.offset <= docsState.limit) {
    docsState.offset = 0;
    loadDocuments();
  }
}, 30000);