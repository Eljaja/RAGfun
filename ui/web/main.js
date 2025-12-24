async function callChatStream(query, onToken, onComplete, onError, onRetrieval) {
  const filters = buildChatFilters();
  const r = await fetch("/api/v1/chat/stream", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      query,
      include_sources: true,
      filters,
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

async function uploadDoc({ file, doc_id, title, uri, source, lang, tags, project_id }) {
  const fd = new FormData();
  fd.append("file", file);
  fd.append("doc_id", doc_id);
  if (title) fd.append("title", title);
  if (uri) fd.append("uri", uri);
  if (source) fd.append("source", source);
  if (lang) fd.append("lang", lang);
  if (tags) fd.append("tags", tags);
  if (project_id) fd.append("project_id", project_id);
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
  const collections = getSelectedValues("files_collections");
  const q = new URLSearchParams();
  q.set("limit", String(limit));
  q.set("offset", String(offset));
  if (collections.length) q.set("collections", collections.join(","));
  const r = await fetch(`/api/v1/documents?${q.toString()}`);
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
  // Page size for /api/v1/documents. We still auto-load all pages,
  // but a larger limit reduces roundtrips.
  limit: 500,
  offset: 0,
};

const collectionsState = {
  items: [], // [{id, count}]
  loaded: false,
};

function getSelectedValues(selectId) {
  const el = document.getElementById(selectId);
  if (!el) return [];
  return Array.from(el.selectedOptions || [])
    .map(o => (o && o.value ? o.value : ""))
    .filter(Boolean);
}

function buildChatFilters() {
  const project_ids = getSelectedValues("chat_collections");
  if (!project_ids.length) return null;
  return { project_ids };
}

function updateDocsMeta() {
  const el = document.getElementById("docs_meta");
  if (!el) return;
  const shown = docsState.items.length;
  const total = docsState.total || 0;
  if (!shown) {
    el.textContent = total ? `Shown: 0 of ${total}` : "";
  } else {
    // Stats by status (based on already loaded docs; in this UI we auto-load all pages).
    const stats = {
      indexed: 0,
      not_indexed: 0,
      queued: 0,
      processing: 0,
      retrying: 0,
      failed: 0,
      unknown: 0,
    };

    for (const doc of docsState.items) {
      const ing = doc && doc.extra && doc.extra.ingestion ? doc.extra.ingestion : null;
      const st = ing && ing.state ? String(ing.state) : "";
      if (st === "queued") stats.queued += 1;
      else if (st === "processing") stats.processing += 1;
      else if (st === "retrying") stats.retrying += 1;
      else if (st === "failed") stats.failed += 1;
      else if (doc && doc.indexed) stats.indexed += 1;
      else if (doc && doc.indexed === false) stats.not_indexed += 1;
      else stats.unknown += 1;
    }

    const left = total ? `Shown: ${shown} of ${total}` : `Shown: ${shown}`;
    const parts = [
      `Indexed: ${stats.indexed}`,
      `Not indexed: ${stats.not_indexed}`,
    ];
    if (stats.queued) parts.push(`Queued: ${stats.queued}`);
    if (stats.processing) parts.push(`Processing: ${stats.processing}`);
    if (stats.retrying) parts.push(`Retrying: ${stats.retrying}`);
    if (stats.failed) parts.push(`Error: ${stats.failed}`);
    if (stats.unknown) parts.push(`Unknown: ${stats.unknown}`);

    el.textContent = `${left} | ${parts.join(" | ")}`;
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
    statusEl.className = "composer-status" + (text.includes("Error") ? " error" : text.includes("OK") ? " success" : "");
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
    statusEl.className = "status" + (text.includes("Error") ? " error" : text.includes("OK") ? " success" : "");
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
    container.innerHTML = '<div style="color: var(--text-secondary); text-align: center; padding: 40px;">No documents found</div>';
    return;
  }

  container.innerHTML = docs
    .map(
      doc => `
    <div class="doc-item">
      <div class="doc-header">
        <div style="flex: 1;">
          <div class="doc-title">${escapeHtml(doc.title || doc.doc_id || "Untitled")}</div>
          <div class="doc-id">ID: ${escapeHtml(doc.doc_id)}</div>
        </div>
        <div class="badges">
          ${doc.storage_id ? '<span class="badge stored">Stored</span>' : ""}
          ${(() => {
            const ing = doc && doc.extra && doc.extra.ingestion ? doc.extra.ingestion : null;
            const st = ing && ing.state ? String(ing.state) : "";
            if (st === "queued") return '<span class="badge queued">Queued</span>';
            if (st === "processing") return '<span class="badge processing">Processing</span>';
            if (st === "retrying") return '<span class="badge queued">Retrying</span>';
            if (st === "failed") return '<span class="badge failed">Error</span>';
            return doc.indexed ? '<span class="badge indexed">Indexed</span>' : '<span class="badge not-indexed">Not indexed</span>';
          })()}
        </div>
      </div>
      <div class="doc-meta">
        ${doc.source ? `<span>Source: ${escapeHtml(doc.source)}</span>` : ""}
        ${doc.project_id ? `<span style="margin-left: 12px;">Collection: ${escapeHtml(doc.project_id)}</span>` : ""}
        ${doc.lang ? `<span style="margin-left: 12px;">Language: ${escapeHtml(doc.lang)}</span>` : ""}
        ${doc.size ? `<span style="margin-left: 12px;">Size: ${formatBytes(doc.size)}</span>` : ""}
        ${doc.stored_at ? `<span style="margin-left: 12px;">Uploaded: ${new Date(doc.stored_at).toLocaleString("en-US")}</span>` : ""}
      </div>
      ${doc.tags && doc.tags.length ? `<div class="doc-meta">Tags: ${doc.tags.map(t => escapeHtml(t)).join(", ")}</div>` : ""}
      <div class="doc-meta" style="margin-top: 10px;">
        <button class="btn btn-danger" data-action="delete-doc" data-doc-id="${escapeHtml(doc.doc_id)}">Delete</button>
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
      const ok = confirm(`Delete document ${docId}?\n\nThis will remove the file from storage and delete chunks from the index.`);
      if (!ok) return;
      try {
        e.currentTarget.disabled = true;
        const res = await deleteDoc(docId);
        await loadDocuments();
        if (res && res.accepted && res.task_id) {
          alert(`Deletion queued.\n\ntask_id=${res.task_id}`);
        }
      } catch (err) {
        e.currentTarget.disabled = false;
        alert(`Delete failed: ${err.message}`);
      }
    });
  });
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

function formatMessageText(text) {
  if (!text) return "";

  // First, escape HTML to prevent XSS
  let escaped = escapeHtml(text);

  // Extract fenced code blocks.
  const codeBlockPlaceholders = [];
  let blockIndex = 0;
  const codeBlockPattern = /```(\w+)?\s*\n([\s\S]*?)```/g;
  escaped = escaped.replace(codeBlockPattern, (_match, lang, code) => {
    const placeholder = `__CODE_BLOCK_${blockIndex}__`;
    const language = lang ? ` data-lang="${lang.trim()}"` : "";
    codeBlockPlaceholders.push(`<pre><code${language}>${code.trim()}</code></pre>`);
    blockIndex += 1;
    return placeholder;
  });

  // Extract inline code.
  const inlineCodePlaceholders = [];
  let inlineIndex = 0;
  escaped = escaped.replace(/`([^`\n]+)`/g, (_match, code) => {
    const placeholder = `__INLINE_CODE_${inlineIndex}__`;
    inlineCodePlaceholders.push(`<code>${code}</code>`);
    inlineIndex += 1;
    return placeholder;
  });

  const applyInlineMarkdown = (value) => {
    let out = value;
    out = out.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
    out = out.replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>");
    out = out.replace(/(^|[^*])\*([^*\n]+)\*(?!\*)/g, "$1<em>$2</em>");
    return out;
  };

  const renderBlocks = (value) => {
    const lines = value.split("\n");
    let html = "";
    let paragraph = [];
    let inUl = false;
    let inOl = false;
    let pendingTable = null;

    const flushParagraph = () => {
      if (!paragraph.length) return;
      const content = applyInlineMarkdown(paragraph.join("<br>"));
      html += `<p>${content}</p>`;
      paragraph = [];
    };

    const closeLists = () => {
      if (inUl) {
        html += "</ul>";
        inUl = false;
      }
      if (inOl) {
        html += "</ol>";
        inOl = false;
      }
    };

    const flushTable = () => {
      if (!pendingTable) return;
      const header = pendingTable.header || [];
      const rows = pendingTable.rows || [];
      const renderRow = (cells, cellTag) => {
        const safe = cells.map(c => `<${cellTag}>${applyInlineMarkdown(c.trim())}</${cellTag}>`).join("");
        return `<tr>${safe}</tr>`;
      };
      let tableHtml = '<table class="md-table">';
      if (header.length) {
        tableHtml += `<thead>${renderRow(header, "th")}</thead>`;
      }
      if (rows.length) {
        tableHtml += `<tbody>${rows.map(r => renderRow(r, "td")).join("")}</tbody>`;
      }
      tableHtml += "</table>";
      html += tableHtml;
      pendingTable = null;
    };

    const parseTableCells = (line) => {
      let trimmed = line.trim();
      if (trimmed.startsWith("|")) trimmed = trimmed.slice(1);
      if (trimmed.endsWith("|")) trimmed = trimmed.slice(0, -1);
      return trimmed.split("|").map(c => c.trim());
    };

    const isTableDivider = (line) => {
      const trimmed = line.trim();
      if (!trimmed.includes("|")) return false;
      const raw = trimmed.replace(/\s+/g, "");
      if (!raw.startsWith("|") && !raw.includes("|")) return false;
      const parts = parseTableCells(trimmed);
      return parts.length >= 2 && parts.every(p => /^:?-{3,}:?$/.test(p));
    };

    for (let i = 0; i < lines.length; i += 1) {
      const line = lines[i];
      const trimmed = line.trim();
      if (pendingTable) {
        if (!trimmed || !trimmed.includes("|")) {
          flushTable();
        } else {
          pendingTable.rows.push(parseTableCells(trimmed));
          continue;
        }
      }
      if (!trimmed) {
        flushParagraph();
        closeLists();
        continue;
      }

      const nextLine = lines[i + 1] || "";
      if (trimmed.includes("|") && isTableDivider(nextLine)) {
        flushParagraph();
        closeLists();
        const header = parseTableCells(trimmed);
        pendingTable = { header, rows: [] };
        i += 1;
        continue;
      }

      const headingMatch = trimmed.match(/^(#{1,3})\s+(.*)$/);
      if (headingMatch) {
        flushParagraph();
        closeLists();
        flushTable();
        const level = headingMatch[1].length;
        const tag = level === 1 ? "h3" : level === 2 ? "h4" : "h5";
        html += `<${tag}>${applyInlineMarkdown(headingMatch[2])}</${tag}>`;
        continue;
      }

      const quoteMatch = trimmed.match(/^>\s+(.*)$/);
      if (quoteMatch) {
        flushParagraph();
        closeLists();
        flushTable();
        html += `<blockquote>${applyInlineMarkdown(quoteMatch[1])}</blockquote>`;
        continue;
      }

      const ulMatch = trimmed.match(/^[-*]\s+(.*)$/);
      if (ulMatch) {
        flushParagraph();
        flushTable();
        if (inOl) {
          html += "</ol>";
          inOl = false;
        }
        if (!inUl) {
          html += "<ul>";
          inUl = true;
        }
        html += `<li>${applyInlineMarkdown(ulMatch[1])}</li>`;
        continue;
      }

      const olMatch = trimmed.match(/^\d+\.\s+(.*)$/);
      if (olMatch) {
        flushParagraph();
        flushTable();
        if (inUl) {
          html += "</ul>";
          inUl = false;
        }
        if (!inOl) {
          html += "<ol>";
          inOl = true;
        }
        html += `<li>${applyInlineMarkdown(olMatch[1])}</li>`;
        continue;
      }

      paragraph.push(line);
    }

    flushParagraph();
    closeLists();
    flushTable();
    return html;
  };

  let html = renderBlocks(escaped);

  // Restore inline code and code blocks.
  inlineCodePlaceholders.forEach((snippet, index) => {
    html = html.replace(`__INLINE_CODE_${index}__`, snippet);
  });
  codeBlockPlaceholders.forEach((block, index) => {
    html = html.replace(`__CODE_BLOCK_${index}__`, block);
  });

  return html;
}

function formatBytes(bytes) {
  if (!bytes) return "0 B";
  const k = 1024;
  const sizes = ["B", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return Math.round(bytes / Math.pow(k, i) * 100) / 100 + " " + sizes[i];
}

async function fetchCollections() {
  const r = await fetch("/api/v1/collections");
  const data = await r.json().catch(() => ({}));
  if (!r.ok) {
    const msg = (data && (data.detail || data.error)) || `HTTP ${r.status}`;
    throw new Error(msg);
  }
  return data;
}

function populateCollectionsUI(items) {
  const list = items || [];

  const datalist = document.getElementById("collections_datalist");
  if (datalist) {
    datalist.innerHTML = list.map(c => `<option value="${escapeHtml(c.id)}"></option>`).join("");
  }

  const mkOptions = list
    .map(
      c =>
        `<option value="${escapeHtml(c.id)}">${escapeHtml(c.id)}${typeof c.count === "number" ? ` (${c.count})` : ""}</option>`
    )
    .join("");

  const filesSel = document.getElementById("files_collections");
  if (filesSel) filesSel.innerHTML = mkOptions;

  const chatSel = document.getElementById("chat_collections");
  if (chatSel) chatSel.innerHTML = mkOptions;
}

async function loadCollections() {
  try {
    const data = await fetchCollections();
    const items = (data && data.collections) || [];
    collectionsState.items = items;
    collectionsState.loaded = true;
    populateCollectionsUI(items);
  } catch (e) {
    console.warn("Failed to load collections:", e);
  }
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
  
  if (role === "assistant") {
    // Format assistant messages with markdown support
    bubbleText.innerHTML = formatMessageText(text || "");
  } else {
    // User messages - plain text with line breaks
    bubbleText.innerHTML = escapeHtml(text || "").replace(/\n/g, '<br>');
  }
  
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
    d.innerHTML = `<summary>Sources (${sources.length})</summary>${sourcesHtml(fmtSources(sources) || [])}`;
    meta.appendChild(d);
  }

  if (context && context.length) {
    const d = document.createElement("details");
    d.open = false;
    d.innerHTML = `<summary>Context (retrieval) (${context.length})</summary>${retrievalHtml(context)}`;
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
    alert("Choose a file");
    return;
  }

  const docIdEl = document.getElementById("doc_id");
  const doc_id = (docIdEl && docIdEl.value ? docIdEl.value.trim() : "") || randId();
  const title = document.getElementById("title").value.trim();
  const uri = document.getElementById("uri").value.trim();
  const source = document.getElementById("source").value.trim();
  const lang = document.getElementById("lang").value.trim();
  const tags = document.getElementById("tags").value.trim();
  const project_id = document.getElementById("upload_collection").value.trim();

  setUploadBusy(true, "Uploading...");
  try {
    const data = await uploadDoc({ file, doc_id, title, uri, source, lang, tags, project_id });
    // Show the actually used doc_id (useful when input was empty and we auto-generated it)
    if (data && data.accepted && data.task_id) {
      setUploadBusy(false, `Queued | task_id=${data.task_id} | doc_id=${doc_id}`);
    } else {
      setUploadBusy(false, `OK | doc_id=${doc_id}`);
    }
    // After each successful upload, generate a new doc_id in UI to avoid accidental overwrite
    if (docIdEl) docIdEl.value = randId();
    // Optional UX: clear file input so the next upload is explicit
    if (fileEl) fileEl.value = "";
    // Refresh document list
    loadDocuments();
    // Refresh collections list (new collection might appear)
    loadCollections();
  } catch (e) {
    setUploadBusy(false, "Error");
    alert(`Upload failed: ${e.message}`);
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

  setBusy(true, "Sending...");
  let answerText = "";

  try {
    await callChatStream(
      q,
      token => {
        answerText += token;
        if (assistant && assistant.bubbleText) {
          assistant.bubbleText.innerHTML = formatMessageText(answerText);
          scrollMessagesToBottom();
        }
      },
      data => {
        // done
        if (assistant && assistant.bubbleText) assistant.bubbleText.innerHTML = formatMessageText(answerText);
        if (assistant && assistant.wrapper && data) {
          setAssistantExtras(assistant.wrapper, { sources: data.sources, context: data.context || latestContext });
        }
        const flags = [];
        if (data && data.partial) flags.push("partial");
        if (data && data.degraded && data.degraded.length) flags.push(`degraded=${data.degraded.join(",")}`);
        setBusy(false, flags.length ? flags.join(" | ") : "OK");
      },
      error => {
        if (assistant && assistant.bubbleText) assistant.bubbleText.innerHTML = escapeHtml(`Error: ${error.message}`).replace(/\n/g, '<br>');
        setBusy(false, "Error");
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
    if (assistant && assistant.bubbleText) assistant.bubbleText.innerHTML = escapeHtml(`Error: ${e.message}`).replace(/\n/g, '<br>');
    setBusy(false, "Error");
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
  const loadMoreBtn = document.getElementById("load_more_docs");
  const refreshBtn = document.getElementById("refresh_docs");
  const applyFilesFiltersBtn = document.getElementById("apply_files_filters");
  const deleteAllBtn = document.getElementById("delete_all_docs");

  const setDocsBusy = busy => {
    if (loadMoreBtn) loadMoreBtn.disabled = busy || (docsState.total && docsState.items.length >= docsState.total);
    if (refreshBtn) refreshBtn.disabled = busy;
    if (applyFilesFiltersBtn) applyFilesFiltersBtn.disabled = busy;
    if (deleteAllBtn) deleteAllBtn.disabled = busy;
  };

  try {
    setDocsBusy(true);

    // Auto-load ALL documents, not only the first page.
    // We render incrementally so the UI shows progress.
    let safety = 0;
    while (true) {
      safety += 1;
      if (safety > 10000) break; // safety valve against infinite loops

      const data = await listDocuments();
      const docs = data.documents || [];

      if (typeof data.total === "number") docsState.total = data.total;

      if (docsState.offset === 0) {
        docsState.items = docs;
      } else {
        // append next page; backend sorts by stored_at desc, so this is stable for browsing
        docsState.items = docsState.items.concat(docs);
      }
      docsState.offset += docs.length;

      renderDocuments(docsState.items);
      updateDocsMeta();

      const total = docsState.total || 0;
      if (!docs.length) break;
      if (total && docsState.items.length >= total) break;
      if (docs.length < docsState.limit) break;

      // Let the browser paint between pages.
      await new Promise(r => setTimeout(r, 0));
    }
  } catch (e) {
    container.innerHTML = `<div style="color: var(--error); text-align: center; padding: 40px;">Failed to load: ${escapeHtml(e.message)}</div>`;
  } finally {
    setDocsBusy(false);
    updateDocsMeta();
  }
}

const refreshBtn = document.getElementById("refresh_docs");
if (refreshBtn) refreshBtn.addEventListener("click", () => {
  docsState.offset = 0;
  loadDocuments();
});

const applyFilesFiltersBtn = document.getElementById("apply_files_filters");
if (applyFilesFiltersBtn) applyFilesFiltersBtn.addEventListener("click", () => {
  docsState.offset = 0;
  docsState.items = [];
  loadDocuments();
});

const deleteAllBtn = document.getElementById("delete_all_docs");
if (deleteAllBtn) deleteAllBtn.addEventListener("click", async () => {
  const ok = confirm(
    "Delete ALL documents?\n\n" +
      "- removes files from document-storage\n" +
      "- deletes chunks from the retrieval index\n\n" +
      "This action cannot be undone."
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
    alert(partial ? `Deleted: ${deleted}. Some degradations/errors occurred -- check the API response.` : `Deleted: ${deleted}.`);
  } catch (e) {
    alert(`Failed to delete all documents: ${e.message}`);
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
loadCollections();

// Auto-refresh document list every 30 seconds
setInterval(() => {
  // Don't disrupt pagination browsing: only refresh the first page
  if (docsState.offset <= docsState.limit) {
    docsState.offset = 0;
    loadDocuments();
  }
}, 30000);
