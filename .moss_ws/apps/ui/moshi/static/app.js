let currentMode = "idle";

// ── Chat ─────────────────────────────────────────────────────────────────

function addMessage(role, text) {
    const div = document.createElement("div");
    div.className = "msg " + role;
    const now = new Date();
    const sender = role === "user" ? "你" : role === "ai" ? "AI" : "";
    div.innerHTML =
        '<span class="sender">' + sender + "</span>" +
        '<span class="time">' + now.toLocaleTimeString() + "</span><br>" +
        escapeHtml(text);
    const msgs = document.getElementById("messages");
    msgs.appendChild(div);
    msgs.scrollTop = msgs.scrollHeight;
}

function escapeHtml(s) {
    const d = document.createElement("div");
    d.textContent = s;
    return d.innerHTML;
}

// ── API ──────────────────────────────────────────────────────────────────

async function api(path, body) {
    if (body === undefined) body = {};
    const r = await fetch("/api" + path, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
    });
    return r.json();
}

// ── Actions ──────────────────────────────────────────────────────────────

async function setMode(mode) {
    currentMode = mode;
    const r = await api("/mode", { mode });
    updateToolbar(mode);
    document.getElementById("status").textContent = r.status || "";
    if (r.msg) addMessage("system", r.msg);
}

async function sendCmd(cmd) {
    const r = await api("/cmd", { cmd });
    document.getElementById("status").textContent = r.status || "";
    if (r.info) document.getElementById("chapter-info").textContent = r.info;
    if (r.msg) addMessage("system", r.msg);
}

async function selectCourse(name) {
    const r = await api("/course/select", { name });
    document.getElementById("status").textContent = r.status || "";
    if (r.msg) addMessage("system", r.msg);
}

async function sendChat() {
    const input = document.getElementById("chat-input");
    const text = input.value.trim();
    if (!text) return;
    input.value = "";
    addMessage("user", text);

    const r = await api("/chat", { text, mode: currentMode });
    if (r.reply) {
        addMessage("ai", r.reply);
    } else if (r.error) {
        addMessage("system", "错误: " + r.error);
    }
    if (r.status) document.getElementById("status").textContent = r.status;
    if (r.info) document.getElementById("chapter-info").textContent = r.info;
}

function updateToolbar(mode) {
    document.querySelectorAll("#toolbar button[data-mode]").forEach(function (b) {
        b.classList.toggle("active", b.dataset.mode === mode);
    });
}

// ── Init ─────────────────────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", async function () {
    // Mode buttons
    document.querySelectorAll("#toolbar button[data-mode]").forEach(function (b) {
        b.addEventListener("click", function () {
            setMode(b.dataset.mode);
        });
    });

    // Course select
    document.getElementById("course-select").addEventListener("change", function () {
        selectCourse(this.value);
    });

    // Control buttons
    document.querySelectorAll("#progress button[data-cmd]").forEach(function (b) {
        b.addEventListener("click", function () {
            sendCmd(b.dataset.cmd);
        });
    });

    // Chat
    document.getElementById("send-btn").addEventListener("click", sendChat);
    document.getElementById("chat-input").addEventListener("keydown", function (e) {
        if (e.key === "Enter") sendChat();
    });

    // Load courses
    const cs = await api("/courses");
    const sel = document.getElementById("course-select");
    (cs || []).forEach(function (c) {
        const o = document.createElement("option");
        o.value = c.name;
        o.text = c.name + " (" + c.chapters + ")";
        sel.appendChild(o);
    });

    // Init state
    const s = await api("/state");
    currentMode = s.mode || "idle";
    updateToolbar(currentMode);
    document.getElementById("status").textContent = s.status || "就绪";
});
