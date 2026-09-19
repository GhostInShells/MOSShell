// ghost-in-bilibili: 可拖拽球(灰=停/绿=运行) + 垂直面板(状态/输入/输出/JS审批)
(function () {
  if (document.getElementById('moss-root')) return;

  const BALL = 48, PANEL_W = 300, GAP = 8;
  let running = false;
  let pollTimer = null;
  const page = () => ({
    url: location.href,
    title: document.title,
    bvid: (location.href.match(/BV[0-9A-Za-z]{10}/) || [])[0] || null,
  });

  const send = (msg) => new Promise((resolve) => {
    try { chrome.runtime.sendMessage(msg, (r) => resolve(r || {})); }
    catch (e) { resolve({ error: e.message }); }
  });

  // ---- ball ----
  const ball = document.createElement('div');
  ball.id = 'moss-root';
  ball.textContent = '·';
  Object.assign(ball.style, {
    position: 'fixed', top: '16px', left: (window.innerWidth - BALL - 16) + 'px',
    zIndex: '2147483647', width: BALL + 'px', height: BALL + 'px', borderRadius: '50%',
    background: '#555', color: '#fff',
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    cursor: 'grab', fontSize: '13px', fontFamily: 'ui-monospace, monospace',
    userSelect: 'none', boxShadow: '0 0 0 2px rgba(255,255,255,.15), 0 3px 14px rgba(0,0,0,.6)',
    transition: 'background .2s',
  });
  document.body.appendChild(ball);
  send({ type: 'config' }).then((r) => { if (r && r.ghostName) ball.textContent = r.ghostName; });

  // ---- panel: 状态 → 输入 → 输出 → JS ----
  const panel = document.createElement('div');
  panel.id = 'moss-panel';
  panel.style.cssText = 'position:fixed;z-index:2147483647;width:' + PANEL_W + 'px;' +
    'background:#0f1318;border:1px solid #2a3340;border-radius:6px;padding:10px;display:none;' +
    'font-family:ui-monospace,monospace;font-size:12px;color:#ddd;box-shadow:0 3px 14px rgba(0,0,0,.5);';
  document.body.appendChild(panel);

  const status = document.createElement('div');
  status.style.cssText = 'margin-bottom:6px;color:#8b97a6;font-size:11px;';
  panel.appendChild(status);

  const input = document.createElement('input');
  input.placeholder = '对 ghost 说…(回车发送)';
  input.style.cssText = 'width:100%;background:#06080b;color:#ddd;border:1px solid #2a3340;' +
    'border-radius:4px;padding:6px 8px;font:inherit;box-sizing:border-box;';
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && input.value.trim()) {
      const text = input.value.trim();
      send({ type: 'input', text });
      input.value = '';
      say('→ 已发送: ' + text);
    }
  });
  panel.appendChild(input);

  const output = document.createElement('div');
  output.style.cssText = 'margin-top:6px;color:#8b97a6;font-size:11px;min-height:14px;max-height:80px;' +
    'overflow:auto;word-break:break-all;';
  panel.appendChild(output);

  const jsArea = document.createElement('div');
  panel.appendChild(jsArea);

  function say(text) { output.textContent = text; }

  function updateStatus() {
    status.innerHTML = running
      ? `<span style="color:#3ddc8f">●</span> 感知中 · ${page().bvid || '非视频页'}`
      : '<span style="color:#555">●</span> 已停';
  }
  updateStatus();

  function positionPanel() {
    const r = ball.getBoundingClientRect();
    const estH = 260;
    let top = r.bottom + GAP;
    let left = r.left;
    if (left + PANEL_W > window.innerWidth - GAP) left = window.innerWidth - PANEL_W - GAP;
    if (left < GAP) left = GAP;
    if (top + estH > window.innerHeight - GAP) top = Math.max(GAP, r.top - estH - GAP);
    panel.style.left = left + 'px';
    panel.style.top = top + 'px';
  }
  function showPanel() { positionPanel(); panel.style.display = 'block'; }
  function hidePanel() { panel.style.display = 'none'; }

  // ---- drag vs click ----
  let drag = null, dragged = false;
  ball.addEventListener('mousedown', (e) => {
    const r = ball.getBoundingClientRect();
    drag = { dx: e.clientX - r.left, dy: e.clientY - r.top };
    dragged = false;
    ball.style.cursor = 'grabbing';
  });
  document.addEventListener('mousemove', (e) => {
    if (!drag) return;
    if (Math.abs(e.clientX - drag.dx - ball.getBoundingClientRect().left) +
        Math.abs(e.clientY - drag.dy - ball.getBoundingClientRect().top) > 2) dragged = true;
    ball.style.left = (e.clientX - drag.dx) + 'px';
    ball.style.top = (e.clientY - drag.dy) + 'px';
    positionPanel();
  });
  document.addEventListener('mouseup', () => { drag = null; ball.style.cursor = 'grab'; });

  ball.addEventListener('click', () => {
    if (dragged) return;
    running = !running;
    ball.style.background = running ? '#3ddc8f' : '#555';
    updateStatus();
    send({ type: 'toggle', state: running ? 'on' : 'off', ...page() });
    if (running) { showPanel(); startPoll(); } else { hidePanel(); stopPoll(); }
  });

  // ---- predefined actions (no eval — bilibili CSP blocks 'unsafe-eval') ----
  function findVideo() {
    let v = document.querySelector('video');
    if (v) return v;
    for (const sel of ['bwp-video', 'bilibili-player', '.bpx-player-video-wrap']) {
      const el = document.querySelector(sel);
      if (el && el.shadowRoot) {
        v = el.shadowRoot.querySelector('video');
        if (v) return v;
      }
    }
    return null;
  }

  const ACTIONS = {
    play: () => { const v = findVideo(); if (!v) throw new Error('video not found'); return v.play(); },
    pause: () => { const v = findVideo(); if (!v) throw new Error('video not found'); v.pause(); return 'paused'; },
    seek: (sec) => { const v = findVideo(); if (!v) throw new Error('video not found'); v.currentTime = sec; return 'seeked ' + sec; },
    speed: (r) => { const v = findVideo(); if (!v) throw new Error('video not found'); v.playbackRate = r; return 'speed ' + r; },
    getTime: () => { const v = findVideo(); if (!v) throw new Error('video not found'); return v.currentTime; },
  };

  function execAction(cmd) {
    const fn = ACTIONS[cmd.action];
    if (!fn) return { ok: false, error: 'unknown action ' + cmd.action };
    try {
      const result = fn(cmd.value);
      return { ok: true, result: result === undefined ? '(undefined)' : String(result) };
    } catch (e) {
      return { ok: false, error: e.name + ': ' + e.message };
    }
  }

  function renderCmd(cmd) {
    jsArea.innerHTML = '';
    const pre = document.createElement('pre');
    pre.textContent = cmd.action + (cmd.value !== undefined && cmd.value !== null ? ' ' + cmd.value : '');
    pre.style.cssText = 'white-space:pre-wrap;word-break:break-all;max-height:160px;overflow:auto;' +
      'background:#06080b;border:1px solid #2a3340;border-radius:4px;padding:6px;margin:0 0 6px;';
    jsArea.appendChild(pre);

    const row = document.createElement('div');
    row.style.cssText = 'display:flex;gap:6px;';
    const mkBtn = (label, bg) => {
      const b = document.createElement('button');
      b.textContent = label;
      b.style.cssText = `flex:1;background:${bg};color:#fff;border:0;border-radius:4px;` +
        'padding:6px;cursor:pointer;font:inherit;';
      return b;
    };
    const accept = mkBtn('accept', '#1d6b45');
    const deny = mkBtn('deny', '#6b1d2a');
    accept.onclick = async () => {
      const result = execAction(cmd);
      await send({ type: 'js_result', id: cmd.id, page: cmd.page, ...result });
      say(result.ok ? '✓ ' + result.result : '✗ ' + result.error);
      jsArea.innerHTML = '';
    };
    deny.onclick = async () => {
      await send({ type: 'js_result', id: cmd.id, page: cmd.page, denied: true });
      say('已拒绝');
      jsArea.innerHTML = '';
    };
    row.appendChild(accept); row.appendChild(deny);
    jsArea.appendChild(row);
  }

  const jsQueue = [];
  async function poll() {
    if (jsArea.children.length) return;  // 正在审批,等结果
    if (jsQueue.length) { renderCmd(jsQueue.shift()); return; }
    const r = await send({ type: 'poll', page: page().url });
    if (Array.isArray(r)) jsQueue.push(...r);
    if (jsQueue.length) renderCmd(jsQueue.shift());
  }

  function startPoll() { stopPoll(); pollTimer = setInterval(poll, 500); }
  function stopPoll() { if (pollTimer) { clearInterval(pollTimer); pollTimer = null; } }
})();
