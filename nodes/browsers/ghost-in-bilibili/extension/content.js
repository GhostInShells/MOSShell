// ghost-in-bilibili content script.
//
// 页面侧:主球(灰=停 / 绿=授权感知本页)+ 环绕卫星(每颗=一项能力,绿=授权)。
// 内容变化(B 站自动播放换视频)靠"url 变化 + video src 变化"双信号检测,不只盯 url。
// 模型下发的 cmd 只走枚举动作表,绝不在页面里 eval —— CSP 禁 unsafe-eval,这也是
// 安全性质:ghost 对页面的权力恰好是这张表。

(function () {
  if (document.getElementById('moss-gib-root')) return;

  const BALL = 44;
  const SAT = 20;
  const GROUPS = [
    { key: 'sense', label: '状态' },
    { key: 'control', label: '控制' },
    { key: 'subtitle', label: '字幕' },
    { key: 'interact', label: '弹幕' },
  ];

  let presence = false;
  const grants = { sense: false, control: false, subtitle: false, interact: false };
  let lastHref = location.href;
  let lastSrc = '';

  const bvid = () => (location.href.match(/BV[0-9A-Za-z]{10}/) || [])[0] || null;
  const title = () => document.title;

  function send(msg) {
    try { chrome.runtime.sendMessage(msg, () => void chrome.runtime.lastError); } catch (e) {}
  }

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

  // ---- UI ---------------------------------------------------------------

  const ball = document.createElement('div');
  ball.id = 'moss-gib-root';
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

  // 卫星:四颗小球围着主球排成十字
  const satWrap = document.createElement('div');
  satWrap.style.cssText = 'position:fixed;z-index:2147483647;pointer-events:none;';
  document.body.appendChild(satWrap);
  const sats = GROUPS.map((g, i) => {
    const s = document.createElement('div');
    s.textContent = g.label[0];
    s.title = `${g.label} (未授权)`;
    Object.assign(s.style, {
      position: 'absolute', width: SAT + 'px', height: SAT + 'px', borderRadius: '50%',
      background: '#444', color: '#fff', display: 'flex', alignItems: 'center',
      justifyContent: 'center', cursor: 'pointer', pointerEvents: 'auto',
      fontSize: '10px', fontFamily: 'ui-monospace, monospace', userSelect: 'none',
      transition: 'background .2s',
    });
    s.addEventListener('click', () => {
      grants[g.key] = !grants[g.key];
      paintSat(s, grants[g.key]);
      s.title = `${g.label} (${grants[g.key] ? '已授权' : '未授权'})`;
      send({ type: 'auth', group: g.key, on: grants[g.key] });
    });
    satWrap.appendChild(s);
    return s;
  });

  function paintSat(s, on) { s.style.background = on ? '#3ddc8f' : '#444'; }

  function placeSats() {
    const r = ball.getBoundingClientRect();
    const cx = r.left + BALL / 2, cy = r.top + BALL / 2;
    const orbit = BALL / 2 + SAT / 2 + 6;
    const pts = [
      [cx, cy - orbit], [cx + orbit, cy], [cx, cy + orbit], [cx - orbit, cy],
    ];
    sats.forEach((s, i) => {
      s.style.left = (pts[i][0] - SAT / 2) + 'px';
      s.style.top = (pts[i][1] - SAT / 2) + 'px';
    });
  }
  placeSats();

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
    placeSats();
  });
  document.addEventListener('mouseup', () => { drag = null; ball.style.cursor = 'grab'; });

  // ---- panel:状态 → 输入 → 输出 ----
  const panel = document.createElement('div');
  panel.style.cssText = 'position:fixed;z-index:2147483647;width:300px;background:#0f1318;' +
    'border:1px solid #2a3340;border-radius:6px;padding:10px;display:none;' +
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
      output.textContent = '→ 已发送: ' + text;
    }
  });
  panel.appendChild(input);

  const output = document.createElement('div');
  output.style.cssText = 'margin-top:6px;color:#8b97a6;font-size:11px;min-height:14px;' +
    'max-height:80px;overflow:auto;word-break:break-all;';
  panel.appendChild(output);

  function say(text) { output.textContent = text; }

  function updateStatus() {
    status.innerHTML = presence
      ? '<span style="color:#3ddc8f">●</span> 感知中 · ' + (bvid() || '非视频页')
      : '<span style="color:#555">●</span> 已停';
  }
  updateStatus();

  function placePanel() {
    const r = ball.getBoundingClientRect();
    let top = r.bottom + 8;
    let left = r.left - (300 - BALL) / 2;
    if (left + 300 > window.innerWidth - 8) left = window.innerWidth - 300 - 8;
    if (left < 8) left = 8;
    if (top + 120 > window.innerHeight - 8) top = Math.max(8, r.top - 120 - 8);
    panel.style.left = left + 'px';
    panel.style.top = top + 'px';
  }

  ball.addEventListener('click', () => {
    if (dragged) return;
    presence = !presence;
    ball.style.background = presence ? '#3ddc8f' : '#555';
    updateStatus();
    send({ type: 'auth', group: null, on: presence });
    panel.style.display = presence ? 'block' : 'none';
    if (presence) placePanel();
  });

  // ---- cmd + say(from SW via tabs.sendMessage) ----
  const ACTIONS = {
    play: async () => { const v = findVideo(); if (!v) throw new Error('video not found'); await v.play(); return 'playing'; },
    pause: () => { const v = findVideo(); if (!v) throw new Error('video not found'); v.pause(); return 'paused'; },
    seek: (sec) => { const v = findVideo(); if (!v) throw new Error('video not found'); v.currentTime = sec; return 'seeked ' + sec; },
    speed: (r) => { const v = findVideo(); if (!v) throw new Error('video not found'); v.playbackRate = r; return 'speed ' + r; },
  };

  async function execAction(cmd) {
    const fn = ACTIONS[cmd.action];
    if (!fn) return { ok: false, error: 'unknown action ' + cmd.action };
    try {
      const result = await fn(cmd.value);
      return { ok: true, result: result === undefined ? '(done)' : String(result) };
    } catch (e) {
      return { ok: false, error: e.name + ': ' + e.message };
    }
  }

  chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
    if (msg.type === 'cmd') {
      execAction(msg).then((result) => {
        send({ type: 'result', cid: msg.cid, ...result });
      });
      sendResponse({ ok: true });
      return false;
    }
    if (msg.type === 'say') {
      say(msg.text);
      sendResponse({ ok: true });
      return false;
    }
  });

  // ---- reporting ---------------------------------------------------------

  function reportContent() {
    send({ type: 'content', bvid: bvid(), title: title(), url: location.href });
  }

  function reportState() {
    const v = findVideo();
    if (!v) return;
    send({
      type: 'state', t: v.currentTime, paused: v.paused,
      rate: v.playbackRate, duration: (v.duration || 0),
    });
  }

  function watchContent() {
    setInterval(() => {
      const v = findVideo();
      const src = v ? (v.currentSrc || v.src) : '';
      if (location.href !== lastHref || src !== lastSrc) {
        lastHref = location.href;
        lastSrc = src;
        reportContent();
        updateStatus();
      }
    }, 1000);
  }

  reportContent();
  watchContent();
  setInterval(reportState, 2000);
})();
