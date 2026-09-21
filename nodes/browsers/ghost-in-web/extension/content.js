// ghost-in-web content script.
//
// 页面侧:图标(灰=停 / 青=授权感知本页)+ 两颗卫星,都排在图标左边可见处:
//   「截」—— 截图发给 ghost,仅这一次(人类点了才推一张)
//   「话」—— 打开/收起与 ghost 的对话面板(面板不常驻,默认收起)
// 除阅读外的一切操作(click/type)先弹一条确认条,人类接受才执行 —— 审批面就在页面上,
// node 不参与审批,只拿到"接受/拒绝"的结果。
//
// 模型下发的 cmd 只走枚举动作表,绝不在页面里 eval —— CSP 禁 unsafe-eval,这也是安全
// 性质:ghost 对页面的权力恰好是这张表。

(function () {
  // ---- local 地址不生效 ----
  // manifest 的 exclude_matches 已挡了 127.0.0.1/localhost,这里再挡私网段与 file://,
  // 让"审计面(node 的页面)与被控面靠地址空间分开"这条成立,而不是靠逻辑判断。
  const host = location.hostname;
  const isLocal =
    location.protocol === 'file:' || location.protocol === 'chrome:' ||
    /^(localhost|0\.0\.0\.0|\[::1\])$/.test(host) || host.endsWith('.local') ||
    /^127\./.test(host) || /^10\./.test(host) || /^192\.168\./.test(host) ||
    /^172\.(1[6-9]|2\d|3[01])\./.test(host);
  if (isLocal) return;
  if (document.getElementById('moss-giw-root')) return;

  const ACCENT = '#2dd4bf'; // 青色 —— 与 bilibili body 的绿明确区分
  const IDLE = '#555';
  const SAT_BG = '#1b222e';
  const BALL = 40;
  const SAT = 24;

  let perceived = false;
  let lastHref = location.href;
  let lastTitle = document.title;

  const refs = new Map(); // ref -> element
  let refSeq = 0;

  function send(msg) {
    try { chrome.runtime.sendMessage(msg, () => void chrome.runtime.lastError); } catch (e) {}
  }

  // ---- 交互元素定位 ------------------------------------------------------

  const INTERACTIVE = 'a[href],button,input,select,textarea,[role=button],[role=link],' +
    '[role=textbox],[onclick],[tabindex]';

  function visible(el) {
    const r = el.getBoundingClientRect();
    if (r.width < 1 || r.height < 1) return false;
    const s = getComputedStyle(el);
    return s.visibility !== 'hidden' && s.display !== 'none' && s.opacity !== '0';
  }

  function labelOf(el) {
    const t = (el.innerText || el.value || el.placeholder || el.getAttribute('aria-label') || '')
      .replace(/\s+/g, ' ').trim();
    return t.slice(0, 80);
  }

  function newRef(el) {
    refSeq += 1;
    const ref = 'r' + refSeq;
    refs.set(ref, el);
    if (refs.size > 500) refs.delete(refs.keys().next().value); // 丢最早
    return ref;
  }

  function findElements(text, limit) {
    const needle = (text || '').toLowerCase();
    const hits = [];
    for (const el of document.querySelectorAll(INTERACTIVE)) {
      if (el.closest('[data-moss]')) continue;  // 跳过自己注入的浮层
      if (!visible(el)) continue;
      const label = labelOf(el);
      if (needle && !(label.toLowerCase().includes(needle))) continue;
      hits.push({ ref: newRef(el), tag: el.tagName.toLowerCase(), text: label });
      if (hits.length >= limit) break;
    }
    return hits;
  }

  function elementOf(ref) {
    const el = refs.get(ref);
    if (!el) throw new Error('ref ' + ref + ' 已失效,先 find');
    return el;
  }

  // ---- 动作表 ------------------------------------------------------------

  function setValue(el, text) {
    if (el.isContentEditable) {
      el.textContent = text;
    } else {
      const proto = el instanceof HTMLTextAreaElement
        ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
      const setter = Object.getOwnPropertyDescriptor(proto, 'value')?.set;
      if (setter) setter.call(el, text); else el.value = text;
    }
    el.dispatchEvent(new Event('input', { bubbles: true }));
    el.dispatchEvent(new Event('change', { bubbles: true }));
  }

  const ACTIONS = {
    read: () => {
      // 读正文前把浮层藏掉,否则 innerText 会把图标/面板文字一起读进来。
      const injected = [ball, satWrap, panel, confirm];
      const prev = injected.map((el) => el.style.display);
      injected.forEach((el) => { el.style.display = 'none'; });
      let text;
      try {
        text = (document.body?.innerText || '').replace(/\n{3,}/g, '\n\n').trim();
      } finally {
        injected.forEach((el, i) => { el.style.display = prev[i]; });
      }
      return text.slice(0, 4000) + (text.length > 4000 ? '\n…(已截断)' : '');
    },
    find: (v) => findElements(v.text, v.limit || 20)
      .map((h) => `${h.ref} <${h.tag}> "${h.text}"`),
    click: (ref) => { elementOf(ref).click(); return 'clicked ' + ref; },
    type: (v) => {
      const el = elementOf(v.ref);
      el.focus();
      setValue(el, v.text);
      return 'typed into ' + v.ref;
    },
  };

  // 有后果的动作要先经人类确认 —— 确认条就是审批面。
  const GATED = new Set(['click', 'type']);

  async function execAction(cmd) {
    const fn = ACTIONS[cmd.action];
    if (!fn) return { ok: false, error: 'unknown action ' + cmd.action };
    try {
      if (GATED.has(cmd.action)) {
        const ok = await confirmBar(describe(cmd));
        if (!ok) return { ok: false, accepted: false, result: 'human rejected' };
      }
      const result = await fn(cmd.value);
      return { ok: true, accepted: true, result: result === undefined ? '(done)' : String(result) };
    } catch (e) {
      return { ok: false, error: e.name + ': ' + e.message };
    }
  }

  function describe(cmd) {
    if (cmd.action === 'click') {
      const el = refs.get(cmd.value);
      return `ghost 想点击「${el ? labelOf(el) : cmd.value}」`;
    }
    if (cmd.action === 'type') {
      const el = refs.get(cmd.value.ref);
      return `ghost 想往「${el ? labelOf(el) : cmd.value.ref}」填入「${cmd.value.text}」`;
    }
    return 'ghost 想做 ' + cmd.action;
  }

  // ---- UI: 图标 ----------------------------------------------------------

  const ball = document.createElement('div');
  ball.id = 'moss-giw-root';
  ball.setAttribute('data-moss', '1');
  ball.textContent = 'W';
  ball.title = 'ghost-in-web · 点我授权 ghost 感知本页(再点撤销)';
  Object.assign(ball.style, {
    position: 'fixed', top: '16px', left: (window.innerWidth - BALL - 16) + 'px',
    zIndex: '2147483647', width: BALL + 'px', height: BALL + 'px',
    borderRadius: '10px', background: IDLE, color: '#fff',
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    cursor: 'grab', fontSize: '13px', fontWeight: '700',
    fontFamily: 'ui-monospace, monospace',
    userSelect: 'none', boxShadow: '0 0 0 2px rgba(255,255,255,.15), 0 3px 14px rgba(0,0,0,.6)',
    transition: 'background .2s',
  });
  document.body.appendChild(ball);

  // ---- UI: 轻提示(toast)—— 卫星点击成功的明确反馈 ------------------------
  const toast = document.createElement('div');
  toast.setAttribute('data-moss', '1');
  toast.style.cssText = 'position:fixed;z-index:2147483647;top:16px;left:50%;' +
    'transform:translateX(-50%);background:#141a22;border:1px solid ' + ACCENT + ';' +
    'color:#d7dee8;border-radius:6px;padding:8px 14px;font-family:ui-monospace,monospace;' +
    'font-size:12px;display:none;box-shadow:0 3px 14px rgba(0,0,0,.6);';
  document.body.appendChild(toast);
  let toastTimer = null;
  function showToast(text, ok = true) {
    toast.textContent = text;
    toast.style.borderColor = ok ? ACCENT : '#e06c75';
    toast.style.display = 'block';
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { toast.style.display = 'none'; }, 1600);
  }

  // ---- UI: 卫星(排在图标左边,保证可见) -----------------------------------

  const satWrap = document.createElement('div');
  satWrap.setAttribute('data-moss', '1');
  // 必须钉 top:0;left:0 —— 否则 fixed 元素落到静态位置(append 到 body 末尾 = 页面底部),
  // 卫星是相对它 absolute 定位的,就会跑到视口外看不见。
  satWrap.style.cssText = 'position:fixed;top:0;left:0;z-index:2147483647;pointer-events:none;';
  document.body.appendChild(satWrap);

  function mkSat(label, title, onClick) {
    const s = document.createElement('div');
    s.textContent = label;
    s.title = title;
    Object.assign(s.style, {
      position: 'absolute', width: SAT + 'px', height: SAT + 'px', borderRadius: '50%',
      background: SAT_BG, color: '#d7dee8', border: '1px solid ' + ACCENT,
      boxShadow: '0 0 0 2px rgba(255,255,255,.25), 0 2px 8px rgba(0,0,0,.6)',
      display: 'flex', alignItems: 'center',
      justifyContent: 'center', cursor: 'pointer', pointerEvents: 'auto',
      fontSize: '11px', fontFamily: 'ui-monospace, monospace', userSelect: 'none',
      transition: 'background .2s',
    });
    s.addEventListener('click', onClick);
    satWrap.appendChild(s);
    return s;
  }

  let shotFlash = null;
  const shotSat = mkSat('截', '截图发给 ghost(仅这一次)', () => {
    if (!perceived) { showToast('先点图标授权感知', false); return; }
    chrome.runtime.sendMessage({ type: 'shot' }, (resp) => {
      if (chrome.runtime.lastError || !resp || !resp.ok) {
        showToast('截图失败', false);
        return;
      }
      shotSat.textContent = '✓';
      shotSat.style.background = ACCENT;
      showToast('截图已发送给 ghost');
      clearTimeout(shotFlash);
      shotFlash = setTimeout(() => {
        shotSat.textContent = '截';
        shotSat.style.background = SAT_BG;
      }, 1200);
    });
  });

  const talkSat = mkSat('话', '打开/收起与 ghost 的对话面板', () => {
    togglePanel();
  });
  let talkFlash = null;
  function pingTalk() {
    talkSat.style.background = ACCENT;
    clearTimeout(talkFlash);
    talkFlash = setTimeout(() => { talkSat.style.background = SAT_BG; }, 800);
  }

  const sats = [shotSat, talkSat];

  function placeSats() {
    const r = ball.getBoundingClientRect();
    const cx = r.left + BALL / 2, cy = r.top + BALL / 2;
    const dx = BALL / 2 + SAT / 2 + 6; // 全部排到图标左边
    sats.forEach((s, i) => {
      const dy = (i - (sats.length - 1) / 2) * (SAT + 6);
      s.style.left = (cx - dx - SAT / 2) + 'px';
      s.style.top = (cy + dy - SAT / 2) + 'px';
    });
  }
  placeSats();

  // ---- 拖动 vs 点击 ------------------------------------------------------

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
    placePanel();
    placeConfirm();
  });
  document.addEventListener('mouseup', () => { drag = null; ball.style.cursor = 'grab'; });

  ball.addEventListener('click', () => {
    if (dragged) return;
    perceived = !perceived;
    ball.style.background = perceived ? ACCENT : IDLE;
    updateStatus();
    send({ type: 'auth', on: perceived });
    if (!perceived) hidePanel(); // 撤销感知时收起面板,不打扰
  });

  // ---- 面板:状态 → 输入 → 输出(默认收起,「话」卫星打开) ------------------

  const panel = document.createElement('div');
  panel.setAttribute('data-moss', '1');
  panel.style.cssText = 'position:fixed;z-index:2147483647;width:240px;background:#0f1318;' +
    'border:1px solid #2a3340;border-radius:6px;padding:8px;display:none;' +
    'font-family:ui-monospace,monospace;font-size:12px;color:#ddd;box-shadow:0 3px 14px rgba(0,0,0,.5);';
  document.body.appendChild(panel);

  const status = document.createElement('div');
  status.style.cssText = 'margin-bottom:6px;color:#8b97a6;font-size:11px;';
  panel.appendChild(status);

  const input = document.createElement('input');
  input.placeholder = '对 ghost 说…(回车发送)';
  input.style.cssText = 'width:100%;background:#06080b;color:#ddd;border:1px solid #2a3340;' +
    'border-radius:4px;padding:5px 7px;font:inherit;box-sizing:border-box;';
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
    'max-height:70px;overflow:auto;word-break:break-all;';
  panel.appendChild(output);

  function flashPanel(text) { output.textContent = text; }

  function updateStatus() {
    status.innerHTML = perceived
      ? `<span style="color:${ACCENT}">●</span> 感知中`
      : '<span style="color:#555">●</span> 已停';
  }
  updateStatus();

  function showPanel() { panel.style.display = 'block'; placePanel(); }
  function hidePanel() { panel.style.display = 'none'; }
  function togglePanel() {
    if (panel.style.display === 'none') showPanel(); else hidePanel();
  }

  function placePanel() {
    const r = ball.getBoundingClientRect();
    let top = r.bottom + 8;
    let left = r.left - (240 - BALL) / 2;
    if (left + 240 > window.innerWidth - 8) left = window.innerWidth - 240 - 8;
    if (left < 8) left = 8;
    if (top + 120 > window.innerHeight - 8) top = Math.max(8, r.top - 120 - 8);
    panel.style.left = left + 'px';
    panel.style.top = top + 'px';
  }

  // ---- 确认条(审批面) ---------------------------------------------------

  const confirm = document.createElement('div');
  confirm.setAttribute('data-moss', '1');
  confirm.style.cssText = 'position:fixed;z-index:2147483647;width:240px;display:none;' +
    'background:#141a22;border:1px solid ' + ACCENT + ';border-radius:6px;padding:10px;' +
    'font-family:ui-monospace,monospace;font-size:12px;color:#ddd;box-shadow:0 3px 14px rgba(0,0,0,.6);';
  document.body.appendChild(confirm);

  const confirmText = document.createElement('div');
  confirmText.style.cssText = 'margin-bottom:8px;line-height:1.4;';
  confirm.appendChild(confirmText);

  const btnRow = document.createElement('div');
  btnRow.style.cssText = 'display:flex;gap:8px;';
  confirm.appendChild(btnRow);

  function mkBtn(label, color) {
    const b = document.createElement('button');
    b.textContent = label;
    b.style.cssText = 'flex:1;padding:5px 0;border-radius:4px;border:1px solid #2a3340;' +
      'background:#1b2230;color:' + color + ';font:inherit;cursor:pointer;';
    btnRow.appendChild(b);
    return b;
  }
  const acceptBtn = mkBtn('接受', ACCENT);
  const rejectBtn = mkBtn('拒绝', '#e06c75');

  let confirmResolve = null;

  function askConfirm(text) {
    confirmText.textContent = text;
    confirm.style.display = 'block';
    placeConfirm();
    return new Promise((resolve) => { confirmResolve = resolve; });
  }

  function settleConfirm(ok) {
    confirm.style.display = 'none';
    const r = confirmResolve;
    confirmResolve = null;
    if (r) r(ok);
  }

  acceptBtn.addEventListener('click', () => settleConfirm(true));
  rejectBtn.addEventListener('click', () => settleConfirm(false));

  function placeConfirm() {
    const r = ball.getBoundingClientRect();
    let top = r.top - 92;
    if (top < 8) top = r.bottom + 8;
    let left = r.left - (240 - BALL) / 2;
    if (left + 240 > window.innerWidth - 8) left = window.innerWidth - 240 - 8;
    if (left < 8) left = 8;
    confirm.style.left = left + 'px';
    confirm.style.top = top + 'px';
  }

  const confirmBar = (text) => askConfirm(text);

  // ---- cmd + say(from SW via tabs.sendMessage) ---------------------------

  chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
    if (msg.type === 'cmd') {
      execAction(msg).then((result) => {
        send({ type: 'result', cid: msg.cid, ...result });
      });
      sendResponse({ ok: true });
      return false;
    }
    if (msg.type === 'say') {
      flashPanel('ghost: ' + msg.text);
      if (panel.style.display === 'none') pingTalk(); // 面板关着就亮「话」卫星提示有新消息
      sendResponse({ ok: true });
      return false;
    }
  });

  // ---- reporting ---------------------------------------------------------

  function reportContent() {
    send({ type: 'content', title: document.title, url: location.href });
  }

  // SPA 导航(点搜索按钮触发的前端路由)会把注入的浮层从 body 里抹掉,而内容脚本
  // 不会重跑。定期检查,球没了就重新挂回去(状态都在闭包里,挂回即恢复)。
  function ensureAttached() {
    if (ball.isConnected) return;
    for (const el of [ball, toast, satWrap, panel, confirm]) {
      if (el && !el.isConnected && document.body) document.body.appendChild(el);
    }
    placeSats();
    if (panel.style.display === 'block') placePanel();
  }

  // 只报内容变化(导航/标题),不报人的行为 —— url 变、标题变就是"内容变了"。
  function watchContent() {
    setInterval(() => {
      ensureAttached();
      if (location.href !== lastHref || document.title !== lastTitle) {
        lastHref = location.href;
        lastTitle = document.title;
        reportContent();
        updateStatus();
      }
    }, 1000);
  }

  // 心跳:SW 空闲 ~30s 会被回收,这条消息既续命又唤醒(sw.js 不把它上行)。
  setInterval(() => send({ type: 'ping' }), 2000);

  reportContent();
  watchContent();
})();
