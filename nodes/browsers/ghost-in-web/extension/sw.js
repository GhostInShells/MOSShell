// ghost-in-web service worker.
//
// 一条 WS 复用一个浏览器 session(一个扩展一个 SW)。MV3 的 SW 空闲 ~30s 会被回收,
// 所以内容脚本每 2s 发一条 `ping` —— 这条消息既续命(每次消息重置空闲计时)又唤醒
// (睡了会被消息叫醒)。这里不靠 chrome.alarms(最小周期 30s,太粗),重连由 ping 自然触发。
//
// 两个必须处理的真问题:
// 1. 首帧竞态 —— 页面一加载内容脚本就发 `content`,但那时 WS 还在 CONNECTING,直接发会
//    静默丢弃。所以加一个出站队列,连上后 flush。
// 2. 截图只能在这里做 —— captureVisibleTab 属于扩展 API,内容脚本调不到。内容脚本发来
//    {type:'shot'},SW 抓当前可见区域,把 dataURL 作为 event 帧上行。

const NODE = 'ws://127.0.0.1:23890/';
const BOOT = Date.now();

let ws = null;
let session = null;
let pending = []; // 出站队列:WS 未 OPEN 时先攒着

// 与 content.js 同一张表。SW 是最后一道门:即使内容脚本被注入到了 local 页面,
// 抓屏也不会发生。
function isLocalUrl(url) {
  try {
    const u = new URL(url);
    const h = u.hostname;
    if (u.protocol === 'file:' || u.protocol === 'chrome:' || u.protocol === 'about:') return true;
    return /^(localhost|0\.0\.0\.0|\[::1\])$/.test(h) || h.endsWith('.local') ||
      /^127\./.test(h) || /^10\./.test(h) || /^192\.168\./.test(h) ||
      /^172\.(1[6-9]|2\d|3[01])\./.test(h);
  } catch (e) { return true; }
}

async function getSession() {
  if (session) return session;
  const got = await chrome.storage.local.get('giw_session');
  session = got.giw_session || 's' + Math.random().toString(36).slice(2, 10);
  if (!got.giw_session) await chrome.storage.local.set({ giw_session: session });
  return session;
}

function sendOrQueue(msg) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(msg));
  } else {
    pending.push(msg);
    if (pending.length > 200) pending.shift(); // 溢出丢最早,防旧帧堆死
  }
}

function ensureConnected() {
  if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) return;
  getSession().then((s) => {
    ws = new WebSocket(NODE);
    ws.onopen = () => {
      ws.send(JSON.stringify({ type: 'hello', session: s, boot: BOOT, ua: navigator.userAgent }));
      while (pending.length) ws.send(JSON.stringify(pending.shift()));
    };
    ws.onmessage = (ev) => {
      let msg = {};
      try { msg = JSON.parse(ev.data); } catch (e) { return; }
      // 下行 cmd / say 都带 tab,路由到对应内容脚本
      if ((msg.type === 'cmd' || msg.type === 'say') && msg.tab != null) {
        chrome.tabs.sendMessage(msg.tab, msg, () => void chrome.runtime.lastError);
      }
    };
    ws.onclose = () => { ws = null; };
    ws.onerror = () => { ws = null; };
  });
}

chrome.runtime.onStartup.addListener(ensureConnected);
chrome.runtime.onInstalled.addListener(ensureConnected);

// 内容脚本 → SW:盖 sender.tab.id 戳后转给 node。SW 睡着时这条消息会把它唤醒。
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  ensureConnected();
  const tab = sender.tab;

  if (msg.type === 'ping') {
    sendResponse({ ok: true }); // 只为续命,不上行
    return false;
  }

  if (msg.type === 'shot') {
    if (!tab || tab.id == null || isLocalUrl(tab.url || '')) {
      sendResponse({ ok: false });
      return false;
    }
    chrome.tabs.captureVisibleTab(tab.windowId, { format: 'png' }, (dataUrl) => {
      if (chrome.runtime.lastError || !dataUrl) {
        // 失败显式上行,别静默吞 —— 审计页能看到为什么截图没成。
        sendOrQueue({ type: 'event', tab: tab.id, kind: 'error',
          detail: 'screenshot failed: ' + (chrome.runtime.lastError?.message || 'empty') });
        sendResponse({ ok: false });
        return;
      }
      sendOrQueue({ type: 'event', tab: tab.id, kind: 'screenshot', data: dataUrl });
      sendResponse({ ok: true });
    });
    return true; // async sendResponse
  }

  if (tab && tab.id != null) msg.tab = tab.id;
  sendOrQueue(msg);
  sendResponse({ ok: true });
  return false;
});
