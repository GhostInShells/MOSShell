// ghost-in-bilibili service worker.
//
// 一条 WS 复用一个浏览器 session(一个扩展一个 SW)。内容脚本每 ~2s 的心跳消息
// 既续命(每次消息重置 SW 的 ~30s 空闲计时)又唤醒(睡了会被消息叫醒),所以这里
// 不靠 chrome.alarms,重连由内容脚本的下一条心跳自然触发。

const NODE = 'ws://127.0.0.1:23880/';
const BOOT = Date.now();

let ws = null;
let session = null;

async function getSession() {
  if (session) return session;
  const got = await chrome.storage.local.get('gib_session');
  session = got.gib_session || 's' + Math.random().toString(36).slice(2, 10);
  if (!got.gib_session) await chrome.storage.local.set({ gib_session: session });
  return session;
}

function ensureConnected() {
  if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) return;
  getSession().then((s) => {
    ws = new WebSocket(NODE);
    ws.onopen = () => {
      ws.send(JSON.stringify({ type: 'hello', session: s, boot: BOOT, ua: navigator.userAgent }));
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
  if (sender.tab && sender.tab.id != null) msg.tab = sender.tab.id;
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(msg));
  }
  sendResponse({ ok: true });
  return false;
});
