// probe #2: 后台 service worker 持有 WebSocket 的存活实验。
//
// 要回答的:SW 会不会在 ~30s 空闲后被回收?靠 20s 一条消息能不能续命?
// 被回收后重连多久?重连时能不能把断线期间的日志补交给 node?
//
// BOOT 是模块级常量 —— SW 每次重启都会重新求值,所以 node 侧看到两个不同的
// boot 时间 = SW 确实重启过。这是判断"是否被回收"最直接的证据。

const NODE = 'ws://127.0.0.1:23881/';
const BOOT = new Date().toISOString();

const buf = [];            // 待补交给 node 的日志;SW 重启会清空,本身就是证据
let ws = null;
let session = null;
let reconnectTimer = null;
let attempts = 0;
let connects = 0;

function note(line) {
  const entry = `${new Date().toTimeString().slice(0, 8)} ${line}`;
  console.log('[probe]', entry);
  buf.push(entry);
  if (buf.length > 300) buf.splice(0, buf.length - 300);
}

async function getSession() {
  if (session) return session;
  const got = await chrome.storage.local.get('probe_session');
  session = got.probe_session || 's' + Math.random().toString(36).slice(2, 8);
  if (!got.probe_session) await chrome.storage.local.set({ probe_session: session });
  return session;
}

function flush() {
  if (!ws || ws.readyState !== WebSocket.OPEN || !buf.length) return;
  const lines = buf.splice(0, buf.length);
  ws.send(JSON.stringify({ type: 'log', lines }));
}

function scheduleReconnect() {
  attempts += 1;
  const delay = Math.min(30000, 1000 * Math.pow(2, Math.min(attempts, 5)));
  note(`reconnect in ${delay}ms (attempt ${attempts})`);
  reconnectTimer = setTimeout(() => connect('retry'), delay);
}

function connect(why) {
  if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) return;
  if (reconnectTimer) { clearTimeout(reconnectTimer); reconnectTimer = null; }
  connects += 1;
  const n = connects;
  getSession().then((s) => {
    note(`connect(why=${why}, n=${n}) session=${s}`);
    ws = new WebSocket(NODE);
    ws.onopen = () => {
      attempts = 0;
      note(`WS open (n=${n})`);
      ws.send(JSON.stringify({ type: 'hello', session: s, boot: BOOT, ua: navigator.userAgent }));
      flush();
    };
    ws.onmessage = (ev) => {
      let msg = {};
      try { msg = JSON.parse(ev.data); } catch (e) { return; }
      if (msg.type === 'ping') {
        ws.send(JSON.stringify({ type: 'pong', n: msg.n }));
        return;
      }
      note(`recv ${msg.type}`);
    };
    ws.onclose = (ev) => {
      note(`WS close code=${ev.code} reason=${ev.reason || '-'} clean=${ev.wasClean}`);
      ws = null;
      scheduleReconnect();
    };
    ws.onerror = () => note('WS error');
  });
}

note(`sw boot ${BOOT}`);

chrome.runtime.onStartup.addListener(() => { note('onStartup'); connect('startup'); });
chrome.runtime.onInstalled.addListener(() => { note('onInstalled'); connect('installed'); });

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  const tab = sender.tab ? sender.tab.id : '?';
  note(`msg tab=${tab} type=${msg.type}`);
  connect('message');           // 懒连接:内容脚本的消息会把睡着的 SW 唤醒
  if (msg.type === 'probe') {
    note(`tab ${tab} checks ${JSON.stringify(msg.checks)}`);
  } else if (msg.type === 'navigated') {
    note(`tab ${tab} navigated -> ${msg.bvid} ${msg.url.slice(0, 90)}`);
  }
  sendResponse({ ok: true, session });
  return true;
});
