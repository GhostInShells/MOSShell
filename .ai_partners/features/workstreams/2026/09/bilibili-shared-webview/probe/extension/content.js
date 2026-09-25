// probe #2 内容脚本:不为功能服务,只为把三条边界事实测出来。
//
//   1) 页面 CSP 是否也约束隔离世界里的 eval —— 设计文档里那条"与直觉相反"的
//      结论,顺手复核一次。
//   2) 内容脚本能不能直连本地 node(跨域)。若能,SW 就不是唯一网络边界,
//      通讯设计可以简化;若不能,SW 必须当边缘。
//   3) EventSource 在内容脚本里构造得出来吗(SSE 方案的生死判据)。
//
// 三者都只上报给 SW,由 SW 转给 node —— 因为 SW 可能就是被这条消息唤醒的。

(async () => {
  const checks = {};

  try {
    checks.eval = 'ok:' + new Function('return 41 + 1')();
  } catch (e) {
    checks.eval = `${e.name}: ${e.message}`;
  }

  try {
    const r = await fetch('http://127.0.0.1:23881/probe', { cache: 'no-store' });
    checks.fetch = `ok:${r.status}:` + (await r.text()).slice(0, 60);
  } catch (e) {
    checks.fetch = `${e.name}: ${e.message}`;
  }

  await new Promise((resolve) => {
    let es;
    let done = false;
    const finish = () => { if (!done) { done = true; resolve(); } };
    try {
      es = new EventSource('http://127.0.0.1:23881/probe');
      checks.eventsource = 'constructed';
      es.onopen = () => { checks.eventsource += ' onopen'; finish(); };
      es.onerror = () => { checks.eventsource += ' onerror'; finish(); };
      setTimeout(() => { checks.eventsource += ' (timeout)'; try { es.close(); } catch (e) {} finish(); }, 3000);
    } catch (e) {
      checks.eventsource = `${e.name}: ${e.message}`;
      finish();
    }
  });

  checks.bvid = (location.href.match(/BV[0-9A-Za-z]{10}/) || [])[0] || null;
  checks.url = location.href;
  checks.title = document.title;

  const send = (kind, payload) => {
    try {
      chrome.runtime.sendMessage({ type: kind, ...payload }, () => void chrome.runtime.lastError);
    } catch (e) { /* SW 睡着时 sendMessage 会抛 —— 本身就值得记 */ }
  };

  send('probe', { checks });

  // B 站视频页可能是部分导航(SPA)。url 变了但内容脚本没重跑 = 需要自己盯。
  let last = location.href;
  setInterval(() => {
    if (location.href === last) return;
    last = location.href;
    send('navigated', {
      bvid: (last.match(/BV[0-9A-Za-z]{10}/) || [])[0] || null,
      url: last,
    });
  }, 2000);
})();
