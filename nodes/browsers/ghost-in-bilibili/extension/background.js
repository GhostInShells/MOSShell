const NODE = 'http://127.0.0.1:23880';

async function postNode(path, obj) {
  try {
    const r = await fetch(`${NODE}${path}`, {
      method: 'POST', headers: { 'Content-Type': 'text/plain' }, body: JSON.stringify(obj),
    });
    return await r.json();
  } catch (e) { return { ok: false, error: `${e.name}: ${e.message}` }; }
}

async function getNode(path) {
  try { return await (await fetch(`${NODE}${path}`)).json(); }
  catch (e) { return { error: `${e.name}: ${e.message}` }; }
}

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  (async () => {
    switch (msg.type) {
      case 'config':
        return sendResponse(await getNode('/config'));
      case 'toggle':
      case 'input':
      case 'js_result':
        await postNode('/ingest', msg);
        return sendResponse({ ok: true });
      case 'poll':
        return sendResponse(await getNode(`/js/drain?page=${encodeURIComponent(msg.page)}`));
      default:
        return sendResponse({ ok: false, error: 'unknown type' });
    }
  })();
  return true;
});
