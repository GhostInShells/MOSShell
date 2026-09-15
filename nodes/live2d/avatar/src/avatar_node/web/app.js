/* Live2D Avatar 页面 — 纯执行器.

   驱动持有所有时间轨迹与待机仲裁; 页面只服从出站帧, 不做任何策略. 动作在模型元数据里
   都是 Loop:True (实测 hiyori 的 10 个 motion), 所以"结束"由驱动计时后发 stop/clear 决定,
   页面绝不自己判断.

   协议 (见 bridge.py), 出站帧:
     {t:"hello",        name, model, canvas, backdrop, params, view, idle, idle_active, parts}
     {t:"params",       v:{paramId:value}}        合并后的参数帧 (30fps)
     {t:"idle",         g, i}                     播待机循环 (动作 Loop:True, 自然循环)
     {t:"motion",       g, i}                     播一个前景动作
     {t:"stop_motion"}                            停所有动作, 参数不动
     {t:"clear_motion"}                           停动作 + 参数回默认 (驱动随后重发状态)
     {t:"expression",   n}                        切表情
     {t:"backdrop",     url}                      换背板
     {t:"view",         scale, x, y}              缩放/平移
     {t:"reset"}                                  停动作 + 参数回默认 + 清表情
*/


(() => {
  const { Live2DModel } = PIXI.live2d;

  const app = new PIXI.Application({
    resizeTo: window,
    backgroundAlpha: 0, // 透明画布, 背板走 CSS 背景
    antialias: true,
    autoDensity: true,
    resolution: window.devicePixelRatio || 1,
  });
  document.getElementById("stage").appendChild(app.view);

  let model = null;
  let ready = false;
  const statusEl = document.getElementById("status");

  // fit = 视口拟合出的基准缩放 + 中心点; view = 驱动下发的缩放倍率 + 归一化偏移.
  let fit = { scale: 1, cx: 0, cy: 0 };
  let view = { scale: 1, x: 0, y: 0 };

  let ws = null;

  function send(msg) {
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(msg));
  }

  function status(text) {
    statusEl.textContent = text;
  }

  function setBackdrop(url) {
    document.body.style.backgroundImage = url ? `url("${url}")` : "none";
  }

  function computeFit(m) {
    fit.scale = Math.min(
      (window.innerWidth * 0.9) / m.width,
      (window.innerHeight * 0.95) / m.height,
    );
    fit.cx = window.innerWidth / 2;
    fit.cy = window.innerHeight / 2;
  }

  function applyView() {
    if (!model) return;
    model.anchor.set(0.5, 0.5);
    model.scale.set(fit.scale * view.scale);
    model.x = fit.cx + view.x * (window.innerWidth / 2);
    model.y = fit.cy + view.y * (window.innerHeight / 2);
  }

  function setParam(id, value) {
    if (!ready) return;
    try {
      model.internalModel.coreModel.setParameterValueById(id, value);
    } catch (e) {
      console.warn("setParam", id, e);
    }
  }

  function stopAllMotions() {
    if (!ready) return;
    model.internalModel.motionManager.stopAllMotions();
  }

  function playMotion(g, i) {
    if (!ready) return;
    stopAllMotions();
    model.motion(g, i); // 动作 Loop:True, 由驱动 stop 结束; 不 await
  }

  function applyParts(parts) {
    if (!parts) return;
    // SDK 原生部件 idle: blink(眼) / breath(呼吸). 关闭即置空, 更新处是可选链, 安全.
    if (parts.blink === false) model.internalModel.eyeBlink = null;
    if (parts.breath === false) model.internalModel.breath = null;
  }

  async function onHello(msg) {
    // 重连 / node 重启都会再触发 hello: 先摘掉并销毁旧模型, 否则会叠成重影.
    if (model) {
      app.stage.removeChild(model);
      try {
        model.destroy();
      } catch (e) {
        console.warn("destroy old model", e);
      }
      model = null;
      ready = false;
    }
    model = await Live2DModel.from(msg.model, { autoInteract: false });
    app.stage.addChild(model);
    computeFit(model);
    if (msg.view) view = { scale: msg.view.scale ?? 1, x: msg.view.x ?? 0, y: msg.view.y ?? 0 };
    applyView();
    // 记下模型默认参数, 供 reset / clear_motion 回退.
    model.internalModel.coreModel.saveParameters();
    ready = true;
    applyParts(msg.parts);
    for (const [id, v] of Object.entries(msg.params || {})) setParam(id, v);
    if (msg.backdrop) setBackdrop(msg.backdrop);
    attachInteraction(model);
    if (msg.idle_active && msg.idle) playMotion(msg.idle.g, msg.idle.i);
    status(`模型已加载: ${msg.name}`);
  }

  // 人类交互 → 感知面: 点击/拖拽回传驱动, 驱动记录进 notice (模型可见), 点击还触发 Tap 动作.
  function attachInteraction(m) {
    m.interactive = true;
    m.buttonMode = true;
    m.hitArea = new PIXI.Rectangle(-m.width / 2, -m.height / 2, m.width, m.height);
    m.on("pointertap", () => send({ t: "interact", act: "tap" }));
    let down = null;
    m.on("pointerdown", (e) => {
      down = { x: e.data.global.x, y: e.data.global.y };
    });
    const endDrag = (e) => {
      if (!down) return;
      const dx = e.data.global.x - down.x;
      const dy = e.data.global.y - down.y;
      down = null;
      if (Math.abs(dx) + Math.abs(dy) > 12) {
        send({ t: "interact", act: "drag", dx: Math.round(dx), dy: Math.round(dy) });
      }
    };
    m.on("pointerup", endDrag);
    m.on("pointerupoutside", endDrag);
  }

  function resetModel() {
    if (!ready) return;
    stopAllMotions();
    model.internalModel.coreModel.loadParameters();
    model.expression();
  }

  function handle(msg) {
    switch (msg.t) {
      case "hello":
        onHello(msg);
        break;
      case "params":
        for (const [id, v] of Object.entries(msg.v || {})) setParam(id, v);
        break;
      case "idle":
        playMotion(msg.g, msg.i);
        break;
      case "motion":
        playMotion(msg.g, msg.i);
        break;
      case "stop_motion":
        stopAllMotions();
        break;
      case "clear_motion":
        stopAllMotions();
        model.internalModel.coreModel.loadParameters();
        break;
      case "expression":
        if (ready) model.expression(msg.n);
        break;
      case "backdrop":
        setBackdrop(msg.url);
        break;
      case "view":
        view = { scale: msg.scale ?? 1, x: msg.x ?? 0, y: msg.y ?? 0 };
        applyView();
        break;
      case "reset":
        resetModel();
        break;
    }
  }

  function connect() {
    const proto = location.protocol === "https:" ? "wss://" : "ws://";
    ws = new WebSocket(`${proto}${location.host}/ws`);
    ws.onopen = () => status("已连接，等待模型…");
    ws.onmessage = (e) => {
      try {
        handle(JSON.parse(e.data));
      } catch (err) {
        console.warn("bad frame", err);
      }
    };
    ws.onclose = () => {
      status("连接断开，重连中…");
      setTimeout(connect, 1000);
    };
    ws.onerror = () => status("连接出错");
  }

  window.addEventListener("resize", () => {
    if (model) {
      computeFit(model);
      applyView();
    }
  });

  connect();
})();
