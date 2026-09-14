/* Live2D Avatar 页面 — 驱动的消费者.

   协议 (见 bridge.py): 页面只收不发. 事件帧:
     {t:"hello",       name, model, canvas, backdrop, params, view}   连接即全量快照
     {t:"params",      v:{paramId:value}}                              合并后的参数帧 (30fps)
     {t:"motion",      g:group, i:index}                               播一个动作
     {t:"expression",  n:name}                                         切一个表情
     {t:"backdrop",    url}                                            换背板
     {t:"view",        scale, x, y}                                    缩放/平移
     {t:"reset"}                                                      参数回默认 + 清表情

   渲染用 pixi-live2d-display (Cubism 4 的薄封装): 它负责 moc3 加载、WebGL 渲染、
   motion/expression/physics/pose. 参数直接写进框架的 CubismModel, 框架会按模型的
   真实上下界夹取 —— 所以 Python 侧传原值即可, 不需要知道 min/max.
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
    // 记下模型默认参数, 供 reset 回退.
    model.internalModel.coreModel.saveParameters();
    ready = true;
    for (const [id, v] of Object.entries(msg.params || {})) setParam(id, v);
    if (msg.backdrop) setBackdrop(msg.backdrop);
    attachInteraction(model);
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
      case "motion":
        if (ready) model.motion(msg.g, msg.i);
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
