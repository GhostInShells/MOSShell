"""DanmakuLayout — 弹幕图文流布局。

文字以弹幕形式从右飘向左，CSS animation 驱动。
背景图 Reflex 原生渲染（和 video_player 同款 list[Image.Image] 模式）。
Canvas 粒子场持续呼吸。
"""

from PIL import Image

import reflex as rx

from framework.events import VideoLocator
from framework.helpers.mixin import NameMixin

# ═══════════════════════════════════════════════════════════════════════════════
# 粒子场脚本（mirror 同款 IIFE）
# ═══════════════════════════════════════════════════════════════════════════════

_PARTICLE_SCRIPT = """
(function S(){
  var c=document.getElementById('danmaku-particles');
  if(!c){setTimeout(S,80);return;}
  var ctx=c.getContext('2d'),W,H;
  function R(){W=c.width=window.innerWidth;H=c.height=window.innerHeight;}
  window.addEventListener('resize',R);R();
  function P(){this.reset();this.x=Math.random()*W;this.y=Math.random()*H;}
  P.prototype.reset=function(){
    this.x=Math.random()*W;this.y=Math.random()*H;
    this.l=Math.random()*6.28;this.sp=0.0002+Math.random()*0.0008;
    this.sz=0.3+Math.random()*1.7;this.op=0.02+Math.random()*0.05;
    this.phase=Math.random()*6.28;
    this.vx=(Math.random()-0.5)*0.2;this.vy=(Math.random()-0.5)*0.2;
  };
  P.prototype.up=function(){
    this.x+=this.vx;this.y+=this.vy;
    if(this.x<-30)this.x=W+30;if(this.x>W+30)this.x=-30;
    if(this.y<-30)this.y=H+30;if(this.y>H+30)this.y=-30;
  };
  P.prototype.draw=function(t){
    var tw=0.5+0.5*Math.sin(t*0.0003+this.phase);
    ctx.beginPath();ctx.arc(this.x,this.y,this.sz,0,6.28);
    ctx.fillStyle='rgba(130,130,210,'+(this.op*tw)+')';ctx.fill();
  };
  var ps=[];for(var i=0;i<30;i++)ps.push(new P());
  function A(t){
    ctx.clearRect(0,0,W,H);
    var g=ctx.createRadialGradient(W/2,H/2,0,W/2,H/2,Math.max(W,H)*0.7);
    g.addColorStop(0,'rgba(10,10,30,0)');g.addColorStop(1,'rgba(3,3,14,0.7)');
    ctx.fillStyle=g;ctx.fillRect(0,0,W,H);
    for(var i=0;i<ps.length;i++){ps[i].up();ps[i].draw(t);}
    requestAnimationFrame(A);
  }
  requestAnimationFrame(A);
})();
"""

# ═══════════════════════════════════════════════════════════════════════════════
# 信号桥接脚本（speed / clear_all 从隐藏 DOM 读取，push 到 CSS 变量）
# ═══════════════════════════════════════════════════════════════════════════════

_SIGNAL_SCRIPT = """
(function(){
  var store=document.getElementById('danmaku-signal-store');
  if(!store){setTimeout(arguments.callee,80);return;}

  var lastClear='',lastSpeed='1.0';
  var layer=document.getElementById('danmaku-layer');

  function sync(){
    var clearEl=store.querySelector('[data-danmaku-clear]');
    var speedEl=store.querySelector('[data-danmaku-speed]');

    // clear_all — 加速现有弹幕
    var cv=clearEl?clearEl.getAttribute('data-danmaku-clear')||'':'';
    if(cv!==''&&cv!==lastClear){
      lastClear=cv;
      if(layer)layer.style.setProperty('--danmaku-speed','6');
      setTimeout(function(){if(layer)layer.style.setProperty('--danmaku-speed',lastSpeed);},300);
    }

    // speed
    var sv=speedEl?speedEl.getAttribute('data-danmaku-speed')||'1.0':'1.0';
    if(sv!==lastSpeed){
      lastSpeed=sv;
      if(layer&&lastClear==='')layer.style.setProperty('--danmaku-speed',sv);
    }
  }

  sync();
  new MutationObserver(sync).observe(store,{childList:true,subtree:true,attributes:true});
})();
"""

# ═══════════════════════════════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════════════════════════════

_DANMAKU_CSS = """
#danmaku-particles {
  position: fixed; top: 0; left: 0; width: 100%; height: 100%;
  z-index: 0; pointer-events: none;
}
#danmaku-layer {
  position: fixed; top: 0; left: 0; width: 100%; height: 100%;
  pointer-events: none; overflow: hidden; z-index: 3;
}

@keyframes danmaku-drift {
  0%   { transform: translateX(100vw); opacity: 0; }
  4%   { opacity: 1; }
  92%  { opacity: 1; }
  100% { transform: translateX(-100%); opacity: 0; }
}

.danmaku-item {
  position: absolute;
  white-space: nowrap;
  padding: 6px 12px;
  border-radius: 8px;
  font-family: "PingFang SC", "Noto Sans SC", sans-serif;
  pointer-events: none;
}

.danmaku-normal {
  animation: danmaku-drift calc(8s / var(--danmaku-speed, 1)) linear forwards;
  background: rgba(8,8,30,0.70);
  color: #e0e0f0;
  font-size: 18px;
}

.danmaku-emphasis {
  animation: danmaku-drift calc(11s / var(--danmaku-speed, 1)) linear forwards;
  background: rgba(8,8,30,0.82);
  color: #a0c4ff;
  font-size: 22px;
}

.danmaku-system {
  animation: danmaku-drift calc(6s / var(--danmaku-speed, 1)) linear forwards;
  background: rgba(20,8,30,0.75);
  color: #c0a0ff;
  font-size: 16px;
}
"""


# ═══════════════════════════════════════════════════════════════════════════════
# Layout Class
# ═══════════════════════════════════════════════════════════════════════════════

LANE_H = 44
LANES = 6


class DanmakuLayout(rx.ComponentState, NameMixin):
    """弹幕图文流布局。

    文字弹幕：Reflex rx.foreach 渲染 DOM 元素 → CSS @keyframes 漂移动画。
    背景图：list[Image.Image]，和 video_player 同款模式，Reflex 原生渲染。
    视频：list[VideoLocator]，全屏自动播放。
    粒子：Canvas IIFE（mirror 同款）。
    """

    danmaku_text: list[str] = []
    danmaku_emphasis: list[str] = []
    danmaku_system: list[str] = []
    wall_images: list[Image.Image] = []
    videos: list[VideoLocator] = []
    danmaku_speed: str = "1.0"
    danmaku_clear_all: str = ""

    @classmethod
    def name(cls) -> str:
        return "danmaku"

    @classmethod
    def get_component(cls, **props) -> rx.Component:
        return rx.box(
            # ── Canvas 环境粒子层 ──
            rx.el.canvas(
                id="danmaku-particles",
                position="fixed",
                top="0",
                left="0",
                width="100%",
                height="100%",
                z_index="0",
                pointer_events="none",
            ),
            # ── 视频背景层（Reflex 渲染，z-1）──
            rx.box(
                rx.foreach(
                    cls.videos,
                    lambda v: rx.box(
                        rx.video(
                            src=v,
                            playing=True,
                            controls=False,
                            muted=True,
                            loop=True,
                            width="100%",
                            height="100%",
                        ),
                        width="100%",
                        height="100%",
                    ),
                ),
                position="fixed",
                top="0",
                left="0",
                width="100%",
                height="100%",
                z_index="1",
                pointer_events="none",
            ),
            # ── 背景图层（Reflex 原生渲染，z-2）──
            rx.box(
                rx.foreach(
                    cls.wall_images,
                    lambda img: rx.image(
                        src=img,
                        object_fit="contain",
                        width="100%",
                        height="100%",
                    ),
                ),
                position="fixed",
                top="0",
                left="0",
                width="100%",
                height="100%",
                z_index="2",
                pointer_events="none",
            ),
            # ── 弹幕文字层（Reflex 渲染 + CSS animation 驱动漂移）──
            rx.box(
                rx.foreach(
                    cls.danmaku_text,
                    lambda text, i: rx.box(
                        text,
                        class_name="danmaku-item danmaku-normal",
                        style={"top": f"{(i % LANES) * LANE_H + 8}px"},
                    ),
                ),
                rx.foreach(
                    cls.danmaku_emphasis,
                    lambda text, i: rx.box(
                        text,
                        class_name="danmaku-item danmaku-emphasis",
                        style={"top": f"{(i % LANES) * LANE_H + 8}px"},
                    ),
                ),
                rx.foreach(
                    cls.danmaku_system,
                    lambda text, i: rx.box(
                        text,
                        class_name="danmaku-item danmaku-system",
                        style={"top": f"{(i % LANES) * LANE_H + 8}px"},
                    ),
                ),
                id="danmaku-layer",
            ),
            # ── 隐藏信号 store（speed + clear_all，JS 读取 → CSS 变量）──
            rx.box(
                rx.el.div(data_danmaku_speed=cls.danmaku_speed),
                rx.el.div(data_danmaku_clear=cls.danmaku_clear_all),
                id="danmaku-signal-store",
                display="none",
            ),
            # ── 底部状态线 ──
            rx.box(
                rx.center(
                    rx.text(
                        "MOSHI",
                        color="rgba(200,210,240,0.04)",
                        font_size="48px",
                        font_weight="200",
                    ),
                    width="100%",
                    height="100%",
                ),
                position="fixed",
                bottom="0",
                left="0",
                width="100%",
                height="28px",
                background="rgba(6,6,18,0.8)",
                border_top="0.5px solid rgba(40,40,70,0.3)",
                z_index="4",
            ),
            # ── 脚本 ──
            rx.script(_PARTICLE_SCRIPT),
            rx.script(_SIGNAL_SCRIPT),
            # ── CSS ──
            rx.html(f"<style>{_DANMAKU_CSS}</style>"),
            width="100%",
            height="100vh",
            overflow="hidden",
            background="#050510",
            **props,
        )
