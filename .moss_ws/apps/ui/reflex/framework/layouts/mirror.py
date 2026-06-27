"""镜像对（Mirror）— 左右两个暗空间，内容逐行凝聚浮现。

用于传统 OS vs AIOS 等对比演示。两侧内容成对出现，
右侧带微延迟形成镜面回响。粒子场在两侧之间缓慢流动。
"""

import reflex as rx

from framework.helpers.mixin import NameMixin

_PARTICLE_SCRIPT = """
(function S(){
  var c=document.getElementById('mirror-particles');
  if(!c){setTimeout(S,80);return;}
  var ctx=c.getContext('2d'),W,H;
  function R(){W=c.width=window.innerWidth;H=c.height=window.innerHeight;}
  window.addEventListener('resize',R);R();
  function P(){this.reset();}
  P.prototype.reset=function(){
    this.x=Math.random()*W;
    this.y=Math.random()*H;
    this.sz=0.2+Math.random()*1.3;
    this.op=0.03+Math.random()*0.15;
    this.sp=0.0001+Math.random()*0.0005;
    this.l=Math.random()*6.28;
    this.ox=W/2+(Math.random()-0.5)*W*0.8;
    this.oy=H/2+(Math.random()-0.5)*H*0.8;
  };
  P.prototype.up=function(){
    this.l+=this.sp;
    var cx=W/2,cy=H/2;
    var r=Math.max(W,H)*0.35;
    var a=this.l+Math.sin(this.l*2.1)*0.25;
    var tx=cx+Math.cos(a)*r*(0.7+0.3*Math.sin(this.l*1.3));
    var ty=cy+Math.sin(a*0.55)*r*0.45;
    this.x+=(tx-this.x)*0.0015;
    this.y+=(ty-this.y)*0.0015;
    if(this.x<-100||this.x>W+100||this.y<-100||this.y>H+100)this.reset();
  };
  P.prototype.draw=function(t){
    ctx.beginPath();ctx.arc(this.x,this.y,this.sz,0,6.28);
    var tw=0.45+0.55*Math.sin(t*0.00015+this.l*4);
    ctx.fillStyle='rgba(110,110,190,'+(this.op*tw)+')';ctx.fill();
  };
  var ps=[];for(var i=0;i<90;i++)ps.push(new P());
  function A(t){
    ctx.clearRect(0,0,W,H);
    for(var i=0;i<ps.length;i++){ps[i].up();ps[i].draw(t);}
    requestAnimationFrame(A);
  }
  requestAnimationFrame(A);
})();
"""

_MIRROR_CSS = """
@keyframes mirror-coalesce {
  0% {
    opacity: 0;
    filter: blur(10px);
    transform: translateY(24px);
  }
  100% {
    opacity: 1;
    filter: blur(0);
    transform: translateY(0);
  }
}

.mirror-left {
  animation: mirror-coalesce 0.7s cubic-bezier(0.22, 0.61, 0.36, 1) both;
}

.mirror-right {
  animation: mirror-coalesce 0.7s cubic-bezier(0.22, 0.61, 0.36, 1) 0.1s both;
}

@keyframes mirror-divider-pulse {
  0%, 100% { border-color: #1a1a3a; }
  50% { border-color: #2a2a5a; }
}

.mirror-divider {
  animation: mirror-divider-pulse 4s ease-in-out infinite;
}
"""


class MirrorLayout(rx.ComponentState, NameMixin):
    """镜像对布局：左右暗空间，内容逐行凝聚浮现。

    用于对比演示（传统 OS vs AIOS）。左侧偏冷蓝，右侧偏暖金，
    行成对出现，右侧带 0.1s 微延迟形成镜面回响。
    """

    left_header: str = ""
    right_header: str = ""
    rows: list[dict] = []
    stats: str = ""

    @classmethod
    def name(cls) -> str:
        return "mirror"

    @classmethod
    def get_component(cls, **props) -> rx.Component:
        return rx.box(
            # 粒子场
            rx.el.canvas(
                id="mirror-particles",
                position="fixed",
                top="0",
                left="0",
                width="100%",
                height="100%",
                z_index="0",
                pointer_events="none",
            ),
            # 内容
            rx.center(
                rx.vstack(
                    # 表头
                    rx.skeleton(
                        rx.hstack(
                            rx.box(
                                rx.heading(
                                    cls.left_header,
                                    size="7",
                                    weight="bold",
                                    color="#7c7cff",
                                    text_align="center",
                                    letter_spacing="0.04em",
                                ),
                                width="50%",
                                padding_x="20px",
                            ),
                            rx.divider(
                                orientation="vertical",
                                height="48px",
                                border_color="#1a1a3a",
                                class_name="mirror-divider",
                            ),
                            rx.box(
                                rx.heading(
                                    cls.right_header,
                                    size="7",
                                    weight="bold",
                                    color="#ffb86c",
                                    text_align="center",
                                    letter_spacing="0.04em",
                                ),
                                width="50%",
                                padding_x="20px",
                            ),
                            width="100%",
                            max_width="760px",
                            justify="center",
                        ),
                        loading=(cls.left_header == "") & (cls.right_header == ""),
                    ),
                    # 对比行
                    rx.foreach(
                        cls.rows,
                        lambda row, i: rx.hstack(
                            rx.box(
                                rx.text(
                                    row["left"],
                                    size="3",
                                    color="#b8b8e0",
                                    text_align="center",
                                    line_height="1.7",
                                ),
                                width="50%",
                                padding_x="20px",
                                padding_y="13px",
                                class_name="mirror-left",
                            ),
                            rx.divider(
                                orientation="vertical",
                                height="16px",
                                border_color="#14142e",
                            ),
                            rx.box(
                                rx.text(
                                    row["right"],
                                    size="3",
                                    color="#e0d0b0",
                                    text_align="center",
                                    line_height="1.7",
                                ),
                                width="50%",
                                padding_x="20px",
                                padding_y="13px",
                                class_name="mirror-right",
                            ),
                            width="100%",
                            max_width="760px",
                            justify="center",
                        ),
                    ),
                    # 底部统计
                    rx.cond(
                        cls.stats != "",
                        rx.box(
                            rx.text(
                                cls.stats,
                                size="2",
                                color="#5a5a7a",
                                text_align="center",
                                letter_spacing="0.06em",
                            ),
                            class_name="mirror-left",
                            padding_top="20px",
                        ),
                    ),
                    spacing="3",
                    align="center",
                    padding="40px",
                ),
                width="100%",
                height="100vh",
                z_index="1",
            ),
            rx.script(_PARTICLE_SCRIPT),
            rx.html(f"<style>{_MIRROR_CSS}</style>"),
            width="100%",
            height="100vh",
            overflow="hidden",
            background="#030310",
            **props,
        )
