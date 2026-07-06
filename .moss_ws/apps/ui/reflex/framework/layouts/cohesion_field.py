"""凝聚场（Cohesion Field）— 内容从空间边缘凝聚成形。

不是「填入槽位」，是「物质降临」：
- 暗空间粒子场持续呼吸
- 每段内容到达时，从模糊到清晰、从边缘到中心凝聚
- 已有内容感知新内容到来，微微震颤
- 所有内容类型共享同一个凝聚空间

字段兼容 HeroLayout / CourseLayout / StageLayout 的所有历史命名，
Ghost 无论对着哪个 layout 写的脚本，命令都能照常工作。
"""

import reflex as rx
from PIL import Image

from framework.helpers.mixin import NameMixin

# ── Canvas 粒子场 JS ──
_PARTICLE_SCRIPT = """
(function S(){
  var c=document.getElementById('cohesion-particles');
  if(!c){setTimeout(S,80);return;}
  var ctx=c.getContext('2d'),W,H;
  function R(){W=c.width=window.innerWidth;H=c.height=window.innerHeight;}
  window.addEventListener('resize',R);R();
  function P(){this.reset();this.x=Math.random()*W;this.y=Math.random()*H;this.l=Math.random()*6.28;}
  P.prototype.reset=function(){
    var a=Math.random()*6.28,d=60+Math.random()*Math.max(W,H)*0.8;
    this.x=W/2+Math.cos(a)*d;this.y=H/2+Math.sin(a*0.6)*d*0.6;
    this.l=Math.random()*6.28;this.sp=0.0002+Math.random()*0.001;
    this.sz=0.5+Math.random()*2;this.op=0.08+Math.random()*0.4;
    this.or=100+Math.random()*Math.max(W,H)*0.6;
  };
  P.prototype.up=function(){
    this.l+=this.sp;var cx=W/2,cy=H/2;
    var r=this.or*(0.82+0.18*Math.sin(this.l*4));
    var a=this.l*2.1+this.or*0.0008;
    var tx=cx+Math.cos(a)*r,ty=cy+Math.sin(a*0.65)*r*0.5;
    this.x+=(tx-this.x)*0.003;this.y+=(ty-this.y)*0.003;
    var dx=this.x-cx,dy=this.y-cy;
    if(Math.sqrt(dx*dx+dy*dy)<25)this.reset();
    if(this.x<-100||this.x>W+100||this.y<-100||this.y>H+100)this.reset();
  };
  P.prototype.draw=function(t){
    var tw=0.6+0.4*Math.sin(t*0.0003+this.l*7);
    ctx.beginPath();ctx.arc(this.x,this.y,this.sz,0,6.28);
    ctx.fillStyle='rgba(130,130,210,'+(this.op*tw)+')';ctx.fill();
  };
  var ps=[];for(var i=0;i<180;i++)ps.push(new P());
  function A(t){
    ctx.clearRect(0,0,W,H);
    var g=ctx.createRadialGradient(W/2,H/2,0,W/2,H/2,Math.max(W,H)*0.7);
    g.addColorStop(0,'rgba(20,20,60,0)');g.addColorStop(1,'rgba(2,2,10,0.6)');
    ctx.fillStyle=g;ctx.fillRect(0,0,W,H);
    for(var i=0;i<ps.length;i++){ps[i].up();ps[i].draw(t);}
    requestAnimationFrame(A);
  }
  requestAnimationFrame(A);
})();
"""

# ── 凝聚动画 + 卡片样式 ──
_COHESION_CSS = """
@keyframes coalesce {
  0% {
    opacity: 0;
    filter: blur(16px);
    transform: translateY(50px) scale(0.82);
  }
  45% {
    opacity: 0.55;
    filter: blur(6px);
  }
  100% {
    opacity: 1;
    filter: blur(0);
    transform: translateY(0) scale(1);
  }
}

.coal-block {
  animation: coalesce 0.95s cubic-bezier(0.22, 0.61, 0.36, 1) both;
  transition: transform 0.7s cubic-bezier(0.22, 0.61, 0.36, 1),
              filter 0.7s ease-out,
              opacity 0.5s ease-out;
}
"""


class CohesionField(rx.ComponentState, NameMixin):
    """凝聚场 — 通用流式内容空间。

    字段覆盖 HeroLayout / CourseLayout / StageLayout 所有历史命名，
    确保 Ghost 对着任何旧 layout 写的脚本命令都能照常工作。
    """

    # 标题 — 兼容 HeroLayout.title / CourseLayout.title / StageLayout.title
    title: str = ""

    # 副标题 — CourseLayout 用 sub_title, StageLayout 用 subtitle，两者共存
    sub_title: str = ""
    subtitle: str = ""

    # 正文 — CourseLayout 用 main_text, StageLayout 用 body
    main_text: str = ""
    body: str = ""

    # 图片 — CourseLayout 用单数 image, StageLayout 用复数 images
    image: list[Image.Image] = []
    images: list[Image.Image] = []

    # StageLayout 的结构化字段 — 暂未接入渲染，保留定义使命令不报错
    status_bars: list[dict] = []
    cards: list[dict] = []

    # CourseLayout 的结构化字段 — 暂未接入渲染，保留定义使命令不报错
    annotations: list[str] = []
    appreciation: str = ""

    @classmethod
    def name(cls) -> str:
        return "cohesion_field"

    @classmethod
    def get_component(cls, **props) -> rx.Component:
        return rx.box(
            # ── 粒子场画布 ──
            rx.el.canvas(
                id="cohesion-particles",
                position="fixed",
                top="0",
                left="0",
                width="100%",
                height="100%",
                z_index="0",
                pointer_events="none",
            ),
            # ── 内容凝聚区 ──
            rx.center(
                rx.vstack(
                    # ── 标题块 ──
                    rx.cond(
                        cls.title != "",
                        rx.box(
                            rx.heading(
                                cls.title,
                                size="9",
                                weight="bold",
                                color="#f2f2ff",
                                text_align="center",
                                letter_spacing="-0.02em",
                                line_height="1.2",
                            ),
                            class_name="coal-block",
                        ),
                    ),
                    # ── 副标题块 (sub_title, CourseLayout 兼容) ──
                    rx.cond(
                        cls.sub_title != "",
                        rx.box(
                            rx.text(
                                cls.sub_title,
                                size="4",
                                color="#7a7ab8",
                                text_align="center",
                                letter_spacing="0.06em",
                                weight="medium",
                            ),
                            class_name="coal-block",
                        ),
                    ),
                    # ── 副标题块 (subtitle, StageLayout 兼容) ──
                    rx.cond(
                        (cls.subtitle != "") & (cls.sub_title == ""),
                        rx.box(
                            rx.text(
                                cls.subtitle,
                                size="4",
                                color="#7a7ab8",
                                text_align="center",
                                letter_spacing="0.06em",
                                weight="medium",
                            ),
                            class_name="coal-block",
                        ),
                    ),
                    # ── 正文块 (main_text, CourseLayout 兼容) ──
                    rx.cond(
                        cls.main_text != "",
                        rx.box(
                            rx.text(
                                cls.main_text,
                                size="3",
                                color="#b8b8e0",
                                text_align="center",
                                line_height="1.9",
                                max_width="640px",
                            ),
                            class_name="coal-block",
                        ),
                    ),
                    # ── 正文块 (body, StageLayout 兼容) ──
                    rx.cond(
                        (cls.body != "") & (cls.main_text == ""),
                        rx.box(
                            rx.text(
                                cls.body,
                                size="3",
                                color="#b8b8e0",
                                text_align="center",
                                line_height="1.9",
                                max_width="640px",
                            ),
                            class_name="coal-block",
                        ),
                    ),
                    # ── 图片块 (image, CourseLayout 兼容) ──
                    rx.foreach(
                        cls.image,
                        lambda img: rx.box(
                            rx.image(
                                src=img,
                                border_radius="12px",
                                max_width="540px",
                                width="100%",
                                height="auto",
                            ),
                            class_name="coal-block",
                        ),
                    ),
                    # ── 图片块 (images, StageLayout 兼容) ──
                    rx.foreach(
                        cls.images,
                        lambda img: rx.box(
                            rx.image(
                                src=img,
                                border_radius="12px",
                                max_width="540px",
                                width="100%",
                                height="auto",
                            ),
                            class_name="coal-block",
                        ),
                    ),
                    spacing="7",
                    align="center",
                    padding="40px",
                ),
                width="100%",
                height="100vh",
                z_index="1",
            ),
            # ── 脚本与样式 ──
            rx.script(_PARTICLE_SCRIPT),
            rx.html(f"<style>{_COHESION_CSS}</style>"),
            width="100%",
            height="100vh",
            overflow="hidden",
            background="#030310",
            **props,
        )
