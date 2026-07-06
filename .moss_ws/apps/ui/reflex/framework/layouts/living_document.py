"""活体手稿（Living Document）— AI 在写一封无限长的、温暖的手写信。

不是「页面切换」，是「书写进行时」：
- 暖色手工纸质感底布
- 内容逐字落下，如墨水渗入纸纤维
- 页面自动跟笔——摄像机始终追随正在被书写的那个字
- 新内容从上一段文字的右下角自然长出，如植物分枝
- 章节之间无硬切——空行、装饰线、自然的段落间距

字段兼容所有历史 layout 命名，Ghost 命令照常工作。
"""

import reflex as rx
from PIL import Image

from framework.helpers.mixin import NameMixin

# ── 纸纹 script（注入 Canvas 做纸纤维噪点叠加）──
_PAPER_GRAIN_SCRIPT = """
(function S(){
  var c=document.getElementById('paper-grain-canvas');
  if(!c){setTimeout(S,80);return;}
  var ctx=c.getContext('2d'),W,H;
  function R(){W=c.width=window.innerWidth;H=c.height=window.innerHeight;}
  window.addEventListener('resize',function(){R();draw();});R();
  var imgData;
  function draw(){
    imgData=ctx.createImageData(W,H);
    var d=imgData.data;
    for(var i=0;i<d.length;i+=4){
      var g=220+Math.floor(Math.random()*35);
      d[i]=g;d[i+1]=g-4+Math.floor(Math.random()*8);d[i+2]=g-10+Math.floor(Math.random()*10);d[i+3]=6+Math.floor(Math.random()*6);
    }
    ctx.putImageData(imgData,0,0);
  }
  draw();
})();
"""

# ── CSS ──
_LIVING_DOC_CSS = """
/* ── 墨迹浮现 ── */
@keyframes inkReveal {
  0% {
    opacity: 0;
    filter: blur(3px);
    transform: translateY(14px);
  }
  60% {
    opacity: 0.8;
    filter: blur(0.5px);
  }
  100% {
    opacity: 1;
    filter: blur(0);
    transform: translateY(0);
  }
}

@keyframes inkPulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.2; }
}

.ink-block {
  animation: inkReveal 0.8s cubic-bezier(0.22, 0.61, 0.36, 1) both;
  transition: transform 0.6s cubic-bezier(0.22, 0.61, 0.36, 1),
              opacity 0.5s ease-out;
}

.ink-cursor {
  display: inline-block; width: 2px; height: 1.1em;
  background: #5c3a1e; margin-left: 1px; vertical-align: text-bottom;
  animation: inkPulse 0.9s ease-in-out infinite;
  border-radius: 1px;
}

/* ── 手稿排版 ── */
.manuscript {
  max-width: 680px;
  margin: 0 auto;
  padding: 120px 48px 60vh;
  font-family: "Noto Serif SC", "Source Han Serif SC", "Songti SC", Georgia, serif;
}

.manuscript h1 {
  font-size: 42px; font-weight: 700;
  color: #2c1810;
  letter-spacing: 0.04em;
  line-height: 1.25;
  position: relative;
}

.manuscript h1::after {
  content: '';
  display: block;
  width: 60px; height: 2px;
  margin-top: 14px;
  background: linear-gradient(90deg, #8b5e3c 0%, transparent 100%);
  border-radius: 1px;
}

.manuscript h2 {
  font-size: 17px; font-weight: 400; font-style: italic;
  color: #8b6e5c;
  letter-spacing: 0.06em;
}

.manuscript p {
  font-size: 17px; line-height: 2.0;
  color: #3d2b1f;
  text-align: justify;
  text-indent: 2em;
}

.manuscript .img-frame {
  display: inline-block;
  padding: 14px;
  background: #faf7f0;
  border: 1px solid rgba(139,94,60,0.2);
  border-radius: 3px;
  box-shadow:
    0 2px 12px rgba(139,94,60,0.12),
    0 0 1px rgba(0,0,0,0.05);
  transform: rotate(-0.5deg);
  transition: transform 0.3s ease;
}

.manuscript .img-frame:hover {
  transform: rotate(0deg) scale(1.01);
}

.manuscript img {
  display: block;
  max-width: 100%; height: auto;
  border-radius: 1px;
}

.manuscript .caption {
  margin-top: 8px;
  font-size: 13px; color: #8b6e5c; font-style: italic;
  text-align: center;
}
"""


class LivingDocument(rx.ComponentState, NameMixin):
    """活体手稿 — AI 的手写信。

    与凝聚场的暗空间相反：这是温暖的、人本的知识腹地。
    Ghost 边说，手稿边生长。没有布局切换，只有翻页。
    """

    # 向后兼容字段
    title: str = ""
    sub_title: str = ""
    subtitle: str = ""
    main_text: str = ""
    body: str = ""
    image: list[Image.Image] = []
    images: list[Image.Image] = []
    annotations: list[str] = []
    appreciation: str = ""
    status_bars: list[dict] = []
    cards: list[dict] = []

    @classmethod
    def name(cls) -> str:
        return "living_document"

    @classmethod
    def get_component(cls, **props) -> rx.Component:
        return rx.box(
            # ── 纸色底层 ──
            rx.box(
                position="fixed",
                inset="0",
                z_index="0",
                background="linear-gradient(180deg, #f7f2e8 0%, #f0e8d8 30%, #ede4d2 60%, #f2ece0 100%)",
            ),
            # ── 纸纤维噪点 canvas ──
            rx.el.canvas(
                id="paper-grain-canvas",
                position="fixed",
                top="0",
                left="0",
                width="100%",
                height="100%",
                z_index="0",
                pointer_events="none",
                opacity="0.35",
            ),
            # ── 底部阴影 ──
            rx.box(
                position="fixed",
                bottom="0",
                left="0",
                right="0",
                height="80px",
                z_index="2",
                pointer_events="none",
                background="linear-gradient(0deg, rgba(180,160,140,0.3) 0%, transparent 100%)",
            ),
            # ── 手稿滚动区 ──
            rx.box(
                rx.box(
                    # 标题
                    rx.cond(
                        cls.title != "",
                        rx.box(
                            rx.heading(
                                cls.title,
                                font_size="42px",
                                font_weight="700",
                                color="#2c1810",
                                letter_spacing="0.04em",
                                line_height="1.25",
                            ),
                            class_name="ink-block",
                            margin_bottom="8px",
                        ),
                    ),
                    # 副标题 (sub_title)
                    rx.cond(
                        cls.sub_title != "",
                        rx.box(
                            rx.text(
                                cls.sub_title,
                                font_size="17px",
                                font_weight="400",
                                font_style="italic",
                                color="#8b6e5c",
                                letter_spacing="0.06em",
                            ),
                            class_name="ink-block",
                            margin_top="-20px",
                            margin_bottom="24px",
                        ),
                    ),
                    # 副标题 (subtitle)
                    rx.cond(
                        (cls.subtitle != "") & (cls.sub_title == ""),
                        rx.box(
                            rx.text(
                                cls.subtitle,
                                font_size="17px",
                                font_weight="400",
                                font_style="italic",
                                color="#8b6e5c",
                                letter_spacing="0.06em",
                            ),
                            class_name="ink-block",
                            margin_top="-20px",
                            margin_bottom="24px",
                        ),
                    ),
                    # 正文 (main_text)
                    rx.cond(
                        cls.main_text != "",
                        rx.box(
                            rx.text(
                                cls.main_text,
                                font_size="17px",
                                line_height="2.0",
                                color="#3d2b1f",
                                text_align="justify",
                                text_indent="2em",
                            ),
                            class_name="ink-block",
                            margin_bottom="28px",
                        ),
                    ),
                    # 正文 (body)
                    rx.cond(
                        (cls.body != "") & (cls.main_text == ""),
                        rx.box(
                            rx.text(
                                cls.body,
                                font_size="17px",
                                line_height="2.0",
                                color="#3d2b1f",
                                text_align="justify",
                                text_indent="2em",
                            ),
                            class_name="ink-block",
                            margin_bottom="28px",
                        ),
                    ),
                    # 图片 (image)
                    rx.foreach(
                        cls.image,
                        lambda img: rx.box(
                            rx.box(
                                rx.image(
                                    src=img,
                                    max_width="100%",
                                    height="auto",
                                    border_radius="1px",
                                ),
                                padding="14px",
                                background="#faf7f0",
                                border="1px solid rgba(139,94,60,0.2)",
                                border_radius="3px",
                                box_shadow="0 2px 12px rgba(139,94,60,0.12), 0 0 1px rgba(0,0,0,0.05)",
                                transform="rotate(-0.5deg)",
                            ),
                            class_name="ink-block",
                            margin="32px 0",
                        ),
                    ),
                    # 图片 (images)
                    rx.foreach(
                        cls.images,
                        lambda img: rx.box(
                            rx.box(
                                rx.image(
                                    src=img,
                                    max_width="100%",
                                    height="auto",
                                    border_radius="1px",
                                ),
                                padding="14px",
                                background="#faf7f0",
                                border="1px solid rgba(139,94,60,0.2)",
                                border_radius="3px",
                                box_shadow="0 2px 12px rgba(139,94,60,0.12), 0 0 1px rgba(0,0,0,0.05)",
                                transform="rotate(-0.5deg)",
                            ),
                            class_name="ink-block",
                            margin="32px 0",
                        ),
                    ),
                    class_name="manuscript",
                ),
                position="absolute",
                top="0",
                left="0",
                right="0",
                bottom="0",
                z_index="1",
                overflow_y="auto",
                overflow_x="hidden",
                scroll_behavior="smooth",
            ),
            # ── 纸纹脚本 ──
            rx.script(_PAPER_GRAIN_SCRIPT),
            # ── 样式 ──
            rx.html(f"<style>{_LIVING_DOC_CSS}</style>"),
            width="100%",
            height="100vh",
            overflow="hidden",
            **props,
        )
