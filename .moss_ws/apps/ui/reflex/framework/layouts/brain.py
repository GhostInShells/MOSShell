"""Brain 突触拓扑 — 中心大脑向外辐射连接周围节点。纯 CSS 实现。

CSS animation/keyframes 驱动所有动画 —— 零 Canvas，零 JS 引擎。
节点以圆形拓扑环绕中心，辉光强度跟随 value。

字段兼容 MatrixLayout（status_bars: list[CellBar], title: str），
Ghost 无需修改脚本即可从 matrix 切换到 brain。
"""

import reflex as rx
from pydantic import BaseModel, Field

from framework.helpers.mixin import NameMixin


class CellBar(BaseModel):
    """Brain 节点状态 — 与 MatrixLayout 共享模型，Ghost 命令兼容"""

    label: str = Field(default="", description="节点名称")
    value: int = Field(default=0, description="连接强度 0-100")
    color: str = Field(default="#10b981", description="节点颜色")


# ═══════════════════════════════════════════════════════════════════════
# Sizing constants
# ═══════════════════════════════════════════════════════════════════════

_NODE_SIZE = 70        # px — node body width/height
_NODE_RADIUS = 300     # px — distance from center to node center
_CORE_SIZE = 110       # px — central brain width/height
_RING_SIZES = [190, 290, 410]  # px — orbiting ring diameters
_AD_RINGS = [100, 155, 220]    # px — ambient dot ring radii


# ═══════════════════════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════════════════════

_BRAIN_CSS = f"""
/* Scene */
.brain-scene {{
  position: absolute; inset: 0;
  background: radial-gradient(ellipse at 50% 50%, rgba(16,24,48,0.6) 0%, #020210 100%);
}}
.brain-scene::before {{
  content: ''; position: absolute; inset: 0;
  background-image: radial-gradient(circle, rgba(90,110,190,0.045) 1px, transparent 1px);
  background-size: 60px 60px;
  pointer-events: none;
}}

/* Central brain core */
.brain-core {{
  position: absolute; top: 50%; left: 50%; z-index: 2;
  width: {_CORE_SIZE}px; height: {_CORE_SIZE}px; margin: -{_CORE_SIZE//2}px 0 0 -{_CORE_SIZE//2}px;
  border-radius: 50%;
  background: radial-gradient(circle at 38% 38%,
    rgba(255,255,255,0.95) 0%,
    rgba(200,220,255,0.7) 12%,
    rgba(100,150,240,0.35) 40%,
    rgba(50,100,200,0) 70%);
  box-shadow:
    0 0 24px rgba(120,160,255,0.55),
    0 0 70px rgba(70,120,240,0.3),
    0 0 140px rgba(50,100,220,0.14),
    0 0 220px rgba(30,60,180,0.06);
  animation: brain-breathe 3.2s ease-in-out infinite;
}}
@keyframes brain-breathe {{
  0%, 100% {{
    transform: scale(1);
    box-shadow: 0 0 24px rgba(120,160,255,0.55), 0 0 70px rgba(70,120,240,0.3),
                0 0 140px rgba(50,100,220,0.14), 0 0 220px rgba(30,60,180,0.06);
  }}
  50% {{
    transform: scale(1.1);
    box-shadow: 0 0 36px rgba(120,160,255,0.7), 0 0 95px rgba(70,120,240,0.42),
                0 0 190px rgba(50,100,220,0.22), 0 0 280px rgba(30,60,180,0.1);
  }}
}}

/* Orbiting rings */
.brain-ring {{
  position: absolute; top: 50%; left: 50%; border-radius: 50%;
  border: 1px solid rgba(100,150,240,0.14);
  transform: translate(-50%, -50%);
  pointer-events: none; z-index: 1;
  animation: ring-breathe 4.5s ease-in-out infinite;
}}
.brain-ring.r1 {{ width: {_RING_SIZES[0]}px; height: {_RING_SIZES[0]}px; animation-delay: 0s; }}
.brain-ring.r2 {{ width: {_RING_SIZES[1]}px; height: {_RING_SIZES[1]}px; animation-delay: 1.5s; }}
.brain-ring.r3 {{ width: {_RING_SIZES[2]}px; height: {_RING_SIZES[2]}px; animation-delay: 3s; }}
@keyframes ring-breathe {{
  0%, 100% {{ opacity: 0.22; transform: translate(-50%, -50%) scale(1); }}
  50%      {{ opacity: 0.5;  transform: translate(-50%, -50%) scale(1.05); }}
}}

/* Ambient dots — 24 fixed positions, pure CSS */
.brain-ambient {{
  position: absolute; top: 50%; left: 50%; z-index: 1; pointer-events: none;
  width: 0; height: 0;
}}
.brain-ad {{
  position: absolute; width: 4px; height: 4px; border-radius: 50%;
  background: rgba(160,200,255,0.45);
  box-shadow: 0 0 8px rgba(140,190,255,0.3);
  animation: dot-twinkle 3s ease-in-out infinite;
}}
@keyframes dot-twinkle {{
  0%, 100% {{ opacity: 0.3; }}
  50%      {{ opacity: 0.7; }}
}}

/* Node anchor — zero-size at center, rotated + translated */
.brain-node-anchor {{
  position: absolute; top: 50%; left: 50%;
  width: 0; height: 0;
  z-index: 3;
}}

/* Node body — counter-rotated so content stays upright */
.brain-node-body {{
  position: absolute;
  top: -{_NODE_SIZE//2}px; left: -{_NODE_SIZE//2}px;
  width: {_NODE_SIZE}px; height: {_NODE_SIZE}px;
  border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  transition: filter 0.5s ease;
}}

/* Node glow aura */
.brain-node-aura {{
  position: absolute; inset: -18px; border-radius: 50%;
  opacity: 0; transition: opacity 0.6s ease;
  pointer-events: none;
}}
.brain-node-aura.on {{
  opacity: 0.55;
  animation: aura-pulse 2.5s ease-in-out infinite;
}}
@keyframes aura-pulse {{
  0%, 100% {{ transform: scale(1); opacity: 0.45; }}
  50%      {{ transform: scale(1.3); opacity: 0.8; }}
}}

/* Node center dot */
.brain-node-dot {{
  width: 18px; height: 18px; border-radius: 50%;
  transition: transform 0.4s ease, box-shadow 0.5s ease;
  z-index: 1;
}}
.brain-node-dot.on {{ transform: scale(1.3); }}

/* Node progress ring */
.brain-node-progress {{
  position: absolute; inset: -10px; border-radius: 50%;
  opacity: 0.3; pointer-events: none;
  mask: radial-gradient(circle, transparent 56%, black 58%);
  -webkit-mask: radial-gradient(circle, transparent 56%, black 58%);
}}

/* Label (outside node, radial outward) */
.brain-node-label {{
  position: absolute; top: {_NODE_SIZE//2 + 18}px; left: 50%; transform: translateX(-50%);
  white-space: nowrap;
  font-family: "SF Mono", Menlo, Monaco, Consolas, monospace;
  font-size: 13px;
  color: rgba(180,200,240,0.7);
  text-shadow: 0 0 8px rgba(0,0,0,0.8);
  pointer-events: none;
}}

/* Value text (above node) */
.brain-node-value {{
  position: absolute; top: -{_NODE_SIZE//2 + 10}px; left: 50%; transform: translateX(-50%);
  white-space: nowrap;
  font-family: "SF Mono", Menlo, Monaco, Consolas, monospace;
  font-size: 12px; font-weight: bold;
  color: rgba(160,190,230,0.55);
  text-shadow: 0 0 6px rgba(0,0,0,0.7);
  pointer-events: none;
}}

/* HUD */
.brain-title {{
  position: absolute; top: 48px; left: 50%; transform: translateX(-50%);
  font-family: "SF Pro Display", -apple-system, BlinkMacSystemFont, sans-serif;
  font-size: 18px; font-weight: 600; letter-spacing: 3px;
  color: rgba(160,175,220,0.55);
  text-shadow: 0 0 16px rgba(80,100,200,0.2);
  text-transform: uppercase;
  z-index: 5; pointer-events: none;
}}
.brain-subtitle {{
  position: absolute; top: 78px; left: 50%; transform: translateX(-50%);
  font-family: "SF Mono", Menlo, Monaco, Consolas, monospace;
  font-size: 13px;
  color: rgba(120,140,190,0.45);
  z-index: 5; pointer-events: none;
}}

/* Debug */
.brain-debug {{
  position: fixed; top: 4px; right: 8px;
  color: rgba(100,140,220,0.4);
  font-size: 11px; font-family: monospace;
  z-index: 99; pointer-events: none;
}}
"""

# Ambient dot class names: ad0..ad23
_AD_CLASSES = [f"brain-ad ad{i}" for i in range(24)]
# Hardcoded positions: each (ring_index, degrees)
_AD_POSITIONS = [
    # inner ring
    (0, 0), (0, 45), (0, 90), (0, 135), (0, 180), (0, 225), (0, 270), (0, 315),
    # middle ring
    (1, 22), (1, 67), (1, 112), (1, 157), (1, 202), (1, 247), (1, 292), (1, 337),
    # outer ring
    (2, 11), (2, 56), (2, 101), (2, 146), (2, 191), (2, 236), (2, 281), (2, 326),
]


def _ambient_dot_css(i: int) -> str:
    """Generate CSS for one ambient dot with its hardcoded position."""
    ring, deg = _AD_POSITIONS[i]
    r = _AD_RINGS[ring]
    delay = (i * 0.13) % 3.0
    return (
        f".brain-ad.ad{i} {{ "
        f"transform: rotate({deg}deg) translateY(-{r}px); "
        f"animation-delay: {delay:.2f}s; "
        f"}}"
    )


# Append ambient dot position rules to CSS
_BRAIN_CSS += "\n" + "\n".join(_ambient_dot_css(i) for i in range(24))


class BrainLayout(rx.ComponentState, NameMixin):
    """Brain 突触拓扑 —— 中心大脑向外辐射连接各 Cell 节点。

    纯 CSS 实现：CSS animation/keyframes 驱动中枢呼吸、节点辉光脉动。
    节点以圆形拓扑环绕中心，每个节点支持 label / value / color。
    字段兼容 MatrixLayout，Ghost 切换无需修改脚本。
    """

    status_bars: list[CellBar] = []
    title: str = ""
    subtitle: str = ""

    @classmethod
    def name(cls) -> str:
        return "brain"

    @classmethod
    def get_component(cls, **props) -> rx.Component:
        n = cls.status_bars.length()  # Var[int]
        has_title = cls.title != ""
        has_subtitle = cls.subtitle != ""

        return rx.box(
            # Scene background + grid
            rx.box(class_name="brain-scene"),
            # Ambient dots
            rx.box(
                *[rx.box(class_name=c) for c in _AD_CLASSES],
                class_name="brain-ambient",
            ),
            # Orbiting rings
            rx.box(class_name="brain-ring r1"),
            rx.box(class_name="brain-ring r2"),
            rx.box(class_name="brain-ring r3"),
            # Central brain core
            rx.box(class_name="brain-core"),
            # Nodes on circle
            rx.foreach(
                cls.status_bars,
                lambda bar, i: rx.box(
                    # Body: counter-rotated to keep text upright
                    rx.box(
                        # Glow aura
                        rx.box(
                            class_name=rx.cond(
                                bar.value > 0,
                                "brain-node-aura on",
                                "brain-node-aura",
                            ),
                            style={
                                "background": f"radial-gradient(circle, {bar.color}44 0%, transparent 70%)",
                            },
                        ),
                        # Progress ring
                        rx.box(
                            class_name="brain-node-progress",
                            style={
                                "background": f"conic-gradient({bar.color} calc({bar.value} * 3.6 * 1deg), transparent 0deg)",
                            },
                        ),
                        # Center dot
                        rx.box(
                            class_name=rx.cond(
                                bar.value > 0,
                                "brain-node-dot on",
                                "brain-node-dot",
                            ),
                            style={
                                "background": bar.color,
                                "boxShadow": rx.cond(
                                    bar.value > 0,
                                    f"0 0 18px {bar.color}, 0 0 40px {bar.color}88",
                                    f"0 0 10px {bar.color}66",
                                ),
                            },
                        ),
                        # Label
                        rx.cond(
                            bar.label != "",
                            rx.text(bar.label, class_name="brain-node-label"),
                        ),
                        # Value %
                        rx.cond(
                            bar.value > 0,
                            rx.text(f"{bar.value}%", class_name="brain-node-value"),
                        ),
                        class_name="brain-node-body",
                        style=rx.cond(
                            n > 0,
                            {
                                "transform": f"rotate(calc(-1 * {i} * 360deg / max({n}, 1)))",
                            },
                            {},
                        ),
                    ),
                    class_name="brain-node-anchor",
                    style=rx.cond(
                        n > 0,
                        {
                            "transform": f"rotate(calc({i} * 360deg / max({n}, 1))) translateY(-{_NODE_RADIUS}px)",
                        },
                        {"display": "none"},
                    ),
                ),
            ),
            # HUD
            rx.cond(has_title, rx.text(cls.title, class_name="brain-title")),
            rx.cond(has_subtitle, rx.text(cls.subtitle, class_name="brain-subtitle")),
            # Debug watermark
            rx.text("brain css", class_name="brain-debug"),
            # CSS
            rx.html(f"<style>{_BRAIN_CSS}</style>"),
            width="100%",
            height="100vh",
            overflow="hidden",
            background="#020210",
            **props,
        )
