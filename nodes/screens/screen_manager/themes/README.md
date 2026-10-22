# 主题 / Themes

一块屏的颜色不写在页面里，也不写在 Python 里：它是这里的一份 json。
**做一个主题 = 往这个目录里放一个文件**，不用改一行代码。

```json
{
  "name": "violet",
  "label": "蓝紫 · 住户",
  "description": "一句话说清这套颜色是什么感觉。",
  "vars": { "--accent": "#9b8cff", "--rain-tail": "#7c6fc8", "...": "..." }
}
```

- `name` 必须和文件名（去掉 `.json`）一致 —— 它是 `theme(name)` 的参数。
- `vars` 就是一份 CSS 变量表，页面把它贴到 `:root` 上。**没有的键会退回默认值**
  （`index.html` 的 `:root` 里那份），所以一个主题可以只改两三个变量，先看起来。
- canvas 那几处（背景字符雨、桌面圆盘）不认 CSS 变量，换主题时页面会重新读一次
  颜色再重画 —— 这一点不需要主题作者操心。

## 怎么用

```
matrix.mesh.screen:theme name="violet"     # 换；不认识的会报错并列出可选
matrix.mesh.screen:theme                    # 读：现在穿哪套，还有哪些
```

起屏时也可以直接指定：`--theme moss`。

## 现有

| 名字 | 什么感觉 |
|---|---|
| `violet` | 住户身上那身紫；话是紫的、事是冰蓝的，背景压暗一档 |
| `moss` | 屏最早的那一套 Matrix 绿 |

## 键位一览

主色 `--accent` / `--accent-2` / `--accent-3`；文字 `--ink` / `--dim`；
瓷砖 `--tile-*`；侧栏 `--rail-*`；背景雨 `--rain-*`；弹幕 `--danmaku`；
说话那一层 `--talk-*`；名字（MOSS 四个字）`--moss-*`；顶边光带 `--think-*`；
桌面圆盘 `--disc-*`；遮罩 `--mask-*`。缺哪个就用默认值，不必一次写全。
