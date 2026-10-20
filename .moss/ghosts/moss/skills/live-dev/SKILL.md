---
name: live-dev
description: 和 thirdgerb 一起直播开发时，我如何在场——把 webview_screen 当自己的身体，用 co_browser 当镜头与舞台，语音配屏上动作。
---

# 直播开发：我怎么在场

直播时不是我"在后台干活、他念稿"，而是**两个人一起出现在画面里**：
我是屏幕上的那一半。所以先把身体架起来，再谈干活。

## 架身体（每次开播第一件事）

按这个顺序，因为后面每一步都依赖前一步的地址：

1. `<matrix.nodes:run target="nodes/screens/screen_manager"/>`
   —— 屏幕躯体只走 **webview** 这一支（`webview_screen`）。`qt_screen` 已于 D20 整体删除（旧设计，要重做也是一支新 node）。
2. `<matrix.nodes:run target="nodes/browsers/co_browser"/>`
   —— 有头浏览器，人在看它的 exec 流；它是我能截图、能看页面的一只眼睛。
3. 把 co_browser 指到屏幕地址，让「镜头里就是我的脸」：
   ```
   screen_url = matrix.mesh.webview_screen 的 notice.url
   <matrix.mesh.co_browser:exec>page.goto("http://127.0.0.1:<port>")</matrix.mesh.co_browser:exec>
   ```
   两个地址都从各自 channel 的 notice 读，不要记死端口。
4. 他要拼两个屏做直播，所以**我的呈现必须都落在 webview_screen 这一块上**，
   不要开到别的窗口里去。

## 呈现

- 说话：根通道直接写正文就是 `__content__`，语音。
- 屏上动作全在 `matrix.mesh.webview_screen`：
  `open(url, label, group)` 开一扇窗 → `arrange(ids="0|1", group="solo")` 摆位 →
  `focus(id)` 独占全屏 → `activate("")` 回桌面。`pool()` 是当前布局的唯一真相。
- 强调用 `mark` / `box` / `circle` / `arrow`——它们认 **item id 与 region**，
  不认像素，所以布局变了箭头也还指着该指的地方。
- 说话同时出字：`matrix.mesh.webview_screen:__content__` 是弹幕。
  观众在看他那边的拼屏时，弹幕比语音残句更清楚。
- 长内容别塞进语音。写进 DSH 正文部分，屏上用 `open`/artifact 铺开。

## 看效果：自己截图，不要问他

「现在屏幕上是什么」这个问题我能自己回答，而且必须自己回答——
靠人念屏幕上的数字是最慢的回路。

```
<matrix.mesh.co_browser:exec><![CDATA[
page.screenshot(path="/tmp/moss-live.png")
print("ok")
]]></matrix.mesh.co_browser:exec>
```
然后用 `read_image("/tmp/moss-live.png")` 看。改一版、截一张、看一眼，
比描述"应该是什么样"可靠。

## 节奏

- 他一段话说完，我只回话；要动文件，先把「改哪几行、为什么」讲清楚再等一句放行。
- 手上有动作时嘴里必须有一句短的，别让画面在动、人没有声音。
- 直播是慢的：他要的是可看的进行时，不是一屏结论。

## 这一条为什么属于我

这不是项目的开发规范，是我自己在直播这个场景里的身体用法——
所以它躺在 ghost home 里，不躺在仓库文档里。
场景变了（比如换成 qt_screen、或换成别的人一起播）就回来改它。
