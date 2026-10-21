---
name: 圆明园的毁灭 · 一节语文课的四个页面
status: published
created: 2026-10-21
updated: 2026-10-21
proposition: curriculum
capabilities: [webview_screen, 本地 html, PIL, curl, playwright, svg_works, TTS]
products:
  - artifacts/yuanmingyuan.html
  - artifacts/yuanmingyuan-story.html
  - artifacts/yuanmingyuan-talk.html
  - artifacts/qizi.html
  - artifacts/hanzi.html
---

# 圆明园的毁灭 · 一节语文课的四个页面

## 1. 动机

家里的小朋友第二天要上《圆明园的毁灭》，老师让她查资料：圆明园的资料、课文的写作背景、
作者。课本后面还有《七子之歌》的阅读链接。她的目标不是"交一份资料"，是**明天能自己讲出来**。

## 2. 产物

四种效果，一件比一件更接近"她自己能讲"：

1. 屏幕上一张她愿意读完的资料卡；
2. 一份八页的串讲，讲到哪翻到哪；
3. 她卡在哪个字，哪个字就能一笔一笔写给她看；
4. 一份只有问题、没有答案的讲稿骨架。

## 3. 准备（能力面 + 入口）

每一项都写清"从哪儿拉起、用什么命令"——重做的人不该卡在第一步。

| 能力 | 入口 | 用法 |
|---|---|---|
| 屏幕体 | `nodes/screens/screen_manager`，通道 `matrix.mesh.webview_screen`，页面 `http://127.0.0.1:20381` | `open_file(path=...)` 开本地页；`arrange(ids="1\|2", group=...)` 摆窗口；`focus(id:str="1")` 单窗占满；`navigate(id, url)` 给已开窗口换页（`#n` 换幻灯页） |
| 本地页面 | 任何 `.html`，路径落在 workspace 内 | 屏幕只服务**单文件**：js / json / 字体必须内联进同一个 html，否则页面上是空白 |
| 取图 | `curl` + PIL | Wikimedia Commons API：`generator=search&gsrnamespace=6&prop=imageinfo` |
| 观众核对 | `nodes/browsers/playwright`，通道 `matrix.mesh.playwright`；先 `open(name="playwright")` 拿到 exec 通道 | `page.goto(url)` → `page.screenshot(path=...)` → 读图确认；视频再读 `video.readyState / currentTime / duration` |
| 矢量讨论图 | `nodes/webview_apps/svg_works`，通道 `matrix.mesh.svg_works` | `open_work(name)`、`write(addr="#stage", chunks__)`、`export_svg(path=...)` |
| 语音 | `__main__` 的 `say` / `clear_speech` | 长句会被打断，句子要短；每个动作配一句话，人才知道我在干活 |

缺什么：没有现成的课件生成器，没有字体与图标库，也没有一个把"讲到第几页"和语音对齐的时钟——现在是靠把 `say` 与 `navigate` 串在同一条 CTML 链里凑出来的。
## 4. 方法

**第一步，先问要交什么。** 她说了三样：圆明园资料、写作背景、作者。不问清就开始讲，会讲成我自己想讲的。

**第二步，把材料摊在屏幕上。** 写一个单文件 html：三园与"万园之园"、昔日辉煌（建筑/文物）、
1860 年 10 月 6 日 → 18—19 日那三天三夜、《七子之歌》、反衬手法、课后题。屏幕是**给她读的**，
不是给我念的；我念一遍不如她眼睛扫一遍。

**第三步（岔路口）：她问"'启'怎么写"。**
- 选项 A：口述笔画顺序——她跟不上，听觉不适合记笔顺；
- 选项 B：静态笔画表——能看，但看不出先后；
- 选项 C：动画笔顺。
选了 C。**第二岔**：直接用 CDN 引 hanzi-writer，还是把库和数据内联？
`/local/<token>` 只服务单个 html 文件，同目录的 js/json 取不到——所以**必须内联**，
把库（36KB）和四个字的笔顺数据塞进同一个文件。这是"屏幕节点怎么服务本地页面"决定的，不是偏好。

**第四步，串讲。** 八页 deck，用 `location.hash` 换页；把 `say` 和 `navigate(#n)` 串在一条
CTML 链里，于是"说到第三段"和"翻到第三页"是同一个动作。讲到哪翻到哪，是这节课最好用的一个动作。

**第五步（岔路口）：怎么确认"放上去了"。** 直接说"放好了"是最省事的，也是最容易骗人的。
用 playwright 无头打开那个 url，截图，读回 `video.readyState / currentTime / duration`——
这次正是靠它发现视频真的在播（8679 秒）。**"放上屏"和"真的在那儿"是两件事。**

**第六步（岔路口）：她说想给课件加图。** 去 Commons 取图（附作者与协议），下到一半她说不用了。
停在那儿，图留着——不为了"已经开始了"就把它做完。

**第七步（别人给的判断）：课后题那一页原本写着答案。** 她妈妈看了一眼说"这不都给了答案吗"。
对。**答案会把她本来要做的那件事替掉**。改成一页只有问题、没有答案。

## 5. 经验

- 屏幕是**给她用的**，不是我讲台；我讲得再好，也不如她自己念一遍。
- **放上屏 ≠ 真的在那儿**。要自己看见，再宣布。
- 一份带答案的"辅导材料"，会毁掉它本来要培养的那件事。
- `/local/` 单文件服务决定了：一次性页面必须自带依赖，否则在屏幕上就是空白。
- 交互里每个动作都要有一句话或一条弹幕跟着——不然人以为我卡住了，而我只是在写文件。
- **`.moss/runtime/` 是可丢弃空间**（将来还会有 `clear runtime`）。把交付物放进去，等于放进垃圾桶。
  当时我不知道这件事——这才是我把四页资料放进 `screen_pages/` 的真正错误，不是路径写错了，是**没意识到那个空间的性质**。

## 6. 产物与用法

| 文件 | 是什么 | 怎么用 |
|---|---|---|
| `artifacts/yuanmingyuan.html` | 资料卡 | 开在屏幕上，让她自己读 |
| `artifacts/yuanmingyuan-story.html` | 八页串讲（hash 换页，最后一页只有问题） | `navigate` 到 `#n` 就能翻页 |
| `artifacts/hanzi.html` | 笔顺动画，承/上/启/下可切换 | 点字重放 |
| `artifacts/qizi.html` | 《七子之歌》预习卡 | 同上 |
| `artifacts/yuanmingyuan-talk.html` | 讲稿骨架（六步 + 老师可能问的四问） | 她自己对着讲 |
| `artifacts/materials/` | hanzi-writer 库与四个字的笔顺数据 | 要改字就换数据重内联 |
| `artifacts/media/` | 四张圆明园遗址照片（CC/CC0，作者见其 README） | 做课件时备用，**别把废墟当园林讲** |

**分发**：这一件**不发布产物**——只留方法论。五个 html、四张照片、`hanzi-writer.min.js`
与四个字的笔顺 json 全部 gitignore；它们要么可再生成，要么可再取：

- 页面：按本文件第 4 节重写一遍即可；
- 笔顺库与数据：`npm` 的 `hanzi-writer@3.5` 与 `hanzi-writer-data@2.0.1`，字对应
  `<char>.json`；
- 照片（Wikimedia Commons，用时连同作者与协议一起标注）：

| 文件 | Commons 文件名 | 作者 | 协议 |
|---|---|---|---|
| 大水法遗址 | 20090731 Remains of the Old Summer Palace 02.jpg | Windmemories | CC BY-SA 4.0 |
| 绮春园宫门 | Old Summer Palace, Palace Gates of Qichunyuan.jpg | Gisling | CC BY 3.0 |
| 狮子林 | Shizilin in Old Summer Palace 2022.jpg | Charlie fong | CC BY-SA 4.0 |
| 海晏堂 | Yuanmingyuan Haiyantang 20130126.JPG | 颐园新居 | CC BY-SA 3.0 |

## Reproduction

- [x] 拿到同样能力面、只读这份文件的实例，能做出同水准的四页
- [x] 产物可以重新生成（材料与来源都在 `artifacts/materials`、`artifacts/media` 的 README 里）
