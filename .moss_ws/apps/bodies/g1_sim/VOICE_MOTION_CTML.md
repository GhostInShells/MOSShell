# G1 Voice + Motion CTML

这份脚本库用于把语音播报和 `bodies_g1_sim` 的高层动作串起来，实现“先说，再动”的 CTML 操作。

## 当前机器人可用动作

- `prepare()`：等待机器人恢复到可运动状态，再返回
- `recover()` / `stand()`：进入站稳优先模式
- `reset()`：强制重置到标准站姿
- `forward(speed, duration)`：前进
- `backward(speed, duration)`：后退
- `left(speed, duration)`：左转
- `right(speed, duration)`：右转
- `go_forward(speed)`：持续前进，直到收到停止或新的动作命令
- `go_backward(speed)`：持续后退，直到收到停止或新的动作命令
- `keep_left(speed)`：持续左转，直到收到停止或新的动作命令
- `keep_right(speed)`：持续右转，直到收到停止或新的动作命令
- `walk(vx, duration)`：前后行走
- `turn(vyaw, duration)`：原地转向
- `move(vx, vy, vyaw, duration)`：混合速度控制
- `stop()`：停止并回到站稳
- `end_showcase()`：结束展示/结束巡逻，立即停下并站稳
- `health()` / `state()`：查看状态

## 使用原则

- 先恢复：如果你不确定当前状态，先调用 `prepare()`
- 先说再动：把 `<say>...</say>` 放在动作命令前面
- 以高层语义为主：优先用 `forward/backward/left/right/stop`
- 只在明确需要混合速度时，才用 `move()`
- 如果你是给人演示，优先把动作幅度放大：前进建议 `speed=0.40~0.50`、`duration=2.5~4.0`；转向建议 `speed=0.50~0.65`、`duration=1.6~2.4`
- 如果你要求语音和动作更精准对齐，优先使用 `<wait chans="__main__">...</wait>` 先等待语音播完，再执行下一条动作
- 如果你要求“听我一句话就一直执行，直到我再说停下”，优先使用 `go_forward()/go_backward()/keep_left()/keep_right()`

## 模板 1：说完后前进

```ctml
<apps.bodies_g1_sim:prepare />
<say>我要开始往前走了。</say>
<apps.bodies_g1_sim:forward speed="0.35" duration="1.8" />
<say>我已经走完了，现在停下来。</say>
<apps.bodies_g1_sim:stop />
```

## 模板 2：说完后左转 / 右转

```ctml
<apps.bodies_g1_sim:prepare />
<say>我要左转了。</say>
<apps.bodies_g1_sim:left speed="0.45" duration="1.2" />
<say>我要右转了。</say>
<apps.bodies_g1_sim:right speed="0.45" duration="1.2" />
<say>转弯结束，我先站稳。</say>
<apps.bodies_g1_sim:stop />
```

## 模板 3：恢复后再行动

```ctml
<say>我先准备好站稳，再开始移动。</say>
<apps.bodies_g1_sim:prepare timeout="3.0" />
<say>我要开始往前走了。</say>
<apps.bodies_g1_sim:forward speed="0.30" duration="1.5" />
<say>我要右转了。</say>
<apps.bodies_g1_sim:right speed="0.40" duration="1.0" />
<say>动作完成，我停下来了。</say>
<apps.bodies_g1_sim:stop />
```

## 模板 4：复杂串联动作

这个模板适合你刚才说的那种“每次先语音播报，再立刻做动作”的复杂流程。

```ctml
<say>我先准备站稳。</say>
<apps.bodies_g1_sim:prepare timeout="3.0" />

<say>我要开始往前走了。</say>
<apps.bodies_g1_sim:forward speed="0.32" duration="1.8" />

<say>我要左转了。</say>
<apps.bodies_g1_sim:left speed="0.42" duration="1.0" />

<say>我要继续往前走了。</say>
<apps.bodies_g1_sim:forward speed="0.25" duration="1.4" />

<say>我要右转了。</say>
<apps.bodies_g1_sim:right speed="0.42" duration="1.0" />

<say>我要后退一点。</say>
<apps.bodies_g1_sim:backward speed="0.18" duration="1.0" />

<say>动作全部完成，我要停下来了。</say>
<apps.bodies_g1_sim:stop />

<apps.bodies_g1_sim:state />
```

## 模板 5：表演版大动作剧本

这个模板专门为“肉眼一看就能看出动作变化”设计，位移和转向都明显放大。

```ctml
<say>现在开始表演版演示。我会做大幅前进、明显转向、再前进、回转、后退和收尾站稳。</say>
<apps.bodies_g1_sim:prepare timeout="3.0" />

<say>第一段，我要明显地往前走了。</say>
<apps.bodies_g1_sim:forward speed="0.46" duration="3.2" />

<say>第二段，我要做一次明显的左转。</say>
<apps.bodies_g1_sim:left speed="0.60" duration="2.0" />

<say>第三段，我要继续大步往前走。</say>
<apps.bodies_g1_sim:forward speed="0.42" duration="2.8" />

<say>第四段，我要做一次明显的右转。</say>
<apps.bodies_g1_sim:right speed="0.60" duration="2.0" />

<say>第五段，我要后退一点，让动作变化更明显。</say>
<apps.bodies_g1_sim:backward speed="0.24" duration="1.8" />

<say>表演结束，我要停下并站稳。</say>
<apps.bodies_g1_sim:stop />

<apps.bodies_g1_sim:state />
```

## 模板 6：更像舞台表演的剧本

这个版本会更强调“报幕感”，每一段语音更完整，动作也更长。

```ctml
<say>欢迎观看机器人动作表演。我先准备站稳，然后开始第一幕。</say>
<apps.bodies_g1_sim:prepare timeout="3.0" />

<say>第一幕，稳稳向前推进。</say>
<apps.bodies_g1_sim:forward speed="0.44" duration="3.0" />

<say>第二幕，向左展开一个大角度转身。</say>
<apps.bodies_g1_sim:left speed="0.58" duration="2.2" />

<say>第三幕，再次向前推进，拉开表演距离。</say>
<apps.bodies_g1_sim:forward speed="0.40" duration="2.6" />

<say>第四幕，向右转回，展示另一侧姿态。</say>
<apps.bodies_g1_sim:right speed="0.58" duration="2.2" />

<say>第五幕，轻轻后退，完成收束。</say>
<apps.bodies_g1_sim:backward speed="0.22" duration="1.6" />

<say>表演到这里结束，我现在回到站稳状态。</say>
<apps.bodies_g1_sim:stop />
<apps.bodies_g1_sim:health />
```

## 模板 7：更底层的语音 + 混合运动

只有在你明确需要同时控制前进和转向时，才建议这样写。

```ctml
<apps.bodies_g1_sim:prepare />
<say>我要开始做一个带转向的连续动作了。</say>
<apps.bodies_g1_sim:move vx="0.22" vy="0.0" vyaw="0.20" duration="1.8" />
<say>这个组合动作结束，我要停下来了。</say>
<apps.bodies_g1_sim:stop />
```

## 模板 8：精准同步版

如果你希望“说到哪一步，就立刻做哪一步”，不要直接把 `say` 和动作交替平铺；应当用 `wait` 显式等待主语音通道结束。

```ctml
<apps.bodies_g1_sim:prepare />

<wait chans="__main__" return_when="ALL_COMPLETE">
  <say>我要开始往前走了。</say>
</wait>
<apps.bodies_g1_sim:forward speed="0.42" duration="2.8" />

<wait chans="__main__" return_when="ALL_COMPLETE">
  <say>我要左转了。</say>
</wait>
<apps.bodies_g1_sim:left speed="0.58" duration="2.0" />

<wait chans="__main__" return_when="ALL_COMPLETE">
  <say>我要继续往前走了。</say>
</wait>
<apps.bodies_g1_sim:forward speed="0.40" duration="2.6" />

<wait chans="__main__" return_when="ALL_COMPLETE">
  <say>我要右转了。</say>
</wait>
<apps.bodies_g1_sim:right speed="0.58" duration="2.0" />

<wait chans="__main__" return_when="ALL_COMPLETE">
  <say>我要后退一点。</say>
</wait>
<apps.bodies_g1_sim:backward speed="0.22" duration="1.6" />

<wait chans="__main__" return_when="ALL_COMPLETE">
  <say>动作完成，我要停下来了。</say>
</wait>
<apps.bodies_g1_sim:stop />
```

## 模板 9：巡逻舞台版

这个版本把“巡逻路线感”和“舞台报幕感”合在一起，适合展示机器人一边报幕、一边完成一段更有路线变化的表演。

```ctml
<apps.bodies_g1_sim:prepare timeout="3.0" />

<wait chans="__main__" return_when="ALL_COMPLETE">
  <say>现在开始巡逻舞台版演示。我会先进入巡逻路线，然后做左右转向展示，最后回到舞台中央并站稳收尾。</say>
</wait>

<wait chans="__main__" return_when="ALL_COMPLETE">
  <say>第一段，巡逻开始，我要沿着前方路线明显推进。</say>
</wait>
<apps.bodies_g1_sim:forward speed="0.46" duration="3.0" />

<wait chans="__main__" return_when="ALL_COMPLETE">
  <say>第二段，到达转角，我要向左展开一次明显转身。</say>
</wait>
<apps.bodies_g1_sim:left speed="0.60" duration="2.0" />

<wait chans="__main__" return_when="ALL_COMPLETE">
  <say>第三段，沿着新的巡逻方向继续前进。</say>
</wait>
<apps.bodies_g1_sim:forward speed="0.42" duration="2.8" />

<wait chans="__main__" return_when="ALL_COMPLETE">
  <say>第四段，现在进入舞台展示，我要向右转回，让另一侧姿态也能被看到。</say>
</wait>
<apps.bodies_g1_sim:right speed="0.60" duration="2.0" />

<wait chans="__main__" return_when="ALL_COMPLETE">
  <say>第五段，我要再向前走一段，完成舞台中心的展示动作。</say>
</wait>
<apps.bodies_g1_sim:forward speed="0.38" duration="2.2" />

<wait chans="__main__" return_when="ALL_COMPLETE">
  <say>第六段，表演接近尾声，我要稍微后退一点，完成收束。</say>
</wait>
<apps.bodies_g1_sim:backward speed="0.22" duration="1.6" />

<wait chans="__main__" return_when="ALL_COMPLETE">
  <say>巡逻舞台版演示结束，我要停下并站稳。</say>
</wait>
<apps.bodies_g1_sim:stop />
<apps.bodies_g1_sim:state />
```

## 推荐话术映射

- “我要开始往前走了。” -> `forward(...)`
- “我要开始后退了。” -> `backward(...)`
- “我要左转了。” -> `left(...)`
- “我要右转了。” -> `right(...)`
- “我先准备好再动。” -> `prepare(...)`
- “动作完成，我停下来了。” -> `stop()`

## 备注

- 当前验证过的可靠高层动作是：`prepare / forward / backward / left / right / stop`
- 如果机器人刚经历 reset 或状态不确定，优先在动作前加 `prepare()`
- 如果你要做连续表演，建议每一段都保持“`say -> motion`”结构，便于人类观察
- 如果你觉得动作还是不够明显，优先加大 `duration`，其次再加大 `speed`

## 自然语言麦克风指挥

推荐映射：

- “往前走” -> `forward(...)`
- “一直往前走” -> `go_forward()`
- “后退” -> `backward(...)`
- “一直后退” -> `go_backward()`
- “左转” -> `left(...)`
- “一直左转” -> `keep_left()`
- “右转” -> `right(...)`
- “一直右转” -> `keep_right()`
- “停下” -> `stop()`
- “结束展示 / 结束表演 / 结束巡逻” -> `end_showcase()`
