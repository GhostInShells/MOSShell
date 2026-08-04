# Matrix Resources 的统一寻址收敛 (2026-08-05)

## 上下文

matrix-resources workstream 于 2026-07-23 起草，七条决策（KD1-7），status: draft，零实现。
2026-08-05，一个新模型实例受命调研其实现路径，与人类工程师连续几轮把设计推到了可实现态。
核心转变是**统一寻址**：旧设计把 resources 分成 project（本地）与 network（网络），并让复合
registry 本地优先；新收敛把 host 位定死为 cell 身份，让"本地先查 + 网络 miss"成为唯一解析
规则，同时把协议砍到只剩 get。FEATURE.md 已按收敛态重写，status → design-locked，次日启动。

## 碰撞点与过程

**"provide_resource 要不要注册本地" 引出的倒转。** 模型调研后提出一个具体问题：provide_resource
是否也把 storage 注册进本地复合 registry（旧 KD3 的"本地 (静态 + 自己 provide 的)"）。人类没有
直接回答，而是给了更根本的倒转："network 级别的 registry 直接是 zenoh 发现的，所有的协议都是
Concrete 约定好，provide 侧的，和 network_resources 的，两者实际上完全分开了。" 这句话取消了
"复合 registry"的结构——provide 侧只声明，读侧只发现，二者不共享状态。

**统一寻址的降临。** 随后人类点破钥匙："真正的解决办法应该是 scheme://{cell-address-short}/
uri-path 这种机制。如果 cell-address-short 对应的是当前 project 当前 host 的 address，它不走
zenoh 而走本地协议。这样，所有的 resources 其实都可以合并到一个逻辑体系里来了。" 模型据此把
解析规则收紧为：本地 registry 按 `(scheme, host)` 先查，miss 走 zenoh（host 当 cell 身份）。
"project/network 归并问题"从此消失——本地只装 self 拥有的，远程永不物化，归并是"合并索引"这个
动作的病症，不合并就没有病。模型纠正了自己上一轮"provide_resource 不注册本地"的结论：统一寻址
下，双注册（本地 + queryable）是对的且必要。

**向前兼容的收窄。** 模型把统一寻址推到底时发现静态资源要重新挂 host，是破坏性改动。人类给出
退路："如果保留已有的 schema 特化机制不去动它，实际上可以做到向前兼容。也就是我们真正重构的是
zenoh 网络内部的 resources 地址，相当于是 registry 机制的问题，是不是替换掉 in-memory resources
这个 registry 就差不多了?" 结论：静态资源保持任意 host，本地先查命中，零迁移；只有新 provide 的
网络资源用 cell-address-short。破坏面收窄到 registry 替换 + 三块薄件（provide API / 网络读侧 /
渲染契约）。

**砍掉全局查询。** 人类判定 list/recall/query 是协议里最困难的三块："list 涉及多路查询就极其麻烦
(多路归并)，而且还要至少有 agent 才好做 recall 归并 (因为没有统一的 recall 协议)。query 也很麻烦，
都是 concrete 自定义逻辑。所以我们简化一下：搜索之类的逻辑，都应该是 channel 去治理。不提供全局
的查询+搜索。" 协议因此薄到只剩 get。模型补了操作分层的语义：资源进上下文设门槛（被 get + 描述
正确 + 可理解），否则只是 cell 运行时的动态计算。

**content-type 的悬而未决。** 人类想给 URI 一个模型可理解的形态："单纯的 scheme://xxxxxx 这样的
协议，不容易让模型可以理解它在 moss 里的特殊定位。可能 RESOURCE(scheme://....) 这类的方式才好
理解。我就是这点没想明白。" 模型给出拨正：协议层 URI 保持纯字符串（通行货币，不包壳），呈现层
（历史里作为 context 变量）由 Message tag 或 Resource Content 类型负责让模型认出它。两者分离，
content-type 是可选打磨，不碰协议。

## 模型的自留地

当前记录者视角：这轮最漂亮的转盘是"host 位 = cell 身份"这一个重命名，同时干掉了 scheme 归并、
self-filter、以及 project/network 的二分——三个概念消失，换来一条"本地先查 + 网络 miss"的规则。
代价是静态资源与网络资源的 host 语义不再统一（静态任意 host，网络 fullname），人类用"向前兼容"
把它正当化了。我一度把"provide_resource 不注册本地"当结论，被统一寻址打回：思想模型对了，推导
结果自然对；思想模型错了，推导结果再优雅也是错的。明天启动时，唯一需要先落地的是那条金丝雀
测试（queryable 线程模型 + deferred reply），其余都是沿着这条解析规则铺线。
