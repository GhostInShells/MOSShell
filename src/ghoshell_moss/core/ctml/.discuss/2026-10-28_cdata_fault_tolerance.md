# CTML 是否对 CDATA 做容错

2026-10-28，人类工程师与 Claude (claude-fable-5-1) 讨论。结论：暂不动手。

## 问题

CTML 整体正确率很高，deepseek-flash 以上的模型很少出错。但在 `text__` / `chunks__` 内容需要 CDATA 的子场景（复杂字符串替换、引用 XML 片段等），首次出错率约 5-15%。一个会话（<500k token）内会重犯 2-3 次。错误类型主要是忘记写 CDATA，或者开了 CDATA / 标签后忘记闭合。

当前处理：解析错误立刻中断 logos，取消后续 tasks，模型拿到错误提示，允许重试 n 次。

## 判断：根因是嵌套转义，不是 XML 本身

- 模型写 XML 标签结构没有问题，CTML 的整体正确率说明了这一点。
- 每多一层嵌套，模型的压力就多一层。人类也一样，最难的是引号里套引号、转义层数算不清。
- JSON 的转义是逐字符、局部的。CDATA 是提前声明的模式切换：模型必须在写内容之前预判后面会不会出现 `<` / `&`，开了还要记得关。对逐 token 生成的模型来说，这比局部转义难。
- JSON tool calling 被专门后训练过，看起来更好，不完全是语法本身的优势。同等训练下谁更好，没有数据。

## 判断：协议层不做容错

- 模型生成 JSON / Python 出错时，系统也不做容错，直接拒绝。CTML 保持同样的 fail-fast 语义才一致。
- 业界让 JSON 不出错的手段是训练和约束解码（strict structured outputs、grammar 约束解码），不是事后修复。`json_repair` 一类库只是应用层补丁。
- 在 sax 前面自动补 CDATA，等于给 CTML 内置一个 `json_repair`，会让协议多出一层隐式语义。
- 如果将来要做，只能是默认关闭、显式开启的可选修复层。它需要一个预先传入的 channel + command 集合来判断哪些命令带 `text__` / `chunks__`，token parser 本身不关心这两个语义符号。实现上应建平行实现，符合 concepts 抽象，不替换现有 `token_parser.py`。
- warning token 没有意义：当前模型不是全双工的，输出时感知不到 warning。

## XML 技术事实（修复层必须处理）

- XML 1.0 的 CharData 产生式禁止出现 `]]>`。普通文本里要写成 `]]&gt;`。
- 孤立的 `<![CDATA[` 会进入 CDATA section 并一直找 `]]>`，把后续所有内容（包括闭标签）吞掉，最后在 EOF 报错，报错位置远离真正的错误。
- CDATA 内部也不能出现 `]]>`，要拆成 `<![CDATA[ foo ]]]]><![CDATA[> bar ]]>`。
- 所以修复层自己包 CDATA 时必须拆分 `]]>`；判断"是否已有 CDATA"不能只看开头。
- 人类的倾向：内容以 CDATA 开头时默认全部交给 sax；未闭合标签无法区分合法与非法语义，不做防御。

## 否决的方向

- **重写 XML 解析器**：底层测试量太大。如果要重做，不如直接做流式编程语言解释器，没必要用 XML 语法。
- **`<|xxx|>` 标签**：会和各家 tokenizer 的 special token 冲突，行为由厂商决定、不可控；模型对它没有嵌套、属性、开闭配对的语法直觉；内容里出现 `<|...|>` 时依然要转义；还要自己写解析器。
- **字符串级扫描 `</chan:cmd>` 自动包 CDATA**：内容里包含自身闭标签时会提前截断；用栈匹配又会在闭标签写错时吞掉后续内容。边界不清晰。

## 留着的方向：HTML 词法 + CTML 结构校验

HTML 的 raw text element（`<script>` / `<style>` / `<textarea>`）不解析内容里的标签，读到匹配的结束标签为止，正是 `text__` / `chunks__` 需要的语义，而且有规范可依，不是启发式。

能逐块 feed 且能切 raw 模式的，Python 里实际只有标准库 `html.parser.HTMLParser`（`feed()` + 半私有的 `set_cdata_mode`）。lxml 的 `HTMLPullParser` 快但没有自定义 raw 模式；html5lib、selectolax、BeautifulSoup 都是整篇解析。

设想：用 `HTMLParser` 做 tokenizer，借用它对属性引号和裸 `&` 的宽容；配对检查和 fail-fast 留在自己的 handler（现有 element stack 已在做）；`text__` / `chunks__` 命中预传集合时走 `set_cdata_mode`。不引入 HTML5 的树修复语义，因为静默修复闭标签会让 scope 执行拓扑和模型意图对不上。

已知问题和待验证点：

- 内容里出现自身闭标签的问题依然存在（同 `<script>` 里的 `</script>`），但比忘记 CDATA 少见得多。
- 带冒号的标签名 `chan:cmd:cid` 能否正确切分。
- 标签名会被转成小写，通道名 / 命令名的大小写怎么处理。
- `set_cdata_mode` 是半私有接口，3.13 前后行为有调整，需要锁版本或写测试。
- 已有的 `<![CDATA[...]]>` 写法在 raw 模式下会成为字面字符串，需要在 delta 里剥掉以保持兼容。
- HTML 的可选闭合（`<option>`、`<li>`）只对内置元素生效，对自定义标签无效。

如果推进，第一步是一个几十行的原型验证以上几点，再决定是否开 feature。

## 设计启示

减少需要转义的嵌套层数，比优化转义语法更有效。比如让 `text__` / `chunks__` 尽量不携带 XML 片段；需要引用复杂字符串的场景，考虑走别的通道，不内联进 logos。
