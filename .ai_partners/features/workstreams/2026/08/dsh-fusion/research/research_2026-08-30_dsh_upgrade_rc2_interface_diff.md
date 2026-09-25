# dsh 升级 0.1.1-rc.2 — interface diff 与图片/表单语义澄清

> 源码锚定、可回放、非推测。源码路径均相对 `research/source/deepseek-harness/`。
> 本次把 vendor 源码从 0.1.0-rc.5 升到 0.1.1-rc.2，diff interface 变动，并澄清
> 图片转换路径与 `ContextForm` 的消费点。是下次升级的基线。

---

## 一、升级信息

| 项 | 值 |
|---|---|
| 旧 | `0.1.0-rc.5` (`abe560f81e`) |
| 新 | `0.1.1-rc.2` (`b150a551b8`) |
| 跨度 | 855 commits |
| 绝对最新（未采用） | `0.1.2-alpha.1` (`cd5ef81481`)，1934 commits，08-28 |

dsh 需持续追版本——否则 MOSS 无法让使用者 vendor 一个版本下来。每次升级 diff
interface 变动即可，技术实现变动很快。

## 二、MOSS 侧对齐点（3 个，实际只需 0 个）

### 1. `assistant/message` 加 `interrupted?: true`（决定不补）

`core/session/src/types.ts:268-273`：turn 被 mid-stream cancel 时，已 delivered 的
text/reasoning 前缀作为 `assistant/message` 事件，带 `interrupted: true`。目的是让投影层
（`deriveMessages` 是 verbatim pass-through）不重新从 turn 边界推导中断。

**决定不补**。此字段是 dsh 中断记账的早期形态，未必稳定发布，MOSS 侧当前不消费，
不为此对齐。

### 2. `host.describe` 加 `home: string`（零影响）

`host/apiproxy/src/api/host.ts:50`：返回值新增 `home`（host account home directory，
Web 显示缩写）。MOSS 侧 `client.py:152` 有 `host_describe()` facade 但**全仓库零调用点**，
此字段连对齐都不需要。

### 3. `EncodedImageAttachment`（base64 上传，图片转换的关键）

见下节。

## 三、图片转换路径（base64 → attachment ref → 请求前投影）

`attachment/attachment/src/admission.ts`（rc.2 新增）+ `index.ts` 确认：

```
EncodedImageAttachment { mediaType, data: base64 string }   // wire 上传形式
  → decodeBase64 (拒绝非 canonical base64)
  → validateImageBatch (maxImagesPerMessage / maxMessageImageBytes / mediaTypes)
  → saveImage → ImageAttachmentRef { attachmentId: "sha256:...", ... }  // content-addressed
  → [请求模型前] ImageRequestPolicy → RequestImageAttachment (variantId + provider 兼容编码)
```

关键：`ImageBlock`（`llm/llm/src/types.ts:71`）只持 `ImageAttachmentRef`，**无 inline base64
入口**。MOSS 的 base64 图片必须 decode 后 `saveImage`（吃 `Uint8Array`，非 base64 字符串）
拿 content-addressed 引用。这是 `message_mapper.to_content_block` 对 image 抛
`NotImplementedError` 的根因——纯映射器没有 attachment service 句柄，造不出 ref。

## 四、imageLimits 超限 = 抛错拒绝，非 surface 清理

`validateImageBatch`（`attachment/index.ts`）在 **upload admission** 时强制上限，超了抛
`AttachmentError`：`TOO_MANY_IMAGES` / `IMAGES_TOO_LARGE` / `UNSUPPORTED_IMAGE_TYPE`。
不是"到上限自动在 surface 清理"。模型侧的图片上限是另一层 `ImageRequestPolicy`
（maxPixels/maxBytes），请求前投影/降采样。

## 五、ContextForm 的消费点：UI 渲染层，不进 prompt

`ContextForm`（`llm/llm/src/message.ts:48-60`）是语义标签，消费者是 **client UI**：

- `client/runtime/src/client/sessions/context-provenance.ts:113` `contextForm()` 读
  `source.form`，返回 `KnownContextForm`，不认识则降级 opaque。
- **模型侧不读 form**：`core/session/src/surface.ts:96` `deriveEventMessage` 对
  `user/message` 是 verbatim pass-through，只投影 `content`。

所以 form 不是"xml 容器 + 顶层 prompt 描述机制"，也不是 message 多态（多态在
`Message.role` + `ContentBlock.type`）。form 只决定"这条注入在人类 UI 里怎么折叠"。

**推论**：真正决定 memory 在模型上下文位置的，是 `surfaceOp`（append/replace）+ seq，
不是 form。memory 装线的核心是"append/replace 落到 surface 最前"，form 只是顺手皮肤。

## 六、compact 用 `plugin:'compact'`，不是 snapshot form

`compaction/compaction/src/checkpoint.ts:19`：`COMPACT_CHECKPOINT_MARKER = {kind:'plugin',
plugin:'compact'}`，靠 `isCompactCheckpointSource` 识别。与 `form:'snapshot'`
（`RuntimeContextProjection` 的 warm 层）是两个不同 producer、不同 source 标记。

所以"对话历史压缩的数据在 snapshot 里"不准确——压缩走 `plugin:'compact'`，
`form:'snapshot'` 是 runtime-context（warm 层）。memory（ground + 摘要）对齐哪个
结构需单独裁决，不是默认 snapshot。
