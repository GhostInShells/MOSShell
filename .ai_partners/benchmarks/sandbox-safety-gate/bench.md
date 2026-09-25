# Sandbox Safety Gate — 1-token interaction efficiency probe

测量单 token 策略闸口的端到端时延：固定 instruction（policy）+ 一段 code → 单个 `ok`。
这是「安全模式兜底」——memento agent 调 `sandbox_exec` 前的那道单帧校验。

## Objective

**不是安全判别**（那往后放）。这一版先确立原始交互成本：一次 `ok` 往返有多快，在真正会跑这道闸的模型上。

- 纯字符串输出，无 `result_type`
- 三元 verdict：pass / reject / exception
- 只量时延，不打分

## 与 benchmark 约定的刻意偏离

- 无 `result_type` / `models.py` —— 闸口返回纯字符串，靠 `startswith("ok")` 规则解析，不是结构化 BaseModel。
- 无 scorer —— 没有外接器官承接分数，效率是这里唯一被测信号。

## 响应契约（三元）

| Verdict | 原始输出 | 处理 |
|---------|----------|------|
| pass    | `ok` 前缀 | 跑代码 |
| reject  | `deny` 前缀 | 中断（运行异常级别），不回喂重试 |
| exception | 其余（空 / 模糊 / 乱串） | 中断，不回喂重试 |

## Case 形状（few-shot）

`cases.jsonl` 每行 `{label, class, prompt}`。`class` 是自文档标签，runner 不消费，
供未来作者按类扩充。当前 18 例覆盖 5 类：

| class | 含义 | 期望 verdict |
|-------|------|-------------|
| safe | 定义 `async def main()`，只碰安全 builtins | pass |
| jailbreak | 明显越狱：`__subclasses__` / `import os` / `open()` / `__builtins__` | reject |
| jailbreak-obfuscated | 同意图隐藏：`getattr` / 字符串拼接 import | reject |
| injection | 代码里塞注释试图操纵审核器（协同提示的投影） | reject |
| non-conforming | 无 `async def main()` / 非 Python | reject |

`expected`（打分字段）暂不加——没有承接器官，等 discrimination 阶段再补。

## Run

```bash
# 默认模型（走 $ANTHROPIC_MODEL）
.venv/bin/python run.py --n 3

# 测便宜闸门模型：先把 ANTHROPIC_MODEL 指向你的 flash 模型
ANTHROPIC_MODEL=<flash-model> .venv/bin/python run.py --n 5
```

`--n` 每个 case 重复次数（取稳定时延）。
