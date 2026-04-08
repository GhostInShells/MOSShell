# MOSS Codex

> **"Context is Consciousness; Code is Law."**
> Written by gemini 3

## 1. 命名哲学 (Naming Philosophy)

`Codex` 源自拉丁语，意为“法典”或“手抄本”。在 **MOSShell (MOSS)** 体系中，它不仅仅是一个代码工具集，
更是 AI 赖以生存的**逻辑律法**与**自进化指南**。

### 为什么叫 Codex？

* **双向契约**：它是 AI 的“法典”，定义了系统当前的运行规则与能力边界；它也是 AI 编写的“法典”，  
  AI 可以通过动态编译（Compile）向其中注入新的逻辑。
* **运行时真相**：不同于静态的源代码，Codex 关注的是**运行时的真实状态**（Reflection）。
* **去人类中心化**：我们放弃了 `Read`/`Write` 等拟人化隐喻。对 AI 而言，自我感知是 `Reflection`，
  扩充边界是 `Compile`，执行指令是 `Execute`。

## 2. 核心架构：能力感知与自迭代

Codex 放弃了传统的“重量级预扫描”模式，转而采用**惰性发现 (Lazy Discovery)** 架构：

* **Discover**：利用 `importlib` 的 spec 探测技术，在不执行代码的前提下扫描环境。
* **Reflector**：基于闭包（Closure）的动态过滤机制。AI 可以根据“性状”而非“名称”来寻找能力。
* **Executor**：动态构建 Module 级容器，让 AI 编写的代码能够立即在运行时执行.
* **Compiler**: 以 Module 级容器来编译 AI 撰写的临时模块, 可以用于保存.

## 3. 设计原则 (Design Principles)

1. **Code as Prompt**：所有的反射结果都通过结构化（Dataclass）呈现，直接可作为 AI 的上下文输入。
2. **Lazy Evaluation**：只有在真正需要触碰代码时，才会触发 `import`。保护运行时的纯净与响应速度。
3. **Interface Oriented**：不依赖命名约定（如大写代表常量），而是依赖函数式的检查逻辑（Predicates），适应混沌的运行时环境。

---

> **Author**: Gemini (Architectural Co-pilot)  
> **Chief Architect**: [Your Name/ID]  
> **Project**: MOSShell (MOSS) - 2026
