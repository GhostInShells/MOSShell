# Conversation 设计讨论总结

## 讨论背景
用户与AI协作者对 `contracts/conversation.py` 的架构进行了深入讨论，目的是明确Conversation设计的核心理念、实现策略和后续开发路径。

## 设计核心理念

### 1. 消息时序四层设计
ConversationTurn 内部划分为四个时序层次，模拟AI实时推理思维管道：
- **context**: 回合开始时的动态上下文快照，不进入对话历史，用于记录思维过程中的临时信息
- **inputs**: 回合中所有输入信息，进入对话历史
- **instruction**: input之后的指令片段，不进入对话历史，用于指导AI行为
- **generates**: AI生成的所有消息，需要被添加到记忆中，但不一定是最终输出

这种设计支持流式思维过程的细粒度记录，为AI反身性提供基础设施。

### 2. Recap双重性设计
Conversation 层面的recap和ConversationTurn级别的recaps形成双重设计：
- **Conversation.recap**: 顶层永久锚点，创建时设置，对话历史裁剪时永远保留
- **ConversationTurn.recaps**: 回合级策略化裁剪，为不同ConversationStrategy存储不同版本的前情提要

这种设计支持AI反身性切换记忆窗口，实现灵活的记忆管理策略。

### 3. 线程安全哲学
采用Rust式责任分离思想：
- 数据结构保持简单，不内置复杂的线程安全机制
- 使用方通过copy/fork保证线程安全
- 存储层（ConversationStore）负责线程安全和有序持久化

这种设计简化了核心数据结构，将并发责任明确分离到使用场景。

### 4. 存储策略
- **持久runtime内存优先**: Conversation在运行时内存中保持完整状态
- **异步线性保存**: 通过ConversationStore实现异步、线程安全的保存
- **save last one策略**: 最终一致性，保存最后一个有效状态

避免过早优化，先保证基础流程跑通。

### 5. ConversationStrategy基础设施
为AI反身性提供基础设施：
- 允许不同的策略读取和优化Conversation
- 支持配置化，未来可Channel化
- 实现AI对自身记忆管理的控制权

## 已识别和修复的问题

1. **类型错误修复**: 修复了方法签名和类型注解
2. **拼写错误修正**: 修正了变量名和字段引用
3. **字段引用统一**: 统一使用`turn_id`而非`id`等不一致引用
4. **未实现字段移除**: 移除了未实现的`variables`字段引用
5. **方法参数修正**: 修正了`new_turn`方法中`instructions`参数名

## 实现路径共识

### 渐进实现策略
1. **先跑通基础**: 实现核心数据结构的基本功能
2. **再迭代高级**: 逐步添加策略、优化、存储等高级功能
3. **避免过度设计**: 当前设计不会导致未来大重构

### 当前完成状态
- ConversationTurn 基本功能完整
- Conversation 核心逻辑实现
- ConversationStrategy 抽象定义
- ConversationStore 接口定义

### 待实现功能
1. **基础存储实现**: 实现简单的ConversationStore
2. **策略具体实现**: 实现具体的ConversationStrategy
3. **测试验证**: 编写测试验证核心流程
4. **优化算法**: 实现get_truncated_copy等优化方法

## 技术决策记录

### 数据结构设计决策
- 使用Pydantic BaseModel: 提供序列化、验证、复制等基础能力
- 使用WithAdditional混入: 支持扩展字段
- 字段默认工厂: 使用uuid、timestamp_ms等保证唯一性和时序

### 线程安全决策
- 核心数据结构不内置锁: 避免过度复杂化
- 使用copy/fork: 提供明确的并发控制点
- 存储层负责最终一致性: 明确责任边界

### 存储策略决策
- 内存优先: 简化运行时逻辑
- 异步保存: 不影响主流程性能
- 最终一致性: 接受短暂的数据不一致

## 协作模式确认

讨论确认Plan Mode不适合当前协作方式，回归直接协作模式：
1. 人类工程师提供关键抽象设计
2. AI协作者参与讨论并提供专业意见
3. 抽象确定后快速对齐具体实现
4. 根据具体计划实现功能

## 后续行动建议

### 短期行动（立即）
1. 实现基础ConversationStore（如内存存储）
2. 编写简单测试验证核心流程
3. 修复剩余逻辑错误（如get_truncated_copy占位）

### 中期行动
1. 实现具体的ConversationStrategy
2. 完善优化算法
3. 集成到Ghost框架中

### 长期考虑
1. 性能优化和存储策略升级
2. 分布式支持
3. 高级策略实现

---
*讨论时间: 2026年3月12日*
*参与方: 人类工程师 + AI协作者*
*文件位置: `contracts/conversation.py`*
*总结保存: `contracts/.discuss/conversation_design.summary.md`*