from .channel import (
    Builder,
    Channel,
    ChannelBroker,
    ChannelFullPath,
    ChannelMeta,
    ChannelPaths,
    ChannelProvider,
    ChannelUtils,
    CommandFunction,
    ContextMessageFunction,
    LifecycleFunction,
    PrompterFunction,
    R,
    StringType,
)
from .command import (
    RESULT,
    BaseCommandTask,
    CancelAfterOthersTask,
    Command,
    CommandDeltaType,
    CommandDeltaTypeMap,
    CommandError,
    CommandErrorCode,
    CommandMeta,
    CommandTask,
    CommandTaskStack,
    CommandTaskState,
    CommandTaskStateType,
    CommandToken,
    CommandTokenType,
    CommandType,
    CommandWrapper,
    PyCommand,
    make_command_group,
)
from .errors import CommandError, CommandErrorCode, FatalError, InterpretError
from .interpreter import (
    CommandTaskCallback,
    CommandTaskParseError,
    CommandTaskParserElement,
    CommandTokenCallback,
    CommandTokenParser,
    Interpreter,
)
from .shell import (
    InterpreterKind,
    MOSSShell,
)
from .speech import (
    TTS,
    AudioFormat,
    BufferEvent,
    ClearEvent,
    DoneEvent,
    NewStreamEvent,
    Speech,
    SpeechEvent,
    SpeechProvider,
    SpeechStream,
    StreamAudioPlayer,
    TTSAudioCallback,
    TTSBatch,
    TTSInfo,
)
from .states import MemoryStateStore, State, StateBaseModel, StateModel, StateStore
from .topics import *

"""
基于代码完成自解释的思路, 定义了 MOSS 架构中所有的关键抽象. 

当前的模块, 所有的抽象设计可以通过 ghostos 的 prompter 机制自动反射出来. 尚未实装到 ghoshell. 

简单解释一下设计思想: 

1. command: 基于 code as prompt 思想, 可以将任何语言的函数定义成一个面向模型的 python async 函数,
            模型可以用代码方式理解.
            这是一种面向模型的胶水语言思路. 不过现阶段只做到了函数级别.
            在 "面向模型的高级编程语言" 思想中, command 对应了模型可用的 "函数".
            
2. channel: 为一组 command 提供一个控制单元, 可以对大模型表征所有的 command, 也封装了通讯协议用来调用它们.
            channel 本身支持树形嵌套, 原理和 python 中一个 module import 另一个 module 一样. 
            在 "面向模型的高级编程语言" 思想中, channel 对应了类似 python module 的 "模块".
            
3. shell:   提供一个可以持续运行的 runtime, 用来执行模型所有下发的 command 指令. 
            同时维护多轨 的 并行/阻塞 生命周期.
            shell 的核心职责是持续调度 command 分发, 并且双工地拿到 command 的返回值. 

4. interpreter: 用来将大模型的流式输出, 解析成 CommandTask 对象 (对标 python 中的 coroutine), 输入给 shell.

5. errors:  在 MOSS 架构中通用的异常处理机制. 定义不同级别的异常, 用来做故障恢复. 
            预设的异常至少有四种类型: 
            - 可忽略的异常, 不打断模型的一轮输出执行. 
            - 解释级别的异常, 立刻中断模型的一轮输出, 并且提示模型输出有错误. 
            - 会话级别的异常, 由于错误会导致 agent 无法继续持续, 所以需要删除掉致命的交互轮次, 用错误提示取代.
            - 致命异常, 错误会导致整个 AI 运行失败. 可能必须强制停止, 或者做灾难性遗忘, 消除掉相关记忆.  
            目前 errors 模块设计未完成, 预计在 beta 版本中完善. 
            
6. speech:  在 AI 的输出中最重要的是自然语言的输出, 而且这些输出通常要转化为语音.
            考虑到 realtime actions 中, AI 的输出是语音和动作交替的,
            shell 必须要感知到一段语音已经播放完, 再执行后面的动作.
            同时考虑到主流模型无法直接输出语音 item, 还需要走 流式或非流式的 tts
            这些功能点合并到一起, 就需要定义一个特殊的 speech 对象实现.
            
            预计在某个正式版本中, 彻底废除 speech 模块, 使用普通的 channel 来替代它. 
            
7. states:  一种多个 channel 共享的状态广播机制, 可以用前端 vue/react 框架的 state 去理解它. 
            当大模型修改了某个 state 数据结构时, 会广播给所有监听这个 state 的 channel, 从而变更对应行为.
            举个简单的例子, 当模型选择 "情绪低落" 时, 所有的肢体轨道都应该对这个状态做反应. 

8. topics:  alpha 版本未完成的实验性功能. 预计 channel 之间可以通过 topic 进行状态通讯. 
            可以理解为 ros/ros2 体系的 topic 对象. 
            一个视觉的 channel 可以广播 "注意对象" 的相对座标, 驱动其它软件比如数字人的 channel 调整面部朝向.
            在 MOSS 架构下的 Topic 帧率应该没有 ros2 高 (ros2 基于 dds 分发, 而 MOSS 基于云端 mqtt 广播)
            只要做到符合大模型思考的秒级频率即可. 
            这个功能预计在 beta 版以后再逐步实现. 
"""
