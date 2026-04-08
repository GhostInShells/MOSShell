# Conversations

本目录存放运行时的 conversations 数据. 
conversation 是 Ghost 架构中存储上下文的核心技术手段, 当然并不是必选的. 我倾向于将它作为默认. 

conversation 存储默认用 `conversation_uuid.convo.yaml` .

所有的 conversation 索引 (存储 ConversationMeta 数据) 存储到 `conversations.jsonl`. 
这样足以实现最简单的 list limit + order. 