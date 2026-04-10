# 关于 Sessions

本目录存放运行时生成的 Session 数据.
本质上每次 Ghost 运行的时候, 都应该生成一个新的 Session, 用来隔离存放运行时可能产生的各种临时数据. 这些数据只在 Session
中存在.

# session 子目录

`runtime/sessions` 目录通过子目录隔离不同的 session 上下文.

MOSS 的 session 子目录按 `session_uuid` 的方式约定存储.
所有 session 的索引通过 `sessions.jsonl`, 这样可以 tail / list. 在人力有限的情况下, 放弃做任何复杂的数据库实现. 
