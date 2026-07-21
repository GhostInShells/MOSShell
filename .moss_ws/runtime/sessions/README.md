# Sessions

MOSS 运行时会开辟存储空间, 用来保存持久化的数据. 
系统提供不同的隔离级别, 用来存储和发现这些持久化的数据. 

默认的隔离级别有: 

- session_scope: 按指定的 scope 进行数据隔离, 不同 scope 下的 session 全部不互通. 
- session_id: 按会话本身的 id 来进行隔离. 每次重新运行时不指定 id, 就会产生不同的隔离环境. 
