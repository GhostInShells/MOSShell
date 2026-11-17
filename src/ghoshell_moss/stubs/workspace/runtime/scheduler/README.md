# scheduler

Ghost 自己的时间规划。每个 ghost 一个子目录 `ghost-{name}/`，里面是：

```
state.yml          # 自增 id 的计数器
tasks/*.yml        # 一条计划一个文件，文件名就是 id
fired/YYYY/MM/     # 一次性计划的归档，保留 30 天
```

计划文件的真相在磁盘上：改一个文件、删一个文件都直接生效（`mindflow.scheduler:reload`
之后）。目录由 `SchedulerNucleus` 写入，也可以被 node 直接读来做渲染。

本目录不进版本库：`.gitignore` 里的 `*` 覆盖所有运行时内容，只保留本文件与忽略规则。
