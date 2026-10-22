---
state: waiting
priority: 1.5
tags: [example]
links: [from-spark-to-done:related, a-file-that-was-deleted:depend]
---

waiting 和 done 别混：一个是"卡在别处"，一个是"这件事结束了"。混起来，
清单就会假装很干净。

（上面那条 depend 指向一个已经不存在的文件——渲染时会画成断头线。）
