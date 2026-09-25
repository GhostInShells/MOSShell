# backdrop/ — 通用背板图

背板是页面自己的图层（一张大图铺在模型后面），与模型包解耦（KD6）。放 `.png/.jpg/.webp`
到这里，页面与 `set_backdrop` 命令就能用它。

- 图片不入库（本地测试资产，见 node 根 `.gitignore`）。
- 启动时 node 枚举这里的图片，自动映射命令面里就会出现 `set_backdrop(name)`。
- 换背板：CTML `<avatar:set_backdrop name="foo.png" />`；或形象作者在 channel.py 里调
  `avatar.set_backdrop("/backdrop/foo.png")`。
