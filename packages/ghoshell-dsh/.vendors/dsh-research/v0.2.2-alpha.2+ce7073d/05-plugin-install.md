# 05 — dsh plugin 是怎么装进去的

## 动机

要在自己起的 dsh 里加一个插件（host 半边 + **browser 半边**），每次都要重新查一遍"插件放哪、
profile 怎么认"。这里把机械部分钉死。写给三个月后的自己。

## 路径（可重放）

```sh
cd .vendors/dsh-research/v0.2.2-alpha.2+ce7073d/source

# profile 的形状：bundles → patch → --patch overlay
sed -n '1,40p' packages/boot/app-boot/README.md
# 现成范例（我们自己的 ghost）：
cat src/ghoshell_moss/ghosts/dolores/dsh_stubs/profiles/web/cordis.patch.yml
cat src/ghoshell_moss/ghosts/dolores/dsh_stubs/profiles/web/package.json
cat src/ghoshell_moss/ghosts/dolores/dsh_stubs/profiles/web/cordis.yml
grep -n "_sync_dsh_plugin" -A 12 src/ghoshell_moss/ghosts/dolores/_runtime.py
```

实盘目录：`$DSH_HOME/profiles/web/`（本机 ghost home 的是 `.moss/ghosts/moss/.dsh/profiles/web/`）

## 结论

- `[读码]` **profile 是一个目录**：`$DSH_HOME/profiles/<name>/`。名字 `web` 对 `dsh --profile web`。
- `[读码]` **`package.json` 只做一件事**：`dsh.profile.bundles = ["@deepseek-ai/dsh-base",
  "@deepseek-ai/dsh-web-app"]` —— 决定用哪套组合包。**"一次只能一个 web profile"指的是这套
  bundles + 一份 dist，不是"只能装一个插件"**。
- `[读码]` **组合顺序**：`bundles` → `cordis.patch.yml` → `--patch` 覆盖。`cordis.yml` 本身是空数组，
  写着 "Edit cordis.patch.yml, not this file."
- `[读码]` **`cordis.patch.yml` 两个动作**：`insert`（`{id, name}`，`name` 可以是相对本目录的
  `./xxx.ts`）和按 `id` 的 `disabled: true`。dolores 就是靠后者压掉 `ui-deliverables` 与
  `file-reference-local`。
- `[读码]` **host 半边可以是一个裸 `.ts` 文件**：`_sync_dsh_plugin()` 把
  `dsh_plugin/moss-dolores-ghost-plugin.ts` 拷到 `profiles/web/` 下，patch 里用
  `name: './moss-dolores-ghost-plugin.ts'` 插入。不需要打包、不需要 npm 发布。
- `[读码]` **profile 目录本身是 pnpm workspace**（`pnpm-workspace.yaml`: `packages: ['.']`,
  `nodeLinker: hoisted`）。所以**装一个真正的包**的方式就是：把它挂进 profile 的
  `dependencies`，在 profile 目录里 `pnpm install`。
- `[读码]` **browser 半边要一个有 package.json 的"包根"，但不需要 npm/pnpm 安装。**
  `dsh-client-modules` 的 `locatePkgJson` 明确把 `./`、`file:`、绝对路径列为 `pathLike`；
  对 pathLike 它不算 expectedPackageName，而是从解析出的模块 URL **往上找第一个 package.json**
  （`nearestPackage`），读它的 `dsh.client` 和 `exports["./client"]`。
  所以：**在 profile 目录里建一个子目录、给它自己的 package.json，就够了。**
  实测旁证：`profiles/web/node_modules/` 是空目录，dolores 的 profile 从没跑过 `pnpm install`。
- `[读码]` **为什么 dolores 那个插件没有 client 半边**：它的 `.ts` 直接躺在 `profiles/web/` 下，
  往上找到的第一个 package.json 就是 `profiles/web/package.json`，那里没有 `dsh.client`，
  于是 `resolveMeta` 返回 null —— 不是 client row。
- `[实测]` 纯脚本起法跑通：`DshLauncher(DshLauncherConfig(port=3084, home=Path('/tmp/...')))`，
  `async with` 返回即可用，`launcher.token()` 拿到 token，`launcher.config.base_url` 是 `http://127.0.0.1:3084`。
  `readiness_path` 字段**声明了但没被使用** —— 就绪是 push 式（mux WS 连上）。
- `[实测]` `launcher.call(path, payload)` 的 `path` **必须带前导斜杠**。写 `"session/list"` 会拼成
  `http://127.0.0.1:3084session/list` 并抛 `InvalidURL: Invalid port: '3084session'`。

## 观点

装一个**带 browser 半边**的插件，机械步骤只有两步，**不用 pnpm、不用发布**：

```
$DSH_HOME/profiles/web/
  cordis.patch.yml
  moss-bridge/            ← 自己的目录，就是它的"包根"
    package.json          ← {name, type:"module", exports:{".":"./host.js", "./client":"./client.js"},
                             dsh:{client:{platform:"web"}}}
    host.js
    client.js
```

```yaml
# cordis.patch.yml
- insert:
    - id: moss-bridge
      name: './moss-bridge'
```

**形状上唯一必须记住的一条**：`nearestPackage` 只认**往上遇到的第一个 package.json**（pathLike
时不算名字）。所以**插件目录必须自带 package.json**；把 `.js`/`.ts` 直接扔在 `profiles/web/` 下
会认领到 profile 自己的 manifest，然后被判成"不是 client row"。

`pnpm` 只在一种情况下需要：插件依赖第三方 npm 包（要在 profile 里解析到）。自己写的、只 import
dsh 已有包的插件不需要。

**失效条件**：
- 若上游把 `dsh.client` 从 package.json 挪到别处（如 profile 级声明），"自带 manifest"这条要重读。
- 若 `nearestPackage` 改成严格匹配包名（pathLike 也算一个期望名），相对目录就得改成包名。
- 若 `app-boot` 的 patch 语义改成"只能 disable 不能 insert"，dolores 的插法就作废。
