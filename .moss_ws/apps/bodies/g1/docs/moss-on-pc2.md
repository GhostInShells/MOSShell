# MOSS 装机记录 — G1 PC2

Jetson Orin NX 上安装 MOSS 的完整流程与问题日志。目标：可复现。

## 核心范式：双帐号下的环境继承

PC2 出厂只有 `unitree` 帐号，所有开发栈（cyclonedds_ws、unitree_sdk2-main C++ 版本、ROS、CUDA）齐备且 `.bashrc` 已配好。加固时我们创建 `moss` 帐号作为 MOSS 应用运行身份，但 **moss 帐号 shell 完全干净**——cyclonedds 不在 PATH/LD 里、CUDA 环境变量没继承、Python 工具链也要自建。

**装机原则**：跨帐号共享的工具栈（特别是 DDS、CUDA、ROS）必须以系统级方式暴露（`/etc/profile.d/*.sh` 或 `/usr/local/`），不在 moss 帐号下重复安装。MOSS-specific 的环境（uv、venv、ghoshell-moss）才在 moss 帐号下独立管理。

这个范式的反面教训：开始很容易认为"PC2 没装开发栈，需要补"，实际是"换了帐号、看不见"。每发现一个 import 失败，先去 unitree 帐号 `find` 一下，大概率已经存在。

## 环境基线

| 项目 | 值 |
|------|-----|
| 硬件 | Jetson Orin NX Developer Kit |
| L4T | R35.3.1 |
| OS | Ubuntu 20.04.6 LTS (kernel 5.10.104-tegra aarch64) |
| CUDA | 11.4 (nvcc build 11.4.315) |
| 磁盘 | 1.8TB NVMe |
| 内存 | 16GB（可用 12GB） |
| Python 系统版本 | 3.8 |
| 出厂用户 | `unitree / 123` |
| 加固用户 | `moss`（应用运行身份） |

## 安装步骤

### 0. 连接 PC2

PC2 通过 G1 内部交换机连接，WiFi 默认关闭。首次连接需用以太网进入：

```
Mac ──(USB 网卡)──→ G1 交换机 ──→ PC2 (192.168.123.164)
```

```bash
# Mac 端：手动配静态 IP 进入 G1 内网（无 DHCP）
sudo ifconfig en<N> 192.168.123.100 netmask 255.255.255.0
ssh unitree@192.168.123.164
# 密码: 123（官方默认）
```

### 1. 安全加固（首次登录后立即执行）

```bash
# 创建独立用户
sudo adduser moss
sudo usermod -aG sudo moss

# 防火墙：仅放行 SSH + 内网段
# Jetson 内核缺 xt_rt 模块，必须先关 IPv6
sudo sed -i 's/^IPV6=yes/IPV6=no/' /etc/default/ufw
sudo ufw allow 22/tcp
sudo ufw allow from 192.168.123.0/24   # DDS/MOSS 内网通讯需要
sudo ufw enable
```

### 2. WiFi 配置

```bash
# 开启射频
sudo nmcli radio wifi on

# 扫描并连接
nmcli device wifi list
sudo nmcli device wifi connect <SSID> password <密码>

# 验证
ip addr show wlan0 | grep "inet "
```

WiFi 自启 service 配置（跨重启保持）：

```bash
sudo tee /etc/systemd/system/wifi-enable.service << 'EOF'
[Unit]
Description=Enable WiFi radio only
After=network.target nvwifibt.service
Wants=network.target

[Service]
Type=oneshot
ExecStart=/usr/bin/nmcli radio wifi on
ExecStart=/usr/bin/iwconfig wlan0 power off
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable wifi-enable.service
```

### 3. 网络依赖源

中国大陆环境按需配置 APT/pip 镜像源（阿里云、清华等），此处从略。NVIDIA Jetson 相关源（`/etc/apt/sources.list.d/`）不应替换。

### 4. Python 工具链

```bash
# 安装 uv（推荐通过 pipx 隔离）
sudo apt install python3.8-venv   # uv 依赖 venv 模块
pip3 install pipx                  # 如未预装
# pipx 不继承 pip 源配置，国内按需 export PIP_INDEX_URL=<镜像>
pipx install uv
```

### 5. cyclonedds 跨帐号共享

unitree-sdk2py 依赖 `cyclonedds==0.10.2` Python 包，它是 C 扩展的薄绑定，构建时需要找到 cyclonedds C 库（`libddsc.so` + 头文件）。Unitree 出厂在 `unitree` 帐号下已完整 build 了 cyclonedds 0.10.2：

```
~/cyclonedds_ws/
├── src/cyclonedds-0.10.2/    # 源码
├── build/cyclonedds/lib/      # build artifacts
└── install/cyclonedds/        # install prefix（CYCLONEDDS_HOME 指向这里）
    ├── lib/libddsc.so.0.10.2
    └── include/dds/dds.h
```

版本 0.10.2 与 Python wheel 钉死的版本完美对齐——不需要自己 build。系统级共享配置：

```bash
sudo tee /etc/profile.d/cyclonedds.sh << 'EOF'
export CYCLONEDDS_HOME=/home/unitree/cyclonedds_ws/install/cyclonedds
export LD_LIBRARY_PATH=$CYCLONEDDS_HOME/lib:${LD_LIBRARY_PATH:-}
EOF

# 当前 session 立即生效（新 SSH session 自动 source）
source /etc/profile.d/cyclonedds.sh
echo $CYCLONEDDS_HOME
```

这之后任何帐号的任何 venv 都能 build/import cyclonedds，无需重复安装。`uv sync` 时 wheel 的 setup.py 会读 `CYCLONEDDS_HOME` 找头文件链接。

**版本核验**: `find ~/cyclonedds_ws/src -name "dds.h"` 路径中应含 `cyclonedds-0.10.2`。如果未来升级且版本漂移，调整 unitree-sdk2py 或自建 0.10.2。

### 6. 部署 MOSS

```bash
# 以 moss 用户操作
sudo su - moss

# 创建仓库目录（与 GitHub 路径保持一致）
mkdir -p ~/github.com/GhostInShells/MOSShell
cd ~/github.com/GhostInShells/MOSShell

# 初始化为可 push 的 Git 仓库
git init
git config receive.denyCurrentBranch updateInstead

# 回到 Mac：推送代码
# git remote add g1 moss@<PC2_WiFi_IP>:/home/moss/github.com/GhostInShells/MOSShell
# git push g1 dev

# PC2 上检出并同步依赖
git checkout dev
uv python install 3.11
uv sync --active --all-extras
```

## 问题日志

| # | 问题 | 解决 |
|---|------|------|
| 1 | pipx 安装 uv 后未进入 PATH | `python -m pipx install uv` 可运行 |
| 2 | `uv sync` 需要 venv 模块 | `sudo apt install python3.8-venv` |
| 3 | 系统 Python 3.8，moss 项目需要 3.11 | `uv python install 3.11` 自动下载 aarch64 预编译版本 |
| 4 | pipx install 走 PyPI 较慢 | pipx 不继承 pip 源配置，需 `export PIP_INDEX_URL=<镜像>` 后运行 |
| 5 | uv Python 下载太慢或 404 | uv 有独立镜像体系：`UV_INDEX_URL`（PyPI 包）和 `UV_PYTHON_INSTALL_MIRROR`（预编译 Python）。后者镜像路径可能不兼容，自行验证 |
| 6 | Claude Code npm 安装失败 (`install.cjs` exit 1) | 可能 linux-arm64 无预编译二进制。开发模式仍是 Mac 端写代码 → `git push g1` |
| 7 | `ufw enable` 失败：`Couldn't load match 'rt'` | Jetson L4T 内核缺 `xt_rt` ip6tables 模块。`sed -i 's/^IPV6=yes/IPV6=no/' /etc/default/ufw` 后重 enable |
| 8 | g1 app `uv sync` 失败：`Could not locate cyclonedds. Try to set CYCLONEDDS_HOME` | unitree 帐号下 `~/cyclonedds_ws/install/cyclonedds/` 已 build 0.10.2。配 `/etc/profile.d/cyclonedds.sh` 系统级共享（见上文步骤 5）|
| 9 | "PC2 看起来没装开发栈"误判（初次跑 `dds/01_env_check.sh` 报缺 cyclonedds） | 实际是换了 moss 帐号、看不见 unitree 出厂栈。每发现 import 失败先去 unitree 帐号 `find` 一下 |

---
