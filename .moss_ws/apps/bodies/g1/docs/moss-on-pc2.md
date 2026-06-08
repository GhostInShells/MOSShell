# MOSS 装机记录 — G1 PC2

Jetson Orin NX 上安装 MOSS 的完整流程与问题日志。目标：可复现。

## 环境基线

| 项目 | 值 |
|------|-----|
| 硬件 | Jetson Orin NX |
| OS | Ubuntu 20.04.6 LTS (GNU/Linux 5.10.104-tegra aarch64) |
| CUDA | 11.4 (nvcc build 11.4.315) |
| 磁盘 | 2TB NVMe, 已用 21GB |
| 内存 | 15GB |
| Python 系统版本 | 3.8 |
| 默认用户 | `unitree / 123`（官方默认，装机后创建独立用户） |

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

# 防火墙：仅放行 SSH
sudo ufw allow 22
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

### 5. 部署 MOSS

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

---
