# G1 硬件环境记录

## 硬件连接拓扑

（逐步记录）

## 网络拓扑

### 架构总览

G1 内部通过一台**交换机**组网，PC1、PC2、LiDAR 均接入交换机。外部设备通过以太网口连接交换机后，可与三者通信。

| 设备 | 内部 IP | 说明 |
|------|---------|------|
| 交换机 | — | G1 内部中枢，所有设备通过它互联 |
| PC1 | 192.168.123.161 | 运控专用。运行所有运控程序、蓝牙/WiFi 管理、宇树云 OTA。**不开放二开访问** |
| PC2 | 192.168.123.164 | Jetson Orin NX，开发者工控机。**唯一二开入口** |
| LiDAR | 192.168.123.120 | Mid-360，独立 IP |

### 关键关系

- **PC1 是网络入口**：通过蓝牙驱动 App → App 配置 WiFi → PC1 连接外网。宇树云 OTA、WebRTC、故障上报均走 PC1
- **PC2 默认无外网**：PC1 不开放时，PC2 只能通过交换机内网通信。需要**手动开启 WiFi** 才能直连外网
- **外部开发机进交换机**：用以太网线连 G1 交换机 → 配静态 IP → 可直接 SSH 到 PC2
- **默认凭据**: `unitree / 123`（官方默认，装完机应修改）

### 连接路径

**路径一：以太网进交换机（调试/装机）**
```
Mac ──(USB网卡)──→ 交换机 ──→ PC2 (192.168.123.164)
```
- Mac 需手动配静态 IP 进入 192.168.123.0/24 子网
- G1 内网无 DHCP，外部设备必须手动配 IP

**路径二：PC2 WiFi 直连（日常二开）**
```
Mac ──(WiFi 路由器)── PC2
```
- PC2 默认 WiFi 射频关闭，需先通过路径一登录开启
- 这是更便捷的日常二开方式，但绕过了交换机隔离

## PC2 规格

| 项目 | 内容 |
|------|------|
| 硬件 | Jetson Orin NX Developer Kit |
| L4T | R35.3.1 (2023-03-19) |
| OS | Ubuntu 20.04.6 LTS (kernel 5.10.104-tegra aarch64) |
| CPU | 8c ARMv8 @ 1.98GHz |
| 内存 | 16GB（可用 12GB） |
| 磁盘 | 1.8TB NVMe |
| 默认用户 | `unitree / 123`（出厂帐号，所有 ROS/DDS/CUDA 栈在此） |
| 加固用户 | `moss`（MOSS 应用运行帐号，环境需从 unitree 继承） |
| WiFi 网卡 | `wlan0`（默认关闭，需手动开启） |
| 以太网 | `eth0`，连接交换机，IP 192.168.123.164 |
| 路由 | wlan0 走外网（DHCP），eth0 走 G1 内网（123.0/24），按 metric 分流 |

### 双帐号范式（重要）

PC2 出厂只有 `unitree` 帐号，所有开发栈（cyclonedds_ws、unitree_sdk2-main、ROS、CUDA）齐备。加固时我们创建 `moss` 帐号作为 MOSS 应用运行身份，但 moss 帐号 shell 是干净的——所有跨帐号共享的工具栈必须以系统级方式暴露（`/etc/profile.d/` 或 `/usr/local/`），不在 moss 帐号下重复安装。这是 PC2 装机的核心范式。

## 连接流程

### 2026-06-07 — 首次实机连接

**物理准备**:
1. 吊架放置 G1，站立姿态
2. 安装电池
3. Mac 通过 USB 以太网 dongle 网线连 G1 交换机
4. 遥控面板开机

**开机步骤**:
1. 电源开关短按 → 亮灯
2. 长按 → 开机启动
3. 手机 App 通过蓝牙连接 PC1，配 WiFi → 已连接

**进入 PC2**:
1. Mac 配静态 IP: `sudo ifconfig en7 192.168.123.100 netmask 255.255.255.0`
2. SSH 登录: `ssh unitree@192.168.123.164` (密码 `123`)

**WiFi 开启**:
1. `nmcli radio wifi on` — 开启射频
2. `nmcli device wifi list` — 扫描可用 WiFi
3. `nmcli device wifi connect <SSID> password <密码>` — 连接
4. PC2 获得本地 WiFi IP，之后可走 WiFi SSH

**待办**:
- 路由器绑定 PC2 MAC 地址，固定 IP
- 配 WiFi 自启 systemd service（见 2026-02 设计文档）
- 创建独立用户帐号，修改默认密码

### 2026-06-14 — WiFi 自启与防火墙加固

**WiFi 自动重连**: 用 NetworkManager 持久 profile 替代命令式 `nmcli connect`。在路由器范围外不会"死"，NM 后台静默扫描，进了范围自动连上。

```bash
sudo nmcli connection add type wifi \
    con-name "<别名>" ifname wlan0 ssid "<SSID>" \
    -- wifi-sec.key-mgmt wpa-psk wifi-sec.psk "<密码>" \
    connection.autoconnect yes \
    connection.autoconnect-priority 100
sudo nmcli connection up "<别名>"
```

**辅助 service**（可选兜底，确保射频开 + 关省电）：`/etc/systemd/system/wifi-enable.service`
```ini
[Unit]
Description=Enable WiFi radio only
After=network.target nvwifibt.service
Wants=network.target
[Service]
Type=oneshot
ExecStart=/usr/bin/nmcli radio wifi on
ExecStartPost=/bin/sh -c 'sleep 2 && /usr/bin/iwconfig wlan0 power off || true'
RemainAfterExit=yes
[Install]
WantedBy=multi-user.target
```

**ufw 配置**:
```bash
# Jetson 内核缺 xt_rt 模块，必须先关 IPv6
sudo sed -i 's/^IPV6=yes/IPV6=no/' /etc/default/ufw
sudo ufw allow 22/tcp
sudo ufw allow from 192.168.123.0/24    # 内网整段，DDS/MOSS 通讯需要
sudo ufw enable
```

---

## 安全考量

### WiFi 直连的风险

PC2 通过 WiFi 直连外网**绕过了交换机隔离**。原本的架构中，PC2 与外部网络隔离开，外部只能通过 PC1 的蓝牙/WiFi 管理链路间接通信。开启 PC2 WiFi 后：
- PC2 直接暴露在局域网中
- PC2 可以出站到外网（安装依赖等），但入站也被开放
- **必须**：创建独立帐号、修改默认密码、UFW 仅放行必要端口

### 待加固项

| 事项 | 优先级 | 状态 |
|------|--------|------|
| 创建独立用户帐号，禁用或改密 unitree | 装机后立即 | 已完成（moss 用户） |
| UFW 仅放行 SSH (22) + 内网段 | 装机后立即 | 已完成（2026-06-14） |
| 考虑 SSH key-only 认证 | 日常使用前 | 待办 |

---

## 问题日志

| # | 问题 | 解决 |
|---|------|------|
| 1 | Mac 连交换机后拿 self-assigned IP (169.254.x.x)，ping 不通 PC2 | G1 内网无 DHCP，手动配静态 IP 192.168.123.100/24 |
| 2 | WiFi 连上后 PC2 无 SSH 入口 | PC2 WiFi 射频默认关闭，需以太网路径一进入后手动开启 |
| 3 | 扩展坞 USB 口不稳定 | 换口后正常 |
| 4 | `ufw enable` 报 `Couldn't load match 'rt': No such file or directory` | Jetson L4T 内核未编 `xt_rt` ip6tables 模块。改 `/etc/default/ufw` 把 `IPV6=yes` 设为 `IPV6=no` 即可，内网通讯本就是 IPv4 |
| 5 | PC2 在路由器下的 IP 与交换机管理 IP 混淆（误用 192.168.3.11 vs 实际 192.168.3.13） | 路由器内 PC2 是 DHCP 分配的设备 IP（如 3.13），192.168.3.x 段中可能另有交换机管理界面 IP（如 3.11，App 注册时用）。用 `nmcli -t -f IP4.ADDRESS dev show wlan0` 在 PC2 本地确认，或路由器后台按 MAC 查 |
| 6 | moss 帐号下 `uv sync` 失败 `Could not locate cyclonedds. Try to set CYCLONEDDS_HOME` | unitree-sdk2py 依赖的 cyclonedds==0.10.2 Python wheel 是 C 扩展薄绑定，需系统已有 cyclonedds C 库。unitree 帐号下已 build 在 `~/cyclonedds_ws/install/cyclonedds/`，对应版本 0.10.2 完美匹配。系统级共享方案见 `moss-on-pc2.md` 的 cyclonedds 跨帐号共享章节 |

---
