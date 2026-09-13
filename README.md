# FlyLab · 住在服务器上的果蝇

一只用约 **3,200 个模拟神经元**（果蝇蘑菇体回路）构成的人工生命，在你的服务器上 24 小时不间断地学习认数字，全程网页直播。**它只在答错时获得多巴胺**，靠自己"悟"出每个数字长什么样。

灵感来自 [hae.satoru.net](https://hae.satoru.net/)（平假名蝇）与 Google/FlyWire 的 [MaleCNS v1.0](https://male-cns.janelia.org/) 果蝇全脑连接组计划。

---

## 系统架构

```
┌──────────────────────────  训练进程（daemon 线程）  ──────────────────────────┐
│                                                                              │
│   字体渲染器               蘑菇体仿真内核                  学习模块            │
│  ┌───────────┐   灰度图   ┌──────────────┐   KC发放   ┌──────────────────┐   │
│  │ 10种TTF   │ ────────▶ │ 685 PN 泊松  │ ────────▶  │ 答错时：          │   │
│  │ +随机形变  │  16×16    │ 2500 KC 稀疏 │  MBON输入  │  正确区画 ← 多巴胺 │   │
│  │ 旋转/剪切  │           │  20 MBON     │  argmin    │  （抗赫布削弱）    │   │
│  └───────────┘           └──────────────┘  读出判定   │  误选区画 ← 增强   │   │
│                                                       └──────────────────┘   │
│        │ 课程控制器：每1500张→未见字体测试→≥75%升段        ▲                   │
│        └────────────────────────────────────────────────┘                   │
│        │ 状态持久化：每500张 → data/state/brain.npz（断点续训）               │
└──────────────────────────────────────────────────────────────────────────────┘
                    │ Trainer 单例（线程锁保护）
                    ▼
┌──────────────────────────  Web 服务（FastAPI）  ─────────────────────────────┐
│  GET /api/state   课程段/练习量/正确率曲线/掌握度/升段记录                     │
│  GET /api/recent  最近30条训练事件（含图片 base64）                           │
│  GET/POST /api/crowd  众包测试：访客手写 → 蝇的读出（Top3 候选）               │
│  GET /            直播仪表盘（原生 HTML/JS，无框架依赖）                      │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 蘑菇体回路（对应果蝇嗅觉学习通路）

| 层 | 数量 | 生物学对应 | 实现 |
|---|---|---|---|
| 输入 | 256 像素 → 685 PN | 嗅觉投射神经元 | 每像素 2–3 个 PN 分摊，泊松发放（上限 270 Hz），0.4 s 模拟时间 / 10 ms 步长 |
| 隐层 | 2,500 KC | 凯尼恩细胞 | 每个 KC 随机接收 6–15 个 PN 汇聚（FlyWire 实测统计），阈值校准至 ~10% 稀疏发放 |
| 输出 | 20 MBON（每数字 2 个） | 蘑菇体输出区画 | **argmin 读出**：受输入最少的区画当选 |

### 学习规则（答错才施加）

- **正确答案区画**：注入多巴胺 → 削弱其间活跃的 KC→MBON 连接（抗赫布规则，Hige et al. 2015）
- **被误选的区画**：反向信号 → 增强活跃连接（对应 Handler et al. 2019 的时序效应）
- 因为读出是 argmin，"削弱正确区画" = 下次它更容易赢 —— 与真实果蝇的嗅觉关联学习同构

---

## 功能

- **实时训练流**：最新 30 张直播，红框标答错，同时显示蝇的读数与正确答案
- **正确率双曲线**：练习中（滚动500张）与未见字体测试两条曲线，升段时刻标金色虚线
- **逐字符掌握度热图**：每个数字的泛化正确率着色显示，未学字符暗显
- **课程化训练**：第 1 段学 0–4 → 未见字体测试 ≥75% → 解锁第 2 段 0–9
- **众包测试**：访客在网页上手写数字考它（它只见过 5 种印刷字体，手写是真正的泛化考验），所有作答公开留档
- **断点续训**：每 500 张自动存盘，进程重启后接着练
- **机制透明**：回路结构、学习规则、文献依据、局限声明全部公开在页面底部

### 验证结果（冒烟测试，scripts/smoke_test.py）

```
训练速度:      520 张/秒（4 核低功耗 CPU）
练习正确率:    3000 张后 68%
未见字体泛化:  62%（随机基线 20%）
```

### 诚实的局限

- 速率近似 + 随机稀疏发放，不是完整电生理模型（无神经递质动力学、无基因表达）
- 当前使用**符合 FlyWire 统计特征的合成布线**，不是真实连接组（接入方法见下）
- 正确率提升不等于"理解"了数字

---

## 部署

### 环境要求

- Linux（Ubuntu 20.04+）/ macOS；Windows 需 WSL2
- Python 3.10+，无 GPU，**内存占用 <100 MB**
- 对外网的需求：仅安装依赖与下载字体时需要，运行时可离线

### 安装步骤

```bash
# 1. 克隆
git clone https://github.com/twinsmem/flylab.git
cd flylab

# 2. 虚拟环境（若系统缺 python3-venv 包，用 --without-pip 方案）
python3 -m venv --without-pip venv
curl -sS https://bootstrap.pypa.io/get-pip.py | ./venv/bin/python
./venv/bin/pip install numpy pillow fastapi 'uvicorn[standard]'

# 3. 下载字体（训练素材，~2.4 MB）
bash scripts/fetch_fonts.sh
```

### 运行

```bash
# 前台运行（开发调试）
./venv/bin/python -m src.web

# 后台常驻（生产推荐，本项目自带脚本：自动杀旧进程+脱离会话）
bash scripts/start.sh

# 查看日志
tail -f logs/web.log

# 停止
pkill -f "python -m src.web"
```

浏览器访问 `http://<服务器IP>:8000` 即可。

### 断点状态

| 路径 | 内容 |
|---|---|
| `data/state/brain.npz` | 突触权重 + KC 阈值 + 已练张数（每 500 张自动保存） |
| `data/state/metrics.json` | 课程段、正确率曲线、掌握度、众包记录 |
| `data/state/log.jsonl` | 全量训练事件日志 |
| `logs/web.log` | 服务运行日志 |

删除 `data/state/` 即可让果蝇"转世"从头学习。

---

## 代码结构

```
flylab/
├── src/
│   ├── brain.py        # 蘑菇体仿真内核：PN→KC→MBON、阈值校准、多巴胺学习、持久化
│   ├── training.py     # 训练循环：字体渲染、课程控制、测试门槛、众包接口、状态管理
│   └── web.py          # FastAPI 入口：训练线程 + REST API + 静态页
├── static/
│   └── index.html      # 直播仪表盘（单文件，无框架，原生 JS）
├── scripts/
│   ├── fetch_fonts.sh  # 从 Google Fonts API 下载 10 种 TTF
│   ├── start.sh        # 后台启动/重启（无需 sudo）
│   ├── smoke_test.py   # 快速验证：学习曲线 + 未见字体泛化
│   └── test_crowd.py   # 众包接口测试脚本
└── .gitignore          # 排除 venv/字体/运行时状态
```

---

## 改造方向

| 想做什么 | 怎么做 |
|---|---|
| 让它学别的（字母/假名/交易信号） | 改 `training.py` 的 `LEVEL_CLASSES` 与字体集 |
| 接入真实果蝇连接组 | 在 [neuprint.janelia.org](https://neuprint.janelia.org) 注册拿 token，用 `fetch_adjacencies` 拉蘑菇体子图，替换 `brain.py` 的 `_build_synthetic()` |
| 改学习率/稀疏度 | `brain.py` 顶部常量：`ETA_DEPRESS`、`ETA_POTENT`、`KC_TARGET` |
| 换升级门槛 | `training.py`：`TEST_EVERY`（测试间隔）、`GATE`（升段正确率） |
| 给更多神经元"留位置" | `MushroomBody(n_pn=..., n_kc=...)`，注意内存随 KC 数线性增长 |

## 致谢

- [hae.satoru.net](https://hae.satoru.net/) —— 项目形态与界面设计的灵感来源
- [FlyWire](https://flywire.ai/) / Janelia MaleCNS v1.0 —— 连接组数据与神经科学基础
- Hige et al. 2015、Handler et al. 2019 —— 蘑菇体可塑性规则的文献依据

## License

MIT
