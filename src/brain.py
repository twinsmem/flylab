"""蘑菇体（mushroom body）仿真内核。

结构对应果蝇嗅觉学习回路：
    PN（嗅觉投射神经元，685 个）→ KC（凯尼恩细胞，2500 个，~10% 稀疏发放）
    → MBON（蘑菇体输出神经元，每类 2 个，argmin 读出：受输入最少的区画当选）。

连接组说明：目前使用符合真实统计特征的合成布线（每个 KC 接收 6~15 个 PN 随机汇聚，
与 FlyWire 实测一致）。接入真实 MaleCNS/FAFB 蘑菇体子图时，只需用真实索引替换
_build_synthetic() 生成的连接矩阵，其余逻辑不变。

学习规则（答错才施加，仿 hae.satoru.net）：
    - 正确答案区画：多巴胺 → 活跃 KC→MBON 连接削弱（抗赫布，Hige et al. 2015）
    - 被误选的区画：反向信号 → 活跃连接增强（Handler et al. 2019 的时序对应）
由于读出是 argmin，削弱正确区画 = 下次它更容易赢。
"""

import numpy as np

PN_MAX_HZ = 270.0      # PN 最大发放率（原站同款）
SIM_T = 0.4            # 每次呈现模拟 0.4 秒神经时间
DT = 0.01              # 10 ms 时间步 → 40 个 bin
KC_TARGET = 0.10       # KC 目标稀疏度 ~10%
W_MIN, W_MAX = 0.0, 1.0
ETA_DEPRESS = 0.020    # 多巴胺削弱步长
ETA_POTENT = 0.010     # 反向增强步长


class MushroomBody:
    def __init__(self, n_pn=685, n_kc=2500, n_classes=10, mbons_per_class=2,
                 seed=42, img_px=16 * 16):
        self.rng = np.random.default_rng(seed)
        self.n_pn, self.n_kc = n_pn, n_kc
        self.n_classes, self.mpc = n_classes, mbons_per_class
        self.n_mbon = n_classes * mbons_per_class
        self.t_present = 0  # 已呈现样本数

        self._build_pn_map(img_px)
        self._build_synthetic()

        # KC 阈值：用噪声输入校准，使平均发放率 ≈ KC_TARGET
        theta = self._calibrate_threshold()
        self.kc_theta = theta

        # 权重与统计（可持久化）
        self.w = self.rng.uniform(0.05, 0.15, size=(self.n_kc, self.n_mbon)).astype(np.float32)
        # 每个 MBON 只连接一部分 KC（~12%），其余强制为 0，贴近真实区画输入稀疏性
        mask = self.rng.random((self.n_kc, self.n_mbon)) < 0.12
        self.w = np.where(mask, self.w, np.float32(0.0)).astype(np.float32)
        # 区画输入基线（学習前=100 归一化用）：每列初始权重总和
        self.col_sum0 = self.w.sum(axis=0).astype(np.float32)
        # KC 二维示意位置（双半球蘑菇体轮廓，仅用于可视化）
        self.kc_xy = self._build_kc_positions()
        # 最近一次活动/强化快照（供网站展示）
        self.last_activity = None   # present() 填充
        self.last_reinforce = None  # reinforce() 填充

    def _build_kc_positions(self, seed=123):
        """生成 5,177 个 KC 的二维示意位置：双半球，每半球 = 萼+柄+叶 三簇高斯。"""
        rng = np.random.default_rng(seed)
        blobs = [  # (cx, cy, sx, sy, 比例) —— 近似蘑菇体轮廓
            (0.30, 0.35, 0.10, 0.12, 0.55),  # 蕈体萼（细胞体聚集）
            (0.42, 0.55, 0.05, 0.13, 0.25),  # 中柄
            (0.46, 0.78, 0.09, 0.06, 0.20),  # 输出叶
        ]
        pts = []
        for mirror in (False, True):
            for cx, cy, sx, sy, wt in blobs:
                n = int(round(self.n_kc / 2 * wt))
                x = rng.normal(cx, sx, n)
                y = rng.normal(cy, sy, n)
                if mirror:
                    x = 1.0 - x
                pts.append(np.stack([x, y], axis=1))
        xy = np.concatenate(pts).astype(np.float32)
        if len(xy) >= self.n_kc:
            xy = xy[:self.n_kc]
        else:  # 取整误差补齐
            extra = rng.integers(0, len(xy), size=self.n_kc - len(xy))
            xy = np.concatenate([xy, xy[extra]])
        np.clip(xy, 0.02, 0.98, out=xy)
        return xy

    # ---------- 布线 ----------
    def _build_pn_map(self, img_px):
        """每个像素分配 2~3 个 PN（256 像素 × ~2.7 ≈ 685）。"""
        assign = []
        pn = 0
        extra = self.n_pn - img_px * 2  # 多出来的 PN 分给前 extra 个像素
        for px in range(img_px):
            k = 3 if px < extra else 2
            assign.append(np.arange(pn, pn + k))
            pn += k
        self.pn_map = assign  # list[ndarray]

    def _build_synthetic(self):
        """PN→KC 随机汇聚：每个 KC 收 6~15 个 PN。"""
        n_in = self.rng.integers(6, 16, size=self.n_kc)
        rows = np.repeat(np.arange(self.n_kc), n_in)
        cols = np.concatenate([self.rng.permutation(self.n_pn)[:k] for k in n_in])
        self.pk_rows, self.pk_cols = rows, cols
        # 汇聚强度的倒数作为增益：让不同 KC 的输入量级可比
        self.pk_gain = (1.0 / n_in).astype(np.float32)

    def _calibrate_threshold(self, n_cal=200):
        """用「字符状」稀疏图样校准 KC 阈值，使平均稀疏度达到 KC_TARGET。

        真实输入是黑底白字：多数像素接近 0，少量笔画像素较亮。
        用全幅噪声校准会高估输入、抬高阈值（实测稀疏度掉到 ~4%）。
        """
        currents = np.empty((n_cal, self.n_kc), dtype=np.float32)
        for i in range(n_cal):
            img = np.zeros(256, dtype=np.float32)
            if i >= n_cal // 5:  # 80% 有笔画，20% 近全黑
                n_ink = self.rng.integers(15, 50)
                idx = self.rng.choice(256, size=n_ink, replace=False)
                img[idx] = self.rng.uniform(0.4, 1.0, size=n_ink)
            pn_rate = self._image_to_rate(img)
            spikes = self.rng.random((len(DT_BINS), self.n_pn)) < (pn_rate * DT)
            cur = np.zeros(self.n_kc, dtype=np.float32)
            np.add.at(cur, self.pk_rows, spikes.sum(axis=0)[self.pk_cols].astype(np.float32))
            currents[i] = cur * self.pk_gain
        flat = currents.ravel()
        theta = np.quantile(flat, 1.0 - KC_TARGET)
        return float(theta)

    def _image_to_rate(self, img256):
        """16×16 灰度 → PN 发放率（Hz）。"""
        rate = np.zeros(self.n_pn, dtype=np.float32)
        for px, pns in enumerate(self.pn_map):
            # 每像素总输入 = intensity × PN_MAX_HZ，由 2~3 个 PN 分摊
            rate[pns] = img256[px] * PN_MAX_HZ / len(pns)
        return rate

    # ---------- 前向 ----------
    def present(self, img256, rng=None):
        """呈现一幅 16×16 图像（展平 256 灰度 0~1），返回 (kc_spikes, mbon_input)。"""
        rng = rng or self.rng
        rate = self._image_to_rate(img256)
        n_bins = int(round(SIM_T / DT))
        spikes = (rng.random((n_bins, self.n_pn)) < (rate[None, :] * DT))
        kc_in = np.zeros(self.n_kc, dtype=np.float32)
        np.add.at(kc_in, self.pk_rows, spikes.sum(axis=0)[self.pk_cols].astype(np.float32))
        kc_in *= self.pk_gain
        # 稳定 Sigmoid：输入越高于阈值越倾向发放，保留随机性
        p = 1.0 / (1.0 + np.exp(-(kc_in - self.kc_theta) * 6.0))
        kc_spikes = rng.random(self.n_kc) < p
        mbon_input = kc_spikes.astype(np.float32) @ self.w
        n_active = int(kc_spikes.sum())
        self.last_activity = {
            "pn_hz": float(rate.mean()),          # 全体 PN 平均发放率
            "kc_active": n_active,                # 本次发火的 KC 数
            "kc_frac": float(kc_spikes.mean()),   # KC 群体稀疏度
            "kc_hz": float(kc_spikes.mean() / SIM_T),  # 单 KC 平均发放率
            "t": self.t_present,
        }
        return kc_spikes, mbon_input

    def comp_values(self, mbon_input, n_active, active_classes=None):
        """各区画对当前图像的输入强度，「学習前 = 100」归一化（仿原站 しくみ 页）。

        基线 = 初始权重下同等数量活跃 KC 的期望输入（按活跃比例缩放的列和）。
        学习会削弱势区画的列和 → 训练过的区画数值 < 100；argmin 当选。
        """
        if n_active < 5:
            return {c: 100.0 for c in range(self.n_classes)}
        base = self.col_sum0.reshape(self.n_classes, self.mpc).sum(axis=1)
        cur = mbon_input.reshape(self.n_classes, self.mpc).sum(axis=1)
        scale = base * (n_active / self.n_kc)
        scale = np.where(scale <= 0, 1e-9, scale)
        vals = 100.0 * cur / scale
        if active_classes is not None:
            return {int(c): float(vals[c]) for c in active_classes}
        return {c: float(vals[c]) for c in range(self.n_classes)}

    def decide(self, mbon_input, active_classes):
        """argmin 读出：active_classes 中受输入最少的区画对应的类别当选。"""
        best_c, best_v = None, np.inf
        for c in active_classes:
            v = mbon_input[c * self.mpc:(c + 1) * self.mpc].sum()
            if v < best_v:
                best_c, best_v = c, v
        return best_c

    # ---------- 学习 ----------
    def reinforce(self, kc_spikes, true_class, chosen_class):
        """答错时施加多巴胺（正确区画削弱）与反向信号（误选区画增强）。"""
        if true_class == chosen_class:
            return
        act = kc_spikes.astype(np.float32)
        cols_t = slice(true_class * self.mpc, (true_class + 1) * self.mpc)
        cols_c = slice(chosen_class * self.mpc, (chosen_class + 1) * self.mpc)
        self.w[:, cols_t] = np.clip(self.w[:, cols_t] - ETA_DEPRESS * act[:, None],
                                    W_MIN, W_MAX)
        self.w[:, cols_c] = np.clip(self.w[:, cols_c] + ETA_POTENT * act[:, None],
                                    W_MIN, W_MAX)
        # 记录「这次学到最多」的 KC：活跃且对两个被修改区画连接权重最大的那个
        impact = act * (self.w[:, cols_t].sum(axis=1) + self.w[:, cols_c].sum(axis=1))
        self.last_reinforce = {
            "kc": int(np.argmax(impact)),
            "frac": float(act.mean()),
            "t": self.t_present,
        }

    # ---------- 持久化 ----------
    def save(self, path):
        np.savez_compressed(path, w=self.w, theta=self.kc_theta, t=self.t_present,
                            col_sum0=self.col_sum0)

    def load(self, path):
        z = np.load(path)
        self.w, self.kc_theta, self.t_present = z["w"], float(z["theta"]), int(z["t"])
        # 旧存档没有基线：以当前权重为基线（此后学习相对此刻归一）
        self.col_sum0 = (z["col_sum0"] if "col_sum0" in z
                         else self.w.sum(axis=0).astype(np.float32))


DT_BINS = np.arange(int(round(SIM_T / DT)))
