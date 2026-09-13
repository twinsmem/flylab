"""MuJoCo 物理果蝇仿真线程（/mj 页面数据源）。

- 加载 static/mj/fruitfly.xml（flybody 模型本地副本，单位 cm，dt=2e-4s）
- 三角步态 CPG（参数经 scripts/_scan_gait.py 扫描验证）
- 按墙钟时间配速，周期性发布全刚体位姿快照 (nbody×7: pos+quat[wxyz])
- NaN / 大倾角 / 走出边界 自动重置
"""
import os
import threading
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MJDIR = os.path.join(ROOT, "static", "mj")

# 步态参数（scripts/smoke_mj.py 验证：freq=4, swing=0.7, lift=0.35, claw=0）
FREQ, SWING, LIFT, CLAW = 4.0, 0.7, 0.35, 0.0
TURN_AMP = 0.10          # 左右腿摆幅差 → 缓慢转向 wandering
BOUND_X = 9.0            # 走到 ±BOUND_X 重置
SETTLE_S = 0.5
CHUNK = 50               # 每次物理步进块（10ms 仿真时间 @dt=2e-4）


class MjFly:
    def __init__(self):
        import mujoco
        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(os.path.join(MJDIR, "fruitfly.xml"))
        self.model.opt.timestep = 0.0002
        self.model.opt.noslip_iterations = 0
        self.data = mujoco.MjData(self.model)

        self.body_names = [self.model.body(i).name for i in range(self.model.nbody)]
        self.nbody = self.model.nbody
        self.thorax = self.model.body("thorax").id
        self._legs = []
        for i in range(6):
            side = "left" if i < 3 else "right"
            tn = i % 3 + 1
            self._legs.append({
                "claw": self.model.actuator(f"adhere_claw_T{tn}_{side}").id,
                "abduct": self.model.actuator(f"coxa_abduct_T{tn}_{side}").id,
                "coxa": self.model.actuator(f"coxa_T{tn}_{side}").id,
                "femur": self.model.actuator(f"femur_T{tn}_{side}").id,
                "tibia": self.model.actuator(f"tibia_T{tn}_{side}").id,
            })
        self.cr = self.model.actuator_ctrlrange.copy()
        # 三角步态相位：L1,R2,R3 同相；L2,L3,R1 反相
        self._tripod = [0.0 if i in (0, 3, 4) else np.pi for i in range(6)]

        self.lock = threading.Lock()
        self._snap = None          # (nbody×7 float32, sim_t, rtf, ncon)
        self._reset()
        self._stop = False
        threading.Thread(target=self._loop, daemon=True, name="mjsim").start()

    # ---------- 控制 ----------
    def _clamp(self, aid, v):
        lo, hi = self.cr[aid]
        return float(np.clip(v, lo, hi))

    def _gait(self, turn_k=0.0):
        """摆动相 sin>0: 抬腿(femur-)+前摆(coxa+)；支撑相: tibia+ 后蹬。
        turn_k∈[-1,1]: 左右 coxa 摆幅差实现转向。"""
        ph = self.data.time * FREQ * 2 * np.pi
        for i, leg in enumerate(self._legs):
            s = np.sin(ph + self._tripod[i])
            k = turn_k if i < 3 else -turn_k
            self.data.ctrl[leg["claw"]] = CLAW if s < 0 else 0.0
            self.data.ctrl[leg["abduct"]] = 0.0
            self.data.ctrl[leg["coxa"]] = self._clamp(leg["coxa"], SWING * s * (1 + k))
            self.data.ctrl[leg["femur"]] = self._clamp(leg["femur"], -LIFT * max(0.0, s))
            self.data.ctrl[leg["tibia"]] = self._clamp(leg["tibia"], LIFT * 0.8 * max(0.0, -s))

    def _reset(self):
        mj = self.mujoco
        mj.mj_resetData(self.model, self.data)
        self.data.qpos[2] = 0.10
        for _ in range(int(SETTLE_S / self.model.opt.timestep)):
            self.data.ctrl[:] = 0.0
            mj.mj_step(self.model, self.data)

    # ---------- 主循环 ----------
    def _loop(self):
        mj = self.mujoco
        wall0 = time.perf_counter()
        sim0 = self.data.time
        frames = 0
        stat_t0 = wall0
        tilt_bad_since = None
        while not self._stop:
            target = (time.perf_counter() - wall0) * 1.0 + sim0
            for _ in range(CHUNK):
                self._gait(TURN_AMP * np.sin(0.10 * 2 * np.pi * self.data.time))
                mj.mj_step(self.model, self.data)
                frames += 1
                if not np.isfinite(self.data.qpos).all():
                    self._reset()
                    wall0 = time.perf_counter()
                    sim0 = self.data.time
                    tilt_bad_since = None
                    break
            now = time.perf_counter()
            rtf = (self.data.time - sim0) / max(now - wall0, 1e-6)
            # 安全检查：倾角 > 50° 持续 0.3s 或越界 → 重置
            tilt = np.degrees(np.arccos(np.clip(
                self.data.xmat[self.thorax].reshape(3, 3)[2, 2], -1, 1)))
            if tilt > 50:
                if tilt_bad_since is None:
                    tilt_bad_since = self.data.time
                elif self.data.time - tilt_bad_since > 0.3:
                    self._reset()
                    wall0 = now
                    sim0 = self.data.time
                    tilt_bad_since = None
            else:
                tilt_bad_since = None
            if abs(self.data.xpos[self.thorax][0]) > BOUND_X or \
               abs(self.data.xpos[self.thorax][1]) > BOUND_X:
                self._reset()
                wall0 = now
                sim0 = self.data.time
            # 发布快照
            buf = np.empty((self.nbody, 7), np.float32)
            buf[:, :3] = self.data.xpos
            buf[:, 3:] = self.data.xquat
            with self.lock:
                self._snap = (buf, float(self.data.time), float(rtf), int(self.data.ncon))
            if now - stat_t0 > 2.0:      # 每 2s 校准一次配速基准
                wall0, sim0, stat_t0 = now, self.data.time, now
                frames = 0
            else:
                time.sleep(0.001)

    def snapshot(self):
        with self.lock:
            return self._snap
