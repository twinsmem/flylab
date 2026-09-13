"""flybody 官方任务环境仿真线程（/mj 页面数据源，严格 TuragaLab/flybody 路线）。

- walk_imitation() 官方步行模仿环境（推理模式：内置参考轨迹 + ghost）
- 控制信号：回放参考轨迹关节角（官方 HDF5 数据集到手后无缝切换到真实数据）
- 渲染：原生 mujoco.Renderer 桥接（physics.model.ptr），osmesa 软渲染，
  关闭阴影/反射（软光栅化瓶颈，320×240 从 2.0s/帧 降到 0.4s/帧）
- 输出：JPEG 帧缓存（MJPEG 推流用）+ 统计信息
"""
import io
import os
import sys
import threading
import time

if sys.platform != "win32":
    os.environ.setdefault("MUJOCO_GL", "osmesa")   # 服务器无头软渲染；Windows 本机用默认 GL

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RENDER_W, RENDER_H = 320, 240


class FlybodySim:
    def __init__(self, ref_path: str | None = None):
        from flybody.fly_envs import walk_imitation
        import mujoco

        self.mujoco = mujoco
        self.env = walk_imitation(ref_path=ref_path) if ref_path else walk_imitation()
        self.env.reset()
        self.action_dim = self.env.action_spec().shape[0]
        physics = self.env.physics
        self.renderer = mujoco.Renderer(physics.model.ptr, height=RENDER_H, width=RENDER_W)
        # 软渲染优化：关阴影/反射（保留纹理与天空盒）
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0

        self._lock = threading.Lock()
        self._jpeg = None          # 最新 JPEG 帧 bytes
        self._stats = {"fps": 0.0, "sim_t": 0.0, "steps": 0, "reward": 0.0,
                       "cam_id": 1, "ref_path": bool(ref_path)}
        self._cam = 1
        self._stop = False
        threading.Thread(target=self._loop, daemon=True, name="flybody-sim").start()

    # ---------- 控制 ----------
    def _reference_action(self):
        """回放参考轨迹关节角（跟踪 ghost 的最简控制器）。"""
        task = self.env._task
        step = int(round(self.env.physics.data.time / task.control_timestep))
        step = min(step, task._episode_steps)
        qpos = task._ref_qpos[min(step, len(task._ref_qpos) - 1)]
        return qpos[7:7 + self.action_dim].astype(np.float64)

    def _loop(self):
        import PIL.Image

        n, t_stat0 = 0, time.perf_counter()
        while not self._stop:
            timestep = self.env.step(self._reference_action())
            # episode 结束（轨迹走完/超距）→ 官方 reset 开下一段
            if timestep.last():
                self.env.reset()
            try:
                self.renderer.update_scene(self.env.physics.data.ptr, camera=self._cam)
            except Exception:   # 非法机位等渲染异常 → 回退默认机位
                self._cam = 1
                self.renderer.update_scene(self.env.physics.data.ptr, camera=1)
            px = self.renderer.render()
            buf = io.BytesIO()
            PIL.Image.fromarray(px).save(buf, "JPEG", quality=80)
            with self._lock:
                self._jpeg = buf.getvalue()
                self._stats["sim_t"] = float(self.env.physics.data.time)
                self._stats["steps"] += 1
                self._stats["reward"] = float(timestep.reward or 0.0)
            n += 1
            now = time.perf_counter()
            if now - t_stat0 >= 2.0:
                with self._lock:
                    self._stats["fps"] = n / (now - t_stat0)
                n, t_stat0 = 0, now

    # ---------- 对外接口 ----------
    def set_camera(self, cam_id: int):
        self._cam = int(cam_id)

    def jpeg(self) -> bytes | None:
        with self._lock:
            return self._jpeg

    def stats(self) -> dict:
        with self._lock:
            return dict(self._stats)
