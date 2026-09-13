"""步态参数扫描（临时工具）：reset->settle->walk，找不摔倒且有位移的参数组合。"""
import itertools
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import mujoco

model = mujoco.MjModel.from_xml_path(os.path.join(ROOT, "static", "mj", "fruitfly.xml"))
model.opt.timestep = 0.0002
model.opt.noslip_iterations = 0
data = mujoco.MjData(model)
THORAX = model.body("thorax").id
A = {n: model.actuator(n).id for n in (
    *[f"coxa_abduct_T{t}_{s}" for t in (1, 2, 3) for s in ("left", "right")],
    *[f"coxa_T{t}_{s}" for t in (1, 2, 3) for s in ("left", "right")],
    *[f"femur_T{t}_{s}" for t in (1, 2, 3) for s in ("left", "right")],
    *[f"tibia_T{t}_{s}" for t in (1, 2, 3) for s in ("left", "right")],
    *[f"adhere_claw_T{t}_{s}" for t in (1, 2, 3) for s in ("left", "right")])}
CR = model.actuator_ctrlrange.copy()
TRIPOD = [0 if i in (0, 3, 4) else np.pi for i in range(6)]


def clamp(aid, v):
    lo, hi = CR[aid]
    return np.clip(v, lo, hi)


def trial(swing, lift, claw, csign, fsign, tsign, freq=2.0, walk_s=2.0):
    mujoco.mj_resetData(model, data)
    data.qpos[2] = 0.10
    mujoco.mj_forward(model, data)
    for _ in range(int(0.4 / model.opt.timestep)):
        data.ctrl[:] = 0.0
        mujoco.mj_step(model, data)
    x0 = data.xpos[THORAX].copy()
    max_tilt = 0.0
    n = int(walk_s / model.opt.timestep)
    for k in range(n):
        if k % 10 == 0:
            ph = data.time * freq * 2 * np.pi
            for i in range(6):
                s = np.sin(ph + TRIPOD[i])
                side = "left" if i < 3 else "right"
                tn = i % 3 + 1
                claw_n, abd_n = f"adhere_claw_T{tn}_{side}", f"coxa_abduct_T{tn}_{side}"
                cox_n, fem_n, tib_n = f"coxa_T{tn}_{side}", f"femur_T{tn}_{side}", f"tibia_T{tn}_{side}"
                data.ctrl[A[claw_n]] = claw if s < 0 else 0.0
                data.ctrl[A[abd_n]] = 0.0
                data.ctrl[A[cox_n]] = clamp(A[cox_n], csign * swing * s)
                data.ctrl[A[fem_n]] = clamp(A[fem_n], fsign * lift * max(0.0, s))
                data.ctrl[A[tib_n]] = clamp(A[tib_n], tsign * lift * 0.8 * max(0.0, -s))
        mujoco.mj_step(model, data)
        if not np.isfinite(data.qpos).all():
            return None
        max_tilt = max(max_tilt, np.degrees(np.arccos(
            np.clip(data.xmat[THORAX].reshape(3, 3)[2, 2], -1, 1))))
    dx = data.xpos[THORAX][0] - x0[0]
    dy = data.xpos[THORAX][1] - x0[1]
    return dx, dy, max_tilt, data.xpos[THORAX][2]


print(f"{'freq':>5} {'swing':>5} {'lift':>5} {'claw':>4} | "
      f"{'dx(cm)':>7} {'dy':>6} {'max倾角':>6} {'末高mm':>5}")
for freq, swing, lift, claw in itertools.product(
        (2.0, 3.0, 4.0), (0.7,), (0.35,), (0.0, 0.15, 0.3)):
    r = trial(swing, lift, claw, 1, -1, 1, freq=freq, walk_s=3.0)
    if r is None:
        print(f"{freq:5} {swing:5} {lift:5} {claw:4} | NaN")
        continue
    dx, dy, tilt, z = r
    flag = " ✓" if tilt < 40 and abs(dx) > 0.05 else ""
    print(f"{freq:5} {swing:5} {lift:5} {claw:4} | "
          f"{dx:7.3f} {dy:6.3f} {tilt:6.1f} {z * 10:5.2f}{flag}")
