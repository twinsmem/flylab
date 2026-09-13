"""MuJoCo 物理果蝇冒烟测试：加载 flybody 模型 + 步速基准 + 三角步态稳定性。

用法: python scripts/smoke_mj.py [--fast] [key=val ...] [静态目录=static/mj]
    --fast        timestep 2e-4 + 关闭 noslip + pyramidal cone + 放大接触时间常数
    freq=2.0      步态频率 Hz
    swing=0.2     coxa 前后摆幅 (rad)
    lift=0.15     femur/tibia 抬腿幅 (rad)
    claw=0.3      支撑相爪部粘附强度 (0=关)
    tlen=6.0      步行测试时长 (s)
"""
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_args = [a for a in sys.argv[1:] if not a.startswith("--") and "=" not in a]
MJDIR = _args[0] if _args else os.path.join(ROOT, "static", "mj")

import mujoco

model = mujoco.MjModel.from_xml_path(os.path.join(MJDIR, "fruitfly.xml"))
if "--dt2" in sys.argv:
    model.opt.timestep = 0.0002   # 原始 solref(0.0002,1) 在此步长下站立/行走均稳定
if "--noslip0" in sys.argv:
    model.opt.noslip_iterations = 0
if "--cone0" in sys.argv:
    model.opt.cone = 0  # pyramidal
data = mujoco.MjData(model)

_opt = {a.split("=")[0]: float(a.split("=")[1]) for a in sys.argv[1:] if "=" in a}
FREQ, SWING = _opt.get("freq", 4.0), _opt.get("swing", 0.7)
LIFT, CLAW, TLEN = _opt.get("lift", 0.35), _opt.get("claw", 0.0), _opt.get("tlen", 6.0)

print(f"nq={model.nq} nv={model.nv} nu={model.nu} nbody={model.nbody} "
      f"njnt={model.njnt} ngeom={model.ngeom} dt={model.opt.timestep} "
      f"noslip={model.opt.noslip_iterations} freq={FREQ} swing={SWING} "
      f"lift={LIFT} claw={CLAW}")

THORAX = model.body("thorax").id
ABDUCT = [model.actuator(f"coxa_abduct_T{t}_{s}").id for t in (1, 2, 3) for s in ("left", "right")]
COXA = [model.actuator(f"coxa_T{t}_{s}").id for t in (1, 2, 3) for s in ("left", "right")]
FEMUR = [model.actuator(f"femur_T{t}_{s}").id for t in (1, 2, 3) for s in ("left", "right")]
TIBIA = [model.actuator(f"tibia_T{t}_{s}").id for t in (1, 2, 3) for s in ("left", "right")]
CLAWS = [model.actuator(f"adhere_claw_T{t}_{s}").id for t in (1, 2, 3) for s in ("left", "right")]
CR = model.actuator_ctrlrange.copy()

# 同侧腿顺序 T1,T2,T3；三角步态组：L1+R2+R3 与 R1+L2+L3
TRIPOD = [0 if i in (0, 3, 4) else np.pi for i in range(6)]  # L1,L2,L3,R1,R2,R3


def clamp(aid, v):
    lo, hi = CR[aid]
    return np.clip(v, lo, hi)


def report(tag):
    x, y, z = data.xpos[THORAX]
    tilt = np.degrees(np.arccos(np.clip(data.xmat[THORAX].reshape(3, 3)[2, 2], -1, 1)))
    print(f"  [{tag} t={data.time:5.2f}s] 胸高={z * 10:5.2f}mm 位=({x:6.3f},{y:6.3f})cm "
          f"倾角={tilt:5.1f}° 接触={data.ncon}")


def settle(seconds=0.5):
    data.ctrl[:] = 0.0
    for _ in range(int(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)


def gait_ctrl(phase):
    """三角步态：静止姿势(ctrl=0)为基线。摆动相 sin>0 抬腿(femur-)前摆(coxa+)；
    支撑相 tibia+ 后蹬。符号经参数扫描验证（scripts/_scan_gait.py）。"""
    for i in range(6):
        s = np.sin(phase + TRIPOD[i])
        data.ctrl[CLAWS[i]] = CLAW if s < 0 else 0.0
        data.ctrl[ABDUCT[i]] = clamp(ABDUCT[i], 0.0)
        data.ctrl[COXA[i]] = clamp(COXA[i], SWING * s)
        data.ctrl[FEMUR[i]] = clamp(FEMUR[i], -LIFT * max(0.0, s))
        data.ctrl[TIBIA[i]] = clamp(TIBIA[i], LIFT * 0.8 * max(0.0, -s))


def run(seconds, walk=False, log=True, ctrl_every=10):
    n = int(seconds / model.opt.timestep)
    t0 = time.perf_counter()
    for k in range(n):
        if walk and k % ctrl_every == 0:   # 控制信号 1kHz 足够
            gait_ctrl(data.time * FREQ * 2 * np.pi)
        mujoco.mj_step(model, data)
        if not np.isfinite(data.qpos).all():
            print(f"  !! NaN at t={data.time:.4f}s")
            return False
        if log and k % int(0.5 / model.opt.timestep) == 0:
            report("步行" if walk else "滑行")
    wall = time.perf_counter() - t0
    print(f"  实时率 {seconds / wall:.2f}x ({wall / n * 1e6:.0f}μs/步)")
    return True


mujoco.mj_resetData(model, data)
data.qpos[2] = 0.10
mujoco.mj_forward(model, data)
report("初始")
settle(0.5)
report("静置")
run(TLEN, walk=True)
report("步行结束")
data.ctrl[:] = 0.0    # 停止指令：回到站立
run(2.0, walk=False)
report("停止后")
