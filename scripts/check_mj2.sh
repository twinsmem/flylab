#!/usr/bin/env bash
# osmesa 安装后重试 MuJoCo 渲染 + 模型结构检查
set -u
echo "== apt install =="
apt-get install -y -q libosmesa6 libegl1 libgl1 libgles2 2>&1 | tail -1
VENV=/opt/flylab/venv
export MUJOCO_GL=osmesa
echo "== render test =="
$VENV/bin/python - <<'EOF'
import mujoco, numpy as np
m = mujoco.MjModel.from_xml_path("/opt/flybody/flybody/fruitfly/assets/fruitfly.xml")
r = mujoco.Renderer(m, 240, 320)
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
r.update_scene(d)
img = r.render()
print("osmesa render OK", img.shape, img.mean())
print("nq", m.nq, "nu", m.nu, "ngeom", m.ngeom, "njnt", m.njnt, "nkey", m.nkey)
acts = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
print("actuator sample:", acts[:14])
keys = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_KEY, i) for i in range(m.nkey)]
print("keyframes:", keys)
jnts = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)]
print("front-left leg joints:", [j for j in jnts if j and ("T1L" in j or "L1" in j or "tarsus1L" in j)][:14])
EOF
echo "== perf test: 60 steps + 20 renders =="
$VENV/bin/python - <<'EOF'
import time
import mujoco
m = mujoco.MjModel.from_xml_path("/opt/flybody/flybody/fruitfly/assets/fruitfly.xml")
d = mujoco.MjData(m)
r = mujoco.Renderer(m, 360, 480)
mujoco.mj_forward(m, d)
t0 = time.time()
for _ in range(60):
    d.ctrl[:] = 0
    mujoco.mj_step(m, d)
t1 = time.time()
for _ in range(20):
    r.update_scene(d)
    r.render()
t2 = time.time()
print(f"60 steps: {t1-t0:.2f}s, 20 renders(480x360): {t2-t1:.2f}s")
EOF
