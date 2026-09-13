#!/usr/bin/env bash
# GL 后端性能对比：osmesa vs egl(mesa llvmpipe) vs vulkan(lavapipe)，含 LP_NUM_THREADS 多线程
set -u
VENV=/opt/flylab/venv
apt-get install -y -q libegl-mesa0 libvulkan1 mesa-vulkan-drivers vulkan-tools 2>&1 | tail -1

$VENV/bin/python - <<'EOF'
import os, time
import numpy as np

XML = "/opt/flybody/flybody/fruitfly/assets/fruitfly.xml"

def bench(gl, W, H, n=12, env_extra=None):
    env = dict(os.environ)
    env["MUJOCO_GL"] = gl
    if env_extra:
        env.update(env_extra)
    # 子进程隔离 GL 状态
    code = f"""
import mujoco, time
m = mujoco.MjModel.from_xml_path("{XML}")
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
r = mujoco.Renderer(m, {H}, {W})
r.update_scene(d); r.render()  # warmup
t0 = time.time()
for _ in range({n}):
    r.update_scene(d); r.render()
print(f"{gl} {{W}}x{{H}}: {{(time.time()-t0)/{n}*1000:.0f}} ms/frame")
"""
    import subprocess
    p = subprocess.run(["/opt/flylab/venv/bin/python", "-c", code], env=env,
                       capture_output=True, text=True, timeout=120)
    out = (p.stdout + p.stderr).strip().splitlines()
    print(out[-1] if out else f"{gl}: no output")

for gl, extra in [("osmesa", {"LP_NUM_THREADS": "8"}),
                  ("egl", {"LP_NUM_THREADS": "8"}),
                  ("vulkan", {})]:
    try:
        bench(gl, 480, 360, env_extra=extra)
    except Exception as e:
        print(gl, "EXC", str(e)[:100])
EOF
