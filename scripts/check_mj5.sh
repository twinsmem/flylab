#!/usr/bin/env bash
set -u
VENV=/opt/flylab/venv
echo "== install xvfb =="
apt-get install -y -q xvfb libgl1-mesa-dri libglx-mesa0 2>&1 | tail -1
$VENV/bin/python - <<'EOF'
import os, subprocess

XML = "/opt/flybody/flybody/fruitfly/assets/fruitfly.xml"

def bench(gl, W, H, n=12, extra=None, display=None):
    env = dict(os.environ)
    env["MUJOCO_GL"] = gl
    if extra:
        env.update(extra)
    if display:
        env["DISPLAY"] = display
    code = (
        "import mujoco, time\n"
        f"m = mujoco.MjModel.from_xml_path({XML!r})\n"
        "d = mujoco.MjData(m)\nmujoco.mj_forward(m, d)\n"
        f"r = mujoco.Renderer(m, {H}, {W})\n"
        "r.update_scene(d); r.render()\n"
        "t0 = time.time()\n"
        f"for _ in range({n}):\n    r.update_scene(d); r.render()\n"
        f"dt = (time.time()-t0)/{n}*1000\n"
        f"print({gl!r}, {W!r}, 'x', {H!r}, ':', round(dt), 'ms/frame')\n")
    p = subprocess.run(["/opt/flylab/venv/bin/python", "-c", code], env=env,
                       capture_output=True, text=True, timeout=180)
    out = (p.stdout + p.stderr).strip().splitlines()
    print(out[-1] if out else f"{gl}: no output")

bench("osmesa", 480, 360, extra={"LP_NUM_THREADS": "8", "GALLIUM_DRIVER": "llvmpipe"})
EOF
echo "== xvfb + glfw =="
Xvfb :99 -screen 0 1280x800x24 &
sleep 2
DISPLAY=:99 MUJOCO_GL=glfw /opt/flylab/venv/bin/python - <<'EOF'
import mujoco, time
m = mujoco.MjModel.from_xml_path("/opt/flybody/flybody/fruitfly/assets/fruitfly.xml")
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
r = mujoco.Renderer(m, 360, 480)
r.update_scene(d); r.render()
t0 = time.time()
for _ in range(20):
    r.update_scene(d); r.render()
dt = (time.time()-t0)/20*1000
print("glfw+xvfb 480x360:", round(dt), "ms/frame")
r2 = mujoco.Renderer(m, 480, 640)
r2.update_scene(d); r2.render()
t0 = time.time()
for _ in range(20):
    r2.update_scene(d); r2.render()
print("glfw+xvfb 640x480:", round((time.time()-t0)/20*1000), "ms/frame")
EOF
kill %1 2>/dev/null
