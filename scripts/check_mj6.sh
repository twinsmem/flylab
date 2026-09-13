#!/usr/bin/env bash
set -u
echo "== apt update + install xvfb =="
apt-get update -q 2>&1 | tail -1
apt-get install -y -q xvfb libgl1-mesa-dri libglx-mesa0 2>&1 | tail -1
echo "== xvfb + glfw bench =="
Xvfb :99 -screen 0 1280x800x24 &
XVFB_PID=$!
sleep 2
DISPLAY=:99 MUJOCO_GL=glfw /opt/flylab/venv/bin/python - <<'EOF'
import mujoco, time
m = mujoco.MjModel.from_xml_path("/opt/flybody/flybody/fruitfly/assets/fruitfly.xml")
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
for W, H in [(480, 360), (640, 480)]:
    r = mujoco.Renderer(m, H, W)
    r.update_scene(d); r.render()
    t0 = time.time()
    for _ in range(20):
        r.update_scene(d); r.render()
    print(f"glfw+xvfb {W}x{H}:", round((time.time()-t0)/20*1000), "ms/frame")
    r.close()
EOF
kill $XVFB_PID 2>/dev/null
