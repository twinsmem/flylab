#!/usr/bin/env bash
# flybody 安装与渲染可行性检查
set -u
VENV=/opt/flylab/venv
if ! $VENV/bin/python -c "import flybody" 2>/dev/null; then
  echo "== installing flybody (core) =="
  $VENV/bin/pip install -q git+https://github.com/TuragaLab/flybody.git 2>&1 | tail -3
fi
echo "== versions =="
$VENV/bin/python - <<'EOF'
import mujoco, numpy as np
print("mujoco", mujoco.__version__)
try:
    import flybody
    print("flybody import OK")
except Exception as e:
    print("flybody FAIL:", e)
EOF
echo "== GL check: try egl then osmesa =="
for GL in egl osmesa glfw; do
  MUJOCO_GL=$GL $VENV/bin/python - <<EOF
import os
ok = True
try:
    from mujoco import _render  # noqa
    import mujoco
    m = mujoco.MjModel.from_xml_path("/opt/flybody/flybody/fruitfly/assets/fruitfly.xml")
    r = mujoco.Renderer(m, height=240, width=320)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    r.update_scene(d)
    img = r.render()
    print("$GL render OK", img.shape)
except Exception as e:
    print("$GL FAIL:", str(e)[:160])
EOF
done
echo "== model summary =="
MUJOCO_GL=osmesa $VENV/bin/python - <<'EOF'
try:
    import mujoco
    m = mujoco.MjModel.from_xml_path("/opt/flybody/flybody/fruitfly/assets/fruitfly.xml")
    print("nq", m.nq, "nu", m.nu, "ngeom", m.ngeom, "njnt", m.njnt)
    legs = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
    print("first actuators:", legs[:12])
    print("actuator count:", m.nu)
    print("keyframes:", m.nkey, [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_KEY, i) for i in range(m.nkey)])
except Exception as e:
    print("model FAIL:", str(e)[:200])
EOF
