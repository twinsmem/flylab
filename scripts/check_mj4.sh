#!/usr/bin/env bash
set -u
VENV=/opt/flylab/venv
$VENV/bin/python - <<'EOF'
import os, subprocess

XML = "/opt/flybody/flybody/fruitfly/assets/fruitfly.xml"

def bench(gl, W, H, n=12, extra=None):
    env = dict(os.environ)
    env["MUJOCO_GL"] = gl
    if extra:
        env.update(extra)
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

bench("osmesa", 480, 360, extra={"LP_NUM_THREADS": "8"})
bench("osmesa", 320, 240, extra={"LP_NUM_THREADS": "8"})
bench("osmesa", 480, 360, extra={"LP_NUM_THREADS": "1"})
p = subprocess.run(["/opt/flylab/venv/bin/python", "-c",
                    "import os; os.environ['MUJOCO_GL']='egl'; import mujoco; print('Renderer' in dir(mujoco))"],
                   capture_output=True, text=True, timeout=60)
print("egl Renderer available:", p.stdout.strip(), p.stderr.strip()[:120])
EOF
