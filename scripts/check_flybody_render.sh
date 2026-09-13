#!/usr/bin/env bash
# flybody 官方路线验证：import 测试 + MUJOCO_GL 渲染后端逐个验证
set -u
cd /opt/flybody
PY=./venv/bin/python

echo "=== 1) 基础 import ==="
$PY - <<'EOF'
import flybody, dm_control, mujoco, numpy as np
print("flybody OK | mujoco", mujoco.__version__, "| numpy", np.__version__)
EOF

echo "=== 2) MUJOCO_GL 渲染后端测试（各 render 10 帧，320x240） ==="
for backend in egl osmesa glfw; do
  echo "--- backend=$backend ---"
  MUJOCO_GL=$backend $PY - <<'EOF' 2>&1 | tail -5
import os, time
backend = os.environ["MUJOCO_GL"]
import numpy as np
try:
    from flybody.fly_envs import walk_imitation
    env = walk_imitation()
    t0 = time.perf_counter()
    for _ in range(10):
        action = np.random.normal(size=59)
        env.step(action)
    px = env.physics.render(camera_id=1, height=240, width=320)
    dt = time.perf_counter() - t0
    print(f"{backend}: step+render 10帧 OK, {dt:.2f}s, 帧shape={px.shape}, 均值={px.mean():.1f}")
except Exception as e:
    print(f"{backend}: FAIL -> {type(e).__name__}: {str(e)[:200]}")
EOF
done

echo "=== 3) xvfb + glfw 组合（若 xvfb 存在） ==="
if command -v Xvfb >/dev/null 2>&1; then
  Xvfb :99 -screen 0 640x480x24 &
  XPID=$!
  sleep 1
  DISPLAY=:99 MUJOCO_GL=glfw $PY - <<'EOF' 2>&1 | tail -3
import time
import numpy as np
try:
    from flybody.fly_envs import walk_imitation
    env = walk_imitation()
    t0 = time.perf_counter()
    for _ in range(10):
        env.step(np.random.normal(size=59))
    px = env.physics.render(camera_id=1, height=240, width=320)
    print(f"xvfb+glfw: OK {time.perf_counter()-t0:.2f}s shape={px.shape}")
except Exception as e:
    print(f"xvfb+glfw: FAIL -> {type(e).__name__}: {str(e)[:200]}")
EOF
  kill $XPID 2>/dev/null
else
  echo "Xvfb 未安装，跳过"
fi
echo "=== 渲染验证完成 ==="
