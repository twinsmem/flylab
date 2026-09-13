#!/usr/bin/env bash
# 渲染优化测试 v2：经 .ptr 桥接到原生 mujoco.Renderer，关闭昂贵 flags
set -u
cd /opt/flybody
MUJOCO_GL=osmesa ./venv/bin/python -u - <<'EOF' > /tmp/mj_bench3.log 2>&1
import time
import numpy as np
import mujoco
from flybody.fly_envs import walk_imitation

env = walk_imitation()
env.reset()
for _ in range(10):
    env.step(np.random.normal(size=59))

physics = env.physics
native_model = physics.model.ptr   # 原生 mujoco.MjModel
print("native model type:", type(native_model).__name__)

H, W = 240, 320

def bench(tag, flags_off, n=12):
    renderer = mujoco.Renderer(native_model, height=H, width=W)
    flags_map = {
        'shadow': mujoco.mjtRndFlag.mjRND_SHADOW,
        'reflection': mujoco.mjtRndFlag.mjRND_REFLECTION,
        'skybox': mujoco.mjtRndFlag.mjRND_SKYBOX,
    }
    for name, flag in flags_map.items():
        if name in flags_off:
            renderer.scene.flags[flag] = 0
    renderer.update_scene(physics.data.ptr, camera=1)
    renderer.render()
    t0 = time.perf_counter()
    for _ in range(n):
        env.step(np.random.normal(size=59))
        renderer.update_scene(physics.data.ptr, camera=1)
        px = renderer.render()
    dt = (time.perf_counter() - t0) / n
    print(f"{tag}: {dt*1000:.0f} ms/帧(step+render) -> {1/dt:.2f} fps")
    renderer.close()
    return px

px = bench("全特效", set())
px = bench("关阴影+反射", {"shadow", "reflection"})
px = bench("关阴影+反射+天空盒", {"shadow", "reflection", "skybox"})
import PIL.Image
PIL.Image.fromarray(px).save('/tmp/opt_render.png')
print("saved")
import os; os._exit(0)
EOF
cat /tmp/mj_bench3.log
scp_fail=0
echo "=== 完成 ==="
