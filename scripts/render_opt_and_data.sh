#!/usr/bin/env bash
# 渲染优化测试：原生 mujoco.Renderer + 关闭昂贵渲染 flags + 分辨率扫描
# 同时用 curl 修复 figshare 数据集下载
set -u
cd /opt/flybody

echo "=== 1) figshare 数据集 curl 直下 ==="
mkdir -p /opt/flybody-data
cd /opt/flybody-data
curl -sL --max-time 560 -o walk_dataset.zip "https://janelia.figshare.com/ndownloader/files/51196868" &
WPID=$!
curl -sL --max-time 560 -o policies.zip "https://janelia.figshare.com/ndownloader/files/44815195" &
PPID2=$!
wait $WPID; echo "walk zip: $(ls -la walk_dataset.zip 2>/dev/null | awk '{print $5}') bytes"
wait $PPID2; echo "policies zip: $(ls -la policies.zip 2>/dev/null | awk '{print $5}') bytes"
cd /opt/flybody

echo "=== 2) 原生 Renderer + flags 优化帧率测试 ==="
MUJOCO_GL=osmesa ./venv/bin/python -u - <<'EOF' > /tmp/mj_bench2.log 2>&1
import time
import numpy as np
import mujoco
from flybody.fly_envs import walk_imitation

env = walk_imitation()
env.reset()
for _ in range(10):
    env.step(np.random.normal(size=59))

physics = env.physics
H, W = 240, 320

def bench(tag, flags_off, n=10):
    renderer = mujoco.Renderer(physics.model, height=H, width=W)
    scene_flag_names = {
        'shadow': mujoco.mjtRndFlag.mjRND_SHADOW,
        'reflection': mujoco.mjtRndFlag.mjRND_REFLECTION,
        'skybox': mujoco.mjtRndFlag.mjRND_SKYBOX,
    }
    for name, flag in scene_flag_names.items():
        if name in flags_off:
            renderer.scene.flags[flag] = 0
    # 预热 1 帧
    renderer.update_scene(physics.data, camera=1)
    renderer.render()
    t0 = time.perf_counter()
    for _ in range(n):
        env.step(np.random.normal(size=59))
        renderer.update_scene(physics.data, camera=1)
        px = renderer.render()
    dt = (time.perf_counter() - t0) / n
    print(f"{tag}: {dt*1000:.0f} ms/帧(step+render) -> {1/dt:.2f} fps")
    return px

px = bench("全特效", set())
px = bench("关阴影+反射", {"shadow", "reflection"})
px = bench("关阴影+反射+天空盒", {"shadow", "reflection", "skybox"})
import PIL.Image
PIL.Image.fromarray(px).save('/tmp/opt_render.png')
print("saved /tmp/opt_render.png")
import os; os._exit(0)
EOF
cat /tmp/mj_bench2.log

echo "=== 3) 解压数据集 ==="
cd /opt/flybody-data
for z in walk_dataset.zip policies.zip; do
  if [ -s "$z" ]; then
    file "$z" | head -1
    unzip -oq "$z" -d "${z%.zip}" && echo "解压 $z OK" || echo "解压 $z FAIL（可能非zip）"
  fi
done
find /opt/flybody-data -maxdepth 3 | head -30
du -sh /opt/flybody-data/* 2>/dev/null
echo "=== 完成 ==="
