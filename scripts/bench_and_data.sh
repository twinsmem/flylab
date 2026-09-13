#!/usr/bin/env bash
# osmesa 渲染性能基准 + 官方数据集下载（walking 数据集 + trained-policies）
set -u
cd /opt/flybody
echo "=== osmesa 帧率基准 ==="
MUJOCO_GL=osmesa ./venv/bin/python -u - <<'EOF' > /tmp/mj_bench.log 2>&1
import time
import numpy as np
from flybody.fly_envs import walk_imitation
env = walk_imitation()
for _ in range(20):
    env.step(np.random.normal(size=59))
t0 = time.perf_counter()
for _ in range(50):
    env.step(np.random.normal(size=59))
t_step = (time.perf_counter() - t0) / 50
for res in (240, 480):
    t0 = time.perf_counter()
    for _ in range(20):
        env.physics.render(camera_id=1, height=res, width=int(res * 4 / 3))
    t_render = (time.perf_counter() - t0) / 20
    print(f"render {res}p: {t_render * 1000:.0f} ms/frame | physics step: {t_step * 1000:.1f} ms")
    print(f"=> 实时预算内 fps ≈ {1 / (t_step + t_render):.1f}")
import os; os._exit(0)
EOF
cat /tmp/mj_bench.log

echo "=== 下载 walking-imitation-dataset ==="
./venv/bin/python -u - <<'EOF'
from flybody.download_data import figshare_download
figshare_download(['walking-imitation-dataset'], dest_path='/opt/flybody-data')
print("walking dataset done")
EOF
du -sh /opt/flybody-data/* 2>/dev/null
echo "=== 下载 trained-policies ==="
./venv/bin/python -u - <<'EOF'
from flybody.download_data import figshare_download
figshare_download(['trained-policies'], dest_path='/opt/flybody-data')
print("policies done")
EOF
du -sh /opt/flybody-data/* 2>/dev/null
find /opt/flybody-data -maxdepth 3 | head -40
echo "=== 全部完成 ==="
