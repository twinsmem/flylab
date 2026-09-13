#!/usr/bin/env bash
# 官方数据集下载（TuragaLab/flybody 官方 download_data.py，figshare 源）
set -u
cd /opt/flybody
PY=./venv/bin/python
$PY - <<'EOF'
from flybody.download_data import figshare_download
import time
t0 = time.time()
figshare_download(['walking-imitation-dataset'], dest_path='/opt/flybody-data')
print(f"walking-imitation-dataset 完成 {time.time()-t0:.0f}s")
EOF
echo "=== walking 数据集完成 ==="
du -sh /opt/flybody-data/* 2>/dev/null
