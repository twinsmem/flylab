#!/usr/bin/env bash
# 在 192.168.1.227 上按官方说明安装 flybody core（TuragaLab）
# 服务器系统 Python 3.10.12 符合官方 >=3.10 要求；无 GPU，跳过 cudatoolkit
set -e
cd /opt
if [ ! -d flybody ]; then
  git clone --depth 1 https://github.com/TuragaLab/flybody.git
else
  echo "flybody 目录已存在，跳过 clone"
fi
cd flybody
python3 --version
# 服务器缺 python3-venv 包时用 --without-pip 方案（flylab 部署时的旧经验）
python3 -m venv --without-pip venv 2>/dev/null || python3 -m venv venv
if [ ! -f venv/bin/pip ]; then
  curl -sS https://bootstrap.pypa.io/get-pip.py | ./venv/bin/python
fi
./venv/bin/pip install -e .
./venv/bin/python - <<'EOF'
import flybody
print("flybody import OK:", flybody.__file__)
import dm_control
print("dm_control OK:", dm_control.__version__ if hasattr(dm_control, "__version__") else "?")
EOF
echo "=== 安装完成 ==="
