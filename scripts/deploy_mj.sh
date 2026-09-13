#!/usr/bin/env bash
# flylab 服务器部署更新：拉代码 + 安装 /mj 与 /mj3d 依赖 + 重启服务
set -e
cd /opt/flylab
git pull origin main 2>&1 | tail -2

echo "=== 安装 /mj3d 依赖（mujoco 物理位姿流）==="
./venv/bin/pip install --quiet mujoco pillow

echo "=== 安装 /mj 依赖（flybody core，含 dm_control）==="
./venv/bin/pip install --quiet -e /opt/flybody 2>&1 | tail -3

echo "=== 检查服务进程 ==="
ps aux | grep -E "uvicorn|src.web" | grep -v grep || echo "无运行中服务"
