#!/usr/bin/env bash
# FlyLab 启动/重启脚本（无需 sudo）。用法： bash ~/flylab/start.sh
cd ~/flylab || exit 1
# 停掉旧进程
pkill -f "python -m src.web" 2>/dev/null
sleep 1
# 后台常驻：setsid 脱离会话，nohup 忽略挂断
setsid nohup ./venv/bin/python -m src.web > logs/web.log 2>&1 < /dev/null &
echo "started, pid=$!"
sleep 2
tail -n 5 logs/web.log
