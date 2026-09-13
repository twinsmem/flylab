#!/usr/bin/env bash
# FlyLab 启动/重启脚本（无需 sudo）
# 用法: bash start.sh [项目根目录]（默认为脚本所在目录的上一级）
set -u
ROOT="${1:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$ROOT" || exit 1
mkdir -p logs

# 停掉旧进程：用 pidfile 精确匹配，避免 pkill -f 自匹配误判
if [ -f logs/web.pid ]; then
  OLDPID=$(cat logs/web.pid)
  kill "$OLDPID" 2>/dev/null && echo "killed old pid=$OLDPID"
  rm -f logs/web.pid
  sleep 1
fi
# 兜底：精确匹配命令行（-x 语义用 pgrep -f 全串），排除自身
for pid in $(pgrep -f "venv/bin/python -m src[.]web"); do
  [ "$pid" != "$$" ] && kill "$pid" 2>/dev/null && echo "killed leftover pid=$pid"
done
sleep 1

# 后台常驻：setsid 脱离会话，nohup 忽略挂断
setsid nohup ./venv/bin/python -m src.web > logs/web.log 2>&1 < /dev/null &
echo $! > logs/web.pid
echo "started, pid=$(cat logs/web.pid)"
sleep 3
tail -n 5 logs/web.log
