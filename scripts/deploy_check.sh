#!/usr/bin/env bash
# v2 部署验证
set -u
cd /opt/flylab || exit 1
git pull -q || exit 1
bash scripts/start.sh /opt/flylab
sleep 5
echo '--- pages ---'
for p in / /records /mechanism /about /static/style.css; do
  printf '%s -> ' "$p"
  curl -s -o /dev/null -w '%{http_code}\n' "http://localhost:8000$p"
done
echo '--- api ---'
curl -s http://localhost:8000/api/state | head -c 150; echo
curl -s http://localhost:8000/api/kc-positions | head -c 60; echo
curl -s http://localhost:8000/api/activity | head -c 250; echo
curl -s http://localhost:8000/api/quiz-next | head -c 80; echo
