#!/usr/bin/env bash
set -u
apt-get install -y xvfb 2>&1 | grep -E "^(E:|The following|  )" | head -20
echo "---"
apt-cache policy xvfb libgl1-mesa-dri 2>/dev/null | head -12
