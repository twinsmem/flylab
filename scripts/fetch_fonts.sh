#!/usr/bin/env bash
# 从 Google Fonts CSS API 下载 10 种静态 TTF 字体
# 用法: bash fetch_fonts.sh [项目根目录]（默认为脚本所在目录的上一级）
set -u
ROOT="${1:-$(cd "$(dirname "$0")/.." && pwd)}"
mkdir -p "$ROOT/data/fonts"
cd "$ROOT/data/fonts" || exit 1
families=("Roboto" "Open+Sans" "Lato" "Montserrat" "Oswald" "Merriweather" "Noto+Sans" "PT+Sans" "Ubuntu" "Source+Sans+3")
for fam in "${families[@]}"; do
  name=$(echo "$fam" | tr -d '+')
  url=$(curl -s "https://fonts.googleapis.com/css2?family=${fam}" | grep -o 'https://fonts.gstatic.com/[^)]*\.ttf' | head -1)
  if [ -n "$url" ]; then
    curl -s "$url" -o "${name}.ttf" && echo "OK ${name}"
  else
    echo "FAIL ${fam}"
  fi
done
echo "----"
ls -la
