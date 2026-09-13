#!/usr/bin/env bash
# 从 Google Fonts CSS API 下载 10 种静态 TTF 字体
set -u
cd ~/flylab/data/fonts || exit 1
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
