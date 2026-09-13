#!/usr/bin/env bash
# 检查官方 policy 加载方式（test-tf.py + getting-started.ipynb）
echo "=== test-tf.py 前 80 行 ==="
head -80 /opt/flybody/tests/test-tf.py
echo "=== getting-started.ipynb 中的 policy 相关代码 ==="
/opt/flybody/venv/bin/python - <<'EOF'
import json, re
for name in ('getting-started.ipynb', 'fly-env-examples.ipynb', 'controller-reuse-vision-flight.ipynb'):
    try:
        nb = json.load(open(f'/opt/flybody/docs/{name}'))
    except Exception as e:
        print(name, 'unreadable:', e)
        continue
    print(f'--- {name} ---')
    for c in nb['cells']:
        src = ''.join(c['source'])
        if re.search(r'policy|actor|checkpoint|saved|load', src, re.I):
            print(src[:500])
            print('---')
EOF
