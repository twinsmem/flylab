"""冒烟测试：渲染→前向→判读→学习效果快速验证（3000 张内应显著高于随机）。"""
import sys, time
import numpy as np
sys.path.insert(0, ".")
from src.brain import MushroomBody
from src.training import render_char, TRAIN_FONTS, TEST_FONTS, LEVEL_CLASSES

t0 = time.time()
brain = MushroomBody(n_classes=10)
print(f"init: {time.time()-t0:.1f}s, theta={brain.kc_theta:.2f}")

rng = np.random.default_rng(7)
classes = [int(c) for c in LEVEL_CLASSES[1]]  # 0~4

# 稀疏度抽查
img = render_char("3", TRAIN_FONTS[0], rng)
kc, mi = brain.present(img, rng)
print(f"KC sparsity: {kc.mean():.3f} (target 0.10)")

# 学习曲线：只练 0~4
t0 = time.time()
n = 3000
window = []
marks = {}
for i in range(n):
    ch = str(rng.integers(0, 5))
    img = render_char(ch, TRAIN_FONTS[rng.integers(len(TRAIN_FONTS))], rng)
    kc, mi = brain.present(img, rng)
    pred = brain.decide(mi, classes)
    ok = pred == int(ch)
    if not ok:
        brain.reinforce(kc, int(ch), pred)
    window.append(ok)
    if (i + 1) in (100, 500, 1500, 3000):
        marks[i + 1] = sum(window[-500:]) / len(window[-500:])
print(f"train {n} imgs in {time.time()-t0:.1f}s ({n/(time.time()-t0):.0f} img/s)")
for k, v in marks.items():
    print(f"  after {k}: acc(last500)={v:.3f}")

# 未见字体测试
ok, tot = 0, 0
for c in classes:
    for _ in range(100):
        img = render_char(str(c), TEST_FONTS[rng.integers(len(TEST_FONTS))], rng)
        _, mi = brain.present(img, rng)
        ok += (brain.decide(mi, classes) == int(c)); tot += 1
print(f"unseen-font test acc: {ok/tot:.3f} (random = 0.20)")
