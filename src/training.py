"""训练循环：字体渲染 → 蘑菇体仿真 → argmin 判定 → 错答强化 → 课程化解锁。

课程设计仿 hae.satoru.net：
    - 第 1 段只学 0~4，第 2 段解锁 0~9
    - 每练 1500 张用「从未见过的 5 种字体」测试，≥75% 才升段
    - 新段字符占练习量约一半（此处段 2 全量混练）
状态持久化到 data/state/，进程重启自动续训。
"""

import base64
import io
import json
import os
import threading
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .brain import MushroomBody

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONT_DIR = os.path.join(ROOT, "data", "fonts")
STATE_DIR = os.path.join(ROOT, "data", "state")
IMG_PX = 16

TRAIN_FONTS = ["Roboto", "OpenSans", "Lato", "Montserrat", "Oswald"]
TEST_FONTS = ["Merriweather", "NotoSans", "PTSans", "Ubuntu", "SourceSans3"]
LEVEL_CLASSES = {1: ["0", "1", "2", "3", "4"], 2: ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]}
TEST_EVERY = 1500
GATE = 0.75
ATTEMPT_MAX = 3  # 出题最多尝试次数（每次独立泊松采样，仿原站「一発/2回目/3回目/諦めた」）

_font_cache = {}


def get_font(name, size=12):
    key = (name, size)
    if key not in _font_cache:
        _font_cache[key] = ImageFont.truetype(os.path.join(FONT_DIR, name + ".ttf"), size)
    return _font_cache[key]


def render_char(ch, font_name, rng, size=IMG_PX):
    """渲染字符 + 随机形变（旋转/缩放/平移/剪切），返回 (256,) 灰度 0~1。"""
    scale = 6  # 先在大画布上画，再缩到 16×16
    canvas = Image.new("L", (size * scale, size * scale), 0)
    d = ImageDraw.Draw(canvas)
    f = get_font(font_name, int(size * scale * 0.8))
    bbox = d.textbbox((0, 0), ch, font=f)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    d.text(((canvas.width - w) / 2 - bbox[0], (canvas.height - h) / 2 - bbox[1]),
           ch, fill=255, font=f)
    canvas = canvas.transform(
        canvas.size, Image.AFFINE,
        (1, np.tan(rng.uniform(-0.15, 0.15)), -rng.uniform(-2, 2) * scale,
         np.tan(rng.uniform(-0.15, 0.15)), 1, -rng.uniform(-2, 2) * scale),
        resample=Image.BILINEAR)
    canvas = canvas.rotate(rng.uniform(-8, 8), resample=Image.BILINEAR, fillcolor=0)
    small = canvas.resize((size, size), Image.BILINEAR)
    arr = np.asarray(small, dtype=np.float32) / 255.0
    # 轻微尺度抖动
    arr = np.clip(arr * rng.uniform(0.9, 1.15), 0, 1)
    return arr.ravel()


def img_to_b64(arr256, upscale=8):
    im = Image.fromarray((arr256.reshape(IMG_PX, IMG_PX) * 255).astype(np.uint8))
    im = im.resize((IMG_PX * upscale, IMG_PX * upscale), Image.NEAREST)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


class Trainer:
    def __init__(self):
        os.makedirs(STATE_DIR, exist_ok=True)
        self.brain = MushroomBody(n_pn=685, n_kc=5177, n_classes=10)
        self.level = 1
        self.log_path = os.path.join(STATE_DIR, "log.jsonl")
        self.metrics_path = os.path.join(STATE_DIR, "metrics.json")
        self.recent = []          # 最近训练事件（供直播）
        self.crowd_log = []       # 访客手写作答档案
        self.quiz_log = []        # 自动出题档案
        self.level_events = []    # 升级事件
        self.train_curve = []     # [(t, train_acc)]
        self.test_curve = []      # [(t, test_acc)]
        self.mastery = {}         # {"0": 0.92, ...} 最新未见字体测试逐类正确率
        self.last_train_activity = None  # 最近一次训练呈现的活动快照（脑活动侧栏/机制页）
        self.lock = threading.Lock()
        self._load_state()
        self.running = True

    # ---------- 持久化 ----------
    def _brain_path(self):
        return os.path.join(STATE_DIR, "brain.npz")

    def _save_state(self):
        self.brain.save(self._brain_path())
        with open(self.metrics_path, "w", encoding="utf8") as f:
            json.dump({
                "level": self.level,
                "presented": self.brain.t_present,
                "train_curve": self.train_curve[-2000:],
                "test_curve": self.test_curve[-2000:],
                "mastery": self.mastery,
                "level_events": self.level_events[-50:],
                "crowd_log": self.crowd_log[-150:],
                "quiz_log": self.quiz_log[-150:],
            }, f, ensure_ascii=False)

    def _load_state(self):
        if os.path.exists(self._brain_path()):
            self.brain.load(self._brain_path())
        if os.path.exists(self.metrics_path):
            with open(self.metrics_path, encoding="utf8") as f:
                m = json.load(f)
            self.level = m["level"]
            self.train_curve = [tuple(x) for x in m.get("train_curve", [])]
            self.test_curve = [tuple(x) for x in m.get("test_curve", [])]
            self.mastery = m.get("mastery", {})
            self.level_events = m.get("level_events", [])
            self.crowd_log = m.get("crowd_log", [])
            self.quiz_log = m.get("quiz_log", [])

    def _append_log(self, ev):
        with open(self.log_path, "a", encoding="utf8") as f:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")

    # ---------- 测试 ----------
    def run_test(self):
        """用未见过的字体测试当前已学字符，返回 (总体正确率, 逐类正确率)。"""
        classes = [int(c) for c in LEVEL_CLASSES[self.level]]
        rng = np.random.default_rng(int(time.time()))
        n_per = 40
        ok = {c: 0 for c in classes}
        for c in classes:
            for fname in TEST_FONTS:
                for _ in range(n_per // len(TEST_FONTS)):
                    img = render_char(str(c), fname, rng)
                    _, mi = self.brain.present(img, rng)
                    if self.brain.decide(mi, classes) == c:
                        ok[c] += 1
        accs = {str(c): ok[c] / n_per for c in classes}
        total = sum(ok.values()) / (n_per * len(classes))
        return total, accs

    # ---------- 训练主循环 ----------
    def train_forever(self):
        rng = np.random.default_rng()
        level_classes = LEVEL_CLASSES[self.level]
        # 新段字符占一半练习量
        old = level_classes[:5] if self.level == 2 else level_classes
        new = level_classes[5:] if self.level == 2 else []
        window = []  # 滚动正确率（最近 500）
        since_save = 0

        while self.running:
            self.brain.t_present += 1
            t = self.brain.t_present
            if new and rng.random() < 0.5:
                ch = rng.choice(old + new)
            else:
                ch = rng.choice(old)
            fname = TRAIN_FONTS[rng.integers(len(TRAIN_FONTS))]
            img = render_char(ch, fname, rng)
            classes_now = [int(c) for c in LEVEL_CLASSES[self.level]]

            with self.lock:
                kc, mi = self.brain.present(img, rng)
                pred = self.brain.decide(mi, classes_now)
                correct = (pred == int(ch))
                if not correct:
                    self.brain.reinforce(kc, int(ch), pred)
                n_active = int(kc.sum())
                # 实时活动快照（脑活动侧栏 / 机制页区画条形图）
                self.last_train_activity = {
                    "t": t, "char": ch, "img": img_to_b64(img),
                    "pn_hz": self.brain.last_activity["pn_hz"],
                    "kc_hz": self.brain.last_activity["kc_hz"],
                    "kc_frac": self.brain.last_activity["kc_frac"],
                    "kc_active": n_active,
                    "comp": self.brain.comp_values(mi, n_active, classes_now),
                    "active_kc": np.nonzero(kc)[0][:600].astype(int).tolist(),
                    "last_reinforce": self.brain.last_reinforce,
                }

            window.append(1 if correct else 0)
            if len(window) > 500:
                window.pop(0)

            ev = {
                "kind": "train", "t": t, "char": ch, "font": fname,
                "pred": str(pred), "correct": correct,
                "img": img_to_b64(img),
            }
            self.recent.append(ev)
            self.recent = self.recent[-40:]
            if not correct or t % 25 == 0:
                self._append_log(ev)

            if len(window) == 500 and t % 100 == 0:
                self.train_curve.append([t, sum(window) / len(window)])

            if t % TEST_EVERY == 0:
                total, accs = self.run_test()
                self.test_curve.append([t, total])
                self.mastery = accs
                self._append_log({"kind": "test", "t": t, "acc": total})
                if total >= GATE and self.level < max(LEVEL_CLASSES):
                    self.level += 1
                    self.level_events.append({"t": t, "level": self.level, "acc": total})
                    self._append_log({"kind": "levelup", "t": t, "level": self.level})
                    level_classes = LEVEL_CLASSES[self.level]
                    old = level_classes[:5] if self.level == 2 else level_classes
                    new = level_classes[5:] if self.level == 2 else []
                    window = []

            since_save += 1
            if since_save >= 500:
                self._save_state()
                since_save = 0
            time.sleep(0.005)  # 节流：直播节奏友好，CPU 占用低

    # ---------- 出题（自动 + 访客手写，统一逻辑） ----------
    def quiz_next(self, rng=None):
        """出一道题：随机选当前课程段字符 + 未见过的字体（70%）/ 训练字体（30%）。"""
        rng = rng or np.random.default_rng(int(time.time() * 1e6) % 2**31)
        classes = LEVEL_CLASSES[self.level]
        ch = str(classes[rng.integers(len(classes))])
        pool = TEST_FONTS if rng.random() < 0.7 else TRAIN_FONTS
        fname = pool[rng.integers(len(pool))]
        img = render_char(ch, fname, rng)
        return {"char": ch, "font": fname, "pixels": img}

    def quiz(self, pixels, label, auto=False):
        """蝇读出一幅图：最多尝试 3 次（每次独立泊松采样），全部存档。

        auto=True 为自动出题模式（服务器出字自己答），False 为访客手写。
        结果标签：first / second / third / gaveup。
        """
        classes = [int(c) for c in LEVEL_CLASSES[self.level]]
        if label not in [str(c) for c in classes]:
            return None
        rng = np.random.default_rng(
            (int(time.time() * 1e6) % 2**31) ^ (self.brain.t_present * 7919))
        attempts = []
        with self.lock:
            for i in range(ATTEMPT_MAX):
                kc, mi = self.brain.present(pixels, rng)
                pred = self.brain.decide(mi, classes)
                act = self.brain.last_activity
                attempts.append({
                    "pred": str(pred),
                    "correct": pred == int(label),
                    "comp": self.brain.comp_values(mi, int(kc.sum()), classes),
                    "pn_hz": act["pn_hz"], "kc_hz": act["kc_hz"],
                    "kc_frac": act["kc_frac"], "kc_active": act["kc_active"],
                    "active_kc": np.nonzero(kc)[0][:400].astype(int).tolist(),
                })
                if attempts[-1]["correct"]:
                    break
        result = "gaveup"
        for i, a in enumerate(attempts):
            if a["correct"]:
                result = ["first", "second", "third"][i]
                break
        rec = {
            "kind": "auto" if auto else "crowd",
            "t": self.brain.t_present,
            "time": time.strftime("%H:%M:%S"),
            "char": label, "attempts": attempts, "result": result,
            "img": img_to_b64(pixels),
        }
        log = self.quiz_log if auto else self.crowd_log
        log.append(rec)
        del log[:-400]
        self._append_log(rec)
        return rec
