"""网站 v2 冒烟测试：本地起 uvicorn，逐个打 API 与页面。"""
import json
import os
import sys
import threading
import time
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import uvicorn

from src.web import app

BASE = "http://127.0.0.1:8021"
fails = []


def get(path, expect_json=True):
    with urllib.request.urlopen(BASE + path, timeout=15) as r:
        body = r.read()
        return (r.status, json.loads(body) if expect_json else body[:200])


def post(path, payload):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as r:
        return r.status, json.loads(r.read())


config = uvicorn.Config(app, host="127.0.0.1", port=8021, log_level="error")
server = uvicorn.Server(config)
t = threading.Thread(target=server.run, daemon=True)
t.start()
for _ in range(100):
    if server.started:
        break
    time.sleep(0.2)
assert server.started, "uvicorn failed to start"
print("server up")

# 页面
for path, name in [("/", "quiz"), ("/records", "records"),
                   ("/mechanism", "mechanism"), ("/about", "about")]:
    st, body = get(path, expect_json=False)
    assert st == 200 and b"FlyLab" in body, f"page {name} broken"
    print(f"page {name}: 200")

# API
st, s = get("/api/state")
assert s["classes_now"] and s["presented"] >= 0
print("state: level", s["level"], "presented", s["presented"])

st, r = get("/api/recent")
print("recent:", len(r["events"]), "events")

st, r = get("/api/kc-positions")
assert r["n"] == 5177 and len(r["xy"]) == 5177
print("kc-positions:", r["n"])

st, r = get("/api/quiz-next")
assert r["char"] in s["classes_now"] and len(r["pixels"]) == 256
print(f"quiz-next: char={r['char']} font={r['font']}")

st, q = post("/api/quiz", {"pixels": r["pixels"], "label": r["char"], "auto": True})
assert q["char"] == r["char"] and 1 <= len(q["attempts"]) <= 3
last = q["attempts"][-1]
assert 0 < last["kc_active"] < 5177 and last["pn_hz"] > 0
print(f"quiz(auto): result={q['result']} attempts={len(q['attempts'])} "
      f"kc_active={last['kc_active']} pn={last['pn_hz']:.1f}Hz comp={ {k: round(v,1) for k,v in list(last['comp'].items())[:3]} }")

# 手写：噪声图
rng = np.random.default_rng(1)
img = np.zeros(256, dtype=np.float32)
idx = rng.choice(256, 30, replace=False)
img[idx] = rng.uniform(0.5, 1.0, 30)
st, q2 = post("/api/quiz", {"pixels": img.tolist(), "label": r["char"], "auto": False})
print(f"quiz(crowd): result={q2['result']} attempts={len(q2['attempts'])}")

st, rec = get("/api/records")
assert rec["summary"]["total"] >= 2
print(f"records: total={rec['summary']['total']} first_rate={rec['summary']['first_rate']}")

time.sleep(2)  # 让训练线程跑几步
st, a = get("/api/activity")
if a["activity"]:
    act = a["activity"]
    print(f"activity: t={act['t']} char={act['char']} comp_keys={len(act['comp'])} "
          f"active_kc={len(act['active_kc'])} lr={act['last_reinforce'] is not None}")
    vals = sorted(act["comp"].values())
    print(f"comp range: {vals[0]:.1f} ~ {vals[-1]:.1f}")
else:
    print("activity: (not yet)")

server.should_exit = True
t.join(timeout=5)
print("SMOKE OK" if not fails else f"FAILS: {fails}")
