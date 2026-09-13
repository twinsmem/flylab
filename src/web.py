"""FastAPI 入口：4 页网站（出题/记录/机制/其他）+ 训练直播 + 出题 API。

启动：
    cd /opt/flylab && ./venv/bin/python -m src.web
默认监听 0.0.0.0:8000，局域网浏览器直接访问 http://<服务器IP>:8000

页面结构（仿 hae.satoru.net）：
    /          出题页  quiz.html      —— 自动出题 + 手写作答 + 脑活动侧栏 + 训练流
    /records   记录页  records.html   —— 成绩卡 + 曲线 + 掌握度 + 作答档案
    /mechanism 机制页  mechanism.html —— 区画输出条形图 + KC 解剖散点 + 机制文档
    /about     其他页  about.html     —— 项目说明 + 依赖清单
"""

import os
import threading

import numpy as np
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .training import Trainer, IMG_PX, LEVEL_CLASSES, img_to_b64

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC = os.path.join(ROOT, "static")

app = FastAPI(title="FlyLab")
app.mount("/static", StaticFiles(directory=STATIC), name="static")
trainer: Trainer = None  # type: ignore
_kc_xy_cache = None


class Drawing(BaseModel):
    """16×16 灰度图（展平 256 个 0~1 浮点）+ 题目字符；auto=True 为自动出题模式。"""
    pixels: list[float]
    label: str
    auto: bool = False


@app.on_event("startup")
def _start():
    global trainer
    trainer = Trainer()
    threading.Thread(target=trainer.train_forever, daemon=True, name="trainer").start()


# ---------- 页面 ----------
@app.get("/")
def page_quiz():
    return FileResponse(os.path.join(STATIC, "quiz.html"))


@app.get("/records")
def page_records():
    return FileResponse(os.path.join(STATIC, "records.html"))


@app.get("/mechanism")
def page_mechanism():
    return FileResponse(os.path.join(STATIC, "mechanism.html"))


@app.get("/about")
def page_about():
    return FileResponse(os.path.join(STATIC, "about.html"))


# ---------- 数据 API ----------
@app.get("/api/state")
def state():
    return {
        "level": trainer.level,
        "presented": trainer.brain.t_present,
        "train_curve": trainer.train_curve[-300:],
        "test_curve": trainer.test_curve[-300:],
        "mastery": trainer.mastery,
        "level_events": trainer.level_events,
        "classes_now": LEVEL_CLASSES[trainer.level],
    }


@app.get("/api/recent")
def recent():
    return {"events": trainer.recent[-30:]}


@app.get("/api/activity")
def activity():
    """最近一次训练呈现的脑活动快照（约每 50ms 更新）。"""
    return {"activity": trainer.last_train_activity}


@app.get("/api/kc-positions")
def kc_positions():
    """KC 二维示意位置（双半球），前端缓存后用于两张散点图。"""
    global _kc_xy_cache
    if _kc_xy_cache is None:
        xy = trainer.brain.kc_xy
        _kc_xy_cache = [[round(float(x), 3), round(float(y), 3)] for x, y in xy]
    return {"n": len(_kc_xy_cache), "xy": _kc_xy_cache}


@app.get("/api/quiz-next")
def quiz_next():
    """出一道题：返回字符 + 渲染图 + 像素（自动模式原样回传，手写模式忽略像素）。"""
    q = trainer.quiz_next()
    return {
        "char": q["char"], "font": q["font"],
        "img": img_to_b64(q["pixels"]),
        "pixels": [round(float(v), 3) for v in q["pixels"]],
    }


@app.post("/api/quiz")
def quiz(d: Drawing):
    if d.auto:
        pass  # 自动模式：pixels 为服务器渲染的题目图
    elif len(d.pixels) != IMG_PX * IMG_PX:
        return JSONResponse({"error": f"need {IMG_PX*IMG_PX} values"}, status_code=400)
    arr = np.clip(np.asarray(d.pixels, dtype=np.float32), 0, 1)
    rec = trainer.quiz(arr, d.label, auto=d.auto)
    if rec is None:
        return JSONResponse({"error": "label not in current level classes"}, status_code=400)
    return {
        "char": rec["char"], "result": rec["result"],
        "attempts": [{"pred": a["pred"], "correct": a["correct"],
                      "comp": a["comp"], "pn_hz": a["pn_hz"], "kc_hz": a["kc_hz"],
                      "kc_frac": a["kc_frac"], "kc_active": a["kc_active"]}
                     for a in rec["attempts"]],
    }


@app.get("/api/records")
def records():
    """记录页数据：作答档案（自动+手写）+ 汇总统计。"""
    def _valid(evs):
        return [e for e in evs if "attempts" in e][-30:]

    auto, crowd = _valid(trainer.quiz_log), _valid(trainer.crowd_log)
    all_recs = [e for e in trainer.quiz_log + trainer.crowd_log if "attempts" in e]
    n = len(all_recs)
    first = sum(1 for e in all_recs if e["result"] == "first")
    within3 = sum(1 for e in all_recs if e["result"] != "gaveup")
    return {
        "summary": {
            "total": n,
            "first_rate": first / n if n else None,
            "within3_rate": within3 / n if n else None,
        },
        "auto": auto[::-1],
        "crowd": crowd[::-1],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
