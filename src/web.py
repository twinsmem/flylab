"""FastAPI 入口：5 页网站（出题/记录/机制/其他/物理3D）+ 训练直播 + 出题 API。

启动：
    cd /opt/flylab && ./venv/bin/python -m src.web
默认监听 0.0.0.0:8000，局域网浏览器直接访问 http://<服务器IP>:8000

页面结构（仿 hae.satoru.net）：
    /          出题页  quiz.html      —— 自动出题 + 手写作答 + 脑活动侧栏 + 训练流
    /records   记录页  records.html   —— 成绩卡 + 曲线 + 掌握度 + 作答档案
    /mechanism 机制页  mechanism.html —— 区画输出条形图 + KC 解剖散点 + 机制文档
    /about     其他页  about.html     —— 项目说明 + 依赖清单
    /mj        物理3D  mj.html        —— flybody 官方任务环境（MuJoCo 物理渲染推流）
    /mj3d      3D对比  mj3d.html      —— 同模型 three.js 网格渲染（WS 位姿流，60fps）
"""

import asyncio
import os
import threading
import time

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
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
    global trainer, mj_sim, mj_fly
    trainer = Trainer()
    threading.Thread(target=trainer.train_forever, daemon=True, name="trainer").start()
    # three.js 位姿流仿真（/mj3d 页，本地 mujoco 原生驱动）
    # 注意：先启动 mjsim（默认 GL 后端），再启动 flybody_sim（Linux 上切 osmesa）
    try:
        from .mjsim import MjFly
        mj_fly = MjFly()
    except Exception as e:  # noqa: BLE001
        print(f"[mj3d] 位姿仿真不可用: {e}")
        mj_fly = None
    # flybody 官方任务环境（/mj 页，MJPEG 渲染流；服务器 /opt/flybody venv 提供）
    try:
        from .flybody_sim import FlybodySim
        mj_sim = FlybodySim()
    except Exception as e:  # noqa: BLE001 - 缺依赖时不影响主站
        print(f"[mj] flybody 仿真不可用: {e}")
        mj_sim = None


mj_sim = None  # type: ignore
mj_fly = None  # type: ignore


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


@app.get("/mj")
def page_mj():
    return FileResponse(os.path.join(STATIC, "mj.html"))


@app.get("/mj3d")
def page_mj3d():
    return FileResponse(os.path.join(STATIC, "mj3d.html"))


# ---------- 物理3D（flybody 官方任务环境 · MJPEG 推流） ----------
@app.get("/mj/stream.mjpg")
def mj_stream():
    """MJPEG 流：multipart/x-mixed-replace，浏览器 <img> 直接消费。
    帧率由仿真线程产生速度决定（osmesa 软渲染约 2-3 fps，慢动作效果）。"""
    if mj_sim is None:
        return JSONResponse({"error": "flybody 仿真不可用"}, status_code=503)

    def gen():
        boundary = b"--frame\r\n"
        last = None
        while True:
            jpg = mj_sim.jpeg()
            if jpg is not None and jpg is not last:
                last = jpg
                yield (boundary + b"Content-Type: image/jpeg\r\nContent-Length: "
                       + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
            else:
                time.sleep(0.02)

    from fastapi.responses import StreamingResponse
    return StreamingResponse(gen(),
                             media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/mj/stats")
def mj_stats():
    """仿真统计（fps / 仿真时间 / 累计步 / 回报 / 相机）。"""
    if mj_sim is None:
        return {"error": "flybody 仿真不可用"}
    return mj_sim.stats()


@app.post("/mj/camera")
def mj_camera(cam_id: int):
    """切换渲染相机（0=跟随镜头, 1=轨道全景, 2+ 为模型内置机位）。"""
    if mj_sim is None:
        return JSONResponse({"error": "flybody 仿真不可用"}, status_code=503)
    mj_sim.set_camera(cam_id)
    return {"cam": cam_id}


# ---------- three.js 版位姿 WebSocket（/mj3d 页） ----------
@app.websocket("/ws/mj")
async def ws_mj(ws: WebSocket):
    """二进制位姿流：先发 JSON 元信息，之后每帧 nbody×7 float32 (pos+quat[wxyz])，
    周期性穿插 JSON 统计帧（rtf/sim_t/ncon）。"""
    await ws.accept()
    if mj_fly is None:
        await ws.send_json({"error": "mujoco 不可用"})
        await ws.close()
        return
    await ws.send_json({
        "nbody": mj_fly.nbody,
        "names": mj_fly.body_names,
        "thorax": mj_fly.thorax,
    })
    try:
        stat_cnt = 0
        while True:
            snap = mj_fly.snapshot()
            if snap is None:
                await asyncio.sleep(0.05)
                continue
            buf, sim_t, rtf, ncon = snap
            await ws.send_bytes(buf.tobytes())
            stat_cnt += 1
            if stat_cnt >= 60:   # 每 60 帧（约1s）发一次统计
                stat_cnt = 0
                await ws.send_json({"rtf": rtf, "sim_t": sim_t, "ncon": ncon})
            await asyncio.sleep(1 / 60)
    except WebSocketDisconnect:
        pass
    except Exception:
        try:
            await ws.close()
        except Exception:
            pass


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
