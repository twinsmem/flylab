"""FastAPI 入口：训练线程 + 直播 API + 众包测试接口。

启动：
    cd ~/flylab && ./venv/bin/python -m src.web
默认监听 0.0.0.0:8000，局域网浏览器直接访问 http://<服务器IP>:8000
"""

import os
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from .training import Trainer, IMG_PX, LEVEL_CLASSES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC = os.path.join(ROOT, "static")

app = FastAPI(title="FlyLab")
trainer: Trainer = None  # type: ignore


class Drawing(BaseModel):
    """16×16 灰度图，展平 256 个 0~1 浮点（前端 canvas 缩放后上报）。"""
    pixels: list[float]


@app.on_event("startup")
def _start():
    global trainer
    trainer = Trainer()
    import threading
    threading.Thread(target=trainer.train_forever, daemon=True, name="trainer").start()


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))


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


@app.get("/api/crowd")
def crowd():
    return {"events": trainer.crowd_log[-30:]}


@app.post("/api/crowd")
def crowd_test(d: Drawing):
    if len(d.pixels) != IMG_PX * IMG_PX:
        return JSONResponse({"error": f"need {IMG_PX*IMG_PX} values"}, status_code=400)
    import numpy as np
    arr = np.clip(np.asarray(d.pixels, dtype=np.float32), 0, 1)
    rec = trainer.crowd_test(arr)
    return {"pred": rec["pred"], "top3": rec["top3"]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
