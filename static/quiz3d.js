/* FlyLab 3D 舞台：程序化果蝇（three.js，无外部模型）
 *
 * 忠实还原 hae.satoru.net 的核心演出：
 *   - 果蝇走向写字位，用前腿在地面逐笔"写出它的答案"
 *   - 答错：红色划掉，换一格重写（最多 3 次）
 *   - 答对：头部多巴胺绿色粒子爆发 + 绿光脉冲 + 振翅欢呼
 * 出题字符按当前答案渲染（服务器已算好每次尝试，这里按序演出）。
 *
 * 对外接口：
 *   Fly3D.ready                    — 是否可用（WebGL 失败时为 false）
 *   Fly3D.newProblem()             — 新题：清空地面字迹，苍蝇归位
 *   Fly3D.writeAttempt(pred, correct, slotIdx) → Promise — 演出一次尝试
 */
(function () {
  'use strict';

  function fail() { document.body.classList.add('no3d'); }

  var container = document.getElementById('stage3d');
  if (!container || typeof THREE === 'undefined') { fail(); return; }

  var renderer;
  try {
    renderer = new THREE.WebGLRenderer({ antialias: true });
  } catch (e) { fail(); return; }
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  container.appendChild(renderer.domElement);

  var scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0d1117);
  scene.fog = new THREE.Fog(0x0d1117, 14, 34);

  var camera = new THREE.PerspectiveCamera(45, 1, 0.1, 100);

  /* ---------- 灯光 ---------- */
  scene.add(new THREE.HemisphereLight(0x3a4048, 0x0c0e12, 1.0));
  var sun = new THREE.DirectionalLight(0xffffff, 1.1);
  sun.position.set(4, 8, 3);
  scene.add(sun);
  var dopamineLight = new THREE.PointLight(0x7ee787, 0, 6);
  scene.add(dopamineLight);
  var errorLight = new THREE.PointLight(0xff7b72, 0, 6);
  scene.add(errorLight);

  /* ---------- 地面 ---------- */
  var ground = new THREE.Mesh(
    new THREE.PlaneGeometry(80, 80),
    new THREE.MeshStandardMaterial({ color: 0x171b22, roughness: 0.95 }));
  ground.rotation.x = -Math.PI / 2;
  scene.add(ground);

  /* ---------- 地面写字层 ---------- */
  var W_CANVAS = 1280, H_CANVAS = 640;
  var W_WORLD = 12.8, H_WORLD = 6.4, W0 = -1.6, Z0 = 0.6;
  var writeCanvas = document.createElement('canvas');
  writeCanvas.width = W_CANVAS; writeCanvas.height = H_CANVAS;
  var wctx = writeCanvas.getContext('2d');
  var writeTex = new THREE.CanvasTexture(writeCanvas);
  var writePlane = new THREE.Mesh(
    new THREE.PlaneGeometry(W_WORLD, H_WORLD),
    new THREE.MeshBasicMaterial({ map: writeTex, transparent: true, depthWrite: false }));
  writePlane.rotation.x = -Math.PI / 2;
  writePlane.position.set(W0 + W_WORLD / 2, 0.006, Z0 + H_WORLD / 2);
  scene.add(writePlane);

  function worldToPx(x, z) {
    return [(x - W0) / W_WORLD * W_CANVAS, (z - Z0) / H_WORLD * H_CANVAS];
  }
  function clearWriting() { wctx.clearRect(0, 0, W_CANVAS, H_CANVAS); writeTex.needsUpdate = true; }
  function inkDot(x, z, color, r) {
    var p = worldToPx(x, z);
    wctx.fillStyle = color;
    wctx.beginPath(); wctx.arc(p[0], p[1], r || 7, 0, Math.PI * 2); wctx.fill();
    writeTex.needsUpdate = true;
  }
  function scribble(slotX, chW, chZ0, chH) {
    var a = worldToPx(slotX, chZ0), b = worldToPx(slotX + chW, chZ0 + chH);
    wctx.strokeStyle = '#ff5c52'; wctx.lineWidth = 14; wctx.lineCap = 'round';
    wctx.beginPath(); wctx.moveTo(a[0], a[1]); wctx.lineTo(b[0], b[1]); wctx.stroke();
    wctx.beginPath(); wctx.moveTo(b[0], a[1]); wctx.lineTo(a[0], b[1]); wctx.stroke();
    writeTex.needsUpdate = true;
  }

  /* ---------- 字符 → 地面书写路径 ---------- */
  var CHAR_W = 2.3, CHAR_H = 2.7;   // 槽位内字符尺寸（世界单位）
  var SLOT_GAP = 3.3;               // 槽位间距
  function slotRect(i) { return { x: 0.5 + i * SLOT_GAP, z: 2.9, w: CHAR_W, h: CHAR_H }; }

  function charPath(ch, slot) {
    var S = 88;
    var cv = document.createElement('canvas');
    cv.width = S; cv.height = S;
    var c = cv.getContext('2d');
    c.fillStyle = '#fff';
    c.font = 'bold 68px "Segoe UI", sans-serif';
    c.textAlign = 'center'; c.textBaseline = 'middle';
    c.fillText(ch, S / 2, S / 2 + 2);
    var img = c.getImageData(0, 0, S, S).data;
    var pts = [];
    for (var y = 0; y < S; y += 2)
      for (var x = 0; x < S; x += 2)
        if (img[(y * S + x) * 4 + 3] > 120)
          pts.push([slot.x + (x + 1) / S * slot.w, slot.z + (1 - (y + 1) / S) * slot.h]);
    if (!pts.length) return [];
    // 贪心最近邻排序，从靠近苍蝇的一端起笔
    var cur = pts[0], ordered = [cur], used = new Array(pts.length); used[0] = true;
    for (var n = 1; n < pts.length; n++) {
      var best = -1, bd = 1e9;
      for (var i = 0; i < pts.length; i++) {
        if (used[i]) continue;
        var d = (pts[i][0] - cur[0]) * (pts[i][0] - cur[0]) + (pts[i][1] - cur[1]) * (pts[i][1] - cur[1]);
        if (d < bd) { bd = d; best = i; }
      }
      used[best] = true; cur = pts[best]; ordered.push(cur);
    }
    return ordered;
  }

  /* ---------- 程序化果蝇 ---------- */
  var matBody = new THREE.MeshStandardMaterial({ color: 0xb08a56, roughness: 0.7 });
  var matDark = new THREE.MeshStandardMaterial({ color: 0x2a2018, roughness: 0.8 });
  var matEye = new THREE.MeshStandardMaterial({ color: 0xc41e2a, roughness: 0.35, metalness: 0.1 });
  var matWing = new THREE.MeshStandardMaterial({ color: 0xdfe8ef, roughness: 0.4, transparent: true, opacity: 0.28, side: THREE.DoubleSide });

  var fly = new THREE.Group();
  var body = new THREE.Group();   // bob 用内层
  fly.add(body);
  scene.add(fly);

  function blob(sx, sy, sz, x, y, z, mat) {
    var m = new THREE.Mesh(new THREE.SphereGeometry(1, 18, 14), mat || matBody);
    m.scale.set(sx, sy, sz); m.position.set(x, y, z);
    body.add(m); return m;
  }
  blob(0.42, 0.34, 0.52, 0, 0.56, 0.05);            // 胸
  blob(0.36, 0.32, 0.50, 0, 0.52, -0.62);           // 腹
  blob(0.34, 0.30, 0.34, 0, 0.52, 0.12, matDark);   // 颈/胸背纹
  var head = blob(0.26, 0.25, 0.26, 0, 0.60, 0.52); // 头
  var eyeL = blob(0.13, 0.16, 0.11, -0.16, 0.66, 0.62, matEye);
  var eyeR = blob(0.13, 0.16, 0.11, 0.16, 0.66, 0.62, matEye);
  blob(0.07, 0.06, 0.09, 0, 0.48, 0.68, matDark);   // 口器

  // 触角
  [-1, 1].forEach(function (s) {
    var a = new THREE.Mesh(new THREE.CylinderGeometry(0.012, 0.006, 0.3, 6), matDark);
    a.position.set(s * 0.09, 0.78, 0.62);
    a.rotation.set(0.7, 0, s * 0.35);
    body.add(a);
  });
  // 翅膀
  var wings = [];
  [-1, 1].forEach(function (s) {
    var w = new THREE.Mesh(new THREE.PlaneGeometry(0.62, 0.30), matWing);
    w.position.set(s * 0.26, 0.82, -0.18);
    w.rotation.set(-Math.PI / 2 + 0.25, 0, s * 0.35);
    w.userData.side = s;
    body.add(w); wings.push(w);
  });

  /* 腿：世界空间两点定段（肩→膝→足尖），每帧重算 */
  var L1 = 0.52, L2 = 0.58;
  var HIP = [
    [-0.24, 0.50, 0.42], [0.24, 0.50, 0.42],    // 前 pair
    [-0.30, 0.46, 0.02], [0.30, 0.46, 0.02],    // 中 pair
    [-0.24, 0.48, -0.40], [0.24, 0.48, -0.40]   // 后 pair
  ];
  var SIDE = [-1, 1, -1, 1, -1, 1];
  var PHASE = [0, Math.PI, Math.PI, 0, 0, Math.PI];  // 三角步态
  var REST_Z = [0.10, 0.10, 0.0, 0.0, -0.10, -0.10];
  var legs = HIP.map(function (h, i) {
    var femur = new THREE.Mesh(new THREE.CylinderGeometry(0.030, 0.022, 1, 6), matDark);
    var tibia = new THREE.Mesh(new THREE.CylinderGeometry(0.018, 0.012, 1, 6), matBody);
    var tip = new THREE.Mesh(new THREE.SphereGeometry(0.035, 8, 6), matDark);
    scene.add(femur); scene.add(tibia); scene.add(tip);
    return {
      hip: new THREE.Vector3(h[0], h[1], h[2]),
      rest: new THREE.Vector3(h[0] + SIDE[i] * 0.40, 0.02, h[2] + REST_Z[i] + SIDE[i] * 0.06),
      femur: femur, tibia: tibia, tip: tip,
      phase: PHASE[i], i: i
    };
  });
  var WRITER = 0;  // 前左腿写字

  var UP = new THREE.Vector3(0, 1, 0);
  function aimSegment(mesh, a, b, radiusScale) {
    var dir = new THREE.Vector3().subVectors(b, a);
    var len = dir.length();
    if (len < 1e-4) len = 1e-4;
    mesh.scale.set(radiusScale, len, radiusScale);
    mesh.position.copy(a).add(b).multiplyScalar(0.5);
    mesh.quaternion.setFromUnitVectors(UP, dir.normalize());
  }
  function solveKnee(hipW, footW, out) {
    var d = new THREE.Vector3().subVectors(footW, hipW);
    var dist = Math.min(Math.max(d.length(), Math.abs(L1 - L2) + 0.02), L1 + L2 - 0.02);
    var a = (L1 * L1 - L2 * L2 + dist * dist) / (2 * dist);
    var h = Math.sqrt(Math.max(0, L1 * L1 - a * a));
    var dir = d.normalize();
    out.copy(hipW).addScaledVector(dir, a);
    out.y += h;  // 膝盖朝上拱
    return out;
  }

  /* ---------- 状态机 ---------- */
  var flyPos = new THREE.Vector3(-1.2, 0, 3.1);
  var flyYaw = Math.PI * 0.15;
  fly.position.copy(flyPos); fly.rotation.y = flyYaw;
  var walkTarget = null;      // {x,z} 走向
  var walkResolve = null;
  var writer = null;          // {points, idx, color, r} 写字任务
  var buzzing = 0;            // 振翅计时
  var shakeT = 0;             // 抖动计时
  var bobT = 0;

  function walkTo(x, z) {
    return new Promise(function (res) {
      walkTarget = new THREE.Vector3(x, 0, z);
      walkResolve = res;
    });
  }

  var tmpA = new THREE.Vector3(), tmpB = new THREE.Vector3(), tmpK = new THREE.Vector3();
  var footScratch = new THREE.Vector3();
  var lookScratch = new THREE.Vector3();
  var clock = new THREE.Clock();

  function tick() {
    requestAnimationFrame(tick);
    var dt = Math.min(clock.getDelta(), 0.05);
    bobT += dt;

    /* 走路 */
    var moving = false;
    if (walkTarget) {
      tmpA.set(walkTarget.x - fly.position.x, 0, walkTarget.z - fly.position.z);
      var dist = tmpA.length();
      if (dist < 0.04) {
        walkTarget = null;
        if (walkResolve) { var r = walkResolve; walkResolve = null; r(); }
      } else {
        moving = true;
        tmpA.normalize();
        var spd = 1.7 * dt;
        fly.position.addScaledVector(tmpA, Math.min(spd, dist));
        var wantYaw = Math.atan2(tmpA.x, tmpA.z);
        var dy = wantYaw - fly.rotation.y;
        while (dy > Math.PI) dy -= Math.PI * 2;
        while (dy < -Math.PI) dy += Math.PI * 2;
        fly.rotation.y += dy * Math.min(1, dt * 7);
      }
    }

    /* 身体起伏 */
    body.position.y = moving ? Math.abs(Math.sin(bobT * 14)) * 0.045 : Math.sin(bobT * 2.2) * 0.012;
    if (shakeT > 0) {
      shakeT -= dt;
      body.rotation.z = Math.sin(bobT * 55) * 0.16 * (shakeT / 0.4);
    } else body.rotation.z = 0;

    /* 翅膀 */
    var flap = buzzing > 0 ? 40 : 3.2;
    if (buzzing > 0) buzzing -= dt;
    wings.forEach(function (w) {
      w.rotation.z = w.userData.side * (0.35 + Math.sin(bobT * flap) * (buzzing > 0 ? 0.5 : 0.08));
    });

    /* 写字推进：cur 平滑趋向下一个墨点，到达即落墨 */
    var writerFoot = null;
    if (writer) {
      var tgt = writer.points[writer.idx];
      if (!writer.cur) writer.cur = new THREE.Vector3(tgt[0], 0.03, tgt[1]);
      writer.cur.lerp(tmpB.set(tgt[0], 0.03, tgt[1]), Math.min(1, dt * 3.2));
      writerFoot = writer.cur;
      fly.updateMatrixWorld();
      var hipW = fly.localToWorld(tmpA.copy(legs[WRITER].hip));
      if (writer.cur.distanceTo(hipW) > L1 + L2 - 0.08) {
        // 够不着：苍蝇朝笔画挪过去
        tmpB.set(writer.cur.x - fly.position.x, 0, writer.cur.z - fly.position.z);
        if (tmpB.length() > 0.02) fly.position.addScaledVector(tmpB.normalize(), dt * 0.9);
      }
      if (writer.cur.distanceTo(tmpB.set(tgt[0], 0.03, tgt[1])) < 0.15) {
        inkDot(tgt[0], tgt[1], writer.color, writer.r);
        writer.idx++;
        if (writer.idx >= writer.points.length) {
          var done = writer.onDone;
          writer = null;
          if (done) done();
        }
      }
    }

    /* 腿世界摆位（肩→膝→足尖，每帧解析 IK） */
    for (var li = 0; li < legs.length; li++) {
      var L = legs[li];
      var hipW2 = fly.localToWorld(tmpA.copy(L.hip));
      var foot;
      if (writerFoot && li === WRITER) {
        foot = writerFoot;
      } else {
        foot = footScratch.copy(L.rest).applyMatrix4(fly.matrixWorld);
        foot.y = 0.02;
        if (moving) {
          var sw = Math.sin(bobT * 13 + L.phase);
          fly.getWorldDirection(tmpK);
          foot.addScaledVector(tmpK, sw * 0.16);
          foot.y = 0.02 + Math.max(0, sw) * 0.10;
        }
      }
      solveKnee(hipW2, foot, tmpK);
      aimSegment(L.femur, hipW2, tmpK, 1);
      aimSegment(L.tibia, tmpK, foot, 1);
      L.tip.position.copy(foot);
    }

    /* 相机跟随：写字时看字，平时看蝇前方 */
    var lookAt;
    if (writer && writer.focus) lookAt = writer.focus;
    else {
      fly.getWorldDirection(lookScratch);
      lookAt = lookScratch.multiplyScalar(1.2).add(fly.position);
      lookAt.y = 0.3;
    }
    camTargetPos.lerp(lookAt, Math.min(1, dt * 3));
    updateCamera();

    /* 光衰减 */
    dopamineLight.intensity = Math.max(0, dopamineLight.intensity - dt * 3);
    errorLight.intensity = Math.max(0, errorLight.intensity - dt * 4);

    updateParticles(dt);
    renderer.render(scene, camera);
  }

  /* ---------- 相机轨道（拖拽旋转 + 滚轮缩放） ---------- */
  var camYaw = -0.5, camPitch = 0.62, camDist = 8.6;
  var camTargetPos = new THREE.Vector3(0.6, 0.3, 3.2);
  function updateCamera() {
    var cy = Math.cos(camPitch), sy = Math.sin(camPitch);
    camera.position.set(
      camTargetPos.x + camDist * sy * Math.sin(camYaw),
      camTargetPos.y + camDist * cy,
      camTargetPos.z + camDist * sy * Math.cos(camYaw));
    camera.lookAt(camTargetPos);
  }
  (function () {
    var drag = null;
    var el = renderer.domElement;
    el.style.touchAction = 'none';
    el.addEventListener('pointerdown', function (e) { drag = { x: e.clientX, y: e.clientY }; el.setPointerCapture(e.pointerId); });
    el.addEventListener('pointermove', function (e) {
      if (!drag) return;
      camYaw -= (e.clientX - drag.x) * 0.006;
      camPitch = Math.min(1.35, Math.max(0.18, camPitch + (e.clientY - drag.y) * 0.005));
      drag = { x: e.clientX, y: e.clientY };
    });
    el.addEventListener('pointerup', function () { drag = null; });
    el.addEventListener('wheel', function (e) {
      e.preventDefault();
      camDist = Math.min(16, Math.max(3.5, camDist + e.deltaY * 0.01));
    }, { passive: false });
  })();

  function resize() {
    var w = container.clientWidth, h = container.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
  window.addEventListener('resize', resize);

  /* ---------- 多巴胺粒子 ---------- */
  var particles = [];
  function dopamineBurst(worldPos) {
    var N = 140;
    var geo = new THREE.BufferGeometry();
    var pos = new Float32Array(N * 3), vel = [];
    for (var i = 0; i < N; i++) {
      pos[i * 3] = worldPos.x; pos[i * 3 + 1] = worldPos.y; pos[i * 3 + 2] = worldPos.z;
      var a = Math.random() * Math.PI * 2, r = Math.random() * 0.8 + 0.3;
      vel.push([Math.cos(a) * r * 0.5, Math.random() * 1.6 + 0.6, Math.sin(a) * r * 0.5]);
    }
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    var mat = new THREE.PointsMaterial({
      color: 0x7ee787, size: 0.09, transparent: true, opacity: 1,
      blending: THREE.AdditiveBlending, depthWrite: false
    });
    var pts = new THREE.Points(geo, mat);
    scene.add(pts);
    particles.push({ pts: pts, vel: vel, life: 1.3, t: 0 });
    dopamineLight.position.copy(worldPos);
    dopamineLight.intensity = 3.2;
    buzzing = 0.9;
  }
  function updateParticles(dt) {
    for (var i = particles.length - 1; i >= 0; i--) {
      var P = particles[i];
      P.t += dt;
      var arr = P.pts.geometry.attributes.position.array;
      for (var j = 0; j < P.vel.length; j++) {
        arr[j * 3] += P.vel[j][0] * dt;
        arr[j * 3 + 1] += P.vel[j][1] * dt;
        arr[j * 3 + 2] += P.vel[j][2] * dt;
        P.vel[j][1] += 0.4 * dt;   // 上飘
      }
      P.pts.geometry.attributes.position.needsUpdate = true;
      P.pts.material.opacity = Math.max(0, 1 - P.t / P.life);
      if (P.t >= P.life) {
        scene.remove(P.pts);
        P.pts.geometry.dispose(); P.pts.material.dispose();
        particles.splice(i, 1);
      }
    }
  }

  /* ---------- 对外接口 ---------- */
  var START = { x: -1.2, z: 3.6 };

  function sleep(ms) { return new Promise(function (r) { setTimeout(r, ms); }); }

  window.Fly3D = {
    ready: true,
    newProblem: function () {
      clearWriting();
      writer = null;
      return walkTo(START.x, START.z);
    },
    writeAttempt: function (pred, correct, slotIdx) {
      var slot = slotRect(slotIdx);
      var standX = slot.x - 1.35, standZ = slot.z + 1.15;
      return walkTo(standX, standZ).then(function () {
        return new Promise(function (resolve) {
          var pts = charPath(pred, slot);
          if (!pts.length) { resolve(); return; }
          writer = {
            points: pts, idx: 0, lastPt: null,
            color: '#f0f6fc', r: 6.5,
            focus: new THREE.Vector3(slot.x + slot.w / 2, 0.3, slot.z + slot.h / 2),
            onDone: function () {
              writer = null;
              sleep(300).then(function () {
                if (correct) {
                  var headW = fly.localToWorld(new THREE.Vector3(0, 0.6, 0.5));
                  dopamineBurst(headW);
                  sleep(1100).then(resolve);
                } else {
                  scribble(slot.x, slot.w, slot.z, slot.h);
                  errorLight.position.set(slot.x + slot.w / 2, 0.8, slot.z + slot.h / 2);
                  errorLight.intensity = 2.4;
                  shakeT = 0.4;
                  sleep(600).then(resolve);
                }
              });
            }
          };
        });
      });
    }
  };

  resize();
  tick();
})();
