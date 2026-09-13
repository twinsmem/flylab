/* FlyLab /mj —— flybody(MuJoCo 物理仿真) 位姿流 + three.js 渲染。
 *
 * 链路：服务端 mujoco 线程 --WS(二进制 nbody×7 float32)--> 本文件
 *   1. 拉取 /static/mj/fruitfly.xml，解析 defaults/材质/刚体树/网格 geom
 *   2. 并行加载 meshes/*.stl，焊接顶点 + 面法线累积 → 平滑法线
 *   3. 每刚体一个 Group，WS 帧直接写入 pos/quat，three.js 自己算世界矩阵
 */
(function () {
  "use strict";
  var stage = document.getElementById("stage3d");
  var $fps = document.getElementById("fpsBadge");
  var $simT = document.getElementById("simT");
  var $rtf = document.getElementById("rtf");
  var $ncon = document.getElementById("ncon");

  if (!window.THREE || !window.WebSocket) {
    document.body.classList.add("nomj");
    if ($fps) $fps.textContent = "three.js 不可用";
    return;
  }

  /* ---------- 场景基础 ---------- */
  var scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0d1117);
  scene.fog = new THREE.Fog(0x0d1117, 8, 30);

  var camera = new THREE.PerspectiveCamera(45, 1, 0.005, 100);
  var renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  stage.appendChild(renderer.domElement);

  scene.add(new THREE.HemisphereLight(0x9db4c9, 0x1c2318, 0.9));
  var sun = new THREE.DirectionalLight(0xffffff, 0.85);
  sun.position.set(2, 5, 1.5);
  scene.add(sun);
  var rim = new THREE.DirectionalLight(0x88aaff, 0.25);
  rim.position.set(-3, 2, -2);
  scene.add(rim);

  /* 地面：暗色板 + 1mm 网格 */
  scene.add(new THREE.Mesh(
    new THREE.PlaneGeometry(60, 60),
    new THREE.MeshStandardMaterial({ color: 0x11151b, roughness: 0.97 })
  ).rotateX(-Math.PI / 2));
  var grid = new THREE.GridHelper(10, 100, 0x2d3742, 0x1b222b);
  grid.material.transparent = true;
  grid.material.opacity = 0.55;
  scene.add(grid);

  /* 胸部接触投影（假 AO 小圆片，跟随胸部） */
  var shadow = new THREE.Mesh(
    new THREE.CircleGeometry(0.09, 24),
    new THREE.MeshBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.35 })
  );
  shadow.rotation.x = -Math.PI / 2;
  shadow.position.y = 0.001;
  scene.add(shadow);

  /* ---------- 轨道相机（拖拽旋转 / 滚轮缩放 / 双击复位） ---------- */
  var orbit = { yaw: 0.7, pitch: 0.55, dist: 1.6 };
  var orbitHome = { yaw: 0.7, pitch: 0.55, dist: 1.6 };
  var followTarget = new THREE.Vector3(0, 0, 0.1);
  var dragging = false, lastX = 0, lastY = 0;

  function applyCamera() {
    var cp = Math.cos(orbit.pitch), sp = Math.sin(orbit.pitch);
    camera.position.set(
      followTarget.x + orbit.dist * cp * Math.sin(orbit.yaw),
      followTarget.y + orbit.dist * sp,
      followTarget.z + orbit.dist * cp * Math.cos(orbit.yaw));
    camera.lookAt(followTarget);
  }
  stage.addEventListener("pointerdown", function (e) {
    dragging = true; lastX = e.clientX; lastY = e.clientY;
    stage.setPointerCapture(e.pointerId);
  });
  stage.addEventListener("pointermove", function (e) {
    if (!dragging) return;
    orbit.yaw -= (e.clientX - lastX) * 0.008;
    orbit.pitch = Math.max(0.05, Math.min(1.5, orbit.pitch + (e.clientY - lastY) * 0.006));
    lastX = e.clientX; lastY = e.clientY;
  });
  stage.addEventListener("pointerup", function () { dragging = false; });
  stage.addEventListener("wheel", function (e) {
    e.preventDefault();
    orbit.dist = Math.max(0.15, Math.min(25, orbit.dist * Math.exp(e.deltaY * 0.0012)));
  }, { passive: false });
  stage.addEventListener("dblclick", function () {
    orbit.yaw = orbitHome.yaw; orbit.pitch = orbitHome.pitch; orbit.dist = orbitHome.dist;
  });

  function resize() {
    var w = stage.clientWidth, h = stage.clientHeight;
    camera.aspect = w / h; camera.updateProjectionMatrix();
    renderer.setSize(w, h);
  }
  window.addEventListener("resize", resize);

  /* ---------- XML 解析：defaults / 材质 / 刚体树 ---------- */
  function attrNums(el, name, dflt) {
    var v = el.getAttribute(name);
    if (v == null) return dflt;
    return v.trim().split(/\s+/).map(Number);
  }

  /* class → 已合并的 {geom:{}, mesh:{}} 属性表（沿 defaults 树自根向下合并） */
  function buildDefaults(doc) {
    var classGeom = {}, classMesh = {};
    function walk(el, parentGeom, parentMesh) {
      var g = Object.assign({}, parentGeom), m = Object.assign({}, parentMesh);
      var i, ch;
      for (i = 0; i < el.children.length; i++) {
        ch = el.children[i];
        if (ch.tagName === "geom") Object.assign(g, attrObj(ch));
        else if (ch.tagName === "mesh") Object.assign(m, attrObj(ch));
      }
      var cname = el.getAttribute("class");
      if (cname) { classGeom[cname] = g; classMesh[cname] = m; }
      for (i = 0; i < el.children.length; i++) {
        ch = el.children[i];
        if (ch.tagName === "default") walk(ch, g, m);
      }
    }
    function attrObj(el) {
      var o = {}, as = el.attributes;
      for (var i = 0; i < as.length; i++) o[as[i].name] = as[i].value;
      return o;
    }
    var root = doc.querySelector("mujoco > default");
    if (root) walk(root, {}, {});
    return { geom: classGeom, mesh: classMesh };
  }

  function parseXML(text) {
    var doc = new DOMParser().parseFromString(text, "text/xml");
    var defs = buildDefaults(doc);
    var baseGeom = defs.geom[""] || {};   // 根 default（无 class 名）

    /* 材质表 */
    var materials = {};
    doc.querySelectorAll("asset > material").forEach(function (el) {
      materials[el.getAttribute("name")] = {
        rgba: attrNums(el, "rgba", [0.6, 0.6, 0.6, 1]),
        shininess: parseFloat(el.getAttribute("shininess") || "0.5"),
      };
    });
    /* 网格名 → 文件名 + 统一缩放（root default mesh.scale） */
    var meshScale = (defs.mesh[""] && defs.mesh[""].scale || "0.1 0.1 0.1")
      .trim().split(/\s+/).map(Number);
    var meshFiles = {};
    doc.querySelectorAll("asset > mesh").forEach(function (el) {
      meshFiles[el.getAttribute("name")] = el.getAttribute("file").replace(/\.stl$/, "");
    });

    /* 刚体树：索引与 MuJoCo body id 对齐（0=world 虚拟节点） */
    var bodies = [{ name: "world", parent: -1, pos: [0, 0, 0], quat: [1, 0, 0, 0], cls: null, geoms: [] }];
    function addBody(el, parentIdx, inheritedClass) {
      var cls = el.getAttribute("childclass") || inheritedClass;
      var idx = bodies.length;
      bodies.push({
        name: el.getAttribute("name"),
        parent: parentIdx,
        pos: attrNums(el, "pos", [0, 0, 0]),
        quat: attrNums(el, "quat", [1, 0, 0, 0]),
        cls: cls,
        geoms: [],
      });
      var i, ch;
      for (i = 0; i < el.children.length; i++) {
        ch = el.children[i];
        if (ch.tagName === "body") addBody(ch, idx, cls);
      }
    }
    var world = doc.querySelector("worldbody");
    var i, ch;
    for (i = 0; i < world.children.length; i++) {
      ch = world.children[i];
      if (ch.tagName === "body") addBody(ch, 0, null);
    }

    /* geom 收集：仅渲染型 mesh geom（type 显式或经 class 链为 mesh） */
    function collectGeoms(bodyEl, body) {
      var ownClass = body.cls;
      var i, ch;
      for (i = 0; i < bodyEl.children.length; i++) {
        ch = bodyEl.children[i];
        if (ch.tagName !== "geom") continue;
        var clsName = ch.getAttribute("class") || ownClass;
        var cg = clsName && defs.geom[clsName] ? defs.geom[clsName] : baseGeom;
        var type = ch.getAttribute("type") || cg.type || "sphere";
        if (type !== "mesh") continue;
        var meshName = ch.getAttribute("mesh");
        if (!meshName || !meshFiles[meshName]) continue;
        var matName = ch.getAttribute("material") || cg.material || null;
        body.geoms.push({
          file: meshFiles[meshName],
          pos: attrNums(ch, "pos", [0, 0, 0]),
          quat: attrNums(ch, "quat", [1, 0, 0, 0]),
          mat: matName,
          rgba: ch.getAttribute("rgba"),
        });
      }
    }
    /* 遍历 XML body 元素（与 addBody 同序，bodies[0] 为虚拟 world）把 geom 挂到 bodies */
    var bi = 1;
    function attach(el, inheritedClass) {
      var cls = el.getAttribute("childclass") || inheritedClass;
      collectGeoms(el, bodies[bi++]);
      var i, ch;
      for (i = 0; i < el.children.length; i++) {
        ch = el.children[i];
        if (ch.tagName === "body") attach(ch, cls);
      }
    }
    for (i = 0; i < world.children.length; i++) {
      ch = world.children[i];
      if (ch.tagName === "body") attach(ch, null);
    }
    return { bodies: bodies, meshScale: meshScale, materials: materials };
  }

  /* ---------- STL 加载：焊接顶点 + 平滑法线 ---------- */
  function loadSTL(url, scale) {
    return fetch(url).then(function (r) {
      if (!r.ok) throw new Error(url + " -> " + r.status);
      return r.arrayBuffer();
    }).then(function (buf) {
      var dv = new DataView(buf);
      var nTri = dv.getUint32(80, true);
      var pos = new Float32Array(nTri * 9);
      var o = 84;
      for (var t = 0; t < nTri; t++) {
        for (var k = 0; k < 9; k++) {
          pos[t * 9 + k] = dv.getFloat32(o + 12 + k * 4, true) * scale;
        }
        o += 50;
      }
      /* 焊接（0.5μm 栅格）+ 面法线累积 → 平滑 */
      var nV = nTri * 3;
      var map = new Map(), idx = new Uint32Array(nV), verts = [];
      for (var v = 0; v < nV; v++) {
        var key = Math.round(pos[v * 3] * 2e6) + "," +
                  Math.round(pos[v * 3 + 1] * 2e6) + "," +
                  Math.round(pos[v * 3 + 2] * 2e6);
        var j = map.get(key);
        if (j === undefined) {
          j = verts.length / 3; map.set(key, j);
          verts.push(pos[v * 3], pos[v * 3 + 1], pos[v * 3 + 2]);
        }
        idx[v] = j;
      }
      var vn = new Float32Array(verts.length);
      for (var t2 = 0; t2 < nV; t2 += 3) {
        var a = idx[t2] * 3, b = idx[t2 + 1] * 3, c = idx[t2 + 2] * 3;
        var ux = verts[b] - verts[a], uy = verts[b + 1] - verts[a + 1], uz = verts[b + 2] - verts[a + 2];
        var wx = verts[c] - verts[a], wy = verts[c + 1] - verts[a + 1], wz = verts[c + 2] - verts[a + 2];
        var nx = uy * wz - uz * wy, ny = uz * wx - ux * wz, nz = ux * wy - uy * wx;
        for (var s = 0; s < 3; s++) {
          var vi = idx[t2 + s] * 3;
          vn[vi] += nx; vn[vi + 1] += ny; vn[vi + 2] += nz;
        }
      }
      for (var v2 = 0; v2 < vn.length; v2 += 3) {
        var l = Math.hypot(vn[v2], vn[v2 + 1], vn[v2 + 2]) || 1;
        vn[v2] /= l; vn[v2 + 1] /= l; vn[v2 + 2] /= l;
      }
      var g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(verts), 3));
      g.setAttribute("normal", new THREE.BufferAttribute(vn, 3));
      g.setIndex(new THREE.BufferAttribute(idx, 1));
      return g;
    });
  }

  function makeMaterial(rgba, shininess) {
    /* 官方模型角质层为半透明；three.js 多层壳体半透明会互相穿透显脏，强制不透明 */
    return new THREE.MeshStandardMaterial({
      color: new THREE.Color(rgba[0], rgba[1], rgba[2]),
      roughness: Math.max(0.15, 1 - shininess),
      metalness: 0.0,
      transparent: false,
      opacity: 1.0,
      side: THREE.DoubleSide,
    });
  }

  /* ---------- 构建 + WS 接入 ---------- */
  var groups = null, thoraxIdx = -1, t0 = performance.now(), frames = 0, fpsT = t0;

  function buildScene(parsed) {
    var geoms = [];
    groups = [null];   // index 0 = world（虚拟）
    parsed.bodies.forEach(function (b, i) {
      if (i === 0) return;
      var g = new THREE.Group();
      b.geoms.forEach(function (ge) {
        var mdef = ge.mat ? parsed.materials[ge.mat] : null;
        var rgba = ge.rgba ? ge.rgba.trim().split(/\s+/).map(Number)
          : (mdef ? mdef.rgba : [0.55, 0.45, 0.35, 1]);
        var shin = mdef ? mdef.shininess : 0.5;
        geoms.push({ file: ge.file, rgba: rgba, shininess: shin,
                     body: i, pos: ge.pos, quat: ge.quat });
      });
      groups.push(g);   // 位姿归零；首帧 WS 到达后直接写世界位姿
      if (b.name === "thorax") thoraxIdx = i;
    });
    /* MuJoCo WS 帧给的是每个刚体的世界系位姿(xpos/xquat) → 全部平铺直挂 scene，
       若按身体树嵌套会把世界位姿再乘父级变换，导致整蝇炸散 */
    parsed.bodies.forEach(function (b, i) {
      if (i === 0) return;
      scene.add(groups[i]);
    });
    /* 材质缓存 + geom mesh 实例化 */
    var matCache = {};
    var fileSet = {};
    geoms.forEach(function (ge) {
      var key = ge.rgba.join(",") + "|" + ge.shininess;
      if (!matCache[key]) matCache[key] = makeMaterial(ge.rgba, ge.shininess);
      ge.material = matCache[key];
      fileSet[ge.file] = true;
    });
    var files = Object.keys(fileSet);
    var done = 0;
    $fps.textContent = "加载网格 0/" + files.length;
    var geoCache = {};
    return Promise.all(files.map(function (f) {
      return loadSTL("/static/mj/meshes/" + f + ".stl", parsed.meshScale[0] || 0.1)
        .then(function (geo) { geoCache[f] = geo; })
        .catch(function () { geoCache[f] = null; })
        .then(function () {
          done++;
          $fps.textContent = "加载网格 " + done + "/" + files.length;
        });
    })).then(function () {
      geoms.forEach(function (ge) {
        if (!geoCache[ge.file]) return;
        var mesh = new THREE.Mesh(geoCache[ge.file], ge.material);
        mesh.position.fromArray(ge.pos);
        mesh.quaternion.set(ge.quat[1], ge.quat[2], ge.quat[3], ge.quat[0]);  // wxyz -> xyzw
        groups[ge.body].add(mesh);
      });
      $fps.textContent = "连接 WS…";
    });
  }

  function connectWS(parsed) {
    var proto = location.protocol === "https:" ? "wss://" : "ws://";
    var ws = new WebSocket(proto + location.host + "/ws/mj");
    ws.binaryType = "arraybuffer";
    var latest = null;
    ws.onmessage = function (ev) {
      if (typeof ev.data === "string") {
        var msg = JSON.parse(ev.data);
        if (msg.error) {
          document.body.classList.add("nomj");
          $fps.textContent = msg.error;
          return;
        }
        if (msg.sim_t === undefined) return;   // (重)连时的 nbody 元数据帧
        $simT.textContent = msg.sim_t.toFixed(1);
        $rtf.textContent = msg.rtf.toFixed(2);
        $ncon.textContent = msg.ncon;
        return;
      }
      latest = new Float32Array(ev.data);
    };
    ws.onclose = function () {
      $fps.textContent = "WS 断开，2s 后重连";
      setTimeout(function () { connectWS(parsed); }, 2000);
    };
    ws.onopen = function () { $fps.textContent = "实时"; };
    (function pump() {
      requestAnimationFrame(pump);
      if (!latest || !groups) return;
      for (var i = 1; i < groups.length; i++) {
        var o = i * 7;
        groups[i].position.set(latest[o], latest[o + 1], latest[o + 2]);
        groups[i].quaternion.set(latest[o + 4], latest[o + 5], latest[o + 6], latest[o + 3]);  // wxyz -> xyzw
      }
      if (thoraxIdx > 0) {
        var o7 = thoraxIdx * 7;
        followTarget.lerp(new THREE.Vector3(latest[o7], latest[o7 + 1], latest[o7 + 2]), 0.08);
        shadow.position.x = followTarget.x; shadow.position.z = followTarget.z;
        shadow.material.opacity = Math.max(0.12, 0.4 - latest[o7 + 2] * 1.2);
      }
      applyCamera();
      renderer.render(scene, camera);
      frames++;
      var now = performance.now();
      if (now - fpsT > 1000) {
        $fps.textContent = "实时 · " + Math.round(frames * 1000 / (now - fpsT)) + " fps";
        frames = 0; fpsT = now;
      }
    })();
  }

  fetch("/static/mj/fruitfly.xml").then(function (r) { return r.text(); })
    .then(parseXML)
    .then(function (parsed) { return buildScene(parsed).then(function () { return parsed; }); })
    .then(function (parsed) {
      resize();
      applyCamera();
      connectWS(parsed);
    })
    .catch(function (e) {
      document.body.classList.add("nomj");
      console.error(e);
      $fps.textContent = "加载失败: " + (e && e.message || e);
      document.getElementById("mjNotice").innerHTML =
        "页面初始化失败：" + (e && e.message || e);
    });
})();
