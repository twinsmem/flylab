"""将 flybody (TuragaLab) 的 OBJ 网格一次性转换为二进制 STL，并生成本地副本 XML。

用法（在本机运行，需先 sparse-clone flybody）：
    python scripts/convert_flybody_obj.py <flybody_assets_dir> [输出目录=static/mj]

- 只处理 fruitfly.xml <asset> 里引用的 .obj（未引用的 .blend 等不拷贝）
- OBJ -> binary STL（三角形原样保留；法线由客户端合并顶点后重算）
- 生成 fruitfly.xml 副本：.obj -> .stl，worldbody 注入地板，头部加来源注释
- 单位说明：OBJ 坐标 × XML <mesh scale="0.1"> = 模型坐标（厘米），缩放保持由
  MuJoCo XML 与客户端各自施加，STL 内不烘焙缩放。
"""
import os
import re
import struct
import sys

ASSET_RE = re.compile(r'<mesh\s+name="([^"]+)"\s+file="([^"]+\.obj)"')
WORLD_INJECT = """<worldbody>
    <!-- FlyLab: 演示用地板（flybody 原模型无地板） -->
    <geom name="floor" type="plane" size="200 200 0.5" friction="1.0"
          rgba="0.16 0.17 0.20 1"/>"""


def parse_obj(path):
    """返回 (顶点列表, 法线列表, 三角形列表)；三角形 = (i0,i1,i2, n0,n1,n2)。"""
    verts, norms, tris = [], [], []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                p = line.split()
                verts.append((float(p[1]), float(p[2]), float(p[3])))
            elif line.startswith("vn "):
                p = line.split()
                norms.append((float(p[1]), float(p[2]), float(p[3])))
            elif line.startswith("f "):
                idx = []
                for tok in line.split()[1:]:
                    # 形如 v / v/vt / v/vt/vn / v//vn，索引可为负
                    parts = tok.split("/")
                    vi = int(parts[0])
                    ni = int(parts[2]) if len(parts) >= 3 and parts[2] else None
                    if vi < 0:
                        vi += len(verts) + 1
                    if ni is not None and ni < 0:
                        ni += len(norms) + 1
                    idx.append((vi - 1, ni))
                # 扇形三角化（防御性，flybody OBJ 本身已是三角形）
                for k in range(1, len(idx) - 1):
                    a, b, c = idx[0], idx[k], idx[k + 1]
                    tris.append((a[0], b[0], c[0], a[1], b[1], c[1]))
    return verts, norms, tris


def write_stl(path, verts, norms, tris):
    """二进制 STL：法线 = 三角面几何法线（客户端将重新平滑）。"""
    with open(path, "wb") as f:
        f.write(b"FlyLab flybody OBJ->STL" .ljust(80, b"\0"))
        f.write(struct.pack("<I", len(tris)))
        pack = struct.pack
        for i0, i1, i2, n0, n1, n2 in tris:
            ax, ay, az = verts[i0]
            bx, by, bz = verts[i1]
            cx, cy, cz = verts[i2]
            ux, uy, uz = bx - ax, by - ay, bz - az
            vx, vy, vz = cx - ax, cy - ay, cz - az
            nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
            ln = (nx * nx + ny * ny + nz * nz) ** 0.5 or 1.0
            f.write(pack("<12fH", nx / ln, ny / ln, nz / ln,
                         ax, ay, az, bx, by, bz, cx, cy, cz, 0))


def main():
    assets = sys.argv[1].rstrip("/\\")
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static", "mj")
    os.makedirs(os.path.join(out, "meshes"), exist_ok=True)

    xml_path = os.path.join(assets, "fruitfly.xml")
    with open(xml_path, "r", encoding="utf-8") as f:
        xml = f.read()

    files = sorted(set(m.group(2) for m in ASSET_RE.finditer(xml)))
    print(f"引用网格 {len(files)} 个")

    total = 0
    for fn in files:
        src = os.path.join(assets, fn)
        verts, norms, tris = parse_obj(src)
        dst = os.path.join(out, "meshes", fn[:-4] + ".stl")
        write_stl(dst, verts, norms, tris)
        total += os.path.getsize(dst)
        print(f"  {fn:42s} {len(verts):8d}v {len(tris):8d}tri "
              f"-> {os.path.getsize(dst) / 1e6:6.2f} MB")
    print(f"STL 合计: {total / 1e6:.1f} MB")

    # XML 副本：obj -> stl + meshdir + 注入地板 + 来源注释
    xml = xml.replace(".obj", ".stl")
    xml = xml.replace('angle="radian"/>', 'angle="radian" meshdir="meshes"/>', 1)
    xml = xml.replace("<worldbody>", WORLD_INJECT, 1)
    header = ("""<!-- FlyLab 本地副本：源自 TuragaLab/flybody (Apache-2.0)
     https://github.com/TuragaLab/flybody  flybody/fruitfly/assets/fruitfly.xml
     改动：网格引用 .obj -> .stl；worldbody 注入演示地板。
     Vaxenburg et al., "Whole-body physics simulation of fruit fly locomotion",
     Nature 643:1312-1320, 2025. -->
""")
    xml = header + xml
    with open(os.path.join(out, "fruitfly.xml"), "w", encoding="utf-8") as f:
        f.write(xml)
    print(f"已写出 {os.path.join(out, 'fruitfly.xml')}")


if __name__ == "__main__":
    main()
