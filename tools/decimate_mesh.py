"""One-time mesh decimation: raw CAD STL -> low-poly STL for MuJoCo.

MuJoCo's STL decoder rejects meshes with more than 200_000 faces and convex-hull /
render cost grows with triangle count, so the raw Fusion 360 exports (base_link ~1.08M
tris) must be simplified. This reads a binary STL, merges duplicate vertices into an
indexed mesh, decimates with a quadric method, and writes a binary STL.

Usage:
    python -m tools.decimate_mesh <in.stl> <out.stl> --target-faces 80000
"""

import argparse
import struct

import numpy as np


def read_binary_stl(path):
    with open(path, "rb") as f:
        f.read(80)
        n = struct.unpack("<I", f.read(4))[0]
        raw = np.frombuffer(f.read(n * 50), dtype=np.uint8).reshape(n, 50)
    floats = raw[:, :48].copy().view("<f4").reshape(n, 12)
    tris = floats[:, 3:12].reshape(n * 3, 3)  # v1, v2, v3 per facet
    # Merge duplicate vertices into an indexed mesh.
    verts, inv = np.unique(np.round(tris, 6), axis=0, return_inverse=True)
    faces = inv.reshape(n, 3)
    return verts.astype(np.float64), faces.astype(np.int64)


def write_binary_stl(path, verts, faces):
    tri = verts[faces]  # (F, 3, 3)
    n = tri.shape[0]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norm = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, norm, out=np.zeros_like(normals), where=norm > 0)
    rec = np.zeros((n, 50), dtype=np.uint8)
    payload = np.concatenate([normals[:, None, :], tri], axis=1).astype("<f4")  # (F,4,3)
    rec[:, :48] = payload.reshape(n, 12).view(np.uint8).reshape(n, 48)
    with open(path, "wb") as f:
        f.write(b"\x00" * 80)
        f.write(struct.pack("<I", n))
        f.write(rec.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--target-faces", type=int, default=80000)
    args = ap.parse_args()

    import fast_simplification

    verts, faces = read_binary_stl(args.src)
    reduction = max(0.0, 1.0 - args.target_faces / len(faces))
    out_v, out_f = fast_simplification.simplify(verts, faces, target_reduction=reduction)
    write_binary_stl(args.dst, out_v, out_f)
    print(f"{args.src}: {len(faces)} -> {len(out_f)} faces ({len(out_v)} verts) -> {args.dst}")


if __name__ == "__main__":
    main()
