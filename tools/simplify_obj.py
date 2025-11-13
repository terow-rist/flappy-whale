#!/usr/bin/env python3
"""
Triangulate + fast simplification (vertex clustering) for OBJ files.

Usage:
  cd ~/flappy-whale
  python3 tools/simplify_obj.py \
    --input assets/cute_whale/SubTool-0-10233600.OBJ \
    --output assets/whale_simplified.obj \
    --reduction 0.7

You should run for both whale and stalactite files.
"""

import os, sys, math, argparse

def read_obj(path):
    verts = []
    faces = []
    others = []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("v "):
                parts = line.split()
                if len(parts) < 4: continue
                x,y,z = float(parts[1]), float(parts[2]), float(parts[3])
                verts.append((x,y,z))
            elif line.startswith("f "):
                parts = line.split()[1:]
                idxs = []
                for p in parts:
                    if "/" in p:
                        v = p.split("/")[0]
                        if v == "": continue
                        idxs.append(int(v) - 1 if int(v) > 0 else len(verts) + int(v))
                    else:
                        idxs.append(int(p) - 1 if int(p)>0 else len(verts) + int(p))
                faces.append(idxs)
            else:
                others.append(line)
    return verts, faces, others

def triangulate_faces(faces):
    tri = []
    for f in faces:
        if len(f) < 3:
            continue
        if len(f) == 3:
            tri.append((f[0], f[1], f[2]))
        else:
            a = f[0]
            for i in range(1, len(f)-1):
                tri.append((a, f[i], f[i+1]))
    return tri

def vertex_clustering(verts, tris, reduction_ratio=0.6, min_vertices=500):
    # bounding box
    xs = [v[0] for v in verts]
    ys = [v[1] for v in verts]
    zs = [v[2] for v in verts]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    minz, maxz = min(zs), max(zs)
    dx, dy, dz = maxx-minx, maxy-miny, maxz-minz
    longest = max(dx, dy, dz)
    if longest == 0:
        longest = 1.0
    # determine approx target vertex count
    original_vcnt = len(verts)
    target_vcnt = max(min_vertices, int(original_vcnt * (1.0 - reduction_ratio)))
    # estimate cubic root to define grid resolution
    approx_cell_count = max(1, int(round(target_vcnt ** (1.0/3.0))))
    cx = dx / approx_cell_count if approx_cell_count>0 else dx+1e-6
    cy = dy / approx_cell_count if approx_cell_count>0 else dy+1e-6
    cz = dz / approx_cell_count if approx_cell_count>0 else dz+1e-6
    # avoid zero cell sizes
    if cx == 0: cx = longest/approx_cell_count + 1e-6
    if cy == 0: cy = longest/approx_cell_count + 1e-6
    if cz == 0: cz = longest/approx_cell_count + 1e-6

    clusters = {}
    cluster_sum = []
    vmap = {}
    for i,v in enumerate(verts):
        ix = int(math.floor((v[0]-minx)/cx))
        iy = int(math.floor((v[1]-miny)/cy))
        iz = int(math.floor((v[2]-minz)/cz))
        key = (ix,iy,iz)
        if key not in clusters:
            clusters[key] = len(cluster_sum)
            cluster_sum.append([0.0,0.0,0.0,0])  # sumx,sumy,sumz,count
        idx = clusters[key]
        cluster_sum[idx][0] += v[0]
        cluster_sum[idx][1] += v[1]
        cluster_sum[idx][2] += v[2]
        cluster_sum[idx][3] += 1
        vmap[i] = idx

    new_verts = [(s[0]/s[3], s[1]/s[3], s[2]/s[3]) for s in cluster_sum]

    # remap triangles and remove degenerate
    new_tris = []
    for a,b,c in tris:
        ia = vmap.get(a)
        ib = vmap.get(b)
        ic = vmap.get(c)
        if ia is None or ib is None or ic is None: continue
        if ia == ib or ib == ic or ia == ic: continue
        new_tris.append((ia,ib,ic))

    return new_verts, new_tris

def write_obj(path_out, verts, tris):
    with open(path_out, "w") as f:
        f.write("# simplified OBJ\n")
        for v in verts:
            f.write("v %f %f %f\n" % (v[0], v[1], v[2]))
        for a,b,c in tris:
            f.write("f %d %d %d\n" % (a+1, b+1, c+1))

def process(infile, outfile, reduction):
    print("Reading:", infile)
    verts, faces, other = read_obj(infile)
    print("Vertices read:", len(verts), "polygons read:", len(faces))
    tris = triangulate_faces(faces)
    print("Triangles after triangulation:", len(tris))
    new_verts, new_tris = vertex_clustering(verts, tris, reduction_ratio=reduction)
    print("After clustering -> vertices:", len(new_verts), "triangles:", len(new_tris))
    write_obj(outfile, new_verts, new_tris)
    print("Wrote:", outfile)
    return len(new_verts), len(new_tris)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", "-i", required=True, help="input OBJ")
    p.add_argument("--output", "-o", required=True, help="output OBJ")
    p.add_argument("--reduction", "-r", type=float, default=0.6, help="reduction ratio (0..0.95)")
    args = p.parse_args()
    if not os.path.exists(args.input):
        print("ERROR: input file not found:", args.input)
        sys.exit(2)
    process(args.input, args.output, args.reduction)

if __name__ == "__main__":
    main()
