#!/usr/bin/env python
"""HE 辅助 IHC 核融合（后处理，不改模型）。

思路（用户定稿）：
  1. HE 正常跑，轮廓做基准。
  2. IHC 用宽松参数跑（cellprob_threshold 低如 -1，flow_threshold 大），宁可多检。
  3. 有 HE 重叠的 IHC 核留下，无重叠的当假阳性扔掉；
     HE 有但 IHC 没检出的，把 HE 轮廓直接补回来。
  4. 融合结果画到 IHC 图上做双色叠加。

坐标前提：HE.svs 与 *_warped.tiff 已配准，两边 dat 的 proc_dimensions 一致
（实测 2519980 两边都是 [25296 26909] @0.5mpp）。若不一致则按 proc_dimensions
比例把 HE 缩放到 IHC 坐标系再匹配。

匹配规则（抗配准误差，proc 坐标系，默认 0.5mpp）：
  IHC 保留条件（三者满足其一）：与任一 HE 的 bbox_IoU >= iou_thresh；
    IHC 质心落在 HE bbox 内；IHC 质心与最近 HE 质心距离 <= dist_thresh px。
  HE 需补充条件：无任何保留 IHC 与之满足上述任一条件。
  注：warped 配准常有十几 px 残差，dist_thresh 默认 25px（@0.5mpp 约 12.5um）。

单对用法：
  python fuse_he_ihc.py --he_dat <HE.dat> --ihc_dat <ER_warped.dat> \\
    --out_dat <fused.dat> --ihc_wsi <ER_warped.tiff> --viz_dir <viz_dir>

 批量用法（dat_dir 里同时有 " HE.dat" 和 " *_warped.dat"）：
  python fuse_he_ihc.py --dat_dir .../dat --out_dir .../fused \\
    --ihc_wsi_dir .../CNB_Registration --viz_mpp 0.5

 直接叠加用法（跳过融合过滤，HE 全量 + IHC 全量合并）：
  python fuse_he_ihc.py --dat_dir .../dat --out_dir .../overlay \\
    --ihc_wsi_dir .../CNB_Registration --direct_overlay --no_viz
  单对同理加 --direct_overlay 即可。

 输出：
  out_dir/dat/<IHC_stem>_HEfused.dat   融合 dat（格式同 cellpose .dat，可直接进 qupath 导出）
  out_dir/viz/<IHC_stem>/<stem>_HEfused_full.tif  整张双色 BigTIFF（绿=IHC保留，红=HE补回）
  out_dir/viz/<IHC_stem>/<stem>_HEfused_preview.png 整张缩小预览图
  out_dir/viz_he/<HE_stem>/<stem>_HE_full.tif  HE 图 + HE 标注（需 --he_wsi_dir）
  out_dir/fuse_stats.csv               每对的 统计

 HE viz（--he_wsi 单对 / --he_wsi_dir 批量给了就画，--no_he_viz 关闭）：
  以 HE 原图为底图画 HE.dat 全量标注（红），与双色 viz 共用同一套分条带内核。
"""

import argparse
import csv
import os
import pathlib
import time
import uuid

import cv2
import joblib
import numpy as np
import tqdm
from scipy.spatial import cKDTree
from shapely.geometry import box as shapely_box
from shapely.strtree import STRtree


def _load_nuclei(dat_path):
    info = joblib.load(dat_path)
    nuc = info.get("Nuclei", {})
    return info, nuc


def _proc_dims(info):
    d = np.array(info["proc_dimensions"]).ravel()
    return int(d[0]), int(d[1])  # H, W


def _xy_box(b):
    """dat box [[r0,c0],[r1,c1]] -> [c0,r0,c1,r1]."""
    return np.array([b[0][1], b[0][0], b[1][1], b[1][0]], dtype=np.float64)


def _bbox_iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = ix1 - ix0, iy1 - iy0
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    aa = max(0.0, (a[2] - a[0])) * max(0.0, (a[3] - a[1]))
    bb = max(0.0, (b[2] - b[0])) * max(0.0, (b[3] - b[1]))
    union = aa + bb - inter
    return inter / union if union > 0 else 0.0


def _scale_entry(entry, sx, sy):
    """HE -> IHC 坐标缩放（contour XY，box YX，centroid XY）。"""
    e = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in entry.items()}
    e["contour"] = e["contour"].astype(np.float64) * np.array([sx, sy])
    e["box"] = e["box"].astype(np.float64) * np.array([[sy, sx], [sy, sx]])
    e["centroid"] = e["centroid"].astype(np.float64) * np.array([sx, sy])
    return e


def fuse_pair(he_dat, ihc_dat, out_dat=None, iou_thresh=0.1, dist_thresh=25.0,
             direct_overlay=False):
    """HE/IHC 融合或直接叠加。

    direct_overlay=False（默认）：走 IoU/质心距离过滤（见模块 docstring）。
    direct_overlay=True：跳过一切过滤，IHC 全保留 + HE 全量（缩放到 IHC
      坐标系后）直接合并，fused_n = he_n + ihc_n。

    注：dat 很大时（几十万核）joblib 加载就要几十秒，本函数会分阶段打日志，
    看到日志在走就是没卡。批量模式建议用 fuse_loaded + HE 缓存，避免同一
    HE.dat 被重复加载。
    """
    t0 = time.perf_counter()
    print(f"[fuse] loading HE: {he_dat}", flush=True)
    he_info, he_nuc = _load_nuclei(he_dat)
    print(f"[fuse] HE loaded: n={len(he_nuc)} ({time.perf_counter() - t0:.1f}s)", flush=True)
    t1 = time.perf_counter()
    print(f"[fuse] loading IHC: {ihc_dat}", flush=True)
    ihc_info, ihc_nuc = _load_nuclei(ihc_dat)
    print(f"[fuse] IHC loaded: n={len(ihc_nuc)} ({time.perf_counter() - t1:.1f}s)", flush=True)
    return fuse_loaded(he_info, ihc_info, out_dat, iou_thresh, dist_thresh,
                       direct_overlay, t0=t0)


def fuse_loaded(he_info, ihc_info, out_dat=None, iou_thresh=0.1, dist_thresh=25.0,
                direct_overlay=False, t0=None):
    """已加载 dat 的融合内核（he_info/ihc_info 为 joblib 载入的 dict）。

    只读 he_info/ihc_info，不修改输入，可安全复用同一 HE 给多个 IHC 配对。
    """
    t0 = t0 if t0 is not None else time.perf_counter()
    he_nuc = he_info["Nuclei"]
    ihc_nuc = ihc_info["Nuclei"]
    print(f"[fuse] HE n={len(he_nuc)}, IHC n={len(ihc_nuc)}", flush=True)

    # 坐标对齐：HE 缩放到 IHC proc 坐标系
    hh, hw = _proc_dims(he_info)
    ih, iw = _proc_dims(ihc_info)
    sx, sy = (iw / hw) if hw else 1.0, (ih / hh) if hh else 1.0
    scaled = abs(sx - 1.0) > 1e-6 or abs(sy - 1.0) > 1e-6

    he_keys = list(he_nuc.keys())
    he_boxes = np.array([_xy_box(_scale_entry(he_nuc[k], sx, sy)["box"]) if scaled
                         else _xy_box(he_nuc[k]["box"]) for k in he_keys])
    he_cent = np.array([np.asarray(he_nuc[k]["centroid"], dtype=float) * ([sx, sy] if scaled else [1, 1])
                        for k in he_keys])
    t_idx = time.perf_counter()
    print(f"[fuse] building HE index ({len(he_keys)})...", flush=True)
    he_tree = STRtree([shapely_box(*b) for b in he_boxes]) if len(he_keys) else None
    he_kd = cKDTree(he_cent) if len(he_keys) else None

    ihc_keys = list(ihc_nuc.keys())
    ihc_boxes = np.array([_xy_box(ihc_nuc[k]["box"]) for k in ihc_keys])
    ihc_cent = np.array([np.asarray(ihc_nuc[k]["centroid"], dtype=float) for k in ihc_keys])
    print(f"[fuse] index built ({time.perf_counter() - t_idx:.1f}s)", flush=True)

    def _query(tree, qbox):
        try:
            cand = tree.query(qbox)
            if len(cand) and isinstance(cand[0], (int, np.integer)):
                return [int(c) for c in cand]
        except Exception:
            pass
        return []

    # 直接叠加：跳过一切过滤，IHC 全保留 + HE 全量合并
    if direct_overlay:
        kept_idx = np.arange(len(ihc_keys))
        supp_idx = list(range(len(he_keys)))
    else:
        # IHC 质心最近 HE 距离（向量化，一次算完）
        if he_kd is not None and len(ihc_keys):
            ihc_nn_dist, _ = he_kd.query(ihc_cent, k=1)
        else:
            ihc_nn_dist = np.full(len(ihc_keys), np.inf)

        keep_mask = np.zeros(len(ihc_keys), dtype=bool)
        if len(he_keys) and len(ihc_keys):
            # 配准残差内有 HE 细胞的直接留（向量化），只对剩下的跑 bbox 精查
            keep_mask = ihc_nn_dist <= dist_thresh
            todo = np.nonzero(~keep_mask)[0]
            print(f"[fuse] IHC x HE filter: dist-pass {int(keep_mask.sum())}/{len(ihc_keys)}, "
                  f"bbox-check {len(todo)}", flush=True)
            for i in tqdm.tqdm(todo, desc="IHC x HE filter", ncols=90, leave=False):
                cand = _query(he_tree, shapely_box(*ihc_boxes[i]))
                hit = False
                for j in cand:
                    if _bbox_iou(ihc_boxes[i], he_boxes[j]) >= iou_thresh:
                        hit = True
                        break
                if not hit:  # 质心落在 HE bbox 内也算命中
                    cx, cy = ihc_cent[i]
                    for j in cand:
                        b = he_boxes[j]
                        if b[0] <= cx <= b[2] and b[1] <= cy <= b[3]:
                            hit = True
                            break
                keep_mask[i] = hit

        kept_idx = np.nonzero(keep_mask)[0]
        kept_boxes = ihc_boxes[kept_idx] if len(kept_idx) else np.zeros((0, 4))
        kept_cent = ihc_cent[kept_idx] if len(kept_idx) else np.zeros((0, 2))
        kept_tree = STRtree([shapely_box(*b) for b in kept_boxes]) if len(kept_idx) else None
        kept_kd = cKDTree(kept_cent) if len(kept_idx) else None
        if kept_kd is not None and len(he_keys):
            he_nn_dist, _ = kept_kd.query(he_cent, k=1)
        else:
            he_nn_dist = np.full(len(he_keys), np.inf)

        # HE 补回：无保留 IHC 命中的 HE（向量化先排除有邻居的，只查剩下的）
        supp_idx = []
        if len(he_keys):
            todo_he = np.nonzero(~(he_nn_dist <= dist_thresh))[0]
            print(f"[fuse] HE supplement: dist-skip {len(he_keys) - len(todo_he)}/{len(he_keys)}, "
                  f"bbox-check {len(todo_he)}", flush=True)
            for j in tqdm.tqdm(todo_he, desc="HE supplement", ncols=90, leave=False):
                hit = False
                if kept_tree is not None and len(kept_idx):
                    cand = _query(kept_tree, shapely_box(*he_boxes[j]))
                    for c in cand:
                        if _bbox_iou(he_boxes[j], kept_boxes[c]) >= iou_thresh:
                            hit = True
                            break
                    if not hit:
                        for c in cand:
                            cx, cy = kept_cent[c]
                            b = he_boxes[j]
                            if b[0] <= cx <= b[2] and b[1] <= cy <= b[3]:
                                hit = True
                                break
                if not hit:
                    supp_idx.append(j)

    he_source = "he" if direct_overlay else "he_supplement"
    print(f"[fuse] merging kept_ihc={len(kept_idx)} + he={len(supp_idx)}...", flush=True)
    fused = {}
    for i in tqdm.tqdm(kept_idx, desc="Merge IHC", ncols=90, leave=False):
        k = ihc_keys[int(i)]
        e = {kk: (vv.copy() if isinstance(vv, np.ndarray) else vv) for kk, vv in ihc_nuc[k].items()}
        e["source"] = "ihc"
        fused[uuid.uuid4().hex] = e
    for j in tqdm.tqdm(supp_idx, desc="Merge HE", ncols=90, leave=False):
        k = he_keys[int(j)]
        e = _scale_entry(he_nuc[k], sx, sy) if scaled else \
            {kk: (vv.copy() if isinstance(vv, np.ndarray) else vv) for kk, vv in he_nuc[k].items()}
        e["source"] = he_source
        fused[uuid.uuid4().hex] = e

    stats = {
        "he_n": len(he_keys), "ihc_n": len(ihc_keys),
        "kept_ihc": int(len(kept_idx)), "dropped_ihc": int(len(ihc_keys) - len(kept_idx)),
        "he_supplement": int(len(supp_idx)), "fused_n": len(fused),
        "mode": "direct_overlay" if direct_overlay else "filter",
        "iou_thresh": iou_thresh, "dist_thresh": dist_thresh, "scaled": scaled, "sx": sx, "sy": sy,
        "secs": round(time.perf_counter() - t0, 1),
    }
    out = {"Nuclei": fused,
           "proc_resolution": ihc_info["proc_resolution"],
           "base_resolution": ihc_info["base_resolution"],
           "proc_dimensions": np.array(ihc_info["proc_dimensions"]),
           "base_dimensions": np.array(ihc_info["base_dimensions"])}
    if out_dat:
        os.makedirs(os.path.dirname(out_dat) or ".", exist_ok=True)
        print(f"[fuse] saving {out_dat} (n={len(fused)})...", flush=True)
        joblib.dump(out, out_dat)
        print(f"[fuse] saved ({time.perf_counter() - t0:.1f}s total)", flush=True)
    return out, stats


def _viz_overlay(wsi_path, info_or_dat, viz_dir, file_stem, color_of, viz_mpp=0.5, strip=2048):
    """整图叠加内核：分条带读 WSI 基准层，画轮廓到整张 canvas，
    存单张 BigTIFF（<file_stem>_full.tif）+ 一张小预览 PNG。color_of(entry) -> BGR 颜色。"""
    from tiatoolbox.wsicore.wsireader import WSIReader
    import tifffile
    info = joblib.load(info_or_dat) if isinstance(info_or_dat, str) else info_or_dat
    nuc = info["Nuclei"]
    proc_mpp = float(np.array(info["proc_resolution"]["resolution"]).flat[0])
    scale = proc_mpp / float(viz_mpp)
    proc_h, proc_w = [int(x) for x in np.array(info["proc_dimensions"]).ravel()[:2]]
    viz_w, viz_h = int(round(proc_w * scale)), int(round(proc_h * scale))

    items = []  # (y0, y1, pts_full_int32, color_bgr)，pts 为整图 viz 坐标
    for e in nuc.values():
        cnt = np.asarray(e["contour"], dtype=float) * scale
        color = color_of(e)
        pts = np.round(cnt).astype(np.int32)
        pts[:, 0] = np.clip(pts[:, 0], 0, viz_w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, viz_h - 1)
        items.append((pts[:, 1].min(), pts[:, 1].max(), pts, color))
    # 轮廓按条带预分组：每条带只遍历与之相交的轮廓，避免 66k×13 次循环
    n_strips = (viz_h + strip - 1) // strip
    buckets = [[] for _ in range(n_strips)]
    for it in items:
        y0, y1 = it[0], it[1]
        for s in range(int(y0) // strip, int(y1) // strip + 1):
            if 0 <= s < n_strips:
                buckets[s].append(it)

    os.makedirs(viz_dir, exist_ok=True)
    canvas = np.zeros((viz_h, viz_w, 3), dtype=np.uint8)

    # 读图：优先 tifffile 按块直读基准层（warped tiff 是 128 tiled JPEG，比
    # openslide/tiatoolbox 快数倍）；失败回退 WSIReader（HE 的 .svs 一般走这条）。
    base_mpp = float(np.array(info["base_resolution"]["resolution"]).flat[0])
    f = float(viz_mpp) / base_mpp  # 1 viz 像素 = f 个基准像素
    reader = None
    try:
        import tifffile as _tf
        import zarr as _zarr
        _fh = _tf.TiffFile(wsi_path)
        _z = _zarr.open(_fh.aszarr(), mode="r")
        _bh, _bw = int(_z.shape[0]), int(_z.shape[1])
        if _z.ndim == 3 and _z.shape[0] == 3:  # 极少数 (C,H,W) 存储
            raise ValueError("unexpected CHW layout")
        reader = ("zarr", _fh, _z, _bh, _bw)
    except Exception as ex:
        print(f"[WARN] tifffile 直读失败，用 WSIReader 回退：{ex}")

    if reader is None:
        from tiatoolbox.wsicore.wsireader import WSIReader
        reader = ("wsi", WSIReader.open(wsi_path))

    for si, ty in enumerate(tqdm.tqdm(range(0, viz_h, strip), desc="Viz full", ncols=90)):
        ty1 = min(ty + strip, viz_h)
        h = ty1 - ty
        try:
            if reader[0] == "zarr":
                _, _, _z, _bh, _bw = reader
                r0 = min(_bh - 1, int(round(ty * f)))
                r1 = min(_bh, int(round(ty1 * f)))
                if r1 <= r0:
                    continue
                img = np.asarray(_z[r0:r1, :, :])
                if img.shape[-1] == 4:
                    img = img[..., :3]
                img = cv2.resize(img, (viz_w, h), interpolation=cv2.INTER_AREA)
            else:
                img = reader[1].read_rect(
                    location=(0, ty), size=(viz_w, h),
                    resolution=float(viz_mpp), units="mpp", coord_space="resolution")
                if img.shape[-1] == 4:
                    img = img[..., :3]
        except Exception as ex:
            print(f"[WARN] strip y={ty} skip: {ex}")
            continue
        tile = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        oh, ow = tile.shape[:2]
        off = np.array([0, ty], dtype=np.int32)
        for y0, y1, pts, color in buckets[si]:
            cv2.polylines(tile, [pts - off], True, color, 1)
        canvas[ty:ty + oh, :ow] = tile
    try:
        if reader[0] == "zarr":
            reader[1].close()
    except Exception:
        pass

    tif_path = os.path.join(viz_dir, f"{file_stem}_full.tif")
    tifffile.imwrite(tif_path, cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB),
                     tile=(1024, 1024), compression="deflate", bigtiff=True)
    k = 2048 / max(viz_w, viz_h)
    preview = cv2.resize(canvas, (max(1, int(viz_w * k)), max(1, int(viz_h * k))),
                         interpolation=cv2.INTER_AREA)
    png_path = os.path.join(viz_dir, f"{file_stem}_preview.png")
    cv2.imwrite(png_path, preview)
    print(f"viz: {tif_path} ({viz_w}x{viz_h}) + preview")
    return tif_path


def viz_dual(ihc_wsi, fused_dat_or_info, viz_dir, viz_mpp=0.5, strip=2048):
    """整张双色叠加：绿=IHC，红=HE（过滤模式是 HE 补回，direct_overlay 模式是 HE 全量）。
    底图为 IHC 配准图，分条带读取，存单张 BigTIFF + 一张小预览 PNG。"""
    stem = pathlib.Path(ihc_wsi).stem
    return _viz_overlay(ihc_wsi, fused_dat_or_info, viz_dir, f"{stem}_HEfused",
                        lambda e: (0, 255, 0) if e.get("source", "ihc") == "ihc" else (0, 0, 255),
                        viz_mpp, strip)


def viz_he(he_wsi, he_dat_or_info, viz_dir, viz_mpp=0.5, strip=2048, color=(0, 0, 255)):
    """HE 图 + HE 标注单色叠加（默认红=HE 核，与双色图里的 HE 颜色一致）。
    底图为 HE 原图（.svs），用 HE.dat 自身的 proc/base 分辨率对齐，
    存单张 BigTIFF（<HE_stem>_HE_full.tif）+ 一张小预览 PNG。"""
    stem = pathlib.Path(he_wsi).stem
    return _viz_overlay(he_wsi, he_dat_or_info, viz_dir, f"{stem}_HE",
                        lambda e: color, viz_mpp, strip)


def _pair_batch(dat_dir):
    files = sorted(os.listdir(dat_dir))
    he = {f[:-len(" HE.dat")].strip(): f for f in files if f.endswith(" HE.dat")}
    # IHC: "<caseID> <STAIN>_warped.dat"，如 "2519980 ER_warped.dat"，
    # caseID 即 HE 的 stem（如 "2519980"）。一张 HE 配 4 张 IHC，
    # 同一文件夹放多个 case 也能各自配上。
    pairs, missing = [], []
    for f in files:
        if not f.endswith("_warped.dat"):
            continue
        prefix = f[:-len("_warped.dat")].strip()  # 如 "2519980 ER"
        match = None
        for stem in he:
            if prefix == stem or prefix.startswith(stem + " "):
                if match is None or len(stem) > len(match):
                    match = stem
        if match is None:
            missing.append(prefix)
        else:
            pairs.append((prefix, he[match], f))
    pairs.sort()
    return pairs, missing


def _find_wsi(wsi_dir, stems, exts):
    """按候选名 + 后缀找 WSI 文件，返回首个存在的路径，否则 None。"""
    for s in stems:
        for ext in exts:
            c = os.path.join(wsi_dir, s + ext)
            if os.path.isfile(c):
                return c
    return None


def main():
    p = argparse.ArgumentParser(description="HE-assisted IHC fusion + dual-color viz")
    p.add_argument("--he_dat", default=None)
    p.add_argument("--ihc_dat", default=None)
    p.add_argument("--out_dat", default=None)
    p.add_argument("--ihc_wsi", default=None, help="IHC 配准图（*_warped.tiff），用于双色 viz 底图")
    p.add_argument("--viz_dir", default=None)
    p.add_argument("--dat_dir", default=None, help="批量模式：同时含 HE.dat 与 *_warped.dat 的目录")
    p.add_argument("--out_dir", default=None, help="批量模式输出根目录（建 dat/ viz/ [viz_he/] fuse_stats.csv）")
    p.add_argument("--ihc_wsi_dir", default=None, help="批量模式 IHC 配准图目录")
    p.add_argument("--he_wsi_dir", default=None, help="批量模式 HE 原图目录（.svs），给了就画 HE viz，每个 HE 只画一次")
    p.add_argument("--no_he_viz", action="store_true", help="不画 HE viz（默认：给了 HE 图就画）")
    p.add_argument("--he_wsi", default=None, help="单对模式 HE 原图（.svs），给了就画 HE viz（除非 --no_he_viz）")
    p.add_argument("--he_viz_dir", default=None, help="单对模式 HE viz 输出目录，默认与 --viz_dir 相同")
    p.add_argument("--iou_thresh", type=float, default=0.1)
    p.add_argument("--dist_thresh", type=float, default=25.0,
                   help="质心配准容差 px（proc 坐标，0.5mpp 下 25px≈12.5um）")
    p.add_argument("--direct_overlay", action="store_true",
                   help="直接叠加：跳过 IoU/距离过滤，HE 全量 + IHC 全量合并")
    p.add_argument("--no_viz", action="store_true")
    p.add_argument("--viz_mpp", type=float, default=0.5)
    p.add_argument("--viz_strip", type=int, default=2048,
                   help="整图分条带读取高度（viz 像素），默认 2048")
    a = p.parse_args()

    if a.dat_dir:  # 批量
        assert a.out_dir, "--dat_dir 需配 --out_dir"
        pairs, missing = _pair_batch(a.dat_dir)
        print(f"paired {len(pairs)}, no-HE skipped {len(missing)}: {missing[:5]}")
        dat_out = os.path.join(a.out_dir, "dat")
        rows = []
        he_cache = {}  # 同一 HE.dat 只加载一次（1 个 HE 配多个 IHC 时省重复加载）
        he_viz_done = set()  # 已画过 HE viz 的 he_stem（每个 HE 只画一次）
        for n, (prefix, he_f, ihc_f) in enumerate(tqdm.tqdm(pairs, desc="Fuse batch", ncols=90), 1):
            ihc_stem = pathlib.Path(ihc_f).stem
            out_dat = os.path.join(dat_out, f"{ihc_stem}_HEfused.dat")
            print(f"[{n}/{len(pairs)}] {prefix} ...", flush=True)
            he_path = os.path.join(a.dat_dir, he_f)
            if he_path not in he_cache:
                print(f"[fuse] loading HE: {he_path}", flush=True)
                t_he = time.perf_counter()
                he_cache[he_path] = joblib.load(he_path)
                print(f"[fuse] HE loaded: n={len(he_cache[he_path]['Nuclei'])} "
                      f"({time.perf_counter() - t_he:.1f}s, cached for reuse)", flush=True)
            else:
                print(f"[fuse] HE cache hit: {he_f}", flush=True)
            ihc_path = os.path.join(a.dat_dir, ihc_f)
            print(f"[fuse] loading IHC: {ihc_path}", flush=True)
            t_ihc = time.perf_counter()
            ihc_info = joblib.load(ihc_path)
            print(f"[fuse] IHC loaded: n={len(ihc_info['Nuclei'])} "
                  f"({time.perf_counter() - t_ihc:.1f}s)", flush=True)
            _, st = fuse_loaded(he_cache[he_path], ihc_info, out_dat,
                                a.iou_thresh, a.dist_thresh, a.direct_overlay)
            st["prefix"] = prefix
            st["out"] = out_dat
            if not a.no_viz and a.ihc_wsi_dir:
                wsi = _find_wsi(a.ihc_wsi_dir, [ihc_stem], (".tiff", ".tif", ".svs"))
                if wsi:
                    viz_dual(wsi, out_dat, os.path.join(a.out_dir, "viz", ihc_stem),
                             a.viz_mpp, a.viz_strip)
                else:
                    print(f"[WARN] {ihc_stem}: IHC 图没找到，跳过 viz")
            if not a.no_he_viz and a.he_wsi_dir:
                he_stem = pathlib.Path(he_f).stem
                if he_stem not in he_viz_done:
                    cand = [he_stem] + ([he_stem[:-3]] if he_stem.endswith(" HE") else [])
                    he_wsi = _find_wsi(a.he_wsi_dir, cand,
                                       (".svs", ".tiff", ".tif", ".mrxs", ".ndpi", ".scn"))
                    if he_wsi:
                        viz_he(he_wsi, he_cache[he_path],
                               os.path.join(a.out_dir, "viz_he", he_stem),
                               a.viz_mpp, a.viz_strip)
                        he_viz_done.add(he_stem)
                    else:
                        print(f"[WARN] {he_stem}: HE 图没找到，跳过 HE viz")
            rows.append(st)
            print(f"{prefix} [{st['mode']}]: HE={st['he_n']} IHC={st['ihc_n']} keep={st['kept_ihc']} "
                  f"drop={st['dropped_ihc']} supp={st['he_supplement']} fused={st['fused_n']} {st['secs']}s")
        if rows:
            with open(os.path.join(a.out_dir, "fuse_stats.csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            print(f"stats -> {a.out_dir}/fuse_stats.csv")
    else:  # 单对
        assert a.he_dat and a.ihc_dat and a.out_dat, "单对模式需 --he_dat --ihc_dat --out_dat"
        _, st = fuse_pair(a.he_dat, a.ihc_dat, a.out_dat, a.iou_thresh, a.dist_thresh,
                          a.direct_overlay)
        print(f"[{st['mode']}] HE={st['he_n']} IHC={st['ihc_n']} keep={st['kept_ihc']} "
              f"drop={st['dropped_ihc']} supp={st['he_supplement']} fused={st['fused_n']} {st['secs']}s")
        print(f"fused -> {a.out_dat}")
        if not a.no_viz and a.ihc_wsi and a.viz_dir:
            viz_dual(a.ihc_wsi, a.out_dat, a.viz_dir, a.viz_mpp, a.viz_strip)
        if not a.no_he_viz and a.he_wsi:
            he_viz_dir = a.he_viz_dir or a.viz_dir
            if he_viz_dir:
                viz_he(a.he_wsi, a.he_dat, he_viz_dir, a.viz_mpp, a.viz_strip)
            else:
                print("[WARN] 要画 HE viz 需给 --he_viz_dir（或 --viz_dir）")


if __name__ == "__main__":
    main()
