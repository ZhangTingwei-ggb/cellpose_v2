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
  out_dir/viz_he/<HE_stem>/<stem>_fused_on_HE_full.tif  融合结果画在 HE 底上
      （--on_he fused；IHC 配准图有黑边/视野不足时用这张看完整 HE 视野）
  out_dir/fuse_stats.csv               每对的 统计

 HE viz（--he_wsi 单对 / --he_wsi_dir 批量给了就画，--no_he_viz 关闭）：
  以 HE 原图为底图画 HE.dat 全量标注（红），与双色 viz 共用同一套分条带内核。
  --on_he fused：把融合 dat 整体缩放到 HE 坐标系后画在 HE 底上，
  绿=IHC 保留、红=HE 补回；不受 IHC 配准图黑边影响。
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


def _viz_overlay(wsi_path, info_or_dat, viz_dir, file_stem, color_of, viz_mpp=0.5, strip=2048,
                 alpha_of=None, fill=False):
    """整图叠加内核：分条带读 WSI 基准层，画轮廓/填充到整张 canvas，
    存单张 BigTIFF（<file_stem>_full.tif）+ 一张小预览 PNG。

    color_of(entry) -> BGR 颜色；alpha_of(entry) -> 填充透明度 0-1。
    fill=False 时只画轮廓；fill=True 时先按 (颜色, alpha) 分组填充，再画轮廓。
    """
    from tiatoolbox.wsicore.wsireader import WSIReader
    import tifffile
    info = joblib.load(info_or_dat) if isinstance(info_or_dat, str) else info_or_dat
    nuc = info["Nuclei"]
    proc_mpp = float(np.array(info["proc_resolution"]["resolution"]).flat[0])
    scale = proc_mpp / float(viz_mpp)
    proc_h, proc_w = [int(x) for x in np.array(info["proc_dimensions"]).ravel()[:2]]
    viz_w, viz_h = int(round(proc_w * scale)), int(round(proc_h * scale))

    items = []  # (y0, y1, pts_full_int32, color_bgr, alpha)，pts 为整图 viz 坐标
    for e in nuc.values():
        cnt = np.asarray(e["contour"], dtype=float) * scale
        color = color_of(e)
        alpha = float(alpha_of(e)) if alpha_of is not None else 0.0
        alpha = min(1.0, max(0.0, alpha))
        pts = np.round(cnt).astype(np.int32)
        pts[:, 0] = np.clip(pts[:, 0], 0, viz_w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, viz_h - 1)
        items.append((pts[:, 1].min(), pts[:, 1].max(), pts, color, alpha))
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
        if not hasattr(_z, "shape") and hasattr(_z, "keys"):
            # OME aszarr 返回 Group（如天津 warped tiff），取最高分辨率层 '0'
            _keys = list(_z.keys())
            _z = _z["0"] if "0" in _keys else _z[_keys[0]]
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
        # 填充：同一 (颜色, alpha) 组只做一次 mask + 一次混合；
        # 组内多边形重叠不会重复叠 alpha，避免颜色越叠越深。
        if fill:
            groups = {}
            for y0, y1, pts, color, alpha in buckets[si]:
                if alpha <= 0:
                    continue
                groups.setdefault((tuple(int(c) for c in color), alpha), []).append(pts - off)
            for (color, alpha), polys in groups.items():
                mask = np.zeros((oh, ow), dtype=np.uint8)
                # 一次 fillPoly 画整组，避免 1.6M 次 Python->OpenCV 调用
                cv2.fillPoly(mask, polys, 1)
                m = mask.astype(bool)
                if not m.any():
                    continue
                c = np.array(color, dtype=np.float32)
                tile[m] = (tile[m].astype(np.float32) * (1.0 - alpha) + c * alpha).astype(np.uint8)
        outlines = {}
        for y0, y1, pts, color, alpha in buckets[si]:
            outlines.setdefault(tuple(int(c) for c in color), []).append(pts - off)
        for color, polys in outlines.items():
            cv2.polylines(tile, polys, True, color, 1)
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


def viz_dual(ihc_wsi, fused_dat_or_info, viz_dir, viz_mpp=0.5, strip=2048,
             fill=False, alpha_he=0.5, alpha_ihc=0.8):
    """整张双色叠加：绿=IHC，红=HE（过滤模式是 HE 补回，direct_overlay 模式是 HE 全量）。
    底图为 IHC 配准图，分条带读取，存单张 BigTIFF + 一张小预览 PNG。"""
    stem = pathlib.Path(ihc_wsi).stem
    return _viz_overlay(ihc_wsi, fused_dat_or_info, viz_dir, f"{stem}_HEfused",
                        lambda e: (0, 255, 0) if e.get("source", "ihc") == "ihc" else (0, 0, 255),
                        viz_mpp, strip,
                        alpha_of=lambda e: alpha_ihc if e.get("source", "ihc") == "ihc" else alpha_he,
                        fill=fill)


def viz_he(he_wsi, he_dat_or_info, viz_dir, viz_mpp=0.5, strip=2048, color=(0, 0, 255),
           fill=False, fill_alpha=0.5):
    """HE 图 + HE 标注单色叠加（默认红=HE 核，与双色图里的 HE 颜色一致）。
    底图为 HE 原图（.svs），用 HE.dat 自身的 proc/base 分辨率对齐，
    存单张 BigTIFF（<HE_stem>_HE_full.tif）+ 一张小预览 PNG。"""
    stem = pathlib.Path(he_wsi).stem
    return _viz_overlay(he_wsi, he_dat_or_info, viz_dir, f"{stem}_HE",
                        lambda e: color, viz_mpp, strip,
                        alpha_of=lambda e: fill_alpha, fill=fill)


def viz_ihc(ihc_wsi, ihc_dat_or_info, viz_dir, viz_mpp=0.5, strip=2048,
            color=(0, 255, 0), fill=False, fill_alpha=0.8):
    """IHC 图 + IHC 标注单色叠加（默认绿=IHC 核）。
    底图为 IHC 配准图（*_warped.tiff），用 IHC.dat 自身的 proc/base 分辨率对齐，
    存单张 BigTIFF（<IHC_stem>_IHC_full.tif）+ 一张小预览 PNG。"""
    stem = pathlib.Path(ihc_wsi).stem
    return _viz_overlay(ihc_wsi, ihc_dat_or_info, viz_dir, f"{stem}_IHC",
                        lambda e: color, viz_mpp, strip,
                        alpha_of=lambda e: fill_alpha, fill=fill)


def _he_base_view(he_info, ihc_info, mode="both"):
    """拼一个 HE 坐标系的合成 info，供 _viz_overlay 以 HE 为底图画图。

    mode="ihc"：只含 IHC（缩放到 HE 坐标系，绿）；mode="both"：HE（红）+ IHC（绿）。
    用原始 he/ihc info（不是 fused），direct_overlay 下即全量叠加。
    """
    if isinstance(he_info, str):
        he_info = joblib.load(he_info)
    if isinstance(ihc_info, str):
        ihc_info = joblib.load(ihc_info)
    hh, hw = _proc_dims(he_info)
    ih, iw = _proc_dims(ihc_info)
    sx = (hw / iw) if iw else 1.0  # IHC -> HE 缩放（与 fuse 里 HE->IHC 反方向）
    sy = (hh / ih) if ih else 1.0
    nuc = {}
    if mode in ("both", "he"):
        for k, e in he_info["Nuclei"].items():
            c = {kk: (vv.copy() if isinstance(vv, np.ndarray) else vv) for kk, vv in e.items()}
            c["source"] = "he"
            nuc[uuid.uuid4().hex] = c
    if mode in ("both", "ihc"):
        for k, e in ihc_info["Nuclei"].items():
            c = _scale_entry(e, sx, sy)
            c["source"] = "ihc"
            nuc[uuid.uuid4().hex] = c
    return {"Nuclei": nuc,
            "proc_resolution": he_info["proc_resolution"],
            "base_resolution": he_info["base_resolution"],
            "proc_dimensions": np.array(he_info["proc_dimensions"]),
            "base_dimensions": np.array(he_info["base_dimensions"])}


def viz_ihc_on_he(he_wsi, he_info, ihc_info, viz_dir, ihc_stem=None, viz_mpp=0.5, strip=2048,
                  fill=False, fill_alpha=0.8):
    """HE 为底 + 只画 IHC 标注（绿）。存 <ihc_stem>_on_HE_full.tif + preview。"""
    view = _he_base_view(he_info, ihc_info, mode="ihc")
    tag = ihc_stem or pathlib.Path(he_wsi).stem
    return _viz_overlay(he_wsi, view, viz_dir, f"{tag}_on_HE",
                        lambda e: (0, 255, 0), viz_mpp, strip,
                        alpha_of=lambda e: fill_alpha, fill=fill)


def viz_both_on_he(he_wsi, he_info, ihc_info, viz_dir, ihc_stem=None, viz_mpp=0.5, strip=2048,
                   fill=False, alpha_he=0.5, alpha_ihc=0.8):
    """HE 为底 + HE(红)+IHC(绿)同时画（两套全量标注，不做融合消除）。
    存 <ihc_stem>_both_on_HE_full.tif + preview。"""
    view = _he_base_view(he_info, ihc_info, mode="both")
    tag = ihc_stem or pathlib.Path(he_wsi).stem
    return _viz_overlay(he_wsi, view, viz_dir, f"{tag}_both_on_HE",
                        lambda e: (0, 255, 0) if e.get("source") == "ihc" else (0, 0, 255),
                        viz_mpp, strip,
                        alpha_of=lambda e: alpha_ihc if e.get("source") == "ihc" else alpha_he,
                        fill=fill)


def _he_view_fused(he_info, fused_info):
    """融合 dat（IHC proc 坐标）缩放到 HE proc 坐标系，保留 source 等字段。

    IHC 配准图边缘常有黑边（moving 没盖满 fixed 视野），直接用 IHC 底画会
    看起来被裁；以 HE 为底、把融合核整体缩放过来即可看到完整 HE 视野。
    """
    if isinstance(he_info, str):
        he_info = joblib.load(he_info)
    if isinstance(fused_info, str):
        fused_info = joblib.load(fused_info)
    hh, hw = _proc_dims(he_info)
    ih, iw = _proc_dims(fused_info)
    sx = (hw / iw) if iw else 1.0
    sy = (hh / ih) if ih else 1.0
    nuc = {k: _scale_entry(e, sx, sy) for k, e in fused_info["Nuclei"].items()}
    return {"Nuclei": nuc,
            "proc_resolution": he_info["proc_resolution"],
            "base_resolution": he_info["base_resolution"],
            "proc_dimensions": np.array(he_info["proc_dimensions"]),
            "base_dimensions": np.array(he_info["base_dimensions"])}


def viz_fused_on_he(he_wsi, he_info, fused_info, viz_dir, ihc_stem=None,
                    viz_mpp=0.5, strip=2048, fill=False, alpha_he=0.5, alpha_ihc=0.8):
    """HE 为底画融合结果（绿=IHC 保留，红=HE 补回）。

    与 IHC 底的双色图互补：IHC 配准图有黑边/视野小于 HE 时，这张能看完整
    HE 视野。存 <ihc_stem>_fused_on_HE_full.tif + preview。
    """
    view = _he_view_fused(he_info, fused_info)
    tag = ihc_stem or pathlib.Path(he_wsi).stem
    return _viz_overlay(he_wsi, view, viz_dir, f"{tag}_fused_on_HE",
                        lambda e: (0, 255, 0) if e.get("source") == "ihc" else (0, 0, 255),
                        viz_mpp, strip,
                        alpha_of=lambda e: alpha_ihc if e.get("source") == "ihc" else alpha_he,
                        fill=fill)


def _pair_batch(dat_dir):
    import re
    files = sorted(os.listdir(dat_dir))
    he = {f[:-len(" HE.dat")].strip(): f for f in files if f.endswith(" HE.dat")}
    # 也接受 "2617860-2.dat" 这种 HE 命名（IHC 为 "2617860-E_warped.dat"）。
    # 排除 IHC / 融合 / 阳性输出，避免把结果 dat 当成 HE。
    for f in files:
        if not f.endswith(".dat"):
            continue
        if (f.endswith("_warped.dat") or f.endswith("_HEfused.dat")
                or f.endswith("_pos.dat") or f.startswith("RegProj_")
                or "registered" in f or f.endswith(" HE.dat")):
            continue
        he.setdefault(pathlib.Path(f).stem, f)
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
        if match is None and len(he) == 1:
            # 回退：目录里只有 1 个 HE（如 2615223_to_1-HE_1 HE.dat 配
            # 2615223_to_1-IHC_*_warped.dat），默认全配给它
            match = next(iter(he))
        if match is None:
            missing.append(prefix)
        else:
            pairs.append((prefix, he[match], f))
    # 新命名：HE "2615223-1.dat"，IHC
    # "RegProj_2615223-C_to_2615223-1-moving_to_fixed_registered.ome.dat"
    # 编号 1/2/3/4 对应 C/E/K/P，按 fixed 编号配对
    he2 = {}
    for f in files:
        m = re.match(r"^(.+)-(\d+)\.dat$", f)
        if m and not f.startswith("RegProj_") and "registered" not in f:
            he2[(m.group(1), m.group(2))] = f
    for f in files:
        if not f.endswith("registered.ome.dat"):
            continue
        m = re.match(r"^RegProj_(.+)-([A-Za-z]+)_to_(.+)-(\d+)-moving_to_fixed_registered\.ome\.dat$", f)
        if not m:
            continue
        key = (m.group(3), m.group(4))
        tag = f"{m.group(2)}_to_{key[0]}-{key[1]}"  # 如 C_to_2615223-1
        if key in he2:
            pairs.append((tag, he2[key], f))
        else:
            missing.append(tag)
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
    p.add_argument("--on_he", type=str, default="none",
                   choices=["none", "ihc", "both", "fused", "all"],
                   help="以 HE 为底额外画图：ihc=只画IHC(绿)，both=HE(红)+IHC(绿)，"
                        "fused=融合结果(绿=IHC保留/红=HE补回)，all=三种都画；"
                        "需 --he_wsi_dir/--he_wsi")
    p.add_argument("--on_ihc", type=str, default="none",
                   choices=["none", "ihc"],
                   help="以 IHC 配准图为底额外画 IHC 标注：ihc=只画IHC(绿)；"
                        "需 --ihc_wsi_dir/--ihc_wsi")
    p.add_argument("--ihc_viz_dir", default=None,
                   help="单对模式 IHC 底 IHC 标注输出目录，默认 --viz_dir")
    p.add_argument("--fill", action="store_true",
                   help="标注多边形填充（默认只画轮廓）")
    p.add_argument("--he_fill_alpha", type=float, default=0.5,
                   help="HE 标注填充透明度 0-1，默认 0.5")
    p.add_argument("--ihc_fill_alpha", type=float, default=0.8,
                   help="IHC 标注填充透明度 0-1，默认 0.8")
    p.add_argument("--viz_mpp", type=float, default=0.5)
    p.add_argument("--viz_strip", type=int, default=2048,
                   help="整图分条带读取高度（viz 像素），默认 2048")
    a = p.parse_args()
    for _name in ("he_fill_alpha", "ihc_fill_alpha"):
        _v = getattr(a, _name)
        if not 0.0 <= _v <= 1.0:
            p.error(f"--{_name} 必须在 0-1 之间，当前 {_v}")

    if a.dat_dir:  # 批量
        assert a.out_dir, "--dat_dir 需配 --out_dir"
        pairs, missing = _pair_batch(a.dat_dir)
        print(f"paired {len(pairs)}, no-HE skipped {len(missing)}: {missing[:5]}")
        dat_out = os.path.join(a.out_dir, "dat")
        rows = []
        he_cache = {}  # 同一 HE.dat 只加载一次（1 个 HE 配多个 IHC 时省重复加载）
        he_viz_done = set()  # 已画过 HE viz 的 he_stem（每个 HE 只画一次）
        he_wsi_cache = {}  # he_stem -> HE 图路径（None 表示找不到，避免重复查找）
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
            fused_info, st = fuse_loaded(he_cache[he_path], ihc_info, out_dat,
                                         a.iou_thresh, a.dist_thresh, a.direct_overlay)
            st["prefix"] = prefix
            st["out"] = out_dat
            if not a.no_viz and a.ihc_wsi_dir:
                wsi = _find_wsi(a.ihc_wsi_dir, [ihc_stem], (".tiff", ".tif", ".svs"))
                if wsi:
                    viz_dual(wsi, out_dat, os.path.join(a.out_dir, "viz", ihc_stem),
                             a.viz_mpp, a.viz_strip, a.fill,
                             a.he_fill_alpha, a.ihc_fill_alpha)
                else:
                    print(f"[WARN] {ihc_stem}: IHC 图没找到，跳过 viz")
            if a.on_ihc == "ihc" and a.ihc_wsi_dir:
                wsi = _find_wsi(a.ihc_wsi_dir, [ihc_stem], (".tiff", ".tif", ".svs"))
                if wsi:
                    viz_ihc(wsi, ihc_info, os.path.join(a.out_dir, "viz_ihc", ihc_stem),
                            a.viz_mpp, a.viz_strip, fill=a.fill,
                            fill_alpha=a.ihc_fill_alpha)
                else:
                    print(f"[WARN] {ihc_stem}: IHC 图没找到，跳过 IHC 底 viz")
            elif a.on_ihc == "ihc":
                print("[WARN] --on_ihc ihc 需给 --ihc_wsi_dir")
            he_stem = pathlib.Path(he_f).stem
            if a.he_wsi_dir and (not a.no_he_viz or a.on_he != "none") and he_stem not in he_wsi_cache:
                cand = [he_stem] + ([he_stem[:-3]] if he_stem.endswith(" HE") else [])
                he_wsi_cache[he_stem] = _find_wsi(a.he_wsi_dir, cand,
                                                 (".svs", ".tiff", ".tif", ".mrxs", ".ndpi", ".scn"))
                if he_wsi_cache[he_stem] is None:
                    print(f"[WARN] {he_stem}: HE 图没找到，跳过 HE viz/on_he")
            he_wsi_path = he_wsi_cache.get(he_stem)
            if not a.no_he_viz and he_wsi_path:
                if he_stem not in he_viz_done:
                    viz_he(he_wsi_path, he_cache[he_path],
                           os.path.join(a.out_dir, "viz_he", he_stem),
                           a.viz_mpp, a.viz_strip, fill=a.fill,
                           fill_alpha=a.he_fill_alpha)
                    he_viz_done.add(he_stem)
            if a.on_he in ("ihc", "all") and he_wsi_path:
                viz_ihc_on_he(he_wsi_path, he_cache[he_path], ihc_info,
                              os.path.join(a.out_dir, "viz_he", he_stem),
                              ihc_stem, a.viz_mpp, a.viz_strip,
                              fill=a.fill, fill_alpha=a.ihc_fill_alpha)
            if a.on_he in ("both", "all") and he_wsi_path:
                viz_both_on_he(he_wsi_path, he_cache[he_path], ihc_info,
                               os.path.join(a.out_dir, "viz_he", he_stem),
                               ihc_stem, a.viz_mpp, a.viz_strip,
                               fill=a.fill, alpha_he=a.he_fill_alpha,
                               alpha_ihc=a.ihc_fill_alpha)
            if a.on_he in ("fused", "all") and he_wsi_path:
                viz_fused_on_he(he_wsi_path, he_cache[he_path], fused_info,
                                os.path.join(a.out_dir, "viz_he", he_stem),
                                ihc_stem, a.viz_mpp, a.viz_strip,
                                fill=a.fill, alpha_he=a.he_fill_alpha,
                                alpha_ihc=a.ihc_fill_alpha)
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
        fused_info, st = fuse_pair(a.he_dat, a.ihc_dat, a.out_dat, a.iou_thresh, a.dist_thresh,
                                   a.direct_overlay)
        print(f"[{st['mode']}] HE={st['he_n']} IHC={st['ihc_n']} keep={st['kept_ihc']} "
              f"drop={st['dropped_ihc']} supp={st['he_supplement']} fused={st['fused_n']} {st['secs']}s")
        print(f"fused -> {a.out_dat}")
        if not a.no_viz and a.ihc_wsi and a.viz_dir:
            viz_dual(a.ihc_wsi, a.out_dat, a.viz_dir, a.viz_mpp, a.viz_strip,
                     a.fill, a.he_fill_alpha, a.ihc_fill_alpha)
        if a.on_ihc == "ihc" and a.ihc_wsi:
            ihc_viz_dir = a.ihc_viz_dir or a.viz_dir
            if ihc_viz_dir:
                viz_ihc(a.ihc_wsi, a.ihc_dat, ihc_viz_dir, a.viz_mpp, a.viz_strip,
                        fill=a.fill, fill_alpha=a.ihc_fill_alpha)
            else:
                print("[WARN] 要画 IHC 底 viz 需给 --ihc_viz_dir（或 --viz_dir）")
        elif a.on_ihc == "ihc":
            print("[WARN] on_ihc 需给 --ihc_wsi")
        if not a.no_he_viz and a.he_wsi:
            he_viz_dir = a.he_viz_dir or a.viz_dir
            if he_viz_dir:
                viz_he(a.he_wsi, a.he_dat, he_viz_dir, a.viz_mpp, a.viz_strip,
                       fill=a.fill, fill_alpha=a.he_fill_alpha)
            else:
                print("[WARN] 要画 HE viz 需给 --he_viz_dir（或 --viz_dir）")
        if a.on_he != "none" and a.he_wsi:
            he_viz_dir = a.he_viz_dir or a.viz_dir
            if he_viz_dir:
                _ihc_stem = pathlib.Path(a.ihc_dat).stem
                if a.on_he in ("ihc", "all"):
                    viz_ihc_on_he(a.he_wsi, a.he_dat, a.ihc_dat, he_viz_dir,
                                  _ihc_stem, a.viz_mpp, a.viz_strip,
                                  fill=a.fill, fill_alpha=a.ihc_fill_alpha)
                if a.on_he in ("both", "all"):
                    viz_both_on_he(a.he_wsi, a.he_dat, a.ihc_dat, he_viz_dir,
                                   _ihc_stem, a.viz_mpp, a.viz_strip,
                                   fill=a.fill, alpha_he=a.he_fill_alpha,
                                   alpha_ihc=a.ihc_fill_alpha)
                if a.on_he in ("fused", "all"):
                    viz_fused_on_he(a.he_wsi, a.he_dat, fused_info, he_viz_dir,
                                    _ihc_stem, a.viz_mpp, a.viz_strip,
                                    fill=a.fill, alpha_he=a.he_fill_alpha,
                                    alpha_ihc=a.ihc_fill_alpha)
            else:
                print("[WARN] on_he 需给 --he_viz_dir（或 --viz_dir）")
        elif a.on_he != "none":
            print("[WARN] on_he 需给 --he_wsi")


if __name__ == "__main__":
    main()
