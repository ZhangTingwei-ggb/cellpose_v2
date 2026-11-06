#!/usr/bin/env python
"""IHC 整体阳性占比（阳性核 / 全部检出核）+ HE 为底橙蓝可视化。

不碰 fuse_he_ihc.py 的融合逻辑，只消费它的结果。

输入（三样）：
  融合 dat（*_HEfused.dat，过滤模式，一个细胞一条轮廓，IHC 坐标系）
    兼容直接传原始 IHC *_warped.dat（此时全按 IHC 核处理）。
  IHC 配准图（*_warped.tiff）：读 DAB 信号用。
  HE 原图 + HE.dat：只用于 HE 底可视化的坐标系（不要可视化可不给）。

算法：
  1. 融合核表（IHC 保留 + HE 补回）逐核读 DAB：IHC 图分条带直读，
     RGB->OD，反卷积拆 DAB 通道（Ruifrok H&E-DAB 矩阵手写实现），
     contour 内平均 DAB OD 为阳性分数。检出来源不决定阴阳，
     HE 补回的核也照样打分（IHC 漏的是轮廓不是染色）。
  2. 成片定阈值二分阴阳：默认固定 DAB OD 0.2（--dab_thresh 可调；
     蓝色复染串扰上限约 0.2，棕色真阳性一般 >0.25，抽查 preview 后按需调），
     另有 triangle/otsu 自动方法备选（阴阳极不平衡时慎用）。
     近黑墨迹伪影核剔除不计。已打过分的 pos dat 可直接重切阈值（秒级，不读图）。
  3. 输出：每个核回写 dab_score/pos；positivity.csv；HE 为底
     橙=阳性、蓝=阴性的整张 BigTIFF + preview。

批量：
  python ihc_positivity.py --dat_dir .../dat --fuse_dir .../fused/dat \\
    --out_dir .../pos --ihc_wsi_dir .../IHC --he_wsi_dir .../HE
单对：
  python ihc_positivity.py --fuse_dat ..._HEfused.dat --ihc_wsi ....tiff \\
    --out_pos ..._pos.dat --he_dat "... HE.dat" --he_wsi .../HE.svs \\
    --he_viz_dir .../viz_he
只要打分不要 HE 图：不给 --he_wsi_dir/--he_wsi 即可（--on_he none）。
"""

import argparse
import csv
import os
import pathlib

import cv2
import joblib
import numpy as np
import tqdm

from fuse_he_ihc import (
    _proc_dims,
    _scale_entry,
    _xy_box,
    _pair_batch,
    _find_wsi,
    _viz_overlay,
)

# Ruifrok H&E-DAB 吸光向量（行=H/E/DAB），DAB 为所求通道
_HED_M = np.array([[0.65, 0.70, 0.29],
                   [0.07, 0.99, 0.11],
                   [0.27, 0.57, 0.82]], dtype=np.float64)
_DAB_W = np.linalg.inv(_HED_M)[:, 2].astype(np.float32)  # OD(3) . w -> DAB

# 近黑像素视为墨迹/杂质/折叠伪影，不参与打分（棕 DAB 最暗通道仍 > 该值）
INK_MAX = 50


def dab_channel(rgb):
    """uint8 RGB -> DAB OD 单通道 float32（与 skimage rgb2hed 第三通道一致）。"""
    od = -np.log10(np.clip((rgb.astype(np.float32) + 1.0) / 255.0, 1e-6, 1.0))
    return od.dot(_DAB_W)


def triangle_threshold(scores, nbins=512):
    """三角法阈值（单峰长尾分布用；Otsu 在阴阳极不平衡时会切进阴性峰内）。"""
    s = np.asarray(scores, dtype=np.float64)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return 0.0
    hi = float(np.percentile(s, 99.99))
    lo = float(s.min())
    if hi <= lo:
        return lo
    hist, edges = np.histogram(s, bins=nbins, range=(lo, hi))
    peak = int(np.argmax(hist))
    last = int(np.nonzero(hist > 0)[0][-1])
    if last <= peak:
        return float(edges[peak])
    x0, y0 = float(peak), float(hist[peak])
    x1, y1 = float(last), float(hist[last])
    dx, dy = x1 - x0, y1 - y0
    norm = float(np.hypot(dx, dy)) or 1.0
    idx = np.arange(len(hist))
    dist = np.abs(dy * idx - dx * hist + x1 * y0 - y1 * x0) / norm
    dist[:peak] = -1.0
    t = int(np.argmax(dist))
    return float((edges[t] + edges[t + 1]) / 2.0)


def otsu_threshold(scores, nbins=256):
    """一维 Otsu（numpy 手写，无额外依赖）。"""
    s = np.asarray(scores, dtype=np.float64)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return 0.0
    lo, hi = float(s.min()), float(s.max())
    if hi <= lo:
        return lo
    hist, edges = np.histogram(s, bins=nbins, range=(lo, hi))
    p = hist / hist.sum()
    omega = np.cumsum(p)
    mu = np.cumsum(p * np.arange(nbins))
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b2 = np.where(denom > 0, (mu_t * omega - mu) ** 2 / denom, 0.0)
    idx = int(np.argmax(sigma_b2))
    return float((edges[idx] + edges[idx + 1]) / 2.0)


def _open_wsi_array(wsi_path):
    """tifffile+zarr 基准层直读（兼容 OME Group），返回 (fh, z, bh, bw)。"""
    import tifffile as _tf
    import zarr as _zarr
    _fh = _tf.TiffFile(wsi_path)
    _z = _zarr.open(_fh.aszarr(), mode="r")
    if not hasattr(_z, "shape") and hasattr(_z, "keys"):
        _keys = list(_z.keys())
        _z = _z["0"] if "0" in _keys else _z[_keys[0]]
    return _fh, _z, int(_z.shape[0]), int(_z.shape[1])


def score_fused(fused_info, wsi_path, strip_rows=512):
    """对融合核表逐核打 DAB 分，返回 scores（与 keys 同序）。

    fused 轮廓是 proc 坐标（IHC 坐标系）；按 f=proc_mpp/base_mpp 放到
    基准层像素 Huge 上打分，不做插值。跨条带的核用 sum/count 累加后平均。
    """
    nuc = fused_info["Nuclei"]
    keys = list(nuc.keys())
    n = len(keys)
    proc_mpp = float(np.array(fused_info["proc_resolution"]["resolution"]).flat[0])
    base_mpp = float(np.array(fused_info["base_resolution"]["resolution"]).flat[0])
    f = proc_mpp / base_mpp

    boxes = np.array([_xy_box(nuc[k]["box"]) * f for k in keys])  # x0,y0,x1,y1 基准层
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, None)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, None)
    conts = [np.asarray(nuc[k]["contour"], dtype=float) * f for k in keys]

    sums = np.zeros(n, dtype=np.float64)
    cnts = np.zeros(n, dtype=np.int64)
    ink_frac = np.zeros(n, dtype=np.float64)  # 核内近黑像素占比（伪影指示）

    _fh, _z, bh, bw = _open_wsi_array(wsi_path)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, bw)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, bh)
    y0_all = boxes[:, 1].astype(np.int64)
    y1_all = np.maximum(boxes[:, 3].astype(np.int64), y0_all + 1)
    try:
        for r0 in tqdm.tqdm(range(0, bh, strip_rows), desc="DAB scoring", ncols=90):
            r1 = min(r0 + strip_rows, bh)
            sel = np.nonzero((y0_all < r1) & (y1_all > r0))[0]
            if len(sel) == 0:
                continue
            block = np.asarray(_z[r0:r1, :, :])
            if block.shape[-1] == 4:
                block = block[..., :3]
            for i in sel:
                x0 = int(max(0, boxes[i][0]))
                x1 = int(min(bw, max(x0 + 1, boxes[i][2])))
                ya = int(max(r0, y0_all[i]))
                yb = int(min(r1, y1_all[i]))
                if yb <= ya or x1 <= x0:
                    continue
                pts = np.round(conts[i] - np.array([x0, ya])).astype(np.int32)
                m = np.zeros((yb - ya, x1 - x0), dtype=np.uint8)
                cv2.fillPoly(m, [pts], 1)
                inside = m > 0
                k_all = int(inside.sum())
                if k_all == 0:
                    continue
                pix = block[ya - r0:yb - r0, x0:x1][inside]
                ink = pix.max(axis=-1) < INK_MAX  # 近黑=墨迹/杂质
                ink_frac[i] += float(ink.sum())
                cnts[i] += k_all
                keep = pix[~ink]
                if keep.shape[0] > 0:
                    od = -np.log10(np.clip((keep.astype(np.float32) + 1.0) / 255.0, 1e-6, 1.0))
                    sums[i] += float(od.dot(_DAB_W).sum())
    finally:
        try:
            _fh.close()
        except Exception:
            pass
    scores = np.divide(sums, np.maximum(cnts, 1),
                       out=np.zeros_like(sums), where=cnts > 0)
    ink_frac = np.divide(ink_frac, np.maximum(cnts, 1),
                         out=np.zeros_like(ink_frac), where=cnts > 0)
    return keys, scores, ink_frac


def label_and_save(fused_info_or_path, wsi_path, out_pos_dat, thresh=0.2,
                   strip_rows=512, thr_method="fixed"):
    """打分 + 定阈值 + 回写 pos dat，返回 stats dict（含 keys/scores/thresh）。

    输入 dat 若已有 dab_score（即 *_pos.dat），直接复用分数只重切阈值，
    不再读图（调阈值秒级完成）。
    """
    info = joblib.load(fused_info_or_path) if isinstance(fused_info_or_path, str) else fused_info_or_path
    nuc0 = next(iter(info["Nuclei"].values()))
    if "dab_score" in nuc0:
        keys = list(info["Nuclei"].keys())
        scores = np.array([info["Nuclei"][k]["dab_score"] for k in keys], dtype=float)
        ink_frac = np.array([1.0 if info["Nuclei"][k].get("artifact") else 0.0 for k in keys])
        print("[pos] reuse stored dab_score, skip image reading", flush=True)
    else:
        keys, scores, ink_frac = score_fused(info, wsi_path, strip_rows)
    # 伪影核（核内近黑像素过半，如墨迹）：不参与阴阳统计，单独计数
    evaluable = ink_frac <= 0.5
    if thr_method == "otsu":
        th = otsu_threshold(scores[evaluable])
        method_used = "otsu"
    elif thr_method == "triangle":
        th = triangle_threshold(scores[evaluable])
        method_used = "triangle"
    else:
        th = float(thresh) if thresh is not None else 0.2
        method_used = "fixed"
    pos_all = np.zeros(len(keys), dtype=bool)
    pos_all[evaluable] = scores[evaluable] >= th
    nuc = info["Nuclei"]
    for k, s, p, f_ink, ev in zip(keys, scores, pos_all, ink_frac, evaluable):
        nuc[k]["dab_score"] = float(s)
        nuc[k]["pos"] = bool(p)
        nuc[k]["artifact"] = bool(not ev)
    if out_pos_dat:
        os.makedirs(os.path.dirname(out_pos_dat) or ".", exist_ok=True)
        joblib.dump(info, out_pos_dat)
    n_eval = int(evaluable.sum())
    n_pos = int(pos_all.sum())
    return {"info": info, "keys": keys, "scores": scores, "thresh": th,
            "method": method_used,
            "total": int(len(keys)), "evaluable": n_eval,
            "artifact_n": int(len(keys) - n_eval),
            "pos_n": n_pos, "neg_n": n_eval - n_pos,
            "pct": float(n_pos / n_eval) if n_eval else 0.0}


def _he_view_labeled(he_info, labeled_info, ihc_stem=None):
    """融合 dat（IHC 坐标，带 pos）缩放到 HE 坐标系，供 HE 底画图。"""
    if isinstance(he_info, str):
        he_info = joblib.load(he_info)
    hh, hw = _proc_dims(he_info)
    ih, iw = _proc_dims(labeled_info)
    sx = (hw / iw) if iw else 1.0
    sy = (hh / ih) if ih else 1.0
    nuc = {}
    for k, e in labeled_info["Nuclei"].items():
        if e.get("artifact"):
            continue  # 墨迹伪影核不画，避免污染橙蓝两色
        c = _scale_entry(e, sx, sy)
        nuc[k] = c
    return {"Nuclei": nuc,
            "proc_resolution": he_info["proc_resolution"],
            "base_resolution": he_info["base_resolution"],
            "proc_dimensions": np.array(he_info["proc_dimensions"]),
            "base_dimensions": np.array(he_info["base_dimensions"])}


def viz_pos_on_he(he_wsi, he_info, labeled_info, viz_dir, ihc_stem=None,
                  viz_mpp=0.5, strip=2048):
    """HE 为底：橙=阳性，蓝=阴性。存 <ihc_stem>_pos_on_HE_full.tif + preview。"""
    view = _he_view_labeled(he_info, labeled_info, ihc_stem)
    tag = ihc_stem or pathlib.Path(he_wsi).stem
    return _viz_overlay(he_wsi, view, viz_dir, f"{tag}_pos_on_HE",
                        lambda e: (0, 165, 255) if e.get("pos") else (255, 0, 0),
                        viz_mpp, strip)


def main():
    p = argparse.ArgumentParser(description="IHC overall positivity + HE-base orange/blue viz")
    p.add_argument("--dat_dir", default=None, help="批量：原始 HE.dat + *_warped.dat 目录（用于配对）")
    p.add_argument("--fuse_dir", default=None, help="批量：*_HEfused.dat 目录")
    p.add_argument("--out_dir", default=None, help="批量输出根目录（dat/ viz_he/ positivity.csv）")
    p.add_argument("--ihc_wsi_dir", default=None)
    p.add_argument("--he_wsi_dir", default=None, help="给了就画 HE 底橙蓝图")
    p.add_argument("--fuse_dat", default=None, help="单对：融合 dat（或原始 IHC dat）")
    p.add_argument("--ihc_wsi", default=None)
    p.add_argument("--out_pos", default=None, help="单对：输出 pos dat")
    p.add_argument("--he_dat", default=None, help="单对 HEviz 用")
    p.add_argument("--he_wsi", default=None)
    p.add_argument("--he_viz_dir", default=None)
    p.add_argument("--dab_thresh", type=float, default=0.2,
                   help="固定阈值（默认 0.2；仅 --thr_method fixed 生效）")
    p.add_argument("--thr_method", type=str, default="fixed",
                   choices=["fixed", "triangle", "otsu"],
                   help="定阈值方法：fixed（默认，用 --dab_thresh），triangle/otsu 自动（阴阳不平衡时慎用）")
    p.add_argument("--strip_rows", type=int, default=512, help="DAB 打分基准层条带高度")
    p.add_argument("--viz_mpp", type=float, default=0.5)
    p.add_argument("--viz_strip", type=int, default=2048)
    a = p.parse_args()

    if a.dat_dir:
        assert a.fuse_dir and a.out_dir and a.ihc_wsi_dir, "--dat_dir 需配 --fuse_dir --out_dir --ihc_wsi_dir"
        pairs, missing = _pair_batch(a.dat_dir)
        print(f"paired {len(pairs)}, no-HE skipped {len(missing)}: {missing[:5]}")
        dat_out = os.path.join(a.out_dir, "dat")
        he_cache, he_wsi_cache = {}, {}
        rows = []
        for n, (prefix, he_f, ihc_f) in enumerate(tqdm.tqdm(pairs, desc="Positivity batch", ncols=90), 1):
            ihc_stem = pathlib.Path(ihc_f).stem
            fuse_dat = os.path.join(a.fuse_dir, f"{ihc_stem}_HEfused.dat")
            if not os.path.isfile(fuse_dat):
                print(f"[WARN] {prefix}: 融合 dat 不存在 {fuse_dat}，跳过")
                continue
            wsi = _find_wsi(a.ihc_wsi_dir, [ihc_stem], (".tiff", ".tif", ".svs"))
            if not wsi:
                print(f"[WARN] {ihc_stem}: IHC 图没找到，跳过")
                continue
            print(f"[{n}/{len(pairs)}] {prefix} scoring ...", flush=True)
            out_pos = os.path.join(dat_out, f"{ihc_stem}_HEfused_pos.dat")
            st = label_and_save(fuse_dat, wsi, out_pos, a.dab_thresh, a.strip_rows,
                                a.thr_method)
            row = {"prefix": prefix, "total": st["total"], "evaluable": st["evaluable"],
                   "artifact": st["artifact_n"], "pos": st["pos_n"],
                   "neg": st["neg_n"], "pct": round(st["pct"], 4),
                   "thresh": round(st["thresh"], 4), "method": st["method"],
                   "out": out_pos}
            if a.he_wsi_dir:
                he_path = os.path.join(a.dat_dir, he_f)
                if he_path not in he_cache:
                    he_cache[he_path] = joblib.load(he_path)
                he_stem = pathlib.Path(he_f).stem
                if he_stem not in he_wsi_cache:
                    cand = [he_stem] + ([he_stem[:-3]] if he_stem.endswith(" HE") else [])
                    he_wsi_cache[he_stem] = _find_wsi(a.he_wsi_dir, cand,
                                                      (".svs", ".tiff", ".tif", ".mrxs", ".ndpi", ".scn"))
                he_wsi = he_wsi_cache[he_stem]
                if he_wsi:
                    viz_pos_on_he(he_wsi, he_cache[he_path], st["info"],
                                  os.path.join(a.out_dir, "viz_he", he_stem),
                                  ihc_stem, a.viz_mpp, a.viz_strip)
                else:
                    print(f"[WARN] {he_stem}: HE 图没找到，跳过 HE viz")
            rows.append(row)
            print(f"{prefix}: total={row['total']} eval={row['evaluable']} art={row['artifact']} "
                  f"pos={row['pos']} neg={row['neg']} pct={row['pct']} th={row['thresh']}")
        if rows:
            with open(os.path.join(a.out_dir, "positivity.csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            print(f"stats -> {a.out_dir}/positivity.csv")
    else:
        assert a.fuse_dat and a.ihc_wsi and a.out_pos, "单对需 --fuse_dat --ihc_wsi --out_pos"
        st = label_and_save(a.fuse_dat, a.ihc_wsi, a.out_pos, a.dab_thresh, a.strip_rows,
                            a.thr_method)
        print(f"total={st['total']} eval={st['evaluable']} art={st['artifact_n']} "
              f"pos={st['pos_n']} neg={st['neg_n']} pct={st['pct']:.4f} th={st['thresh']:.4f}\npos -> {a.out_pos}")
        if a.he_wsi and a.he_dat:
            he_viz_dir = a.he_viz_dir or os.path.dirname(a.out_pos)
            _stem = pathlib.Path(a.fuse_dat).stem
            _tag = _stem.replace("_HEfused_pos", "").replace("_HEfused", "")
            viz_pos_on_he(a.he_wsi, a.he_dat, st["info"], he_viz_dir,
                          _tag, a.viz_mpp, a.viz_strip)


if __name__ == "__main__":
    main()
