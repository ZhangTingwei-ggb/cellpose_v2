#!/usr/bin/env python
"""三张图（不做融合消除）：

  1. HE 底图 + HE 标注（红）
  2. IHC 配准图 + IHC 标注（绿）
  3. HE 底图 + HE 标注（红）+ IHC 标注（绿），两套全量叠加

只读 dat 和 WSI，不做融合、不写融合 dat；适合反复调透明度。

单对：
  python viz_three.py \
    --he_dat <HE.dat> --ihc_dat <IHC_warped.dat> \
    --he_wsi <HE.svs> --ihc_wsi <IHC_warped.tiff> \
    --out <输出根> --fill --he_fill_alpha 0.5 --ihc_fill_alpha 0.8

批量：
  python viz_three.py \
    --dat_dir <含HE.dat和*_warped.dat的目录> \
    --he_wsi_dir <HE图目录> --ihc_wsi_dir <IHC图目录> \
    --out <输出根> --fill --he_fill_alpha 0.5 --ihc_fill_alpha 0.8

输出：
  <out>/viz_he/<HE名>/<HE名>_HE_full.tif + _preview.png
  <out>/viz_ihc/<IHC名>/<IHC名>_IHC_full.tif + _preview.png
  <out>/viz_he/<HE名>/<IHC名>_both_on_HE_full.tif + _preview.png
"""
import argparse
import os
import pathlib

import joblib

from fuse_he_ihc import _find_wsi, _pair_batch, viz_both_on_he, viz_he, viz_ihc


def _check_alpha(p, name, v):
    if not 0.0 <= v <= 1.0:
        p.error(f"--{name} 必须在 0-1 之间，当前 {v}")


def main():
    p = argparse.ArgumentParser(description="三张图：HE+HE、IHC+IHC、HE+HE+IHC（不融合消除）")
    p.add_argument("--he_dat", default=None, help="单对：HE dat")
    p.add_argument("--ihc_dat", default=None, help="单对：IHC dat")
    p.add_argument("--he_wsi", default=None, help="单对：HE 原图")
    p.add_argument("--ihc_wsi", default=None, help="单对：IHC 配准图")
    p.add_argument("--dat_dir", default=None, help="批量：含 HE.dat 和 *_warped.dat 的目录")
    p.add_argument("--he_wsi_dir", default=None, help="批量：HE 原图目录")
    p.add_argument("--ihc_wsi_dir", default=None, help="批量：IHC 配准图目录")
    p.add_argument("--out", required=True, help="输出根目录")
    p.add_argument("--fill", action="store_true", default=True,
                   help="填充标注（默认开；用 --no_fill 关闭）")
    p.add_argument("--no_fill", dest="fill", action="store_false", help="只画轮廓，不填充")
    p.add_argument("--he_fill_alpha", type=float, default=0.5,
                   help="HE 标注填充 alpha，0-1，默认 0.5（50%% 不透明）")
    p.add_argument("--ihc_fill_alpha", type=float, default=0.8,
                   help="IHC 标注填充 alpha，0-1，默认 0.8（80%% 不透明）")
    p.add_argument("--viz_mpp", type=float, default=0.5, help="可视化 mpp，默认 0.5")
    p.add_argument("--viz_strip", type=int, default=2048, help="分条带读取高度，默认 2048")
    a = p.parse_args()
    _check_alpha(p, "he_fill_alpha", a.he_fill_alpha)
    _check_alpha(p, "ihc_fill_alpha", a.ihc_fill_alpha)

    out = pathlib.Path(a.out)
    if a.dat_dir:
        if not (a.he_wsi_dir and a.ihc_wsi_dir):
            p.error("批量模式需 --he_wsi_dir 和 --ihc_wsi_dir")
        pairs, missing = _pair_batch(a.dat_dir)
        print(f"paired {len(pairs)}, no-HE skipped {len(missing)}: {missing[:5]}", flush=True)
        if not pairs:
            raise SystemExit("[ERR] 没有配对成功的 HE/IHC dat，检查 dat 命名")
        he_cache = {}
        for n, (prefix, he_f, ihc_f) in enumerate(pairs, 1):
            he_stem = pathlib.Path(he_f).stem
            ihc_stem = pathlib.Path(ihc_f).stem
            he_dat = os.path.join(a.dat_dir, he_f)
            ihc_dat = os.path.join(a.dat_dir, ihc_f)
            he_wsi = _find_wsi(a.he_wsi_dir, [he_stem],
                               (".svs", ".tiff", ".tif", ".mrxs", ".ndpi", ".scn"))
            ihc_wsi = _find_wsi(a.ihc_wsi_dir, [ihc_stem], (".tiff", ".tif", ".svs"))
            if not he_wsi or not ihc_wsi:
                print(f"[WARN] {prefix}: 找不到 HE 或 IHC 图，跳过", flush=True)
                continue
            print(f"[{n}/{len(pairs)}] {prefix} ...", flush=True)
            if he_dat not in he_cache:
                print(f"[load] HE {he_dat}", flush=True)
                he_cache[he_dat] = joblib.load(he_dat)
            he_info = he_cache[he_dat]
            ihc_info = joblib.load(ihc_dat)
            viz_he(he_wsi, he_info, out / "viz_he" / he_stem,
                   a.viz_mpp, a.viz_strip, fill=a.fill, fill_alpha=a.he_fill_alpha)
            viz_ihc(ihc_wsi, ihc_info, out / "viz_ihc" / ihc_stem,
                    a.viz_mpp, a.viz_strip, fill=a.fill, fill_alpha=a.ihc_fill_alpha)
            viz_both_on_he(he_wsi, he_info, ihc_info, out / "viz_he" / he_stem,
                           ihc_stem, a.viz_mpp, a.viz_strip, fill=a.fill,
                           alpha_he=a.he_fill_alpha, alpha_ihc=a.ihc_fill_alpha)
    else:
        if not (a.he_dat and a.ihc_dat and a.he_wsi and a.ihc_wsi):
            p.error("单对模式需 --he_dat --ihc_dat --he_wsi --ihc_wsi")
        he_stem = pathlib.Path(a.he_dat).stem
        ihc_stem = pathlib.Path(a.ihc_dat).stem
        print(f"[load] HE {a.he_dat}", flush=True)
        he_info = joblib.load(a.he_dat)
        print(f"[load] IHC {a.ihc_dat}", flush=True)
        ihc_info = joblib.load(a.ihc_dat)
        viz_he(a.he_wsi, he_info, out / "viz_he" / he_stem,
               a.viz_mpp, a.viz_strip, fill=a.fill, fill_alpha=a.he_fill_alpha)
        viz_ihc(a.ihc_wsi, ihc_info, out / "viz_ihc" / ihc_stem,
                a.viz_mpp, a.viz_strip, fill=a.fill, fill_alpha=a.ihc_fill_alpha)
        viz_both_on_he(a.he_wsi, he_info, ihc_info, out / "viz_he" / he_stem,
                       ihc_stem, a.viz_mpp, a.viz_strip, fill=a.fill,
                       alpha_he=a.he_fill_alpha, alpha_ihc=a.ihc_fill_alpha)

    print(f"\n完成：\n  {out}/viz_he/<HE名>/<HE名>_HE_full.tif"
          f"\n  {out}/viz_ihc/<IHC名>/<IHC名>_IHC_full.tif"
          f"\n  {out}/viz_he/<HE名>/<IHC名>_both_on_HE_full.tif", flush=True)


if __name__ == "__main__":
    main()
