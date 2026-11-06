#!/usr/bin/env python
"""一条命令：HE/IHC dat 合并 -> 融合 -> 阳性占比 + HE 底橙/蓝图。

四个输入目录：
  --he_img   HE 原图目录（.svs / .tiff）
  --ihc_img  IHC 配准图目录（*_warped.tiff）
  --he_dat   HE dat 目录
  --ihc_dat  IHC dat 目录

做的事：
  1. 把两个 dat 目录软链接到 <out>/dat_all（同名 dat 会报错，不覆盖）。
  2. 调 fuse_he_ihc.py 生成 <out>/fused/dat/*_HEfused.dat。
  3. 调 ihc_positivity.py 生成：
       <out>/pos/dat/*_pos.dat
       <out>/pos/positivity.csv
       <out>/pos/viz_he/<HE名>/<IHC名>_pos_on_HE_full.tif + preview
       （橙=阳性，蓝=阴性；--no_pos_viz 可关）

用法：
  python run_fuse_pos.py \
    --he_img  <HE图目录> \
    --ihc_img <IHC配准图目录> \
    --he_dat  <HE_dat目录> \
    --ihc_dat <IHC_dat目录> \
    --out     <输出根目录>

只重切阈值时不用重跑本脚本，直接：
  python ihc_positivity.py --fuse_dat <out>/pos/dat/<...>_pos.dat \
    --ihc_wsi <占位> --out_pos <...> --dab_thresh 0.3
"""

import argparse
import os
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent


def _dat_files(d):
    p = pathlib.Path(d)
    if not p.is_dir():
        raise SystemExit(f"[ERR] dat 目录不存在：{d}")
    files = sorted(p.glob("*.dat"))
    if not files:
        raise SystemExit(f"[ERR] {d} 里没有 *.dat")
    return files


def merge_dats(he_dat_dir, ihc_dat_dir, dat_all):
    """两个 dat 目录软链接到同一目录（同名不同源直接报错，避免配错）。"""
    dat_all.mkdir(parents=True, exist_ok=True)
    for src in _dat_files(he_dat_dir) + _dat_files(ihc_dat_dir):
        dst = dat_all / src.name
        if dst.exists() or dst.is_symlink():
            if dst.resolve() == src.resolve():
                continue
            raise SystemExit(f"[ERR] dat 重名且来源不同：\n  {dst.resolve()}\n  {src.resolve()}")
        os.symlink(src.resolve(), dst)
    print(f"[1/2] dat 合并 -> {dat_all}（{len(list(dat_all.glob('*.dat')))} 个）", flush=True)


def main():
    p = argparse.ArgumentParser(description="一条命令跑融合 + 阳性占比（HE 底橙/蓝图）")
    p.add_argument("--he_img", required=True, help="HE 原图目录")
    p.add_argument("--ihc_img", required=True, help="IHC 配准图目录（*_warped.tiff）")
    p.add_argument("--he_dat", required=True, help="HE dat 目录")
    p.add_argument("--ihc_dat", required=True, help="IHC dat 目录")
    p.add_argument("--out", required=True, help="输出根目录")
    p.add_argument("--dab_thresh", type=float, default=0.2,
                   help="DAB OD 阈值（默认 0.2，仅 fixed 方法生效）")
    p.add_argument("--thr_method", default="fixed", choices=["fixed", "triangle", "otsu"])
    p.add_argument("--viz_mpp", type=float, default=0.5)
    p.add_argument("--viz_strip", type=int, default=2048)
    p.add_argument("--no_pos_viz", action="store_true", help="只算占比，不画橙蓝图")
    p.add_argument("--keep_fuse_viz", action="store_true",
                   help="保留融合的 IHC 底绿/红双色图（默认关，省时间）")
    p.add_argument("--dry_run", action="store_true", help="只打印命令，不执行")
    a = p.parse_args()

    out = pathlib.Path(a.out)
    dat_all = out / "dat_all"
    fused = out / "fused"
    pos = out / "pos"

    merge_dats(a.he_dat, a.ihc_dat, dat_all)

    fuse_cmd = [
        sys.executable, str(HERE / "fuse_he_ihc.py"),
        "--dat_dir", str(dat_all),
        "--out_dir", str(fused),
        "--ihc_wsi_dir", a.ihc_img,
        "--viz_mpp", str(a.viz_mpp),
        "--viz_strip", str(a.viz_strip),
    ]
    if not a.keep_fuse_viz:
        fuse_cmd += ["--no_viz", "--no_he_viz"]

    pos_cmd = [
        sys.executable, str(HERE / "ihc_positivity.py"),
        "--dat_dir", str(dat_all),
        "--fuse_dir", str(fused / "dat"),
        "--out_dir", str(pos),
        "--ihc_wsi_dir", a.ihc_img,
        "--dab_thresh", str(a.dab_thresh),
        "--thr_method", a.thr_method,
        "--viz_mpp", str(a.viz_mpp),
        "--viz_strip", str(a.viz_strip),
    ]
    if not a.no_pos_viz:
        pos_cmd += ["--he_wsi_dir", a.he_img]

    print("[2/2] 融合：", " ".join(fuse_cmd), flush=True)
    if not a.dry_run:
        subprocess.run(fuse_cmd, check=True)
        fused_files = sorted((fused / "dat").glob("*_HEfused.dat"))
        if not fused_files:
            raise SystemExit(
                "[ERR] 融合没有生成任何 *_HEfused.dat，通常是 HE/IHC dat 命名没能配对；"
                "请看上面的 paired/no-HE skipped 统计。")
        print("[2/2] 阳性占比：", " ".join(pos_cmd), flush=True)
        subprocess.run(pos_cmd, check=True)
        pos_files = sorted((pos / "dat").glob("*_pos.dat"))
        if not pos_files:
            raise SystemExit(
                "[ERR] 阳性没有生成任何 *_pos.dat，请检查上面 Positivity batch 的配对/缺失提示。")
        print(f"\n完成：\n  {pos / 'positivity.csv'}\n  {pos / 'viz_he'}/<HE名>/*_pos_on_HE_preview.png", flush=True)
    else:
        print("[dry_run] 阳性占比：", " ".join(pos_cmd), flush=True)


if __name__ == "__main__":
    main()
