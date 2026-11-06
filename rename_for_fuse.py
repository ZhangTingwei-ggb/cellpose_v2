#!/usr/bin/env python3
"""Bridge wsireg outputs -> fuse_he_ihc naming convention (dry-run by default).

Direction: moving=IHC (aligned), fixed=HE (reference, untouched).

fuse_he_ihc batch mode expects, per case, e.g.:
    HE dat :  "2615223-1 HE.dat"          + HE image : "2615223-1.svs"
    IHC dat:  "2615223-1 C_warped.dat"    + IHC image: "2615223-1 C_warped.tiff"
  (IHC prefix "<HE> <STAIN>" must start with "<HE> ".)

wsireg GUI instead produces, per pair folder:
    "<Prefix>_<IHC>_to_<HE>-moving_to_fixed_registered.ome.tiff"

This script, driven by reg_list.csv (moving=IHC, fixed=HE):
  mode stage1 (default): find each pair's registered OME-TIFF under --reg-root
                         and link/copy it to "<HE> <STAIN>_warped.tiff".
  mode stage2: after cellpose, rename "<HE>.dat" -> "<HE> HE.dat"
               (IHC dats need no rename if cellpose ran on the _warped.tiff).

Run without --execute to preview only.
"""

import argparse
import csv
import os
import re
import sys
from pathlib import Path

DEFAULT_CSV = "/home/linjiatai/\u684c\u9762/reg_list.csv"


def base(name: str) -> str:
    """Strip one image suffix (.svs/.tiff/.tif) for matching & naming."""
    n = name.strip()
    low = n.lower()
    for ext in (".ome.tiff", ".tiff", ".tif", ".svs"):
        if low.endswith(ext):
            return n[: -len(ext)]
    return n


def stain_of(ihc_name: str) -> str:
    """Trailing stain letter: 2615223-C -> C, 2616117-13C -> C."""
    m = re.search(r"([CEKP])$", ihc_name)
    if not m:
        raise ValueError(f"cannot parse stain from {ihc_name!r}")
    return m.group(1)


def find_registered(reg_root: Path, moving: str, fixed: str):
    cands = sorted(reg_root.rglob(f"*{moving}_to_{fixed}*registered*.ome.tiff"))
    cands += sorted(reg_root.rglob(f"*{moving}_to_{fixed}*registered*.tiff"))
    # de-dup (second glob overlaps first)
    seen, out = set(), []
    for c in cands:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def stage1(csv_path, reg_root, out_dir, mode, execute):
    reg_root, out_dir = Path(reg_root), Path(out_dir)
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8-sig")))
    print(f"[stage1] {len(rows)} pairs | reg-root={reg_root} | out={out_dir} | mode={mode}")
    if execute:
        out_dir.mkdir(parents=True, exist_ok=True)
    ok = miss = multi = 0
    for r in rows:
        ihc, he = base(r["moving"]), base(r["fixed"])
        dst = out_dir / f"{he} {stain_of(ihc)}_warped.tiff"
        found = find_registered(reg_root, ihc, he)
        if not found:
            print(f"  MISS  {ihc} -> {he}  (no registered file under reg-root)")
            miss += 1
            continue
        if len(found) > 1:
            print(f"  MULTI {ihc} -> {he}  ({len(found)} candidates, use first)")
            multi += 1
        src = found[0]
        print(f"  {'LINK ' if mode == 'symlink' else 'COPY '} {src.name}\n        -> {dst}")
        if execute:
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            if mode == "symlink":
                dst.symlink_to(src.resolve())
            else:
                import shutil
                shutil.copy2(src, dst)
        ok += 1
    print(f"[stage1] done: linked={ok} missing={miss} multi={multi} (dry-run, nothing written)"
          if not execute else f"[stage1] done: linked={ok} missing={miss} multi={multi}")


def stage2(csv_path, dat_dir, execute):
    dat_dir = Path(dat_dir)
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8-sig")))
    hes = sorted({base(r["fixed"]) for r in rows})
    print(f"[stage2] {len(hes)} HE dats in {dat_dir}")
    for m in hes:
        src = dat_dir / f"{m}.dat"
        dst = dat_dir / f"{m} HE.dat"
        if dst.exists():
            print(f"  SKIP  {dst.name} already exists")
        elif src.exists():
            print(f"  RENAME {src.name} -> {dst.name}")
            if execute:
                src.rename(dst)
        else:
            print(f"  MISS  {src.name} not found")


def main():
    ap = argparse.ArgumentParser(description="wsireg -> fuse_he_ihc rename bridge")
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--mode", choices=["stage1", "stage2"], default="stage1")
    ap.add_argument("--reg-root", default="wsireg_output", help="wsireg batch output root")
    ap.add_argument("--out-dir", default="warped_tiff", help="where to place *_warped.tiff")
    ap.add_argument("--link", choices=["symlink", "copy"], default="symlink")
    ap.add_argument("--dat-dir", default="dat", help="cellpose dat dir (stage2)")
    ap.add_argument("--execute", action="store_true", help="actually write; default dry-run")
    a = ap.parse_args()
    if a.mode == "stage1":
        stage1(a.csv, a.reg_root, a.out_dir, a.link, a.execute)
    else:
        stage2(a.csv, a.dat_dir, a.execute)


if __name__ == "__main__":
    main()
