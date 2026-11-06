
  python fuse_he_ihc.py \
  --dat_dir /media/linjiatai/086399513677BED7/zhang/dataset/20260924_gd_he_ihc/fuse_test_18_2617860/dat_all \
  --out_dir /media/linjiatai/086399513677BED7/zhang/dataset/20260924_gd_he_ihc/fuse2/fused_direct \
  --ihc_wsi_dir /media/linjiatai/086399513677BED7/zhang/dataset/20260924_gd_he_ihc/reg_ihc_test_18_2617860 \
  --direct_overlay --no_he_viz


python run_fuse_pos.py \
  --he_img  /media/linjiatai/086399513677BED7/zhang/dataset/20260924_gd_he_ihc/reg_he_test_18_2617860 \
  --ihc_img /media/linjiatai/086399513677BED7/zhang/dataset/20260924_gd_he_ihc/reg_ihc_test_18_2617860 \
  --he_dat  /media/linjiatai/086399513677BED7/zhang/dataset/20260924_gd_he_ihc/core_out/fuse_contours/he \
  --ihc_dat /media/linjiatai/086399513677BED7/zhang/dataset/20260924_gd_he_ihc/core_out/fuse_contours/ihc \
  --out     /media/linjiatai/086399513677BED7/zhang/dataset/20260924_gd_he_ihc/core_out/fuse_contours_out \
  --dab_thresh 0.1 --keep_fuse_viz

python run_wsi_cellpose.py --model cpsam --gpu 0   --input_dir /media/linjiatai/086399513677BED7/zhang/dataset/20260924_gd_he_ihc/reg_ihc_3   --output_dir /media/linjiatai/086399513677BED7/zhang/dataset/20260924_gd_he_ihc/cellpose_ihc_3   --tile_shape 4096 --ambiguous_size 64 --wsi_proc_mag 0.5   --cellprob_threshold -1 --flow_threshold 1.0   --assume_mpp 0.26264

# Cellpose WSI 推理 + HE 辅助 IHC 融合 + 阳性占比

流程：两遍 cellpose 推理（`run_wsi_cellpose.py` 写 `.dat`）→ 后处理融合（`fuse_he_ihc.py`）→ 阳性占比（`ihc_positivity.py`）。后两步纯 CPU，不占 GPU。

前提：HE 与 IHC 配准图已配准；`proc_dimensions` 不一致时按比例缩放对齐。

## 0. dat 命名与配对规则

`--dat_dir` 里同时放两类文件（批量要求同一个目录，先拷到一起）：

- 旧命名：`... HE.dat`（带空格）算 HE，如 `2519980 HE.dat`；`..._warped.dat` 算 IHC，按 caseID 前缀配对；目录里只有 1 个 HE 时所有 `_warped.dat` 默认配给它。
- 新命名（RegProj）：HE `2615223-1.dat`，IHC `RegProj_2615223-C_to_2615223-1-moving_to_fixed_registered.ome.dat`，按 fixed 编号配对。编号与抗体对应：1↔C、2↔E、3↔K、4↔P。

IHC 底图按 `dat` 同名找（`...registered.ome.dat` ↔ `...registered.ome.tiff`）；HE 图按 `<HE名>.svs` 找。

## 1. WSI 推理（run_wsi_cellpose.py）

HE 正常跑（轮廓做基准）：

```bash
python run_wsi_cellpose.py --model cpsam --gpu 0 \
  --input_dir <HE_wsi_dir> --output_dir <he_out> \
  --tile_shape 4096 --ambiguous_size 64 --wsi_proc_mag 0.5 --diameter 20
```
python run_wsi_cellpose.py --model cpsam --gpu 0 \
  --input_dir /media/linjiatai/086399513677BED7/zhang/dataset/20260924_gd_he_ihc/reg_he_test_18_2617860 \
  --output_dir /media/linjiatai/086399513677BED7/zhang/dataset/20260924_gd_he_ihc/cellpose_he_test_18_2617860 \
  --tile_shape 4096 --ambiguous_size 64 --wsi_proc_mag 0.5 --diameter 20 

IHC 宽松跑（宁可多检，融合阶段再过滤）：

```bash





python run_wsi_cellpose.py --model cpsam --gpu 0 \
  --input_dir /media/linjiatai/086399513677BED7/zhang/dataset/20260924_gd_he_ihc/test4/ihc_reg \
  --output_dir /media/linjiatai/086399513677BED7/zhang/dataset/20260924_gd_he_ihc/test4/ihc_dat \
  --tile_shape 4096 --ambiguous_size 64 --wsi_proc_mag 0.5 \
  --cellprob_threshold -1 --flow_threshold 1.0 \
  --assume_mpp 0.26264










- `tiff` 输入加 `--wsi_file_ext .tiff`；40 倍图不用改参数（自动降采样到 0.5mpp）。
- 已有 `dat` 重跑会自动 `Skip existing`，只补新增；换参数重跑某张先删其 `dat`。
- 输出：`output_dir/dat/<片名>.dat`（`{Nuclei:{uuid:{box,contour,centroid}}, proc/base 分辨率与尺寸}`）。

## 2. 融合（fuse_he_ihc.py）

两种模式：默认过滤融合（IHC 有 HE 重叠才留，无重叠扔掉；HE 有而 IHC 没检出的补回来，计数用这个）；`--direct_overlay` 跳过一切过滤直接合并（只看图用，**计数禁用**，会重复计数）。

批量：

```bash
python fuse_he_ihc.py \
  --dat_dir <dat_all> --out_dir <fused> \
  --ihc_wsi_dir <IHC_reg_dir> --he_wsi_dir <HE_dir>
```

python viz_three.py \
  --he_dat  /media/linjiatai/086399513677BED7/zhang/dataset/20260924_gd_he_ihc/cellpose_he_test_18_2617860/dat/2617860-2.dat \
  --ihc_dat /media/linjiatai/086399513677BED7/zhang/dataset/20260924_gd_he_ihc/cellpose_ihc_test_18_2617860/dat/2617860-E_warped.dat \
  --he_wsi  /media/linjiatai/086399513677BED7/zhang/dataset/20260924_gd_he_ihc/reg_he_test_18_2617860/2617860-2.svs \
  --ihc_wsi /media/linjiatai/086399513677BED7/zhang/dataset/20260924_gd_he_ihc/reg_ihc_test_18_2617860/2617860-E_warped.tiff \
  --out /media/linjiatai/086399513677BED7/zhang/dataset/20260924_gd_he_ihc/three_viz1 \
  --fill \
  --he_fill_alpha 0.5 \
  --ihc_fill_alpha 0.5

three_viz/viz_he/2617860-2/2617860-2_HE_full.tif
three_viz/viz_he/2617860-2/2617860-2_HE_preview.png

three_viz/viz_ihc/2617860-E_warped/2617860-E_warped_IHC_full.tif
three_viz/viz_ihc/2617860-E_warped/2617860-E_warped_IHC_preview.png

three_viz/viz_he/2617860-2/2617860-E_warped_both_on_HE_full.tif
three_viz/viz_he/2617860-2/2617860-E_warped_both_on_HE_preview.png
第三张就是：
HE 原图为底
红 = HE 自己的全部标注
绿 = IHC 自己的全部标注
不做融合消除，两套全量叠加

单对：

```bash
python fuse_he_ihc.py \
  --he_dat  /media/linjiatai/086399513677BED7/zhang/dataset/20260924_gd_he_ihc/core_out/fuse_registered/dat_all/2617860-2.dat \
  --ihc_dat /media/linjiatai/086399513677BED7/zhang/dataset/20260924_gd_he_ihc/core_out/fuse_registered/dat_all/2617860-E_warped.dat \
  --out_dat /media/linjiatai/086399513677BED7/zhang/dataset/20260924_gd_he_ihc/core_out/direct \
  --ihc_wsi /media/linjiatai/086399513677BED7/zhang/dataset/20260924_gd_he_ihc/reg_he_test_18_2617860/2617860-2.svs \
  --viz_dir /media/linjiatai/086399513677BED7/zhang/dataset/20260924_gd_he_ihc/core_out/direct \
  --direct_overlay
# 直接叠加加 --direct_overlay；只融合不画图加 --no_viz
```



输出：`out_dir/dat/*_HEfused.dat`（`source`：过滤模式 `ihc` / `he_supplement`，叠加模式 `ihc` / `he`）、`out_dir/viz/<IHC名>/` 双色 BigTIFF（绿=IHC，红=HE）+ preview、`out_dir/viz_he/<HE名>/`（`--on_he fused` 时含 `<IHC名>_fused_on_HE_full.tif`）、`out_dir/fuse_stats.csv`。

注意：IHC 坐标是基线像素（真值约 0.26µm/px）时，`--dist_thresh` 默认 25 只≈6.6µm 太严；实测 `--dist_thresh 30`（≈7.8µm）单细胞贴合最好（中位偏移 19px、超 25px 仅 22%），50 则太松（中位 26px、一半以上飘）。

## 3. 可视化（HE 为底）

`full.tif` 是 BigTIFF，用 QuPath / ImageJ 看；先看 `preview.png`。颜色：双色图绿=IHC、红=HE；阳性图橙=阳性、蓝=阴性。

- 双色（IHC 为底）：默认画，`--no_viz` 关闭。
- `--on_he`（需 HE 图）：`ihc` 只画 IHC（绿），`both` 同时画 HE+IHC，`fused` 把融合结果（绿=IHC 保留、红=HE 补回）画到 HE 底上，`all` 全画，`none`（默认）不画。单对输出目录用 `--he_viz_dir`。
- IHC 配准图 `*_warped.tiff` 边缘常有黑边（moving 没盖满 fixed 视野），IHC 底的双色图会如实带出黑边；想看完整体 HE 视野用 `--on_he fused`。

```bash
python fuse_he_ihc.py --dat_dir <dat_all> --out_dir <fused> \
  --he_wsi_dir <HE_dir> --direct_overlay --no_viz --on_he ihc
```

## 4. 阳性占比（ihc_positivity.py）

整体阳性占比 = 阳性核 / 全部检出核（分母含间质/淋巴核，比临床“只数肿瘤”口径偏低，属探索性指标；肿瘤/正常分区等医生圈 ROI 后再加分组）。

算法：融合核表（IHC 保留 + HE 补回）逐核读 DAB——IHC 图分条带直读，RGB→OD，反卷积拆 DAB 通道，contour 内平均 OD 为分数，检出来源不决定阴阳；近黑墨迹伪影核剔除不计；默认固定阈值 DAB OD 0.2（蓝复染串扰上限约 0.2，棕真阳性一般 >0.25，`--dab_thresh` 可调；`triangle`/`otsu` 自动方法备选，阴阳不平衡时慎用）。`pos.dat` 存全部分数，重切阈值秒级完成不读图。HER2（膜染色）不适用这套核算法，跳过。

```

正确做法：先融合，再算阳性 + 画橙蓝图
等 IHC 推理结束、2617860-E_warped.dat 出现后：

HE_DAT=/media/linjiatai/086399513677BED7/zhang/dataset/20260924_gd_he_ihc/cellpose_he_test_18_2617860/dat/2617860-2.dat
IHC_DAT=/media/linjiatai/086399513677BED7/zhang/dataset/20260924_gd_he_ihc/cellpose_ihc_test_18_2617860/dat/2617860-E_warped.dat
IHC_WSI=/media/linjiatai/086399513677BED7/zhang/dataset/20260924_gd_he_ihc/reg_ihc_test_18_2617860/2617860-E_warped.tiff
HE_WSI=/media/linjiatai/086399513677BED7/zhang/dataset/20260924_gd_he_ihc/reg_he_test_18_2617860/2617860-2.svs
OUT=/media/linjiatai/086399513677BED7/zhang/dataset/20260924_gd_he_ihc/fuse_test_18_2617860
1）融合 HE + IHC
python fuse_he_ihc.py \
  --he_dat "$HE_DAT" \
  --ihc_dat "$IHC_DAT" \
  --out_dat "$OUT/2617860-E_warped_HEfused.dat" \
  --no_viz --no_he_viz
2）阳性占比 + 橙蓝 HE 可视化
python ihc_positivity.py \
  --fuse_dat "$OUT/2617860-E_warped_HEfused.dat" \
  --ihc_wsi "$IHC_WSI" \
  --out_pos "$OUT/2617860-E_warped_HEfused_pos.dat" \
  --he_dat "$HE_DAT" \
  --he_wsi "$HE_WSI" \
  --he_viz_dir "$OUT"


输出：`positivity.csv`（总数/可评/伪影/阳性/阴性/占比/阈值）、`dat/*_pos.dat`（每核 `dab_score`/`pos`/`artifact`）、`viz_he/<HE名>/*_pos_on_HE_full.tif` + preview（橙=阳性、蓝=阴性）。

### 一条命令跑融合 + 阳性（run_fuse_pos.py）

四个目录（HE 图 / IHC 配准图 / HE dat / IHC dat）→ 自动合并 dat → 融合 → 阳性占比 + 橙蓝图：

```bash
python run_fuse_pos.py \
  --he_img  "<HE图目录>" \
  --ihc_img "<IHC配准图目录>" \
  --he_dat  "<HE_dat目录>" \
  --ihc_dat "<IHC_dat目录>" \
  --out     "<输出根目录>"
```

输出：`<out>/fused/dat/*_HEfused.dat`、`<out>/pos/positivity.csv`、`<out>/pos/dat/*_pos.dat`、`<out>/pos/viz_he/<HE名>/*_pos_on_HE_preview.png`（橙=阳性，蓝=阴性）。默认跳过融合的绿/红图省时间（`--keep_fuse_viz` 保留）；`--no_pos_viz` 只算占比；`--dab_thresh` / `--thr_method` 原样透传。

## 5. 可调参数

```text
--iou_thresh  默认 0.1，bbox 交并比阈值（--direct_overlay 下不生效）
--dist_thresh 默认 25（IHC proc 像素）；IHC 基线像素时填 30（--direct_overlay 下不生效）
--direct_overlay 直接叠加，不做过滤（只看图，计数禁用）
--he_wsi / --he_wsi_dir  HE 图（单对路径 / 批量目录），给了就画 HE viz
--he_viz_dir  单对 HE viz 输出目录，默认与 --viz_dir 相同
--on_he       ihc / both / fused / all，默认 none
--dab_thresh  默认 0.2（仅 positivity 的 fixed 方法生效）
--thr_method  fixed（默认）/ triangle / otsu
--no_he_viz   关闭 HE viz（on_he 不受影响）
--no_viz      关闭 IHC 底双色图（不管 HE viz）
--viz_mpp     默认 0.5；--viz_strip 默认 2048（内存小调小）
```

## 6. 实测参考

- 小片：HE 57k、IHC 100k → 保留 44k / 扔掉 56k / 补回 22k / 融合 66k，约 20 秒。
- 大片（HE 327k + IHC 41~57 万）：加载两个 `dat` 约 1 分钟，融合约 13 秒，可视化每对约 5 分钟；阳性打分每张约 5 分钟（纯 CPU）。
- 批量同一 `HE.dat` 只加载一次（`HE cache hit`）。

## 7. 环境

`conda activate linjiatai_4090`（以实际 env 名为准）。依赖 tiatoolbox、openslide、cellpose、torch、opencv、shapely、joblib、tqdm（positivity 无需新增依赖）。
