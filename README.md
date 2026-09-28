# 联动三步：Cellpose 推理 + HE 辅助 IHC 融合

都是 cellpose 推理（run_wsi_cellpose.py 写 dat），再用 fuse_he_ihc.py 融合。

1）HE 正常跑（轮廓做基准）：

python run_wsi_cellpose.py --model cpsam --gpu 0 \
  --input_dir /media/linjiatai/086399513677BED7/zhang/dataset/CNB_Registration_Cellpose_v2/HE_test \
  --output_dir /media/linjiatai/086399513677BED7/zhang/dataset/CNB_Registration_Cellpose_v2/HE_output \
  --tile_shape 4096 --ambiguous_size 64 --wsi_proc_mag 0.5 --diameter 20

2）IHC 宽松跑一遍（宁可多检，以 ER 为例）：

python run_wsi_cellpose.py --model cpsam --gpu 0 \
  --input_dir /media/linjiatai/086399513677BED7/zhang/dataset/CNB_Registration_Cellpose_v2/IHC_test \
  --output_dir /media/linjiatai/086399513677BED7/zhang/dataset/CNB_Registration_Cellpose_v2/IHC_output \
  --tile_shape 4096 --ambiguous_size 64 --wsi_proc_mag 0.5 \
  --cellprob_threshold -1 --flow_threshold 1.0

3）融合（两遍 dat 到手后，单对 / 批量二选一）：

单对（不用拷 dat）：

python fuse_he_ihc.py \
  --he_dat ".../he_out/dat/2519980 HE.dat" \
  --ihc_dat ".../ihc_loose_out/dat/2519980 ER_warped.dat" \
  --out_dat ".../HEfused/dat/2519980 ER_warped_HEfused.dat" \
  --ihc_wsi ".../CNB_Registration/2519980 ER_warped.tiff" \
  --viz_dir ".../HEfused/viz/2519980 ER_warped"

批量（先把两遍的 dat 拷到同一个 dat_all/，再跑）：

python fuse_he_ihc.py \
--dat_dir /media/linjiatai/086399513677BED7/zhang/dataset/CNB_Registration_Cellpose_v2/IHC_output/dat \
--out_dir /media/linjiatai/086399513677BED7/zhang/dataset/CNB_Registration_Cellpose_v2/fuse \
--ihc_wsi_dir /media/linjiatai/086399513677BED7/zhang/dataset/CNB_Registration_Cellpose_v2/IHC

只融合不画图加 --no_viz（只关双色图）。HE viz 默认不画，给了 --he_wsi_dir 才画，详见下文 HE viz 一节。

直接叠加（不用融合过滤，HE 全量 + IHC 全量合并）时，单对 / 批量都只需加 --direct_overlay：

python fuse_he_ihc.py \
--dat_dir  /media/linjiatai/086399513677BED7/zhang/dataset/CNB_Registration_Cellpose/dat\
--out_dir /media/linjiatai/086399513677BED7/zhang/dataset/CNB_Registration_Cellpose_v2/fuse \
--ihc_wsi_dir /media/linjiatai/086399513677BED7/zhang/dataset/CNB_Registration_Cellpose_v2/IHC \
--direct_overlay --no_viz

---

# HE 辅助 IHC 融合（fuse_he_ihc.py）

IHC 染色淡，Cellpose 自己分不好。用 HE 的准轮廓来带 IHC，后处理融合，不改模型、不重训。

做法（默认过滤模式）：
1. HE 正常跑，轮廓做基准。
2. IHC 用宽松参数跑一遍（cellprob_threshold 调低如 -1，flow_threshold 调大），宁可多检。
3. 有 HE 重叠的 IHC 核留下，无重叠的当假阳性扔掉；HE 有但 IHC 没检出的，把 HE 轮廓直接补回来。
4. 融合结果画到 IHC 图上，双色叠加看效果。

两种模式：默认是上面这套 IoU / 质心距离过滤融合；加 `--direct_overlay` 则跳过一切过滤，
HE 全量 + IHC 全量直接合并（HE 仍按 `proc_dimensions` 比例缩放到 IHC 坐标系），`fused_n = he_n + ihc_n`。

## 单对用法

python fuse_he_ihc.py \
  --he_dat  "dat/2519980 HE.dat" \
  --ihc_dat "dat/2519980 ER_warped.dat" \
  --out_dat "fused/2519980 ER_warped_HEfused.dat" \
  --ihc_wsi "CNB_Registration/2519980 ER_warped.tiff" \
  --viz_dir "fused/viz/2519980 ER_warped"

只融合不画图加 --no_viz（只关双色图）。HE viz 见下文 HE viz 一节，默认不画。

直接叠加时单对命令同理加 --direct_overlay：

python fuse_he_ihc.py \
  --he_dat  "dat/2519980 HE.dat" \
  --ihc_dat "dat/2519980 ER_warped.dat" \
  --out_dat "overlay/2519980 ER_warped_HEfused.dat" \
  --ihc_wsi "CNB_Registration/2519980 ER_warped.tiff" \
  --viz_dir "overlay/viz/2519980 ER_warped" \
  --direct_overlay

## 批量用法

dat 目录里同时放 " HE.dat" 和 " *_warped.dat"，按前缀自动配对：

python fuse_he_ihc.py \
  --dat_dir     /media/linjiatai/086399513677BED7/zhang/dataset/CNB_Registration_Cellpose/dat \
  --out_dir     /media/linjiatai/086399513677BED7/zhang/dataset/CNB_Registration_Cellpose_HEfused \
  --ihc_wsi_dir /media/linjiatai/086399513677BED7/zhang/dataset/CNB_Registration

直接叠加时批量命令加 --direct_overlay（建议 out_dir 换一个目录，如 ..._overlay，避免和过滤模式结果混在一起）：

python fuse_he_ihc.py \
  --dat_dir     /media/linjiatai/086399513677BED7/zhang/dataset/CNB_Registration_Cellpose/dat \
  --out_dir     /media/linjiatai/086399513677BED7/zhang/dataset/CNB_Registration_Cellpose_overlay \
  --ihc_wsi_dir /media/linjiatai/086399513677BED7/zhang/dataset/CNB_Registration \
  --direct_overlay --no_viz

输出：
  out_dir/dat/*_HEfused.dat   融合 dat（格式同 cellpose .dat，可直接进 qupath 导出）
  out_dir/viz/<IHC名>/*_HEfused_full.tif  整张双色 BigTIFF：绿=IHC，红=HE（过滤模式是 HE 补回，叠加模式是 HE 全量）
  out_dir/viz/<IHC名>/*_HEfused_preview.png 整张缩小预览图
  out_dir/viz_he/<HE名>/*_HE_full.tif  HE 图 + HE 标注（给了 --he_wsi_dir 才画，下同）
  out_dir/viz_he/<HE名>/*_HE_preview.png 同上预览图
  out_dir/fuse_stats.csv      每对的 HE 数 / IHC 数 / 保留 / 扔掉 / 补回 / 融合数 / mode 列（filter 或 direct_overlay）

## 直接叠加模式（--direct_overlay）

- 行为：跳过 `--iou_thresh` / `--dist_thresh` 过滤，IHC 全部保留、HE 全部合并；HE 坐标仍按
  `proc_dimensions` 比例缩放到 IHC 坐标系（两边一致时无缩放）。
- dat 内 `source` 标记：IHC 为 `ihc`，HE 为 `he`（过滤模式下 HE 补回记为 `he_supplement`）；
  viz 双色画图不变：绿=IHC，红=HE。
- stats：`kept_ihc = ihc_n`、`dropped_ihc = 0`、`he_supplement = he_n`、`fused_n = he_n + ihc_n`，
  `mode = direct_overlay`。

## HE viz（以 HE 图为底图画 HE 标注）

默认只画 IHC 底图的双色图。给了 HE 原图就加画一张 HE 底图 + HE 全量标注（红色，与双色图里的 HE 同色）；
不给就不画。用 `--no_he_viz` 可强制关闭。

单对（`--he_viz_dir` 不给则和双色图放同一目录，文件名不同不会覆盖）：

python fuse_he_ihc.py \
  --he_dat  "dat/2519980 HE.dat" \
  --ihc_dat "dat/2519980 ER_warped.dat" \
  --out_dat "fused/2519980 ER_warped_HEfused.dat" \
  --ihc_wsi "CNB_Registration/2519980 ER_warped.tiff" \
  --viz_dir "fused/viz/2519980 ER_warped" \
  --he_wsi  "HE/2519980 HE.svs" \
  --he_viz_dir "fused/viz_he/2519980 HE"

批量（每个 HE 只画一次，输出到 `out_dir/viz_he/<HE名>/`，HE 图按 `<HE名>.svs` / 去掉 ` HE` 后的名自动找）：

python fuse_he_ihc.py \
  --dat_dir     /media/linjiatai/086399513677BED7/zhang/dataset/CNB_Registration_Cellpose/dat \
  --out_dir     /media/linjiatai/086399513677BED7/zhang/dataset/CNB_Registration_Cellpose_HEfused \
  --ihc_wsi_dir /media/linjiatai/086399513677BED7/zhang/dataset/CNB_Registration \
  --he_wsi_dir  /media/linjiatai/086399513677BED7/zhang/dataset/HE \
  --direct_overlay --no_viz

## IHC 宽松参数那遍怎么跑

正常跑 run_wsi_cellpose.py，只改两个阈值（以 ER 为例）：

python run_wsi_cellpose.py --model cpsam --gpu 0 \
  --input_dir .../CNB_Registration --output_dir .../dat_ihc_loose \
  --cellprob_threshold -1 --flow_threshold 1.0

HE 那遍用默认参数正常跑。两遍 dat 到手后用本脚本融合。

## 可调参数

  --iou_thresh  默认 0.1，bbox 交并比阈值（--direct_overlay 下不生效）
  --dist_thresh 默认 25（proc 像素，0.5mpp 下约 12.5um），质心配准容差；
                warped 配准残差大就调大，配准准就调小（--direct_overlay 下不生效）
  --direct_overlay  直接叠加：跳过 IoU/距离过滤，HE 全量 + IHC 全量合并（viz 仍是绿=IHC、红=HE）
  --he_wsi / --he_wsi_dir  HE 原图（单对 .svs 路径 / 批量目录），给了就画 HE viz，不给不画
  --he_viz_dir    单对模式 HE viz 输出目录，默认与 --viz_dir 相同
  --no_he_viz     不画 HE viz（即使给了 HE 图）
  --no_viz        不画双色 viz（只控制 IHC 底那张，不管 HE viz）
  --viz_mpp     双色整图分辨率，默认 0.5
  --viz_strip   整图分条带读取高度，默认 2048（机器内存小就调小）

## 实测参考

2519980 ER：HE 57k、IHC 100k → 保留 44k、扔掉 56k、HE 补回 22k、融合 66k，单对约 20 秒。

大数据注意：HE 33 万 + IHC 41 万这种规模，光加载两个 dat 就要约 1 分钟，单对几分钟正常；
程序分阶段打 `[fuse]` 日志，有日志在走就是没卡。批量时同一 `HE.dat` 只加载一次（`HE cache hit`），
4 对里第 2～4 对会快约半分钟。

---

# WSI 推理（run_wsi_cellpose.py）

Cellpose 整片推理入口，输出 .dat（HE / IHC 两遍都是它跑的）。

## 四种用法

1）只推理（最快，不画图）：

python run_wsi_cellpose.py --model cpsam --gpu 0 \
  --input_dir .../CNB_Registration --output_dir .../dat \
  --tile_shape 4096 --ambiguous_size 64 --wsi_proc_mag 0.5 --diameter 20

2）推理 + 高清分块图（原图分辨率切小图，放大看边界）：

python run_wsi_cellpose.py ... --save_viz_highres --viz_highres_tile 2048 --viz_highres_max_tiles 16

3）更细的高清图：

python run_wsi_cellpose.py ... --save_viz_highres --viz_highres_tile 1024 --viz_highres_mpp 0.20 --viz_highres_max_tiles 36

4）已有 dat 补画高清图（不重推理）：

python run_wsi_cellpose.py --input_dir ... --output_dir ... \
  --model cpsam --gpu 0 --save_viz_highres --viz_highres_tile 2048 --viz_highres_max_tiles 4
已有 dat 会跳过推理，只补画 viz_highres。

## 常用参数

  --model               cpsam / cyto / cyto2 / cyto3 / 本地路径，默认 cpsam
  --gpu                 GPU 号，默认 0；空字符串用 CPU
  --input_dir           WSI 目录（必填）
  --output_dir          输出目录（必填）
  --msk_dir             组织掩膜目录（同名 png，可选）；给了就只跑有掩膜的片子
  --wsi_file_ext        默认 .svs；找不到会自动试 .tif/.tiff/.ndpi/.mrxs/.scn
  --tile_shape          后处理瓦片边长，默认 4096；显存不够改 2048
  --ambiguous_size      瓦片重叠去重边界，默认 64
  --wsi_proc_mag        处理分辨率 mpp，默认 0.5（20x）；0.25 是 40x
  --diameter            细胞直径，默认 None 自动估计；HE 在 0.5mpp 建议 15-20，太大易粘连
  --flow_threshold      默认 0.4，越大检出越多
  --cellprob_threshold  默认 0.0，越小检出越多（IHC 宽松那遍用 -1）
  --min_size            最小面积，默认 15
  --save_thumb          缩略图到 thumb/
  --save_mask           掩膜到 mask/
  --save_viz_highres    高清分块到 viz_highres/<片名>/
  --viz_highres_tile      默认 2048（1024 更细，4096 视野大）
  --viz_highres_mpp       默认原图 base mpp；0.20 更清晰，0.5 与处理一致
  --viz_highres_max_tiles 默认 16，均匀采样；-1 全出（大片子上千张，慎用）

## 输出结构

  output_dir/dat/<片名>.dat            joblib：{Nuclei:{uuid:{box,contour,centroid}}, proc/base 分辨率与尺寸}
  output_dir/thumb/<片名>.png          缩略图（--save_thumb）
  output_dir/mask/<片名>.png           掩膜（--save_mask）
  output_dir/viz_highres/<片名>/*.png  高清小图（--save_viz_highres）
  output_dir/logs/wsi_cellpose_*.log   日志

## 环境

conda activate linjiatai_4090。依赖 tiatoolbox、openslide、cellpose、torch、opencv、shapely、joblib、tqdm。
