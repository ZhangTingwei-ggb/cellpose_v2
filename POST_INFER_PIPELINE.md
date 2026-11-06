# Cellpose WSI 推理之后的全部命令说明（融合 / 可视化 / 阳性率）

适用范围：`fuse_he_ihc.py`（融合 + 各种可视化）、`ihc_positivity.py`（DAB 阳性率 + 橙蓝图）、
`run_fuse_pos.py`（一条命令串起上面两步）。

前置条件：已经跑完 Cellpose 推理，产出 `.dat`：

- HE dat：`<cellpose_he_out>/dat/<HE名>.dat`
- IHC dat：`<cellpose_ihc_out>/dat/<IHC名>_warped.dat`

---

## 0. 三个脚本一览

| 脚本 | 作用 | 模式 |
|---|---|---|
| `run_fuse_pos.py` | 合并 dat 目录 → 融合 → 阳性率 + HE 底橙蓝图 | 只有批量（目录级） |
| `fuse_he_ihc.py` | HE/IHC 核融合，输出 `*_HEfused.dat` + 多种可视化 | 单对（文件）/ 批量（目录） |
| `ihc_positivity.py` | 对融合核逐个打 DAB 分，统计阳性率，HE 底橙蓝图 | 单对（文件）/ 批量（目录） |

颜色约定：

| 颜色 | BGR | 含义 |
|---|---|---|
| 橙 | `(0, 165, 255)` | 阳性核 |
| 蓝 | `(255, 0, 0)` | 阴性核 |
| 绿 | `(0, 255, 0)` | IHC 保留的核 |
| 红 | `(0, 0, 255)` | HE 补回的核 / HE 自己的核 |

---

## 1. `run_fuse_pos.py`（推荐：一条命令）

### 1.1 用法

```bash
python run_fuse_pos.py \
  --he_img  <HE原图目录> \
  --ihc_img <IHC配准图目录> \
  --he_dat  <HE_dat目录> \
  --ihc_dat <IHC_dat目录> \
  --out     <输出根目录>
```

### 1.2 参数

| 参数 | 必填 | 默认 | 说明 |
|---|---|---|---|
| `--he_img` | ✔ | - | HE 原图目录（`.svs`/`.tiff`），用于画 HE 底图 |
| `--ihc_img` | ✔ | - | IHC 配准图目录（`*_warped.tiff`），DAB 打分和 IHC 底图用 |
| `--he_dat` | ✔ | - | HE dat 目录（里面是 `*.dat`） |
| `--ihc_dat` | ✔ | - | IHC dat 目录（里面是 `*_warped.dat`） |
| `--out` | ✔ | - | 输出根目录 |
| `--dab_thresh` | | `0.2` | DAB OD 固定阈值（仅 `--thr_method fixed` 生效） |
| `--thr_method` | | `fixed` | `fixed` / `triangle` / `otsu` |
| `--viz_mpp` | | `0.5` | 可视化分辨率（µm/px） |
| `--viz_strip` | | `2048` | 整图分条带读取高度（viz 像素） |
| `--no_pos_viz` | | 关 | 只算占比，不画橙蓝图 |
| `--keep_fuse_viz` | | 关 | 额外保留 IHC 底绿/红双色融合图（慢，默认关） |
| `--dry_run` | | 关 | 只打印将要执行的命令，不真跑 |

### 1.3 输出

```text
<out>/dat_all/                              # HE/IHC dat 软链接合集（融合的输入）
<out>/fused/dat/<IHC名>_HEfused.dat        # 融合核表
<out>/fused/fuse_stats.csv                  # 融合统计
<out>/pos/dat/<IHC名>_HEfused_pos.dat       # 带 dab_score / pos / artifact
<out>/pos/positivity.csv                    # 阳性率汇总
<out>/pos/viz_he/<HE名>/<IHC名>_pos_on_HE_full.tif    # HE 底橙蓝全图
<out>/pos/viz_he/<HE名>/<IHC名>_pos_on_HE_preview.png # 预览
```

### 1.4 实例

```bash
python run_fuse_pos.py \
  --he_img  /media/linjiatai/086399513677BED7/zhang/dataset/20260924_gd_he_ihc/reg_he_test_18_2617860 \
  --ihc_img /media/linjiatai/086399513677BED7/zhang/dataset/20260924_gd_he_ihc/reg_ihc_test_18_2617860 \
  --he_dat  /media/linjiatai/086399513677BED7/zhang/dataset/20260924_gd_he_ihc/cellpose_he_test_18_2617860/dat \
  --ihc_dat /media/linjiatai/086399513677BED7/zhang/dataset/20260924_gd_he_ihc/cellpose_ihc_test_18_2617860/dat \
  --out     /media/linjiatai/086399513677BED7/zhang/dataset/20260924_gd_he_ihc/fuse_test_18_2617860
```

---

## 2. `fuse_he_ihc.py`（融合 + 可视化）

### 2.1 单对模式（文件级）

```bash
python fuse_he_ihc.py \
  --he_dat  <HE.dat> \
  --ihc_dat <IHC_warped.dat> \
  --out_dat <out>_HEfused.dat \
  --ihc_wsi <IHC_warped.tiff> \
  --viz_dir <viz目录> \
  --he_wsi  <HE.svs> \
  --he_viz_dir <HE viz目录> \
  --on_he both
```

必填：`--he_dat`、`--ihc_dat`、`--out_dat`。

### 2.2 批量模式（目录级）

```bash
python fuse_he_ihc.py \
  --dat_dir <含HE.dat和*_warped.dat的目录> \
  --out_dir <输出根目录> \
  --ihc_wsi_dir <IHC配准图目录> \
  --he_wsi_dir  <HE原图目录> \
  --on_he both
```

必填：`--dat_dir`、`--out_dir`。

### 2.3 全部参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--he_dat` | - | 单对：HE dat 文件路径 |
| `--ihc_dat` | - | 单对：IHC dat 文件路径 |
| `--out_dat` | - | 单对：融合 dat 输出文件路径 |
| `--ihc_wsi` | - | 单对：IHC 配准图，作 IHC 底双色图底图 |
| `--viz_dir` | - | 单对：IHC 底融合双色图输出目录 |
| `--dat_dir` | - | 批量：同时含 HE.dat 和 `*_warped.dat` 的目录 |
| `--out_dir` | - | 批量：输出根目录（建 `dat/ viz/ viz_he/ fuse_stats.csv`） |
| `--ihc_wsi_dir` | - | 批量：IHC 配准图目录 |
| `--he_wsi_dir` | - | 批量：HE 原图目录（`.svs`），给了才画 HE 系可视化 |
| `--he_wsi` | - | 单对：HE 原图（`.svs`） |
| `--he_viz_dir` | 同 `--viz_dir` | HE 系可视化输出目录 |
| `--iou_thresh` | `0.1` | IHC 核与 HE 核的 bbox IoU 阈值 |
| `--dist_thresh` | `25.0` | 质心配准容差（proc 坐标 px；0.5 mpp 下 25 px ≈ 12.5 µm） |
| `--direct_overlay` | 关 | 跳过 IoU/距离过滤，HE 全量 + IHC 全量直接合并 |
| `--no_viz` | 关 | 不画 IHC 底双色融合图 |
| `--no_he_viz` | 关 | 不画 HE 底 HE 标注图（`--on_he` 不受影响） |
| `--on_he` | `none` | 以 HE 为底的额外可视化，见下表 |
| `--on_ihc` | `none` | 以 IHC 配准图为底额外画 IHC 标注；`ihc`=只画 IHC |
| `--ihc_viz_dir` | 同 `--viz_dir` | 单对模式 IHC 底 IHC 标注输出目录 |
| `--fill` | 关 | 标注多边形填充（默认只画轮廓） |
| `--he_fill_alpha` | `0.5` | HE 标注填充 alpha（0-1，0.5=50% 不透明） |
| `--ihc_fill_alpha` | `0.8` | IHC 标注填充 alpha（0-1，0.8=80% 不透明） |
| `--viz_mpp` | `0.5` | 可视化分辨率（µm/px） |
| `--viz_strip` | `2048` | 整图分条带读取高度 |

### 2.4 `--on_he` 取值（都以 HE 原图为底）

| 值 | 输出 | 颜色 |
|---|---|---|
| `none` | 不画 | - |
| `ihc` | `<IHC名>_on_HE_full.tif` | 绿 = IHC 核 |
| `both` | `<IHC名>_both_on_HE_full.tif` | **红 = HE 核，绿 = IHC 核**（你要的“HE 底 + HE 标注 + IHC 标注”） |
| `fused` | `<IHC名>_fused_on_HE_full.tif` | 绿 = IHC 保留，红 = HE 补回 |
| `all` | 上面三种全画 | - |

需要 `--he_wsi`（单对）或 `--he_wsi_dir`（批量）+ `--he_viz_dir`/`--viz_dir`。

### 2.5 输出

```text
# 批量
<out_dir>/dat/<IHC名>_HEfused.dat
<out_dir>/viz/<IHC名>/<IHC名>_HEfused_full.tif + _preview.png   # IHC 底：绿=IHC, 红=HE补回
<out_dir>/viz_ihc/<IHC名>/<IHC名>_IHC_full.tif + _preview.png     # IHC 底：IHC 自己的核（绿），--on_ihc ihc
<out_dir>/viz_he/<HE名>/<HE名>_HE_full.tif + _preview.png          # HE 底：HE 自己的核（红）
<out_dir>/viz_he/<HE名>/<IHC名>_on_HE_full.tif + _preview.png       # --on_he ihc
<out_dir>/viz_he/<HE名>/<IHC名>_both_on_HE_full.tif + _preview.png  # --on_he both
<out_dir>/viz_he/<HE名>/<IHC名>_fused_on_HE_full.tif + _preview.png # --on_he fused
<out_dir>/fuse_stats.csv
```

### 2.6 三张图一条命令（HE 底 HE、IHC 底 IHC、HE 底 HE+IHC）

```bash
python fuse_he_ihc.py \
  --dat_dir <dat_all目录> \
  --out_dir <输出根> \
  --ihc_wsi_dir <IHC配准图目录> \
  --he_wsi_dir  <HE原图目录> \
  --on_he both \
  --on_ihc ihc \
  --fill \
  --he_fill_alpha 0.5 \
  --ihc_fill_alpha 0.8 \
  --no_viz
```

- `--on_he both` 同时触发默认的 `viz_he`（HE 底 + HE 标注）
- `--on_ihc ihc` 生成 IHC 底 + IHC 标注
- `--on_he both` 生成 HE 底 + HE(红) + IHC(绿)
- `--no_viz` 跳过 IHC 底的融合双色图（不需要可省时间）
- `--fill` 打开填充；alpha 是“不透明度”，0.5=50% 不透明，0.8=80% 不透明

### 2.7 dat 配对规则（`--dat_dir`）

- **HE dat**：任何不是 `*_warped.dat` / `*_HEfused.dat` / `*_pos.dat` / `RegProj_*` / `*registered*` 的 `.dat`
  （旧命名 `xxx HE.dat`、新命名 `2617860-2.dat` 都支持）。
- **IHC dat**：`*_warped.dat` 或 `RegProj_*_registered.ome.dat`。
- 目录里**只有一个 HE** 时，所有 IHC 自动配到它。
- 配不上会打印 `no-HE skipped [...]`，`run_fuse_pos.py` 现在会在融合产物为 0 时直接报错。

### 2.8 融合规则

- IHC 核保留条件（任一满足）：与某 HE 核 `bbox_IoU >= iou_thresh`；IHC 质心落在 HE bbox 内；
  IHC 质心与最近 HE 质心距离 `<= dist_thresh`。
- HE 核补充条件：没有任何被保留的 IHC 核与它满足上述条件。
- HE 与 IHC 的 `proc_dimensions` 不一致时，按比例把 HE 缩放到 IHC 坐标系再匹配。

---

## 3. `ihc_positivity.py`（DAB 阳性率 + 橙蓝图）

### 3.1 单对模式

```bash
python ihc_positivity.py \
  --fuse_dat  <融合dat 或 原始IHC dat> \
  --ihc_wsi   <IHC_warped.tiff> \
  --out_pos   <输出_pos.dat> \
  --he_dat    <HE.dat>          # 仅用于 HE 底图
  --he_wsi    <HE.svs>          # 仅用于 HE 底图
  --he_viz_dir <HE viz目录>
```

必填：`--fuse_dat`、`--ihc_wsi`、`--out_pos`。

### 3.2 批量模式

```bash
python ihc_positivity.py \
  --dat_dir     <含HE.dat和*_warped.dat的目录> \
  --fuse_dir    <含*_HEfused.dat的目录> \
  --out_dir     <输出根目录> \
  --ihc_wsi_dir <IHC配准图目录> \
  --he_wsi_dir  <HE原图目录>     # 给了才画 HE 底橙蓝图
```

必填：`--dat_dir`、`--fuse_dir`、`--out_dir`、`--ihc_wsi_dir`。

### 3.3 全部参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--fuse_dat` | - | 单对：融合 dat（`*_HEfused.dat`）或原始 IHC dat |
| `--ihc_wsi` | - | 单对：IHC 配准图（DAB 打分读这张） |
| `--out_pos` | - | 单对：输出 pos dat 文件路径 |
| `--he_dat` | - | 单对：HE dat，仅用于 HE 底图绘制 |
| `--he_wsi` | - | 单对：HE 原图，仅用于 HE 底图绘制 |
| `--he_viz_dir` | - | 单对：HE 底图输出目录 |
| `--dat_dir` | - | 批量：原始 HE.dat + `*_warped.dat` 目录（用于配对） |
| `--fuse_dir` | - | 批量：`*_HEfused.dat` 目录 |
| `--out_dir` | - | 批量：输出根目录（建 `dat/ viz_he/ positivity.csv`） |
| `--ihc_wsi_dir` | - | 批量：IHC 配准图目录 |
| `--he_wsi_dir` | - | 批量：HE 原图目录，给了才画 HE 底橙蓝图 |
| `--dab_thresh` | `0.2` | 固定阈值（仅 `--thr_method fixed` 生效） |
| `--thr_method` | `fixed` | `fixed` / `triangle` / `otsu`（阴阳极不平衡时慎用自动阈值） |
| `--strip_rows` | `512` | DAB 打分时基准层条带高度 |
| `--viz_mpp` | `0.5` | 可视化分辨率 |
| `--viz_strip` | `2048` | 整图分条带读取高度 |

### 3.4 输出

```text
# 单对
<out_pos>                                             # 带 dab_score / pos / artifact
<he_viz_dir>/<IHC名>_pos_on_HE_full.tif + _preview.png # 橙=阳性 蓝=阴性

# 批量
<out_dir>/dat/<IHC名>_HEfused_pos.dat
<out_dir>/positivity.csv
<out_dir>/viz_he/<HE名>/<IHC名>_pos_on_HE_full.tif + _preview.png
```

### 3.5 打分逻辑

- 用融合核轮廓（proc 坐标）按 `f = proc_mpp / base_mpp` 放大回基准层像素，在 **IHC 原图全分辨率** 上取核内像素。
- Ruifrok H&E-DAB 颜色反卷积求 DAB 光学密度，核内取平均 OD 作为 `dab_score`。
- 核内近黑像素（墨迹/折叠）超过 50% 记为 `artifact`，**不参与阴阳统计**（也不画图）。
- `dab_score >= dab_thresh` 判阳性。

### 3.6 阈值敏感性示例（2617860-E）

| `--dab_thresh` | 阳性率 |
|---|---|
| 0.20 | 0.55% |
| 0.15 | 0.76% |
| 0.10 | 1.11% |
| 0.05 | 13.77% |

### 3.7 只改阈值、不重跑（几秒~几分钟）

如果 `_pos.dat` 里已经有 `dab_score`，再喂给脚本会直接复用分数、只重切阴阳：

```bash
python ihc_positivity.py \
  --fuse_dat <上一步的 *_pos.dat> \
  --ihc_wsi  <IHC_warped.tiff> \
  --out_pos  <新的_pos.dat> \
  --dab_thresh 0.1 \
  --he_dat <HE.dat> --he_wsi <HE.svs> --he_viz_dir <viz目录>
```

日志出现 `[pos] reuse stored dab_score, skip image reading` 即表示没重读图。

---

## 4. 三张图专用：`viz_three.py`（不做融合，最快）

只读 dat + WSI，不写融合 dat，适合反复调透明度。

### 4.1 单对

```bash
python viz_three.py \
  --he_dat  <HE.dat> \
  --ihc_dat <IHC_warped.dat> \
  --he_wsi  <HE.svs> \
  --ihc_wsi <IHC_warped.tiff> \
  --out     <输出根> \
  --fill --he_fill_alpha 0.5 --ihc_fill_alpha 0.8
```

### 4.2 批量

```bash
python viz_three.py \
  --dat_dir <含HE.dat和*_warped.dat的目录> \
  --he_wsi_dir  <HE图目录> \
  --ihc_wsi_dir <IHC图目录> \
  --out <输出根> \
  --fill --he_fill_alpha 0.5 --ihc_fill_alpha 0.8
```

### 4.3 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--he_dat` / `--ihc_dat` | - | 单对 dat（文件） |
| `--he_wsi` / `--ihc_wsi` | - | 单对 WSI（文件） |
| `--dat_dir` / `--he_wsi_dir` / `--ihc_wsi_dir` | - | 批量 |
| `--out` | 必填 | 输出根目录 |
| `--fill` / `--no_fill` | 填充开 | 填充或只画轮廓 |
| `--he_fill_alpha` | `0.5` | HE 填充 alpha（不透明度） |
| `--ihc_fill_alpha` | `0.8` | IHC 填充 alpha（不透明度） |
| `--viz_mpp` | `0.5` | 可视化分辨率 |
| `--viz_strip` | `2048` | 分条带读取高度 |

### 4.4 输出

```text
<out>/viz_he/<HE名>/<HE名>_HE_full.tif + _preview.png
<out>/viz_ihc/<IHC名>/<IHC名>_IHC_full.tif + _preview.png
<out>/viz_he/<HE名>/<IHC名>_both_on_HE_full.tif + _preview.png
```

> alpha 是“不透明度”：0.5 = 50% 不透明（半透明），0.8 = 80% 不透明。
> 如果你说的“80% 透明度”是“80% 透明”，应填 `--ihc_fill_alpha 0.2`。

---

## 5. 常见坑

1. **`--fuse_dat` / `--ihc_wsi` / `--out_pos` / `--he_dat` / `--he_wsi` 都是“文件”，只有 `--he_viz_dir` / `--viz_dir` / `--out_dir` 是“目录”。**
2. **`--fuse_dat` 传原始 IHC dat 时，分母只有 IHC 核**；要包含 HE 补回的核，必须先跑融合用 `*_HEfused.dat`。
3. **dat 配对靠文件名**：`2617860-2.dat` + `2617860-E_warped.dat` 这种命名现在支持了；若 `paired 0` 说明命名没匹上。
4. **HE 图用 `.svs` 的 mpp 是 OpenSlide 读的**；IHC 配准 `.tiff` 若 `ResolutionUnit=NONE`，推理时需要 `--assume_mpp`。
5. **融合和阳性的坐标换算**都靠 dat 里的 `proc_resolution` / `base_resolution` / `proc_dimensions`，不要手改 dat。
6. **整图预览是缩略图**，几千个阳性点可能肉眼看不出来；需要时用全分辨率 `*_full.tif` 或裁局部看。
