# 用于精确连接采样基准的真实空间连接数据集

## 摘要

本文描述了三个为空间连接采样构造的真实数据基准：CMAB-Spatial-Join-0.08B、GeoLife-Spatial-Join-0.15B 和 COCO-Spatial-Join-1.23B。三个数据集都将真实对象转换为轴对齐盒或超矩形，并将得到的关系存储为 Parquet，但它们强调空间连接采样的不同方面。CMAB 是一个二维建筑轮廓与影响区域基准，来源于中国全国尺度的屋顶几何数据。GeoLife 是一个三维与四维时空相遇基准，来源于 GPS 轨迹。COCO 是一个十亿规模的三维视觉对象基准，来源于 MS COCO 2017 标注以及确定性的区域候选框。三者共同提供了互补的真实工作负载，用于测试无需物化完整连接结果即可从大规模盒相交连接中进行精确均匀采样的算法。

## 本实验仓库的获取策略

本文后续关于 GIS 清洗、GeoLife 轨迹解析和 COCO RPN 推理的说明用于解释三个公开数据集的上游 provenance，不是 `/home/dhy/PhD/impl` 必须再次执行的构建步骤。本实验仓库从空 `data-root` 开始时，只导入以下已经完成的 Hugging Face 数据集：

- `DannHiroaki/CMAB-Spatial-Join-0.08B@41e3c90fa42fc8eede910404fe3db29ad3897b81`；
- `DannHiroaki/Geolife-Spatial-Join-0.15B@a9b8439beb16de106f6ff3f54c73c6b6964d77af`；
- `DannHiroaki/COCO-Spatial-Join-1.23B@2e5f2a1ba741ba1148f0b2f42209a9da4635a6cb`。

`data_sources.lock.json` 冻结 repo revision、所需资产、字节数与 SHA-256；`configs/real_data.json` 冻结从成品表到实验 workload 的选择规则。仓库不得下载原始 GIS、`.plt`、COCO 图像或模型权重，不得运行 upstream builder、PROJ 或 Detectron2。CMAB 和 GeoLife 只下载所需 Parquet/JSON；COCO 的大 rectangle shards 只按选中 image 的 Parquet row groups 做 HTTP range 读取。

唯一入口为 `scripts/data/prepare_real_data.sh`。该入口只负责成品资产校验、确定性子集选择、必要的盒派生、规范二进制写出和最终验收；Setup 中的实验均使用这些冻结输出。

## 目的与通用基准模型

主实验中使用的合成数据生成器适合进行可控压力测试，但空间连接采样器也应当在真实数据上进行评估。真实空间数据包含独立合成盒模型难以捕捉的现象：空间聚集、层级行政结构、高度倾斜的对象尺寸、重复轨迹、时间突发、维度相关性、图像级局部性，以及候选区域之间的大量重叠。本文描述的三个数据集旨在暴露这些效应，同时保留干净的几何表示。

每个数据集本质上都是一组轴对齐矩形或超矩形。每条记录表示一个盒：

$$
b = \prod_{i=1}^{d} I_i(b),
$$

其中每个区间 $I_i(b)$ 要么是闭区间 $[L_i(b),U_i(b)]$，要么是半开区间 $[L_i(b),U_i(b))$，具体取决于数据集。一个空间连接工作负载会选择两个带颜色的关系，通常记为 $\mathcal R$ 和 $\mathcal S$，并要求得到集合：

$$
\mathcal J = \{(r,s)\in \mathcal R\times\mathcal S: r \cap s \neq \varnothing\}.
$$

对于连接采样而言，目标输出不是完整集合 $\mathcal J$，因为它可能过大而无法物化。相反，任务是生成独立的、有放回的样本，并且这些样本在 $\mathcal J$ 上精确均匀。

这三个数据集有意覆盖不同的情形。

- CMAB 提供大规模二维地理空间工作负载，包含真实建筑分布和可调影响半径。
- GeoLife 提供三维与四维整数超矩形，其相交关系编码时空相遇。
- COCO 提供十亿规模的半开三维工作负载，在生成的候选框之间具有极高的图像内重叠。

#### 边界约定

ANCHOR 的原始算法手稿和 KDS 基线使用半开盒与严格重叠。三个数据集的约定并不完全一致：CMAB 和 GeoLife 被指定为包含端点的闭区间相交语义，而 COCO 被指定为半开语义。实验中应明确报告这种差异。对于 GeoLife 这样的整数闭区间，只要所有坐标都位于整数网格上且 $u+1$ 可表示，闭谓词 $[\ell,u]$ 就可以表示为半开区间 $[\ell,u+1)$。对于 CMAB 这样的浮点闭区间，可以直接实现闭区间谓词，也可以为所有比较系统统一采用半开约定；由于边界接触的配对会受到不同处理，因此应说明所作选择。

## 数据集概览

表 1 总结了三个基准。这里展示的行数遵循公开数据集卡片和构建器清单。对于 CMAB，Hugging Face 数据集卡片同时暴露了按层级的配置和按省份的配置，因此其目录级行数可能会跨重叠配置重复计数同一批物理文件。基准名称中的 “0.08B” 应理解为已发布基准的近似不相交四层级 AABB 规模，而公开卡片当前报告的是所有暴露配置合计 150,831,244 行。

**表 1. 三个真实数据集的高层比较。**

| 数据集 | 维度与坐标 | 真实来源 | 主要几何对象 | 规模与作用 |
| --- | --- | --- | --- | --- |
| CMAB-Spatial-Join-0.08B | 二维投影米制坐标 | CMAB 全国尺度建筑屋顶几何数据 | 基础建筑 AABB，以及四个层级下扩展得到的影响 AABB | 建筑/影响连接，包含城市偏斜、行政分区和多半径选择率。公开卡片报告各配置合计 150,831,244 行；数据集名称反映约 0.08B 条已发布 AABB 记录。 |
| GeoLife-Spatial-Join-0.15B | 三维 $(x,y,t)$ 与四维 $(x,y,z,t)$ 整数坐标；$x,y,z$ 以厘米计，$t$ 以 Unix epoch 毫秒计 | GeoLife GPS Trajectories v1.3 | 三个相遇层级下以点为中心的时空超矩形 | 六个“层级—维度”分组中共有 148,923,081 条矩形记录；公开数据集卡片还暴露 18,670 行轨迹字典，因此总行数为 148,941,751。用于测试高维轨迹相遇连接。 |
| COCO-Spatial-Join-1.23B | 三维半开盒 $(x,y,z)$；图像像素坐标加图像索引 $z$ 切片 | MS COCO 2017 标注与 Detectron2 RPN 候选框 | Ground-truth 检测框，以及每张图像恰好 10,000 个 RPN 候选框 | 公开卡片报告 rect 与 image 子集共 1,233,890,069 行；rect 工作负载包含 1,232,870,000 个候选框以及 COCO GT 框。 |

## CMAB-Spatial-Join-0.08B

### 来源与目标

CMAB-Spatial-Join-0.08B 是一个二维地理空间 AABB 基准，来源于 CMAB 建筑屋顶数据集；其原始论文描述了一个中国全国尺度的多属性建筑数据集 [1, 2, 7, 8]。该基准将每个清洗后的屋顶几何对象转换为基础包围矩形和若干扩展影响矩形。它旨在用于在大规模真实建筑分布上评估空间连接系统与空间连接采样器。

构建器接受 Shapefile、GeoPackage 和 FileGDB 目录等 GIS 容器。它确定性地扫描输入树，按排序顺序处理文件，并记录相对源文件路径和源要素 id 等可追溯字段。行政元数据通过由 `StCity_id` 索引的 CMAB 元数据填充；当行政字段不完整时，构建器会退回到基于路径的推断和确定性的占位符。

### 坐标系统与几何清洗

输入几何对象按照文件内部的坐标参考系统解释。如果 CRS 缺失或无法解析，构建器回退到 WGS 1984 / EPSG:4326。所有输出矩形和派生几何统计量都在基于米的投影 CRS 中计算；默认目标 CRS 是覆盖中国的 Albers Equal Area 投影。因此，输出坐标以米为单位，使影响半径和几何尺寸可直接解释。

构建器仅接受多边形屋顶几何对象。如果一个要素的几何为空、不是 Polygon 或 MultiPolygon、无法修复为合法几何、投影失败、投影后边界非有限，或基础包围盒退化，则该要素会被丢弃。可选尺寸过滤器还可以丢弃投影包围盒异常大的几何对象。这些清洗规则对连接采样很重要，因为退化或非有限矩形会破坏几何谓词和精确采样器假设。

### 基础 AABB 与扩展影响 AABB

令 $G$ 表示某栋建筑的投影屋顶几何。其基础 AABB 为：

$$
\mathrm{AABB}(G) = [x_{\min},y_{\min},x_{\max},y_{\max}],
$$

派生字段为：

$$
w=x_{\max}-x_{\min},\qquad
  h=y_{\max}-y_{\min},\qquad
  a=wh,
$$

中心坐标为：

$$
c_x=(x_{\min}+x_{\max})/2,
  \qquad
  c_y=(y_{\min}+y_{\max})/2.
$$

对于每栋建筑，构建器还会生成扩展矩形。如果影响距离为 $d$，则扩展 AABB 为：

$$
[e x_{\min}, e y_{\min}, e x_{\max}, e y_{\max}]
  = [x_{\min}-d, y_{\min}-d, x_{\max}+d, y_{\max}+d].
$$

它表示一个 $L_\infty$ 影响区域：在所选矩形相交约定下，当另一个基础盒位于源建筑的方形影响包络内时，它会与扩展矩形相交。

每个清洗后的建筑在四个层级数据集中各出现一次。跨层级变化的字段只有层级 id、选定的影响距离 $d_m$ 和扩展坐标。基础 AABB、建筑标识符、功能类别和行政字段保持不变。

### 功能类别与影响层级

输出建筑功能被规范化为五个类别：

| 编码 | 功能类别 |
| --- | --- |
| 0 | 居住 |
| 1 | 商业 |
| 2 | 公共服务 |
| 3 | 办公 |
| 4 | 工业 |

影响距离同时取决于功能类别和层级。默认距离表见表 2。

**表 2. 默认 CMAB 影响距离（单位：米）。**

| 功能类别 | Level 1 | Level 2 | Level 3 | Level 4 |
| --- | --- | --- | --- | --- |
| 居住 | 500 | 800 | 1000 | 1500 |
| 商业 | 800 | 1000 | 1500 | 2500 |
| 公共服务 | 1000 | 2000 | 3000 | 5000 |
| 办公 | 1000 | 2000 | 3000 | 5000 |
| 工业 | 2000 | 5000 | 8000 | 10000 |

该表形成了自然的选择率梯度。Level 1 相对局部；Level 4 可能生成密度高得多的连接，尤其是公共服务、办公和工业建筑。依赖功能的半径很有用，因为它们即使在同一空间区域内也会产生异质的盒尺寸分布。

### 输出布局与模式

发布的数据集组织为分区 Parquet 数据集：

```text
cmab_spatial_join/level_1/
cmab_spatial_join/level_2/
cmab_spatial_join/level_3/
cmab_spatial_join/level_4/
```

每个层级按省份分区。根目录还包含 `dataset_metadata.json`、`file_manifest.parquet` 和 `summary_stats.parquet`。文件清单记录分片路径、行数、SHA256 哈希、列集合，以及每个文件的包围盒范围。汇总统计表包含按层级、省份和功能编码分组的统计量，包括基础矩形与扩展矩形尺寸的分位数。

四个层级共享同一个模式。最重要的字段列于表 3。该基准同时保留几何属性和语义属性，使实验可以在全局、省份、城市、功能或层级维度上运行。

**表 3. CMAB 主要字段。**

| 字段组 | 字段 | 含义 |
| --- | --- | --- |
| 标识符 | `building_uid`, `shape_id` | 稳定的建筑标识符与源 shape 标识符。64 位 UID 由行政键和 shape id 使用 xxHash64 派生。 |
| 类别 | `func`, `func_code`, `level`, `d_m` | 规范化功能类别、数字功能编码、工作负载层级和以米计的影响距离。 |
| 行政元数据 | `province`, `city`, `district`, `admin_level`, `stcity_id`, `block_id` | 用于分区和区域过滤工作负载的区域属性。 |
| 基础 AABB | `xmin`, `ymin`, `xmax`, `ymax` | 以米计的投影屋顶包围矩形。 |
| 基础几何统计 | `bbox_w_m`, `bbox_h_m`, `bbox_area_m2`, `cx`, `cy` | 基础矩形的宽度、高度、面积和中心。 |
| 扩展 AABB | `exmin`, `eymin`, `exmax`, `eymax` | 对应层级的影响矩形。 |
| 源属性 | `height_m`, `bu_area_m2` | 可用时的建筑高度和建筑面积属性。 |
| 可追溯性 | `source_file`, `source_fid` | 输入文件路径与源要素 id。 |

### 推荐连接工作负载

CMAB 基准支持若干自然工作负载。

#### 影响到基础连接

对于固定层级 $\ell$，定义左关系 $\mathcal R_\ell$ 为层级 $\ell$ 的扩展 AABB，右关系 $\mathcal S$ 为基础 AABB。连接对 $(r,s)$ 表示建筑 $s$ 的基础矩形与建筑 $r$ 的影响矩形相交。这是该数据集最直接的解释：它采样在依赖功能的影响距离下受另一建筑影响或与另一建筑空间接近的建筑。

#### 影响自连接

对于固定层级 $\ell$，两侧都使用扩展 AABB，并在适当时过滤相同建筑 id。这衡量影响区域之间的重叠。它通常比影响到基础连接更稠密，因此是对输出基数敏感采样器的有用压力测试。

#### 区域过滤连接

同样的任务可以在按省份、城市、区县或 `stcity_id` 过滤后运行。这使实验能够评估采样器是否能在密度差异很大的区域之间平滑扩展。例如，稠密沿海省份和大型都市区域很可能生成与稀疏西部或农村区域不同的重叠结构。

#### 功能过滤连接

由于功能类别具有不同半径，实验还可以按功能过滤记录或给记录着色。示例包括居住到商业影响采样、工业到基础影响采样，或完整的跨功能连接工作负载矩阵。

### 为什么 CMAB 对空间连接采样有用

CMAB 在大规模上考验二维真实地理空间索引。主要难点不是高维，而是空间偏斜和异质选择率。建筑在城市中强烈聚集；省份之间的行数相差数个数量级；依赖功能的影响距离在同一数据集中产生多个盒尺寸分布；四个层级形成系统性的选择率扫描。在 CMAB 上表现良好的采样器应能同时处理小型局部连接和高密度城市连接，而不依赖均匀空间分布假设。

## GeoLife-Spatial-Join-0.15B

### 来源与目标

GeoLife-Spatial-Join-0.15B 来源于 Microsoft Research 的 GeoLife GPS Trajectories v1.3 [3, 4, 9, 10]。源数据包含 182 名用户采集的 GPS 轨迹。每个轨迹点包含纬度、经度、海拔和时间戳信息。构建器将轨迹点转换为轴对齐超矩形，使“两个点在空间和时间上是否接近？”这个查询变成纯粹的盒相交连接。

该基准包含六个分组：

- 基于 $(x,y,t)$ 的三维分组，包含 Level 1、2、3；
- 基于 $(x,y,z,t)$ 的四维分组，包含 Level 1、2、3，并且仅使用海拔有效的点。

公开制品中，每个三维层级包含 24,876,977 行，每个四维层级包含 24,764,050 行，总计 148,923,081 条矩形记录。加上 18,670 行轨迹字典，Hugging Face 的总行数为 148,941,751。三维分组覆盖 182 名用户和 18,670 条轨迹；在海拔过滤后，四维分组覆盖 182 名用户和 18,645 条轨迹。

### 点提取

构建器期望输入为解压后的 GeoLife 1.3 目录，其中包含 `Data/`，并包含类似 `Data/000/` 的用户目录。它递归收集所有 `.plt` 文件，将路径规范化为相对于 `Data/` 的路径，按字典序排序，并按该确定性顺序处理文件。

对于每个 `.plt` 文件，前六行头部会被跳过。后续每一行都会被解析为一个 GPS 点。相关字段包括纬度、经度、以英尺计的海拔、日期和时间。只有当纬度位于 $[-90,90]$、经度位于 $[-180,180]$ 且时间戳可解析时，该点才会保留。每个被接受的点会获得：

- 从用户目录解析出的 `user_id`；
- 规范化轨迹路径 `traj_src`；
- `traj_src` 的确定性哈希 `traj_id`；
- 轨迹内点索引 `point_idx`；
- 纬度、经度、海拔和编码后的时间。

### 坐标编码

空间坐标从 WGS84 / EPSG:4326 投影到 Web Mercator / EPSG:3857。得到的米制坐标被转换为整数厘米：

$$
x_{\mathrm{cm}} = \operatorname{round\_away\_from\_zero}(100x_{\mathrm{m}}),
  \qquad
  y_{\mathrm{cm}} = \operatorname{round\_away\_from\_zero}(100y_{\mathrm{m}}).
$$

时间戳被解释为 UTC，并转换为 Unix epoch 毫秒：

$$
t_{\mathrm{ms}}=\lfloor 1000\cdot \mathrm{timestamp\_seconds}\rfloor.
$$

海拔以英尺读取并转换为厘米：

$$
z_{\mathrm{cm}}=\operatorname{round\_away\_from\_zero}(100\cdot 0.3048\cdot \mathrm{alt}_{ft}).
$$

只有当 `alt_ft != -777` 且转换后的海拔位于 $[-500\,\mathrm{m},10000\,\mathrm{m}]$ 时，一个点才对四维分组有效。

这种整数编码对精确采样实验非常有利。它消除了生成的超矩形端点中的浮点歧义，并使边界转换变得显式。

### 相遇层级与盒构造

每个层级由空间阈值 $\Delta d$ 和时间阈值 $\Delta t$ 定义。以 $(x,y,t)$ 为中心的点会变成一个三维盒，其空间半宽为 $r_d=\Delta d/2$，时间半宽为 $r_t=\Delta t/2$：

$$
\begin{aligned}
  x_{\min} &= x_{\mathrm{cm}}-r_{d,\mathrm{cm}}, & x_{\max} &= x_{\mathrm{cm}}+r_{d,\mathrm{cm}},\\
  y_{\min} &= y_{\mathrm{cm}}-r_{d,\mathrm{cm}}, & y_{\max} &= y_{\mathrm{cm}}+r_{d,\mathrm{cm}},\\
  t_{\min} &= t_{\mathrm{ms}}-r_{t,\mathrm{ms}}, & t_{\max} &= t_{\mathrm{ms}}+r_{t,\mathrm{ms}}.
\end{aligned}
$$

对于四维记录，该盒额外包含：

$$
z_{\min}=z_{\mathrm{cm}}-r_{d,\mathrm{cm}},
  \qquad
  z_{\max}=z_{\mathrm{cm}}+r_{d,\mathrm{cm}}.
$$

固定层级表见表 4。

**表 4. GeoLife 相遇阈值与半宽。**

| 层级 | $\Delta d$ (m) | $r_{d,\mathrm{cm}}$ | $\Delta t$ (s) | $r_{t,\mathrm{ms}}$ |
| --- | --- | --- | --- | --- |
| 1 | 20 | 1,000 | 60 | 30,000 |
| 2 | 50 | 2,500 | 300 | 150,000 |
| 3 | 200 | 10,000 | 1200 | 600,000 |

该数据集使用闭区间。因此，同一层级中的两个三维盒相交，当且仅当其底层点满足：

$$
|x_1-x_2|\le \Delta d,
  \qquad
  |y_1-y_2|\le \Delta d,
  \qquad
  |t_1-t_2|\le \Delta t.
$$

对于四维，条件还额外包括 $|z_1-z_2|\le \Delta d$。因此，矩形相交连接精确实现了一个 $L_\infty$ 时空相遇谓词。

### 输出布局与模式

输出目录具有如下逻辑结构：

```text
geolife_spatial_join/manifest.json
geolife_spatial_join/dims=3/level=1/part-*.parquet
geolife_spatial_join/dims=3/level=2/part-*.parquet
geolife_spatial_join/dims=3/level=3/part-*.parquet
geolife_spatial_join/dims=4/level=1/part-*.parquet
geolife_spatial_join/dims=4/level=2/part-*.parquet
geolife_spatial_join/dims=4/level=3/part-*.parquet
geolife_spatial_join/dict/trajectories.parquet
```

矩形模式紧凑且取整数值。公共字段包括 `rect_id:int64`、`traj_id:int64`、`user_id:int32` 和 `point_idx:int32`。三维边界字段为 `x_min_cm`、`x_max_cm`、`y_min_cm`、`y_max_cm`、`t_min_ms` 和 `t_max_ms`。四维分组还包含 `z_min_cm` 和 `z_max_cm`。轨迹字典将 `traj_id` 映射回 `traj_src`、用户 id、保留点数量和时间跨度。

清单记录不可变的构建参数，包括 CRS 选择、单位、阈值层级、海拔有效性规则、精确行数、文件列表和哈希规则。确定性 id 方案使用种子为 0 的 xxHash64。

### 推荐连接工作负载

#### 全点相遇连接

对于固定的 `dims` 和 `level`，将记录与自身连接，并过滤掉相同的 `rect_id`。这会采样中心在 $L_\infty$ 时空意义上彼此接近的轨迹点对。如果目标是避免重复的对称配对，可以施加 `rect_id_left < rect_id_right`；如果算法期望跨颜色连接，则使用该关系的两个逻辑副本并采样有序对。

#### 跨用户相遇连接

通过 `user_id_left != user_id_right` 过滤自连接。这会将工作负载转换为跨用户相遇基准，并避免同一轨迹中相邻点之间的平凡匹配。它尤其适合在具有语义意义的相遇条件下评估采样器。

#### 轨迹对连接

使用轨迹字典选择轨迹子集或用户组。例如，可以将一个选定用户集合的点与另一个用户集合的点连接；如果外部添加了工作日/周末标签，也可以连接工作日轨迹和周末轨迹。

#### 维度比较

在三维和四维中运行相同层级。四维版本加入海拔，因此会改变选择率和索引复杂度。这有助于比较运行时间强烈依赖维度的算法。

### 为什么 GeoLife 对空间连接采样有用

GeoLife 是一个具有强时间和空间相关性的高维真实工作负载。轨迹中的连续点彼此接近；用户会重复访问同一地点；活动突发会在时间上形成稠密局部邻域。三个层级提供受控的选择率扫描，而三维/四维划分测试算法对维度的敏感性。由于所有坐标都经过整数编码，它也非常适合精确实现闭区间语义或离散化的半开语义。

## COCO-Spatial-Join-1.23B

### 来源与目标

COCO-Spatial-Join-1.23B 是一个十亿规模的视觉空间连接基准，构建自 MS COCO 2017 检测数据，以及由 Detectron2 Faster R-CNN（ResNet-50-FPN backbone）生成的确定性区域候选框 [5, 6, 11, 12]。该基准使用 `train2017` 和 `val2017` 图像划分，共包含 118,287 张训练图像、5,000 张验证图像和 123,287 张图像。对于每张图像，它存储两类矩形：

- GT 框：由 COCO 检测标注转换而来，包括 `iscrowd` 等于 0 或 1 的标注；
- Proposal 框：由 Faster R-CNN 模型的 RPN 输出生成，每张图像恰好包含 10,000 个候选框。

结果是一个非常大的图像平面盒集合，并且图像内重叠很强。这使其成为采样器的优秀压力测试，因为采样器必须处理巨大的连接基数，并从高度重叠的局部实例中反复采样。

### 三维半开表示

每个对象表示为一个三维半开盒：

$$
b=[x_{\min},x_{\max})\times[y_{\min},y_{\max})\times[z_{\min},z_{\max}).
$$

$x$ 和 $y$ 坐标是原始 COCO 图像像素坐标系中的 float32 坐标。$z$ 坐标是图像切片。图像以确定性顺序排列：先放所有 `train2017` 图像，再放所有 `val2017` 图像；每个划分内部按 COCO 图像 id 升序排序。如果一张图像的索引为 `z_idx`，则该图像中的每个对象都会获得：

$$
z_{\min}=\texttt{z\_idx},
  \qquad
  z_{\max}=\texttt{z\_idx}+1.
$$

因此，不同图像的盒在 $z$ 维度上永不相交，而同一图像的所有盒共享同一个单位 $z$ 区间。

相交谓词是严格的半开重叠：

$$
\max(L_i(b),L_i(b')) < \min(U_i(b),U_i(b'))
  \quad\text{for every } i\in\{x,y,z\}.
$$

边界接触不计为相交。这与 ANCHOR 和 KDS 使用的半开语义一致，无需进行边界转换。

### 候选框生成与确定性

候选框来源是 Detectron2 Faster R-CNN 的 RPN 候选框生成器，而不是最终 ROI-head 检测结果。构建器记录模型配置 `COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml`、checkpoint URL 与 SHA256、PyTorch 和 Detectron2 版本、CUDA/cuDNN 版本、硬件信息以及确定性设置。公开构建清单记录 `seed_root=12345`，禁用 cuDNN benchmarking 和 TF32 等非确定性开关，并启用确定性算法设置。

为了获得一个大规模、高重叠的候选框池，RPN 测试配置使用：

$$
\texttt{PRE\_NMS\_TOPK\_TEST}=20000,
\quad
\texttt{POST\_NMS\_TOPK\_TEST}=20000,
\quad
\texttt{NMS\_THRESH}=1.0.
$$

实现会验证每张图像的候选池至少包含 10,000 个盒。候选框会被映射回原始图像坐标系，裁剪到图像边界，规范化为有限 float32 坐标，确定性排序，并截断为前 10,000 个。存储的 proposal 分数为：

$$
\sigma(\mathrm{objectness\_logit}) = \frac{1}{1+\exp(-\mathrm{logit})},
$$

并以 float32 存储。确定性排序键为：分数降序，然后元组 $(x_{\min},y_{\min},x_{\max},y_{\max})$ 升序，然后原始候选索引升序。最终 proposal rank 在每张图像内为 `1..10000`。

### 坐标规范化

写入之前，每个 GT 或 proposal 框必须满足：

$$
0\le x_{\min}<x_{\max}\le \mathrm{width},
  \qquad
  0\le y_{\min}<y_{\max}\le \mathrm{height}.
$$

非有限值会在裁剪前以确定性方式处理。退化盒使用 float32 `nextafter` 修复：如果 `max_x <= min_x`，则将 `max_x` 推进到大于 `min_x` 的下一个可表示浮点数；如果这超过图像边界，则将 `max_x` 设为边界，并将 `min_x` 移到前一个可表示浮点数。$y$ 方向采用同样规则。该规范化保证严格的半开盒有效性。

### 输出布局与模式

输出结构为：

```text
coco-spatial-1b/meta/build_manifest.json
coco-spatial-1b/meta/stats.json
coco-spatial-1b/data/images.parquet
coco-spatial-1b/data/rects/train2017/shard-*.parquet
coco-spatial-1b/data/rects/val2017/shard-*.parquet
```

每个分片覆盖按 `z_idx` 连续排列的至多 1,024 张图像；训练和验证分片不会混合。分片文件名基于起始图像索引，例如 `shard-000000.parquet` 和 `shard-001024.parquet`。每个 rect 分片内部的行按 `rect_id` 排序。

图像表每张图像一行，字段包括 `split`、`coco_image_id`、`file_name`、`width`、`height` 和 `z_idx`。rect 表包含表 5 中的字段。

**表 5. COCO rect 主要字段。**

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `rect_id` | int64 | 确定性矩形 id。 |
| `z_min`, `z_max` | int32 | 图像切片区间，其中 `z_max = z_min + 1`。 |
| `min_x`, `min_y`, `max_x`, `max_y` | float32 | 原始像素坐标中的半开图像平面盒。 |
| `type` | int8 | GT 为 0，proposal 为 1。 |
| `rank` | int16 | GT 为 0；proposal 为 1 到 10,000。 |
| `score` | float32 | GT 为 1.0；proposal 为 RPN objectness logit 的 sigmoid。 |
| `category_id` | int16 | GT 的 COCO 类别 id；proposal 为 -1。 |
| `iscrowd` | int8 | GT 的 COCO `iscrowd`；proposal 为 0。 |
| `coco_ann_id` | int64 | GT 的 COCO annotation id；proposal 为 -1。 |
| `coco_image_id` | int32 | 原始 COCO 图像 id。 |

确定性矩形 id 方案对每张图像使用固定步长 20,000。对于图像索引 $z$，定义：

$$
\mathrm{base}=20000z.
$$

GT 框按 COCO annotation id 排序，并分配 id $\mathrm{base}+0,\mathrm{base}+1,\ldots$。Proposal 框分配为：

$$
\texttt{rect\_id}=\mathrm{base}+10000+(\texttt{rank}-1).
$$

该步长将 GT id 和 proposal id 分离，并使 id 可由图像反推。

公开构建清单记录 123,287 张图像和 1,232,870,000 个 proposal。Hugging Face 卡片报告公开子集中共有 1,233,890,069 行，其中 `images` 子集约 123k 行，`rects` 子集约 1.23B 行。结合这些公开计数可推出 rect 关系中包含 896,782 个 GT annotation 框：

$$
1{,}233{,}890{,}069 - 1{,}232{,}870{,}000 - 123{,}287 = 896{,}782.
$$

### 推荐连接工作负载

#### 任务 A：GT $\times$ Proposal

令 $\mathcal R$ 为所有 `type=0` 的记录，令 $\mathcal S$ 为所有 `type=1` 的记录。该连接采样同一图像中 ground-truth 对象与 RPN proposal 严格重叠的配对。该工作负载在语义上接近目标检测中的 proposal recall，并产生一个非常大但结构化的连接。

#### 任务 B：Proposal $\times$ Proposal

对每张图像，按 rank 分割 proposal：

$$
\mathcal R=\{\texttt{type=1},\ 1\le \texttt{rank}\le 5000\},
  \qquad
  \mathcal S=\{\texttt{type=1},\ 5001\le \texttt{rank}\le 10000\}.
$$

该连接采样同一图像内彼此重叠的 proposal 对。这是一个纯粹的高重叠压力测试。$z$ 维度使全局工作负载可以看作逐图像连接的不相交并；但算法也可以在完整三维关系上运行。

#### 按划分的工作负载

同样的任务可以分别在 `train2017`、`val2017` 或选定分片范围上运行。由于分片在 `z_idx` 上连续，它们提供了可复现的前缀或区间。通过改变 rank 分割可以使任务 B 更稠密，不过默认基准分割为 ranks 1–5000 对 5001–10000。

### 为什么 COCO 对空间连接采样有用

COCO 是三个数据集中最大的一个，其结构也与经典地理空间数据非常不同。它的盒位于小型图像坐标系中，但由 $z$ 维度隔开。在单张图像内，proposal 数量被有意设置得很多且高度重叠，因此局部连接基数可能非常大。跨图像之间完全没有重叠。这形成了一种清晰的块对角几何结构：按图像分区很容易，但每张图像内部很难。优秀的采样器应能利用或承受这种结构，而无需物化所有 proposal-proposal 或 GT-proposal 相交。

## 空间连接采样中的实验使用

### 构造带颜色关系

大多数精确连接采样器都被定义在两个带颜色关系 $\mathcal R$ 和 $\mathcal S$ 上。三个数据集自然地以不同方式产生这样的关系。

- 在 CMAB 中，将扩展盒作为 $\mathcal R$，将基础盒作为 $\mathcal S$，用于影响到基础连接。对于自连接式工作负载，创建两个逻辑副本，并在适当时过滤相同建筑 id。
- 在 GeoLife 中，对同一层级/维度表使用两个逻辑副本进行相遇连接，或者按用户 id、轨迹 id、时间范围或用户组分割记录，以形成真正的跨颜色连接。
- 在 COCO 中，直接使用语义记录类型：任务 A 使用 GT 对 proposal，任务 B 使用前半 rank proposal 对后半 rank proposal。

在采样有序对时，应明确对称配对是否都保留。例如，对单表自连接可以在两个副本的跨颜色连接实现中同时产生 $(a,b)$ 和 $(b,a)$。如果需要无序对，则施加严格 id 顺序，并在所有系统中一致调整。

### 报告连接语义

每个实验都应报告实现使用的区间约定。建议报告表如下：

| 数据集 | 原生约定 | 实现说明 |
| --- | --- | --- |
| CMAB | 构建器工作负载描述中的包含端点矩形相交 | 说明实验使用闭区间相交还是统一半开约定。 |
| GeoLife | 闭整数区间 | 可通过在每个整数维度上将 $u$ 替换为 $u+1$，精确转换为半开区间。 |
| COCO | 半开严格重叠 | 直接匹配半开算法手稿。 |

这主要影响边界接触情形。边界接触在连续地理空间数据中可能很少见，但在整数编码数据中按定义并非可以忽略。

### 推荐指标

对于每个数据集和工作负载，至少报告：

- 两侧记录数 $|\mathcal R|$ 和 $|\mathcal S|$；
- 维度 $d$ 与区间约定；
- 可获得时的估计或精确连接基数 $|\mathcal J|$；
- 预处理时间和内存；
- 持久化索引大小；
- 跨若干数量级样本数 $t$ 的查询时间；
- 采样器是否返回有放回的 i.i.d. 有序样本；
- 对均匀性和有放回重复行为的验证方法。

对于非常大的工作负载，精确 $|\mathcal J|$ 本身也可能很昂贵。在这种情况下，应报告采样器使用的计数方法、基线计数器或独立估计器。关键是区分索引构建成本、计数成本和单样本成本。

### 扩展协议

这些数据集支持自然的缩放和扩展协议。

#### CMAB 扩展

按层级、省份、城市或功能类别扩展。层级控制盒扩展和选择率；省份控制行数和空间密度；功能过滤控制半径分布。

#### GeoLife 扩展

按维度、层级、用户子集或轨迹子集扩展。三维/四维比较隔离维度效应；三个层级隔离阈值/选择率效应。

#### COCO 扩展

按划分、分片范围或图像子集扩展。由于分片在 `z_idx` 上连续，它们提供可复现的前缀或区间。任务 B 可以通过改变 rank 分割变得更稠密，不过默认基准分割是 ranks 1–5000 对 5001–10000。

## 基准情形比较

这三个数据集是互补的，而不是冗余的。

- CMAB 测试具有真实行政结构和城市偏斜的大规模二维地理空间连接。它接近经典空间数据库工作负载。
- GeoLife 测试高维时空连接。对于复杂度依赖维度的算法，以及同时涉及空间和时间阈值的谓词，它是最自然的基准。
- COCO 测试十亿规模高重叠连接。它不是地理数据集，但由于目标 proposal 会在每张图像内产生大量重叠矩形，因此它是非常强的盒相交工作负载。

空间连接采样器在不同数据集上的预期优势和风险不同。在 CMAB 上，空间分区和区域级局部性很重要。在 GeoLife 上，时间局部性、重复轨迹点和整数边界处理很重要。在 COCO 上，主导挑战是巨量 proposal 框以及每个图像切片内部的稠密局部连接。

## 建议的论文表述

以下段落可以直接用于实验章节。

*真实数据集。*  我们在三个由独立应用领域构造的真实空间连接基准上进行评估。CMAB-Spatial-Join-0.08B 是一个二维建筑基准，来源于 CMAB 屋顶几何数据；每栋建筑贡献一个基础 AABB 和四个依赖层级的影响 AABB，影响距离由建筑功能类别决定。GeoLife-Spatial-Join-0.15B 是一个来源于 GeoLife GPS 轨迹的时空相遇基准；每个轨迹点被编码为三维 $(x,y,t)$ 或四维 $(x,y,z,t)$ 整数超矩形，每次相交都精确对应于 $L_\infty$ 距离/时间阈值下的接近关系。COCO-Spatial-Join-1.23B 是一个来源于 MS COCO 2017 的十亿规模视觉基准；它将 ground-truth 检测框和每张图像恰好 10,000 个确定性 Detectron2 RPN proposal 存储为半开三维盒，并使用图像索引作为分离 $z$ 坐标。这些数据集分别覆盖二维地理空间偏斜、高维时空相关性和十亿规模图像局部高重叠连接。

## 参考文献与数据访问

三个已发布数据集及其参考构建器可在以下位置获取：

- CMAB 构建器：<https://github.com/DANNHIROAKI/CMAB-Spatial-Join-0.08B-Builder>；数据集：<https://huggingface.co/datasets/DannHiroaki/CMAB-Spatial-Join-0.08B>。
- GeoLife 构建器：<https://github.com/DANNHIROAKI/Geolife-Spatial-Join-0.15B-Builder>；数据集：<https://huggingface.co/datasets/DannHiroaki/Geolife-Spatial-Join-0.15B>。
- COCO 构建器：<https://github.com/DANNHIROAKI/COCO-Spatial-Join-1B-Builder>；数据集：<https://huggingface.co/datasets/DannHiroaki/COCO-Spatial-Join-1.23B>。

## 参考文献

1. Y. Zhang, H. Zhao, and Y. Long. CMAB: A Multi-Attribute Building Dataset of China. *Scientific Data*, 12:430, 2025. DOI: <https://doi.org/10.1038/s41597-025-04730-5>。
2. Y. Zhang, H. Zhao, and Y. Long. CMAB–The World's First National-Scale Multi-Attribute Building Dataset. Figshare dataset, 2025. DOI: <https://doi.org/10.6084/m9.figshare.27992417>。
3. Microsoft Research. GeoLife GPS Trajectories v1.3. Dataset and user guide, 2011. <https://www.microsoft.com/en-us/research/publication/geolife-gps-trajectory-dataset-user-guide/>。
4. Y. Zheng, X. Xie, and W.-Y. Ma. GeoLife: A Collaborative Social Networking Service among User, Location and Trajectory. *IEEE Data Engineering Bulletin*, 33(2):32–39, 2010。
5. T.-Y. Lin, M. Maire, S. Belongie, L. Bourdev, R. Girshick, J. Hays, P. Perona, D. Ramanan, C. L. Zitnick, and P. Dollár. Microsoft COCO: Common Objects in Context. In *European Conference on Computer Vision*, 2014. <https://arxiv.org/abs/1405.0312>。
6. Y. Wu, A. Kirillov, F. Massa, W.-Y. Lo, and R. Girshick. Detectron2. <https://github.com/facebookresearch/detectron2>, 2019。
7. DANNHIROAKI. CMAB-Spatial-Join-0.08B-Builder. <https://github.com/DANNHIROAKI/CMAB-Spatial-Join-0.08B-Builder>。
8. DANNHIROAKI. CMAB-Spatial-Join-0.08B. Hugging Face dataset. <https://huggingface.co/datasets/DannHiroaki/CMAB-Spatial-Join-0.08B>。
9. DANNHIROAKI. Geolife-Spatial-Join-0.15B-Builder. <https://github.com/DANNHIROAKI/Geolife-Spatial-Join-0.15B-Builder>。
10. DANNHIROAKI. Geolife-Spatial-Join-0.15B. Hugging Face dataset. <https://huggingface.co/datasets/DannHiroaki/Geolife-Spatial-Join-0.15B>。
11. DANNHIROAKI. COCO-Spatial-Join-1B-Builder. <https://github.com/DANNHIROAKI/COCO-Spatial-Join-1B-Builder>。
12. DANNHIROAKI. COCO-Spatial-Join-1.23B. Hugging Face dataset. <https://huggingface.co/datasets/DannHiroaki/COCO-Spatial-Join-1.23B>。
