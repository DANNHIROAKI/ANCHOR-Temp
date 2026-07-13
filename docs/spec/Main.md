# 跨颜色盒相交连接的精确独立采样：ANCHOR 与 Range-Tree Baselines

## 摘要

本文研究两个颜色集合之间的 $d$ 维轴对齐半开盒相交连接，并给出四个精确独立均匀采样算法：**ANCHOR-Compiled (AC)**、**ANCHOR-Streaming (AS)**、**SweepRT** 与 **LiftedRT**。四个算法共享同一个目标：给定输出长度 $t$，若连接非空，则返回 $t$ 个来自连接结果的带放回、有序、独立均匀样本；若连接为空且 $t>0$，则返回 `EMPTY-INSTANCE`。本文允许两侧盒具有任意且彼此不同的边长，并保留重复几何坐标下的对象身份。

## 主要贡献

1. **ANCHOR 分解。** 通过当前维度的锚定所有权规则和对侧秩区间的 dyadic 规范覆盖，将跨颜色盒相交连接分解为互不相交的后缀子连接。
2. **AC。** 将 ANCHOR 分解完全展开到终端一维实例，物化终端原子，并以 $O(t)$ 时间回答长度为 $t$ 的精确独立均匀采样查询。
3. **AS。** 不保存完整分解树，而是流式路由、递归计数和按配额采样；以过滤后规模 $N$ 计，其核心采样时间为 $O(N\log^{d-1}(N+1)+t)$，辅助空间为 $O(N)$。
4. **Range-tree baselines。** LiftedRT 将 $d$ 维盒相交转化为 $2d$ 维静态正交范围查询；SweepRT 用扫描线处理第一维，并将剩余 $d-1$ 维转化为 $2d-2$ 维动态正交范围查询。

全文所有概率结论均在精确离散随机模型下陈述，不依赖浮点概率近似。时间界以固定维度、精确比较和单位代价离散随机原语为基础；若将随机原语展开为无偏随机位上的拒绝采样，相应界解释为期望时间界，输出分布仍然精确。

## 相关工作与问题边界

正交对象相交与盒相交枚举是计算几何中的经典问题，Edelsbrunner 与 Maurer 给出了这一问题的早期算法研究 [1]。关系数据库中的结果采样和连接采样研究了如何在不完整物化查询结果的情况下生成简单随机样本 [9,10]。独立范围采样关注从一次范围查询结果中生成独立均匀样本：一维点集、低维空间点集和区间对象分别已有专门数据结构 [2–4]。空间范围连接采样进一步研究了由统一窗口尺寸诱导的二维点—点范围连接采样 [5]。

本文处理的对象和输出语义与上述范围查询模型不同。输入两侧均由任意大小的 $d$ 维盒组成，连接谓词是两盒在全部维度上的非空相交；算法必须在不枚举连接结果的前提下，返回连接对上的精确带放回 i.i.d. 样本。ANCHOR 直接对 box–box 相交连接建立不交所有权分解；LiftedRT 与 SweepRT 则给出基于标准正交范围结构的参照实现。本文只陈述固定维度内的最坏情形渐近界和精确分布保证，不主张高维、外存、并行或近似模型下的最优性。

---

# Part I. 统一问题、符号、模型与主结果

## 1. 输入对象与半开盒语义

原始输入由两个颜色集合组成：

$$
\mathcal R_0,\mathcal S_0.
$$

每个对象 $q$ 是一个 $d$ 维坐标轴对齐半开盒：

$$
q=\prod_{j=1}^{d}[L_j(q),U_j(q)).
$$

维度满足 $d\in\mathbb Z_{\ge1}$，并在复杂度分析中视为固定常数。端点来自支持精确全序比较的数值域；LiftedRT 所用的取负运算可以由反向次序键等价实现，因此算法实质上只依赖精确比较。若某个对象 $q$ 在至少一个维度 $j$ 上满足

$$
L_j(q)\ge U_j(q),
$$

则该盒为空盒，不可能参与任何非空相交。对于 $t=0$ 的采样请求，算法可以直接返回空序列；除此之外，所有算法首先在线性时间内扫描原始输入并确定性地丢弃空盒。保留对象集合记为

$$
\mathcal R\subseteq\mathcal R_0,
\qquad
\mathcal S\subseteq\mathcal S_0.
$$

原始规模与过滤后规模分别记为

$$
N_0=|\mathcal R_0|+|\mathcal S_0|,
\qquad
N=|\mathcal R|+|\mathcal S|\le N_0.
$$

之后默认所有保留对象都满足

$$
L_j(q)<U_j(q),\qquad j=1,\ldots,d.
$$

对象身份始终保留。即使两个盒具有完全相同的几何坐标，只要对象 id 不同，它们就是两个不同输入对象，并应在连接和采样中作为不同结果计数。若两侧 id 只在各自集合内部唯一，则统一使用 $(\mathrm{side},\mathrm{id})$ 作为全局对象键。所有稳定排序都以该全局对象键作为最终 tie-breaker。

若过滤后 $\mathcal R=\emptyset$ 或 $\mathcal S=\emptyset$，则不存在跨颜色相交连接对。所有算法在构建线段树、range tree 或排序视图之前处理这一情形：正长度采样请求返回 `EMPTY-INSTANCE`。因此后文每次实际构建带填充树时，其真实叶子数均为正。

对两个一维半开区间，有

$$
[L_j(a),U_j(a))\cap [L_j(b),U_j(b))\ne\emptyset
$$

当且仅当

$$
L_j(a)<U_j(b)\quad\text{且}\quad L_j(b)<U_j(a).
$$

因此两个 $d$ 维盒相交，当且仅当它们在每个维度上都满足上述两个严格不等式。记

$$
a\sim b
$$

表示 $a$ 与 $b$ 相交。

## 2. 连接目标与采样目标

跨颜色相交连接定义为

$$
J=\{(a,b)\in\mathcal R\times\mathcal S:a\sim b\}.
$$

查询输入为整数

$$
t\in\mathbb Z_{\ge0}.
$$

输出规则如下：

1. 若 $t=0$，返回空序列。
2. 若 $t>0$ 且 $J=\emptyset$，返回 `EMPTY-INSTANCE`。
3. 若 $t>0$ 且 $J\ne\emptyset$，返回有序序列

$$
\mathbf Z=(Z_1,\ldots,Z_t)\in J^t,
$$

并满足

$$
Z_1,\ldots,Z_t\overset{\mathrm{i.i.d.}}{\sim}\operatorname{Unif}(J).
$$

等价地，对每个 $(p_1,\ldots,p_t)\in J^t$，有

$$
\Pr[\mathbf Z=(p_1,\ldots,p_t)]=|J|^{-t}.
$$

采样是带放回的，输出顺序有意义。同一个连接对可以出现在多个输出位置。

## 3. 精确离散随机模型

所有分布保证均在 exact word-RAM 模型下陈述。RAM 字长为 $w=\Theta(\log(N_0+t+2))$，常数因子足以容纳本文出现的多项式大小整数。端点比较是精确的单位代价操作；所有计数、权重、偏移量和配额均以精确非负整数保存。由于 $|J|\le |\mathcal R||\mathcal S|\le N^2$，连接计数和权重占用常数个 RAM 字。固定 $d$ 时，标签数、$kw_i$、$kW$、索引和配额等中间量均为 $N_0+t$ 的多项式大小整数，因而占用常数个 RAM 字。常数个 RAM 字整数之间的加减、乘法、比较、数组访问以及下述精确随机原语调用按 $O(1)$ RAM 操作计。输入 $t$ 被假定为可寻址输出长度，因此输出数组本身占用 $O(t)$ 个字。

所有随机原语的不同调用使用相互独立的随机性。本文的 $O(1)$ 随机原语口径是标准单位代价接口；若从无偏随机位实现，则下面的拒绝采样消耗期望 $O(\log(q-p+1))$ 个随机位和期望常数次 $O(w)$ 位字块尝试。此时全文的随机运行时间界均为期望界，确定性的数据结构操作界保持最坏情形成立。

算法使用如下离散随机原语。

### 3.1 均匀整数

给定整数 $p<q$，`UniformInteger(p,q)` 从集合

$$
\{p,p+1,\ldots,q-1\}
$$

中精确均匀返回一个整数。若 $q-p=1$，直接返回 $p$；否则令 $b=\lceil\log_2(q-p)\rceil$，反复读取 $b$ 个无偏随机位得到 $X\in[0,2^b)$，仅当 $X<q-p$ 时返回 $p+X$。接受概率大于 $1/2$，故期望尝试次数小于 $2$；该过程不使用取模截断，并且不存在有限的最坏情形尝试次数。

### 3.2 精确加权索引

给定 $k$ 个非负整数权重 $w_1,\ldots,w_k$，总权重

$$
W=\sum_{i=1}^k w_i>0,
$$

`BuildIndex` 在 $O(k)$ 时间和 $O(k)$ 空间内构建一个精确整数 alias 索引 [6]，使每次 `SampleIndex()` 返回标签 $i$ 的概率为

$$
\frac{w_i}{W}.
$$

具体地，把每个标签的缩放质量写为 $k w_i$，把每个 alias 槽的容量写为 $W$。标准 small/large 配对过程在线性时间内为每个槽构造整数阈值 $\theta_i\in[0,W]$ 和 alias 标签 $a_i$。一次查询独立生成

$$
I\sim\operatorname{Unif}\{1,\ldots,k\},
\qquad
R\sim\operatorname{Unif}\{0,\ldots,W-1\},
$$

若 $R<\theta_I$ 则返回 $I$，否则返回 $a_I$。全部比较均为整数比较，因此概率严格为 $w_i/W$；零权重标签永不被返回。

其正确性来自质量守恒不变式：每个 alias 槽的总质量始终为 $W$；处理一个 small 标签时，把其剩余质量放入自身槽，并用一个 large 标签补足该槽的亏空，随后从 large 标签的剩余质量中减去同样的量。因而构造结束后，标签 $i$ 分布在全部 $k$ 个槽中的总整数质量仍为 $kw_i$。均匀选择一个槽并在容量 $W$ 内均匀选择整数位置后，标签 $i$ 的概率为

$$
\frac{kw_i}{kW}=\frac{w_i}{W}.
$$

理论表中的 $O(1)$ 加权抽样时间采用这一精确整数 alias 索引。若改用精确前缀和与二分搜索，则分布完全相同，但在 $k$ 个标签上每次抽样需要 $O(\log(k+1))$ 时间。该替代只改变相关采样步骤的时间，不改变正确性。

### 3.3 多项式配额

给定 $h\ge0$ 与权重 $w_1,\ldots,w_k$，$W=\sum_iw_i>0$，`DrawQuotas` 返回

$$
(H_1,\ldots,H_k)\sim
\operatorname{Mult}\left(h;\frac{w_1}{W},\ldots,\frac{w_k}{W}\right).
$$

可以通过构建一次精确加权索引、执行 $h$ 次独立加权抽样、并统计各标签出现次数实现，时间 $O(k+h)$，空间 $O(k)$。

### 3.4 洗牌

Fisher–Yates 洗牌在线性时间内对有限序列生成均匀随机排列。AS 中，递归子块按实现顺序生成输出，最终洗牌用于恢复目标有序 i.i.d. 序列的可交换分布。配额、不同递归子块内部的抽样以及洗牌使用彼此独立的随机性。

## 4. 复杂度记号

原始输入规模与过滤后规模分别为

$$
N_0=|\mathcal R_0|+|\mathcal S_0|,
\qquad
N=|\mathcal R|+|\mathcal S|\le N_0.
$$

读取、检查并删除空盒需要 $O(N_0)$ 时间；过滤可以原地完成，也可以写入大小为 $O(N)$ 的紧凑数组。所有几何索引、排序视图和递归复杂度均以过滤后规模 $N$ 表示。除特别说明外，维度 $d$ 为常数，所有 $O(\cdot)$ 隐含常数可依赖于 $d$。

表中各列采用以下起始状态：`CountTime` 包含原始输入读取与过滤，并构造连接计数或算法所需的全局权重；LiftedRT 的 `SampleTime` 从静态 range tree、所有度数和外层索引已经构造完毕开始，SweepRT 的 `SampleTime` 包含给定第一遍权重后的第二遍重放，AS 的 `SampleTime` 是不持久保存递归权重的一次完整 `Sample` 调用，AC 的 `SampleTime` 从编译结构和全局原子索引已经构造完毕开始。`E2ETime` 包含空盒过滤、初始排序或索引构造以及一次长度为 $t$ 的采样。空间复杂度以 RAM 字为单位，不计调用方持有的只读原始输入 $O(N_0)$；它包含过滤后的紧凑输入 $O(N)$、算法持久结构、工作缓冲区和输出序列 $O(t)$，其中紧凑输入空间被表中上界吸收。

对数统一写作 $\log(N+1)$。这既覆盖 $N=0,1$ 的边界，也不改变 $N\ge2$ 时的通常渐近含义。当 $d=1$ 时，AC/AS 的递归指数 $d-1$ 为 $0$，但过滤后输入仍需构造排序视图，成本为 $O(N\log(N+1))$。SweepRT 在 $d=1$ 时剩余维度为空，range-tree 部分退化为活跃对侧集合上的计数与均匀采样，但事件排序成本 $O(N\log(N+1))$ 必须计入端到端时间。

## 5. 四个算法的主结果

下表给出 $d\ge2$ 时从原始输入开始的复杂度。所有 $N_0$ 项来自读取和过滤，所有数据结构项以过滤后的 $N$ 表示。

| 算法 | CountTime | SampleTime | E2ETime | 空间复杂度 |
|---|---:|---:|---:|---:|
| **LiftedRT** | $O\!\left(N_0+N\log^{2d}(N+1)\right)$ | $O\!\left(t+\min\{N,t\}\log^{2d}(N+1)\right)$ | $O\!\left(N_0+N\log^{2d}(N+1)+t\right)$ | $O\!\left(N\log^{2d-1}(N+1)+t\right)$ |
| **SweepRT** | $O\!\left(N_0+N\log^{2d-2}(N+1)\right)$ | $O\!\left(N\log^{2d-2}(N+1)+t\log(N+1)\right)$ | $O\!\left(N_0+N\log^{2d-2}(N+1)+t\log(N+1)\right)$ | $O\!\left(N\log^{2d-3}(N+1)+N+t\right)$ |
| **AS** | $O\!\left(N_0+N\log^{d-1}(N+1)\right)$ | $O\!\left(N\log^{d-1}(N+1)+t\right)$ | $O\!\left(N_0+N\log^{d-1}(N+1)+t\right)$ | $O(N+t)$ |
| **AC** | $O\!\left(N_0+N\log^{d-1}(N+1)\right)$ | $O(t)$ | $O\!\left(N_0+N\log^{d-1}(N+1)+t\right)$ | $O\!\left(N\log^{d-1}(N+1)+t\right)$ |

AC 行的 `CountTime` 指完整编译时间，包括终端原子物化和全局精确加权索引构造；AS 行的 `CountTime` 指独立的连接计数调用。AS 的一次端到端采样可以直接运行 `Sample`，不要求先额外执行一遍 `Count`。

当 $d=1$ 时：

- AC/AS 从过滤后输入构造排序视图需要 $O(N\log(N+1))$ 时间。排序视图已给定后，AS 的终端原子构造与采样为 $O(N+t)$；AC 在预先保存正原子及其全局精确加权索引后，查询为 $O(t)$。两者的原始输入端到端时间均为 $O(N_0+N\log(N+1)+t)$。
- SweepRT 的事件排序为 $O(N\log(N+1))$。使用第 15.4 节给出的可随机访问活跃数组时，两遍扫描和活跃集合采样为 $O(N+t)$，因此原始输入端到端时间为 $O(N_0+N\log(N+1)+t)$，空间为 $O(N+t)$。
- LiftedRT 仍可直接使用 $D=2$ 的静态 range tree；其一般公式在 $d=1$ 时继续成立，即 CountTime 为 $O(N_0+N\log^2(N+1))$，空间为 $O(N\log(N+1)+t)$。

若过滤后任一颜色集合为空，则四个算法均在过滤后直接记录 $|J|=0$。对 $t>0$，端到端时间为 $O(N_0)$，返回 `EMPTY-INSTANCE`；对 $t=0$，可以直接返回空序列而不构造任何索引。

这四个算法都返回 $J$ 上的精确带放回 i.i.d. 均匀样本。AC 和 AS 使用锚定所有权分解，维度指数为 $d-1$；SweepRT 和 LiftedRT 使用 range tree，将盒相交转化为正交范围查询，因此维度指数分别为 $2d-2$ 和 $2d$。

---

# Part II. ANCHOR-Based Algorithms：AC 与 AS

ANCHOR 的核心是按维度递归地分解连接。每一层只负责认证当前维度的相交关系，并把剩余维度交给后缀递归。当前维度的分解由两个规则组成：一是锚定所有权规则，二是对侧秩区间的规范 dyadic 覆盖。最终到达一维时，连接被表示为互不相交的终端原子。

## 6. 后缀连接与排序视图

对任意两个局部颜色集合 $A,B$，定义从维度 $\ell$ 开始的后缀连接：

$$
J^{(\ell)}(A,B)=
\{(a,b)\in A\times B:
 a,b\text{ 在每个维度 }\ell,\ell+1,\ldots,d\text{ 上相交}\}.
$$

顶层连接为 $J^{(1)}(\mathcal R,\mathcal S)$。层级 $\ell$ 的局部实例称为前缀已认证，如果 $A\times B$ 中每一对都已经在维度 $1,\ldots,\ell-1$ 上相交。顶层实例在真空意义下前缀已认证。ANCHOR 在层级 $\ell$ 构造的每个局部块都会在当前维度 $\ell$ 上认证其完整笛卡尔积，因此递归到 $\ell+1$ 后继续保持前缀已认证。

层级 $\ell$ 的排序视图记为

$$
\mathfrak V_\ell(A,B)=
(A^{L_\ell},A^{L_{\ell+1}},\ldots,A^{L_d},A^{U_d};
 B^{L_\ell},B^{L_{\ell+1}},\ldots,B^{L_d},B^{U_d}).
$$

其中 $A^{L_j}$ 表示 $A$ 按 $L_j$ 稳定排序得到的列表，$A^{U_d}$ 表示 $A$ 按 $U_d$ 稳定排序得到的列表；$B$ 侧类似。对于 $\ell<d$，当前层只需要在对侧 $L_\ell$ 排序视图中对端点 $U_\ell(\cdot)$ 做秩查询，因此不需要排序的 $U_\ell$ 视图。终端层 $\ell=d$ 使用 $U_d$ 排序视图进行一维扫描。

对一侧 $Y$，定义

$$
\operatorname{pos}^{(\ell)}_Y(y)
$$

为对象 $y$ 在 $Y^{L_\ell}$ 中的零基位置。对阈值 $\alpha$，定义

$$
\operatorname{lb}^{(\ell)}_Y(\alpha)=|\{y\in Y:L_\ell(y)<\alpha\}|,
$$

$$
\operatorname{ub}^{(\ell)}_Y(\alpha)=|\{y\in Y:L_\ell(y)\le\alpha\}|.
$$

当维度明确时省略上标。所有秩值都相对于当前局部视图计算，而不是相对于全局输入数组计算。该局部性是 ANCHOR 正确性的必要条件：同一对象可能因前面维度的所有权分解出现在多个局部实例中，并在不同局部实例中具有不同的局部秩；终端层也必须使用对应的终端局部数组解码原子。

## 7. 锚定所有权分解

固定一个两侧均非空的非终端后缀实例 $J^{(\ell)}(A,B)$，其中 $\ell<d$。一侧为空的实例由递归入口直接返回，不进入本节分解。当前维度为 $\ell$。ANCHOR 规定：当 $L_\ell(a)=L_\ell(b)$ 时，该对由 $A$ 侧锚拥有。严格较大的左端点情形由另一侧锚拥有。

### 7.1 后继秩区间

对 $a\in A$，定义 $B$ 侧后继区间

$$
X_B^{(\ell)}(a)=
\left[
\operatorname{lb}_B^{(\ell)}(L_\ell(a)),
\operatorname{lb}_B^{(\ell)}(U_\ell(a))
\right).
$$

该区间恰好包含所有满足

$$
L_\ell(a)\le L_\ell(b)<U_\ell(a)
$$

的 $b\in B$。

对 $b\in B$，定义 $A$ 侧后继区间

$$
X_A^{(\ell)}(b)=
\left[
\operatorname{ub}_A^{(\ell)}(L_\ell(b)),
\operatorname{lb}_A^{(\ell)}(U_\ell(b))
\right).
$$

该区间恰好包含所有满足

$$
L_\ell(b)<L_\ell(a)<U_\ell(b)
$$

的 $a\in A$。这里 $\operatorname{ub}_A(L_\ell(b))$ 排除了相等左端点的 $A$ 对象，与 $A$ 侧拥有并列情形的规则一致。

若 $L_\ell(a)\le L_\ell(b)<U_\ell(a)$，则当前维度的另一个相交不等式 $L_\ell(a)<U_\ell(b)$ 由 $b$ 的非退化性 $L_\ell(b)<U_\ell(b)$ 推出。对称情形相同。因此当前维度只需检查后继左端点是否落入锚的开右秩区间。

### 7.2 规范 dyadic 覆盖

对非空一侧 $Y\in\{A,B\}$，在 $Y^{L_\ell}$ 的位置 $0,1,\ldots,|Y|-1$ 上构建一棵带填充的满二叉线段树 $T_Y^{(\ell)}$。若 $|Y|$ 不是 $2$ 的幂，则添加虚拟叶子到下一个 $2$ 的幂。虚拟叶子不对应任何对象。若某一侧为空，当前局部连接为空，算法在建树前已经返回。

一个节点 $v$ 表示真实区间

$$
I_Y(v)=[\lambda(v),\rho(v))\cap[0,|Y|),
$$

其中 $[\lambda(v),\rho(v))$ 是填充树中的 dyadic 区间。忽略 $I_Y(v)=\emptyset$ 的节点。

对任意秩区间 $X=[p,q)\subseteq[0,|Y|)$，定义 $\operatorname{Can}_Y(X)$ 为所有满足下列条件的节点 $v$ 的集合：

1. $I_Y(v)\ne\emptyset$；
2. $I_Y(v)\subseteq X$；
3. 不存在 $v$ 的真祖先 $v'$ 满足 $I_Y(v')\ne\emptyset$ 且 $I_Y(v')\subseteq X$。

最大性按树祖先关系定义。该定义在填充树上仍唯一：若不同填充节点诱导同一个截断后的真实区间，只有祖先最高且包含于 $X$ 的节点会被选入规范覆盖。

**引理 7.1（规范覆盖性质）。** 对每个秩区间 $X=[p,q)\subseteq[0,|Y|)$，区间族

$$
\{I_Y(v):v\in\operatorname{Can}_Y(X)\}
$$

两两不相交，并且其并集为 $X$。此外，

$$
|\operatorname{Can}_Y(X)|=O(\log(|Y|+1)),
$$

并且在固定树深度上，$X$ 贡献 $O(1)$ 个被选中的规范节点和 $O(1)$ 个仍需继续向下路由的边界片段。

**证明。** 任意位置 $r\in X$ 位于填充树中唯一一条根到叶路径上。包含 $r$ 的真实叶区间包含于 $X$，所以该路径上至少有一个非空真实区间包含于 $X$ 的节点。取其中最高者，它按定义属于 $\operatorname{Can}_Y(X)$，因此 $r$ 被覆盖。

若两个被选节点存在祖先—后代关系，则后代节点的真祖先也包含于 $X$，与后代的最大性矛盾。因此任意两个被选节点在树上不可比。二叉线段树中不可比节点的 dyadic 区间不相交，截断后的真实区间也不相交。

在固定深度上，与 $X$ 相交的 dyadic 区间形成一个连续段。除接触 $X$ 左边界和右边界的区间外，中间区间要么完全包含于 $X$ 并在该深度被选中，要么已经位于更高层某个被选节点之下。于是每个深度只有常数个被选规范节点和常数个边界片段。树高为 $O(\log(|Y|+1))$，引理成立。$\square$

### 7.3 带标签局部块

两棵当前规范树形成一个带标签的不交并：

$$
\mathcal Z_\ell(A,B)=
(\{\mathsf B\}\times T_B^{(\ell)})
\sqcup
(\{\mathsf A\}\times T_A^{(\ell)}).
$$

标签是数学对象的一部分；$A$ 树节点和 $B$ 树节点即使编号相同也不是同一个局部块。

对 $B$ 树带标签节点 $z=(\mathsf B,v)$，定义

$$
A_z^{(\ell)}=
\{a\in A:v\in\operatorname{Can}_B(X_B^{(\ell)}(a))\},
$$

$$
B_z^{(\ell)}=
\{b\in B:\operatorname{pos}_B^{(\ell)}(b)\in I_B(v)\}.
$$

对 $A$ 树带标签节点 $z=(\mathsf A,u)$，定义

$$
A_z^{(\ell)}=
\{a\in A:\operatorname{pos}_A^{(\ell)}(a)\in I_A(u)\},
$$

$$
B_z^{(\ell)}=
\{b\in B:u\in\operatorname{Can}_A(X_A^{(\ell)}(b))\}.
$$

同一个对象可能出现在多个局部块中；互不相交的是连接对的归属，而不是对象集合本身。

**引理 7.2（局部块前缀认证）。** 若 $(A,B)$ 在层级 $\ell$ 前缀已认证，则每个局部实例 $(A_z^{(\ell)},B_z^{(\ell)})$ 在层级 $\ell+1$ 前缀已认证。

**证明。** 维度 $1,\ldots,\ell-1$ 的相交认证由包含关系继承。只需证明当前维度 $\ell$。

若 $z=(\mathsf B,v)$，则对任意 $a\in A_z^{(\ell)}$ 有 $I_B(v)\subseteq X_B^{(\ell)}(a)$；对任意 $b\in B_z^{(\ell)}$ 有 $\operatorname{pos}_B^{(\ell)}(b)\in I_B(v)$。因此

$$
L_\ell(a)\le L_\ell(b)<U_\ell(a).
$$

又因为 $b$ 在该维度非退化，$L_\ell(b)<U_\ell(b)$，于是 $L_\ell(a)<U_\ell(b)$。所以 $a,b$ 在维度 $\ell$ 相交。

若 $z=(\mathsf A,u)$，同理可得

$$
L_\ell(b)<L_\ell(a)<U_\ell(b).
$$

结合 $a$ 的非退化性 $L_\ell(a)<U_\ell(a)$，得到 $L_\ell(b)<U_\ell(a)$。因此当前维度也已认证。$\square$

**定理 7.3（锚定规范分解）。** 对任意满足 $A\ne\emptyset$、$B\ne\emptyset$ 的后缀实例 $(A,B)$ 和任意 $\ell<d$，有

$$
J^{(\ell)}(A,B)=
\biguplus_{z\in\mathcal Z_\ell(A,B)}
J^{(\ell+1)}\bigl(A_z^{(\ell)},B_z^{(\ell)}\bigr),
$$

其中空局部实例可省略。

**证明。** 取任意 $(a,b)\in J^{(\ell)}(A,B)$。该对在维度 $\ell$ 相交，所以

$$
L_\ell(a)<U_\ell(b),\qquad L_\ell(b)<U_\ell(a).
$$

若 $L_\ell(a)\le L_\ell(b)$，则

$$
\operatorname{pos}_B^{(\ell)}(b)\in X_B^{(\ell)}(a).
$$

由引理 7.1，$\operatorname{Can}_B(X_B^{(\ell)}(a))$ 将该秩区间不交覆盖。因此存在唯一节点 $v\in\operatorname{Can}_B(X_B^{(\ell)}(a))$ 使 $\operatorname{pos}_B^{(\ell)}(b)\in I_B(v)$。于是该对出现在带标签节点 $z=(\mathsf B,v)$ 的局部实例中。唯一性来自规范覆盖的不交性；它不会出现在 $A$ 树项中，因为 $X_A^{(\ell)}(b)$ 只包含左端点严格大于 $L_\ell(b)$ 的 $A$ 对象。

若 $L_\ell(b)<L_\ell(a)$，对称地存在唯一 $u\in\operatorname{Can}_A(X_A^{(\ell)}(b))$，使得 $\operatorname{pos}_A^{(\ell)}(a)\in I_A(u)$。该对唯一归入 $z=(\mathsf A,u)$，且不会归入 $B$ 树项。

反过来，若 $(a,b)$ 出现在某个 $B$ 树局部实例中，则 $I_B(v)\subseteq X_B^{(\ell)}(a)$ 且 $b$ 的秩位于 $I_B(v)$，故

$$
L_\ell(a)\le L_\ell(b)<U_\ell(a).
$$

结合 $b$ 的非退化性得到 $L_\ell(a)<U_\ell(b)$，所以维度 $\ell$ 相交。$A$ 树情形对称。局部递归项 $J^{(\ell+1)}$ 负责且仅负责剩余维度 $\ell+1,\ldots,d$，因此这些局部后缀连接的不交并正好等于 $J^{(\ell)}(A,B)$。$\square$

## 8. 一维终端原子

当 $\ell=d$ 时，后缀连接是一维连接，可以直接表示为互不相交的原子。并列规则仍然由 $A$ 侧拥有相等左端点的对。

### 8.1 原子定义

对每个 $a\in A$，定义 $A$ 锚定原子

$$
\mathcal A_d(a)=
\{(a,b):b\in B,\ L_d(a)\le L_d(b)<U_d(a)\}.
$$

其基数为

$$
\alpha(a)=
\operatorname{lb}_B^{(d)}(U_d(a))-
\operatorname{lb}_B^{(d)}(L_d(a)).
$$

对每个 $b\in B$，定义 $B$ 锚定原子

$$
\mathcal B_d(b)=
\{(a,b):a\in A,\ L_d(b)<L_d(a)<U_d(b)\}.
$$

其基数为

$$
\beta(b)=
\operatorname{lb}_A^{(d)}(U_d(b))-
\operatorname{ub}_A^{(d)}(L_d(b)).
$$

任意一维相交对要么满足 $L_d(a)\le L_d(b)<U_d(a)$，要么满足 $L_d(b)<L_d(a)<U_d(b)$，二者互斥且覆盖全部一维相交对。因此

$$
J^{(d)}(A,B)=
\left(\biguplus_{a\in A}\mathcal A_d(a)\right)
\uplus
\left(\biguplus_{b\in B}\mathcal B_d(b)\right).
$$

设

$$
W_d=\sum_{a\in A}\alpha(a)+\sum_{b\in B}\beta(b),
$$

则 $W_d=|J^{(d)}(A,B)|$。

一个正原子记录为

$$
\tau=(\mathrm{side},\mathrm{anchor},\mathrm{opp},lo,hi,\omega),
$$

其中 $\mathrm{side}\in\{A,B\}$ 表示锚所在侧，$\mathrm{anchor}$ 是锚对象，$\mathrm{opp}$ 指向当前终端局部实例的对侧 $L_d$ 排序数组，$[lo,hi)$ 是该局部数组中的秩区间，$\omega=hi-lo$ 是原子权重。该指针指向终端局部数组，不指向全局 $L_d$ 数组。

记 $\Omega_\tau$ 为原子记录 $\tau$ 解码出的连接对集合：当 $\mathrm{side}=A$ 时，

$$
\Omega_\tau=
\{(\mathrm{anchor},\mathrm{opp}[j]):lo\le j<hi\};
$$

当 $\mathrm{side}=B$ 时，

$$
\Omega_\tau=
\{(\mathrm{opp}[j],\mathrm{anchor}):lo\le j<hi\}.
$$

因此 $|\Omega_\tau|=\omega$，且输出方向始终为 $(A,B)$。

### 8.2 终端原子计算

**算法：ComputeBaseAtoms$_d(\mathfrak V_d(A,B))$**

1. 为每个 $a\in A$ 准备 $lo_B[a],hi_B[a]$，为每个 $b\in B$ 准备 $lo_A[b],hi_A[b]$。
2. 按递增 $L_d(a)$ 扫描 $A^{L_d}$，同时在 $B^{L_d}$ 中推进指针越过所有满足 $L_d(b)<L_d(a)$ 的对象，设置

$$
lo_B[a]=\operatorname{lb}_B^{(d)}(L_d(a)).
$$

3. 按递增 $U_d(a)$ 扫描 $A^{U_d}$，同时在 $B^{L_d}$ 中推进指针越过所有满足 $L_d(b)<U_d(a)$ 的对象，设置

$$
hi_B[a]=\operatorname{lb}_B^{(d)}(U_d(a)).
$$

4. 按递增 $L_d(b)$ 扫描 $B^{L_d}$，同时在 $A^{L_d}$ 中推进指针越过所有满足 $L_d(a)\le L_d(b)$ 的对象，设置

$$
lo_A[b]=\operatorname{ub}_A^{(d)}(L_d(b)).
$$

5. 按递增 $U_d(b)$ 扫描 $B^{U_d}$，同时在 $A^{L_d}$ 中推进指针越过所有满足 $L_d(a)<U_d(b)$ 的对象，设置

$$
hi_A[b]=\operatorname{lb}_A^{(d)}(U_d(b)).
$$

6. 对每个满足 $hi_B[a]>lo_B[a]$ 的 $a$，输出原子

$$
(A,a,B^{L_d},lo_B[a],hi_B[a],hi_B[a]-lo_B[a]).
$$

7. 对每个满足 $hi_A[b]>lo_A[b]$ 的 $b$，输出原子

$$
(B,b,A^{L_d},lo_A[b],hi_A[b],hi_A[b]-lo_A[b]).
$$

8. 返回正原子列表 $\mathcal T_d$ 与总权重

$$
W_d=\sum_{\tau\in\mathcal T_d}\omega_\tau.
$$

四个边界都由单调扫描得到，因此 ComputeBaseAtoms$_d$ 的时间为 $O(|A|+|B|)$，临时空间为 $O(|A|+|B|)$，输出正原子数量至多为 $|A|+|B|$。

**算法：BaseCount$_d(\mathfrak V_d(A,B))$**

1. 运行 ComputeBaseAtoms$_d(\mathfrak V_d(A,B))$。
2. 返回总权重 $W_d$。

**算法：BaseSample$_d(\mathfrak V_d(A,B),h)$**

1. 若 $h=0$，返回空序列。
2. 运行 ComputeBaseAtoms$_d$，得到 $\mathcal T_d$ 和 $W_d$。
3. 若 $W_d=0$，返回 `EMPTY-INSTANCE`。
4. 在原子权重 $\omega_\tau$ 上构建精确加权索引。
5. 对 $r=1,\ldots,h$：
   1. 以概率 $\omega_\tau/W_d$ 抽取原子 $\tau$；
   2. 从 $\{lo,lo+1,\ldots,hi-1\}$ 中均匀抽取偏移 $j$；
   3. 若 $\mathrm{side}=A$，输出 $(\mathrm{anchor},\mathrm{opp}[j])$；若 $\mathrm{side}=B$，输出 $(\mathrm{opp}[j],\mathrm{anchor})$。
6. 返回长度为 $h$ 的有序序列。

**引理 8.1（终端采样正确性）。** BaseCount$_d$ 返回 $|J^{(d)}(A,B)|$。若 $h>0$ 且 $J^{(d)}(A,B)\ne\emptyset$，BaseSample$_d$ 返回 $h$ 个来自 $J^{(d)}(A,B)$ 的 i.i.d. 均匀样本。若 $h>0$ 且 $J^{(d)}(A,B)=\emptyset$，返回 `EMPTY-INSTANCE`。

**证明。** 上述原子族是不交覆盖，且每个原子权重等于其包含的连接对数量，所以 BaseCount$_d$ 返回连接基数。对任意 $p\in\Omega_\tau$，一次采样返回 $p$ 的概率为

$$
\Pr[p]=\frac{\omega_\tau}{W_d}\cdot\frac1{\omega_\tau}=\frac1{W_d}.
$$

每个输出位置重新独立抽取原子和偏移，因此输出序列由 $h$ 个独立均匀样本组成。$\square$

## 9. RouteTree：流式构造局部视图

AS 不存储完整递归分解，而是在计数或采样时按需路由当前层对象。RouteTree 是实现定理 7.3 的流式原语。

### 9.1 路由方向

在 $B$ 树方向中，源侧为 $A$，后继侧为 $B$。源对象 $a\in A$ 携带区间 $X_B^{(\ell)}(a)$，后继对象 $b\in B$ 携带其在 $B^{L_\ell}$ 中的位置。

在 $A$ 树方向中，源侧为 $B$，后继侧为 $A$。源对象 $b\in B$ 携带区间 $X_A^{(\ell)}(b)$，后继对象 $a\in A$ 携带其在 $A^{L_\ell}$ 中的位置。回调时再把两侧映射回标准顺序 $(A_z,B_z)$。

令下一层需要的排序键集合为

$$
\mathcal K_{\ell+1}=\{L_{\ell+1},L_{\ell+2},\ldots,L_d,U_d\}.
$$

对每个 $K\in\mathcal K_{\ell+1}$，RouteTree 同时维护按 $K$ 排序的源列表和后继列表。

### 9.2 完整路由语义

对某一方向，记源侧为 $S$，后继侧为 $Y$。源对象 $s\in S$ 携带秩区间 $X(s)=[lo(s),hi(s))$；空区间源不参与路由。后继对象 $y\in Y$ 携带位置 $\operatorname{pos}_Y^{(\ell)}(y)$。

在完整路由中，节点 $v$ 处维护：

- 后继列表 $\mathsf{Lat}_K(v)$：位置落在 $I_Y(v)$ 的后继对象，并按键 $K$ 稳定排序。
- 未停止源列表 $\mathsf{Src}_K(v)$：满足 $X(s)\cap I_Y(v)\ne\emptyset$ 且尚未在 $v$ 的任何真祖先处停止的源对象，并按键 $K$ 稳定排序。
- 停止源列表 $\mathsf{Stop}_K(v)$：满足 $I_Y(v)\subseteq X(s)$ 的未停止源对象，并按键 $K$ 稳定排序。

源对象在一条根到叶路径上采用“首次可停止”规则：一旦到达最高的满足 $I_Y(v)\subseteq X(s)$ 的节点 $v$，该源在 $v$ 停止，不再向后代传播。等价地，

$$
s\in\mathsf{Stop}(v)
\quad\Longleftrightarrow\quad
v\in\operatorname{Can}_Y(X(s)).
$$

若 $I_Y(v)\nsubseteq X(s)$，则 $s$ 被发送到所有满足 $I_Y(c)\cap X(s)\ne\emptyset$ 的子节点 $c$。最多两个子节点接收该源。后继对象不复制，只进入包含其位置的唯一子节点。所有分发均稳定，因此从父列表继承的按键顺序在子列表中保持。

### 9.3 活动子树与请求节点

RouteTree 接收两个谓词：

- $\mathsf{Active}(v)$：是否处理节点 $v$ 及其子树；
- $\mathsf{Request}(v)$：是否在节点 $v$ 向回调交付局部视图。

这两个谓词必须满足以下接口条件：

1. $\mathsf{Request}(v)\Rightarrow\mathsf{Active}(v)$。
2. 若 $\mathsf{Active}(v)$ 且 $v$ 不是根，则 $v$ 的父节点也为 active。即活动节点形成祖先闭合集。
3. 即使某个 active 节点没有被 request，仍然执行该节点处的停止判定；停止源不会继续向后代传播。

计数和编译时，所有非空真实节点 active；节点完成停止与后继划分后，若停止源侧和后继侧均非空，则 request。采样时，先由正配额计算子树配额 $Q_v$，令 $\mathsf{Active}(v)$ 当且仅当 $Q_v>0$，令 $\mathsf{Request}(v)$ 当且仅当节点自身配额 $H_v>0$。由于

$$
Q_v=H_v+\sum_{c\text{ 为 }v\text{ 的子节点}}Q_c,
$$

采样时的活动节点自动形成祖先闭合集，并且 $H_v>0$ 蕴含 $Q_v>0$。

**算法：RouteTree$_\ell(S,Y,X,\mathrm{orient},\mathsf{Active},\mathsf{Request},\mathsf{Callback})$**

1. 在 $Y^{L_\ell}$ 的位置上构建带填充线段树。
2. 对每个 $K\in\mathcal K_{\ell+1}$，根后继列表初始化为 $Y^K$，根源列表初始化为 $S^K$ 中非空区间源的稳定限制。
3. 按深度逐层处理节点。若 $\mathsf{Active}(v)=\mathrm{false}$，跳过整个子树。
4. 对每个活动节点 $v$：
   1. 稳定扫描源列表，形成停止源列表和子节点源列表；
   2. 稳定扫描后继列表，形成左右子节点后继列表；
   3. 若 $\mathsf{Request}(v)=\mathrm{true}$，则将停止源列表和当前后继列表按方向映射为 $\mathfrak V_{\ell+1}(A_z,B_z)$ 并调用回调；
   4. 只保留活动子节点的列表。
5. 完成当前深度后释放上一深度列表。

RouteTree 的回调采用同步、串行语义。对节点 $v$ 调用回调时，当前节点的 $\mathsf{Stop}_K(v)$ 与 $\mathsf{Lat}_K(v)$ 在所有 $K\in\mathcal K_{\ell+1}$ 上均保持只读有效；回调返回后，RouteTree 才可复用或释放这些列表的底层缓冲区。子节点列表写入下一深度缓冲区，或由任何不破坏当前节点列表的等价所有权机制产生。同一深度的不同节点按任意确定顺序逐个执行回调，不并发保留多个递归子调用。

除编译终端数组的情形外，回调接收的局部视图只在该次回调的动态作用域内有效。若回调需要在返回后继续引用某个局部数组，则必须在返回前取得该数组的持久所有权；第 11.1 节对 AC 的终端数组采用这一语义。

带标签树节点以 $(\mathrm{orient},\lambda(v),\rho(v))$ 标识，而不以临时内存地址标识。同一局部实例上重复执行 RouteTree 时，带填充树和节点标识完全一致，因此 NodeWeights 得到的 $H_z,Q_z$ 可以无歧义地用于第二次活动路由。

**引理 9.1（局部视图不变式）。** 每当 RouteTree$_\ell$ 在带标签节点 $z$ 调用回调时，传递的视图正好是

$$
\mathfrak V_{\ell+1}(A_z^{(\ell)},B_z^{(\ell)}).
$$

其中每个局部列表都按对应键稳定排序。

**证明。** 先考虑完整路由，再考虑活动子树限制。

完整路由中，对树深度归纳。根节点处，源列表和后继列表分别是对应当前局部视图的稳定限制。假设节点 $v$ 处不变式成立。后继对象按当前位置进入唯一子区间，稳定分发后子列表恰好是该子区间上的后继排序视图。源对象 $s$ 若满足 $I_Y(v)\subseteq X(s)$，则因为 $s$ 尚未在任何祖先停止，$v$ 是从根到当前位置遇到的最高包含节点；这等价于 $v\in\operatorname{Can}_Y(X(s))$。若 $I_Y(v)\nsubseteq X(s)$，则 $s$ 不能在 $v$ 停止，只需沿与 $X(s)$ 相交的边界子区间继续。由此，$\mathsf{Stop}(v)$ 正好等于所有规范覆盖包含 $v$ 的源对象集合，$\mathsf{Lat}(v)$ 正好等于 $I_Y(v)$ 中的后继对象集合。

不同排序键列表由同一对象集合经稳定分发得到，因此包含的对象集合一致，并分别保持按键排序。应用方向映射后，$B$ 树方向得到 $(A_z^{(\ell)},B_z^{(\ell)})$，$A$ 树方向得到同样标准顺序的 $(A_z^{(\ell)},B_z^{(\ell)})$。

若只处理活动子树，由接口条件，任意 requested 节点的所有祖先均 active，并且这些祖先上的停止判定照常执行。因此 requested 节点处的列表与完整路由在该节点处的列表完全一致。故回调视图即为 $\mathfrak V_{\ell+1}(A_z^{(\ell)},B_z^{(\ell)})$。$\square$

**引理 9.2（每层路由体积）。** 对大小 $m=|A|+|B|$ 的当前后缀实例，在任意固定深度上，所有源记录、停止源记录、后继记录和回调局部记录的总数为 $O(m)$。

**证明。** 固定一个排序键。每个后继对象在固定深度上只出现在一个节点中。对任意源区间 $X=[p,q)$，尚未停止的副本只能位于接触 $p$ 或 $q$ 的边界节点中，因此每个深度最多贡献常数个未停止副本。该深度上的停止副本是 $X$ 的规范覆盖节点，由引理 7.1 也只有常数个。回调局部记录由停止源记录和当前后继记录组成，在该深度上不超过同阶路由体积。排序键数量为 $O(d)$，而 $d$ 为常数，结论成立。$\square$

## 10. AS：流式计数与递归采样

### 10.1 计数

**算法：Count$_\ell(\mathfrak V_\ell(A,B))$**

1. 若 $A=\emptyset$ 或 $B=\emptyset$，返回 $0$。
2. 若 $\ell=d$，返回 BaseCount$_d(\mathfrak V_d(A,B))$。
3. 对所有 $a\in A$ 计算 $X_B^{(\ell)}(a)$，对所有 $b\in B$ 计算 $X_A^{(\ell)}(b)$。所有端点秩都相对于当前局部对侧的 $L_\ell$ 排序视图，通过二分搜索精确得到；严格边界与非严格边界分别按 $\operatorname{lb}$ 和 $\operatorname{ub}$ 的定义处理。
4. 令 $W\leftarrow0$。
5. 在 $B$ 树方向运行 RouteTree$_\ell$，所有真实节点 active。节点 $z$ 完成停止与后继划分后，若停止源侧和后继侧均非空，则 request。对每个 requested 节点 $z$，令

$$
w_z=\mathrm{Count}_{\ell+1}(\mathfrak V_{\ell+1}(A_z,B_z)),
$$

并将 $w_z$ 加入 $W$。

6. 在 $A$ 树方向运行同样过程，将递归权重加入 $W$。
7. 返回 $W$。

**引理 10.1（计数正确性）。** 对每个后缀实例，Count$_\ell(\mathfrak V_\ell(A,B))$ 返回 $|J^{(\ell)}(A,B)|$。

**证明。** 对剩余维度数量归纳。基准层由引理 8.1 给出。对 $\ell<d$，RouteTree 由引理 9.1 精确生成每个 requested 局部视图。未被 request 的节点至少停止源侧或后继侧为空，其局部笛卡尔积为空，因而局部后缀连接为空。由归纳假设，回调中递归返回的值为

$$
|J^{(\ell+1)}(A_z^{(\ell)},B_z^{(\ell)})|.
$$

定理 7.3 将 $J^{(\ell)}(A,B)$ 表示为这些局部后缀连接的不交并，因此求和得到精确连接基数。$\square$

### 10.2 节点权重

**算法：NodeWeights$_\ell(\mathfrak V_\ell(A,B))$**

1. 计算所有当前层区间 $X_B^{(\ell)}(a)$ 和 $X_A^{(\ell)}(b)$。
2. 为 $\mathcal Z_\ell(A,B)$ 中所有带标签真实节点初始化 $w_z\leftarrow0$。
3. 在 $B$ 树方向运行全活动 RouteTree。节点 $z$ 完成停止与后继划分后，若停止源侧和后继侧均非空，则 request，并设置

$$
w_z\leftarrow \mathrm{Count}_{\ell+1}(\mathfrak V_{\ell+1}(A_z,B_z)).
$$

4. 在 $A$ 树方向运行同样过程。
5. 返回区间、节点权重数组 $(w_z)_z$ 与

$$
W=\sum_{z\in\mathcal Z_\ell(A,B)}w_z.
$$

**引理 10.2（节点权重正确性）。** NodeWeights$_\ell$ 返回的每个 $w_z$ 等于局部后缀连接大小

$$
|J^{(\ell+1)}(A_z^{(\ell)},B_z^{(\ell)})|,
$$

且总和 $W=|J^{(\ell)}(A,B)|$。

**证明。** 每个 requested 局部视图由引理 9.1 保证正确，递归 Count 的值由引理 10.1 保证正确。未被 request 的节点至少停止源侧或后继侧为空，其局部连接大小为零；对应 $w_z$ 保持为零。总和等于当前后缀连接大小由定理 7.3 的不交分解推出。$\square$

### 10.3 加权配额组合

**引理 10.3（加权配额组合）。** 令

$$
\Omega=\biguplus_{i=1}^k\Omega_i,
\qquad w_i=|\Omega_i|,
\qquad W=\sum_i w_i>0.
$$

先抽取

$$
(H_1,\ldots,H_k)\sim
\operatorname{Mult}\left(h;\frac{w_1}{W},\ldots,\frac{w_k}{W}\right),
$$

再从每个 $\Omega_i$ 中独立均匀抽取 $H_i$ 个样本，将所有块输出按任意确定顺序连接，最后均匀洗牌。所得有序序列由 $h$ 个来自 $\Omega$ 的 i.i.d. 均匀样本组成。

**证明。** 若 $w_i=0$，则该块的多项式配额以概率 $1$ 等于 $0$，删除这些空块不改变算法或分布；以下只对 $w_i>0$ 的块取乘积。固定任意目标序列 $(x_1,\ldots,x_h)\in\Omega^h$。令 $b(r)$ 是满足 $x_r\in\Omega_{b(r)}$ 的唯一块标签，并令

$$
n_i=|\{r:b(r)=i\}|.
$$

算法首先得到配额向量 $(n_1,\ldots,n_k)$ 的概率为

$$
\Pr[H_i=n_i,\ 1\le i\le k]
=
\frac{h!}{\prod_i n_i!}
\prod_i\left(\frac{w_i}{W}\right)^{n_i}.
$$

给定该配额，均匀洗牌把各块标签放到目标位置模式 $(b(1),\ldots,b(h))$ 的概率为

$$
\frac{\prod_i n_i!}{h!}.
$$

给定这些块标签位置，各块内部的带放回独立均匀抽样恰好产生目标元素的概率为

$$
\prod_i w_i^{-n_i}.
$$

三项相乘得到

$$
\Pr[(Z_1,\ldots,Z_h)=(x_1,\ldots,x_h)]
=
W^{-h}.
$$

该值对所有 $\Omega^h$ 中的有序序列相同，故输出分布正是 $\operatorname{Unif}(\Omega)^h$。$\square$

### 10.4 递归采样

**算法：Sample$_\ell(\mathfrak V_\ell(A,B),h)$**

1. 若 $h=0$，返回空序列。
2. 若 $A=\emptyset$ 或 $B=\emptyset$，返回 `EMPTY-INSTANCE`。
3. 若 $\ell=d$，返回 BaseSample$_d(\mathfrak V_d(A,B),h)$。
4. 运行 NodeWeights$_\ell$，得到区间、节点权重 $(w_z)_z$ 和总权重 $W$。
5. 若 $W=0$，返回 `EMPTY-INSTANCE`。
6. 在带标签节点权重上抽取多项式配额

$$
(H_z)_z\sim\operatorname{Mult}\left(h;\left(\frac{w_z}{W}\right)_z\right).
$$

7. 对每棵带标签树，自底向上计算子树选中配额

$$
Q_z=H_z+\sum_{c\text{ 为 }z\text{ 的子节点}}Q_c.
$$

8. 在 $B$ 树方向运行 RouteTree，令 $\mathsf{Active}(z)$ 为 $Q_z>0$，$\mathsf{Request}(z)$ 为 $H_z>0$。对 requested 节点递归调用 Sample$_{\ell+1}$，请求长度为 $H_z$，并把返回序列追加到输出缓冲区。
9. 在 $A$ 树方向执行相同过程。
10. 对输出缓冲区执行 Fisher–Yates 洗牌并返回。

若 $H_z>0$，则精确多项式抽样必然只会选中 $w_z>0$ 的节点，因此对应局部后缀连接非空。活动子树中的非请求祖先仍执行停止判定，保证 requested 后代的局部视图与完整 ANCHOR 分解一致。

## 11. AC：编译终端原子

AC 将 ANCHOR 分解展开经过维度 $1,2,\ldots,d-1$，在所有终端一维实例上生成正原子，并在全局原子权重上建立精确加权索引。

### 11.1 编译过程

令全局原子列表为 $\mathcal T$。编译过程如下。

**算法：Compile$_\ell(\mathfrak V_\ell(A,B))$**

1. 若 $A=\emptyset$ 或 $B=\emptyset$，返回。
2. 若 $\ell=d$：
   1. 运行 ComputeBaseAtoms$_d(\mathfrak V_d(A,B))$；
   2. 将所有正原子追加到全局列表 $\mathcal T$；
   3. 对每个被至少一个正原子引用的终端局部 $L_d$ 排序数组，在当前回调返回前取得持久只读所有权；同一终端局部实例中的全部原子共享该实例的同一份数组；
   4. 对未被任何正原子引用的终端局部数组不取得持久所有权；这些数组由 RouteTree 按第 9.3 节的缓冲生命周期统一释放；
   5. 返回，不再执行后续的非终端步骤。
3. 计算所有当前层区间 $X_B^{(\ell)}(a)$ 和 $X_A^{(\ell)}(b)$。
4. 在 $B$ 树方向运行全活动 RouteTree。节点完成停止与后继划分后，若停止源侧和后继侧均非空，则对带标签节点 $z=(\mathsf B,v)$ 回调调用

$$
\mathrm{Compile}_{\ell+1}(\mathfrak V_{\ell+1}(A_z,B_z)).
$$

5. 在 $A$ 树方向运行同样过程，对带标签节点 $z=(\mathsf A,u)$ 递归编译。

顶层调用 Compile$_1(\mathfrak V_1(\mathcal R,\mathcal S))$ 后，得到全局正原子列表

$$
\mathcal T=\{\tau_1,\ldots,\tau_s\}
$$

以及每个原子权重 $\omega_\tau$。令

$$
W=\sum_{\tau\in\mathcal T}\omega_\tau.
$$

若 $W>0$，AC 在 $\omega_\tau$ 上构造全局精确加权索引；若 $W=0$，不构造采样器。

每个终端局部 $L_d$ 数组只存储一次。原子记录只保存锚对象、局部对侧数组指针、秩区间 $[lo,hi)$ 和权重。全局 $L_d$ 数组不能替代终端局部数组，因为终端局部数组已经隐式编码了前面维度中的所有权选择和相交认证。

持久所有权可以通过移动不可变数组、紧凑复制或引用计数实现。无论采用哪种实现，只要某个原子仍引用终端局部数组，RouteTree 就不得释放或复用该数组的底层缓冲区。每个终端局部实例的 $A^{L_d}$ 与 $B^{L_d}$ 数组至多各物化一次，并由该实例中的全部相关原子共享。持久 backing allocation 的计费大小必须与局部数组长度同阶；若路由缓冲区来自更大的共享 arena，则终端局部数组采用紧凑复制，而不持久 pin 整个 arena。

### 11.2 AC 查询

**算法：AC-Sample$(t)$**

1. 若 $t=0$，返回空序列。
2. 若全局总权重 $W=0$，返回 `EMPTY-INSTANCE`。
3. 初始化长度为 $t$ 的输出数组。
4. 对每个位置 $r=1,\ldots,t$：
   1. 用全局加权索引抽取原子 $\tau$，概率为 $\omega_\tau/W$；
   2. 从 $[lo(\tau),hi(\tau))$ 中均匀抽取偏移 $j$；
   3. 使用 $\tau$ 的锚对象和终端局部对侧数组第 $j$ 项解码连接对。
5. 返回输出序列。

对任意 $p\in\Omega_\tau$，单次输出 $p$ 的概率为

$$
\frac{\omega_\tau}{W}\cdot\frac1{\omega_\tau}=\frac1W.
$$

不同输出位置独立调用全局索引和均匀偏移抽样，因此 AC 查询直接产生有序 i.i.d. 均匀样本。

## 12. ANCHOR 正确性与复杂度

**定理 12.1（AC 与 AS 的精确正确性）。** 在 exact word-RAM 模型下，AS 与 AC 均满足统一采样目标：若 $J\ne\emptyset$，长度为 $t$ 的查询返回 $t$ 个来自 $J$ 的 i.i.d. 均匀样本；若 $J=\emptyset$ 且 $t>0$，返回 `EMPTY-INSTANCE`；若 $t=0$，返回空序列。

**证明。** AS 的证明按剩余维度数量归纳。终端层由引理 8.1 给出。非终端层中，NodeWeights 由引理 10.2 给出精确的局部块权重；定理 7.3 给出当前后缀连接的不交块分解；归纳假设保证每个被选局部块内部递归采样精确；引理 10.3 保证按权重抽多项式配额、块内采样、连接并洗牌后得到整个后缀连接上的 i.i.d. 均匀有序序列。

AC 的编译过程沿定理 7.3 的不交分解递归展开到终端层，再由一维原子覆盖把 $J$ 表示为

$$
J=\biguplus_{\tau\in\mathcal T}\Omega_\tau.
$$

全局按 $\omega_\tau/|J|$ 采原子、原子内均匀采偏移，使每个连接对概率为 $1/|J|$。每个输出位置独立执行该实验，故输出序列为 i.i.d. 均匀。$\square$

**定理 12.2（ANCHOR 复杂度）。** 令 $N=|\mathcal R|+|\mathcal S|$，$d$ 为常数。在顶层排序视图已经构造好的 exact word-RAM 模型中：

1. AS 的计数时间为 $O(N\log^{d-1}(N+1))$，采样时间为 $O(N\log^{d-1}(N+1)+t)$，辅助空间为 $O(N)$，若包含输出则为空间 $O(N+t)$。
2. AC 的编译时间为 $O(N\log^{d-1}(N+1))$，持久空间为 $O(N\log^{d-1}(N+1))$，查询时间为 $O(t)$，若包含输出则空间为 $O(N\log^{d-1}(N+1)+t)$。

从原始输入开始时，读取和过滤需要 $O(N_0)$ 时间，顶层排序视图需要 $O(N\log(N+1))$ 时间；当 $d\ge2$ 时排序项被核心 ANCHOR 界吸收，当 $d=1$ 时排序项是主导初始化成本。

**证明。** 以下均在顶层排序视图已经构造好的模型中分析。记

$$
\Lambda(m)=\log(m+1).
$$

对一个剩余维度数为 $r>1$、局部大小为 $m$ 的调用，令 $m_z=|A_z|+|B_z|$，其中 $z$ 遍历当前层两棵带标签树中实际递归处理的节点。由引理 9.2，将所有树深度和两个方向相加，有

$$
\sum_zm_z=O(m\Lambda(m)),
\qquad
m_z\le m.
$$

先分析 Count。令 $C_r(m)$ 为剩余维度数为 $r$ 时的计数时间。终端层满足

$$
C_1(m)=O(m).
$$

当 $r>1$ 时，当前层的端点秩二分、两次完整 RouteTree 路由及节点回调管理共需 $O(m\Lambda(m))$ 时间。因此

$$
C_r(m)
\le
O(m\Lambda(m))+\sum_z C_{r-1}(m_z).
$$

由归纳假设 $C_{r-1}(x)=O(x\Lambda(x)^{r-2})$，并利用 $m_z\le m$，得到

$$
\begin{aligned}
C_r(m)
&\le O(m\Lambda(m))
 +O\left(\Lambda(m)^{r-2}\sum_zm_z\right)\\
&=O\left(m\Lambda(m)^{r-1}\right).
\end{aligned}
$$

NodeWeights 在当前层执行同样的完整路由，并对每个节点调用一次 Count$_{r-1}$，故满足同一递推和同一时间界。

接着分析 Sample。令 $S_r(m,h)$ 为剩余维度数为 $r$、请求样本数为 $h$ 时的采样时间。终端层由 BaseSample 给出

$$
S_1(m,h)=O(m+h).
$$

对 $r>1$，NodeWeights 的时间为 $O(m\Lambda(m)^{r-1})$。在 $O(m)$ 个带标签真实节点上抽取多项式配额并自底向上计算子树配额，需要 $O(m+h)$ 时间。第二次 RouteTree 只处理正配额诱导的活动子树，其体积不超过完整路由体积，因而需要 $O(m\Lambda(m))$ 时间。正配额子问题满足

$$
\sum_{z:H_z>0}H_z=h,
$$

以及

$$
\sum_{z:H_z>0}m_z
\le
\sum_zm_z
=O(m\Lambda(m)).
$$

由归纳假设，

$$
\begin{aligned}
\sum_{z:H_z>0}S_{r-1}(m_z,H_z)
&\le
\sum_{z:H_z>0}
O\left(m_z\Lambda(m_z)^{r-2}+H_z\right)\\
&\le
O\left(\Lambda(m)^{r-2}
\sum_{z:H_z>0}m_z\right)+O(h)\\
&=O\left(m\Lambda(m)^{r-1}+h\right).
\end{aligned}
$$

因此

$$
S_r(m,h)=O\left(m\Lambda(m)^{r-1}+h\right).
$$

取 $r=d$ 即得到 AS 的 CountTime 与 SampleTime。

空间方面，RouteTree 自身按树深逐层处理；由回调触发的维度递归采用同步、串行的深度优先语义，即一个递归子调用完整返回后，才继续当前 RouteTree 中的后续节点。一个大小为 $m'$ 的活动调用只保留当前深度与下一深度的路由缓冲区、节点权重、配额和活动标记，总大小为 $O(m')$；某个递归子调用返回后，其临时结构立即释放，再处理下一个兄弟节点。任意时刻只有一条长度至多为 $d$ 的维度递归路径处于活动状态，并且路径上每个局部大小都不超过 $N$。由于 $d$ 为常数，所有活动层的临时结构总和为

$$
O(dN)=O(N).
$$

顶层预先分配长度为 $t$ 的输出数组。每个 Sample 调用被指定一个长度为 $h$ 的连续片段，并按确定的块顺序把该片段划分为长度 $H_z$ 的互不相交子片段；对子节点的递归调用只填写其指定子片段。全部子调用返回后，当前调用仅对自己的长度 $h$ 片段执行 Fisher–Yates 洗牌。片段由下标边界描述，不复制输出对象，故输出数据共占 $O(t)$ 个字；包含工作空间后总空间为 $O(N+t)$。

再分析 AC。令 $P_r(m)$ 为 Compile 在剩余维度数为 $r$ 时的编译时间。终端层运行 ComputeBaseAtoms 并追加正原子，所以

$$
P_1(m)=O(m).
$$

对 $r>1$，当前层的端点秩计算和两次完整路由需要 $O(m\Lambda(m))$ 时间，各递归局部视图随后串行编译，因此

$$
P_r(m)
\le
O(m\Lambda(m))+\sum_zP_{r-1}(m_z).
$$

与 Count 的归纳相同，

$$
P_r(m)=O\left(m\Lambda(m)^{r-1}\right).
$$

令 $M_r(m)$ 为 Compile 完成后由该调用产生的持久终端数组和正原子记录的总空间。终端层有

$$
M_1(m)=O(m).
$$

非终端 RouteTree 缓冲区在同步回调返回后释放，不进入持久结构，故对 $r>1$，

$$
M_r(m)\le\sum_zM_{r-1}(m_z).
$$

利用归纳假设和 $\sum_zm_z=O(m\Lambda(m))$，得到

$$
\begin{aligned}
M_r(m)
&\le
O\left(\Lambda(m)^{r-2}\sum_zm_z\right)\\
&=O\left(m\Lambda(m)^{r-1}\right).
\end{aligned}
$$

全局正原子数不超过上述持久记录数。对全部正原子构造精确加权索引所需的线性时间和空间均被 $P_d(N)$ 与 $M_d(N)$ 的界吸收。编译期间的活动 RouteTree 递归栈额外占用 $O(N)$ 临时空间，也被持久空间界吸收。因此 AC 的编译时间和持久空间均为

$$
O\left(N\log^{d-1}(N+1)\right).
$$

查询时，每个输出位置执行一次全局加权原子抽样和一次原子内均匀偏移抽样；在第 3 节的精确 $O(1)$ 加权索引模型下，查询时间为 $O(t)$。包含输出后，总空间为

$$
O\left(N\log^{d-1}(N+1)+t\right).
$$

从原始输入开始时，再加上 $O(N_0)$ 的读取过滤和 $O(N\log(N+1))$ 的顶层排序成本，即得到第 5 节的端到端界。$\square$

---

# Part III. RangeTree-Based Baselines：LiftedRT 与 SweepRT

Range-tree baselines 都把盒相交转换为正交范围查询，再利用 range tree 的 canonical cover 做计数或局部均匀采样。LiftedRT 是静态提升方法：把一侧所有盒提升为 $2d$ 维点，并对另一侧每个盒做一次范围查询。SweepRT 是动态扫描方法：第一维由扫描线处理，剩余 $d-1$ 维提升为 $2(d-1)$ 维点，由动态 range tree 维护活跃集合。

## 13. 共享的 Range-Tree IQS 原语

本节给出 RangeTree baselines 使用的精确范围内独立均匀采样原语。该原语只作为 baseline 的标准数据结构接口使用；AC/AS 不依赖本节。

### 13.1 静态 canonical cover

对 $D$ 维点集 $P$ 和正交范围 $Q$，静态 range tree 查询返回若干 canonical blocks：

$$
P\cap Q=\biguplus_{B\in\mathcal K(Q)}P(B).
$$

这些块两两不交，覆盖 $P\cap Q$。每个块 $B$ 可以表示为某个关联数组中的连续片段，并支持：

1. 返回块大小 $\omega_B=|P(B)|$；
2. 对 $r\in\{0,\ldots,\omega_B-1\}$，返回块中第 $r$ 个对象；
3. 保留对象 id，因此重复坐标点仍作为不同对象计数。

对于固定维度 $D$，标准静态 range tree 的空间为

$$
O(|P|\log^{D-1}(|P|+1)),
$$

查询 canonical cover 的时间和 cover 大小为

$$
O(\log^D(|P|+1)).
$$

这些是固定维标准多层 range tree 的经典界 [8]。

### 13.2 动态活跃版本

SweepRT 使用的动态版本不是任意坐标在线插入结构，而是**静态全集骨架上的活跃标记结构**。扫描开始前，对所有可能出现的投影点预先建立 range-tree skeleton；扫描过程中，`Insert` 和 `Delete` 只切换某个已知对象的 active bit。

先考虑 $D\ge1$。range tree 在前 $D-1$ 个坐标上递归分解；到达最后一个坐标时，每个终端关联结构保存一个按第 $D$ 个坐标排序的对象数组。对正交范围 $Q$，前 $D-1$ 个坐标的 canonical decomposition 访问

$$
K=O\left(\log^{D-1}(N+1)\right)
$$

个终端关联数组。在每个被访问数组 $A$ 中，第 $D$ 个坐标条件对应一个连续下标区间

$$
[\ell_A,r_A).
$$

不同终端数组及区间所代表的对象集合两两不交，并且其并恰好是 $P\cap Q$。以下把这些连续区间称为动态 canonical blocks。

对每个终端关联数组 $A$ 建立一个 Fenwick tree [7]，其第 $x$ 个位置保存对象 $A[x]$ 的当前 active bit。定义

$$
F_A(x)=\sum_{u=0}^{x-1}\mathbf 1[A[u]\text{ is active}],
\qquad 0\le x\le |A|.
$$

于是动态块 $[\ell_A,r_A)$ 的当前活跃权重为

$$
\omega_A=F_A(r_A)-F_A(\ell_A),
$$

可在 $O(\log(N+1))$ 时间内得到。

每个点在前 $D-1$ 层所有搜索路径的组合中出现，因此属于

$$
O\left(\log^{D-1}(N+1)\right)
$$

个终端关联数组。骨架建立时，为每个对象预存其在这些数组中的位置列表。切换该对象的 active bit 时，在每个出现位置执行一次 Fenwick point update，因此

$$
\mathrm{Insert}/\mathrm{Delete}
=
O\left(\log^{D-1}(N+1)\log(N+1)\right)
=
O\left(\log^D(N+1)\right).
$$

一次范围计数查询枚举上述 $K$ 个不交动态块，并对每个块执行两次 Fenwick prefix sum。确定终端区间端点与计算所有块权重的总时间为

$$
O\left(K\log(N+1)\right)
=
O\left(\log^D(N+1)\right).
$$

因此

$$
\mathrm{RT.Count}(Q)=O\left(\log^D(N+1)\right).
$$

对块内 active rank-select，设块 $[\ell_A,r_A)$ 的权重为 $\omega_A>0$，并抽取

$$
\rho\sim\operatorname{Unif}\{0,\ldots,\omega_A-1\}.
$$

令

$$
\tau=F_A(\ell_A)+\rho+1.
$$

Fenwick select 返回满足

$$
F_A(x+1)\ge\tau
$$

的最小下标 $x$。由 $\tau>F_A(\ell_A)$ 且 $\tau\le F_A(r_A)$，必有

$$
\ell_A\le x<r_A,
$$

并且 $A[x]$ 正好是该块内按数组顺序计数的第 $\rho$ 个 active 对象。一次块内选择的时间为 $O(\log(N+1))$。

`RT.Sample(Q,k)` 先在 $O(\log^D(N+1))$ 时间内生成所有非零动态块及其权重，再以 $O(K)$ 时间建立块权重上的精确加权索引。每个样本以 $O(1)$ 时间选择一个块，再以 $O(\log(N+1))$ 时间在块内执行 active rank-select。因此

$$
\mathrm{RT.Sample}(Q,k)
=
O\left(\log^D(N+1)+k\log(N+1)\right).
$$

所有样本均带放回，采样过程不改变 active bits。

所有终端关联数组的总长度为

$$
O\left(N\log^{D-1}(N+1)\right).
$$

Fenwick arrays、对象出现位置列表和 range-tree skeleton 均具有相同数量级。因此动态结构的空间为

$$
O\left(N\log^{D-1}(N+1)\right).
$$

当 $D\ge2$ 时，从未排序点集建立 skeleton、终端关联数组、对象位置列表和全零 Fenwick arrays 的时间同样为 $O(N\log^{D-1}(N+1))$；该成本包含各层关联数组的排序与合并构造。当 $D=1$ 时，结构是一个按第一个坐标稳定排序的全局对象数组及其 Fenwick tree，构造时间为 $O(N\log(N+1))$，空间为 $O(N)$；其更新、计数和采样界仍分别为 $O(\log(N+1))$、$O(\log(N+1))$ 和 $O(\log(N+1)+k\log(N+1))$。

当 $D=0$ 时，范围查询没有坐标条件。每一侧使用一个 active dense vector $V$ 和位置表 $\mathrm{pos}$。插入对象 $q$ 时，把 $q$ 追加到 $V$ 并记录其下标；删除 $q$ 时，用 $V$ 的最后一个对象覆盖 $\mathrm{pos}[q]$ 所在位置，更新被移动对象的位置，再删除末元素。活跃数为 $|V|$，均匀样本由

$$
V[\operatorname{UniformInteger}(0,|V|)]
$$

返回，其中采样调用只在 $|V|>0$ 时发生。因而 `Insert`、`Delete` 和活跃数查询均为 $O(1)$，生成 $k$ 个带放回均匀样本为 $O(k)$，空间为 $O(N)$。SweepRT 的 $d=1$ 情形使用这一退化结构。

### 13.3 范围内独立均匀采样

给定 canonical cover $\mathcal K(Q)$，令

$$
W_Q=\sum_{B\in\mathcal K(Q)}\omega_B=|P\cap Q|,
$$

其中静态版本的 $\omega_B$ 是块大小，动态版本的 $\omega_B$ 是块内当前 active 点数。`RT-Sample(Q,k)` 的步骤为：

1. 计算 $\mathcal K(Q)$ 与各块权重 $\omega_B$。
2. 删除零权重块；若总权重为 $0$，返回空实例标记。
3. 在块权重上构建精确加权索引。
4. 对每个输出位置：先按 $\omega_B/W_Q$ 抽取块 $B$，再在 $B$ 中均匀选择一个对象。

对任意点 $p\in P\cap Q$，设 $B(p)$ 是其唯一所在块，则

$$
\Pr[p]=\frac{\omega_{B(p)}}{W_Q}\cdot\frac1{\omega_{B(p)}}=\frac1{W_Q}.
$$

不同输出位置使用独立随机性，因此 `RT-Sample(Q,k)` 返回 $k$ 个来自 $P\cap Q$ 的 i.i.d. 均匀样本。

## 14. LiftedRT

LiftedRT 对 $\mathcal S$ 侧建立静态 range tree，对每个 $a\in\mathcal R$ 计算其邻居数，并按邻居数进行外层采样。

### 14.1 $2d$ 维提升

对每个 $b\in\mathcal S$，定义 $2d$ 维点

$$
p(b)=\bigl(L_1(b),-U_1(b),L_2(b),-U_2(b),\ldots,L_d(b),-U_d(b)\bigr).
$$

点记录对象 id；坐标相同但 id 不同的对象作为不同点保存。

对每个 $a\in\mathcal R$，定义查询范围

$$
Q(a)=\prod_{j=1}^{d}\left(( -\infty,U_j(a))\times(-\infty,-L_j(a))\right).
$$

于是

$$
p(b)\in Q(a)
$$

当且仅当对所有 $j$ 有

$$
L_j(b)<U_j(a)
\quad\text{且}\quad
-U_j(b)<-L_j(a),
$$

也就是

$$
L_j(b)<U_j(a)
\quad\text{且}\quad
L_j(a)<U_j(b).
$$

因此

$$
p(b)\in Q(a)\iff a\sim b.
$$

严格开边界和重复坐标可用字典序键实现。对边界值 $x$ 和对象 id，使用 $(x,\mathrm{id})$ 排序，并引入哨兵 $\mathrm{id}_{\min}$ 与 $\mathrm{id}_{\max}$。这样

$$
y<x \iff (y,\mathrm{id}(y))<(x,\mathrm{id}_{\min}),
$$

$$
y>x \iff (y,\mathrm{id}(y))>(x,\mathrm{id}_{\max}).
$$

坐标恰好等于开边界的点不会被纳入。

若端点使用可能在取负时溢出的定宽有符号整数，实现可把第二个坐标保存为按 $U_j$ 降序的次序键，而不实际计算 $-U_j$；所有比较结果和证明保持不变。

### 14.2 度数与外层权重

定义

$$
N(a)=\{b\in\mathcal S:a\sim b\},
\qquad c_a=|N(a)|.
$$

由提升等价性，

$$
c_a=|P_{\mathcal S}\cap Q(a)|.
$$

总连接数为

$$
C=\sum_{a\in\mathcal R}c_a=|J|.
$$

LiftedRT 在所有 $c_a>0$ 的 $a$ 上按权重 $c_a$ 建立外层精确加权索引。

### 14.3 预处理算法

**算法：LiftedRT-Preprocess$(\mathcal R,\mathcal S)$**

1. 丢弃两侧空盒。
2. 对每个 $b\in\mathcal S$ 构造提升点 $p(b)$，保留对象 id。
3. 在点集 $P_{\mathcal S}$ 上构建静态 $D=2d$ 维 Range-Tree IQS。
4. 对每个 $a\in\mathcal R$：
   1. 构造 $Q(a)$；
   2. 令 $c_a=RT.Count(Q(a))$；
   3. 若 $c_a>0$，把 $(a,c_a)$ 加入外层权重列表。
5. 令 $C=\sum_a c_a$。
6. 若 $C>0$，构建外层精确加权索引；否则记录空实例。

### 14.4 批量采样算法

**算法：LiftedRT-Sample$(t)$**

1. 若 $t=0$，返回空序列。
2. 若 $C=0$，返回 `EMPTY-INSTANCE`。
3. 初始化输出数组 $Z[1..t]$ 与分组表 `groups`。
4. 对每个位置 $i=1,\ldots,t$：
   1. 从外层索引中按概率 $c_a/C$ 抽取 $a$；
   2. 将位置 $i$ 加入 `groups[a]`。
5. 对每个被选中过的 $a$：
   1. 令 $s_a=|\mathrm{groups}[a]|$；
   2. 调用 $RT.Sample(Q(a),s_a)$，得到 $s_a$ 个来自 $P_{\mathcal S}\cap Q(a)$ 的 i.i.d. 均匀点；
   3. 按 `groups[a]` 中记录的原始位置写回连接对 $(a,b)$。
6. 返回 $Z$。

### 14.5 正确性

**引理 14.1（LiftedRT 度数正确性）。** 对每个 $a\in\mathcal R$，预处理得到的 $c_a$ 等于 $|N(a)|$。

**证明。** 由 $2d$ 维提升等价性，$p(b)\in Q(a)$ 当且仅当 $a\sim b$。range tree 中每个 $b$ 以对象身份单独保存，因此 $RT.Count(Q(a))$ 正好计数满足 $a\sim b$ 的右侧对象数。$\square$

**引理 14.2（LiftedRT 总权重）。** $C=|J|$。

**证明。** 由 $c_a=|N(a)|$，

$$
\sum_{a\in\mathcal R}c_a
=
\sum_{a\in\mathcal R}|\{b\in\mathcal S:a\sim b\}|
=
|\{(a,b)\in\mathcal R\times\mathcal S:a\sim b\}|=|J|.
$$

$\square$

**引理 14.3（单样本均匀性）。** 若 $C>0$，LiftedRT 单个输出位置返回任意 $(a,b)\in J$ 的概率为 $1/|J|$。

**证明。** 外层以概率 $c_a/C$ 选择 $a$。给定 $a$ 后，range-tree IQS 从 $P_{\mathcal S}\cap Q(a)$ 中均匀采点，即从 $N(a)$ 中均匀采 $b$。因此

$$
\Pr[(a,b)]
=\frac{c_a}{C}\cdot\frac1{c_a}
=\frac1C
=\frac1{|J|}.
$$

$\square$

**定理 14.4（LiftedRT 的 IID 正确性）。** 若 $J\ne\emptyset$，LiftedRT-Sample$(t)$ 返回 $t$ 个来自 $J$ 的 i.i.d. 均匀样本。

**证明。** 固定任意目标序列

$$
((a_1,b_1),\ldots,(a_t,b_t))\in J^t.
$$

外层独立抽到左侧序列 $(a_1,\ldots,a_t)$ 的概率为

$$
\prod_{r=1}^t\frac{c_{a_r}}{C}.
$$

给定该左侧序列后，批量分组只改变执行顺序。对每个不同的 $a$，`RT.Sample(Q(a),s_a)` 返回 $N(a)$ 上的带放回 i.i.d. 样本；所有调用使用独立随机性。因此条件于左侧序列，右侧对象恰为 $(b_1,\ldots,b_t)$ 的概率为

$$
\prod_{r=1}^t\frac1{c_{a_r}}.
$$

两式相乘得到目标序列的联合概率

$$
\prod_{r=1}^t
\frac{c_{a_r}}C\frac1{c_{a_r}}
=C^{-t}=|J|^{-t}.
$$

故返回序列的分布为 $\operatorname{Unif}(J)^t$。$\square$

### 14.6 复杂度

令 $D=2d$，$n=|\mathcal R|$，$m=|\mathcal S|$。从提升点集建立标准多层静态 range tree 的时间和持久空间均为

$$
O(m\log^{D-1}(m+1));
$$

该构造成本包含各层关联数组的排序与合并。一次计数查询和一次 canonical cover 构造的时间均为

$$
O(\log^D(m+1)).
$$

预处理时间为

$$
O(m\log^{D-1}(m+1)+n\log^D(m+1)+n),
$$

空间为

$$
O(m\log^{D-1}(m+1)+n).
$$

用过滤后规模 $N=n+m$ 统一表示，数据结构预处理时间为

$$
O(N\log^{2d}(N+1)),
$$

从原始输入开始的 CountTime 为 $O(N_0+N\log^{2d}(N+1))$。空间上界为

$$
O(N\log^{2d-1}(N+1)+t).
$$

批量查询中，外层抽样和写回成本为 $O(t)$。设本次查询中被选中过的不同左侧对象数量为 $G$，则

$$
G\le\min\{t,n\}\le\min\{t,N\}.
$$

每个被选 $a$ 调用一次 $RT.Sample(Q(a),s_a)$，成本为 $O(\log^D(m+1)+s_a)$。总查询时间为

$$
O(t+G\log^D(m+1))
\subseteq
O(t+\min\{N,t\}\log^{2d}(N+1)).
$$

从原始输入开始的端到端时间为

$$
O(N_0+N\log^{2d}(N+1)+t).
$$

## 15. SweepRT

SweepRT 使用扫描线处理第 $1$ 维。每个相交对只会在两个 START 事件中较晚的那个事件处被发现；在该事件处，第 $1$ 维相交已由活跃集合保证，剩余维度由动态 range tree 计数与采样。

### 15.1 事件顺序与活跃集合

对每个盒 $q$ 定义两个事件：

$$
\mathrm{START}(q):x=L_1(q),
\qquad
\mathrm{END}(q):x=U_1(q).
$$

所有事件按如下全序排序：

1. 坐标小者在前；
2. 坐标相等时，$\mathrm{END}$ 在 $\mathrm{START}$ 前；
3. 坐标和类型都相等时，用全局对象 id 打破平局。

END-before-START 与半开语义一致：若 $U_1(r)=L_1(s)$，则 $r$ 会在 $s$ 开始前离开活跃集合，边界相触不会被当作相交。同坐标 START 的 id 顺序只决定同左端点相交对由哪个 START 事件发现；不影响覆盖和均匀性。

扫描维护两个活跃集合 $A_{\mathcal R}$ 和 $A_{\mathcal S}$。遇到 END 事件时，从本侧活跃集合删除该盒。遇到 START 事件 $q$ 时，先用对侧活跃集合形成事件块并进行查询，然后再把 $q$ 插入本侧活跃集合。

### 15.2 START 事件块

令 $e_i=\mathrm{START}(q_i)$ 为扫描序中的第 $i$ 个 START 事件。设投影维数为

$$
h=d-1.
$$

对盒 $q$，记其去掉第 $1$ 维后的投影为

$$
q^\downarrow=\prod_{k=2}^{d}[L_k(q),U_k(q)).
$$

当 $d=1$ 时，上式按空乘积解释为唯一的零维盒；任意两个零维投影都相交。因此该情形下 $K_i$ 就是当前对侧活跃集合，与第 13.2 节的 $D=0$ 结构一致。

当 START 事件 $q_i$ 被处理时，对侧活跃盒已经在第 $1$ 维与 $q_i$ 相交。定义伙伴集合

$$
K_i=
\begin{cases}
\{s\in A_{\mathcal S}:q_i^\downarrow\cap s^\downarrow\ne\emptyset\}, & q_i\in\mathcal R,\\[1mm]
\{s\in A_{\mathcal R}:s^\downarrow\cap q_i^\downarrow\ne\emptyset\}, & q_i\in\mathcal S.
\end{cases}
$$

输出方向固定为 $(\mathcal R,\mathcal S)$，定义

$$
\Phi_i(s)=
\begin{cases}
(q_i,s), & q_i\in\mathcal R,\\
(s,q_i), & q_i\in\mathcal S.
\end{cases}
$$

事件块为

$$
J_i=\{\Phi_i(s):s\in K_i\},
\qquad w_i=|J_i|=|K_i|.
$$

**引理 15.1（事件块不交分解）。** SweepRT 的 START 事件块满足

$$
J=\biguplus_i J_i.
$$

**证明。** 取任意相交对 $(r,s)\in J$。其两个 START 事件中较早者发生时，另一个盒尚未进入活跃集合，因此该对不会被发现。较晚 START 事件发生时，由于两个盒在第 $1$ 维半开相交，较早盒尚未 END，仍在对侧活跃集合中；剩余维度也相交，因此该对属于较晚事件的 $K_i$，进而属于 $J_i$。若两个 START 坐标相同，则事件全序中的较晚 START 会看到较早 START 已插入的对象；这正是同左端点情形的唯一归属。每个相交对只可能在较晚 START 事件处出现，所以事件块两两不交并覆盖 $J$。$\square$

### 15.3 剩余维度到 $2(d-1)$ 维范围查询

将投影维度重新编号为 $j=1,\ldots,h$，对应原始维度 $j+1$。定义

$$
L'_j(q)=L_{j+1}(q),
\qquad
U'_j(q)=U_{j+1}(q).
$$

对活跃盒 $s$，构造 $D=2h=2d-2$ 维点

$$
p(s)=\bigl(L'_1(s),\ldots,L'_h(s),U'_1(s),\ldots,U'_h(s)\bigr).
$$

对 START 盒 $q$，定义范围

$$
Q(q)=
\left(\prod_{j=1}^{h}(-\infty,U'_j(q))\right)
\times
\left(\prod_{j=1}^{h}(L'_j(q),+\infty)\right).
$$

于是对侧活跃盒 $s$ 满足

$$
s^\downarrow\cap q^\downarrow\ne\emptyset
$$

当且仅当

$$
p(s)\in Q(q).
$$

严格边界和重复坐标仍用 $(x,\mathrm{id})$ 字典序键与哨兵 id 实现。坐标等于开边界的活跃点不会被查询返回。

### 15.4 动态 range tree 操作

SweepRT 对两侧分别建立一个静态全集 skeleton：一棵覆盖所有 $\mathcal R$ 侧投影点，另一棵覆盖所有 $\mathcal S$ 侧投影点。扫描过程中，两棵树只维护 active bits；START 事件在对侧树上查询后，将本侧点 active bit 置为 $1$；END 事件将本侧点 active bit 置为 $0$。

对 $d\ge2$，动态维度为 $D=2d-2$。根据第 13.2 节，动态 range tree 支持：

- `Insert` / `Delete`：$O(\log^D(N+1))$；
- `RT-Count(Q)`：$O(\log^D(N+1))$；
- `RT-Sample(Q,k)`：$O(\log^D(N+1)+k\log(N+1))$。

其中 `RT-Sample` 先得到查询范围的 canonical blocks 及其 active 权重，在块权重上做精确加权抽样，再在被选块中用 active rank-select 返回对象。因为采样带放回，active bits 不因采样改变。

当 $d=1$ 时，$D=0$，没有剩余维度条件。每一侧使用第 13.2 节的 active dense vector 和位置表：`Insert` 通过末尾追加完成，`Delete` 通过 swap-with-last 完成，伙伴数就是当前对侧 vector 的长度。块内每个样本独立抽取一个均匀数组下标，因此一次伙伴计数为 $O(1)$，生成 $k$ 个带放回均匀伙伴为 $O(k)$。

### 15.5 两遍算法

SweepRT 是纯两遍算法。

**第一遍：Pass1-Count**

1. 初始化两棵空动态 range tree。
2. 按事件顺序扫描。
3. 若事件为 END，从本侧树删除该对象。
4. 若事件为 START$(q_i)$：
   1. 在对侧树上查询 $Q(q_i)$；
   2. 令 $w_i=RT.Count(Q(q_i))$；
   3. 将 $q_i$ 插入本侧树。
5. 返回所有事件块权重 $w_i$。

**位置分配：Assign-Positions**

1. 令 $W=\sum_iw_i$。
2. 若 $W=0$，记录空实例。
3. 否则在事件权重上构建精确加权索引。
4. 对每个输出位置 $j=1,\ldots,t$，独立抽取事件索引 $I_j$，概率为

$$
\Pr[I_j=i]=\frac{w_i}{W},
$$

并将位置 $j$ 加入列表 $L_i$。

**第二遍：Pass2-Sample**

1. 保留两棵静态 skeleton，并以全零 active bits 开始第二遍；第一遍处理完全部 END 事件后结构已经处于该状态。
2. 按完全相同的事件顺序扫描。
3. 若事件为 END，从本侧树删除该对象。
4. 若事件为 START$(q_i)$：
   1. 若 $L_i$ 非空，在对侧树上调用 $RT.Sample(Q(q_i),|L_i|)$，得到伙伴样本；
   2. 用 $\Phi_i$ 将伙伴映射为固定方向连接对，并写入 $L_i$ 记录的输出位置；
   3. 将 $q_i$ 插入本侧树。
5. 返回输出数组。

若 $t=0$，算法直接返回空序列。若 $t>0$ 且 $W=0$，算法返回 `EMPTY-INSTANCE`。

### 15.6 正确性

**引理 15.2（第一遍权重正确性）。** 第一遍对第 $i$ 个 START 事件返回的 $w_i$ 等于 $|J_i|$。

**证明。** 处理 START$(q_i)$ 时，对侧动态树中恰好存储对侧活跃盒的提升点。由剩余维度范围查询等价性，$RT.Count(Q(q_i))$ 正好计数所有投影与 $q_i^\downarrow$ 相交的对侧活跃盒，也就是 $|K_i|$。由定义 $|J_i|=|K_i|$。$\square$

**引理 15.3（第二遍块内采样）。** 若第二遍在事件 $i$ 调用 $RT.Sample(Q(q_i),k)$，其返回的是 $K_i$ 上的 $k$ 个 i.i.d. 均匀伙伴。

**证明。** 第二遍使用与第一遍完全相同的事件顺序，并在每个 START 事件处同样先查询再插入。因此事件 $i$ 处的对侧活跃点集与第一遍相同。range-tree IQS 对 $P\cap Q(q_i)$ 返回独立均匀点，而该集合与 $K_i$ 一一对应。$\square$

**定理 15.4（SweepRT 的 IID 正确性）。** 若 $J\ne\emptyset$，SweepRT 返回 $t$ 个来自 $J$ 的 i.i.d. 均匀样本。

**证明。** 由引理 15.1 和引理 15.2，

$$
W=\sum_iw_i=|J|.
$$

固定任意目标序列 $(p_1,\ldots,p_t)\in J^t$，并令 $i(r)$ 为满足 $p_r\in J_{i(r)}$ 的唯一事件块。位置分配阶段独立得到事件索引序列 $(i(1),\ldots,i(t))$ 的概率为

$$
\prod_{r=1}^t\frac{w_{i(r)}}W.
$$

给定这些事件索引后，引理 15.3 保证每个被调用事件块内部返回带放回 i.i.d. 均匀伙伴，不同事件调用也使用独立随机性。因此第二遍恰好产生目标连接对序列的条件概率为

$$
\prod_{r=1}^t\frac1{w_{i(r)}}.
$$

联合概率为

$$
\prod_{r=1}^t
\frac{w_{i(r)}}W\frac1{w_{i(r)}}
=W^{-t}=|J|^{-t}.
$$

故输出序列的分布为 $\operatorname{Unif}(J)^t$。$\square$

### 15.7 复杂度

令

$$
D=2d-2.
$$

丢弃空盒后共有 $N$ 个 START 事件和 $N$ 个 END 事件。事件序列只排序一次，并由两遍扫描共同使用。

先考虑 $d\ge2$，此时 $D\ge2$。扫描开始前，两侧静态 skeleton、终端关联数组、对象出现位置列表和全零 Fenwick arrays 的总构建时间与空间为

$$
O\left(N\log^{D-1}(N+1)\right).
$$

生成并排序 $2N$ 个事件的时间为 $O(N\log(N+1))$，事件数组空间为 $O(N)$。由于 $D\ge2$，这两项时间均被后续的 $O(N\log^D(N+1))$ 项吸收。

第一遍包含 $O(N)$ 次 active-bit 更新和 $N$ 次 START 计数查询。由第 13.2 节，

$$
T_{\mathrm{Pass1}}
=
O\left(N\log^D(N+1)\right)
=
O\left(N\log^{2d-2}(N+1)\right).
$$

位置分配在 $O(N)$ 个事件块权重上建立一次精确加权索引，并执行 $t$ 次独立事件抽样，因此

$$
T_{\mathrm{Assign}}=O(N+t).
$$

完整扫描结束时，每个已插入对象都已经处理其 END 事件，所以两棵动态树的 active bits 自然全部回到 $0$，第二遍可直接复用相同静态骨架。若实现选择显式清零全部 Fenwick arrays，其成本至多为

$$
O\left(N\log^{D-1}(N+1)\right),
$$

仍被第二遍的基础扫描成本吸收。

第二遍重放所有事件的基础更新成本为

$$
O\left(N\log^D(N+1)\right).
$$

设

$$
S=|\{i:|L_i|>0\}|.
$$

则

$$
S\le\min\{N,t\},
\qquad
\sum_i|L_i|=t.
$$

每个被选中的事件块执行一次动态范围采样。根据第 13.2 节，这些调用的总成本为

$$
\begin{aligned}
\sum_{i:|L_i|>0}
O\left(\log^D(N+1)+|L_i|\log(N+1)\right)
&=
O\left(S\log^D(N+1)+t\log(N+1)\right).
\end{aligned}
$$

因此，过滤后输入上的总时间为

$$
\begin{aligned}
T_{\mathrm{SweepRT}}
&=
O\left(
N\log^{D-1}(N+1)
+N\log(N+1)
+N\log^D(N+1)\right.\\
&\qquad\left.
+N+t
+S\log^D(N+1)
+t\log(N+1)
\right)\\
&=
O\left(N\log^D(N+1)+t\log(N+1)\right)\\
&=
O\left(N\log^{2d-2}(N+1)+t\log(N+1)\right).
\end{aligned}
$$

最后一步使用了 $S\le N$ 和 $D=2d-2$。从原始输入开始再加上 $O(N_0)$ 的读取与过滤成本。

两棵动态 range tree 的总空间为

$$
O\left(N\log^{D-1}(N+1)\right)
=
O\left(N\log^{2d-3}(N+1)\right).
$$

再加上事件数组、事件块权重、位置列表和输出数组，总空间为

$$
O\left(N\log^{2d-3}(N+1)+N+t\right).
$$

当 $d=1$ 时，$D=0$。事件排序成本为 $O(N\log(N+1))$。两侧 active dense vectors 与位置表的空间为 $O(N)$；每次插入、删除和计数均为 $O(1)$，每个伙伴样本也为 $O(1)$。两遍扫描、位置分配和所有伙伴采样的总成本为 $O(N+t)$。因此从原始输入开始，

$$
T_{\mathrm{SweepRT}}
=
O\left(N_0+N\log(N+1)+t\right),
$$

空间为

$$
O(N+t).
$$

---

# Part IV. 统一结论与实现口径

四个算法共享同一个输出语义：对非空连接 $J$，长度为 $t$ 的输出序列精确服从 $\operatorname{Unif}(J)^t$；对空连接和正查询长度返回 `EMPTY-INSTANCE`；对 $t=0$ 返回空序列。

AC/AS 的优势来自按当前维度递归认证相交关系。锚定所有权规则把每个连接对唯一分配给一个当前层带标签节点；规范 dyadic 覆盖保证每个锚的后继集合被拆成 $O(\log(N+1))$ 个不交局部块；终端一维原子给出可直接采样的最终表示。AS 的 RouteTree 按树深逐层路由，维度递归采用同步、串行、深度优先语义，因此辅助空间为 $O(N)$。AC 取得被正原子引用的终端数组的持久只读所有权，以 $O(N\log^{d-1}(N+1))$ 持久空间换取 $O(t)$ 查询时间。

SweepRT 与 LiftedRT 基于 range tree 的 canonical cover。LiftedRT 直接把完整 $d$ 维盒相交转化为 $2d$ 维静态范围查询；SweepRT 用扫描线处理第一维，并将剩余维度转化为 $D=2d-2$ 维动态范围查询。SweepRT 的动态结构在前 $D-1$ 个坐标上形成终端关联数组，在每个数组的最后一维连续区间上使用 Fenwick active rank-select；当 $d=1$ 时则使用 dense vector 与位置表。二者的正确性都来自同一个加权组合原则：先按不交块大小选择块，再在块内均匀采样。

所有复杂度表述都以精确整数权重和精确离散随机原语为基础，不依赖浮点概率。在单位代价 exact word-RAM 接口下，精确整数 alias 索引支持 $O(1)$ 加权类别抽样；若展开为无偏随机位拒绝采样，随机运行时间界为期望界。若采用精确前缀和与二分搜索替代 alias 索引，则所有分布结论保持不变，在 $k$ 个候选标签上的每次类别抽样需要 $O(\log(k+1))$ 时间。

---

# 参考文献

[1] H. Edelsbrunner and H. A. Maurer. “On the Intersection of Orthogonal Objects.” *Information Processing Letters*, 13:177–181, 1981. DOI: [10.1016/0020-0190(81)90053-3](https://doi.org/10.1016/0020-0190(81)90053-3).

[2] Xiaocheng Hu, Miao Qiao, and Yufei Tao. “Independent Range Sampling.” In *Proceedings of the 33rd ACM SIGMOD-SIGACT-SIGART Symposium on Principles of Database Systems*, pages 246–255, 2014. DOI: [10.1145/2594538.2594545](https://doi.org/10.1145/2594538.2594545).

[3] Dong Xie, Jeff M. Phillips, Michael Matheny, and Feifei Li. “Spatial Independent Range Sampling.” In *Proceedings of the 2021 International Conference on Management of Data*, pages 2023–2035, 2021. DOI: [10.1145/3448016.3452806](https://doi.org/10.1145/3448016.3452806).

[4] Daichi Amagata. “Independent Range Sampling on Interval Data.” In *2024 IEEE 40th International Conference on Data Engineering*, pages 449–461, 2024. DOI: [10.1109/ICDE60146.2024.00041](https://doi.org/10.1109/ICDE60146.2024.00041).

[5] Daichi Amagata. “Random Sampling Over Spatial Range Joins.” In *2025 IEEE 41st International Conference on Data Engineering*, pages 2080–2093, 2025. DOI: [10.1109/ICDE65448.2025.00158](https://doi.org/10.1109/ICDE65448.2025.00158).

[6] Michael D. Vose. “A Linear Algorithm for Generating Random Numbers with a Given Distribution.” *IEEE Transactions on Software Engineering*, 17(9):972–975, 1991. DOI: [10.1109/32.92917](https://doi.org/10.1109/32.92917).

[7] Peter M. Fenwick. “A New Data Structure for Cumulative Frequency Tables.” *Software: Practice and Experience*, 24(3):327–336, 1994. DOI: [10.1002/spe.4380240306](https://doi.org/10.1002/spe.4380240306).

[8] Jon Louis Bentley and Jerome H. Friedman. “Data Structures for Range Searching.” *ACM Computing Surveys*, 11(4):397–409, 1979. DOI: [10.1145/356789.356797](https://doi.org/10.1145/356789.356797).

[9] Frank Olken and Doron Rotem. “Simple Random Sampling from Relational Databases.” In *Proceedings of the 12th International Conference on Very Large Data Bases*, pages 160–169, 1986.

[10] Surajit Chaudhuri, Rajeev Motwani, and Vivek R. Narasayya. “On Random Sampling over Joins.” In *Proceedings of the 1999 ACM SIGMOD International Conference on Management of Data*, pages 263–274, 1999. DOI: [10.1145/304182.304206](https://doi.org/10.1145/304182.304206).
