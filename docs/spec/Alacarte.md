# Alacarte：可控期望输出密度的轴对齐矩形生成原理

## 摘要

本文给出一种面向空间连接数据生成的轴对齐矩形与超矩形生成方法。给定两个输入基数 $n_R,n_S$、维度 $d$、有限宇宙空间 $\mathcal U$、体积与形状分布，以及目标输出密度 $\alpha_{\mathrm{out}}^\star$，方法生成两个 box 集合 $R$ 与 $S$。输出密度定义为

$$
\alpha_{\mathrm{out}}
=
\frac{|\mathcal J(R,S)|}{n_R+n_S},
$$

其中 $\mathcal J(R,S)$ 为相交对象对的集合。

核心思想是引入名义覆盖参数 $C$，将期望输出密度控制转化为一维反问题。给定 $C$ 后，体积分布决定对象尺度，形状分布决定归一化坐标中的各维边长比例，位置则在保证对象位于宇宙空间内的可行区域中条件独立均匀采样。对给定的一维边长，可以解析计算两个区间的相交概率；多维条件相交概率等于各维条件概率的乘积。再对体积与形状随机性取期望，即可得到对象对相交概率 $p(C)$，进而得到期望输出密度

$$
\alpha_{\mathrm{exp}}(C)
=
\frac{n_R n_S}{n_R+n_S}p(C).
$$

在本文采用的尺度族、逐维饱和规则和条件独立均匀位置模型下，$p(C)$ 关于 $C$ 连续，并在 $p(C)<1$ 的范围内严格递增，同时满足 $p(0+)=0$ 与 $p(C)\to1$。因此，对任意严格位于可行区间内部的目标输出密度，都存在唯一的有限覆盖参数。求解时在对数覆盖率空间中构造括区间并执行二分搜索，无需枚举 $R\times S$。

有限样本求解将校准样本与认证样本分离。认证阶段采用方差自适应的经验 Bernstein 区间和预先规定的几何检查点；在目标输出密度与绝对容差保持固定的稀疏输出区间，其样本复杂度随规模因子近线性增长，而不是由分布无关界产生的平方增长。对总置信预算 $\delta$，算法保证

$$
\Pr\left(
\operatorname{CERTIFIED}(\widehat C)
\ \text{且}\ 
\left|
\alpha_{\mathrm{exp}}(\widehat C)
-
\alpha_{\mathrm{out}}^\star
\right|>
\varepsilon_\alpha
\right)
\le\delta.
$$

该方法控制的是随机生成模型下的期望输出密度。单次生成结果仍具有随机波动，不要求其连接基数等于预先指定的确定值。

## 1. 问题定义与适用范围

### 1.1 生成目标

随机模型与保证参数为

$$
n_R,
\quad
n_S,
\quad
d,
\quad
\alpha_{\mathrm{out}}^\star,
\quad
\mathcal U,
\quad
\text{体积分布},
\quad
\mathrm{CV},
\quad
\sigma_{\mathrm{shape}},
\quad
\varepsilon_{\mathrm{geom}},
\quad
\varepsilon_{\alpha},
\quad
\delta.
$$

其中，体积分布与 $\sigma_{\mathrm{shape}}$ 描述逐维饱和前的潜在几何分布；$\varepsilon_{\mathrm{geom}}$ 控制最大相对边长，$\varepsilon_\alpha$ 与 $\delta$ 分别给出期望输出密度容差和总失败概率。随机几何模型以逐维归一化坐标为规范语义。

样本预算、对数覆盖率范围和迭代上限等求解控制量在第 8.4 节给出。

目标是生成两个带对象索引的有限集合

$$
R=\{r_1,\ldots,r_{n_R}\},
\qquad
S=\{s_1,\ldots,s_{n_S}\}.
$$

即使两个对象具有相同的几何坐标，它们仍按不同索引计为不同输入对象。连接结果集定义为

$$
\mathcal J(R,S)
=
\{(r,s)\in R\times S:r\cap s\ne\varnothing\}.
$$

随机模型的目标是使

$$
\mathbb E[\alpha_{\mathrm{out}}]
=
\mathbb E\left[
\frac{|\mathcal J(R,S)|}{n_R+n_S}
\right]
$$

等于给定值 $\alpha_{\mathrm{out}}^\star$。期望取自体积、形状和位置随机性。有限样本求解返回满足第 8.4 节误差与置信语义的覆盖参数。

### 1.2 基本参数域

设

$$
A
:=
\frac{n_R n_S}{n_R+n_S}.
$$

主模型采用以下参数域：

$$
n_R,n_S,d\in\mathbb N_{>0},
$$

$$
0<\alpha_{\mathrm{out}}^\star<A,
$$

$$
\mathrm{CV}\ge0,
\qquad
\sigma_{\mathrm{shape}}\ge0,
$$

$$
0<\varepsilon_{\mathrm{geom}}<\frac12,
\qquad
0<\varepsilon_{\alpha}
<
\min\{\alpha_{\mathrm{out}}^\star,A-\alpha_{\mathrm{out}}^\star\},
\qquad
0<\delta<1.
$$

各体积分布还具有各自的参数约束：`fixed` 要求 $\mathrm{CV}=0$，`exponential` 要求 $\mathrm{CV}=1$，`lognormal` 允许 $\mathrm{CV}\ge0$，零点左截断正态族要求 $0\le \mathrm{CV}<1$。不满足所选分布族约束的参数组合不属于输入域。

严格内部目标给出统一的有限求解语义。

当 $\alpha_{\mathrm{out}}^\star=0$ 时，任意有限 $C>0$ 下的所有边长均几乎处处严格为正。给定正边长后，每一维相交概率严格为正，因此 $p(C)>0$。精确零目标不属于该独立均匀位置模型；空连接可由将 $R$ 与 $S$ 放置在互不相交子区域中的确定性空间分区模型实现。

当 $\alpha_{\mathrm{out}}^\star=A$ 时，需要 $p(C)=1$。由于 $0\le q(C;\omega)\le1$，有限 $C$ 达到该端点的充要条件是

$$
\Pr\left(
\forall k:\
\lambda_{R,k}(C)+\lambda_{S,k}(C)\ge W_k
\right)=1.
$$

等价地，每个维度的一维条件相交概率都必须几乎处处等于 $1$。各边长随机系数分别具有正下界是一个充分条件，但不是必要条件；决定端点是否可达的是每一维两条实际边长之和的本质下界。对于本质下界为 $0$ 的尺度族，通常只有在 $C\to\infty$ 时才能逼近最大输出密度。本文以下只讨论严格内部目标。

### 1.3 输出密度的图论解释

若将 $R$ 和 $S$ 视为二部图的两个顶点集合，将相交对象对视为边，则

$$
|\mathcal J(R,S)|
$$

是二部图的边数，而

$$
\frac{2|\mathcal J(R,S)|}{n_R+n_S}
=
2\alpha_{\mathrm{out}}
$$

是二部图的平均度数。因此，$\alpha_{\mathrm{out}}$ 是输出二元组总数与输入对象总数之比；若按每个输入对象参与的输出二元组数计数，则平均值为 $2\alpha_{\mathrm{out}}$。

由

$$
|\mathcal J(R,S)|\le n_R n_S
$$

可得

$$
0\le\alpha_{\mathrm{out}}\le A.
$$

### 1.4 模型假设

本文采用以下基础假设：

1. 对象均为有限宇宙空间内的轴对齐半开超矩形；
2. 对象体积由统一尺度参数与单位均值随机乘子共同决定；
3. 对象形状由归一化坐标中乘积为 $1$ 的各维形状因子决定；
4. 给定完整边长向量后，各维起点在对应可行区间内独立均匀分布；
5. 集合内对象独立同分布，$R$ 与 $S$ 相互独立；体积、形状和位置的基础随机变量也相互独立。

该模型不包含空间聚簇、热点、对象间排斥、跨集合配准、地图拓扑、非轴对齐几何或时间相关性。引入这些结构后，位置条件分布发生变化，一维闭式相交概率通常也需要重新推导。

### 1.5 研究定位

传统空间连接选择率估计面向已经给定的数据集，目标是估计固定输入上的连接结果规模 [9]。本文研究的是其逆向生成问题：先给定随机几何族和目标期望输出密度，再求使该分布达到目标的尺度参数。解析的一维相交概率在这里既是选择率表达式，也是生成参数反演的目标函数。该区分使本文的保证针对生成分布的期望负载，而不是对任意既有空间数据分布作无参数选择率估计。

## 2. 空间模型与几何不变量

### 2.1 宇宙空间

令

$$
\mathcal U
=
\prod_{k=1}^{d}
[u_k^{\min},u_k^{\max})
$$

为有限的 $d$ 维半开超矩形。记第 $k$ 维跨度为

$$
W_k
=
u_k^{\max}-u_k^{\min}>0,
$$

宇宙空间体积为

$$
V_{\mathcal U}
=
\prod_{k=1}^{d}W_k.
$$

若未指定其他范围，可取

$$
\mathcal U=[0,1)^d.
$$

对任意物理坐标 $x_k$，定义归一化坐标

$$
x_k'
=
\frac{x_k-u_k^{\min}}{W_k}.
$$

该映射将 $\mathcal U$ 映射为 $[0,1)^d$。本文的形状、边长饱和和位置采样均以该归一化坐标为规范语义。记

$$
\rho_k
=
\frac{\lambda_k}{W_k}
$$

为第 $k$ 维相对边长，则

$$
\frac{\operatorname{vol}(b)}{V_{\mathcal U}}
=
\prod_{k=1}^{d}\rho_k.
$$

### 2.2 Box 表示与相交谓词

每个对象表示为正边长轴对齐半开超矩形

$$
b
=
\prod_{k=1}^{d}[\ell_k,u_k),
\qquad
u_k>\ell_k.
$$

两个 box 相交，当且仅当每个维度都具有严格正长度重叠：

$$
r\cap s\ne\varnothing
\quad\Longleftrightarrow\quad
\forall k:\quad
\max(\ell_k^{(r)},\ell_k^{(s)})
<
\min(u_k^{(r)},u_k^{(s)}).
$$

采用半开区间后，仅在端点处接触不计为相交。

### 2.3 几何不变量

对任意生成对象，都应满足

$$
u_k^{\min}
\le
\ell_k
<
u_k
\le
u_k^{\max},
\qquad
k=1,\ldots,d.
$$

因此，每个对象均具有正边长，并完整包含在 $\mathcal U$ 内。后续的边长饱和与位置采样均围绕这一不变量构造。

## 3. 名义覆盖参数与一维反问题

### 3.1 名义覆盖参数

对 $T\in\{R,S\}$，令 $|T|=n_T$。给定覆盖参数 $C>0$，定义单个对象的平均目标体积为

$$
\bar v_T(C)
=
\frac{C\,V_{\mathcal U}}{n_T}.
$$

对应的平均相对体积为

$$
\bar\nu_T(C)
=
\frac{\bar v_T(C)}{V_{\mathcal U}}
=
\frac{C}{n_T}.
$$

因此，若忽略边长饱和，集合 $T$ 中所有对象体积之和的期望为

$$
n_T\bar v_T(C)
=
C\,V_{\mathcal U}.
$$

这一名义覆盖参数沿用 À La Carte 矩形生成模型中的总体积尺度思想 [1,2]。$C$ 是生成分布的尺度参数，不是几何并集覆盖率。它可以大于 $1$，因为不同对象可以相互重叠。本文进一步以对象对相交概率为桥梁，从目标输出密度反求 $C$。

### 3.2 几何总体积比

对一次生成结果，定义集合 $T$ 的几何总体积比为

$$
C_T^{\mathrm{geom}}
=
\frac{1}{V_{\mathcal U}}
\sum_{b\in T}\operatorname{vol}(b).
$$

若体积分布的均值为 $\bar v_T(C)$，且不存在边长饱和，则

$$
\mathbb E[C_T^{\mathrm{geom}}]=C.
$$

逐维饱和只会减小原始边长，因此一般有

$$
\mathbb E[C_T^{\mathrm{geom}}]\le C.
$$

这说明 $C$ 描述的是饱和前的名义总体积尺度，而不是每次生成后必然实现的总体积比。

### 3.3 输出密度反问题

固定体积分布、形状分布、边长饱和规则和位置模型后，$C$ 决定随机边长分布的整体尺度。定义

$$
p(C)
=
\Pr_{r\sim R(C),\,s\sim S(C)}
[r\cap s\ne\varnothing].
$$

由于任意对象对具有相同的边际相交概率，期望输出密度可写为

$$
\alpha_{\mathrm{exp}}(C)
=
A p(C),
\qquad
A=
\frac{n_R n_S}{n_R+n_S}.
$$

于是，输出密度控制转化为标量方程

$$
\text{求 }C>0,
\qquad
\alpha_{\mathrm{exp}}(C)
=
\alpha_{\mathrm{out}}^\star.
$$

后文将说明 $p(C)$ 连续，并在其取值小于 $1$ 的范围内严格递增，且在 $C\downarrow0$ 与 $C\to\infty$ 时分别趋于 $0$ 和 $1$。因此，每个严格内部目标存在唯一有限解。

## 4. 随机几何生成模型

每个 box 依次经过体积采样、形状采样、边长饱和和位置采样。用于求解 $C$ 的概率模型与最终数据生成必须采用同一组数学规则。

### 4.1 以单位均值乘子表示体积分布

对集合 $T\in\{R,S\}$，体积统一写为

$$
V_T(C)
=
\bar v_T(C)Z,
\qquad
Z>0,
\qquad
\mathbb E[Z]=1.
$$

随机变量 $Z$ 的分布不依赖 $C$，因此覆盖参数只负责整体尺度变化。该表示使体积关于 $C$ 线性增长，是单调性证明的基础。

#### `fixed`

$$
Z=1.
$$

所有对象在饱和前具有相同体积 $\bar v_T(C)$。

#### `exponential`

$$
Z\sim\operatorname{Exp}(1).
$$

其均值和变异系数分别为

$$
\mathbb E[Z]=1,
\qquad
\operatorname{CV}(Z)=1.
$$

#### `lognormal`

给定 $c=\mathrm{CV}\ge0$，令

$$
\sigma_v^2
=
\log(1+c^2),
\qquad
\mu_v
=
-\frac12\sigma_v^2,
$$

其中 $\sigma_v^2$ 在数值上按

$$
\sigma_v^2
=
\begin{cases}
\operatorname{log1p}(c^2),
&0\le c\le1,\\[4pt]
2\log c+\operatorname{log1p}(c^{-2}),
&c>1
\end{cases}
$$

计算，以避免先形成可能溢出的 $c^2$。
若 $c>0$ 但 $c^2$ 在当前格式中下溢为 $0$，则提高精度后计算；不得将该情形静默退化为 `fixed`。渐近关系 $\sigma_v\sim c$ 可用于选择所需精度。

并取

$$
\log Z
\sim
\mathcal N(\mu_v,\sigma_v^2).
$$

由对数正态分布的矩公式，

$$
\mathbb E[Z]
=
\exp\left(
\mu_v+\frac12\sigma_v^2
\right)
=1,
$$

且

$$
\frac{\sqrt{\operatorname{Var}(Z)}}{\mathbb E[Z]}
=
\sqrt{e^{\sigma_v^2}-1}
=c.
$$

#### `normal`

正态模式采用零点左截断正态分布，并使截断后的均值为 $1$、变异系数为给定值 $c$。该分布族对应

$$
0\le c<1.
$$

当 $c=0$ 时退化为 `fixed`。当 $0<c<1$ 时，设

$$
X\sim\mathcal N(\mu_0,\sigma_0^2),
\qquad
Y=X\mid X>0.
$$

记

$$
\kappa
=
\frac{\mu_0}{\sigma_0},
\qquad
\Lambda(\kappa)
=
\frac{\phi(\kappa)}{\Phi(\kappa)},
$$

其中 $\phi$ 和 $\Phi$ 分别为标准正态密度函数与分布函数。截断正态的均值和方差为

$$
\mathbb E[Y]
=
\sigma_0\bigl(\kappa+\Lambda(\kappa)\bigr),
$$

$$
\operatorname{Var}(Y)
=
\sigma_0^2
\left[
1-
\kappa\Lambda(\kappa)
-
\Lambda(\kappa)^2
\right].
$$

定义

$$
m(\kappa)
=
\kappa+\Lambda(\kappa),
$$

$$
v(\kappa)
=
1-
\kappa\Lambda(\kappa)
-
\Lambda(\kappa)^2,
$$

以及

$$
c(\kappa)
=
\frac{\sqrt{v(\kappa)}}{m(\kappa)}.
$$

**引理 1**　函数 $c(\kappa)$ 是从 $\mathbb R$ 到 $(0,1)$ 的连续严格递减双射，并满足

$$
\lim_{\kappa\to-\infty}c(\kappa)=1,
\qquad
\lim_{\kappa\to+\infty}c(\kappa)=0.
$$

**证明。** $\Lambda(\kappa)$、$m(\kappa)$ 与 $v(\kappa)$ 在有限 $\kappa$ 上连续，且零点左截断正态的均值和方差严格为正，因此 $c(\kappa)$ 连续。由单侧截断正态的矩域，任意该族分布都满足

$$
0<\operatorname{Var}(Y)<\mathbb E[Y]^2,
$$

故 $0<c(\kappa)<1$。当 $\kappa\to+\infty$ 时，$\Lambda(\kappa)\to0$，从而

$$
m(\kappa)\sim\kappa,
\qquad
v(\kappa)\to1,
\qquad
c(\kappa)\to0.
$$

当 $x=-\kappa\to+\infty$ 时，标准逆 Mills 比渐近式给出

$$
\Lambda(-x)
=
x+x^{-1}-2x^{-3}+10x^{-5}+O(x^{-7}),
$$

因而

$$
m(-x)
=
x^{-1}-2x^{-3}+10x^{-5}+O(x^{-7}),
$$

$$
v(-x)
=
x^{-2}-6x^{-4}+O(x^{-6}),
$$

所以 $v(-x)/m(-x)^2\to1$，即 $c(\kappa)\to1$。

零点左截断正态族的均值域定理表明：对任意 $m_Y>0$ 和 $0<s_Y/m_Y<1$，存在唯一参数对 $(\mu_0,\sigma_0)$ 使截断后均值为 $m_Y$、标准差为 $s_Y$ [6]。对任意给定的 $\kappa$，取 $\sigma_0=1/m(\kappa)$ 与 $\mu_0=\kappa/m(\kappa)$，截断后均值恰为 $1$，变异系数为 $c(\kappa)$。若两个不同的 $\kappa$ 产生同一变异系数，它们便给出具有相同均值与标准差的两组参数，与上述唯一性矛盾。因此 $c(\kappa)$ 连续且单射；结合两个端点极限可知它是从 $\mathbb R$ 到 $(0,1)$ 的严格递减双射。$\square$

因此，给定 $c\in(0,1)$ 时，方程

$$
c(\kappa)=c
$$

具有唯一解。数值求解可从任意有限初始区间出发，按倍增方式扩张端点直至括住目标，再执行一维二分。确定 $\kappa$ 后，令

$$
\sigma_0
=
\frac{1}{\kappa+\Lambda(\kappa)},
\qquad
\mu_0
=
\kappa\sigma_0,
$$

并取 $Z=Y$，即可得到均值为 $1$、变异系数为 $c$ 的正随机变量。

尾部区域中不直接计算 $\Phi(-\kappa)+U\Phi(\kappa)$。记

$$
\overline\Phi(x)=1-\Phi(x),
$$

并令 $\operatorname{isf}_{\mathcal N}^{\log}(t)$ 表示满足

$$
\log\overline\Phi
\left(
\operatorname{isf}_{\mathcal N}^{\log}(t)
\right)
=t
$$

的标准正态对数生存函数逆。若 $U\sim\operatorname{Uniform}(0,1)$ 且数值实现不返回两个端点，可取

$$
G
=
\operatorname{isf}_{\mathcal N}^{\log}
\left(
\log\overline\Phi(-\kappa)+\log U
\right),
$$

$$
Z
=
\frac{\kappa+G}{\kappa+\Lambda(\kappa)}.
$$

于是 $G$ 服从 $N(0,1)$ 在 $(-\kappa,\infty)$ 上的条件分布，且 $Z>0$、$\mathbb E[Z]=1$。若数值环境没有可靠的对数生存函数逆，可使用针对单侧截断正态的接受—拒绝采样器 [3,5]；所采用的采样器在校准、认证和最终生成中必须保持一致。

逆 Mills 比可按尾部安全的形式计算。例如在 $\kappa\le0$ 时，

$$
\Lambda(\kappa)
=
\frac{\sqrt{2/\pi}}
{\operatorname{erfcx}(-\kappa/\sqrt2)}.
$$

在 $\kappa\ll0$ 时，$\kappa+\Lambda(\kappa)$ 及方差因子由专用截断正态矩函数、连分式或相应渐近式直接返回，不形成两个接近数的减法。数值例程应同时返回结果及其舍入包围；无法在当前精度下形成可靠包围时，提高精度或返回明确的数值范围状态。

四种模式在数值计算中统一保留 $\log Z$：`fixed` 取 $0$；`lognormal` 直接采样正态对数变量；`exponential` 对 $U\in(0,1)$ 计算

$$
\log Z
=
\log\bigl(-\operatorname{log1p}(-U)\bigr);
$$

`normal` 计算尾部超越量的对数减去 $\log(\kappa+\Lambda(\kappa))$。后续边长计算不需要在线性域形成未饱和体积。

### 4.2 形状因子

对每个对象采样

$$
z_k
\overset{\mathrm{iid}}{\sim}
\mathcal N(0,\sigma_{\mathrm{shape}}^2),
\qquad
k=1,\ldots,d.
$$

在对数域中中心化：

$$
h_k
=
z_k-
\frac1d\sum_{j=1}^{d}z_j,
$$

并定义

$$
g_k=e^{h_k}.
$$

由此有

$$
\sum_{k=1}^{d}h_k=0,
\qquad
\prod_{k=1}^{d}g_k=1.
$$

因此，形状因子只改变各维相对边长比例，不改变未饱和对象的总体积。

上述等式按实数模型定义。有限精度实现使用补偿求和或成对求和计算中心值，并对全部分量执行同一个对称投影：先计算暂存向量 $\widehat h$，再按

$$
\widehat h_k
\leftarrow
\widehat h_k-
\frac1d\sum_{j=1}^{d}\widehat h_j
$$

重新中心化，必要时提高精度直至残差包围满足预设数值阈值。该过程不把全部残差集中到某一个维度；归约顺序固定，残余的舍入误差纳入后续边长与概率求值的区间包围。当 $d=1$ 时直接令 $h_1=0$。

中心化后，单个分量的方差与不同分量间的协方差为

$$
\operatorname{Var}(\log g_k)
=
\sigma_{\mathrm{shape}}^2
\left(1-\frac1d\right),
$$

$$
\operatorname{Cov}(\log g_i,\log g_j)
=
-\frac{\sigma_{\mathrm{shape}}^2}{d},
\qquad
i\ne j.
$$

任意两维的归一化形状因子之对数比满足

$$
\log\frac{g_i}{g_j}
\sim
\mathcal N(0,2\sigma_{\mathrm{shape}}^2).
$$

当 $\sigma_{\mathrm{shape}}=0$ 时，所有 $g_k=1$，对象在归一化坐标中为超立方体。特别地，当 $d=1$ 时恒有 $h_1=0$，$\sigma_{\mathrm{shape}}$ 不改变生成分布。数值计算保留 $h_k=\log g_k$ 即可，无需显式形成可能溢出或下溢的 $g_k$。

### 4.3 从体积与形状得到边长

本文在逐维归一化坐标中定义形状。对集合 $T$，饱和前相对体积为

$$
\nu_T(C)
=
\frac{V_T(C)}{V_{\mathcal U}}
=
\frac{C}{n_T}Z.
$$

在对数域中记

$$
\eta_T(C)
=
\log\nu_T(C)
=
\log C-\log n_T+\log Z,
$$

并定义第 $k$ 维未饱和对数相对边长

$$
\widetilde\tau_{T,k}(C)
=
\frac{\eta_T(C)}{d}+h_{T,k}.
$$

概念上的未饱和相对边长和物理边长分别为

$$
\widetilde\rho_{T,k}
=e^{\widetilde\tau_{T,k}},
\qquad
\widetilde\lambda_{T,k}
=W_k\widetilde\rho_{T,k}.
$$

由 $\sum_k h_{T,k}=0$，有

$$
\prod_{k=1}^{d}\widetilde\lambda_{T,k}
=
V_{\mathcal U}
\prod_{k=1}^{d}\widetilde\rho_{T,k}
=
V_T(C).
$$

令

$$
\tau_{\max}
=
\operatorname{log1p}(-\varepsilon_{\mathrm{geom}}),
$$

并记

$$
\rho_{\max}
=
e^{\tau_{\max}}
=
1-\varepsilon_{\mathrm{geom}},
\qquad
L_k^{\max}
=
W_k\rho_{\max}.
$$

最终对数相对边长、相对边长和物理边长为

$$
\tau_{T,k}
=
\min\{\widetilde\tau_{T,k},\tau_{\max}\},
$$

$$
\rho_{T,k}=e^{\tau_{T,k}},
\qquad
\lambda_{T,k}=W_k\rho_{T,k}.
$$

饱和后的物理体积满足

$$
V_T^{\mathrm{actual}}
=
V_{\mathcal U}
\exp\left(\sum_{k=1}^{d}\tau_{T,k}\right)
\le
V_T(C).
$$

因此，体积分布与 $\sigma_{\mathrm{shape}}$ 描述饱和前分布，最终分布由上述确定性饱和映射诱导。对几乎每个固定基础随机结果，$C\to\infty$ 时有

$$
\rho_{T,k}(C)
\to
1-\varepsilon_{\mathrm{geom}}.
$$

又因为 $2(1-\varepsilon_{\mathrm{geom}})>1$，每一维的条件相交概率趋于 $1$。达到饱和的尺度阈值可以依赖于基础随机结果，不要求存在对所有随机结果统一适用的有限 $C$。

比较、饱和以及尺度计算均在对数域完成。只有在生成坐标时才指数化最终有限边长；若正边长在选定数值格式中不可表示，则采用更高精度或返回明确的数值范围状态，不将其静默置为零。

### 4.4 位置采样

给定完整相对边长向量 $\boldsymbol\rho$，对每个维度独立采样

$$
Q_k
\sim
\operatorname{Uniform}(0,1).
$$

先在归一化坐标中设置

$$
\ell_k'
=
Q_k(1-\rho_k),
$$

$$
u_k'
=
1-(1-Q_k)(1-\rho_k).
$$

于是

$$
u_k'-\ell_k'=\rho_k,
\qquad
0\le\ell_k'<u_k'\le1.
$$

映射回物理坐标：

$$
\ell_k
=
u_k^{\min}+W_k\ell_k',
\qquad
u_k
=
u_k^{\min}+W_ku_k'.
$$

因此 $u_k-\ell_k=\lambda_k$，且给定 $\lambda_k$ 后，

$$
\ell_k
\sim
\operatorname{Uniform}
\bigl(u_k^{\min},u_k^{\max}-\lambda_k\bigr).
$$

所有对象、维度以及两个集合使用的 $Q_k$ 相互独立，并独立于体积与形状基础变量。实际起点坐标跨维仅在给定完整边长向量后条件独立。

## 5. 一维区间相交概率

### 5.1 闭式公式

令 $W>0$，给定 $a,b\in[0,W]$。取相互独立的

$$
X\sim\operatorname{Uniform}(0,W-a),
\qquad
Y\sim\operatorname{Uniform}(0,W-b),
$$

并定义半开区间

$$
I=[X,X+a),
\qquad
J=[Y,Y+b).
$$

当 $a=W$ 时，约定 $X=0$；当 $b=W$ 时，约定 $Y=0$。则相交概率为

$$
P_{\mathrm{1D}}(a,b;W)
=
\begin{cases}
0,
& a=0\ \text{或}\ b=0,\\[4pt]
1,
& a>0,\ b>0,\ a+b\ge W,\\[8pt]
\dfrac{W(a+b)-a^2-ab-b^2}
{(W-a)(W-b)},
& a>0,\ b>0,\ a+b<W.
\end{cases}
$$

以相对边长 $\rho_R=a/W$、$\rho_S=b/W$ 表示时，尺度 $W$ 完全消去：

$$
P_{\mathrm{1D}}(W\rho_R,W\rho_S;W)
=
P_{\mathrm{1D}}(\rho_R,\rho_S;1).
$$

### 5.2 推导

若 $a=0$ 或 $b=0$，对应区间为空，相交概率为 $0$。以下设 $a,b>0$。

两个半开区间不相交，当且仅当

$$
X+a\le Y
\quad\text{或}\quad
Y+b\le X.
$$

两事件互斥。随机点 $(X,Y)$ 在矩形

$$
\mathcal D
=
[0,W-a]\times[0,W-b]
$$

上均匀分布，其面积为

$$
|\mathcal D|
=
(W-a)(W-b).
$$

当 $a+b<W$ 时，记

$$
c=W-a-b>0.
$$

事件 $X+a\le Y$ 对应的区域面积为

$$
\int_0^c
\bigl[(W-b)-(x+a)\bigr]dx
=
\int_0^c(c-x)dx
=
\frac{c^2}{2}.
$$

由对称性，事件 $Y+b\le X$ 对应的面积也为 $c^2/2$。因此

$$
\Pr(I\cap J=\varnothing)
=
\frac{(W-a-b)^2}{(W-a)(W-b)}.
$$

从而

$$
P_{\mathrm{1D}}(a,b;W)
=
1-
\frac{(W-a-b)^2}{(W-a)(W-b)}.
$$

展开分子可得

$$
P_{\mathrm{1D}}(a,b;W)
=
\frac{W(a+b)-a^2-ab-b^2}
{(W-a)(W-b)}.
$$

当 $a+b>W$ 时，不相交区域为空，故相交概率为 $1$。当 $a+b=W$ 时，只有端点恰好接触的边界配置可能不相交，而这些配置在连续采样域中的测度为 $0$，因此相交概率仍为 $1$。

### 5.3 关于边长的单调性

对固定 $W$，$P_{\mathrm{1D}}(a,b;W)$ 分别关于 $a$ 和 $b$ 非递减。

在 $a,b>0$ 且 $a+b<W$ 的区域，令不相交概率为

$$
F(a,b)
=
\frac{(W-a-b)^2}{(W-a)(W-b)}.
$$

对 $a$ 求对数导数：

$$
\frac{\partial}{\partial a}\log F(a,b)
=
-\frac{2}{W-a-b}
+
\frac{1}{W-a}
<0.
$$

因此 $F$ 关于 $a$ 递减，$1-F$ 关于 $a$ 递增；对 $b$ 同理。跨越 $a+b=W$ 后，概率进入常数 $1$ 区域，因此整个正边长定义域上的相交概率均保持非递减。

### 5.4 对数域表示

在尺度悬殊或概率极小时，直接计算概率乘积容易出现数值下溢。计算时先处理 $a=0$ 或 $b=0$ 的分支；只有在两条边长均为正时，才定义

$$
\ell_x=\log a-\log W,
\qquad
\ell_y=\log b-\log W,
$$

$$
\ell_s
=
\operatorname{logaddexp}(\ell_x,\ell_y)
=
\log\left(\frac{a+b}{W}\right).
$$

对本文的归一化生成模型，可直接取 $\ell_x=\tau_{R,k}$、$\ell_y=\tau_{S,k}$，无需形成物理边长或执行 $\log a-\log W$ 的减法。

其中

$$
\operatorname{logaddexp}(u,v)
=
m+\log\left(e^{u-m}+e^{v-m}\right),
\qquad
m=\max\{u,v\}.
$$

对 $t<0$，定义

$$
\operatorname{log1mexp}(t)
=
\begin{cases}
\log\bigl(-\operatorname{expm1}(t)\bigr),
&-\log 2<t<0,\\[4pt]
\operatorname{log1p}(-e^t),
&t\le-\log 2.
\end{cases}
$$

当 $a,b>0$ 且 $\ell_s<0$ 时，利用恒等式

$$
x+y-x^2-xy-y^2
=
x(1-x-y)+y(1-y),
\qquad
x=\frac aW,
\quad
y=\frac bW,
$$

先计算

$$
\ell_{\mathrm{num}}
=
\operatorname{logaddexp}
\left(
\ell_x+\operatorname{log1mexp}(\ell_s),
\ell_y+\operatorname{log1mexp}(\ell_y)
\right),
$$

再写出

$$
\log P_{\mathrm{1D}}
=
\ell_{\mathrm{num}}
-
\operatorname{log1mexp}(\ell_x)
-
\operatorname{log1mexp}(\ell_y).
$$

完整分支为

$$
\log P_{\mathrm{1D}}(a,b;W)
=
\begin{cases}
-\infty,
&a=0\ \text{或}\ b=0,\\[4pt]
0,
&a>0,\ b>0,\ \ell_s\ge0,\\[4pt]
\ell_{\mathrm{num}}
-\operatorname{log1mexp}(\ell_x)
-\operatorname{log1mexp}(\ell_y),
&\ell_s<0.
\end{cases}
$$

数学上最后一项不大于 $0$，浮点实现返回其与 $0$ 的较小者，以吸收舍入误差。当 $a+b=W$ 时，相交概率为 $1$，但仍存在测度为 $0$ 的端点接触配置，因此该分支表示连续位置分布下的几乎必然相交。

## 6. 多维相交概率与期望输出密度

### 6.1 条件相交概率

给定两个 box 的边长向量

$$
\boldsymbol\lambda^{(r)}
=
(\lambda_1^{(r)},\ldots,\lambda_d^{(r)}),
$$

$$
\boldsymbol\lambda^{(s)}
=
(\lambda_1^{(s)},\ldots,\lambda_d^{(s)}),
$$

由于给定完整边长向量后的位置坐标跨维独立，两个 box 相交等价于所有维度同时相交。因此

$$
\Pr(r\cap s\ne\varnothing
\mid
\boldsymbol\lambda^{(r)},
\boldsymbol\lambda^{(s)})
=
\prod_{k=1}^{d}
P_{\mathrm{1D}}
(\lambda_k^{(r)},\lambda_k^{(s)};W_k).
$$

该结论不要求同一个 box 的各维边长相互独立。共享体积和形状中心化会使各维边长相关，但给定边长后，各维位置仍然条件独立。

### 6.2 对边长分布取期望

令

$$
\omega=(\omega_R,\omega_S),
\qquad
\omega_R\perp\omega_S,
$$

其中 $\omega_R,\omega_S$ 分别包含两个对象的体积乘子与形状基础随机变量。定义

$$
q(C;\omega)
=
\prod_{k=1}^{d}
P_{\mathrm{1D}}
\bigl(
\lambda_k^{(R)}(C;\omega_R),
\lambda_k^{(S)}(C;\omega_S);W_k
\bigr).
$$

对象对相交概率为

$$
p(C)
=
\mathbb E_{\omega}[q(C;\omega)].
$$

若取 $M$ 个独立同分布基础样本

$$
\omega_1,\ldots,\omega_M
\overset{\mathrm{iid}}{\sim}\omega,
$$

则可用样本平均近似

$$
\widehat p_M(C)
=
\frac1M\sum_{j=1}^{M}q(C;\omega_j).
$$

对每个固定且非随机的 $C>0$，由于 $0\le q\le1$，有

$$
\mathbb E[\widehat p_M(C)]=p(C),
$$

并且

$$
\widehat p_M(C)
\xrightarrow{\mathrm{a.s.}}
p(C)
\qquad
(M\to\infty).
$$

位置随机性已经通过一维闭式概率被解析积分，因此不需要显式采样位置再统计相交指示量。令 $I(C)$ 表示进一步采样两个对象的位置后得到的相交指示变量，则

$$
q(C;\omega)
=
\mathbb E[I(C)\mid\omega].
$$

由全方差公式，

$$
\operatorname{Var}(I(C))
=
\mathbb E[q(C;\omega)(1-q(C;\omega))]
+
\operatorname{Var}(q(C;\omega)),
$$

因此

$$
\operatorname{Var}(\widehat p_M(C))
=
\frac{\operatorname{Var}(q(C;\omega))}{M}
\le
\frac{p(C)(1-p(C))}{M}.
$$

同一批基础样本可在不同候选 $C$ 上复用，以保持经验目标函数的单调性。由这些样本选出的 $\widehat C$ 本身是随机变量，固定 $C$ 的无偏等式不直接给出 $p(\widehat C)$ 的有限样本精度；该精度由独立认证样本给出。

### 6.3 从对象对概率到输出密度

令

$$
I_{ij}
=
\mathbf 1[r_i\cap s_j\ne\varnothing].
$$

则

$$
|\mathcal J(R,S)|
=
\sum_{i=1}^{n_R}
\sum_{j=1}^{n_S}I_{ij}.
$$

由期望线性性，且每个对象对具有相同的边际相交概率 $p(C)$，

$$
\mathbb E[|\mathcal J(R,S)|]
=
n_R n_Sp(C).
$$

这里不要求不同 $I_{ij}$ 相互独立。因此

$$
\alpha_{\mathrm{exp}}(C)
=
\frac{\mathbb E[|\mathcal J(R,S)|]}{n_R+n_S}
=
A p(C).
$$

对应的样本平均近似为

$$
\widehat\alpha_M(C)
=
A\widehat p_M(C).
$$

### 6.4 高维概率的对数聚合

高维稀疏情形中，直接计算各维概率乘积可能下溢。对第 $j$ 个基础样本，计算

$$
\log q_j
=
\sum_{k=1}^{d}
\log P_{\mathrm{1D},j,k}.
$$

再通过 `logsumexp` 得到

$$
\log\widehat p_M(C)
=
\operatorname{logsumexp}
(\log q_1,\ldots,\log q_M)
-
\log M.
$$

约定所有 $\log q_j=-\infty$ 时，`logsumexp` 的结果为 $-\infty$。

该式是线性域样本平均的稳定等价计算。一般而言，$\log\widehat p_M(C)$ 不是 $\log p(C)$ 的无偏估计；由 Jensen 不等式，

$$
\mathbb E[\log\widehat p_M(C)]
\le
\log p(C).
$$

求解时计算

$$
\log p^*
=
\log\alpha_{\mathrm{out}}^\star-\log A
$$

并直接比较 $\log\widehat p_M(C)$ 与 $\log p^*$，从而避免不必要的线性域指数化。`logsumexp` 消除概率乘积和样本求和中的浮点下溢；有限样本精度仍由样本量及 $q(C;\omega)$ 的分布决定。

### 6.5 方差自适应独立认证与置信保证

设候选 $\widehat C$ 由校准样本以及已经结束的全部轮次确定。随后生成不参与 $\widehat C$ 选择的新认证样本

$$
\widetilde\omega_1,\ldots,\widetilde\omega_K
\overset{\mathrm{iid}}{\sim}\omega,
\qquad
K\ge2,
$$

并定义

$$
Q_j
=
q(\widehat C;\widetilde\omega_j)
\in[0,1],
$$

$$
\overline Q_K
=
\frac1K\sum_{j=1}^{K}Q_j,
$$

$$
\widehat V_K
=
\frac1{K-1}
\sum_{j=1}^{K}
(Q_j-\overline Q_K)^2.
$$

条件于候选参数，$Q_1,\ldots,Q_K$ 仍为独立同分布的有界随机变量，且

$$
\mathbb E[Q_j\mid\widehat C]
=
p(\widehat C).
$$

对置信预算 $\xi\in(0,1)$，定义经验 Bernstein 半径

$$
r_{\mathrm{EB}}(K,\xi,\widehat V_K)
=
\sqrt{
\frac{2\widehat V_K\log(4/\xi)}{K}
}
+
\frac{7\log(4/\xi)}{3(K-1)}.
$$

由经验 Bernstein 不等式 [7] 的两个单侧形式及并合界，

$$
\Pr\left(
\left|
\overline Q_K-p(\widehat C)
\right|
>
r_{\mathrm{EB}}(K,\xi,\widehat V_K)
\ \middle|\ 
\widehat C
\right)
\le\xi.
$$

认证计算采用有向舍入或区间算术，返回均值包围

$$
[p_K^-,p_K^+]
\ni
\overline Q_K
$$

和样本方差上界

$$
V_K^+
\ge
\widehat V_K.
$$

选择任意可表示的

$$
p_K^{\mathrm{num}}
\in
[p_K^-,p_K^+],
$$

例如区间中点的舍入值，并以上向舍入定义

$$
r_{\mathrm{num}}
=
\max\left\{
p_K^{\mathrm{num}}-p_K^-,
p_K^+-p_K^{\mathrm{num}}
\right\}.
$$

随后以上向舍入计算

$$
r_{\mathrm{stat}}
=
r_{\mathrm{EB}}(K,\xi,V_K^+).
$$

由于 $r_{\mathrm{EB}}$ 关于方差参数非递减，以至少 $1-\xi$ 的条件概率有

$$
p(\widehat C)
\in
\left[
\max\{0,p_K^{\mathrm{num}}-r_{\mathrm{stat}}-r_{\mathrm{num}}\},
\min\{1,p_K^{\mathrm{num}}+r_{\mathrm{stat}}+r_{\mathrm{num}}\}
\right].
$$

因此，输出密度的充分认证条件为

$$
\left|
A p_K^{\mathrm{num}}
-
\alpha_{\mathrm{out}}^\star
\right|
+
A(r_{\mathrm{stat}}+r_{\mathrm{num}})
\le
\varepsilon_\alpha.
$$

认证样本按几何检查点读取。第 $r$ 个候选使用预先规定的样本量

$$
K_s
=
2^{s-1}K_1,
\qquad
s=1,\ldots,S_{\max},
$$

并为每个候选—检查点对分配预算

$$
\xi_{r,s}
=
\frac{\delta}{R_{\max}S_{\max}}.
$$

同一候选的各检查点使用一条全新的认证流的前缀；候选在读取该流前已经固定。虽然不同前缀相互依赖，但每个检查点的覆盖结论分别成立，并合界不要求这些事件独立。因此允许根据已经观察到的区间决定继续取样或停止。

经验 Bernstein 半径同时利用样本方差与取值范围。对

$$
\sigma_q^2
=
\operatorname{Var}(Q_j\mid\widehat C),
$$

标准的经验 Bernstein 停止分析给出检查点对数因子之外的实例依赖样本量 [8]

$$
K
=
\widetilde O\left(
\frac{\sigma_q^2}{\eta^2}
+
\frac1\eta
\right)
$$

以达到概率半宽 $\eta$。本文还有

$$
\sigma_q^2
\le
p(\widehat C)(1-p(\widehat C))
\le
p(\widehat C).
$$

在能够通过目标容差认证的候选附近，$p(\widehat C)\le p^*+\varepsilon_p$。取 $\eta$ 与 $\varepsilon_p$ 同阶并代入

$$
p^*=\frac{\alpha_{\mathrm{out}}^\star}{A},
\qquad
\varepsilon_p=\frac{\varepsilon_\alpha}{A},
$$

可得

$$
K
=
\widetilde O\left(
\frac{A(\alpha_{\mathrm{out}}^\star+\varepsilon_\alpha)}
{\varepsilon_\alpha^2}
+
\frac{A}{\varepsilon_\alpha}
\right).
$$

因此，当 $\alpha_{\mathrm{out}}^\star$ 与 $\varepsilon_\alpha$ 不随 $A$ 增长时，认证样本量关于 $A$ 近线性增长。仅假设 $Q_j\in[0,1]$ 时，这一阶数在最坏情形下一般不能继续降低：若 $Q_j$ 近似均值为 $p=\Theta(1/A)$ 的 Bernoulli 变量，以固定相对精度估计其均值需要 $\Omega(1/p)=\Omega(A)$ 个样本。若目标对应 $p^*=\Theta(1)$，而输出密度绝对容差仍固定，则需要以 $\Theta(1/A)$ 的精度估计一个常数方差均值，此时平方级依赖是该统计任务本身的精度代价。

上述保证控制的是联合错误事件

$$
\Pr\left(
\operatorname{CERTIFIED}
\ \text{且真实误差超过容差}
\right).
$$

除非另有对 $\Pr(\operatorname{CERTIFIED})$ 的正下界，否则该式不应写成条件概率 $\Pr(\text{误差超过容差}\mid\operatorname{CERTIFIED})\le\delta$。

## 7. 目标函数的连续性、严格内部单调性与解唯一性

### 7.1 尺度耦合表示

对本文支持的体积分布，可在同一概率空间上写成

$$
V_T(C;\omega)
=
\frac{C\,V_{\mathcal U}}{n_T}Z_T(\omega),
$$

其中

$$
Z_T(\omega)>0
$$

几乎处处成立，且其分布不依赖 $C$。

将体积乘子、归一化形状和宇宙跨度合并到随机系数中，未饱和物理边长可写为

$$
\widetilde\lambda_{T,k}(C;\omega)
=
C^{1/d}B_{T,k}(\omega),
$$

其中

$$
B_{T,k}(\omega)>0
$$

几乎处处成立；具体地，

$$
B_{T,k}(\omega)
=
W_k n_T^{-1/d}Z_T(\omega)^{1/d}e^{h_{T,k}(\omega)}.
$$

最终边长为

$$
\lambda_{T,k}(C;\omega)
=
\min\left\{
C^{1/d}B_{T,k}(\omega),
L_k^{\max}
\right\}.
$$

### 7.2 连续性与严格内部单调性

**定理 1**　在本文的体积、形状、边长饱和与位置模型下，$p(C)$ 和 $\alpha_{\mathrm{exp}}(C)$ 在 $C>0$ 上连续且非递减。进一步地，对任意 $0<C_1<C_2$，若 $p(C_1)<1$，则

$$
p(C_2)>p(C_1).
$$

因此，$p$ 在其取值严格小于 $1$ 的区间上严格递增，$\alpha_{\mathrm{exp}}(C)=Ap(C)$ 具有相同性质。

**证明。** 固定任意基础随机结果 $\omega$。由于 $C^{1/d}$ 关于 $C$ 严格递增，与常数取最小值保持连续和非递减，因此每个

$$
\lambda_{T,k}(C;\omega)
$$

关于 $C$ 连续且非递减。

对任意 $C>0$，边长严格为正。一维相交概率在正边长域内连续，并分别关于两条边长非递减。因此

$$
P_{\mathrm{1D}}
\bigl(
\lambda_{R,k}(C;\omega),
\lambda_{S,k}(C;\omega);W_k
\bigr)
$$

关于 $C$ 连续且非递减。有限个取值位于 $[0,1]$ 的非负非递减函数之积仍然非递减，所以

$$
q(C;\omega)
$$

关于 $C$ 连续且非递减。

若 $C_1<C_2$，则逐样本有

$$
q(C_1;\omega)
\le
q(C_2;\omega).
$$

两端取期望得到

$$
p(C_1)\le p(C_2).
$$

又因为

$$
0\le q(C;\omega)\le1,
$$

对任意收敛序列 $C_m\to C$，由逐点连续性和支配收敛定理，

$$
\lim_{m\to\infty}p(C_m)
=
\mathbb E\left[
\lim_{m\to\infty}q(C_m;\omega)
\right]
=
p(C).
$$

故 $p(C)$ 连续且非递减。

现取 $0<C_1<C_2$，并固定满足 $q(C_1;\omega)<1$ 的样本。至少存在一个维度 $k$，使该维相交概率严格小于 $1$，从而

$$
\lambda_{R,k}(C_1;\omega)
+
\lambda_{S,k}(C_1;\omega)
<W_k.
$$

两条边不可能同时达到 $L_k^{\max}$，否则 $2L_k^{\max}>W_k$ 将使该维相交概率等于 $1$。所以至少一条边在 $C_1$ 时尚未饱和，并在 $C_2$ 时严格增长。由一维相交概率在 $a+b<W$ 区域内对边长严格递增，该维概率严格增长；其余各维概率为正且非递减，故

$$
q(C_2;\omega)>q(C_1;\omega).
$$

若 $p(C_1)<1$，则集合 $\{\omega:q(C_1;\omega)<1\}$ 具有正概率。在该集合上严格增长，在其余样本上非递减，因此 $p(C_2)>p(C_1)$。乘以正常数 $A$ 即得 $\alpha_{\mathrm{exp}}$ 的结论。$\square$

### 7.3 端点极限

**定理 2**　在 $0<\varepsilon_{\mathrm{geom}}<1/2$ 时，

$$
\lim_{C\downarrow0}p(C)=0,
\qquad
\lim_{C\to\infty}p(C)=1.
$$

**证明。** 固定几乎处处有限且为正的基础随机结果 $\omega$。

当 $C\downarrow0$ 时，

$$
\lambda_{T,k}(C;\omega)
\to0.
$$

每一维的两条边长同时趋于 $0$，因此对应的一维相交概率趋于 $0$，进而

$$
q(C;\omega)\to0.
$$

由 $0\le q\le1$ 和支配收敛定理，得到

$$
p(C)\to0.
$$

当 $C\to\infty$ 时，

$$
\lambda_{T,k}(C;\omega)
\to
L_k^{\max}
=
W_k(1-\varepsilon_{\mathrm{geom}}).
$$

由于 $\varepsilon_{\mathrm{geom}}<1/2$，有

$$
2L_k^{\max}>W_k.
$$

因此每一维的相交概率均趋于 $1$，从而

$$
q(C;\omega)\to1.
$$

再次应用支配收敛定理可得

$$
p(C)\to1.
$$

$\square$

### 7.4 严格内部目标的唯一解

**定理 3**　对任意

$$
0<\alpha_{\mathrm{out}}^\star<A,
$$

存在唯一有限的 $C^*>0$，使得

$$
\alpha_{\mathrm{exp}}(C^*)
=
\alpha_{\mathrm{out}}^\star.
$$

**证明。** 令

$$
p^*
=
\frac{\alpha_{\mathrm{out}}^\star}{A}
\in(0,1).
$$

由定理 2，可取有限的 $C_{\mathrm{lo}}>0$ 和 $C_{\mathrm{hi}}>C_{\mathrm{lo}}$，使得

$$
p(C_{\mathrm{lo}})<p^*<p(C_{\mathrm{hi}}).
$$

由定理 1 的连续性和介值定理，区间 $[C_{\mathrm{lo}},C_{\mathrm{hi}}]$ 中至少存在一个 $C^*$ 满足

$$
p(C^*)=p^*.
$$

因此至少存在一个解，且

$$
\alpha_{\mathrm{exp}}(C^*)
=A p(C^*)
=\alpha_{\mathrm{out}}^\star.
$$

若还存在 $C_1<C_2$ 且 $p(C_1)=p(C_2)=p^*$，则 $p(C_1)=p^*<1$，由定理 1 应有 $p(C_2)>p(C_1)$，矛盾。因此解唯一。$\square$

### 7.5 固定基础样本下的单调性

求解阶段一次性生成基础样本

$$
\omega_1,\ldots,\omega_M,
$$

并在所有候选 $C$ 上复用这些样本。于是

$$
\widehat p_M(C)
=
\frac1M\sum_{j=1}^{M}q(C;\omega_j)
$$

在样本固定后成为确定函数。

由于每个 $q(C;\omega_j)$ 关于 $C$ 非递减，$\widehat p_M(C)$ 也连续且非递减。进一步地，若 $\widehat p_M(C_1)<1$ 且 $C_2>C_1$，则至少一个样本满足 $q(C_1;\omega_j)<1$；由定理 1 证明中的逐样本论证，

$$
q(C_2;\omega_j)>q(C_1;\omega_j),
$$

所以 $\widehat p_M(C_2)>\widehat p_M(C_1)$。因此，每个经验内部目标也具有唯一经验根。复用同一批基础样本使括区间扩张和二分搜索始终基于同一个严格内部单调目标函数。

## 8. 覆盖参数求解

以下简记

$$
\alpha^*
=
\alpha_{\mathrm{out}}^\star,
\qquad
p^*
=
\frac{\alpha^*}{A}.
$$

为区分数学尺度与数值状态，记

$$
q_\theta(\theta;\omega)
=
q(e^\theta;\omega),
\qquad
p_\theta(\theta)
=
p(e^\theta).
$$

$e^\theta$ 在此只是函数复合的数学记号；求解器始终保存 $\theta$，并直接由它计算对数相对边长。

### 8.1 小对象近似与初始值

在固定体积、无形状变化、无饱和且 box 很小时，一维相交概率满足

$$
P_{\mathrm{1D}}(a,b;W)
=
\frac{a+b}{W}
+
O\left(
\frac{(a+b)^2}{W^2}
\right).
$$

因此

$$
p(C)
\approx
C
\left(
 n_R^{-1/d}+n_S^{-1/d}
\right)^d,
$$

并有

$$
\alpha_{\mathrm{exp}}(C)
\approx
\kappa_d C,
$$

其中

$$
\kappa_d
=
A
\left(
 n_R^{-1/d}+n_S^{-1/d}
\right)^d.
$$

由此可取初始覆盖率

$$
C_0
=
\frac{\alpha^*}{\kappa_d}.
$$

为处理极小或极大的尺度，求解器可使用

$$
\theta=
\log C
$$

作为状态，并令

$$
\theta_0
=
\log\alpha^*-
\log\kappa_d.
$$

该近似只用于确定搜索起点，不要求在一般体积与形状分布下保持精确。

### 8.2 对数空间括区间

在固定基础样本下计算

$$
\widehat p_{M,\theta}(\theta)
=
\frac1M\sum_{j=1}^{M}q_\theta(\theta;\omega_j).
$$

精确实数模型中，若

$$
\widehat p_{M,\theta}(\theta_0)<p^*,
$$

则令 $\theta_{\mathrm{lo}}=\theta_0$，并反复执行

$$
\theta\leftarrow\theta+\log2,
$$

直到得到 $\theta_{\mathrm{hi}}$，满足

$$
\widehat p_{M,\theta}(\theta_{\mathrm{hi}})\ge p^*.
$$

若

$$
\widehat p_{M,\theta}(\theta_0)>p^*,
$$

则向相反方向扩张，直到得到

$$
\widehat p_{M,\theta}(\theta_{\mathrm{lo}})
\le
p^*
\le
\widehat p_{M,\theta}(\theta_{\mathrm{hi}}).
$$

固定有限基础样本时，每个 $q_\theta(\theta;\omega_j)$ 在 $\theta\to-\infty$ 时趋于 $0$，在 $\theta\to+\infty$ 时趋于 $1$。因此，样本目标函数也具有相同的端点极限，倍增或减半搜索能够跨过任意严格内部目标。

有限精度求值返回经验均值的包围

$$
[\widehat p_M^-(\theta),\widehat p_M^+(\theta)]
\ni
\widehat p_{M,\theta}(\theta).
$$

方向只在以下情形作出：

$$
\widehat p_M^+(\theta)<p^*
\quad\Longrightarrow\quad
\theta\ \text{位于经验根左侧},
$$

$$
\widehat p_M^-(\theta)>p^*
\quad\Longrightarrow\quad
\theta\ \text{位于经验根右侧}.
$$

若区间包含 $p^*$，则提高算术精度，直到区间能够给出方向，或整个区间已经落入校准残差带 $[p^*-\tau_{\mathrm{train}},p^*+\tau_{\mathrm{train}}]$；后一情形可直接把当前点作为候选。括区间的两个端点在通常情形下分别得到“低于目标”和“高于目标”的区间判定。这样，浮点舍入不会把一次微小的非单调数值波动转化为错误的搜索方向。

若起点恰好满足目标，直接将 $\theta_0$ 作为候选值。实现为 $\theta$ 设置允许范围和最大扩张次数，并在函数值非有限、步进不再改变 $\theta$、数值包围无法收窄或目标未能在允许范围内被括住时返回明确状态。目标函数求值直接使用 $\theta$ 形成对数相对边长，不先计算可能溢出的 $e^\theta$。

### 8.3 对数空间二分

每轮取

$$
\theta_{\mathrm{mid}}
=
\theta_{\mathrm{lo}}
+
\frac{\theta_{\mathrm{hi}}-\theta_{\mathrm{lo}}}{2}.
$$

若

$$
\widehat p_M^+(\theta_{\mathrm{mid}})<p^*,
$$

则更新

$$
\theta_{\mathrm{lo}}
\leftarrow
\theta_{\mathrm{mid}}.
$$

若

$$
\widehat p_M^-(\theta_{\mathrm{mid}})>p^*,
$$

则更新

$$
\theta_{\mathrm{hi}}
\leftarrow
\theta_{\mathrm{mid}}.
$$

若目标落在当前数值包围内，则先提高精度。对包围区间定义经验残差上界

$$
r_{\mathrm{train}}(\theta)
=
\max\left\{
\left|\widehat p_M^-(\theta)-p^*\right|,
\left|\widehat p_M^+(\theta)-p^*\right|
\right\}.
$$

通常在

$$
r_{\mathrm{train}}(\theta_{\mathrm{mid}})
\le
\tau_{\mathrm{train}}
$$

且

$$
\theta_{\mathrm{hi}}-\theta_{\mathrm{lo}}
\le
\tau_\theta
$$

时，取 $\widehat\theta=\theta_{\mathrm{mid}}$。若当前数值包围包含 $p^*$ 且已经整体落入校准残差带，则仅凭前一个条件即可取该点为候选；此时目标函数残差已经直接达到所需精度，不需要依赖参数区间宽度。若中点等于任一端点、函数值非有限、数值包围无法达到要求，或达到最大迭代次数而停止条件尚未成立，则返回相应状态。

在线性概率能够可靠表示时，上述残差可直接由包围端点计算；极小概率下则在对数域进行。若

$$
\ell=\log\widehat p_{M,\theta},
\qquad
\ell^*=\log p^*,
$$

则精确概率残差满足

$$
\log\left|
\widehat p_{M,\theta}-p^*
\right|
=
m+
\operatorname{log1mexp}(-|\ell-\ell^*|),
\qquad
m=\max\{\ell,\ell^*\}.
$$

区间实现对该表达式使用有向舍入。当 $\ell=\ell^*$ 时，概率残差为 $0$，其对数为 $-\infty$。

程序保存并返回 $\widehat\theta$。记

$$
\widehat C=e^{\widehat\theta}
$$

仅用于表示相应的数学尺度，不要求在线性域实际形成 $\widehat C$。

### 8.4 带方差自适应独立认证的求解流程

令

$$
\varepsilon_p
=
\frac{\varepsilon_\alpha}{A}.
$$

求解分为校准求根和独立认证两个阶段。校准样本只用于确定候选覆盖参数；每个候选的认证流在候选固定后才生成，并且不再参与同一候选的选择。

算法控制量满足

$$
R_{\max},S_{\max},M_1,
\mathrm{max\_expand},
\mathrm{max\_bisect}
\in\mathbb N_{>0},
$$

$$
K_1\in\mathbb N,
\qquad
K_1\ge2,
$$

$$
\tau_{\mathrm{train},0}>0,
\qquad
\tau_\theta>0,
\qquad
\theta_{\min}<\theta_{\max}.
$$

第 $r$ 轮采用

$$
M_r=2^{r-1}M_1,
$$

以及

$$
\tau_{\mathrm{train},r}
=
\min\left\{
\frac{\varepsilon_p}{8},
\frac{\tau_{\mathrm{train},0}}{2^{r-1}}
\right\}.
$$

这样，数值求根残差不会占用主要认证容差，并随校准样本量同步收紧。

```text
Algorithm SolveCoverageCertified
输入:
  nR, nS, d, universe, alpha_target
  volume_dist, volume_cv, shape_sigma, eps_geom
  epsilon_alpha, delta
  Rmax, Smax, M1, K1 (K1 >= 2)
  tau_train_0, tau_theta
  theta_min, theta_max, max_expand, max_bisect
输出:
  CERTIFIED(theta, confidence_interval) 或明确状态

1. 稳定计算 A、p_target = alpha_target/A、
   epsilon_p = epsilon_alpha/A
2. 对 r = 1,...,Rmax:
     a. 令 M_r = 2^(r-1) * M1
        tau_train_r = min(epsilon_p/8,
                          tau_train_0/2^(r-1))
     b. 生成或扩展到 M_r 组校准基础样本；
        本轮所有候选 theta 严格复用这些样本
     c. 由小对象近似得到 theta0，并投影到
        [theta_min, theta_max]
     d. 使用经验均值区间构造括区间并执行二分；
        只有当区间整体位于 p_target 的一侧时才更新方向；
        数值包围含 p_target 时提高精度
     e. 在
          r_train <= tau_train_r
        且
          (theta_hi - theta_lo <= tau_theta
           或当前数值包围包含 p_target)
        时得到候选 theta_r；若出现非有限值、浮点停滞、
        数值包围无法收窄、括区间失败或达到迭代上限，
        返回对应状态
     f. 固定 theta_r，启动与全部既有样本独立的认证流
     g. 对 s = 1,...,Smax:
          i.  扩展认证前缀到 K_s = 2^(s-1) * K1
          ii. 令 xi = delta/(Rmax*Smax)
          iii. 以有向舍入或区间算术计算
                 [p_minus, p_plus] 包含认证均值
               以及 V_plus >= 精确样本方差
          iv. 选择属于 [p_minus,p_plus] 的可表示数 p_num，
              并以上向舍入计算
                 r_num = max(p_num-p_minus,
                             p_plus-p_num)
                 r_stat = upward_round(
                   sqrt(2*V_plus*log(4/xi)/K_s)
                   + 7*log(4/xi)/(3*(K_s-1)))
          v.  若 r_num > epsilon_p/8，
               提高精度；仍无法达到时返回
               TARGET_BELOW_NUMERIC_RESOLUTION
          vi. 若
                 abs(p_num-p_target)+r_stat+r_num
                 <= epsilon_p
               则返回
                 CERTIFIED(theta_r,
                   [A*max(0,p_num-r_stat-r_num),
                    A*min(1,p_num+r_stat+r_num)])
          vii.若置信区间与目标容差带
                 [p_target-epsilon_p,
                  p_target+epsilon_p]
               不相交，则结束本轮认证并进入下一校准轮
     h. 若已用尽本轮认证预算且区间仍与目标容差带相交，
        返回 SAMPLE_BUDGET_EXCEEDED
3. 返回 CERTIFICATION_NOT_REACHED
```

第 2.g.vii 步只把已经结束的认证结果用于下一轮校准；下一候选仍由新的认证流评估。因此，自适应轮次不会破坏当前候选与其认证样本之间的独立性。每次目标函数求值只涉及样本中的随机对象对及其 $d$ 个维度，不需要构造或枚举 $n_Rn_S$ 个实际对象对。

输入域、数值范围、样本预算、括区间、二分和认证均具有独立状态，包括 `INVALID_INPUT`、`TARGET_BELOW_NUMERIC_RESOLUTION`、`SAMPLE_BUDGET_EXCEEDED`、`NONFINITE_EVALUATION`、`NUMERICAL_RESOLUTION_EXHAUSTED`、`BRACKET_NOT_FOUND`、`EXPANSION_LIMIT_REACHED`、`BISECTION_LIMIT_REACHED` 与 `CERTIFICATION_NOT_REACHED`。任一状态都不返回带置信精度声明的尺度。

**定理 4（方差自适应独立认证保证）**　上述过程满足

$$
\Pr\left(
\operatorname{CERTIFIED}(\widehat\theta)
\quad\text{且}\quad
\left|
A p(e^{\widehat\theta})
-
\alpha_{\mathrm{out}}^\star
\right|
>
\varepsilon_\alpha
\right)
\le
\delta.
$$

**证明。** 令 $\mathcal H_r^-$ 表示第 $r$ 个候选固定且其认证流尚未生成时的全部历史。条件于 $\mathcal H_r^-$，$\theta_r$ 已经固定，该认证流中的样本仍独立同分布。对任意检查点 $s$，经验 Bernstein 不等式给出

$$
\Pr\left(
\left|
\overline Q_{r,s}
-
p(e^{\theta_r})
\right|
>
r_{\mathrm{EB},r,s}
\ \middle|\ 
\mathcal H_r^{-}
\right)
\le
\xi_{r,s},
$$

其中

$$
\xi_{r,s}
=
\frac{\delta}{R_{\max}S_{\max}}.
$$

数值包围确定性保证

$$
\left|
p_{r,s}^{\mathrm{num}}
-
\overline Q_{r,s}
\right|
\le
r_{\mathrm{num},r,s},
$$

且以 $V_{r,s}^+$ 计算的统计半径不小于精确样本方差对应的经验 Bernstein 半径。若当前检查点的统计事件不发生且认证条件成立，则

$$
\left|
p(e^{\theta_r})-p^*
\right|
\le
\left|
p_{r,s}^{\mathrm{num}}-p^*
\right|
+
r_{\mathrm{stat},r,s}
+
r_{\mathrm{num},r,s}
\le
\varepsilon_p.
$$

对上述条件概率取全期望后，每个检查点的无条件失败概率仍不超过 $\xi_{r,s}$。再对所有至多 $R_{\max}S_{\max}$ 个候选—检查点对应用并合界，错误认证事件的概率不超过

$$
\sum_{r=1}^{R_{\max}}
\sum_{s=1}^{S_{\max}}
\xi_{r,s}
=
\delta.
$$

最后乘以 $A$ 即得结论。$\square$

若存在多次完整执行，第 $u$ 次使用总预算 $\delta^{(u)}$，并要求

$$
\sum_u\delta^{(u)}\le\delta;
$$

每次执行内部再按候选与检查点分配其预算。

### 8.5 确定性批处理

校准阶段的固定样本是指：每个样本索引在所有候选 $\theta$ 上对应完全相同的基础随机变量。样本可常驻内存，也可由可寻址随机数生成器按

$$
(\mathrm{seed},\mathrm{round},j,T,k,\mathrm{variable\ type})
$$

确定性重放。批大小为 $b$ 时，每次目标函数求值按相同索引重建并处理全部校准样本；不同候选参数不得获得不同的随机样本。并行实现采用固定归约顺序，使相同输入产生相同的 $\widehat p_{M,\theta}(\theta)$。

## 9. 完整生成流程

设求解阶段返回经认证的对数覆盖参数 $\widehat\theta$，并以

$$
\widehat C=e^{\widehat\theta}.
$$

表示相应的数学尺度。生成过程直接使用 $\widehat\theta$。分别生成 $R$ 与 $S$；两个集合使用相同的尺度参数、体积分布族、形状分布族、饱和规则和宇宙空间，但彼此独立，并且与校准及认证样本使用独立随机流。

对集合 $T\in\{R,S\}$ 中的每个对象，依次执行：

1. 在对数域计算平均目标物理体积和相对体积

   $$
   \log\bar v_T(\widehat C)
   =
   \widehat\theta+\log V_{\mathcal U}-\log n_T,
   $$

   $$
   \log\bar\nu_T(\widehat C)
   =
   \widehat\theta-\log n_T.
   $$

2. 采样单位均值体积乘子 $Z$，得到

   $$
   \log V
   =
   \log\bar v_T(\widehat C)+\log Z,
   $$

   $$
   \eta_T
   =
   \log\bar\nu_T(\widehat C)+\log Z.
   $$

3. 采样并中心化对数形状变量，得到满足

   $$
   \prod_{k=1}^{d}g_k=1
   $$

   的形状因子；数值计算保留 $h_k=\log g_k$。

4. 在对数域计算相对边长

   $$
   \tau_{T,k}
   =
   \min\{
   \eta_T/d+h_k,
   \operatorname{log1p}(-\varepsilon_{\mathrm{geom}})
   \}.
   $$

   最终相对边长和物理边长为

   $$
   \rho_{T,k}=e^{\tau_{T,k}},
   \qquad
   \lambda_{T,k}=W_k\rho_{T,k}.
   $$

5. 给定完整相对边长向量，在归一化可行起点区间内条件独立均匀采样位置，并映射回 $\mathcal U$，得到 $[\ell_k,u_k)$。

条件于 $\widehat C$，最终生成分布满足

$$
\mathbb E_{\mathrm{gen}}
\left[
\alpha_{\mathrm{out}}
\mid\widehat C
\right]
=
A p(\widehat C).
$$

定理 4 给出的语义是

$$
\Pr_{\mathrm{solver}}\left(
\operatorname{CERTIFIED}(\widehat C)
\quad\text{且}\quad
\left|
\mathbb E_{\mathrm{gen}}
[\alpha_{\mathrm{out}}\mid\widehat C]
-
\alpha_{\mathrm{out}}^\star
\right|
>
\varepsilon_\alpha
\right)
\le
\delta.
$$

该保证针对最终生成分布的条件期望。单次生成数据的输出密度仍是随机变量，其波动取决于数据规模、体积与形状分布、位置分布以及共享对象造成的对象对依赖。

## 10. 复杂度分析

设实际执行 $H\le R_{\max}$ 个候选轮。第 $r$ 轮使用 $M_r$ 个校准样本，括区间产生 $B_r$ 次附加求值，二分产生 $I_r$ 次求值；连同起点求值，校准目标函数共计算

$$
E_r
=
1+B_r+I_r
$$

次。设该轮认证最终读取到检查点 $s_r$，累计认证样本数为

$$
K_{s_r}
=
2^{s_r-1}K_1.
$$

求解与认证总时间为

$$
O\left(
d\sum_{r=1}^{H}
\left[
E_rM_r+K_{s_r}
\right]
\right).
$$

最终生成 $N=n_R+n_S$ 个对象需要 $O(Nd)$ 时间，因此端到端时间为

$$
O\left(
d\sum_{r=1}^{H}
\left[
E_rM_r+K_{s_r}
\right]
+Nd
\right).
$$

若第 $r$ 轮初始括区间宽度为 $\Delta\theta_r$，则仅就区间宽度而言，所需二分次数为

$$
I_r^{\mathrm{width}}
=
\max\left\{
0,
\left\lceil
\log_2
\frac{\Delta\theta_r}{\tau_\theta}
\right\rceil
\right\}.
$$

实际成功路径还需满足校准残差条件，并有

$$
B_r\le\mathrm{max\_expand},
\qquad
I_r\le\mathrm{max\_bisect}.
$$

全部样本常驻内存时，求解工作空间为

$$
O\left(
d\max\{\max_rM_r,K_{S_{\max}}\}
\right).
$$

使用确定性批处理重放时，除最终输出外的工作空间为 $O(bd)$，其中 $b$ 为批大小；最终坐标需要 $O(Nd)$ 存储。

认证复杂度具有实例依赖形式。令

$$
\sigma_q^2(C)
=
\operatorname{Var}(q(C;\omega)).
$$

在固定候选处达到概率半宽 $\eta$ 的几何经验 Bernstein 停止复杂度为

$$
\widetilde O\left(
\frac{\sigma_q^2(C)}{\eta^2}
+
\frac1\eta
\right),
$$

其中 $\widetilde O$ 隐去 $\log(R_{\max}S_{\max}/\delta)$ 以及检查点调度产生的对数因子。由于

$$
\sigma_q^2(C)
\le
p(C)(1-p(C)),
$$

在可认证候选附近取 $\eta=\Theta(\varepsilon_p)$ 可得

$$
K
=
\widetilde O\left(
\frac{A(\alpha_{\mathrm{out}}^\star+\varepsilon_\alpha)}
{\varepsilon_\alpha^2}
+
\frac{A}{\varepsilon_\alpha}
\right).
$$

当目标输出密度与绝对容差保持常数量级时，该项关于 $A$ 近线性增长。若 $p^*=\Theta(1)$，则估计精度仍为 $\Theta(1/A)$，常数方差均值估计本身需要平方级样本量；这一情形由数据规模要求的绝对输出密度精度决定，而不是由分布无关半径造成。

求解过程不包含 $n_Rn_S$ 次实际几何相交判断。对象对位置已经通过一维闭式概率解析积分。以上时间界将选定精度下的一次基本算术视为常数；切换到任意精度算术时，还需乘以相应的位复杂度。

## 11. 数值边界与模型限制

### 11.1 边长饱和

定义全维必交事件

$$
\mathcal E_C
=
\left\{
\forall k:\
\lambda_{R,k}(C)+\lambda_{S,k}(C)\ge W_k
\right\}.
$$

在 $\mathcal E_C$ 上有 $q(C;\omega)=1$，因此

$$
p(C)\ge\Pr(\mathcal E_C).
$$

随着 $C\to\infty$，每条边长趋于

$$
L_k^{\max}
=
W_k(1-\varepsilon_{\mathrm{geom}}),
$$

且 $2L_k^{\max}>W_k$，从而 $\Pr(\mathcal E_C)\to1$。因此，$p(C)\to1$ 的关键不是单独统计已经饱和的边数，而是绝大部分概率质量是否同时在所有维度满足两条边长之和至少为宇宙跨度。

饱和不会破坏单调性，但会带来以下变化：

1. 当 $p(C)$ 进入接近 $1$ 的平台区时，反函数对概率误差更敏感；
2. 原始体积分布和长宽比分布被逐维截断；
3. 不同覆盖率可能对应非常接近的输出密度；
4. 接近最大输出密度的目标可能需要更大的 $\theta$ 搜索范围，并使尺度参数的识别精度下降。

输出密度容差仍由概率残差直接控制；平台区并不必然要求任意缩小 $\theta$ 区间，而是要求求解器同时检查概率残差、参数范围与数值分辨率。

### 11.2 坐标尺度

本文以归一化坐标

$$
x_k'
=
\frac{x_k-u_k^{\min}}{W_k}
$$

作为随机生成模型的规范坐标。体积变量 $\nu$ 表示相对于 $V_{\mathcal U}$ 的体积，形状变量控制未饱和相对边长

$$
\widetilde\rho_k
=
\exp\left(\frac{\eta}{d}+h_k\right).
$$

在逐维饱和以前，

$$
\log\frac{\widetilde\rho_i}{\widetilde\rho_j}
=
h_i-h_j.
$$

逐维饱和后，

$$
\tau_k
=
\min\left\{
\frac{\eta}{d}+h_k,
\tau_{\max}
\right\},
\qquad
\rho_k=e^{\tau_k},
$$

因此相对边长比例为

$$
\log\frac{\rho_i}{\rho_j}
=
\tau_i-\tau_j
$$

即

$$
\log\frac{\rho_i}{\rho_j}
=
\min\left\{
\frac{\eta}{d}+h_i,
\tau_{\max}
\right\}
-
\min\left\{
\frac{\eta}{d}+h_j,
\tau_{\max}
\right\}.
$$

物理坐标中的长宽比为

$$
\frac{\lambda_i}{\lambda_j}
=
\frac{W_i}{W_j}
\exp(\tau_i-\tau_j).
$$

当 $\sigma_{\mathrm{shape}}=0$ 时，对象在归一化坐标中为超立方体；映射回物理坐标后，对象与 $\mathcal U$ 同比例。逐维严格递增的仿射映射保持轴对齐相交关系，因此生成、概率计算和相交判定都可在归一化坐标中进行，仅在输出时映射回物理坐标。

选定的数值格式还应满足 $1-\varepsilon_{\mathrm{geom}}$ 与 $1$ 可区分，并能区分每个输出 box 的两个端点。若不满足，则保留归一化表示或采用更高精度；不把正边长舍入为零。

### 11.3 极低概率与高维

在高维稀疏情形中，

$$
q(C;\omega)
=
\prod_{k=1}^{d}P_{\mathrm{1D},k}
$$

可能极小。此时应在对数域中累计

$$
\log q
=
\sum_{k=1}^{d}\log P_{\mathrm{1D},k},
$$

并使用 `logsumexp` 聚合样本平均。这样可以避免直接概率乘积的下溢，并使极小目标概率参与比较和二分搜索。

对数域处理与有限样本认证承担不同作用：前者保持概率的数值表示，后者量化随机积分误差。即使概率在对数域中可精确表示，有限样本仍可能未充分覆盖对期望有显著贡献的极端体积或形状区域，因此停止条件以独立认证区间为准，而不单独依赖校准残差。

### 11.4 位置模型限制

基础位置模型假设：给定完整边长向量后，每个对象在其可行区域内按维度条件独立均匀放置。以下结构不属于该模型：

1. 空间聚簇与热点；
2. 对象间排斥、遮挡或依赖；
3. $R$ 与 $S$ 之间的位置相关性；
4. 道路、行政区和地图拓扑约束；
5. 非轴对齐矩形和一般多边形；
6. 时间相关性和移动轨迹。

若引入这些结构，可以保留“用标量尺度参数控制对象大小，再反求目标输出密度”的总体思路，但必须重新建立给定几何参数下的相交概率表达式，并重新分析目标函数的单调性与端点性质。

### 11.5 输出密度的含义

本文控制的是经认证覆盖参数下的条件期望

$$
\mathbb E_{\mathrm{gen}}
[\alpha_{\mathrm{out}}\mid\widehat C]
=
A p(\widehat C),
$$

而不是单次生成后的确定值。求解阶段的校准随机性由独立认证控制；最终数据的下列随机性决定单次输出密度围绕该条件期望的波动：

1. 对象体积随机性；
2. 对象形状随机性；
3. 对象位置随机性；
4. 不同对象对之间因共享对象产生的依赖。

因此，目标参数 $\alpha_{\mathrm{out}}^\star$ 应解释为随机模型的期望负载。若研究任务要求每个数据集都具有完全相同的连接基数，则需要使用带全局连接约束的构造，而不能仅依赖独立随机生成模型。

### 11.6 数值协议、随机源与相交语义

基数、宇宙体积与目标概率采用稳定形式计算：

$$
\log A
=
\log n_R+
\log n_S
-
\operatorname{logaddexp}(\log n_R,\log n_S),
$$

$$
\log V_{\mathcal U}
=
\sum_{k=1}^{d}\log W_k,
\qquad
\log p^*
=
\log\alpha_{\mathrm{out}}^\star-
\log A.
$$

小对象初值中的系数按

$$
\log\kappa_d
=
\log A
+
d\,\operatorname{logaddexp}
\left(
-\frac{\log n_R}{d},
-\frac{\log n_S}{d}
\right)
$$

计算。

所有搜索均设置有限的 $\theta$ 范围、最大扩张次数、最大二分次数、非有限值检查和浮点停滞检查。校准方向由经验目标区间与 $p^*$ 的严格分离关系决定；区间包含目标时提高精度，不依据单个舍入值改变二分方向。认证均值与样本方差使用有向舍入或区间算术，统计半径以上向舍入计算 [4]。当 $p^*$ 或 $\varepsilon_\alpha/A$ 低于当前数值类型可可靠表示的范围时，求解器采用更高精度或返回明确状态。

第 5–10 章及定理 1–3 采用连续归一化坐标和半开区间语义。若坐标需要整数化、定点化或十进制舍入，或者目标平台将边界接触计为相交，则应按相应语义重新定义条件相交概率 $q$，并重新分析其连续性、单调性与端点性质。

定理 4 所需的认证接口可以概括为：给定在读取当前认证数据前已经固定的候选参数，认证过程返回区间 $[L,U]$，并满足

$$
\Pr\left(
p(\widehat C)\notin[L,U]
\ \middle|\ 
\widehat C
\right)
\le\xi,
$$

同时数值计算对 $L,U$ 具有确定的包围保证。任何满足该接口的固定样本置信区间或时间一致置信序列都可以代替经验 Bernstein 区间；仅有“估计量取值于 $[0,1]$”和“与候选独立”并不足以推出认证结论。

概率模型假设基础均匀变量与由其产生的对象样本相互独立。确定性伪随机数发生器在种子固定后产生确定序列；即使把种子随机化，也只得到一个联合分布，并不自动证明各认证样本独立同分布。因此，定理 4 的严格概率语义以理想独立随机源为基础。具体实现可以把高质量伪随机数发生器视为该模型的计算实现，但这一做法属于随机源假设，而不是由区间算术推出的结论。校准、认证和最终生成应使用相互分离的随机流，并明确随机数发生器、子流构造与分布变换算法。区间算术只控制数值舍入误差，不把伪随机序列转化为独立随机变量。

严格的软件语义可选择以下两种口径之一：

1. 以实数随机模型为对象，采样例程提供所需的独立样本，特殊函数与几何核提供可证明的分布误差和舍入包围；
2. 以具体实现诱导的随机分布为对象，并为该联合分布建立适用的浓缩界；只有当实现确实产生认证所需的独立同分布 $Q_j$ 时，定理 4 的经验 Bernstein 形式才可直接沿用。

定理 1–3是实数随机模型的结构结论。若数值求解以该模型为目标，区间计算负责包围实数目标函数并保证搜索方向可靠。若直接以具体实现诱导的分布为目标，并且认证样本满足相应浓缩界，则定理 4 的证明结构仍适用于该分布的对象对相交概率；有限精度网格上的解可能不再具有实数意义下的连续性与唯一性，此时算法返回的是通过给定容差认证的可表示尺度。

## 参考文献

1. O. Günther, V. Oria, P. Picouet, J.-M. Saglio, and M. Scholl. *Benchmarking Spatial Joins À La Carte*. Proceedings of the 10th International Conference on Scientific and Statistical Database Management, 1998, pp. 32–41.
2. O. Günther, V. Oria, P. Picouet, J.-M. Saglio, and M. Scholl. *Benchmarking Spatial Joins À La Carte*. *International Journal of Geographical Information Science*, 13(7), 1999, pp. 639–655. <https://doi.org/10.1080/136588199241049>
3. L. Devroye. *Non-Uniform Random Variate Generation*. Springer-Verlag, 1986. <https://doi.org/10.1007/978-1-4613-8643-8>
4. N. J. Higham. *Accuracy and Stability of Numerical Algorithms*, 2nd ed. SIAM, 2002. <https://doi.org/10.1137/1.9780898718027>
5. C. P. Robert. *Simulation of Truncated Normal Variables*. *Statistics and Computing*, 5, 1995, pp. 121–125. <https://doi.org/10.1007/BF00143942>
6. J. del Castillo. *The Singly Truncated Normal Distribution: A Non-Steep Exponential Family*. *Annals of the Institute of Statistical Mathematics*, 46(1), 1994, pp. 57–66. <https://doi.org/10.1007/BF00773592>
7. A. Maurer and M. Pontil. *Empirical Bernstein Bounds and Sample Variance Penalization*. Proceedings of the 22nd Annual Conference on Learning Theory, 2009. <https://arxiv.org/abs/0907.3740>
8. V. Mnih, C. Szepesvári, and J.-Y. Audibert. *Empirical Bernstein Stopping*. Proceedings of the 25th International Conference on Machine Learning, 2008, pp. 672–679. <https://doi.org/10.1145/1390156.1390241>
9. N. An, Z.-Y. Yang, and A. Sivasubramaniam. *Selectivity Estimation for Spatial Joins*. Proceedings of the 17th International Conference on Data Engineering, 2001, pp. 368–375. <https://doi.org/10.1109/ICDE.2001.914849>
