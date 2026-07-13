# ANCHOR 实验设置

本实验设置围绕 `AC`、`AS`、`SweepRT` 与 `LiftedRT` 的运行时间和峰值内存展开，共包含十二组主实验：

| 数据集 | 固定 workload | 主实验 |
|---|---|---|
| Alacarte | 平衡合成盒集合 | $N$-sweep、$t$-sweep、$\alpha$-sweep、shape-sweep、$d$-sweep |
| CMAB-1M | 广东省 943285 个建筑物构成的百万级 2D workload | level-sweep、$t$-sweep |
| GeoLife-3D-1M | 同一批一百万个点生成的 3D workload | level-sweep、$t$-sweep |
| GeoLife-4D-1M | 与 3D 版本共享对象 id 的 4D workload | level-sweep、$t$-sweep |
| COCO-1M | 一个确定性的 100-image、1000000-proposal 3D workload | $t$-sweep |

主比较指标为 `OneShotTime(t)`、`PeakMemoryTotal` 与 `PeakMemoryIncremental`；所有阶段时间、准备后查询时间和辅助结构内存均作为解释性指标单独报告。

## 1. 问题定义与统一语义

输入为两个 $d$ 维坐标轴对齐盒集合：

$$
R=\{r_1,\ldots,r_{|R|}\},\qquad S=\{s_1,\ldots,s_{|S|}\}.
$$

总输入规模统一记为：

$$
N:=|R|+|S|.
$$

平衡 workload 满足：

$$
|R|=\left\lfloor\frac{N}{2}\right\rfloor,
\qquad
|S|=N-|R|.
$$

所有核心实验统一采用半开盒语义：

$$
b=\prod_{j=1}^d [L_j(b),U_j(b)).
$$

两个盒相交当且仅当它们在每一维上严格重叠：

$$
\max(L_j(r),L_j(s)) < \min(U_j(r),U_j(s)),\qquad j=1,\ldots,d.
$$

连接结果定义为：

$$
J=\{(r,s)\in R\times S:r\cap s\neq\emptyset\},
$$

并记：

$$
W:=|J|.
$$

采样目标是返回长度为 $t$ 的有序、带放回、独立均匀样本序列：

$$
X_1,\ldots,X_t \overset{i.i.d.}{\sim} \operatorname{Unif}(J).
$$

即对任意 $(r,s)\in J$：

$$
\Pr[X_i=(r,s)]=\frac{1}{W}.
$$

特殊情况如下：

- 若 $t=0$，返回空样本序列。
- 若 $W=0$ 且 $t>0$，该运行的状态记为 `EMPTY-JOIN`。

输出密度定义为：

$$
\alpha=\frac{|J|}{|R|+|S|}=\frac{W}{N}.
$$

因此在任意平衡 workload 中均有：

$$
W=\alpha N.
$$

对象身份始终保留。即使两个对象具有完全相同的几何坐标，只要对象 id 不同，它们仍是不同的输入对象与不同的连接结果。所有排序均以全局对象键作为最终 tie-breaker。

每个 workload 的端点表示在生成后冻结：合成数据与连续坐标真实数据使用 IEEE 754 binary64，整数时空数据使用有符号 64 位整数。算法只对冻结后的端点做精确全序比较；任何坐标转换、量化与分布采样均在性能计时区间之外完成。

## 2. 主实验算法集合

对任意维度 $d$，主实验算法集合为：

$$
\mathcal A_d=\{\mathrm{LiftedRT},\mathrm{SweepRT},\mathrm{AS},\mathrm{AC}\}.
$$

其中：

| 名称 | 含义 |
|---|---|
| `LiftedRT` | 静态 $2d$ 维提升范围树采样器 |
| `SweepRT` | 两遍扫描线 / 动态范围树采样器 |
| `AS` | ANCHOR-Streaming |
| `AC` | ANCHOR-Compiled |

`LiftedRT` 采用静态 $2d$ 维提升：在 $S$ 侧建立静态范围树；对每个 $r\in R$ 计算 $c_r=|\{s\in S:r\cap s\neq\emptyset\}|$；以 $c_r$ 为外层权重选择 $r$，再在对应的范围查询结果中均匀选择 $s$。该算法使用范围树基线的标准几何提升，并同时支持外层权重采样和内层均匀邻居采样。

实验规则如下：

1. 每个实验配置运行该维度下的所有合法算法。
2. ANCHOR 系列固定为 `AS` 和 `AC`，不随维度改变。
3. 若某算法出现超时或内存溢出，保留该运行的状态和诊断信息。

---

## 3. 时间指标与计时边界

所有时间测量均从两侧原始盒数组已经驻留内存且完成预触页之后开始。调用方预先分配并预触长度为 $t$ 的统一输出缓冲区；缓冲区分配不计时，向其中写入全部 $t$ 个连接对计入算法时间。

以下步骤不计入算法时间：

- 合成数据生成与 Alacarte coverage 求解；
- 文件读取、二进制映射建立和输入页预触；
- Parquet / CSV / 图像解码；
- 数据集下载与清洗；
- 坐标转换、量化、子集构造和盒构造；
- 输出校验、成员检查、校验和计算与结果写盘。

计时使用单调高分辨率时钟，例如 Linux `CLOCK_MONOTONIC_RAW`。主时间区间内不执行 RSS 查询、日志写盘、正确性检查或其他外部监控调用。

### 3.1 主时间指标

四个算法统一使用：

$$
\operatorname{OneShotTime}(t).
$$

`OneShotTime(t)` 定义为：从相同的内存原始盒数组与空输出缓冲区开始，完成空盒检查、算法所需的全部排序和索引构造、得到精确 $W=|J|$，并生成 $t$ 个样本所需的单次端到端核心时间。

每个 $t$ 对应一个全新进程和一套全新的算法内部状态。任何算法都不得复用另一个 $t$ 运行中构建的数据结构。

不同算法的 `OneShotTime(t)` 边界如下：

| 算法 | `OneShotTime(t)` 包含 |
|---|---|
| `LiftedRT` | 空盒检查；提升点构造；在 $S$ 侧构建静态 $2d$ 维范围树；计算全部 $c_r$；得到 $W$；构建外层精确权重索引；生成并写入 $t$ 个样本。 |
| `SweepRT` | 空盒检查；构造并排序全部事件；构建两侧动态 range-tree skeleton、关联数组与 Fenwick 结构；第一遍扫描得到全部 START 事件块权重及 $W$；构建事件权重索引；分配输出位置；第二遍扫描并生成 $t$ 个样本。 |
| `AS` | 空盒检查；构造顶层排序视图；直接执行一次完整的 `AS.Sample(t)`，由该调用内部完成递归权重计算、配额分配、活动路由、终端采样与输出写入，并同时返回精确 $W$。`OneShotTime(t)` 之前不得额外执行独立 `Count`。 |
| `AC` | 空盒检查；构造顶层排序视图；完整编译 ANCHOR 终端原子及全局精确权重索引并得到 $W$；生成并写入 $t$ 个样本。 |

主时间图、主表和算法间 speedup 均以 `OneShotTime(t)` 为准。

### 3.2 阶段时间与辅助时间指标

对具有可复用准备态的算法，在同一次 one-shot 运行内部记录嵌套阶段时间：

| 算法 | `PrepareTime` | `QueryTime(t)` |
|---|---|---|
| `LiftedRT` | 从原始盒数组到静态范围树、全部 $c_r$、精确 $W$ 和外层索引全部就绪。 | 从准备态生成并写入 $t$ 个样本。 |
| `SweepRT` | 事件构造与排序、动态结构构建、第一遍扫描、精确 $W$ 和事件权重索引全部就绪。 | 输出位置分配及完整第二遍扫描。 |
| `AC` | 从原始盒数组到全部持久终端原子、精确 $W$ 和全局权重索引全部就绪。 | 从编译态生成并写入 $t$ 个样本。 |
| `AS` | 不定义持久准备态。 | 不定义。 |

`PrepareTime` 与 `QueryTime(t)` 只用于解释瓶颈；`OneShotTime(t)` 由外层单一计时区间直接测得，不以两个阶段计时值相加代替。

另设两个独立辅助实验：

1. `CountOnlyTime`：从原始盒数组开始，只计算精确 $W$。AS 在该实验中执行独立 `Count`；该数值不与 AS 的 `OneShotTime(t)` 相加。
2. `PreparedQueryTime(t)`：在准备态已经存在时发出一次长度为 $t$ 的查询。该指标适用于 `AC`、`LiftedRT` 和 `SweepRT`；AS 记为 `N/A`。

`CountOnlyTime` 与 `PreparedQueryTime(t)` 在独立进程中测量，不进入主 one-shot 排名。对于 $t$-sweep，主图同时给出 `OneShotTime(t)`；`PreparedQueryTime(t)` 作为辅助图展示可复用索引上的纯查询扩展性。

`CountOnlyTime` 的算法边界为：

| 算法 | `CountOnlyTime` 包含 |
|---|---|
| `LiftedRT` | 提升点与静态 range tree 构建、全部 $c_r$ 计数及 $W$ 求和；不生成外层采样请求。 |
| `SweepRT` | 事件构造与排序、动态 skeleton 构建和第一遍扫描；不执行位置分配与第二遍。 |
| `AS` | 从顶层排序视图开始执行一次独立的递归 `Count`。 |
| `AC` | 完整编译终端原子并求和得到 $W$；该算法没有另一个更轻的持久编译语义。 |

### 3.3 输出、随机性与计时一致性

所有算法使用相同的输出记录格式：每个样本由两个 64 位对象 id 构成，共 16 bytes。输出缓冲区在计时前完整预触，避免按需分配和首次缺页成为算法间差异。

计时结束后计算输出校验和并执行成员检查。编译器不得消除输出写入；程序在计时区间外消费最终校验和。若运行未产生完整的 $t$ 个输出或未返回精确 $W$，该时间记录不得标记为 `OK`。

## 4. 峰值内存指标与双运行协议

每个实验配置、算法、数据标识和采样长度对应一个逻辑运行单元。为避免内存监测改变主计时，同一逻辑运行单元由两个独立进程执行：

1. `measurement_mode="time"`：只记录第 3 节的时间指标，不在主时间区间内读取内存计数器。
2. `measurement_mode="memory"`：重复完全相同的 `OneShot` 流程，只记录峰值内存；该进程中的时间仅作为诊断字段。

两个模式使用相同的冻结 workload、算法参数、$t$ 与采样主种子，不共享运行时数据结构。time run 与 memory run 分别执行一次；二者是同一逻辑运行单元的两种测量模式，不作为统计重复。

### 4.1 输入与输出基线

性能 workload 保存为无压缩规范二进制文件。加载器把文件直接读入最终的连续匿名内存数组，不建立第二份坐标副本；文件句柄和 I/O 缓冲区在建立基线前释放。memory run 由一个算法子进程和一个外部监控进程组成。算法子进程在 `OneShot` 期间不得创建额外工作进程；监控进程通过 `/proc/<pid>/status` 中的 `VmRSS` 读取目标算法进程的当前 resident set size，并统一换算为 bytes。

进入算法前通过控制管道依次完成以下握手：

1. 算法进程加载并预触全部输入页，发送 `INPUT_READY`；监控进程立即读取一次 RSS，记录为 `InputMemory`。
2. 算法进程分配并预触统一输出缓冲区，清空算法内部状态，发送 `BASELINE_READY`；监控进程立即读取一次 RSS，记录为 `BaselineMemory`。
3. 监控进程确认基线记录完成后发送 `START_ACK`；算法进程随后开始完整 `OneShot`。
4. `AC`、`LiftedRT` 和 `SweepRT` 的准备态建立完成时，算法进程发送 `PREPARE_READY`；监控进程立即读取一次 RSS，记录为 `MemoryAfterPrepare`。AS 不发送该事件。
5. `OneShot` 结束后，算法进程发送 `ONESHOT_DONE` 并等待确认；监控进程执行最后一次 RSS 读取后发送 `DONE_ACK`，算法进程方可退出。

因此：

- `InputMemory` 包含程序运行时与完整输入数组；
- `BaselineMemory` 进一步包含统一的 $16t$ bytes 输出缓冲区；
- 两者均不包含算法索引、排序副本、路由缓冲区、range tree、Fenwick tree、alias 表或递归工作区。

### 4.2 固定间隔 RSS 轮询

memory run 在收到 `START_ACK` 后开始轮询。正式实验固定：

$$
\texttt{memory\_poll\_interval\_ms}=5.
$$

监控进程使用单调时钟按绝对时间点调度，每隔 5 ms 读取一次目标算法进程的 `VmRSS`，直到完成第 4.1 节的最后一次读取。若某次读取因瞬时调度延迟晚于目标时间点，不补做密集追赶读取，而是记录实际采样时间并继续到下一个绝对时间点。监控进程与算法进程固定在不同的物理 CPU 核上；机器无法提供独立监控核时，必须在 machine manifest 中记录该事实。

将第 4.1 节记录的 `BaselineMemory` 记为 $M_0$，将随后周期性读取、`PREPARE_READY` 立即读取和 `ONESHOT_DONE` 最后读取所得的 RSS 依次记为：

$$
M_0,M_1,\ldots,M_q.
$$

监控进程自身的 RSS 不计入这些样本。

主内存指标为：

| 指标 | 定义 |
|---|---|
| `PeakMemoryTotal` | $\max_{0\le k\le q}M_k$，即从基线建立后到 one-shot 结束期间观测到的最大进程 RSS bytes。 |
| `PeakMemoryIncremental` | $\max\{0,\mathrm{PeakMemoryTotal}-\mathrm{InputMemory}\}$，包含统一输出缓冲区与算法新增内存。 |
| `PeakMemoryAux` | $\max\{0,\mathrm{PeakMemoryTotal}-\mathrm{BaselineMemory}\}$，近似对应算法辅助结构与工作区。 |
| `ProcessMaxRSS` | 进程结束后由 `wait4` / `getrusage` 得到的生命周期最大 RSS，只作一致性诊断。 |
| `MemoryAfterPrepare` | `AC`、`LiftedRT` 和 `SweepRT` 在准备态建立后由监控进程立即读取的 RSS；AS 记为 `N/A`。 |

主内存图至少报告：

$$
\operatorname{PeakMemoryTotal}
\qquad\text{与}\qquad
\operatorname{PeakMemoryIncremental}.
$$

`PeakMemoryAux` 用于对照理论辅助空间。由于轮询只能观察离散采样时刻，`PeakMemoryTotal` 是固定 5 ms 采样间隔下的峰值 RSS 估计；它可能低于两个采样点之间的瞬时真实峰值。所有正式结果必须使用相同的 RSS 数据源、采样间隔和监控实现，不得在同一组图表中混用不同内存测量后端。

每条 memory record 额外保存：

```text
memory_measurement_backend = procfs_vmrss_polling
memory_poll_interval_ms = 5
peak_is_sampled = true
rss_sample_count = q + 1
```

### 4.3 内存限制与可用性要求

`memory_cap_bytes` 仍作为整套实验统一的机器级内存阈值，并在 machine manifest 中以明确字节数给出。监控进程在每次轮询后检查：

$$
M_k>\texttt{memory\_cap\_bytes}.
$$

若该条件成立，监控进程先向整个算法进程组发送 `SIGTERM`，短暂等待后仍未退出则发送 `SIGKILL`，运行状态记为 `MEMORY-CAP-EXCEEDED`。该阈值是基于离散 RSS 轮询的软限制，实际占用可能在两个采样点之间短暂超过阈值；结果中必须保留最后观测 RSS 和终止信号。

宿主机不得为实验进程启用交换空间。若分配器返回不可恢复的内存失败，或算法进程由系统 OOM 机制终止，状态记为 `OOM`。`MEMORY-CAP-EXCEEDED` 与 `OOM` 分开记录。

若运行环境无法读取目标进程的 `/proc/<pid>/status`，或 `VmRSS` 字段在 memory run 中不可用，对应状态记为 `MEMORY-MEASUREMENT-UNAVAILABLE`，该运行不得生成主峰值内存结果。

### 4.4 两种测量模式的一致性

若同一个逻辑运行单元的 time run 与 memory run 均正常完成，则二者必须得到相同的精确 $W$ 和相同长度的输出；若 $W$ 不一致，标记为 `COUNT-MISMATCH`，若输出校验失败，标记为相应正确性状态。若其中一个模式发生 `TO`、`OOM`、`MEMORY-CAP-EXCEEDED` 或内存测量不可用，则分别保留两个模式的实际状态，不强制把另一个模式改成同一失败状态。

主结果表中：

- 时间只使用 `measurement_mode="time"` 的正常运行；
- 内存只使用 `measurement_mode="memory"` 的正常运行；
- 两种模式的原始记录全部保留，并通过同一个 `run_group_id` 关联。

## 5. 执行环境、随机性、单次运行与失败状态

### 5.1 固定执行环境

主实验采用与理论模型一致的顺序实现：每个算法进程只使用一个软件线程，并固定到同一个物理 CPU 核及其所在 NUMA 节点。所有输入页和算法分配均绑定到该 NUMA 节点。实验期间同一物理核的 SMT sibling 不运行其他任务。

整套实验固定并记录以下环境：

- CPU 型号、微码、socket、物理核与 NUMA 拓扑；
- 内存容量、频率与通道配置；
- 操作系统、Linux kernel、`/proc` 文件系统与 RSS 轮询实现版本；
- 编译器、链接器、优化参数和目标指令集；
- 内存分配器及其配置；
- 随机数发生器与版本；
- CPU governor、Turbo/boost、SMT、THP 与 swap 状态；
- 算法代码 commit、第三方库 commit 和构建产物 SHA-256。

主实验使用固定频率策略：CPU governor 设为 `performance`，Turbo/boost 关闭，透明大页设为 `never`，swap 关闭。任何环境项变化都产生新的 `machine_id`，不得与原结果直接合并。

四个算法编译进同一个 benchmark executable，共享输入层、输出层、计时器、精确随机原语、排序库和内存分配器。每个 measured run 使用全新进程，输入预触完成后才开始计时。每个参数配置只执行一次 time run 和一次 memory run；四个算法按 `AC`、`AS`、`SweepRT`、`LiftedRT` 的固定顺序运行，并把实际顺序写入原始结果。

### 5.2 稳定哈希与数据清单

全文的 `StableHash` 统一定义为：将字段编码为键排序、无多余空白的 UTF-8 canonical JSON，计算 SHA-256，并把前 128 bits 按大端无符号整数解释为排序键。若 128-bit 前缀碰撞，则依次使用完整 SHA-256 和原始稳定对象键打破平局。

每个 workload 发布独立 manifest，至少记录：

- 原始数据版本、下载地址标识与原始文件 SHA-256；
- 预处理代码 commit；
- 坐标系、单位、量化和区间转换规则；
- $R/S$ 对象 id 清单或其 SHA-256；
- 最终规范二进制 workload 文件 SHA-256；
- $N,|R|,|S|,d$、端点类型和坐标范围。

没有完整 manifest 和最终 workload checksum 的数据不得进入主实验。

### 5.3 数据随机性、采样随机性与单次运行

三个标识彼此独立：

- `data_seed`：只决定 Alacarte 合成数据；
- `sample_seed_id`：只决定采样算法的随机选择；
- `process_repeat_id`：为保持结果模式稳定而保留，本实验固定为 $0$，不表示额外重复。

合成数据固定使用：

$$
\texttt{data\_seed}=0.
$$

全部主性能实验固定使用：

$$
\texttt{sample\_seed\_id}=0,
\qquad
\texttt{process\_repeat\_id}=0.
$$

采样主种子定义为：

$$
\operatorname{StableHash}(
\texttt{experiment\_id},
\texttt{dataset\_id},
\texttt{workload\_id},
\texttt{algorithm},
\texttt{sample\_seed\_id}
).
$$

同一 workload 的 $t$-sweep 不把 $t$ 写入主种子，使不同 $t$ 从同一算法随机主流开始；各算法内部使用带 domain tag 的独立子流区分外层权重抽样、块内抽样、配额和洗牌。time run 与 memory run 使用完全相同的主种子。

每个参数配置、算法和测量模式只运行一次。对一个逻辑运行单元，唯一的一次 time run 用于时间指标，唯一的一次 memory run 用于峰值内存指标；不再执行额外 timing repeats、memory repeats 或多数据种子重复。

### 5.4 Timeout、整数安全与失败状态

所有算法 measured run 的 timeout 统一为：

$$
900\ \mathrm{seconds/run}=15\ \mathrm{minutes/run}.
$$

该限制适用于 time run、memory run，以及单独执行时的 `CountOnlyTime` 和 `PreparedQueryTime`。timeout 必须作用于整个算法进程组；到达 900 seconds 后先发送 `SIGTERM`，短暂等待后仍未退出则发送 `SIGKILL`。输入映射、预触、数据下载、数据预处理和结果写盘不计入算法运行的 900 seconds。

连接计数、块权重、alias 构造中的缩放质量、配额和前缀和统一使用 checked unsigned 128-bit integer；写入较窄类型前必须验证范围。任何溢出返回 `INTEGER-OVERFLOW`，不得发生静默回绕。

失败状态包括：

| 状态 | 含义 |
|---|---|
| `OK` | 正常完成，并通过全部运行后检查。 |
| `EMPTY-JOIN` | $W=0$ 且 $t>0$。 |
| `OOM` | 分配器返回不可恢复的内存失败，或算法进程由系统 OOM 机制终止。 |
| `MEMORY-CAP-EXCEEDED` | RSS 轮询监控观测到内存超过 `memory_cap_bytes`，并终止算法进程组。 |
| `TO` | 算法进程组运行超过 900 seconds。 |
| `COUNT-MISMATCH` | 不同算法或两种测量模式得到的 $W$ 不一致。 |
| `MEMBERSHIP-MISMATCH` | 至少一个 sampled pair 不满足相交谓词。 |
| `OUTPUT-LENGTH-MISMATCH` | 未产生恰好 $t$ 个输出。 |
| `UNIFORMITY-VALIDATION-FAILED` | 小规模分布验证未通过预设检验。 |
| `INTEGER-OVERFLOW` | 精确整数或中间量超出实现范围。 |
| `NUMERIC-DEGENERACY` | 预处理后出现空盒、非有限端点或无法表示的正边长。 |
| `DATASET-CONSTRUCTION-FAILED` | 固定规则下无法构造规定规模的 workload。 |
| `MEMORY-MEASUREMENT-UNAVAILABLE` | 目标进程当前 RSS 无法由规定的 procfs 接口读取。 |

OOM、`MEMORY-CAP-EXCEEDED`、TO 和其他失败点保留在主图与原始数据中，不以删除失败运行的方式形成仅含成功样本的扩展性曲线。speedup 只在同一 workload 和同一参数配置下双方均为 `OK` 时计算。

### 5.5 轻量级一键运行脚本

仓库根目录必须提供可执行脚本：

```bash
./run_all_lite.sh
```

该脚本是十二组实验的统一轻量级入口。`lite` 仅表示每个参数配置单次执行并默认跳过辅助实验，不缩减第 7 节和第 8 节规定的 sweep 参数网格。默认调用形式为：

```bash
./run_all_lite.sh \
  --data-root ./data \
  --results-root ./results/lite
```

脚本按以下顺序串行执行，不并行运行两个 benchmark 进程：

1. 检查编译产物、系统环境、可用磁盘空间、`/proc` RSS 接口和结果目录。
2. 检查真实 workload 的最终 checksum；若任一真实 workload 缺失或校验失败，调用第 8.1.1 节规定的 `scripts/data/prepare_real_data.sh --all` 完成 HF 成品资产获取、校验、确定性导入和规范二进制写出。
3. 以 `data_seed=0` 生成第 7 节全部 Alacarte sweep 所需的冻结 workload。
4. 依次执行 Alacarte、CMAB-1M、GeoLife-3D-1M、GeoLife-4D-1M 和 COCO-1M；每个数据集内部按第 7 节和第 8 节列出的完整 sweep 参数顺序运行。
5. 对每个参数配置依次运行 `AC`、`AS`、`SweepRT` 和 `LiftedRT`。每个算法执行一次 `measurement_mode="time"` 和一次 `measurement_mode="memory"`，不执行额外重复。
6. 每个算法 measured run 的 timeout 固定为 900 seconds。出现 `TO`、`OOM`、`MEMORY-CAP-EXCEEDED` 或其他算法级失败时，记录状态和日志后继续下一个算法或配置。
7. 默认只运行主时间和主内存流程，不额外运行 `CountOnlyTime` 与 `PreparedQueryTime`；这些辅助实验由独立命令显式启动。

数据下载与预处理不属于算法 measured run，不使用 900-second benchmark timeout；其失败使对应数据集状态记为 `DATASET-CONSTRUCTION-FAILED`，并跳过依赖该 workload 的实验。脚本至少生成：

```text
results/lite/raw/results.jsonl
results/lite/raw/results.csv
results/lite/logs/
results/lite/manifests/
```

每条记录包含第 9 节规定的字段。脚本再次运行时，checksum 已通过的锁定成品资产和最终 workload 可以复用；任何不完整下载只保留为临时文件，不得被 benchmark 读取。

## 6. 正确性验证

正确性验证与性能测量使用不同进程和不同结果表。验证时间与验证内存不进入任何主性能指标。

### 6.1 确定性单元测试与小规模穷举

以下组件必须分别通过确定性单元测试：

- `UniformInteger` 的拒绝采样边界；
- exact integer alias 的质量守恒；
- multinomial quota 与 Fisher--Yates；
- Fenwick prefix sum 与 active rank-select；
- SweepRT 的事件顺序和两遍活跃集合；
- LiftedRT 的严格开边界提升；
- ANCHOR 的规范覆盖、局部所有权、终端原子解码和局部数组生命周期。

对端点集合 $\{0,1,2,3,4\}$ 上的小规模一维区间进行穷举组合，并构造 $d\in\{1,2,3,4\}$ 的笛卡尔测试族，覆盖：

- 重复几何坐标但对象 id 不同；
- 大量相同左端点或右端点；
- 仅端点接触；
- 严格嵌套；
- 空连接、完全连接与单结果连接。

在可物化实例上显式构造：

$$
J=\{(r,s):r\in R,s\in S,r\cap s\neq\emptyset\},
$$

并检查：

1. $W_{\mathrm{alg}}=|J|$；
2. 所有输出 pair 均属于 $J$；
3. $t=0$ 时输出为空；
4. $W=0,t>0$ 时返回 `EMPTY-JOIN`；
5. time run 与 memory run 的 $W$、状态和输出长度一致。

推荐验证规模为 $|R|=|S|\le 10^3$；真实数据从每个最终 workload manifest 中按稳定对象键取可物化子集。

### 6.2 均匀性与独立性实现检验

对满足 $2\le W\le 2000$ 的物化实例，每个算法至少生成：

$$
T_{\mathrm{val}}=\min\{10^7,\max\{10^6,1000W\}\}
$$

个样本。逐 pair 统计频率并执行 Pearson goodness-of-fit 检验；所有检验采用 family-wise level $0.01$ 的 Holm 校正。另将 pair 通过固定 `StableHash` 映射到 $B=128$ 个桶，对相邻输出的 $B\times B$ 转移频数执行独立性检验，并报告 lag-1 bucket autocorrelation。

统计检验用于识别实现偏差，不替代数学证明。任一检验失败时，该构建版本不得生成主性能结果，除非定位到检验功效或多重比较设置问题并重新冻结验证协议。

### 6.3 百万级运行后检查

百万级 workload 不物化完整连接。每个 `OK` 运行在计时结束后执行：

1. 四个算法的精确 $W$ 一致性检查；
2. 对全部 $t$ 个输出直接检查严格半开相交谓词；
3. 检查对象 id 位于正确颜色集合；
4. 检查输出长度恰为 $t$；
5. 计算输出序列 SHA-256 作为运行记录。

成员检查在计时区间外完成，但任何失败都会使对应性能记录失效。

## 7. 合成数据实验

### 7.1 通用设置

除非某个 sweep 明确改变参数，合成数据实验使用以下默认设置：

| 参数 | 默认值 |
|---|---:|
| 总输入规模 $N=|R|+|S|$ | $2\times10^5$ |
| $|R|$ | $10^5$ |
| $|S|$ | $10^5$ |
| $d$ | $2$ |
| $t$ | $10^5$ |
| target $\alpha$ | $10$ |
| universe | $[0,1)^d$ |
| endpoint dtype | IEEE 754 binary64 |
| volume distribution | fixed |
| `shape_sigma` | $0$ |
| `data_seed` | $0$ |

所有合成数据实验采用 one-factor-at-a-time 设计：每次只改变一个主参数，其余参数保持默认值。

对每个参数组合，Alacarte solver 在性能实验之前求出 coverage 参数，并使用与求解器分离的最终生成随机流产生 $R,S$。求解、认证和数据生成时间均不计入四个采样算法的运行时间。

最终 workload 以 binary64 端点冻结。生成后必须检查：

- 所有端点有限；
- 每一维满足 $L_j<U_j$；
- 不存在 binary64 舍入造成的零长度盒；
- workload checksum 与 manifest 一致。

若出现无法表示的正边长，返回 `NUMERIC-DEGENERACY`，不得把端点静默调整为相等或改变目标分布。Alacarte 的目标密度用于设定生成尺度；其认证标签描述生成模型的期望密度，不直接替代 binary64 workload 上的实测密度。性能分析以最终冻结 workload 上的精确计数为准。每个点同时记录：

$$
\alpha_{\mathrm{target}},
\qquad
\alpha_{\mathrm{realized}}=\frac{W}{N}.
$$

固定 `data_seed=0`。$t$-sweep 只生成一份 $R,S$，所有 $t$ 取值复用同一个 workload 文件与同一个精确 $W$。其他 sweep 在主参数变化时重新生成数据。

每个合成 workload 还记录：

- 各维相对边长分位数；
- 盒体积分位数；
- 最大/最小边长比的中位数、90%、95% 和 99% 分位数；
- 每维发生饱和的对象比例；
- 实际重复端点比例。

---

### G1. 输入规模扩展：$N$-sweep

目的：评估总输入规模变化下的时间与峰值内存扩展性。

固定：

$$
d=2,\qquad t=10^5,\qquad \alpha=10,\qquad \texttt{shape\_sigma}=0.
$$

变化：

$$
N\in\{2\times10^4,10^5,2\times10^5,10^6,2\times10^6\}.
$$

每个点采用平衡输入：

$$
|R|=\left\lfloor\frac{N}{2}\right\rfloor,
\qquad
|S|=N-|R|.
$$

---

### G2. 采样规模扩展：$t$-sweep

目的：评估采样数量对 one-shot 时间、准备后查询时间和峰值内存的影响。

固定：

$$
N=2\times10^5,\qquad d=2,\qquad \alpha=10,\qquad \texttt{shape\_sigma}=0.
$$

变化：

$$
t\in\{10^3,10^4,10^5,10^6,10^7\}.
$$

在固定 `data_seed=0` 下，全部 $t$ 运行复用同一份 $R,S,W$。主图报告 `OneShotTime(t)`；`AC`、`SweepRT` 和 `LiftedRT` 另报告 `PreparedQueryTime(t)`。

---

### G3. 输出密度扩展：$\alpha$-sweep

目的：评估输出密度变化下的时间与峰值内存表现。

固定：

$$
N=2\times10^5,\qquad d=2,\qquad t=10^5,\qquad \texttt{shape\_sigma}=0.
$$

变化：

$$
\alpha\in\{0.1,0.5,1,5,10,50,100,500,1000\}.
$$

在目标期望密度为 $\alpha_{\mathrm{target}}$ 时，目标期望连接规模满足：

$$
\mathbb E[W]\approx\alpha_{\mathrm{target}}N.
$$

因此当 $\alpha_{\mathrm{target}}=1000$ 时：

$$
\mathbb E[W]\approx1000\times2\times10^5=2\times10^8.
$$

作图时同时提供以 $\alpha_{\mathrm{target}}$ 和 $\alpha_{\mathrm{realized}}$ 为横轴的视图；性能解释以 $\alpha_{\mathrm{realized}}$ 为最终密度依据。

---

### G4. 几何形状扩展：shape-sweep

目的：评估盒形状与长宽比变化下的算法表现。

本实验固定 volume distribution 为 fixed，只改变 shape distribution。

固定：

$$
N=2\times10^5,\qquad d=2,\qquad t=10^5,\qquad \alpha=10.
$$

变化：

$$
\texttt{shape\_sigma}\in\{0,0.2,0.4,0.6,0.8,1.0\}.
$$

| `shape_sigma` | 几何含义 |
|---:|---|
| $0$ | 归一化坐标中的正方形 / 超立方体。 |
| $0.2$ | 极轻微长宽比偏斜。 |
| $0.4$ | 轻微长宽比偏斜。 |
| $0.6$ | 中等长宽比偏斜。 |
| $0.8$ | 较强长宽比偏斜。 |
| $1.0$ | 很强长宽比偏斜。 |

对每个 `shape_sigma` 重新求 coverage，使目标 $\alpha$ 保持为 10。结果除 `shape_sigma` 外，同时报告实际长宽比分位数和饱和比例。

---

### G5. 维度扩展：$d$-sweep

目的：评估维度增加时的时间、峰值内存与可完成边界。

固定：

$$
N=2\times10^5,\qquad t=10^5,\qquad \alpha=10,\qquad \texttt{shape\_sigma}=0.
$$

变化：

$$
d\in\{2,3,4,5\}.
$$

对每个 $d$ 均尝试运行：

$$
\mathcal A_d=\{\mathrm{LiftedRT},\mathrm{SweepRT},\mathrm{AS},\mathrm{AC}\}.
$$

结果采用两个 panel：

1. **共同完成 panel**：显示四个算法均为 `OK` 的维度点，用于直接比较运行时间与峰值内存；
2. **完整压力 panel**：显示 $d\in\{2,3,4,5\}$ 的全部状态，包括 `OOM`、`MEMORY-CAP-EXCEEDED` 与 `TO`，用于展示维度扩展边界。

两个 panel 来自同一组运行，不改变 workload，也不以缩小输入规模隐藏高维失败。

## 8. 真实数据实验

本节在固定的百万级真实 workload 上比较 `AC`、`AS`、`SweepRT` 与 `LiftedRT` 的 `OneShotTime(t)` 和峰值内存。真实数据不设目标输出密度，连接规模由数据几何与 level 参数决定，并以精确计数记录：

$$
W=|J|,
\qquad
\alpha_{\mathrm{realized}}=\frac{W}{|R|+|S|}.
$$

所有算法在同一份冻结的 $R,S$ 上返回：

$$
X_1,\ldots,X_t\overset{i.i.d.}{\sim}\operatorname{Unif}(J).
$$

每个真实实验点的主指标为：

$$
\operatorname{OneShotTime}(t),
\qquad
\operatorname{PeakMemoryTotal},
\qquad
\operatorname{PeakMemoryIncremental}.
$$

`PrepareTime`、`QueryTime(t)`、`CountOnlyTime`、`PreparedQueryTime(t)`、`PeakMemoryAux` 和 `MemoryAfterPrepare` 作为解释性指标。

### 8.1 通用协议

#### 8.1.1 真实数据获取、处理与计时范围

仓库不得假定 CMAB、GeoLife 或 COCO 已经存在于工作目录中。真实数据流程必须能够从一个空的 `data-root` 开始，但仓库只导入已经发布在 Hugging Face 上的成品 Parquet/JSON，不从原始 GIS、`.plt`、COCO 图像或模型权重重新构建三个数据集，也不执行投影软件或 Detectron2 推理。

统一目录布局为：

```text
data/
  sources/
    huggingface/<repo-slug>/<revision>/
  workloads/
    cmab_1m/
    geolife_3d_1m/
    geolife_4d_1m/
    coco_1m/
  manifests/
```

仓库提供且只提供一个真实数据准备入口：

```text
scripts/data/prepare_real_data.sh
```

统一命令为：

```bash
./scripts/data/prepare_real_data.sh --all --data-root ./data
```

`--check-only` 只检查依赖、冻结配置与空间预算；`--verify-only` 只验收现有 workload；`--download-only` 只取得所需的静态成品资产。`run_all_lite.sh` 在实验开始时先执行验收，发现真实 workload 缺失或无效时自动调用上述统一入口。

该命令按 CMAB、GeoLife、COCO 的顺序执行以下步骤：

1. 读取 `data_sources.lock.json`，验证 lock schema、Hub repo id、40-hex immutable revision、资产路径、字节数和 SHA-256。
2. 在联网前检查 `numpy`、`pyarrow`、`requests` 与磁盘空间；不要求 GIS、PROJ、Detectron2、PyTorch、CUDA、图像解码器或模型权重。
3. 对 CMAB 和 GeoLife 的锁定 Parquet/JSON 使用可恢复的 `*.partial` 下载，并在原子重命名前验证大小与 SHA-256。COCO 的小型索引和 build manifest 同样完整下载；大规模 rectangle shards 按选中图像所在的 Parquet row group 进行 HTTP byte-range 读取，不下载整个 1.23B-row 数据集。
4. 只对成品表执行实验子集导入：
   - CMAB 对齐四个广东 level 的 `building_uid`，直接读取已发布的 base/expanded AABB 与 function code，再作固定的 function-class/10 km tile 分层哈希划分。
   - GeoLife 从已发布 `dims=4/level=1` 的五个 shards 按列名恢复 altitude-valid 点中心，以 `(traj_id,point_idx)` 为共同稳定键，作 user-disjoint 划分和每侧 500000 点选择，再由同一中心和 Setup 半径派生 3D/4D 三个 levels。
   - COCO 从已发布 image index 选择固定 100 张图像，只读取这些图像的 `type=1, rank=1..10000` proposal rows，并在每图内以发布字段稳定地划分为 $5000/5000$。
5. 写出四个算法共同读取的规范二进制 workload。每个 collection 包含 $R$、$S$、对象 id、端点类型、维度、逐 workload manifest、collection `manifest.json` 和 `checksums.sha256`。
6. 重新读取最终二进制文件，检查 $N,|R|,|S|,d$、全部 $L<U$、对象 id 唯一性、跨 level/维度对象一致性、源 revision、payload SHA-256 和文件 SHA-256。任一检查失败时返回 `DATASET-CONSTRUCTION-FAILED`。

冻结数据源如下：

| 数据集 | Hub 成品数据集与 revision | 仓库读取内容 |
|---|---|---|
| CMAB | `DannHiroaki/CMAB-Spatial-Join-0.08B@41e3c90fa42fc8eede910404fe3db29ad3897b81` | `file_manifest.parquet` 与四个 `level_*/province=guangdong` Parquet。 |
| GeoLife | `DannHiroaki/Geolife-Spatial-Join-0.15B@a9b8439beb16de106f6ff3f54c73c6b6964d77af` | `manifest.json` 与五个 `dims=4/level=1` Parquet shards。 |
| COCO | `DannHiroaki/COCO-Spatial-Join-1.23B@2e5f2a1ba741ba1148f0b2f42209a9da4635a6cb` | `data/images.parquet`、`meta/build_manifest.json`，以及选中 image 所在 rectangle row groups。 |

最终 workload 的规范位置为：

```text
data/workloads/cmab_1m/
data/workloads/geolife_3d_1m/
data/workloads/geolife_4d_1m/
data/workloads/coco_1m/
```

每次 benchmark 启动前只读取最终 workload 和 manifest，不在算法进程中访问网络、读取 Parquet 或执行数据选择。若最终文件缺失、版本不匹配或 checksum 校验失败，benchmark 拒绝启动，并提示先运行 `prepare_real_data.sh`。

真实数据性能计时从规范二进制 workload 已读入内存、全部输入页已预触且统一输出缓冲区已分配之后开始。Hub 下载、Parquet 读取、子集划分、盒派生、格式转换、correctness validation 与输出校验全部位于计时区间之外。

同一 workload 的四个算法读取完全相同的只读盒数组和对象 id。每个 dataset、level、subset、$t$、algorithm 和 `measurement_mode` 形成一条原始运行记录；`process_repeat_id` 字段固定为 $0$。

#### 8.1.2 区间语义与端点类型

所有算法统一使用半开严格重叠语义：

$$
\max(L_j(r),L_j(s))<\min(U_j(r),U_j(s)).
$$

| Dataset | 冻结端点 | 半开语义 |
|---|---|---|
| CMAB | 固定投影平面中的 binary64 AABB | 预处理生成的连续 AABB 直接解释为 $[L,U)$；边界相接不计相交。 |
| GeoLife | centimeter 与 epoch-millisecond 的 int64 closed boxes | 每维执行 $[L,U]\mapsto[L,U+1)$，得到等价的 int64 半开盒。 |
| COCO | 原图坐标中的 binary64 proposal boxes | 研究 workload 统一把连续 XYXY 盒解释为 $[x_1,x_2)\times[y_1,y_2)$；不依赖外部框架的隐含区间约定。 |

所有 workload 在发布前检查每一维 $L<U$。重复坐标保留不同对象 id。

#### 8.1.3 精确连接规模与 level 解释

真实数据的 $W$ 和 $\alpha_{\mathrm{realized}}$ 必须由四算法一致的精确计数得到，不使用估计值。

CMAB 与 GeoLife 的 level 同时改变盒尺度、输出密度、度数分布和局部性，因此称为 application-level sweep。结果同时绘制：

$$
\operatorname{OneShotTime}\ \text{vs. level}
$$

以及：

$$
\operatorname{OneShotTime}\ \text{vs. }\alpha_{\mathrm{realized}}.
$$

#### 8.1.4 单次运行与随机种子

真实 workload 不使用 `data_seed`。子集、对象 id 与坐标均由 manifest 决定。所有真实数据主性能运行固定：

$$
\texttt{sample\_seed\_id}=0,
\qquad
\texttt{process\_repeat\_id}=0.
$$

同一 workload 的 time run 与 memory run 使用相同的采样主种子；`process_repeat_id` 只作为固定为 $0$ 的结果字段保留。每个 dataset、level、subset、$t$ 和 algorithm 只执行一次 time run 与一次 memory run，二者的 $R,S,W$ 与 $\alpha_{\mathrm{realized}}$ 必须完全一致。

#### 8.1.5 真实数据实验族

**R1. Level sweep**

固定：

$$
t=10^5.
$$

改变数据集自身的 level 参数。R1 用于：

- CMAB-1M；
- GeoLife-3D-1M；
- GeoLife-4D-1M。

COCO-1M 不执行人工 level sweep。

**R2. $t$-sweep**

固定同一份 workload，改变：

$$
t\in\{10^3,10^4,10^5,10^6\}.
$$

具有 level 的数据集固定使用 $\ell=2$。同一 dataset/subset 的全部 $t$ 运行复用相同的 $R,S,W$ 与采样主种子。R2 的主图报告 `OneShotTime(t)`；对 `AC`、`SweepRT` 和 `LiftedRT` 同时报告 `PreparedQueryTime(t)`。

### 8.2 CMAB-1M

#### 8.2.1 数据集、坐标系与几何对象

CMAB workload 使用广东省可用建筑物全集，共：

$$
943285
$$

个 buildings。实验名保留为 `CMAB-1M`，表示百万级 workload；所有图表标题同时标明实际总输入规模 $N=943285$，不把该名称解释为恰好 $10^6$ 个对象。

仓库直接读取成品数据集中已经投影和构造好的四个广东省 level，不读取原始建筑几何，也不在本仓库执行 CRS 转换。发布坐标是以 meter 为单位的 Albers Equal Area 平面 binary64；仓库将其冻结标识为 `CMAB-HF-Albers-Equal-Area-m`。四个 level 必须各有 943285 行，并按 `building_uid` 一一对齐；base AABB、function code 与建筑身份必须跨 level 一致。

manifest 保存 Hub repo、immutable revision、每个 Parquet 的 SHA-256、CRS id、导入配置 SHA-256 与 importer SHA-256。上游数据集如何从建筑几何投影、清洗和扩张属于发布数据集的 provenance，不由本仓库重复执行。

每个建筑物在该投影平面中构造 binary64 base AABB：

$$
[x_{\min},x_{\max})\times[y_{\min},y_{\max}).
$$

base AABB、expanded influence AABB、空间 tile、面积与距离统计全部使用同一个 CRS 和 meter 单位。

每个 building 具有一个 function class。设其 base AABB 如上，在 level $\ell$ 下的 expansion distance 为 $d_\ell$ meters，则 expanded influence AABB 为：

$$
[x_{\min}-d_\ell,x_{\max}+d_\ell)
\times
[y_{\min}-d_\ell,y_{\max}+d_\ell).
$$

不同 function class 在各 level 下的 expansion distance 如下，单位为 meters。

| Function class | Level 1 | Level 2 | Level 3 | Level 4 |
|---|---:|---:|---:|---:|
| Residential | 500 | 800 | 1000 | 1500 |
| Commercial | 800 | 1000 | 1500 | 2500 |
| Public service | 1000 | 2000 | 3000 | 5000 |
| Office | 1000 | 2000 | 3000 | 5000 |
| Industry | 2000 | 5000 | 8000 | 10000 |

成品 Parquet 的取得、划分与规范二进制写出均在性能计时之外完成。最终 binary64 workload 文件一经生成即冻结，并由 SHA-256 标识。

#### 8.2.2 广东省子集的确定性分层哈希划分

CMAB 不使用运行时随机二分。广东省全部 buildings 使用确定性分层哈希划分，使 $R$ 与 $S$ 在 function class 和空间位置上保持接近平衡，同时避免依赖原始文件顺序或 building identifier 顺序。

对每个 building $b$，设其 base AABB 中心为：

$$
c_x(b)=\frac{x_{\min}(b)+x_{\max}(b)}{2},
\qquad
c_y(b)=\frac{y_{\min}(b)+y_{\max}(b)}{2}.
$$

使用投影坐标中的 10 km 空间网格定义 tile：

$$
\mathrm{tile}_x(b)=\left\lfloor \frac{c_x(b)}{10000}\right\rfloor,
\qquad
\mathrm{tile}_y(b)=\left\lfloor \frac{c_y(b)}{10000}\right\rfloor.
$$

building 的分层标签定义为：

$$
g(b)=
(\mathrm{function\_class}(b),
\mathrm{tile}_x(b),
\mathrm{tile}_y(b)).
$$

令 $\mathcal G$ 为所有非空分层标签集合。对每个 stratum $G\in\mathcal G$，记其对象数为 $n_G$。目标总规模为：

$$
|B_R|=471643,
\qquad
|B_S|=471642.
$$

每个 stratum 的 $B_R$ 配额由 largest-remainder 规则给出。先令：

$$
q_G^{(0)}=\left\lfloor \frac{n_G}{2}\right\rfloor.
$$

设：

$$
M=471643-\sum_{G\in\mathcal G}q_G^{(0)}.
$$

在所有 $n_G$ 为奇数的 strata 中，按稳定键

$$
H_{\mathrm{stratum}}(G)
=
\operatorname{StableHash}(\texttt{cmab-stratum},G)
$$

升序选择前 $M$ 个 strata，并令这些 strata 的 $B_R$ 配额增加 $1$。最终配额记为 $q_G$。于是：

$$
\sum_{G\in\mathcal G}q_G=471643.
$$

在每个 stratum 内，对 building 使用稳定对象键：

$$
H_{\mathrm{building}}(b)
=
\operatorname{StableHash}(\texttt{cmab-building-split},\mathrm{building\_uid}(b)).
$$

若原始数据没有稳定的 `building_uid`，预处理阶段用建筑物几何、function class 和原始记录稳定字段生成 deterministic id，并把该 id 写入数据清单。每个 stratum 内按 $H_{\mathrm{building}}(b)$ 升序排序，前 $q_G$ 个对象进入 $B_R$，其余对象进入 $B_S$。

由此得到：

$$
|B_R|=471643,
\qquad
|B_S|=471642.
$$

该划分完全确定，不依赖 random seed、文件读取顺序或原始 id 的数值顺序。每个 stratum 内 $R/S$ 的数量差至多为 $1$，整体上 function class 与 10 km 空间 tile 的边际分布保持接近平衡。

对于每个 level $\ell$，定义：

$$
R_\ell=
\{
\text{level-}\ell\text{ expanded AABB of }b:
b\in B_R
\},
$$

$$
S=
\{
\text{base AABB of }b:
b\in B_S
\}.
$$

因此 CMAB 的连接实例为：

$$
J_\ell
=
\{(r,s)\in R_\ell\times S:r\cap s\neq\emptyset\}.
$$

$S$ 在所有 levels 中保持不变，$R_\ell$ 只改变 expansion distance；因此 level sweep 固定输入 building 集合，只改变 level-induced box expansion。

CMAB 数据清单额外记录以下诊断字段：

| 字段 | 含义 |
|---|---|
| `source` | Hub repo、immutable revision、锁定资产路径、字节数与 SHA-256。 |
| `crs_id` | `CMAB-HF-Albers-Equal-Area-m`。 |
| `split_method` | `cmab_hf_stratified_hash_tile_10km_v1`。 |
| `function_class_counts_R/S` | 每个 function class 在 $B_R$ 和 $B_S$ 中的对象数。 |
| `tile_count_balance` | 每个 10 km tile 内 $B_R/B_S$ 数量差的摘要统计。 |
| `aabb_area_summary_R/S` | 两侧 base AABB 面积分布摘要。 |
| `boundary_touching_diagnostic` | half-open 语义下被排除的边界相接诊断量。 |

#### 8.2.3 CMAB-R1：level sweep

CMAB-R1 固定 sample size：

$$
t=10^5.
$$

level 取值为：

$$
\ell\in\{1,2,3,4\}.
$$

每个 level 下运行所有算法，并记录精确连接规模：

$$
W_\ell=|J_\ell|,
$$

以及真实输出密度：

$$
\alpha_{\mathrm{realized},\ell}
=
\frac{W_\ell}{|R_\ell|+|S|}.
$$

CMAB-R1 的核心记录表应至少包含以下字段：

| dataset | level | $|R_\ell|$ | $|S|$ | $W_\ell=|J_\ell|$ | $\alpha_{\mathrm{realized},\ell}$ | $t$ |
|---|---:|---:|---:|---:|---:|---:|
| CMAB-1M | 1 | 471643 | 471642 | exact | exact | $10^5$ |
| CMAB-1M | 2 | 471643 | 471642 | exact | exact | $10^5$ |
| CMAB-1M | 3 | 471643 | 471642 | exact | exact | $10^5$ |
| CMAB-1M | 4 | 471643 | 471642 | exact | exact | $10^5$ |

$W_\ell$ 和 $\alpha_{\mathrm{realized},\ell}$ 必须来自精确计数，而不是估计值。

---

#### 8.2.4 CMAB-R2：$t$-sweep

CMAB-R2 固定中等 density level：

$$
\ell=2.
$$

在该固定 workload 上改变 sample size：

$$
t\in\{10^3,10^4,10^5,10^6\}.
$$

CMAB-R2 中的连接实例固定为：

$$
J_2=\{(r,s)\in R_2\times S:r\cap s\neq\emptyset\}.
$$

对应设置为：

$$
W_2=|J_2|,
\qquad
\alpha_{\mathrm{realized},2}
=
\frac{W_2}{|R_2|+|S|}
$$

在所有 $t$ 取值下保持不变。该实验用于评估 `OneShotTime` 与 `PreparedQueryTime` 随 $t$ 增长的变化。

---

### 8.3 GeoLife-1M

#### 8.3.1 共同点集、坐标与时间语义

仓库不读取 GeoLife GPS Trajectories 1.3 压缩包或 `.plt` 文件。共同候选池直接取自固定 Hub revision 中的 `dims=4/level=1` 五个 Parquet shards，共 24764050 条 altitude-valid published rectangles；这样 3D 与 4D 一开始就来自同一批点。

导入器按列名而非 Parquet 物理列顺序读取 `x_min/x_max`、`y_min/y_max`、`z_min/z_max`、`t_min/t_max`，并依据 level-1 的冻结半宽恢复 int64 点中心。发布表中的水平坐标采用 EPSG:3857 并量化为 centimeters，垂直坐标为 centimeters，时间为 UTC Unix epoch milliseconds；仓库将该坐标标识冻结为：

```text
EPSG:3857-centimeter-plus-epoch-millisecond
```

`(traj_id,point_idx)` 是跨维度和 level 的共同稳定点键。先按 `user_id` 的稳定哈希顺序交替形成 user-disjoint 的 $R/S$ 用户池，再在每侧按 UTC month 与 1 km XY tile 作 largest-remainder quota，并以 keyed SplitMix64 点优先级选择恰好 500000 个点。选中键和 dense object id 在 3D/4D 及全部 levels 中保持不变。

manifest 固定 Hub repo/revision、五个输入 shards 的 SHA-256、候选与选择计数、用户划分、选中点 sidecar checksum、CRS id、导入配置 SHA-256 与 importer SHA-256。本仓库不重复执行上游经纬度投影、海拔单位转换或民用时间解析。


`GeoLife-3D-1M` 使用：

$$
(x,y,t),
$$

`GeoLife-4D-1M` 在同一对象上增加 altitude：

$$
(x,y,z,t).
$$

因此 3D 与 4D 的比较不会混入用户、轨迹、时间或水平空间子集变化。

#### 8.3.2 GeoLife levels

GeoLife level 由水平空间阈值 $\Delta d$ 和时间阈值 $\Delta t$ 决定。三个 levels 为：

| Level | $\Delta d$ | 水平半宽 $r_{xy}$ | $\Delta t$ | 时间半宽 $r_t$ |
|---:|---:|---:|---:|---:|
| 1 | 20 m | 1000 cm | 60 s | 30000 ms |
| 2 | 50 m | 2500 cm | 300 s | 150000 ms |
| 3 | 200 m | 10000 cm | 1200 s | 600000 ms |

3D workload 中，point $p=(x,y,t)$ 在 level $\ell$ 下产生 closed integer box：

$$
[x-r_{xy},x+r_{xy}]
\times
[y-r_{xy},y+r_{xy}]
\times
[t-r_t,t+r_t].
$$

4D workload 中，垂直半宽明确设为：

$$
r_z=r_{xy}.
$$

这是 workload 的固定设计选择，表示水平与垂直维采用相同的 meter 尺度，而不是 GeoLife 的原生任务定义。4D closed integer box 为：

$$
[x-r_{xy},x+r_{xy}]
\times
[y-r_{xy},y+r_{xy}]
\times
[z-r_z,z+r_z]
\times
[t-r_t,t+r_t].
$$

所有 closed integer intervals 在预处理阶段逐维转换为：

$$
[L,U]\mapsto[L,U+1).
$$

该转换在 int64 范围检查通过后执行，并精确保持离散闭区间中的整数点集合。

#### 8.3.3 用户划分与共同的分层 500k 选择

用户划分与点选择只在第 8.3.1 节的共同 altitude-valid 候选池上执行一次。同一用户的 trajectory points 不得同时出现在两侧。

令 $u$ 为 user id，定义：

$$
H_{\mathrm{user}}(u)=
\operatorname{StableHash}(\texttt{geolife-user-split},u).
$$

将 candidate users 按 $H_{\mathrm{user}}(u)$ 升序排序，并交替分配：

$$
U_R=\{u:\operatorname{rank}_H(u)\bmod2=0\},
$$

$$
U_S=\{u:\operatorname{rank}_H(u)\bmod2=1\}.
$$

point 的稳定对象键为：

$$
\operatorname{point\_key}(p)
=(\operatorname{user\_id},\operatorname{trajectory\_id},\operatorname{point\_index}).
$$

在 $U_R$ 和 $U_S$ 两侧分别执行 temporal-spatial stratified deterministic selection。`month(p)` 使用 `Asia/Shanghai` 日历月；1 km 空间 tile 由 centimeter 坐标定义：

$$
\operatorname{tile}_x(p)=\left\lfloor\frac{x(p)}{100000}\right\rfloor,
\qquad
\operatorname{tile}_y(p)=\left\lfloor\frac{y(p)}{100000}\right\rfloor.
$$

分层标签为：

$$
g(p)=(\operatorname{month}(p),\operatorname{tile}_x(p),\operatorname{tile}_y(p)).
$$

对每侧 $C\in\{U_R,U_S\}$，设非空 strata 为 $\mathcal G_C$，stratum $G$ 的候选数为 $n_G$，总候选数为 $n_C$。构造条件为：

$$
n_C\ge500000.
$$

若任一侧不满足该条件，返回 `DATASET-CONSTRUCTION-FAILED`，不得通过更换区域、允许用户跨侧或使用重复点补足。

每侧目标选择数为：

$$
K=500000.
$$

先令：

$$
q_G^{(0)}=
\left\lfloor K\frac{n_G}{n_C}\right\rfloor.
$$

设：

$$
M_C=K-\sum_{G\in\mathcal G_C}q_G^{(0)}.
$$

按小数余量降序选择前 $M_C$ 个 strata，并以：

$$
H_{\mathrm{stratum}}(G)=
\operatorname{StableHash}(\texttt{geolife-stratum},C,G)
$$

打破并列；对应配额增加 $1$。在每个 stratum 内按：

$$
H_{\mathrm{point}}(p)=
\operatorname{StableHash}(\texttt{geolife-point-select},\operatorname{point\_key}(p))
$$

升序选择前 $q_G$ 个 points。最终得到共同点集：

$$
|P_R|=500000,
\qquad
|P_S|=500000.
$$

同一份 $P_R,P_S$ 同时用于 3D 与 4D，并在所有 levels 中保持不变。

GeoLife manifest 至少记录：

| 字段 | 含义 |
|---|---|
| `source` | Hub repo/revision 与五个输入 shards 的 SHA-256。 |
| `crs_id` | `EPSG:3857-centimeter-plus-epoch-millisecond`。 |
| `candidate_relation` | 已发布的 `dims=4/level=1` altitude-valid relation。 |
| `candidate_points_R/S` | 共同候选池划分后的两侧点数。 |
| `selected_point_manifest_R/S` | 两侧 500k point keys 及 SHA-256。 |
| `selected_users_R/S` | 两侧覆盖的用户数。 |
| `selected_trajectories_R/S` | 两侧覆盖的轨迹数。 |
| `time_span_R/S` | 两侧最小与最大 UTC epoch milliseconds。 |
| `month_histogram_R/S` | UTC 日历月分布。 |
| `spatial_tile_summary_R/S` | 1 km tile 分布摘要。 |
| `altitude_summary_R/S` | 已发布 altitude-valid 点的高度摘要。 |

#### 8.3.4 GeoLife-3D 与 GeoLife-4D workloads

GeoLife 部分包含四组实验：

1. `GeoLife-3D-R1`：3D level sweep；
2. `GeoLife-3D-R2`：3D $t$-sweep；
3. `GeoLife-4D-R1`：4D level sweep；
4. `GeoLife-4D-R2`：4D $t$-sweep。

四组实验均使用第 8.3.3 节冻结的同一批一百万个 point ids。3D 与 4D 的差别仅为是否加入 altitude 维及其 level box，因此可以把两者的差异解释为该固定 workload 上的维度效应。

#### 8.3.5 GeoLife-R1：level sweep

GeoLife-R1 对 `GeoLife-3D-1M` 和 `GeoLife-4D-1M` 分别执行。

固定 sample size：

$$
t=10^5.
$$

level 取值为：

$$
\ell\in\{1,2,3\}.
$$

对于 dimension $d\in\{3,4\}$ 和 level $\ell$，定义：

$$
J_{d,\ell}
=
\{(r,s)\in R_{d,\ell}\times S_{d,\ell}:r\cap s\neq\emptyset\}.
$$

记录精确连接规模：

$$
W_{d,\ell}
=
|J_{d,\ell}|,
$$

以及真实输出密度：

$$
\alpha_{\mathrm{realized},d,\ell}
=
\frac{W_{d,\ell}}
{|R_{d,\ell}|+|S_{d,\ell}|}.
$$

由于 GeoLife-1M 中：

$$
|R_{d,\ell}|=500000,
\qquad
|S_{d,\ell}|=500000,
$$

所以：

$$
\alpha_{\mathrm{realized},d,\ell}
=
\frac{W_{d,\ell}}{10^6}.
$$

GeoLife-R1 的核心记录表应至少包含：

| dataset | dimension | level | $|R|$ | $|S|$ | $W=|J|$ | $\alpha_{\mathrm{realized}}$ | $t$ |
|---|---:|---:|---:|---:|---:|---:|---:|
| GeoLife-1M | 3D | 1 | 500000 | 500000 | exact | exact | $10^5$ |
| GeoLife-1M | 3D | 2 | 500000 | 500000 | exact | exact | $10^5$ |
| GeoLife-1M | 3D | 3 | 500000 | 500000 | exact | exact | $10^5$ |
| GeoLife-1M | 4D | 1 | 500000 | 500000 | exact | exact | $10^5$ |
| GeoLife-1M | 4D | 2 | 500000 | 500000 | exact | exact | $10^5$ |
| GeoLife-1M | 4D | 3 | 500000 | 500000 | exact | exact | $10^5$ |

$W$ 和 $\alpha_{\mathrm{realized}}$ 必须对每个 dimension-level 组合单独记录。

---

#### 8.3.6 GeoLife-R2：$t$-sweep

GeoLife-R2 对 `GeoLife-3D-1M` 和 `GeoLife-4D-1M` 分别执行。

固定中等 density level：

$$
\ell=2.
$$

改变 sample size：

$$
t\in\{10^3,10^4,10^5,10^6\}.
$$

对于每个 dimension $d\in\{3,4\}$，GeoLife-R2 的连接实例固定为：

$$
J_{d,2}
=
\{(r,s)\in R_{d,2}\times S_{d,2}:r\cap s\neq\emptyset\}.
$$

对应设置为：

$$
W_{d,2}
=
|J_{d,2}|,
$$

以及：

$$
\alpha_{\mathrm{realized},d,2}
=
\frac{W_{d,2}}{10^6}
$$

在所有 $t$ 取值下保持不变。该实验用于评估 3D 和 4D GeoLife workloads 上 `OneShotTime` 与 `PreparedQueryTime` 随 $t$ 增长的扩展性。

---

### 8.4 COCO-1M

#### 8.4.1 冻结的已发布 proposal relation

COCO-1M 不在本仓库下载 COCO 图像、annotations、Detectron2 代码或模型 checkpoint，也不执行模型推理。仓库直接导入固定 Hub revision 中已经发布的 image index、upstream build manifest 和 rectangle Parquet shards。模型与 RPN 的上游构建信息只作为发布 provenance 继承到 workload manifest。

`data/images.parquet` 给出 123287 张 eligible images 的 `split`、`coco_image_id`、宽高和发布 `z_idx`。`meta/build_manifest.json` 给出 rectangle shards；导入器从选中图像的 `z_idx` 计算所需 row groups，以 pinned revision 和 Hub linked SHA-256 验证远端 shard 身份，再通过 HTTP byte ranges 只读取这些 row groups。

对每张 selected image，只接受发布 relation 中满足以下条件的行：

1. `type=1`，即发布的 RPN proposal，而不是 ground-truth box；
2. `rank` 恰为 $1,2,\ldots,10000$ 且每个 rank 唯一；
3. `coco_image_id`、`z_min/z_max` 与 image index 一致；
4. XYXY 有限且满足 $x_1<x_2,y_1<y_2$。

发布 proposal 坐标的物理类型为 float32。导入时只作精确的 float32-to-float64 promotion，不重新计算、裁剪、NMS 或排序；工作负载统一解释为：

$$
[x_1,x_2)\times[y_1,y_2).
$$

重复几何盒保留不同 proposal id。成品表可用的稳定 proposal 身份为：

$$
(\mathrm{canonical\_split},\mathrm{coco\_image\_id},
\mathrm{rank},\mathrm{rect\_id}).
$$

发布 schema 不包含 upstream `anchor_key`，所以本仓库不得声称复现旧的 anchor-key split。proposal split 使用以上四元组并具有新的显式版本标识。

为避免不同 images 之间相交，每个 2D proposal 提升为 3D box。设 selected image 的确定性索引为 $z_{\mathrm{idx}}$，则：

$$
[x_1,x_2)
\times[y_1,y_2)
\times[z_{\mathrm{idx}},z_{\mathrm{idx}}+1).
$$

最终 manifest 记录 Hub repo/revision、image index 与 build manifest SHA-256、实际访问 shard 的 linked SHA-256/size、每张图像 proposal rows SHA-256、upstream 模型 provenance、importer SHA-256 与最终 workload SHA-256。

#### 8.4.2 确定性 100-image 子集

image pool 为：

$$
\mathcal I=\texttt{train2017}\cup\texttt{val2017}.
$$

候选池就是发布 `images.parquet` 中已经拥有完整 10000 个 RPN proposals 的 123287 张图像。对每张候选 image $i$，定义：

$$
H_{\mathrm{image}}(i)=
\operatorname{StableHash}(
\texttt{coco-image-subset},
\operatorname{split}(i),
\operatorname{image\_id}(i)).
$$

候选 images 按：

$$
(H_{\mathrm{image}}(i),\operatorname{split}(i),\operatorname{image\_id}(i))
$$

升序排序，固定选择前 100 张图像，得到唯一 workload：

```text
subset_id = coco_hash_subset_0
```

按同一顺序为 selected images 分配：

$$
z_{\mathrm{idx}}=0,1,\ldots,99.
$$

图像子集不随 `process_repeat_id`、`sample_seed_id` 或 $t$ 改变。manifest 保存 100 个 `(split,image_id)`、发布 `z_idx`、宽高、排序位置、proposal rows SHA-256 和完整子集 SHA-256。

#### 8.4.3 确定性 proposal split

对每张 selected image，使用第 8.4.1 节排序后的前 10000 个 proposal candidates：

$$
\mathcal P_i=\{p_{i,1},\ldots,p_{i,10000}\}.
$$

对 proposal 定义：

$$
H_{\mathrm{proposal}}(p)=
\operatorname{StableHash}(
\texttt{coco-hf-proposal-split-v1},
\operatorname{canonical\_split},
\operatorname{coco\_image\_id},
\operatorname{rank},
\operatorname{rect\_id}).
$$

每张 image 内按：

$$
(H_{\mathrm{proposal}}(p),\operatorname{rank},\operatorname{rect\_id})
$$

升序排序，前 5000 个进入 $R$，后 5000 个进入 $S$。因此：

$$
|R|=500000,
\qquad
|S|=500000,
\qquad
N=10^6.
$$

该分法保证每张 image 在两侧数量完全平衡，并避免按 objectness rank 的前后段系统性分色。COCO-1M 不执行 box dilation。

COCO manifest 至少记录：

| 字段 | 含义 |
|---|---|
| `source` | Hub repo/revision、静态资产与访问 shards 的 linked SHA-256。 |
| `upstream_builder_manifest_sha256` | 已发布 build manifest 的 checksum。 |
| `model_config_id` / `checkpoint_sha256` | 从发布 manifest 继承的 upstream provenance；本仓库不下载模型。 |
| `proposal_stage` | `hf_published_rpn_top10000`。 |
| `image_subset_id` | 固定为 `coco_hash_subset_0`。 |
| `eligible_image_count` | 发布 image index 的 123287 行。 |
| `proposal_split_method` | `coco_hf_published_hash_v1_balanced_5000_5000`。 |
| `proposal_rank_summary_R/S` | 两侧 objectness rank 分布。 |
| `proposal_score_summary_R/S` | 两侧 objectness score 分布。 |
| `proposal_area_summary_R/S` | 两侧 proposal 面积分布。 |
| `image_size_summary` | 100 张图像的原图宽高分布。 |
| `workload_sha256` | 最终规范二进制 workload checksum。 |

#### 8.4.4 COCO 不执行 level sweep

COCO 没有类似 CMAB influence level 或 GeoLife spatiotemporal threshold 的原生 level 参数。为保持真实数据实验的自然性，COCO-1M 不执行人工 dilation-level sweep，也不使用如下人为参数：

$$
\lambda\in\{1.0,1.25,1.5,2.0\}.
$$

COCO-1M 不执行 R1，只执行 R2。

---

#### 8.4.5 COCO-R2：固定 proposal boxes 上的 $t$-sweep

COCO-R2 在唯一的 `coco_hash_subset_0` workload 上执行。固定连接实例：

$$
J_{\mathrm{COCO}}
=
\{(r,s)\in R\times S:r\cap s\neq\emptyset\}.
$$

记录精确连接规模：

$$
W_{\mathrm{COCO}}=|J_{\mathrm{COCO}}|,
$$

以及：

$$
\alpha_{\mathrm{realized},\mathrm{COCO}}
=
\frac{W_{\mathrm{COCO}}}{10^6}.
$$

改变 sample size：

$$
t\in\{10^3,10^4,10^5,10^6\}.
$$

所有 $t$ 取值使用完全相同的 $R$、$S$、$W_{\mathrm{COCO}}$、$\alpha_{\mathrm{realized},\mathrm{COCO}}$、采样主种子起点和 workload SHA-256。该实验用于评估固定 proposal overlap 结构下 `OneShotTime`、`PreparedQueryTime` 与峰值内存随 $t$ 增长的表现。

COCO-R2 的核心记录表为：

| dataset | subset_id | images | dimension | $|R|$ | $|S|$ | $W=|J|$ | $\alpha_{\mathrm{realized}}$ | $t$ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| COCO-1M | `coco_hash_subset_0` | 100 | 3D | 500000 | 500000 | exact | exact | $10^3$ |
| COCO-1M | `coco_hash_subset_0` | 100 | 3D | 500000 | 500000 | exact | exact | $10^4$ |
| COCO-1M | `coco_hash_subset_0` | 100 | 3D | 500000 | 500000 | exact | exact | $10^5$ |
| COCO-1M | `coco_hash_subset_0` | 100 | 3D | 500000 | 500000 | exact | exact | $10^6$ |

### 8.5 真实数据 workloads 汇总

| Workload | Dimension | $|R|$ | $|S|$ | R1 level sweep | R2 $t$-sweep |
|---|---:|---:|---:|---|---|
| CMAB-1M | 2D | 471643 | 471642 | $\ell\in\{1,2,3,4\}$ | fixed $\ell=2$ |
| GeoLife-3D-1M | 3D | 500000 | 500000 | $\ell\in\{1,2,3\}$ | fixed $\ell=2$ |
| GeoLife-4D-1M | 4D | 500000 | 500000 | $\ell\in\{1,2,3\}$ | fixed $\ell=2$ |
| COCO-1M | 3D | 500000 | 500000 | not used | fixed `coco_hash_subset_0` proposal workload |

因此真实数据包含七组实验：

1. CMAB level-sweep；
2. CMAB $t$-sweep；
3. GeoLife-3D level-sweep；
4. GeoLife-3D $t$-sweep；
5. GeoLife-4D level-sweep；
6. GeoLife-4D $t$-sweep；
7. COCO $t$-sweep。

加上 Alacarte 的五组 sweep，整套主实验共十二组。

每个 workload 的主记录包括：

$$
W,
\qquad
\alpha_{\mathrm{realized}},
\qquad
\operatorname{OneShotTime}(t),
$$

$$
\operatorname{PeakMemoryTotal},
\qquad
\operatorname{PeakMemoryIncremental},
\qquad
\operatorname{PeakMemoryAux}.
$$

对 `AC`、`LiftedRT` 和 `SweepRT` 还记录 `PrepareTime`、`QueryTime(t)`、`PreparedQueryTime(t)` 与 `MemoryAfterPrepare`；所有算法记录独立的 `CountOnlyTime`。

## 9. 原始结果字段

每条原始运行记录至少包含：

| 类别 | 字段 |
|---|---|
| experiment | `experiment_id`, `dataset_type`, `dataset_id`, `workload_id`, `subset_id` |
| run identity | `run_group_id`, `measurement_mode`, `data_seed`, `sample_seed_id`, `process_repeat_id`, `algorithm_order_index` |
| environment | `machine_id`, `code_commit`, `build_sha256`, `compiler_id`, `allocator_id`, `prng_id`, `kernel_id`, `cpu_core`, `numa_node`, `monitor_cpu_core`, `memory_cap_bytes`, `memory_measurement_backend`, `memory_poll_interval_ms` |
| data | `N_total`, `n_R`, `n_S`, `d`, `endpoint_type`, `t`, `workload_sha256` |
| synthetic | `alpha_target`, `alpha_expected`, `alpha_realized`, `shape_sigma`, `coverage_status`, `coverage_interval`, `coverage_solver_config_sha256`, `aspect_ratio_quantiles`, `saturation_fraction` |
| real | `real_dataset`, `level`, `split_method`, `crs_id`, `projection_unit`, `stratification_id`, `image_subset_id`, `proposal_pipeline_id` |
| algorithm | `algorithm`, `orientation`, `algorithm_config_sha256` |
| count | `W`, `alpha_realized`, `count_integer_bits` |
| main time | `OneShotTime` |
| stage time | `PrepareTime`, `QueryTime`, `CountOnlyTime`, `PreparedQueryTime` |
| memory baseline | `InputMemory`, `BaselineMemory`, `output_buffer_bytes` |
| main memory | `PeakMemoryTotal`, `PeakMemoryIncremental`, `PeakMemoryAux`, `ProcessMaxRSS`, `PeakRSSPollBytes`, `MemoryAfterPrepare`, `rss_sample_count`, `peak_is_sampled` |
| correctness | `output_length`, `output_sha256`, `membership_checked`, `count_consistency_checked` |
| status | `status`, `failure_stage`, `memory_cap_exceeded`, `termination_signal`, `timeout_seconds`, `error_message` |
| diagnostics | `boundary_touching_diagnostic`, `spatial_temporal_coverage_summary`, `rank_or_score_summary` |

算法特定诊断字段至少包括：

| 算法 | 字段 |
|---|---|
| `AC` | `terminal_instance_count`, `positive_atom_count`, `persistent_terminal_array_items`, `alias_label_count` |
| `AS` | `recursive_count_calls`, `positive_quota_nodes`, `active_route_nodes`, `max_live_workspace_bytes` |
| `SweepRT` | `nonzero_event_blocks`, `selected_event_blocks`, `skeleton_nodes`, `fenwick_items` |
| `LiftedRT` | `positive_degree_left_objects`, `selected_left_objects`, `canonical_block_queries`, `range_tree_items` |

同一个 `run_group_id` 下应有一条 time record 与一条 memory record。任一模式失败时仍保留该记录；聚合阶段不得覆盖原始状态。

## 10. 聚合、作图与结论口径

### 10.1 单次运行与数据实例

对固定 workload、算法、$t$ 和测量模式，只生成一条原始运行记录，不执行统计重复。对于状态为 `OK` 的记录，time run 的 `OneShotTime` 直接作为该配置的时间值，memory run 的 `PeakMemoryTotal`、`PeakMemoryIncremental` 和 `PeakMemoryAux` 直接作为该配置的内存值；失败记录仍按第 5.4 节保留。结果不取 median、minimum、maximum 或 bootstrap interval。

合成数据的每个参数点固定使用 `data_seed=0`，报告该冻结 workload 的原始时间、原始内存、$W$ 和 $\alpha_{\mathrm{realized}}$。CMAB、GeoLife 与 COCO 报告各自固定 workload 的单次结果。图表不为单次运行构造误差条；全部原始 time record、memory record、状态和日志均随结果发布。

### 10.2 配对 speedup

算法 $A$ 相对 baseline $B$ 的 speedup 在相同 workload、数据实例、$t$ 和测量口径上直接计算：

$$
\operatorname{speedup}(A,B)=\frac{T_B}{T_A}.
$$

只在双方 time run 均为 `OK` 时计算该比值。不得用不同 workload、不同 $t$ 或不同测量模式的数值相除；主扩展图仍显示 OOM、`MEMORY-CAP-EXCEEDED` 与 TO 边界，避免只展示成功配置。

### 10.3 十二组主实验的展示

每个 sweep 至少提供两张主图：

1. `OneShotTime(t)` 或对应横轴下的 one-shot 时间；
2. `PeakMemoryTotal` 与 `PeakMemoryIncremental`。

辅助图包括：

- `PrepareTime` 与 `QueryTime(t)`；
- `PreparedQueryTime(t)`；
- `CountOnlyTime`；
- `PeakMemoryAux` 与 `MemoryAfterPrepare`；
- throughput $t/\operatorname{QueryTime}(t)$，仅对定义了 `QueryTime` 的算法报告。

$t$-sweep 的每个点从原始输入执行 one-shot，因此展示的是单次查询端到端成本；准备后查询图单独说明多查询场景。AS 不具有持久准备态，相关字段保持 `N/A`，不以额外缓存变体替代。

level-sweep 同时以 level 和 $\alpha_{\mathrm{realized}}$ 为横轴。shape-sweep 同时报告实际长宽比分位数。$d$-sweep 同时展示共同完成 panel 与包含失败状态的完整压力 panel。

### 10.4 失败与资源边界

主表记录每个算法最后一个成功的 $N$、$d$、level 和 $t$，以及首次 `OOM`、`MEMORY-CAP-EXCEEDED` 或 `TO` 的配置。失败点不进行数值插值；图中以明确符号和固定的 `memory_cap_bytes`、5 ms RSS 轮询间隔及 900-second timeout 标注。

所有公开结论都以冻结 workload、固定单线程执行环境、明确时间边界和固定 5 ms 间隔的进程 RSS 轮询峰值口径为前提。
