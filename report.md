# 基于差分进化算法的电力系统经济负荷分配优化研究

**实验报告（初稿）**

课程：演化计算及其在能源电力中的应用
算法类型：差分进化算法（Differential Evolution, DE）
研究问题：电力系统经济负荷分配（Economic Load Dispatch, ELD）

---

## 一、实验目的

1. 掌握差分进化算法（DE）的基本流程，理解其变异、交叉、选择三个核心算子的作用机制。
2. 通过 Sphere 与 Rastrigin 两个经典基准函数验证 DE 实现的正确性与全局搜索能力。
3. 选取电力系统中典型的优化问题——经济负荷分配（ELD），用 DE 求解 IEEE 6 机组测试系统，理解约束优化问题的处理方法。
4. 对结果进行可视化与对比分析，初步形成对 DE 算法优缺点和适用场景的工程认识。

## 二、差分进化算法原理

差分进化最早由 Storn 和 Price 在 1997 年提出，是一类基于种群的实数编码全局优化算法。它不依赖目标函数的导数信息，通过个体之间的“差分向量”自适应地引导搜索方向，对低维到中维实数优化问题非常有效。

最常用的变体是 `DE/rand/1/bin`，记当前代种群为 `{x_1, x_2, ..., x_NP}`，对每个个体 `x_i` 执行：

1. **变异**：随机选三个互不相同也不等于 `i` 的索引 `r1, r2, r3`，构造
   `v_i = x_r1 + F · (x_r2 − x_r3)`，其中 `F ∈ (0, 2)` 称为缩放因子。
2. **交叉（二项式）**：以概率 `CR ∈ [0, 1]` 在每一维上从 `v_i` 取分量，至少保留一维来自 `v_i`，得到试验向量 `u_i`。
3. **选择**：若 `f(u_i) ≤ f(x_i)`，则下一代用 `u_i` 替换 `x_i`，否则保留 `x_i`。

差分项 `F·(x_r2 − x_r3)` 在搜索初期方差大，承担探索（exploration）；当种群逐渐收敛时差分项变小，自动转入开发（exploitation），不需要额外的退火机制。

## 三、经典优化问题描述

为了验证算法实现的正确性，本实验采用两个最常用的基准函数：

- **Sphere 函数**（单峰、可分、凸）：
  `f(x) = Σ x_i²`，`x ∈ [−5.12, 5.12]^n`，理论最优 `f* = 0`。
- **Rastrigin 函数**（多峰、不可分，含大量局部最小值）：
  `f(x) = 10n + Σ [x_i² − 10 cos(2π x_i)]`，`x ∈ [−5.12, 5.12]^n`，理论最优 `f* = 0`。

两个函数均设维度 `n = 30`。Sphere 用于检查算法在简单单峰问题上的快速收敛性，Rastrigin 用于检验全局搜索能力。

## 四、ELD 问题背景

经济负荷分配（Economic Load Dispatch, ELD）是电力系统调度的核心问题之一。在已知系统总负荷需求 `PD` 和机组成本特性的前提下，合理分配各台发电机组的出力 `P_i`，使总燃料成本最低，同时满足功率平衡和机组出力上下限。它是一类典型的**含等式约束、不等式约束、非线性目标**的约束优化问题。Noman 和 Iba（2008）的工作明确指出 DE 能够较好地处理此类问题。

## 五、数学模型

记机组台数为 `n`，第 `i` 台机组出力为 `P_i`，其燃料成本采用二次模型：

```
C_i(P_i) = a_i + b_i · P_i + c_i · P_i²
```

总燃料成本最小化模型：

```
min   F(P) = Σ_{i=1..n} ( a_i + b_i P_i + c_i P_i² )
s.t.  Σ_{i=1..n} P_i = PD                     # 功率平衡
      Pmin_i ≤ P_i ≤ Pmax_i,  i = 1, ..., n   # 机组出力上下限
```

本实验使用的数据来自 Al-Roomi 的 *Economic Load Dispatch Test Systems Repository* 中的 IEEE 6-Units Test System No.1：

| 机组 | Pmin (MW) | Pmax (MW) | a | b | c |
| :-: | :-: | :-: | :-: | :-: | :-: |
| 1 | 100 | 500 | 240 | 7.0 | 0.0070 |
| 2 |  50 | 200 | 200 | 10.0 | 0.0095 |
| 3 |  80 | 300 | 220 | 8.5  | 0.0090 |
| 4 |  50 | 150 | 200 | 11.0 | 0.0090 |
| 5 |  50 | 200 | 220 | 10.5 | 0.0080 |
| 6 |  50 | 120 | 190 | 12.0 | 0.0075 |

负荷需求 `PD = 1263 MW`。第一版实现为简化模型，不考虑网损、阀点效应、爬坡约束和禁运区。

## 六、算法设计

### 6.1 公共超参数

| 参数 | Sphere/Rastrigin | ELD |
| :-: | :-: | :-: |
| 种群规模 NP | 50 | 80 |
| 最大代数 G_max | 500 | 800 |
| 缩放因子 F | 0.5 | 0.5 |
| 交叉概率 CR | 0.9 | 0.9 |
| 随机种子 | 42 | 42 |

### 6.2 边界处理

变异得到 `v_i` 后，对越界分量做“反射”修正：
若 `v < lower`，置为 `lower + (lower − v)`；若 `v > upper`，置为 `upper − (v − upper)`，最后再 clip 到 `[lower, upper]`。这种反射式修正比直接 clip 更利于种群多样性。

### 6.3 ELD 约束处理：repair + penalty

`DE/rand/1/bin` 是无约束算法，需要外加约束处理机制：

1. **box 修复**：先把每一维裁剪到 `[Pmin_i, Pmax_i]`，自动满足不等式约束。
2. **等式修复**：计算偏差 `Δ = PD − ΣP`。若 `Δ > 0`，按各机组当前的“上行余量” `(Pmax_i − P_i)` 比例分摊；若 `Δ < 0`，按“下行余量” `(P_i − Pmin_i)` 比例分摊。重复几次直到 `|Δ| < 10⁻⁶`。
3. **penalty**：极少数情况下分摊会因取整误差留下小残差，再叠加 `1e5 · |Δ|` 的惩罚项，几乎不影响最终最优解的成本，但能保证 DE 的“贪心选择”始终偏向可行解。

## 七、程序设计

项目目录结构（详见 `README.md`）：

```
de_power_dispatch/
├── main.py
├── requirements.txt
├── README.md
├── report.md
├── data/eld_6unit.json
├── docs/references.md
├── src/
│   ├── differential_evolution.py
│   ├── benchmark_functions.py
│   ├── economic_load_dispatch.py
│   └── visualization.py
└── results/
    ├── benchmark_results.csv
    ├── eld_results.csv
    ├── sphere_convergence.png
    ├── rastrigin_convergence.png
    ├── eld_convergence.png
    └── eld_dispatch_bar.png
```

模块职责：

- `differential_evolution.py`：从零实现 `DE/rand/1/bin`，提供 `DifferentialEvolution.optimize()` 接口，并支持注入 `repair` 函数。
- `benchmark_functions.py`：定义 Sphere、Rastrigin（以及备用的 Rosenbrock）目标函数及参数。
- `economic_load_dispatch.py`：包含 `ELDSystem` 数据类、`fuel_cost`、`repair_power_balance` 修复函数和 `make_eld_objective` 工厂函数。
- `visualization.py`：基于 matplotlib 绘制收敛曲线和出力柱状图，使用 `Agg` 后端，不弹窗。
- `main.py`：依次跑两个基准函数和 ELD 实验，写入 CSV 与 PNG。

## 八、时空复杂度分析

设维度为 `D`，种群规模为 `NP`，最大代数为 `G`。

- **时间复杂度**：每代要做 `NP` 次变异、交叉、选择，每次 `O(D)`，并伴随一次目标函数评估。整个算法的时间复杂度为 `O(G · NP · (D + T_f))`，其中 `T_f` 是目标函数单次评估的开销。对本实验而言：
  - Sphere/Rastrigin：`G·NP = 500·50 = 25 000` 次评估，`D = 30`。
  - ELD：`G·NP = 800·80 = 64 000` 次评估（修复函数额外引入 `O(D · iter_repair)`，此处 `iter_repair ≤ 50`），`D = 6`。
- **空间复杂度**：主要存储种群矩阵 `O(NP · D)` 和收敛曲线 `O(G)`，整体 `O(NP·D + G)`，在本实验规模下可忽略不计。

## 九、实验结果与可视化说明

实测结果（`seed = 42`）摘录如下：

### 9.1 经典基准函数

| 函数 | 维度 | 最优值 | |最优解|∞ | 评估次数 | 用时 |
| :-: | :-: | :-: | :-: | :-: | :-: |
| Sphere | 30 | 2.59 × 10⁻⁸ | 7.84 × 10⁻⁵ | 25 050 | ≈0.58 s |
| Rastrigin | 30 | 196.42 | 2.17 | 25 050 | ≈0.62 s |

- Sphere 收敛曲线 `results/sphere_convergence.png` 在对数纵轴上近似线性下降，说明 DE 对单峰凸问题具有指数级收敛速度。
- Rastrigin 收敛曲线 `results/rastrigin_convergence.png` 在前 200 代下降较快，后期出现“平台”，是 DE 在 30 维高度多峰问题上的典型表现；可通过加大种群或代数继续改善（属第二版可拓展项）。

### 9.2 ELD 经济负荷分配

| 机组 | P (MW) | Pmin (MW) | Pmax (MW) | C_i (\$/h) |
| :-: | :-: | :-: | :-: | :-: |
| 1 | 446.7073 | 100 | 500 | 4763.78 |
| 2 | 171.2580 |  50 | 200 | 2191.21 |
| 3 | 264.1057 |  80 | 300 | 3092.66 |
| 4 | 125.2168 |  50 | 150 | 1718.50 |
| 5 | 172.1189 |  50 | 200 | 2264.25 |
| 6 |  83.5935 |  50 | 120 | 1245.53 |

总出力 ΣP = **1263.000000 MW**，与负荷需求 PD = 1263 MW 误差为 0，**最小总燃料成本 F\* ≈ 15 275.93 \$/h**，运行约 6.9 s（CPU）。

- ELD 收敛曲线 `results/eld_convergence.png` 在约前 60 代快速降到 1.527 6 × 10⁴ 附近，之后进入精细微调；后期变化在小数点后第 2 位，可视为收敛。
- 出力柱状图 `results/eld_dispatch_bar.png` 直观显示 6 台机组出力均严格落在 `[Pmin, Pmax]` 之间。可以看到：
  - 燃料成本曲线斜率小（b、c 较低）的机组 1 出力最大，符合“边际成本越低承担越多负荷”的经济调度原理。
  - 机组 6 由于 b、c 都较高，被自动压在接近下限。
  - 这与等微增率法（λ-iteration）的解析结果定性一致，进一步说明 DE 求解结果合理。

### 9.3 可视化结果文件清单

- `results/benchmark_results.csv`：两个基准函数的实验结果一览。
- `results/eld_results.csv`：ELD 各机组出力、单机成本、总成本、误差。
- `results/sphere_convergence.png`、`results/rastrigin_convergence.png`、`results/eld_convergence.png`：收敛曲线。
- `results/eld_dispatch_bar.png`：6 机组出力柱状图。

## 十、算法优缺点分析

**优点**

1. 实现简单，超参数少（`NP, F, CR`），新手友好。
2. 不需要梯度信息，对非凸、不连续、非线性目标函数都能应用。
3. 收敛速度在中低维问题上快，且差分项自带步长自适应。
4. 易于扩展约束处理（repair / penalty）和并行评估。

**缺点**

1. 在高维（数百维以上）或高度多峰问题上容易过早收敛，例如本实验 30 维 Rastrigin 仍有较大间隙。
2. 对 `F`、`CR` 的选择敏感，不同问题的最优参数差别较大；课程通用值 `F=0.5, CR=0.9` 是稳健折中，但不是任何问题的最优。
3. 没有理论意义上的全局收敛保证，每次运行结果会随种子有抖动，需要多次实验取平均。
4. 处理强约束问题时仍需要外加约束处理机制，否则会大量产生不可行解。

## 十一、与 GA 和 PSO 的简要对比

| 方面 | DE | GA（遗传算法） | PSO（粒子群） |
| :-: | :-: | :-: | :-: |
| 编码 | 实数 | 通常为二进制/实数 | 实数 |
| 关键算子 | 变异（差分） + 交叉 + 贪心选择 | 选择 + 交叉 + 变异 | 速度更新 + 位置更新 |
| 收敛速度 | 中低维快 | 一般 | 通常更快但易早熟 |
| 多样性 | 差分项自带步长自适应 | 依赖较高的变异率 | 依赖惯性权重和加速因子 |
| 约束处理 | 易接 repair / penalty | 类似 | 类似 |
| 实现复杂度 | 简单 | 中（编码 + 选择策略） | 简单 |

直观感受：在本实验的 ELD 6 机组问题上，DE 在 6.9 秒内得到误差为 0 的解；如果换成传统 GA 通常需要更大的种群和更多代数，PSO 则更容易卡到局部最优需要重启。当然这一对比未做严格的等参数横评，仅作工程经验描述。

## 十二、实验总结

1. 自行实现的 `DE/rand/1/bin` 在 Sphere 上达到 10⁻⁸ 量级，证明算法实现正确。
2. Rastrigin 30 维问题的最优值在 200 左右，反映 DE 在高度多峰问题上的局限，但仍能远离随机解。
3. 在 IEEE 6-Units ELD 测试系统上，使用 repair + penalty 组合的约束处理，最终获得满足等式约束（误差 = 0）、不等式约束（全部在 `[Pmin, Pmax]` 内）的最优解，最小总燃料成本约 **15 275.93 \$/h**。
4. 各机组出力分配和等微增率法定性一致，说明 DE 在工程意义上是可信的全局优化工具。
5. 项目代码结构清晰，可一键复现（`python main.py`），所有结果（CSV / PNG）自动保存，便于撰写报告与答辩展示。

后续可拓展方向：

- 加入网损（B 系数）和阀点效应，使模型更接近实际；
- 引入爬坡约束、禁运区；
- 与 PSO、GA 在统一参数下做横向对比；
- 做 `F`、`CR` 的参数敏感性分析；
- 对每个问题重复运行 30 次，统计均值、最优、最差和标准差。

## 十三、参考文献

1. Storn, R., & Price, K. (1997). Differential Evolution – A Simple and Efficient Heuristic for Global Optimization over Continuous Spaces. *Journal of Global Optimization*, 11, 341–359.
2. Noman, N., & Iba, H. (2008). Differential evolution for economic load dispatch problems. *Electric Power Systems Research*, 78(8), 1322–1331.
3. Wood, A. J., Wollenberg, B. F., & Sheblé, G. B. (2013). *Power Generation, Operation, and Control* (3rd ed.). Wiley.
4. Al-Roomi, A. R. *Economic Load Dispatch Test Systems Repository*. https://al-roomi.org/economic-dispatch/6-units-system/system-i

> 注：本报告中所有数值结果均通过运行 `python main.py`（`seed=42`）即可复现。
