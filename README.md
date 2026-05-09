# 基于差分进化算法的电力系统经济负荷分配优化研究

> Economic Load Dispatch Optimization Based on Differential Evolution Algorithm

## 1. 项目简介

本项目是“演化计算及其在能源电力中的应用”课程实验，对应 PPT 中的**第 4 节**和**第 5 节**：

- **第 4 节**：调通差分进化（Differential Evolution, DE）算法代码，并求解两个经典优化问题——Sphere 函数和 Rastrigin 函数。
- **第 5 节**：在能源电力领域中选定一个优化问题——**电力系统经济负荷分配（Economic Load Dispatch, ELD）**，并使用 DE 进行求解。

本组选定的算法类型为：**差分进化算法（Differential Evolution, DE）**，从零实现 `DE/rand/1/bin` 策略，未直接调用 `scipy.optimize.differential_evolution`。

## 2. 算法简介

差分进化（Storn & Price, 1997）是一类基于种群的启发式实数优化算法，使用“差分向量”指导搜索方向。本项目采用经典的 `DE/rand/1/bin` 策略：

1. **初始化**：在搜索域内均匀采样 `NP` 个个体。
2. **变异**：对每个个体 `x_i`，随机选取 3 个互不相同的个体 `x_r1, x_r2, x_r3`，构造变异向量 `v_i = x_r1 + F * (x_r2 - x_r3)`。
3. **交叉**：对 `v_i` 与 `x_i` 做二项式交叉，至少保留一维来自 `v_i`，其余以概率 `CR` 替换。
4. **选择**：贪心比较，若试验向量适应度优于父代则替换。
5. 重复变异/交叉/选择直到达到最大代数 `G_max`，输出最优个体。

实现位于 `src/differential_evolution.py`，支持自定义 `repair` 修复函数（用于约束优化）。

## 3. 问题建模

### 3.1 经典基准函数

| 函数 | 公式 | 维度 | 范围 | 理论最优 |
| --- | --- | --- | --- | --- |
| Sphere | `f(x) = Σ x_i²` | 30 | [-5.12, 5.12] | 0 |
| Rastrigin | `f(x) = 10n + Σ [x_i² − 10 cos(2π x_i)]` | 30 | [-5.12, 5.12] | 0 |

### 3.2 经济负荷分配（IEEE 6-Units ELD Test System No.1）

目标函数（最小化总燃料成本）：

```
min  F = Σ_{i=1..n} (a_i + b_i P_i + c_i P_i²)
```

约束：

```
Σ P_i = PD          # 功率平衡（等式）
Pmin_i ≤ P_i ≤ Pmax_i   # 出力上下限（不等式）
```

简化处理：暂不考虑网损、阀点效应、爬坡约束、禁运区。约束处理采用 **repair + penalty** 组合：先 clip 到边界，再迭代分摊功率偏差至各机组的可用余量；若残差极小再用线性 penalty。

数据保存在 `data/eld_6unit.json`。

## 4. 项目结构

```
de_power_dispatch/
├── main.py                       # 入口：依次跑基准函数 + ELD
├── requirements.txt              # 依赖：numpy / pandas / matplotlib
├── README.md                     # 本文件
├── report.md                     # 实验报告初稿（中文）
├── data/
│   └── eld_6unit.json            # IEEE 6-Units 简化数据
├── src/
│   ├── __init__.py
│   ├── differential_evolution.py # DE/rand/1/bin 实现
│   ├── benchmark_functions.py    # Sphere / Rastrigin / Rosenbrock
│   ├── economic_load_dispatch.py # ELD 系统、目标函数、修复函数
│   └── visualization.py          # 收敛曲线 / 出力柱状图
├── results/                      # 运行后自动生成
│   ├── benchmark_results.csv
│   ├── eld_results.csv
│   ├── sphere_convergence.png
│   ├── rastrigin_convergence.png
│   ├── eld_convergence.png
│   └── eld_dispatch_bar.png
└── docs/
    └── references.md             # 参考文献清单
```

## 5. 安装依赖

```bash
# 推荐使用项目自带的虚拟环境
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

依赖只包含 `numpy`、`pandas`、`matplotlib`，无 `seaborn`、无 GPU 依赖。

## 6. 运行方法

```bash
python main.py
```

运行流程：

1. Sphere 函数优化（dim=30，500 代）；
2. Rastrigin 函数优化（dim=30，500 代）；
3. ELD 6 机组优化（NP=80，800 代）；
4. 自动保存 CSV 和 PNG 到 `results/`；
5. 终端打印每代最优值、最终最优解和总成本。

随机种子固定为 `seed = 42`，结果可复现。

## 7. 输出结果说明

| 文件 | 含义 |
| --- | --- |
| `results/benchmark_results.csv` | 两个基准函数的最优值、误差、用时、超参数 |
| `results/sphere_convergence.png` | Sphere 函数收敛曲线（对数纵轴） |
| `results/rastrigin_convergence.png` | Rastrigin 函数收敛曲线（对数纵轴） |
| `results/eld_results.csv` | ELD 各机组出力、机组成本、总成本、误差 |
| `results/eld_convergence.png` | ELD 总成本随代数收敛曲线 |
| `results/eld_dispatch_bar.png` | ELD 各机组出力柱状图（含 Pmin/Pmax 误差棒） |

参考运行结果（seed=42）：

- **Sphere**: best ≈ 2.59 × 10⁻⁸（理论最优 0）。
- **Rastrigin**: best ≈ 196.4（30 维多峰问题，DE/rand/1/bin 在该规模下的合理水平；可通过加大种群或迭代代数继续改善）。
- **ELD**: ΣP = 1263.000000 MW（误差 0），最小总燃料成本 ≈ **15275.93 $/h**。
