"""项目入口：依次完成

1. Sphere 函数优化（第 4 节，基础验证）；
2. Rastrigin 函数优化（第 4 节，多峰验证）；
3. ELD 经济负荷分配优化（第 5 节，能源电力应用）；
4. 保存所有 CSV 与 PNG 结果到 ``results/`` 目录；
5. 在终端打印关键结果。

运行方式::

    python main.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# 让 src/ 模块可被导入（无论从项目根还是上级目录运行）
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmark_functions import get_default_problems
from src.differential_evolution import DifferentialEvolution
from src.economic_load_dispatch import (
    fuel_cost,
    ieee_6units_system1,
    make_eld_objective,
)
from src.visualization import plot_convergence, plot_eld_dispatch


RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# DE 公共超参数（按提示词要求）
SEED = 42
POP_SIZE = 50
MAX_GEN = 500
F = 0.5
CR = 0.9


def run_benchmarks() -> pd.DataFrame:
    """第 4 节：经典优化函数实验。"""
    print("=" * 60)
    print("第 4 节：经典优化问题（Sphere / Rastrigin）")
    print("=" * 60)

    problems = get_default_problems(dim=30)
    rows = []

    for prob in problems:
        print(f"\n>> 求解 {prob.name} 函数（dim={prob.dim}）")
        de = DifferentialEvolution(
            func=prob.func,
            bounds=prob.bounds,
            population_size=POP_SIZE,
            F=F,
            CR=CR,
            max_generations=MAX_GEN,
            seed=SEED,
        )
        t0 = time.perf_counter()
        result = de.optimize(verbose=True)
        elapsed = time.perf_counter() - t0

        print(f"  best_fitness = {result.best_fitness:.6e}")
        print(f"  ||best_solution||_inf = {np.max(np.abs(result.best_solution)):.4e}")
        print(f"  评估次数 = {result.n_eval}, 用时 = {elapsed:.2f} s")

        # 保存收敛曲线
        png_path = RESULTS_DIR / f"{prob.name.lower()}_convergence.png"
        plot_convergence(
            history=result.history,
            title=f"DE Convergence on {prob.name} (dim={prob.dim})",
            save_path=png_path,
            log_scale=True,
        )
        print(f"  收敛曲线已保存：{png_path.name}")

        rows.append(
            {
                "function": prob.name,
                "dim": prob.dim,
                "theoretical_optimum": prob.optimum,
                "best_fitness": result.best_fitness,
                "abs_error": abs(result.best_fitness - prob.optimum),
                "n_eval": result.n_eval,
                "time_seconds": round(elapsed, 4),
                "population_size": POP_SIZE,
                "max_generations": MAX_GEN,
                "F": F,
                "CR": CR,
                "seed": SEED,
            }
        )

    df = pd.DataFrame(rows)
    csv_path = RESULTS_DIR / "benchmark_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n基准函数结果已写入：{csv_path}")
    return df


def run_eld() -> pd.DataFrame:
    """第 5 节：电力系统经济负荷分配。"""
    print("\n" + "=" * 60)
    print("第 5 节：电力系统经济负荷分配（IEEE 6-Units ELD）")
    print("=" * 60)

    sys_eld = ieee_6units_system1()
    fitness_fn, repair_fn = make_eld_objective(sys_eld, penalty_factor=1e5)

    # ELD 问题维度较低（6），可用更大的种群和迭代数提高精度
    de = DifferentialEvolution(
        func=fitness_fn,
        bounds=list(zip(sys_eld.Pmin, sys_eld.Pmax)),
        population_size=80,
        F=0.5,
        CR=0.9,
        max_generations=800,
        seed=SEED,
        repair=repair_fn,
    )

    t0 = time.perf_counter()
    result = de.optimize(verbose=True)
    elapsed = time.perf_counter() - t0

    P_opt = repair_fn(result.best_solution)
    total_P = float(P_opt.sum())
    cost = fuel_cost(P_opt, sys_eld)
    err = abs(total_P - sys_eld.PD)

    print("\n>> ELD 最优结果")
    for i, p in enumerate(P_opt, start=1):
        print(f"  P{i} = {p:10.4f} MW   (limits: {sys_eld.Pmin[i-1]:.0f} ~ {sys_eld.Pmax[i-1]:.0f})")
    print(f"  sum(P) = {total_P:.6f} MW, PD = {sys_eld.PD:.2f} MW, |Δ| = {err:.6e} MW")
    print(f"  最小总燃料成本 F* = {cost:.6f} $/h")
    print(f"  评估次数 = {result.n_eval}, 用时 = {elapsed:.2f} s")

    # 保存收敛曲线
    plot_convergence(
        history=result.history,
        title="DE Convergence on IEEE 6-Units ELD",
        save_path=RESULTS_DIR / "eld_convergence.png",
        log_scale=False,
    )
    # 出力柱状图
    plot_eld_dispatch(
        P=P_opt,
        Pmin=sys_eld.Pmin,
        Pmax=sys_eld.Pmax,
        save_path=RESULTS_DIR / "eld_dispatch_bar.png",
        title="ELD Optimal Dispatch (IEEE 6-Units, PD=1263 MW)",
    )

    rows = [
        {
            "unit": f"P{i+1}",
            "power_MW": float(P_opt[i]),
            "Pmin_MW": float(sys_eld.Pmin[i]),
            "Pmax_MW": float(sys_eld.Pmax[i]),
            "cost_$/h": float(
                sys_eld.a[i] + sys_eld.b[i] * P_opt[i] + sys_eld.c[i] * P_opt[i] ** 2
            ),
        }
        for i in range(sys_eld.n)
    ]
    rows.append(
        {
            "unit": "TOTAL",
            "power_MW": total_P,
            "Pmin_MW": float(sys_eld.Pmin.sum()),
            "Pmax_MW": float(sys_eld.Pmax.sum()),
            "cost_$/h": cost,
        }
    )
    rows.append(
        {
            "unit": "DEMAND",
            "power_MW": float(sys_eld.PD),
            "Pmin_MW": np.nan,
            "Pmax_MW": np.nan,
            "cost_$/h": np.nan,
        }
    )
    rows.append(
        {
            "unit": "ABS_ERROR_MW",
            "power_MW": err,
            "Pmin_MW": np.nan,
            "Pmax_MW": np.nan,
            "cost_$/h": np.nan,
        }
    )

    df = pd.DataFrame(rows)
    csv_path = RESULTS_DIR / "eld_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"  ELD 结果表已写入：{csv_path}")
    print(f"  收敛曲线：{(RESULTS_DIR / 'eld_convergence.png').name}")
    print(f"  出力柱状图：{(RESULTS_DIR / 'eld_dispatch_bar.png').name}")
    return df


def main() -> None:
    np.random.seed(SEED)  # 仅供旁路使用，DE 内部使用独立 RNG
    df_bm = run_benchmarks()
    df_eld = run_eld()

    print("\n" + "=" * 60)
    print("全部实验已完成。结果保存在 results/ 目录：")
    for p in sorted(RESULTS_DIR.iterdir()):
        print(f"  - {p.name}")
    print("=" * 60)


if __name__ == "__main__":
    main()
