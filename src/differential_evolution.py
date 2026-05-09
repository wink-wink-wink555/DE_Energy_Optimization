"""差分进化算法 (Differential Evolution, DE)

从零实现 DE/rand/1/bin 策略，不依赖 scipy.optimize.differential_evolution。

DE 基本流程：
    1. 初始化种群（在 [lower, upper] 范围内均匀采样）
    2. 变异 (Mutation): v_i = x_r1 + F * (x_r2 - x_r3)
    3. 交叉 (Crossover, Binomial): 以概率 CR 用 v_i 替换 x_i 的分量，至少保留 1 维
    4. 选择 (Selection): 贪心选择，若 trial 适应度优于父代则替换
    5. 记录每一代的最优适应度，形成收敛曲线
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np


@dataclass
class DEResult:
    """DE 优化结果。"""

    best_solution: np.ndarray   # 最优个体
    best_fitness: float         # 最优适应度
    history: np.ndarray         # 每代最优适应度，形状 (max_generations + 1,)
    n_eval: int                 # 适应度函数评估次数


class DifferentialEvolution:
    """DE/rand/1/bin 实现。

    参数
    ------
    func : Callable
        待优化的目标函数，签名 ``f(x: np.ndarray) -> float``，向最小化方向优化。
    bounds : Sequence
        每一维的搜索范围 [(lo_1, up_1), ..., (lo_d, up_d)]。
    population_size : int
        种群规模 NP。
    F : float
        差分缩放因子，建议 [0.4, 1.0]。
    CR : float
        交叉概率，建议 [0.7, 1.0]。
    max_generations : int
        最大迭代代数 G_max。
    seed : int | None
        随机种子，便于结果复现。
    repair : Callable | None
        可选的修复函数，对越界/违反约束的个体做修复。
        签名 ``repair(x: np.ndarray) -> np.ndarray``。
    """

    def __init__(
        self,
        func: Callable[[np.ndarray], float],
        bounds: Sequence[tuple],
        population_size: int = 50,
        F: float = 0.5,
        CR: float = 0.9,
        max_generations: int = 500,
        seed: Optional[int] = 42,
        repair: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> None:
        self.func = func
        self.bounds = np.asarray(bounds, dtype=float)
        if self.bounds.ndim != 2 or self.bounds.shape[1] != 2:
            raise ValueError("bounds must have shape (dim, 2)")
        self.dim = self.bounds.shape[0]
        self.lower = self.bounds[:, 0]
        self.upper = self.bounds[:, 1]

        self.NP = int(population_size)
        if self.NP < 4:
            raise ValueError("population_size must be >= 4 for DE/rand/1/bin")
        self.F = float(F)
        self.CR = float(CR)
        self.G_max = int(max_generations)
        self.repair = repair

        self.rng = np.random.default_rng(seed)
        self._n_eval = 0

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    def _evaluate(self, x: np.ndarray) -> float:
        """评估单个个体，并计数。"""
        self._n_eval += 1
        return float(self.func(x))

    def _initialize(self) -> np.ndarray:
        """在搜索范围内均匀初始化种群。"""
        pop = self.rng.uniform(
            low=self.lower, high=self.upper, size=(self.NP, self.dim)
        )
        if self.repair is not None:
            pop = np.array([self.repair(ind) for ind in pop])
        return pop

    def _mutate(self, pop: np.ndarray, i: int) -> np.ndarray:
        """DE/rand/1: v = x_r1 + F * (x_r2 - x_r3)。"""
        idxs = [j for j in range(self.NP) if j != i]
        r1, r2, r3 = self.rng.choice(idxs, size=3, replace=False)
        v = pop[r1] + self.F * (pop[r2] - pop[r3])
        # 边界处理：越界则反射到搜索区间内（也可改为重采样/截断）
        v = np.where(v < self.lower, self.lower + (self.lower - v), v)
        v = np.where(v > self.upper, self.upper - (v - self.upper), v)
        v = np.clip(v, self.lower, self.upper)
        return v

    def _crossover(self, target: np.ndarray, donor: np.ndarray) -> np.ndarray:
        """二项式交叉：以概率 CR 用 donor 分量替换 target 分量，至少保留 1 维。"""
        cross_mask = self.rng.random(self.dim) < self.CR
        # 保证至少有一维来自 donor
        j_rand = int(self.rng.integers(0, self.dim))
        cross_mask[j_rand] = True
        trial = np.where(cross_mask, donor, target)
        return trial

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------
    def optimize(self, verbose: bool = False) -> DEResult:
        pop = self._initialize()
        fitness = np.array([self._evaluate(ind) for ind in pop])

        history = np.empty(self.G_max + 1, dtype=float)
        history[0] = fitness.min()

        for g in range(1, self.G_max + 1):
            for i in range(self.NP):
                donor = self._mutate(pop, i)
                trial = self._crossover(pop[i], donor)
                if self.repair is not None:
                    trial = self.repair(trial)
                else:
                    trial = np.clip(trial, self.lower, self.upper)

                f_trial = self._evaluate(trial)
                # 贪心选择
                if f_trial <= fitness[i]:
                    pop[i] = trial
                    fitness[i] = f_trial

            history[g] = fitness.min()
            if verbose and (g % max(1, self.G_max // 10) == 0 or g == 1):
                print(f"  Gen {g:4d}/{self.G_max} | best = {history[g]:.6e}")

        best_idx = int(np.argmin(fitness))
        return DEResult(
            best_solution=pop[best_idx].copy(),
            best_fitness=float(fitness[best_idx]),
            history=history,
            n_eval=self._n_eval,
        )
