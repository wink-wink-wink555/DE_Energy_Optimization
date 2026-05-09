"""电力系统经济负荷分配 (Economic Load Dispatch, ELD)

数据来源：IEEE 6-Units ELD Test System No.1 (Al-Roomi)。
本课程实验采用简化版：
    - 仅考虑二次燃料成本；
    - 仅考虑机组出力上下限和功率平衡约束；
    - 暂不考虑网损、阀点效应、爬坡约束、禁运区。

数学模型：
    minimize  F = sum_i (a_i + b_i P_i + c_i P_i^2)
    s.t.      sum_i P_i = PD            (功率平衡, 等式约束)
              Pmin_i <= P_i <= Pmax_i   (出力上下限, 不等式约束)

约束处理（repair + penalty 组合）：
    1. 先 clip 到 [Pmin, Pmax]；
    2. 通过迭代修复 sum(P_i) 接近 PD（按可用余量分摊偏差）；
    3. 若仍有微小残差，在适应度中加 penalty: penalty_factor * |sum(P) - PD|。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


@dataclass
class ELDSystem:
    """ELD 系统数据容器。"""

    name: str
    n: int
    PD: float                 # 总负荷需求 (MW)
    Pmin: np.ndarray          # 各机组出力下限 (MW)
    Pmax: np.ndarray          # 各机组出力上限 (MW)
    a: np.ndarray             # 二次成本系数
    b: np.ndarray
    c: np.ndarray

    def __post_init__(self) -> None:
        for attr in ("Pmin", "Pmax", "a", "b", "c"):
            setattr(self, attr, np.asarray(getattr(self, attr), dtype=float))
        if not (self.Pmin.size == self.Pmax.size == self.a.size == self.b.size == self.c.size == self.n):
            raise ValueError("ELD parameter array sizes do not match n")
        if not (self.Pmin.sum() <= self.PD <= self.Pmax.sum()):
            raise ValueError("Load demand PD is infeasible w.r.t. Pmin/Pmax sums")


def ieee_6units_system1() -> ELDSystem:
    """IEEE 6-Units ELD Test System No.1 简化版数据。"""
    return ELDSystem(
        name="IEEE 6-Units ELD Test System No.1",
        n=6,
        PD=1263.0,
        Pmin=[100, 50, 80, 50, 50, 50],
        Pmax=[500, 200, 300, 150, 200, 120],
        a=[240, 200, 220, 200, 220, 190],
        b=[7.0, 10.0, 8.5, 11.0, 10.5, 12.0],
        c=[0.0070, 0.0095, 0.0090, 0.0090, 0.0080, 0.0075],
    )


def fuel_cost(P: np.ndarray, sys: ELDSystem) -> float:
    r"""总燃料成本 F = sum_i (a_i + b_i P_i + c_i P_i^2)。"""
    P = np.asarray(P, dtype=float)
    return float(np.sum(sys.a + sys.b * P + sys.c * P * P))


def repair_power_balance(
    P: np.ndarray,
    sys: ELDSystem,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> np.ndarray:
    """对个体做 box clip + 等式约束修复。

    思路：
        - 先把每个分量裁剪到 [Pmin_i, Pmax_i]；
        - 计算 delta = PD - sum(P)；
        - 若 delta != 0，按各机组当前的可用余量 (Pmax - P 或 P - Pmin) 比例分摊；
        - 多次迭代直到 |delta| 足够小或达到上限。
    """
    P = np.clip(np.asarray(P, dtype=float), sys.Pmin, sys.Pmax)
    for _ in range(max_iter):
        delta = sys.PD - P.sum()
        if abs(delta) < tol:
            break
        if delta > 0:
            # 需要增加出力，按 (Pmax - P) 余量分摊
            slack = sys.Pmax - P
        else:
            # 需要减少出力，按 (P - Pmin) 余量分摊
            slack = P - sys.Pmin
        total = slack.sum()
        if total <= 1e-12:
            break  # 没有可调整空间
        share = slack / total
        P = P + share * delta
        P = np.clip(P, sys.Pmin, sys.Pmax)
    return P


def make_eld_objective(
    sys: ELDSystem, penalty_factor: float = 1e5
) -> "tuple[callable, callable]":
    """构造 (适应度函数, 修复函数)。

    适应度函数已经内置了 penalty，可直接交给 DifferentialEvolution。
    修复函数也单独返回，便于 DE 在交叉/变异后立刻应用。
    """

    def _repair(x: np.ndarray) -> np.ndarray:
        return repair_power_balance(x, sys)

    def _fitness(x: np.ndarray) -> float:
        # x 在传入前应该已经 repair 过；这里再保险地 clip 一下
        x = np.clip(np.asarray(x, dtype=float), sys.Pmin, sys.Pmax)
        cost = fuel_cost(x, sys)
        imbalance = abs(x.sum() - sys.PD)
        return cost + penalty_factor * imbalance

    return _fitness, _repair
