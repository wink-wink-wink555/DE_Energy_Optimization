"""经典优化测试函数。

提供两个常用基准函数：
    - Sphere: 单峰、可分、凸函数，用于验证 DE 的基础收敛性。
    - Rastrigin: 多峰、不可分，含大量局部极小值，用于验证全局搜索能力。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class BenchmarkProblem:
    """基准函数定义。"""

    name: str
    func: Callable[[np.ndarray], float]
    dim: int
    lower: float
    upper: float
    optimum: float  # 理论最优函数值

    @property
    def bounds(self):
        return [(self.lower, self.upper)] * self.dim


def sphere(x: np.ndarray) -> float:
    r"""Sphere 函数: f(x) = \sum_i x_i^2 ; 最优解 x*=0, f*=0。"""
    x = np.asarray(x, dtype=float)
    return float(np.sum(x * x))


def rastrigin(x: np.ndarray) -> float:
    r"""Rastrigin 函数: f(x) = 10 n + \sum_i [x_i^2 - 10 cos(2 pi x_i)]; 最优 0。"""
    x = np.asarray(x, dtype=float)
    n = x.size
    return float(10.0 * n + np.sum(x * x - 10.0 * np.cos(2.0 * np.pi * x)))


def rosenbrock(x: np.ndarray) -> float:
    r"""Rosenbrock 函数（扩展实验备用）。"""
    x = np.asarray(x, dtype=float)
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))


def get_default_problems(dim: int = 30) -> list[BenchmarkProblem]:
    """返回课程实验所需的两个基准函数。"""
    return [
        BenchmarkProblem(
            name="Sphere",
            func=sphere,
            dim=dim,
            lower=-5.12,
            upper=5.12,
            optimum=0.0,
        ),
        BenchmarkProblem(
            name="Rastrigin",
            func=rastrigin,
            dim=dim,
            lower=-5.12,
            upper=5.12,
            optimum=0.0,
        ),
    ]
