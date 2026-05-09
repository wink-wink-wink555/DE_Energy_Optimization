"""结果可视化：收敛曲线、ELD 出力柱状图。"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

# 选择非交互后端，便于在无显示环境下保存图片
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_convergence(
    history: np.ndarray,
    title: str,
    save_path: Path,
    log_scale: bool = True,
) -> None:
    """绘制收敛曲线（每代最优适应度）。"""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7.5, 4.5))
    generations = np.arange(history.size)
    if log_scale and np.all(history > 0):
        plt.semilogy(generations, history, linewidth=1.6)
        plt.ylabel("Best fitness (log scale)")
    else:
        plt.plot(generations, history, linewidth=1.6)
        plt.ylabel("Best fitness")
    plt.xlabel("Generation")
    plt.title(title)
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_eld_dispatch(
    P: np.ndarray,
    Pmin: np.ndarray,
    Pmax: np.ndarray,
    save_path: Path,
    title: str = "ELD Optimal Dispatch",
) -> None:
    """绘制 ELD 各机组出力柱状图，并标注上下限。"""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(P)
    idx = np.arange(1, n + 1)

    plt.figure(figsize=(8, 4.8))
    bars = plt.bar(idx, P, color="#3a7bd5", edgecolor="black", label="Output P_i")

    # 上下限参考线（误差棒形式）
    plt.errorbar(
        idx,
        (Pmin + Pmax) / 2,
        yerr=[((Pmin + Pmax) / 2 - Pmin), (Pmax - (Pmin + Pmax) / 2)],
        fmt="none",
        ecolor="grey",
        capsize=8,
        alpha=0.6,
        label="[Pmin, Pmax]",
    )

    for bar, p in zip(bars, P):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 5,
            f"{p:.2f}",
            ha="center",
            fontsize=9,
        )

    plt.xticks(idx, [f"Unit {i}" for i in idx])
    plt.ylabel("Power output (MW)")
    plt.title(title)
    plt.legend(loc="upper right")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
