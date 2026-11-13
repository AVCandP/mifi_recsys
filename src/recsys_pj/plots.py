"""Графики для ноутбука. Все функции сохраняют PNG в папку figures и возвращают фигуру."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#8a8984"]
GRAY, INK = "#8a8984", "#2b2b2b"

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight", "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": GRAY,
    "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.7, "axes.axisbelow": True,
    "legend.frameon": False,
})

BASE_MODELS = ["Popularity", "Content (TF-IDF)", "Item-CF (implicit)", "Item-CF (explicit)", "PureSVD"]


def _save(fig, figures_dir: Path, name: str):
    Path(figures_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(figures_dir) / f"{name}.png")
    return fig


def topics(topic_names: list[str], item_topics: np.ndarray, figures_dir: Path):
    share = pd.Series(np.bincount(item_topics.argmax(axis=1), minlength=len(topic_names)) / len(item_topics),
                      index=topic_names).sort_values()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(share.index, share.values * 100, color=COLORS[0], height=0.7)
    ax.set(title="Тематические категории книг (NMF по тегам)", xlabel="Доля книг с этой главной темой, %")
    ax.grid(axis="y", visible=False)
    return _save(fig, figures_dir, "01_topics")


def hybrid_weights(weights: pd.DataFrame, figures_dir: Path):
    w = weights.div(weights.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(9, 3.4))
    left = np.zeros(len(w))
    for i, col in enumerate(w.columns):
        ax.barh(w.index, w[col], left=left, color=COLORS[i], height=0.6, label=col, edgecolor="white", linewidth=2)
        for y, (l, v) in enumerate(zip(left, w[col])):
            if v >= 0.08:
                ax.text(l + v / 2, y, f"{v:.2f}", ha="center", va="center", color="white", fontsize=9)
        left += w[col].to_numpy()
    ax.set(title="Веса компонентов гибрида по сегментам", xlabel="Доля веса", xlim=(0, 1))
    ax.invert_yaxis()
    ax.grid(False)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=9)
    return _save(fig, figures_dir, "02_hybrid_weights")


def weight_search(search: pd.DataFrame, components: list[str], segment: str, figures_dir: Path):
    data = search[search["Сегмент"] == segment]
    fig, axes = plt.subplots(1, len(components), figsize=(14, 3), sharey=True)
    best = data.loc[data["nDCG@10"].idxmax()]
    for ax, comp, color in zip(axes, components, COLORS):
        ax.scatter(data[comp], data["nDCG@10"], s=10, color=color, alpha=0.45, linewidths=0)
        ax.scatter([best[comp]], [best["nDCG@10"]], s=60, color=INK, zorder=3)
        ax.set(title=comp, xlabel="Вес")
        ax.title.set_fontsize(10)
    axes[0].set_ylabel("nDCG@10 (validation)")
    fig.suptitle(f"Случайный поиск весов: {segment} (чёрная точка — выбранные веса)", x=0.01, ha="left",
                 fontweight="bold")
    fig.tight_layout()
    return _save(fig, figures_dir, "03_weight_search")


def mmr_tradeoff(mmr: pd.DataFrame, chosen: float, figures_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    for ax, col, title in zip(axes, ["nDCG@10", "ILD@10", "Coverage@10"],
                              ["Точность: nDCG@10", "Разнообразие списка: ILD@10", "Покрытие каталога"]):
        ax.plot(mmr["λ"], mmr[col], color=COLORS[0], linewidth=2, marker="o", markersize=5)
        y = mmr.loc[mmr["λ"] == chosen, col].iloc[0]
        ax.scatter([chosen], [y], s=80, color=COLORS[1], zorder=3, label=f"выбрано λ = {chosen}")
        ax.set(title=title, xlabel="λ (1 — без балансировки)")
        ax.invert_xaxis()
    axes[0].legend(loc="lower left")
    fig.tight_layout()
    return _save(fig, figures_dir, "04_mmr_tradeoff")


def training_curves(history: pd.DataFrame, figures_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    axes[0].plot(history["эпоха"], history["loss"], color=COLORS[0], linewidth=2, marker="o", markersize=5)
    axes[0].set(title="Функция потерь на обучении", xlabel="Эпоха", ylabel="Loss")
    axes[1].plot(history["эпоха"], history["val nDCG@10"], color=COLORS[2], linewidth=2, marker="o", markersize=5)
    best = history.loc[history["val nDCG@10"].idxmax()]
    axes[1].scatter([best["эпоха"]], [best["val nDCG@10"]], s=80, color=COLORS[1], zorder=3,
                    label=f"лучшая эпоха: {int(best['эпоха'])}")
    axes[1].set(title="nDCG@10 на validation", xlabel="Эпоха")
    axes[1].legend(loc="lower right")
    fig.tight_layout()
    return _save(fig, figures_dir, "05_wd_training")


def feature_importance(table: pd.DataFrame, figures_dir: Path):
    t = table.sort_values("Падение nDCG@10")
    fig, ax = plt.subplots(figsize=(8, 3.6))
    colors = [COLORS[0] if v > 0 else GRAY for v in t["Падение nDCG@10"]]
    ax.barh(t["Группа признаков"], t["Падение nDCG@10"], color=colors, height=0.65)
    ax.axvline(0, color=GRAY, linewidth=1)
    ax.set(title="Важность групп признаков для Wide & Deep", xlabel="Падение nDCG@10 при перемешивании группы")
    ax.grid(axis="y", visible=False)
    return _save(fig, figures_dir, "06_feature_importance")


def alpha_blend(table: pd.DataFrame, figures_dir: Path):
    alphas = [c for c in table.columns if isinstance(c, (int, float)) or str(c).replace(".", "").isdigit()]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    x = np.array([float(a) for a in alphas])
    for i, (seg, row) in enumerate(table.iterrows()):
        y = row[alphas].astype(float).to_numpy()
        rel = y / y[0]
        ax.plot(x, rel, color=COLORS[i], linewidth=2, label=seg)
        j = int(np.argmax(y))
        ax.scatter([x[j]], [rel[j]], s=60, color=COLORS[i], zorder=3, edgecolor="white", linewidth=2)
    ax.axhline(1, color=GRAY, linewidth=1)
    ax.set(title="Доля Wide & Deep в итоговой смеси", xlabel="α (0 — только взвешенный гибрид, 1 — только W&D)",
           ylabel="nDCG@10 относительно α = 0")
    ax.legend(loc="best")
    return _save(fig, figures_dir, "07_alpha_blend")


def model_comparison(metrics: pd.DataFrame, figures_dir: Path):
    order = metrics.sort_values("nDCG@10").index
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), sharey=True)
    colors = [GRAY if m in BASE_MODELS else COLORS[0] for m in order]
    for ax, col in zip(axes, ["Precision@10", "Recall@10", "nDCG@10"]):
        values = metrics.loc[order, col]
        ax.barh(order, values, color=colors, height=0.68)
        for y, v in enumerate(values):
            ax.text(v, y, f" {v:.3f}", va="center", fontsize=8, color=INK)
        ax.set(title=col)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, values.max() * 1.18)
    handles = [plt.Rectangle((0, 0), 1, 1, color=GRAY), plt.Rectangle((0, 0), 1, 1, color=COLORS[0])]
    fig.legend(handles, ["Отдельные модели из ДЗ", "Гибридные и нейросетевые"], ncol=2, loc="lower center",
               bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Сравнение моделей на тесте (K = 10)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return _save(fig, figures_dir, "08_model_comparison")


def accuracy_vs_diversity(metrics: pd.DataFrame, figures_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    for ax, col, title in zip(axes, ["Coverage@10", "ILD@10"], ["Точность и покрытие каталога",
                                                                  "Точность и разнообразие списка"]):
        xr = metrics[col].max() - metrics[col].min()
        yr = metrics["nDCG@10"].max() - metrics["nDCG@10"].min()
        placed = []
        for m, row in metrics.sort_values("nDCG@10", ascending=False).iterrows():
            base = m in BASE_MODELS
            x, y = row[col], row["nDCG@10"]
            ax.scatter(x, y, s=55, color=GRAY if base else COLORS[0], zorder=3, edgecolor="white", linewidth=1.5)
            dy = 3
            while any(abs(x - px) / xr < 0.35 and abs((y - py) / yr * 260 + (dy - pdy)) < 11 for px, py, pdy in placed):
                dy -= 11
            placed.append((x, y, dy))
            ax.annotate(m, (x, y), xytext=(14 if dy != 3 else 6, dy), textcoords="offset points", fontsize=8,
                        color=INK, va="center",
                        arrowprops=dict(arrowstyle="-", color=GRAY, linewidth=0.7) if dy != 3 else None)
        ax.set(title=title, xlabel=col, ylabel="nDCG@10")
    fig.tight_layout()
    return _save(fig, figures_dir, "09_accuracy_vs_diversity")


def segment_heatmap(table: pd.DataFrame, figures_dir: Path, name: str = "10_segments", title: str = ""):
    fig, ax = plt.subplots(figsize=(1.9 * table.shape[1] + 3.5, 0.42 * table.shape[0] + 1.4))
    rel = table / table.max(axis=0)
    ax.imshow(rel.to_numpy(), cmap="Blues", vmin=0, vmax=1.15, aspect="auto")
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            v = table.iat[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=9,
                    color="white" if rel.iat[i, j] > 0.75 else INK,
                    fontweight="bold" if v == table.iloc[:, j].max() else "normal")
    ax.set_xticks(range(table.shape[1]), table.columns, rotation=0)
    ax.set_yticks(range(table.shape[0]), table.index)
    ax.grid(False)
    ax.set_title(title or "nDCG@10 по сегментам пользователей (жирным — лучший в столбце)")
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    return _save(fig, figures_dir, name)


def cold_start(cold: pd.DataFrame, models: list[str], figures_dir: Path):
    fig, ax = plt.subplots(figsize=(8.5, 4))
    for i, m in enumerate(models):
        d = cold[cold["Модель"] == m].sort_values("Лайков в истории")
        ax.plot(d["Лайков в истории"], d["nDCG@10"], color=COLORS[i], linewidth=2, marker="o", markersize=6, label=m)
    ax.set(title="Новые пользователи: качество при короткой истории", xlabel="Известно понравившихся книг",
           ylabel="nDCG@10 (тест)")
    ax.set_xticks(sorted(cold["Лайков в истории"].unique()))
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    return _save(fig, figures_dir, "11_cold_start")


def bootstrap(table: pd.DataFrame, figures_dir: Path):
    t = table.sort_values("Δ nDCG@10")
    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    y = np.arange(len(t))
    colors = [COLORS[0] if lo > 0 else (COLORS[1] if hi < 0 else GRAY)
              for lo, hi in zip(t["95% ДИ, низ"], t["95% ДИ, верх"])]
    ax.hlines(y, t["95% ДИ, низ"], t["95% ДИ, верх"], color=colors, linewidth=2)
    ax.scatter(t["Δ nDCG@10"], y, color=colors, s=50, zorder=3)
    ax.axvline(0, color=GRAY, linewidth=1)
    ax.set_yticks(y, t["Система"])
    ax.set(title=f"Разница nDCG@10 с лучшей отдельной моделью ({t['Сравнение с'].iloc[0]})",
           xlabel="Δ nDCG@10 и 95% доверительный интервал (бутстрэп по пользователям)")
    ax.grid(axis="y", visible=False)
    return _save(fig, figures_dir, "12_bootstrap")


# ------------------------------------------------------------------ Раздел 9: улучшения
def source_recall(table: pd.DataFrame, figures_dir: Path):
    t = table.copy()
    pools = t["Источник"].str.startswith("Пул")
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.8), gridspec_kw={"width_ratios": [2, 1.2]})
    src = t[~pools].sort_values("Recall@50")
    new = src["Источник"].isin(["EASE", "RP3beta", "SASRec"])
    axes[0].barh(src["Источник"], src["Recall@50"], color=[COLORS[1] if n else GRAY for n in new], height=0.65)
    for y, v in enumerate(src["Recall@50"]):
        axes[0].text(v, y, f" {v:.3f}", va="center", fontsize=8)
    axes[0].set(title="Recall@50 каждого источника (validation)", xlim=(0, src["Recall@50"].max() * 1.2))
    p = t[pools]
    axes[1].bar(range(len(p)), p["Recall@50"], color=[GRAY, COLORS[1], COLORS[1]][:len(p)], width=0.6)
    for x, (v, s) in enumerate(zip(p["Recall@50"], p["Размер пула"])):
        axes[1].text(x, v, f"{v:.1%}\n~{s:.0f} книг", ha="center", va="bottom", fontsize=8)
    axes[1].set_xticks(range(len(p)), [s.split(" ")[1] for s in p["Источник"]])
    axes[1].set(title="Доля релевантных в пуле", ylim=(0, p["Recall@50"].max() * 1.3))
    for ax in axes:
        ax.grid(axis="y" if ax is axes[0] else "x", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=GRAY), plt.Rectangle((0, 0), 1, 1, color=COLORS[1])]
    fig.legend(handles, ["Источники проекта", "Новые источники (H1, H5)"], ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return _save(fig, figures_dir, "13_source_recall")


def ablation_steps(metrics: pd.DataFrame, rows: list[tuple[str, str]], figures_dir: Path):
    """Кумулятивные шаги улучшений: подпись → имя строки в таблице метрик теста."""
    labels = [r[0] for r in rows]
    values = [metrics.loc[r[1], "nDCG@10"] for r in rows]
    fig, ax = plt.subplots(figsize=(10, 0.5 * len(rows) + 1.2))
    y = np.arange(len(rows))[::-1]
    ax.barh(y, values, color=[GRAY] + [COLORS[0]] * (len(rows) - 2) + [COLORS[1]], height=0.62)
    for yy, v, prev in zip(y, values, [None] + values[:-1]):
        delta = "" if prev is None else f"  ({(v / values[0] - 1) * 100:+.0f}% к C0)"
        ax.text(v, yy, f" {v:.3f}{delta}", va="center", fontsize=8.5)
    ax.set_yticks(y, labels)
    ax.set(title="Вклад гипотез: nDCG@10 на тесте", xlim=(0, max(values) * 1.3))
    ax.grid(axis="y", visible=False)
    return _save(fig, figures_dir, "14_ablation_steps")


def stages(table: pd.DataFrame, figures_dir: Path):
    """Сравнение этапов: ДЗ → итоговый проект → улучшения (столбцы «Этап», «Система», «nDCG@10»)."""
    stage_colors = {s: c for s, c in zip(table["Этап"].unique(), [GRAY, COLORS[0], COLORS[1]])}
    fig, ax = plt.subplots(figsize=(10, 0.45 * len(table) + 1.3))
    y = np.arange(len(table))[::-1]
    ax.barh(y, table["nDCG@10"], color=[stage_colors[s] for s in table["Этап"]], height=0.62)
    for yy, v in zip(y, table["nDCG@10"]):
        ax.text(v, yy, f" {v:.3f}", va="center", fontsize=8.5)
    ax.set_yticks(y, table["Система"])
    ax.set(title="nDCG@10 на тесте: домашнее задание → итоговый проект → улучшения", xlim=(0, table["nDCG@10"].max() * 1.15))
    ax.grid(axis="y", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in stage_colors.values()]
    ax.legend(handles, list(stage_colors), loc="upper center", bbox_to_anchor=(0.45, -0.12), ncol=3)
    return _save(fig, figures_dir, "15_stages")


def cold_ext(table: pd.DataFrame, models: list[str], figures_dir: Path):
    fig, ax = plt.subplots(figsize=(8.5, 4))
    for i, m in enumerate(models):
        d = table[table["Модель"] == m].sort_values("Лайков в истории")
        ax.plot(d["Лайков в истории"], d["nDCG@10"], color=COLORS[i], linewidth=2, marker="o", markersize=6,
                label=m, linestyle="--" if m.startswith("Проект") else "-")
    ax.set(title="Новые пользователи: проект и улучшения на одних пользователях", xlabel="Известно понравившихся книг",
           ylabel="nDCG@10 (тест)")
    ax.set_xticks(sorted(table["Лайков в истории"].unique()))
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    return _save(fig, figures_dir, "16_cold_start_ext")
