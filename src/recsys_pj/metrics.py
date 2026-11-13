"""Этап «оценка качества»: Precision@K, Recall@K, nDCG@K и метрики разнообразия списка."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp


def metrics_at_k(hits: np.ndarray, n_relevant: np.ndarray, k: int) -> dict[str, np.ndarray]:
    """Метрики по матрице попаданий (пользователи × позиции), список уже упорядочен.

    Precision@K = hits/K; Recall@K = hits/|R_u|; nDCG@K = DCG/IDCG, где IDCG — DCG идеального списка
    из min(K, |R_u|) релевантных книг. Пользователи без релевантных книг в оценку не попадают.
    """
    top = hits[:, :k].astype(np.float32)
    n_hits = top.sum(axis=1)
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    ideal = np.concatenate(([0.0], np.cumsum(discounts)))[np.minimum(k, n_relevant)]
    return {f"Precision@{k}": n_hits / k,
            f"Recall@{k}": n_hits / n_relevant,
            f"nDCG@{k}": (top * discounts).sum(axis=1) / ideal}


def self_check() -> None:
    """Проверка формул на примере, посчитанном вручную."""
    m = metrics_at_k(np.array([[1, 0, 1]]), np.array([4]), 3)
    expected = (1 + 1 / np.log2(4)) / (1 + 1 / np.log2(3) + 1 / np.log2(4))
    assert np.isclose(m["Precision@3"][0], 2 / 3) and np.isclose(m["Recall@3"][0], 0.5)
    assert np.isclose(m["nDCG@3"][0], expected)
    assert np.isclose(metrics_at_k(np.array([[1, 1, 0]]), np.array([2]), 3)["nDCG@3"][0], 1.0)


def top_items(scores: np.ndarray, k: int) -> np.ndarray:
    """Индексы k лучших элементов каждой строки по убыванию."""
    k = min(k, scores.shape[1])
    part = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(scores, part, axis=1), axis=1, kind="stable")
    return np.take_along_axis(part, order, axis=1)


def hits_matrix(lists: np.ndarray, users: np.ndarray, relevance: sp.csr_matrix, batch: int = 4000) -> np.ndarray:
    out = []
    for s in range(0, len(users), batch):
        rel = relevance[users[s:s + batch]].toarray() > 0
        part = lists[s:s + batch]
        out.append(np.where(part >= 0, np.take_along_axis(rel, np.maximum(part, 0), axis=1), False))
    return np.vstack(out)


@dataclass
class Evaluation:
    """Списки рекомендаций и метрики для каждого пользователя."""
    name: str
    users: np.ndarray
    lists: np.ndarray
    per_user: dict[str, np.ndarray]

    def mean(self) -> dict[str, float]:
        return {k: float(v.mean()) for k, v in self.per_user.items()}


def evaluate_lists(name: str, lists: np.ndarray, users: np.ndarray, relevance: sp.csr_matrix,
                   ks: tuple[int, ...]) -> Evaluation:
    hits = hits_matrix(lists, users, relevance)
    n_rel = relevance[users].getnnz(axis=1)
    per_user: dict[str, np.ndarray] = {}
    for k in ks:
        per_user.update(metrics_at_k(hits, n_rel, k))
    return Evaluation(name, users, lists, per_user)


class BeyondAccuracy:
    """Метрики качества списка вне точности: покрытие, новизна, разнообразие, доля «длинного хвоста»."""

    def __init__(self, item_popularity: np.ndarray, item_embeddings: np.ndarray, tail_quantile: float = 0.9) -> None:
        pop = item_popularity.astype(np.float64)
        self.self_info = -np.log2((pop + 1) / (pop.sum() + len(pop)))
        self.is_tail = pop < np.quantile(pop, tail_quantile)  # вне 10% самых популярных
        self.emb = item_embeddings / np.maximum(np.linalg.norm(item_embeddings, axis=1, keepdims=True), 1e-9)
        self.n_items = len(pop)

    def __call__(self, lists: np.ndarray, k: int) -> dict[str, float]:
        top = lists[:, :k]
        e = self.emb[top]                                   # users × k × d
        sims = np.einsum("ukd,uld->ukl", e, e)
        pair_mean = (sims.sum(axis=(1, 2)) - k) / (k * (k - 1))
        return {f"Coverage@{k}": len(np.unique(top)) / self.n_items,
                f"Novelty@{k}": float(self.self_info[top].mean()),
                f"ILD@{k}": float((1 - pair_mean).mean()),
                f"LongTail@{k}": float(self.is_tail[top].mean())}


def summary_row(ev: Evaluation, beyond: BeyondAccuracy, k: int) -> dict[str, float]:
    return {"Модель": ev.name, **ev.mean(), **beyond(ev.lists, k)}


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 1000, seed: int = 42) -> tuple[float, float, float]:
    """Разница средних (a - b) по одним и тем же пользователям и её 95% доверительный интервал."""
    rng = np.random.default_rng(seed)
    diff = a - b
    means = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(n_boot)])
    return float(diff.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def segment_table(evals: list[Evaluation], segments: pd.Series, metric: str) -> pd.DataFrame:
    """Среднее значение метрики по сегментам пользователей (segments индексирован номером пользователя)."""
    rows = {}
    for ev in evals:
        seg = segments.reindex(ev.users).to_numpy()
        rows[ev.name] = pd.Series(ev.per_user[metric]).groupby(seg).mean()
    return pd.DataFrame(rows).T
