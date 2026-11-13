"""Гибридизация: взвешенное объединение по сегментам пользователей, подбор весов, балансировка MMR."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .features import CandidateBatch
from .metrics import metrics_at_k, top_items

SEGMENTS = ("Новые", "Умеренные", "Активные")


def segment_of(n_likes: np.ndarray, new_max: int, active_min: int) -> np.ndarray:
    """0 — новые (≤ new_max лайков), 1 — умеренные, 2 — активные (> active_min)."""
    return np.where(n_likes <= new_max, 0, np.where(n_likes > active_min, 2, 1))


def segment_labels(new_max: int, active_min: int) -> list[str]:
    return [f"Новые (≤{new_max} лайков)", f"Умеренные ({new_max + 1}–{active_min})", f"Активные (>{active_min})"]


def lists_from_scores(cand: np.ndarray, scores: np.ndarray, valid: np.ndarray, k: int) -> np.ndarray:
    """Топ-k книг из пула по скору; пустые позиции пула не выбираются."""
    s = np.where(valid, scores, -np.inf)
    order = top_items(s, k)
    lists = np.take_along_axis(cand, order, axis=1)
    return np.where(np.take_along_axis(valid, order, axis=1), lists, -1)


@dataclass
class WeightedHybrid:
    """Скор = Σ_m w_m(сегмент) · z_m, где z_m — z-оценка скора базовой модели внутри списка пользователя."""
    component_names: list[str]
    weights: dict[int, np.ndarray]          # сегмент -> веса компонентов
    new_max: int
    active_min: int
    name: str = "Гибрид (взвешенный)"

    def score(self, batch: CandidateBatch) -> np.ndarray:
        seg = segment_of(batch.n_likes, self.new_max, self.active_min)
        W = np.stack([self.weights[s] for s in seg])            # U × M
        return np.einsum("upm,um->up", batch.z, W)

    def recommend(self, batch: CandidateBatch, k: int) -> np.ndarray:
        return lists_from_scores(batch.cand, self.score(batch), batch.valid, k)

    def weights_table(self, labels: list[str]) -> pd.DataFrame:
        return pd.DataFrame({labels[s]: w for s, w in self.weights.items()}, index=self.component_names).T


@dataclass
class TuningSet:
    """Пул кандидатов с z-оценками и метками релевантности — данные для подбора весов и порогов."""
    z: np.ndarray            # N × P × M
    valid: np.ndarray        # N × P
    labels: np.ndarray       # N × P (bool)
    n_relevant: np.ndarray   # N — всего релевантных в отложенном периоде (не только в пуле)
    segment: np.ndarray      # N
    tag: np.ndarray          # N — происхождение строки: "full" или "n=…" для укороченных историй

    def subset(self, mask: np.ndarray) -> "TuningSet":
        return TuningSet(self.z[mask], self.valid[mask], self.labels[mask], self.n_relevant[mask],
                         self.segment[mask], self.tag[mask])

    @staticmethod
    def concat(parts: list["TuningSet"]) -> "TuningSet":
        return TuningSet(*(np.concatenate([getattr(p, f) for p in parts]) for f in
                           ("z", "valid", "labels", "n_relevant", "segment", "tag")))


def ndcg_for_scores(ts: TuningSet, scores: np.ndarray, k: int) -> np.ndarray:
    s = np.where(ts.valid, scores, -np.inf)
    order = top_items(s, k)
    hits = np.take_along_axis(ts.labels & ts.valid, order, axis=1)
    return metrics_at_k(hits, ts.n_relevant, k)[f"nDCG@{k}"]


def tune_weights(ts: TuningSet, n_samples: int, k: int, seed: int) -> tuple[np.ndarray, pd.DataFrame]:
    """Случайный поиск по симплексу весов (Дирихле) + локальное уточнение вокруг лучшей точки.

    Возвращает лучшие веса и таблицу всех проверенных вариантов с nDCG@k.
    """
    rng = np.random.default_rng(seed)
    M = ts.z.shape[2]
    candidates = [np.eye(M)[m] for m in range(M)] + [np.full(M, 1 / M)]
    candidates += list(rng.dirichlet(np.ones(M), size=n_samples // 2))
    rows = []

    def check(w):
        value = float(ndcg_for_scores(ts, ts.z @ w, k).mean())
        rows.append({**{f"w{m}": w[m] for m in range(M)}, f"nDCG@{k}": value})
        return value

    for w in candidates:
        check(w)
    best = np.array([rows[int(np.argmax([r[f"nDCG@{k}"] for r in rows]))][f"w{m}"] for m in range(M)])
    for scale in (0.15, 0.07, 0.03):                      # локальное уточнение
        for _ in range((n_samples - len(candidates)) // 3):
            w = np.clip(best + rng.normal(0, scale, M), 0, None)
            w = w / w.sum() if w.sum() > 0 else best
            if check(w) >= max(r[f"nDCG@{k}"] for r in rows[:-1]):
                best = w
    table = pd.DataFrame(rows)
    best = table.iloc[table[f"nDCG@{k}"].idxmax()][[f"w{m}" for m in range(M)]].to_numpy(dtype=np.float64)
    return best, table


def mmr_rerank(cand: np.ndarray, scores: np.ndarray, valid: np.ndarray, emb: np.ndarray,
               lam: float, k: int, depth: int) -> np.ndarray:
    """Maximal Marginal Relevance: λ·relevance − (1−λ)·max сходство с уже выбранными.

    relevance — скор, приведённый к [0, 1] внутри первых depth кандидатов пользователя;
    сходство — косинус контентных эмбеддингов книг.
    """
    top = lists_from_scores(cand, scores, valid, depth)
    s = np.where(valid, scores, -np.inf)
    top_scores = -np.sort(-s, axis=1)[:, :depth]
    ok = top >= 0
    lo = np.where(ok, top_scores, np.inf).min(axis=1, keepdims=True)
    hi = np.where(ok, top_scores, -np.inf).max(axis=1, keepdims=True)
    rel = np.where(ok, (top_scores - lo) / np.maximum(hi - lo, 1e-9), -np.inf)
    if lam >= 1.0:
        return top[:, :k]
    E = emb[np.maximum(top, 0)]                                          # U × D × d
    max_sim = np.zeros(top.shape, dtype=np.float32)
    taken = ~ok
    rows = np.arange(len(top))
    out = np.full((len(top), k), -1)
    for step in range(k):
        value = lam * rel - (1 - lam) * max_sim
        value[taken] = -np.inf
        j = value.argmax(axis=1)
        out[:, step] = np.where(np.isfinite(value[rows, j]), top[rows, j], -1)
        taken[rows, j] = True
        sim = np.einsum("udk,uk->ud", E, E[rows, j])
        max_sim = np.maximum(max_sim, sim)
    return out


@dataclass
class Router:
    """Итоговая система: смесь нейросетевого ранжировщика и взвешенного гибрида с долей α по сегменту.

    score = α_s · z(ранжировщик) + (1 − α_s) · z(взвешенный гибрид), затем MMR для разнообразия.
    α_s = 1 — только нейросеть, α_s = 0 — только классический гибрид.
    """
    ranker: object
    hybrid: WeightedHybrid
    alpha: dict[int, float]
    new_max: int
    active_min: int
    item_emb: np.ndarray
    mmr_lambda: float = 1.0
    mmr_depth: int = 100
    name: str = "Итоговая гибридная система"

    def score(self, batch: CandidateBatch) -> np.ndarray:
        seg = segment_of(batch.n_likes, self.new_max, self.active_min)
        a = np.array([self.alpha[s] for s in seg], dtype=np.float32)[:, None]
        mixed = a * _standardize(self.ranker.score(batch), batch.valid) + \
            (1 - a) * _standardize(self.hybrid.score(batch), batch.valid)
        return np.where(batch.valid, mixed, -np.inf)

    def recommend(self, batch: CandidateBatch, k: int, mmr_lambda: float | None = None) -> np.ndarray:
        lam = self.mmr_lambda if mmr_lambda is None else mmr_lambda
        return mmr_rerank(batch.cand, self.score(batch), batch.valid, self.item_emb, lam, k, self.mmr_depth)


def _standardize(scores: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """z-оценка скоров внутри пула пользователя по заполненным позициям."""
    s = np.where(valid, scores, np.nan).astype(np.float64)
    mu = np.nanmean(s, axis=1, keepdims=True)
    sd = np.nanstd(s, axis=1, keepdims=True)
    return np.nan_to_num((s - mu) / np.where(sd > 0, sd, 1), nan=0.0).astype(np.float32)
