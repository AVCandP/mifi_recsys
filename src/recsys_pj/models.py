"""Базовые модели из домашнего задания, приведённые к общему интерфейсу.

Каждая модель обучает только «книжную» часть (`fit`), а скор пользователя считает по переданной истории
(`score(users, history)`). Поэтому одна и та же обученная модель обслуживает и полную историю, и
укороченную (новые пользователи), и новых пользователей без переобучения.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .data import History, Indexer


class BaseModel:
    name = "base"

    def score(self, users: np.ndarray, hist: History) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


class PopularityModel(BaseModel):
    """Неперсонализированный рейтинг: log(1 + число лайков книги)."""
    name = "Popularity"

    def fit(self, hist: History) -> "PopularityModel":
        self.item_scores = np.log1p(hist.B.getnnz(axis=0)).astype(np.float32)
        return self

    def score(self, users, hist):
        return np.broadcast_to(self.item_scores, (len(users), len(self.item_scores))).copy()


def build_tfidf(books: pd.DataFrame, content_tags: pd.DataFrame, idx: Indexer, top_tags: int) -> sp.csr_matrix:
    """TF-IDF профиль книги: название + до top_tags содержательных тегов (теги — цельные токены)."""
    tag_text = (content_tags.sort_values(["book_id", "count"], ascending=[True, False])
                .groupby("book_id").head(top_tags).groupby("book_id")["tag_name"].agg(" ".join))
    b = books.set_index("book_id").reindex(idx.book_ids)
    titles = b["original_title"].fillna(b["title"]).fillna("").str.lower()
    profiles = titles + " " + tag_text.reindex(idx.book_ids).fillna("").to_numpy()
    vec = TfidfVectorizer(token_pattern=r"(?u)\b\w[\w-]+\b", stop_words="english", min_df=2, sublinear_tf=True)
    return vec.fit_transform(profiles.to_numpy()).astype(np.float32).tocsr()


class ContentModel(BaseModel):
    """Близость книги к TF-IDF-профилю пользователя (вес оценки 4 — 1, оценки 5 — 2)."""
    name = "Content (TF-IDF)"

    def __init__(self, tfidf: sp.csr_matrix, threshold: int = 4) -> None:
        self.tfidf, self.threshold = tfidf, threshold

    def fit(self, hist: History) -> "ContentModel":
        return self

    def user_profiles(self, users, hist) -> sp.csr_matrix:
        R = hist.R[users]
        likes = R.multiply(R >= self.threshold).tocsr()
        likes.data = likes.data - (self.threshold - 1)
        return normalize(likes @ self.tfidf, axis=1)

    def score(self, users, hist):
        return (self.user_profiles(users, hist) @ self.tfidf.T).toarray()


def _keep_top_k(values: np.ndarray, offset: int, k: int):
    top = np.argpartition(-values, k - 1, axis=1)[:, :k]
    tv = np.take_along_axis(values, top, axis=1)
    keep = tv > 0
    rows = np.repeat(np.arange(values.shape[0]), k)[keep.ravel()] + offset
    return rows, top[keep], tv[keep]


def item_similarity(X: sp.csr_matrix, B: sp.csr_matrix, shrink: float, k: int, chunk: int = 500) -> sp.csr_matrix:
    """Косинусная схожесть книг со сглаживанием по числу общих читателей, k соседей на книгу.

    Считается по блокам: память O(chunk × I) вместо O(I²).
    """
    n = X.shape[1]
    X_csc, B_csc = X.tocsc(), B.tocsc()
    norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=0)).ravel())
    norms[norms == 0] = 1.0
    parts = []
    for start in range(0, n, chunk):
        cols = np.arange(start, min(start + chunk, n))
        dot = (X_csc[:, cols].T @ X).toarray()
        co = dot if X is B else (B_csc[:, cols].T @ B).toarray()
        values = dot / (norms[cols, None] * norms[None, :])
        if shrink:
            values = values * co / (co + shrink)
        values[np.arange(len(cols)), cols] = 0.0
        parts.append(_keep_top_k(values, start, k))
    return sp.csr_matrix((np.concatenate([p[2] for p in parts]),
                          (np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]))),
                         shape=(n, n), dtype=np.float32)


def center_by_user(R: sp.csr_matrix) -> tuple[sp.csr_matrix, np.ndarray]:
    counts = np.maximum(R.getnnz(axis=1), 1)
    means = np.asarray(R.sum(axis=1)).ravel() / counts
    C = R.copy().astype(np.float32)
    C.data = C.data - np.repeat(means, np.diff(C.indptr)).astype(np.float32)
    return C, means


class ItemCFImplicit(BaseModel):
    """Item-Based CF по лайкам: скор = сумма схожестей книги с понравившимися."""
    name = "Item-CF (implicit)"

    def __init__(self, shrink: float, k: int) -> None:
        self.shrink, self.k = shrink, k

    def fit(self, hist: History) -> "ItemCFImplicit":
        self.sim = item_similarity(hist.B, hist.B, self.shrink, self.k)
        self.sim_t = self.sim.T.tocsr()
        return self

    def score(self, users, hist):
        return (hist.B[users] @ self.sim_t).toarray()


class ItemCFExplicit(BaseModel):
    """Item-Based CF по центрированным оценкам: скор — прогноз оценки со штрафом за малую опору."""
    name = "Item-CF (explicit)"

    def __init__(self, shrink: float, k: int, support_shrink: float) -> None:
        self.shrink, self.k, self.support_shrink = shrink, k, support_shrink

    def fit(self, hist: History) -> "ItemCFExplicit":
        centered, _ = center_by_user(hist.R)
        rated = (hist.R != 0).astype(np.float32).tocsr()
        self.sim = item_similarity(centered, rated, self.shrink, self.k)
        self.sim_t = self.sim.T.tocsr()
        return self

    def score(self, users, hist):
        R = hist.R[users]
        centered, means = center_by_user(R)
        rated = (R != 0).astype(np.float32)
        num = (centered @ self.sim_t).toarray()
        den = (rated @ self.sim_t).toarray()
        adj = np.divide(num, den + self.support_shrink, out=np.zeros_like(num), where=den > 0)
        return means[:, None].astype(np.float32) + adj


class PureSVD(BaseModel):
    """Усечённое SVD матрицы лайков: скор = B_u · V · Vᵀ (проекция истории на факторы книг)."""
    name = "PureSVD"

    def __init__(self, factors: int, seed: int = 42) -> None:
        self.factors, self.seed = factors, seed

    def fit(self, hist: History) -> "PureSVD":
        _, s, vt = svds(hist.B, k=self.factors, random_state=self.seed)
        order = np.argsort(-s)
        self.singular_values = s[order]
        self.item_factors = vt[order].astype(np.float32)
        return self

    def score(self, users, hist):
        return np.asarray(hist.B[users] @ self.item_factors.T) @ self.item_factors


def fit_base_models(hist: History, tfidf: sp.csr_matrix, cfg) -> dict[str, BaseModel]:
    """Обучает все базовые модели на одной истории. Порядок словаря задаёт порядок признаков."""
    models: list[BaseModel] = [
        PopularityModel().fit(hist),
        ContentModel(tfidf, cfg.relevance_threshold).fit(hist),
        ItemCFImplicit(cfg.itemcf_shrink, cfg.itemcf_implicit_k).fit(hist),
        ItemCFExplicit(cfg.itemcf_shrink, cfg.itemcf_explicit_k, cfg.itemcf_support_shrink).fit(hist),
        PureSVD(cfg.puresvd_factors, cfg.seed).fit(hist),
    ]
    return {m.name: m for m in models}
