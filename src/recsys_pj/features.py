"""Расширенные признаки и генерация кандидатов.

* Признаки книг: популярность, средняя оценка и разброс оценок, энтропия распределения оценок, метаданные,
  тематические категории (NMF по содержательным тегам).
* Признаки пользователей: средняя оценка, число оценок и лайков, доля лайков, «мейнстримность»,
  тематический профиль и разнообразие интересов. Все они считаются по переданной истории, поэтому работают
  и для укороченных историй новых пользователей.
* Признаки взаимодействия: скоры всех базовых моделей, тематическое сходство с историей, книги того же
  автора в истории, отклонение средней оценки книги от средней оценки пользователя, число моделей-источников.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize

from .data import History, Indexer
from .metrics import top_items
from .models import BaseModel

ACTIVITY_BINS = np.array([0, 3, 10, 30, 60])  # границы корзин числа лайков пользователя
N_POP_BUCKETS = 10


@dataclass
class Topics:
    item_topics: np.ndarray      # книги × темы, строки нормированы на сумму 1
    names: list[str]

    @property
    def n_topics(self) -> int:
        return self.item_topics.shape[1]


def build_topics(content_tags: pd.DataFrame, idx: Indexer, n_topics: int, top_tags: int, seed: int) -> Topics:
    """Тематические категории книг: NMF по TF-IDF матрице «книга × содержательный тег»."""
    tags = (content_tags.sort_values(["book_id", "count"], ascending=[True, False])
            .groupby("book_id").head(top_tags))
    tag_codes, tag_names = pd.factorize(tags["tag_name"])
    X = sp.csr_matrix((np.log1p(tags["count"].to_numpy(dtype=np.float32)),
                       (idx.items(tags["book_id"]), tag_codes)), shape=(idx.n_items, len(tag_names)))
    X = TfidfTransformer(sublinear_tf=False).fit_transform(X)
    nmf = NMF(n_components=n_topics, init="nndsvda", random_state=seed, max_iter=400)
    W = nmf.fit_transform(X)
    names = [" / ".join(tag_names[np.argsort(-h)[:3]]) for h in nmf.components_]
    W = W / np.maximum(W.sum(axis=1, keepdims=True), 1e-9)
    W[W.sum(axis=1) == 0] = 1.0 / n_topics
    return Topics(W.astype(np.float32), names)


def content_embeddings(tfidf: sp.csr_matrix, dim: int, seed: int) -> np.ndarray:
    """Плотные эмбеддинги книг (SVD от TF-IDF) — для разнообразия списка и MMR."""
    emb = TruncatedSVD(n_components=dim, random_state=seed).fit_transform(tfidf)
    return normalize(emb).astype(np.float32)


def _first_author(authors: pd.Series) -> pd.Series:
    return authors.fillna("").str.split(",").str[0].str.strip()


class FeatureBuilder:
    """Считает признаки книг по обучающей истории и признаки пользователей/пар по переданной истории."""

    def __init__(self, stats_hist: History, books: pd.DataFrame, idx: Indexer, topics: Topics,
                 threshold: int) -> None:
        self.topics, self.threshold = topics, threshold
        R = stats_hist.R
        n_items = R.shape[1]
        counts = R.getnnz(axis=0).astype(np.float32)
        likes = stats_hist.B.getnnz(axis=0).astype(np.float32)
        sums = np.asarray(R.sum(axis=0)).ravel()
        sq = np.asarray(R.multiply(R).sum(axis=0)).ravel()
        mean = np.divide(sums, counts, out=np.zeros(n_items), where=counts > 0)
        std = np.sqrt(np.maximum(np.divide(sq, counts, out=np.zeros(n_items), where=counts > 0) - mean ** 2, 0))
        global_mean = sums.sum() / max(counts.sum(), 1)
        m = 20.0  # сглаживание средней к глобальной для редких книг
        mean_shrunk = (sums + m * global_mean) / (counts + m)
        # Энтропия распределения оценок 1..5 — «разнообразие оценок» книги
        dist = np.vstack([np.asarray((R == v).sum(axis=0)).ravel() for v in range(1, 6)]).T.astype(np.float64)
        p = dist / np.maximum(dist.sum(axis=1, keepdims=True), 1)
        entropy = -(np.where(p > 0, p * np.log2(np.where(p > 0, p, 1)), 0)).sum(axis=1)

        b = books.set_index("book_id").reindex(idx.book_ids)
        year = b["original_publication_year"].fillna(b["original_publication_year"].median()).clip(1800, 2017)

        self.item_log_ratings = np.log1p(counts)
        self.item_mean = mean_shrunk.astype(np.float32)
        self.item_frame = pd.DataFrame({
            "item_log_ratings": self.item_log_ratings,
            "item_log_likes": np.log1p(likes),
            "item_like_share": np.divide(likes, counts, out=np.zeros(n_items, np.float32), where=counts > 0),
            "item_mean_rating": mean_shrunk,
            "item_std_rating": std,
            "item_rating_entropy": entropy,
            "item_goodreads_avg": b["average_rating"].fillna(b["average_rating"].mean()).to_numpy(),
            "item_goodreads_log_count": np.log1p(b["ratings_count"].fillna(0).to_numpy()),
            "item_year": ((year - 2000) / 20).to_numpy(),
            "item_topic_focus": topics.item_topics.max(axis=1),
        }).astype(np.float32)
        self.item_matrix = self.item_frame.to_numpy()
        self.item_top_topic = topics.item_topics.argmax(axis=1).astype(np.int64)
        edges = np.quantile(self.item_log_ratings, np.linspace(0, 1, N_POP_BUCKETS + 1)[1:-1])
        self.item_pop_bucket = np.searchsorted(edges, self.item_log_ratings).astype(np.int64)

        authors = _first_author(b["authors"])
        codes, self.author_names = pd.factorize(authors)
        self.item_author = codes.astype(np.int64)
        self.n_authors = len(self.author_names)

    # ---------- признаки пользователя ----------
    def user_features(self, users: np.ndarray, hist: History) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        R, B = hist.R[users], hist.B[users]
        n_r = R.getnnz(axis=1).astype(np.float32)
        n_l = B.getnnz(axis=1).astype(np.float32)
        s = np.asarray(R.sum(axis=1)).ravel()
        sq = np.asarray(R.multiply(R).sum(axis=1)).ravel()
        mean = np.divide(s, n_r, out=np.full(len(users), 3.9, np.float64), where=n_r > 0)
        std = np.sqrt(np.maximum(np.divide(sq, n_r, out=np.zeros(len(users)), where=n_r > 0) - mean ** 2, 0))
        rated = (R != 0).astype(np.float32)
        mainstream = np.divide(np.asarray(rated @ self.item_log_ratings).ravel(), n_r,
                               out=np.zeros(len(users), np.float64), where=n_r > 0)
        profile = np.asarray(B @ self.topics.item_topics)
        profile = np.divide(profile, profile.sum(axis=1, keepdims=True),
                            out=np.full_like(profile, 1 / self.topics.n_topics), where=profile.sum(axis=1, keepdims=True) > 0)
        topic_entropy = -(np.where(profile > 0, profile * np.log2(np.where(profile > 0, profile, 1)), 0)).sum(axis=1)
        frame = pd.DataFrame({
            "user_log_ratings": np.log1p(n_r), "user_log_likes": np.log1p(n_l),
            "user_like_share": np.divide(n_l, n_r, out=np.zeros(len(users), np.float32), where=n_r > 0),
            "user_mean_rating": mean, "user_std_rating": std,
            "user_mainstream": mainstream, "user_topic_entropy": topic_entropy,
        }).astype(np.float32)
        return frame, profile.astype(np.float32), n_l

    @staticmethod
    def activity_bucket(n_likes: np.ndarray) -> np.ndarray:
        return np.searchsorted(ACTIVITY_BINS, n_likes, side="right").astype(np.int64) - 1 + (n_likes > 0)

    def author_counts(self, users: np.ndarray, hist: History) -> sp.csr_matrix:
        """Матрица пользователь × автор: число понравившихся книг автора."""
        B = hist.B[users].tocoo()
        return sp.csr_matrix((B.data, (B.row, self.item_author[B.col])), shape=(len(users), self.n_authors))


@dataclass
class CandidateBatch:
    users: np.ndarray            # индексы пользователей
    cand: np.ndarray             # пользователи × P, -1 — пустая позиция
    valid: np.ndarray            # пользователи × P
    z: np.ndarray                # пользователи × P × M — z-оценки скоров базовых моделей
    features: np.ndarray         # пользователи × P × F
    user_top_topic: np.ndarray   # пользователи
    user_activity: np.ndarray    # пользователи
    n_likes: np.ndarray          # пользователи
    single_lists: dict[str, np.ndarray]  # топ-K каждой базовой модели по всему каталогу
    hist_items: list                     # понравившиеся книги каждого пользователя (для Deep-части)


class CandidateGenerator:
    """Объединяет топ-N каждой базовой модели в общий пул, исключает прочитанное и считает признаки."""

    def __init__(self, models: dict[str, BaseModel], builder: FeatureBuilder, n_per_model: int, list_k: int,
                 history_len: int = 100, seed: int = 42) -> None:
        self.models, self.builder = models, builder
        self.history_len, self.seed = history_len, seed
        self.n_per_model, self.list_k = n_per_model, list_k
        self.model_names = list(models)
        self.feature_names = (
            list(builder.item_frame.columns)
            + ["user_log_ratings", "user_log_likes", "user_like_share", "user_mean_rating", "user_std_rating",
               "user_mainstream", "user_topic_entropy"]
            + [f"score::{n}" for n in self.model_names] + [f"z::{n}" for n in self.model_names]
            + ["topic_affinity", "author_likes", "mean_diff", "pop_diff", "n_sources"]
        )

    def generate(self, users: np.ndarray, hist: History) -> CandidateBatch:
        n_users, M = len(users), len(self.models)
        seen = hist.R[users].tocoo()
        raw_tops, stats, thresholds, scores = [], [], [], []
        for model in self.models.values():
            s = np.asarray(model.score(users, hist), dtype=np.float32)
            mu, sd = s.mean(axis=1), s.std(axis=1)
            stats.append((mu, np.where(sd > 0, sd, 1.0)))
            s[seen.row, seen.col] = -np.inf                       # фильтрация уже прочитанного
            top = top_items(s, self.n_per_model)
            raw_tops.append(top)
            thresholds.append(np.take_along_axis(s, top[:, -1:], axis=1)[:, 0])
            scores.append(s)
        # Объединение списков в пул с удалением повторов
        cand = np.sort(np.hstack(raw_tops), axis=1)
        dup = np.zeros_like(cand, dtype=bool)
        dup[:, 1:] = cand[:, 1:] == cand[:, :-1]
        cand[dup] = -1
        valid = cand >= 0
        safe = np.maximum(cand, 0)

        raw = np.stack([np.take_along_axis(s, safe, axis=1) for s in scores], axis=2)        # U × P × M
        mu = np.stack([m for m, _ in stats], axis=1)[:, None, :]
        sd = np.stack([d for _, d in stats], axis=1)[:, None, :]
        z = ((raw - mu) / sd).astype(np.float32)
        n_sources = np.stack([raw[:, :, m] >= thresholds[m][:, None] for m in range(M)], axis=2).sum(axis=2)

        b = self.builder
        user_frame, profile, n_likes = b.user_features(users, hist)
        user_mat = user_frame.to_numpy()
        item_mat = b.item_matrix[safe]                                                          # U × P × Fi
        topic_aff = np.einsum("upt,ut->up", b.topics.item_topics[safe], profile)
        authors = b.author_counts(users, hist).toarray()
        author_likes = np.log1p(np.take_along_axis(authors, b.item_author[safe], axis=1))
        mean_diff = b.item_mean[safe] - user_frame["user_mean_rating"].to_numpy()[:, None]
        pop_diff = b.item_log_ratings[safe] - user_frame["user_mainstream"].to_numpy()[:, None]
        signed_log = np.sign(raw) * np.log1p(np.abs(raw))
        features = np.concatenate([
            item_mat,
            np.broadcast_to(user_mat[:, None, :], (n_users, cand.shape[1], user_mat.shape[1])),
            signed_log, z,
            np.stack([topic_aff, author_likes, mean_diff, pop_diff, n_sources], axis=2),
        ], axis=2).astype(np.float32)
        features[~valid] = 0.0
        z[~valid] = 0.0
        single = {name: top[:, :self.list_k] for name, top in zip(self.model_names, raw_tops)}
        rng = np.random.default_rng(self.seed)
        B = hist.B
        hist_items = []
        for u in users:
            items = B.indices[B.indptr[u]:B.indptr[u + 1]].astype(np.int64)
            hist_items.append(rng.choice(items, self.history_len, replace=False) if len(items) > self.history_len else items)
        return CandidateBatch(users, cand, valid, z, features, profile.argmax(axis=1).astype(np.int64),
                              b.activity_bucket(n_likes), n_likes, single, hist_items)
