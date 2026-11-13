"""Проведение исследования улучшений (раздел 9 ноутбука) поверх обученного основного пайплайна."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .data import History
from .features import FeatureBuilder
from .hybrid import mmr_rerank, segment_of
from .improvements import EASE, RP3beta, SASRec, SeriesInfo, source_recall
from .metrics import BeyondAccuracy, evaluate_lists, metrics_at_k, paired_bootstrap, top_items
from .neural import GBDTRanker, RankerData, WideDeepRanker, ranked_ndcg

BASE_NAMES = ["item_log_ratings", "item_log_likes", "item_like_share", "item_mean_rating", "item_std_rating",
              "item_rating_entropy", "item_goodreads_avg", "item_goodreads_log_count", "item_year", "item_topic_focus",
              "user_log_ratings", "user_log_likes", "user_like_share", "user_mean_rating", "user_std_rating",
              "user_mainstream", "user_topic_entropy", "topic_affinity", "author_likes", "mean_diff", "pop_diff"]


@dataclass
class StudyConfig:
    candidates_per_model: int = 50
    ease_lambdas: tuple = (250.0, 1000.0, 4000.0)
    rp3_betas: tuple = (0.3, 0.5, 0.7)
    source_tune_users: int = 3000
    sasrec_epochs: int = 4
    tune_users: int = 3000          # половина A — веса и ранняя остановка, половина B — выбор смеси
    cold_tune_users: int = 1500
    rank_users: int = 10000
    truncated_share: float = 0.25
    lgbm_trees: int = 400
    cold_eval_users: int = 5000
    max_test_users: int | None = None


@dataclass
class Variant:
    """Конфигурация для абляции: какие источники кандидатов и признаки доступны ранжировщику."""
    name: str
    models: list[str]
    series: bool = False


class ExtBatch:
    """Пул кандидатов от расширенного набора источников со всеми признаками в «мастер»-формате."""

    def __init__(self, users, cand, valid, master, flags, n_likes, user_topic, user_act, hist_items, single):
        self.users, self.cand, self.valid, self.master, self.flags = users, cand, valid, master, flags
        self.n_likes, self.user_topic, self.user_act, self.hist_items, self.single = \
            n_likes, user_topic, user_act, hist_items, single


class ExtGenerator:
    def __init__(self, models: dict, builder: FeatureBuilder, series: SeriesInfo, n_per: int, list_k: int = 20,
                 history_len: int = 100, seed: int = 42) -> None:
        self.models, self.builder, self.series = models, builder, series
        self.n_per, self.list_k, self.history_len, self.seed = n_per, list_k, history_len, seed
        self.names = list(models)
        M = len(self.names)
        self.columns = (BASE_NAMES + [f"score::{n}" for n in self.names] + [f"z::{n}" for n in self.names]
                        + [f"flag::{n}" for n in self.names] + series.feature_names)
        self.col = {c: i for i, c in enumerate(self.columns)}
        self.M = M

    def generate(self, users: np.ndarray, hist: History) -> ExtBatch:
        b, n_users = self.builder, len(users)
        seen = hist.R[users].tocoo()
        tops, stats, thresholds, scores = [], [], [], []
        for model in self.models.values():
            s = np.asarray(model.score(users, hist), dtype=np.float32)
            mu, sd = s.mean(axis=1), s.std(axis=1)
            stats.append((mu, np.where(sd > 0, sd, 1.0)))
            s[seen.row, seen.col] = -np.inf
            top = top_items(s, self.n_per)
            tops.append(top)
            thresholds.append(np.take_along_axis(s, top[:, -1:], axis=1)[:, 0])
            scores.append(s)
        cand = np.sort(np.hstack(tops), axis=1)
        dup = np.zeros_like(cand, dtype=bool)
        dup[:, 1:] = cand[:, 1:] == cand[:, :-1]
        cand[dup] = -1
        valid = cand >= 0
        safe = np.maximum(cand, 0)
        raw = np.stack([np.take_along_axis(s, safe, axis=1) for s in scores], axis=2)
        mu = np.stack([m for m, _ in stats], axis=1)[:, None, :]
        sd = np.stack([d for _, d in stats], axis=1)[:, None, :]
        z = (raw - mu) / sd
        flags = np.stack([raw[:, :, m] >= thresholds[m][:, None] for m in range(self.M)], axis=2)
        user_frame, profile, n_likes = b.user_features(users, hist)
        user_mat = user_frame.to_numpy()
        topic_aff = np.einsum("upt,ut->up", b.topics.item_topics[safe], profile)
        authors = b.author_counts(users, hist).toarray()
        author_likes = np.log1p(np.take_along_axis(authors, b.item_author[safe], axis=1))
        mean_diff = b.item_mean[safe] - user_frame["user_mean_rating"].to_numpy()[:, None]
        pop_diff = b.item_log_ratings[safe] - user_frame["user_mainstream"].to_numpy()[:, None]
        master = np.concatenate([
            b.item_matrix[safe], np.broadcast_to(user_mat[:, None, :], (n_users, cand.shape[1], user_mat.shape[1])),
            np.stack([topic_aff, author_likes, mean_diff, pop_diff], axis=2),
            np.sign(raw) * np.log1p(np.abs(raw)), z, flags.astype(np.float32),
            self.series.features(users, hist, cand),
        ], axis=2).astype(np.float32)
        master[~valid] = 0
        rng = np.random.default_rng(self.seed)
        hist_items = []
        for u in users:
            items = hist.B.indices[hist.B.indptr[u]:hist.B.indptr[u + 1]].astype(np.int64)
            hist_items.append(rng.choice(items, self.history_len, replace=False) if len(items) > self.history_len else items)
        single = {n: t[:, :self.list_k] for n, t in zip(self.names, tops)}
        return ExtBatch(users, cand, valid, master, flags & valid[:, :, None], n_likes,
                        profile.argmax(axis=1).astype(np.int64), FeatureBuilder.activity_bucket(n_likes), hist_items, single)


def flatten(batch: ExtBatch, Y: sp.csr_matrix | None, tag: str) -> tuple[RankerData, np.ndarray]:
    """Пул пользователей → плоские строки мастер-признаков (+ число лайков в истории каждого пользователя)."""
    v = batch.valid
    if Y is not None:
        rel = Y[batch.users].toarray() > 0
        labels = np.take_along_axis(rel, np.maximum(batch.cand, 0), axis=1) & v
        n_rel = rel.sum(axis=1)
    else:
        labels, n_rel = np.zeros_like(v), np.zeros(len(batch.users), dtype=np.int64)
    sizes = v.sum(axis=1)
    data = RankerData(batch.master[v].astype(np.float16), batch.cand[v].astype(np.int64), labels[v].astype(np.float32),
                      np.r_[0, np.cumsum(sizes)], batch.hist_items, batch.user_topic, batch.user_act, n_rel,
                      np.array([tag] * len(batch.users)))
    return data, batch.n_likes


def row_user(data: RankerData) -> np.ndarray:
    return np.repeat(np.arange(data.n_users), np.diff(data.offsets))


def variant_data(master: RankerData, gen: ExtGenerator, variant: Variant) -> RankerData:
    """Строки и столбцы мастер-данных, доступные конфигурации (пул — только её источники)."""
    flag_cols = [gen.col[f"flag::{n}"] for n in variant.models]
    keep = master.features[:, flag_cols].sum(axis=1) > 0
    cols = list(range(len(BASE_NAMES))) + [gen.col[f"score::{n}"] for n in variant.models] + \
        [gen.col[f"z::{n}"] for n in variant.models]
    if variant.series:
        cols += [gen.col[c] for c in gen.series.feature_names]
    rows = master.features[keep]
    X = np.concatenate([rows[:, cols], rows[:, flag_cols].astype(np.float32).sum(axis=1, keepdims=True).astype(rows.dtype)],
                       axis=1)
    del rows
    sizes = np.bincount(row_user(master)[keep], minlength=master.n_users)
    return RankerData(X, master.items[keep], master.labels[keep], np.r_[0, np.cumsum(sizes)], master.hist_items,
                      master.user_topic, master.user_activity, master.n_relevant, master.tag)


def standardize_flat(data: RankerData, scores: np.ndarray) -> np.ndarray:
    ru = row_user(data)
    n = np.maximum(np.bincount(ru, minlength=data.n_users), 1)
    mu = np.bincount(ru, weights=scores, minlength=data.n_users) / n
    var = np.bincount(ru, weights=scores ** 2, minlength=data.n_users) / n - mu ** 2
    sd = np.sqrt(np.maximum(var, 0))
    return ((scores - mu[ru]) / np.where(sd > 0, sd, 1)[ru]).astype(np.float32)


def padded(data: RankerData, scores: np.ndarray):
    """Плоские скоры → (пользователи × P) кандидаты, скоры, маска, метки."""
    sizes = np.diff(data.offsets)
    P = max(int(sizes.max()), 1)
    ru = row_user(data)
    pos = np.arange(len(scores)) - np.repeat(data.offsets[:-1], sizes)
    cand = np.full((data.n_users, P), -1, dtype=np.int64)
    S = np.full((data.n_users, P), -np.inf, dtype=np.float32)
    L = np.zeros((data.n_users, P), dtype=bool)
    cand[ru, pos], S[ru, pos], L[ru, pos] = data.items, scores, data.labels > 0
    return cand, S, cand >= 0, L


def lists_ndcg(lists: np.ndarray, cand: np.ndarray, labels: np.ndarray, n_rel: np.ndarray, k: int = 10) -> np.ndarray:
    pos = np.where(labels, cand, -2)
    hits = np.stack([(lists[:, j:j + 1] == pos).any(axis=1) & (lists[:, j] >= 0) for j in range(k)], axis=1)
    return metrics_at_k(hits, n_rel, k)[f"nDCG@{k}"]


class LambdaMARTRanker:
    """H3: LightGBM LambdaMART — оптимизирует nDCG напрямую через попарные градиенты по спискам пользователей."""

    def __init__(self, trees: int, seed: int, name: str = "LambdaMART") -> None:
        import lightgbm as lgb
        self.name = name
        self.model = lgb.LGBMRanker(objective="lambdarank", n_estimators=trees, learning_rate=0.05, num_leaves=63,
                                    min_child_samples=100, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                    lambdarank_truncation_level=30, random_state=seed, n_jobs=2, verbose=-1)

    def fit(self, train: RankerData, valid: RankerData) -> "LambdaMARTRanker":
        import lightgbm as lgb

        def groups(d):
            sizes = np.diff(d.offsets)
            pos = np.bincount(row_user(d), weights=d.labels, minlength=d.n_users)
            keep_user = (sizes > 0) & (pos > 0)
            rows = np.repeat(keep_user, sizes)
            return d.features[rows].astype(np.float32), d.labels[rows].astype(int), sizes[keep_user]

        X, y, g = groups(train)
        Xv, yv, gv = groups(valid)
        import warnings
        with warnings.catch_warnings():  # eval_set устарел в новых LightGBM, но нужен для совместимости со старыми
            warnings.simplefilter("ignore")
            self.model.fit(X, y, group=g, eval_set=[(Xv, yv)], eval_group=[gv], eval_at=[10],
                           callbacks=[lgb.early_stopping(30, verbose=False)])
        self.best_iteration = self.model.best_iteration_
        return self

    def predict_data(self, data: RankerData) -> np.ndarray:
        return np.concatenate([self.model.predict(data.features[i:i + 500_000].astype(np.float32))
                               for i in range(0, len(data.features), 500_000)]) if len(data.features) else np.zeros(0)


def tune_weights_flat(data: RankerData, Z: np.ndarray, samples: int, seed: int) -> tuple[np.ndarray, float]:
    """Случайный поиск весов z-оценок источников по nDCG@10 на плоских данных."""
    rng = np.random.default_rng(seed)
    M = Z.shape[1]
    cands = [np.eye(M)[m] for m in range(M)] + [np.full(M, 1 / M)] + list(rng.dirichlet(np.ones(M), samples))
    best_w, best = None, -1.0
    for w in cands:
        v = float(ranked_ndcg(data, Z @ w).mean())
        if v > best:
            best, best_w = v, w
    for scale in (0.1, 0.05):
        for _ in range(samples // 4):
            w = np.clip(best_w + rng.normal(0, scale, M), 0, None)
            if w.sum() == 0:
                continue
            w = w / w.sum()
            v = float(ranked_ndcg(data, Z @ w).mean())
            if v > best:
                best, best_w = v, w
    return best_w, best


class ImprovementStudy:
    """Последовательная проверка гипотез H0–H5 на тех же данных, разбиении и метриках, что и основной пайплайн."""

    OLD = ["Popularity", "Content (TF-IDF)", "Item-CF (implicit)", "Item-CF (explicit)", "PureSVD"]

    def __init__(self, pipe, scfg: StudyConfig | None = None) -> None:
        self.pipe, self.cfg, self.s = pipe, pipe.cfg, scfg or StudyConfig()
        self.sp = pipe.splits
        self.rng = np.random.default_rng(pipe.cfg.seed + 100)
        self.series = SeriesInfo(pipe.raw.books, self.sp.idx)
        self.results: dict[str, pd.DataFrame] = {}
        self.timer = pipe.timer
        self.variants = [
            Variant("C0: 5 источников из проекта", self.OLD),
            Variant("C1: + EASE и RP3beta", self.OLD + ["EASE", "RP3beta"]),
            Variant("C2: + признаки серий", self.OLD + ["EASE", "RP3beta"], series=True),
            Variant("C3: + SASRec", self.OLD + ["EASE", "RP3beta", "SASRec"], series=True),
        ]

    # ------------------------------------------------------------------ H0
    def check_order(self) -> "ImprovementStudy":
        with self.timer("9.1. H0: порядок строк и серии книг"):
            self.results["h0_order"] = self.series.order_check(self.pipe.raw.ratings, self.sp.idx)
            s = self.series
            self.results["series_stats"] = pd.DataFrame({
                "Значение": [int((s.item_series >= 0).sum()), s.n_series,
                             int(pd.Series(s.item_series[s.item_series >= 0]).value_counts().ge(2).sum())]},
                index=["Книг в сериях", "Серий", "Серий с 2+ книгами в каталоге"])
        return self

    # ------------------------------------------------------------------ H1 + H5: новые источники
    def _recall_table(self, models: dict, users: np.ndarray, hist: History, Y) -> pd.DataFrame:
        rel = Y[users].toarray() > 0
        n_rel = np.maximum(rel.sum(axis=1), 1)
        seen = hist.R[users].tocoo()
        hits = {}
        for name, model in models.items():
            s = np.asarray(model.score(users, hist), dtype=np.float32)
            s[seen.row, seen.col] = -np.inf
            top = top_items(s, 100)
            for k in (50, 100):
                h = np.zeros_like(rel)
                np.put_along_axis(h, top[:, :k], True, axis=1)
                hits[(name, k)] = h
        rows = [{"Источник": n, "Recall@50": float(((hits[(n, 50)] & rel).sum(1) / n_rel).mean()),
                 "Recall@100": float(((hits[(n, 100)] & rel).sum(1) / n_rel).mean())} for n in models]
        for v in (self.variants[0], self.variants[1], self.variants[-1]):
            for k in (50,):
                union = np.zeros_like(rel)
                for n in v.models:
                    union |= hits[(n, k)]
                rows.append({"Источник": f"Пул {v.name.split(':')[0]} (топ-50 каждого)", "Recall@50": float(((union & rel).sum(1) / n_rel).mean()),
                             "Recall@100": np.nan, "Размер пула": float(union.sum(1).mean())})
        return pd.DataFrame(rows)

    def fit_sources(self) -> "ImprovementStudy":
        sp_, cfg = self.sp, self.cfg
        eligible = np.flatnonzero(sp_.Y_val.getnnz(axis=1) > 0)
        self.perm = self.rng.permutation(eligible)
        src_users = np.sort(self.perm[:self.s.source_tune_users])
        with self.timer("9.2. H1: EASE и RP3beta (подбор λ и β на validation)"):
            rows, best = [], {}
            for lam in self.s.ease_lambdas:
                m = EASE(lam).fit(sp_.hist_fit)
                r = self._single_recall(m, src_users, sp_.hist_fit, sp_.Y_val)
                rows.append({"Модель": "EASE", "Параметр": f"λ = {lam:g}", "val Recall@50": r})
                if r > best.get("EASE", (-1, None))[0]:
                    best["EASE"] = (r, m, lam)
                del m
                import gc; gc.collect()
            for beta in self.s.rp3_betas:
                m = RP3beta(beta).fit(sp_.hist_fit)
                r = self._single_recall(m, src_users, sp_.hist_fit, sp_.Y_val)
                rows.append({"Модель": "RP3beta", "Параметр": f"β = {beta:g}", "val Recall@50": r})
                if r > best.get("RP3beta", (-1, None))[0]:
                    best["RP3beta"] = (r, m, beta)
            self.results["source_tuning"] = pd.DataFrame(rows)
            self.ease_lambda, self.rp3_beta = best["EASE"][2], best["RP3beta"][2]
            self.new_fit = {"EASE": best["EASE"][1], "RP3beta": best["RP3beta"][1]}
            print(self.results["source_tuning"].round(4).to_string())
        with self.timer("9.3. H5: обучение SASRec (cross-entropy, CPU)"):
            self.sasrec = SASRec(sp_.idx.n_items, epochs=self.s.sasrec_epochs, seed=cfg.seed).fit(
                sp_.hist_fit, src_users, sp_.Y_val)
            self.new_fit["SASRec"] = self.sasrec
            self.results["sasrec_history"] = pd.DataFrame(self.sasrec.history)
        with self.timer("9.4. Доля релевантных книг по источникам и в пуле (validation)"):
            models = {**self.pipe.gen_fit.models, **self.new_fit}
            self.results["source_recall_val"] = self._recall_table(models, src_users, sp_.hist_fit, sp_.Y_val)
            print(self.results["source_recall_val"].round(4).to_string())
        return self

    @staticmethod
    def _single_recall(model, users, hist, Y, k=50):
        s = np.asarray(model.score(users, hist), dtype=np.float32)
        seen = hist.R[users].tocoo()
        s[seen.row, seen.col] = -np.inf
        top = top_items(s, k)
        rel = Y[users].toarray() > 0
        return float((np.take_along_axis(rel, top, axis=1).sum(1) / np.maximum(rel.sum(1), 1)).mean())

    # ------------------------------------------------------------------ данные
    def _collect(self, gen, users, hist, Y, tag):
        parts, nl = [], []
        for s in range(0, len(users), 1000):
            d, n = flatten(gen.generate(users[s:s + 1000], hist), Y, tag)
            parts.append(d)
            nl.append(n)
        return RankerData.concat(parts), np.concatenate(nl)

    def build_data(self) -> "ImprovementStudy":
        sp_, cfg, s = self.sp, self.cfg, self.s
        with self.timer("9.5. Расширенный пул кандидатов и признаки для validation"):
            self.gen_fit = ExtGenerator({**self.pipe.gen_fit.models, **self.new_fit}, self.pipe.gen_fit.builder,
                                        self.series, s.candidates_per_model, seed=cfg.seed)
            rest = self.perm[s.source_tune_users:]
            tune = np.sort(rest[:s.tune_users])
            rank = np.sort(rest[s.tune_users:s.tune_users + s.rank_users])
            half_a = set(self.rng.choice(tune, len(tune) // 2, replace=False).tolist())
            parts, nls, halves = [], [], []
            d, n = self._collect(self.gen_fit, tune, sp_.hist_fit, sp_.Y_val, "full")
            parts.append(d); nls.append(n); halves.append([u in half_a for u in tune])
            cold = tune[: s.cold_tune_users]
            for k in cfg.cold_histories:
                d, n = self._collect(self.gen_fit, cold, self.pipe._truncated_hist(sp_.train_fit, cold, k), sp_.Y_val, f"n={k}")
                parts.append(d); nls.append(n); halves.append([u in half_a for u in cold])
            self.tune_data = RankerData.concat(parts)
            self.tune_nlikes = np.concatenate(nls)
            self.tune_half_a = np.concatenate(halves)
            parts, nls = [], []
            d, n = self._collect(self.gen_fit, rank, sp_.hist_fit, sp_.Y_val, "full")
            parts.append(d); nls.append(n)
            trunc = rank[: int(len(rank) * s.truncated_share)]
            for j, k in enumerate(cfg.cold_histories):
                us = trunc[j::len(cfg.cold_histories)]
                d, n = self._collect(self.gen_fit, us, self.pipe._truncated_hist(sp_.train_fit, us, k), sp_.Y_val, f"n={k}")
                parts.append(d); nls.append(n)
            self.rank_data = RankerData.concat(parts)
            pool = np.diff(self.tune_data.offsets)[self.tune_data.tag == "full"]
            print(f"  строк для обучения: {len(self.rank_data.labels):,}; средний пул (8 источников): {pool.mean():.0f}")
        return self

    # ------------------------------------------------------------------ обучение ранжировщиков (H1, H2, H3, H5)
    def train_rankers(self) -> "ImprovementStudy":
        cfg, sp_ = self.cfg, self.sp
        A = np.flatnonzero(self.tune_half_a)
        B = np.flatnonzero(~self.tune_half_a)
        self.users_a, self.users_b = A, B
        rows = []
        self.rankers = {}
        tag_b = self.tune_data.tag[B]

        def report(name, variant, scores_b, extra=None):
            nd = ranked_ndcg(variant_b, scores_b)
            rows.append({"Вариант": name, "Ранжировщик": extra or "Wide & Deep",
                         "Средний пул": float(np.diff(variant_b.offsets)[tag_b == "full"].mean()),
                         "Доля релевантных в пуле": float((np.bincount(row_user(variant_b), weights=variant_b.labels,
                                                                       minlength=variant_b.n_users) /
                                                           np.maximum(variant_b.n_relevant, 1))[tag_b == "full"].mean()),
                         "nDCG@10 (полные истории)": float(nd[tag_b == "full"].mean()),
                         "nDCG@10 (укороченные)": float(nd[tag_b != "full"].mean())})
            return nd

        self.val_ndcg = {}
        for variant in self.variants:
            with self.timer(f"9.6. Wide & Deep: {variant.name}"):
                tr = variant_data(self.rank_data, self.gen_fit, variant)
                full_v = variant_data(self.tune_data, self.gen_fit, variant)
                va, variant_b = full_v.subset(A), full_v.subset(B)
                wd = WideDeepRanker(self.pipe.gen_fit.builder, sp_.idx.n_items, tr.features.shape[1], cfg,
                                    name=variant.name).fit(tr, va, verbose=False)
                self.rankers[variant.name] = (variant, wd)
                self.val_ndcg[variant.name] = report(variant.name, variant, wd.predict_data(variant_b))
                print(f"  {rows[-1]}")
                if variant is self.variants[0] or variant is self.variants[-1]:
                    lm = LambdaMARTRanker(self.s.lgbm_trees, cfg.seed, f"LambdaMART ({variant.name[:2]})").fit(tr, va)
                    self.rankers[lm.name] = (variant, lm)
                    self.val_ndcg[lm.name] = report(variant.name, variant, lm.predict_data(variant_b), "LambdaMART (H3)")
                    print(f"  {rows[-1]}")
                if variant is self.variants[-1]:
                    gb = GBDTRanker(cfg, name="GBDT pointwise (C3)").fit(tr)
                    self.rankers[gb.name] = (variant, gb)
                    self.val_ndcg[gb.name] = report(variant.name, variant, gb.predict_data(variant_b), "GBDT pointwise")
                    self.c3_a, self.c3_b = va, variant_b
                del tr, full_v
                import gc; gc.collect()
        self.results["ablation_val"] = pd.DataFrame(rows)
        return self

    # ------------------------------------------------------------------ H4: честный выбор смеси
    def _hybrid_scores(self, data: RankerData, nlikes: np.ndarray) -> np.ndarray:
        M = len(self.variants[-1].models)
        Z = data.features[:, len(BASE_NAMES) + M: len(BASE_NAMES) + 2 * M]
        seg = segment_of(nlikes, self.cfg.segment_new_max, self.cfg.segment_active_min)
        W = np.stack([self.hybrid_w[s] for s in seg])[row_user(data)]
        return (Z * W).sum(axis=1)

    def _component_scores(self, data: RankerData, nlikes: np.ndarray) -> dict:
        c3 = self.variants[-1].name
        return {"W&D": standardize_flat(data, self.rankers[c3][1].predict_data(data)),
                "LambdaMART": standardize_flat(data, self.rankers["LambdaMART (C3)"][1].predict_data(data)),
                "гибрид": standardize_flat(data, self._hybrid_scores(data, nlikes))}

    def final_scores(self, data: RankerData, nlikes: np.ndarray) -> np.ndarray:
        comp = self._component_scores(data, nlikes)
        seg = segment_of(nlikes, self.cfg.segment_new_max, self.cfg.segment_active_min)[row_user(data)]
        out = comp["W&D"].copy()
        for s, (other, w) in self.blend.items():
            if other is not None:
                m = seg == s
                out[m] = (1 - w) * comp["W&D"][m] + w * comp[other][m]
        return out

    def select_final(self) -> "ImprovementStudy":
        cfg = self.cfg
        with self.timer("9.7. H4: веса гибрида на половине A, выбор смеси, MMR и правила серий на половине B"):
            M = len(self.variants[-1].models)
            nl_a, nl_b = self.tune_nlikes[self.users_a], self.tune_nlikes[self.users_b]
            seg_a = segment_of(nl_a, cfg.segment_new_max, cfg.segment_active_min)
            seg_b = segment_of(nl_b, cfg.segment_new_max, cfg.segment_active_min)
            Z = self.c3_a.features[:, len(BASE_NAMES) + M: len(BASE_NAMES) + 2 * M]
            w_glob, v_glob = tune_weights_flat(self.c3_a, Z, 120, cfg.seed)
            self.hybrid_w = {}
            for s in range(3):
                us = np.flatnonzero(seg_a == s)
                sub = self.c3_a.subset(us)
                Zs = sub.features[:, len(BASE_NAMES) + M: len(BASE_NAMES) + 2 * M]
                w, v = tune_weights_flat(sub, Zs, 120, cfg.seed + s)
                own = float(ranked_ndcg(sub, Zs @ w).mean())
                shared = float(ranked_ndcg(sub, Zs @ w_glob).mean())
                self.hybrid_w[s] = w if own >= shared * 1.01 else w_glob
            self.results["hybrid_weights_ext"] = pd.DataFrame(self.hybrid_w, index=self.variants[-1].models).T.rename(
                index=dict(enumerate(self.pipe.seg_labels)))
            comp = self._component_scores(self.c3_b, nl_b)
            base = ranked_ndcg(self.c3_b, comp["W&D"])
            rows, self.blend = [], {}
            grid = np.round(np.arange(0.1, 1.0, 0.1), 1)
            for s in range(3):
                users_s = seg_b == s
                best = (None, 0.0, float(base[users_s].mean()))
                for other in ("гибрид", "LambdaMART"):
                    for w in grid:
                        nd = ranked_ndcg(self.c3_b, (1 - w) * comp["W&D"] + w * comp[other])
                        if nd[users_s].mean() > best[2]:
                            best = (other, float(w), float(nd[users_s].mean()), nd)
                if best[0] is not None:
                    d, lo, hi = paired_bootstrap(best[3][users_s], base[users_s], seed=cfg.seed)
                    accept = lo > 0
                else:
                    d = lo = hi = 0.0
                    accept = False
                self.blend[s] = (best[0], best[1]) if accept else (None, 0.0)
                rows.append({"Сегмент": self.pipe.seg_labels[s], "W&D (C3)": float(base[users_s].mean()),
                             "Лучшая смесь": f"W&D + {best[0]}, w = {best[1]}" if best[0] else "—",
                             "nDCG@10 смеси": best[2], "Δ": d, "95% ДИ, низ": lo, "95% ДИ, верх": hi,
                             "Принято": "смесь" if accept else "только W&D"})
            self.results["blend_choice"] = pd.DataFrame(rows)
            print(self.results["blend_choice"].round(4).to_string())
            # MMR и правило серий — на пользователях половины B с полной историей
            full_b = np.flatnonzero(self.tune_data.tag[self.users_b] == "full")
            data = self.c3_b.subset(full_b)
            scores = self.final_scores(data, nl_b[full_b])
            cand, S, valid, L = padded(data, scores)
            rows = []
            for lam in cfg.mmr_lambdas:
                lists = mmr_rerank(cand, S, valid, self.pipe.item_emb, lam, cfg.main_k, cfg.mmr_depth)
                for rule in (False, True):
                    l2 = self.series.fix_order(lists) if rule else lists
                    rows.append({"λ": lam, "Правило серий": rule,
                                 "nDCG@10": float(lists_ndcg(l2, cand, L, data.n_relevant).mean()),
                                 **self.pipe.beyond(l2, cfg.main_k)})
            mmr = pd.DataFrame(rows)
            base_nd = mmr[(mmr["λ"] == 1.0) & ~mmr["Правило серий"]]["nDCG@10"].iloc[0]
            ok = mmr[(mmr["nDCG@10"] >= base_nd * (1 - cfg.mmr_max_ndcg_loss)) & ~mmr["Правило серий"]]
            self.mmr_lambda = float(ok["λ"].min())
            at = mmr[mmr["λ"] == self.mmr_lambda].set_index("Правило серий")["nDCG@10"]
            self.use_rule = bool(at[True] >= at[False] * 0.999)
            self.results["mmr_rule_val"] = mmr
            print(f"  λ = {self.mmr_lambda}; правило серий: {'включено' if self.use_rule else 'не принято'}")
        return self

    # ------------------------------------------------------------------ переобучение на train_full и тест
    def refit_full(self) -> "ImprovementStudy":
        sp_ = self.sp
        with self.timer("9.8. Переобучение EASE и RP3beta на train_full"):
            del self.new_fit, self.gen_fit, self.tune_data, self.rank_data
            new_full = {"EASE": EASE(self.ease_lambda).fit(sp_.hist_full), "RP3beta": RP3beta(self.rp3_beta).fit(sp_.hist_full),
                        "SASRec": self.sasrec}
            self.gen_full = ExtGenerator({**self.pipe.gen_full.models, **new_full}, self.pipe.gen_full.builder,
                                         self.series, self.s.candidates_per_model, seed=self.cfg.seed)
            for _, model in self.rankers.values():
                if isinstance(model, WideDeepRanker):
                    model.builder = self.pipe.gen_full.builder
        return self

    def systems(self):
        names = {v.name: v for v in self.variants}
        out = {f"W&D {n.split(':')[0]}": (v, self.rankers[n][1]) for n, v in names.items()}
        for key in ("LambdaMART (C0)", "LambdaMART (C3)", "GBDT pointwise (C3)"):
            out[key] = self.rankers[key]
        return out

    def _pass(self, users, hist, Y, label) -> dict:
        k = max(self.cfg.k_values)
        lists: dict[str, list] = {}
        pool = {v.name: [] for v in (self.variants[0], self.variants[-1])}
        for s in range(0, len(users), 1000):
            eb = self.gen_full.generate(users[s:s + 1000], hist)
            master, nl = flatten(eb, None, "test")
            seen = hist.R[eb.users].toarray() > 0
            rel = Y[eb.users].toarray() > 0
            for v in pool:
                var = next(x for x in self.variants if x.name == v)
                cols = [self.gen_full.names.index(n) for n in var.models]
                in_pool = eb.flags[:, :, cols].any(axis=2)
                pool[v].append((np.take_along_axis(rel, np.maximum(eb.cand, 0), axis=1) & in_pool).sum(1) / np.maximum(rel.sum(1), 1))
            for name in ("EASE", "RP3beta", "SASRec"):
                lists.setdefault(name, []).append(eb.single[name])
            cache = {}
            for name, (variant, model) in self.systems().items():
                if variant.name not in cache:
                    cache[variant.name] = variant_data(master, self.gen_full, variant)
                d = cache[variant.name]
                cand, S, valid, _ = padded(d, model.predict_data(d))
                lists.setdefault(name, []).append(np.take_along_axis(cand, top_items(S, k), axis=1))
            d = cache[self.variants[-1].name]
            cand, S, valid, _ = padded(d, self.final_scores(d, nl))
            plain = np.take_along_axis(cand, top_items(S, k), axis=1)
            mmr = mmr_rerank(cand, S, valid, self.pipe.item_emb, self.mmr_lambda, k, self.cfg.mmr_depth)
            lists.setdefault("Улучшенная система (без MMR)", []).append(plain)
            lists.setdefault("Улучшенная система (MMR, без правила серий)", []).append(mmr)
            lists.setdefault("Улучшенная система (итог)", []).append(self.series.fix_order(mmr) if self.use_rule else mmr)
            for name in lists:
                l = lists[name][-1]
                assert not np.any(np.take_along_axis(seen, np.maximum(l, 0), axis=1) & (l >= 0)), name
        self.pool_recall = {v: float(np.concatenate(x).mean()) for v, x in pool.items()}
        print(f"  [{label}] доля релевантных книг в пуле: " + ", ".join(f"{k.split(':')[0]} {v:.1%}" for k, v in self.pool_recall.items()))
        return {n: evaluate_lists(n, np.vstack(v), users, Y, self.cfg.k_values) for n, v in lists.items()}

    def evaluate_test(self) -> "ImprovementStudy":
        cfg = self.cfg
        with self.timer("9.9. Тест: все варианты улучшений"):
            users = self.pipe.test_users
            if self.s.max_test_users:
                users = np.sort(self.rng.choice(users, self.s.max_test_users, replace=False))
            self.test_users = users
            self.test_evals = self._pass(users, self.sp.hist_full, self.sp.Y_test, "тест")
            self.results["pool_recall_test_ext"] = pd.DataFrame({"Доля релевантных книг теста в пуле": self.pool_recall})
            self.results["metrics_test_ext"] = pd.DataFrame(
                [{"Модель": n, **e.mean(), **self.pipe.beyond(e.lists, cfg.main_k)} for n, e in self.test_evals.items()]
            ).set_index("Модель")
            old = self.pipe.test_evals
            same = np.array_equal(old["Итоговая система (с MMR)"].users, users)
            rows = []
            if same:
                for ref in ("Итоговая система (с MMR)", "Wide & Deep", "Item-CF (explicit)"):
                    for new in ("Улучшенная система (итог)", "W&D C3", "LambdaMART (C3)"):
                        d, lo, hi = paired_bootstrap(self.test_evals[new].per_user["nDCG@10"],
                                                     old[ref].per_user["nDCG@10"], seed=cfg.seed)
                        rows.append({"Новая система": new, "Сравнение с (проект)": ref, "Δ nDCG@10": d,
                                     "95% ДИ, низ": lo, "95% ДИ, верх": hi,
                                     "Относительно, %": 100 * d / old[ref].mean()["nDCG@10"]})
            self.results["bootstrap_ext"] = pd.DataFrame(rows)
            seg = self.pipe.user_segments
            self.results["segments_test_ext"] = pd.DataFrame({
                n: pd.Series(e.per_user["nDCG@10"]).groupby(seg.reindex(e.users).to_numpy()).mean()
                for n, e in self.test_evals.items()}).T[self.pipe.seg_labels]
        return self

    def cold_start(self) -> "ImprovementStudy":
        cfg = self.cfg
        with self.timer("9.10. Новые пользователи: улучшенная и прежняя системы на одних пользователях"):
            users = np.sort(self.rng.choice(self.pipe.test_users, min(self.s.cold_eval_users, len(self.pipe.test_users)),
                                            replace=False))
            rows = []
            keep_new = ["W&D C0", "W&D C3", "LambdaMART (C3)", "Улучшенная система (итог)", "SASRec", "EASE"]
            keep_old = ["Popularity", "Wide & Deep", "Итоговая система (с MMR)"]
            for n in cfg.cold_histories:
                hist_n = self.pipe._truncated_hist(self.sp.train_full, users, n)
                new = self._pass(users, hist_n, self.sp.Y_test, f"n={n}")
                old = self.pipe._run_pass(users, hist_n, self.sp.Y_test, f"n={n}, проект")
                for name in keep_old:
                    rows.append({"Лайков в истории": n, "Модель": f"Проект: {name}", **old[name].mean()})
                for name in keep_new:
                    rows.append({"Лайков в истории": n, "Модель": f"Улучшения: {name}", **new[name].mean()})
            self.results["cold_start_ext"] = pd.DataFrame(rows)
        return self

    def save(self) -> "ImprovementStudy":
        from pathlib import Path
        out = Path(self.cfg.results_dir) / "improvements"
        out.mkdir(parents=True, exist_ok=True)
        for name, frame in self.results.items():
            frame.to_csv(out / f"{name}.csv", encoding="utf-8-sig")
        return self

    def run(self) -> "ImprovementStudy":
        return (self.check_order().fit_sources().build_data().train_rankers().select_final()
                .refit_full().evaluate_test().cold_start().save())
