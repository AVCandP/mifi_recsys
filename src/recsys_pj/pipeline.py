"""Сквозной пайплайн: загрузка → предобработка → обучение → генерация рекомендаций → оценка качества.

Запуск из командной строки (повторный запуск на новых данных — укажите другую папку с CSV того же формата):

    python -m recsys_pj --data-dir data --out-dir .            # полный прогон
    python -m recsys_pj --data-dir data --out-dir . --quick    # быстрый прогон для проверки
    python -m recsys_pj --data-dir data --out-dir . --with-improvements   # + раздел 9 (улучшения)
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .config import Config
from .data import History, RawData, Splits, load_data, make_splits, truncate_history
from .features import CandidateBatch, CandidateGenerator, FeatureBuilder, Topics, build_topics, content_embeddings
from .hybrid import (Router, TuningSet, WeightedHybrid, lists_from_scores, mmr_rerank, ndcg_for_scores,
                     segment_labels, segment_of, tune_weights)
from .metrics import BeyondAccuracy, Evaluation, evaluate_lists, metrics_at_k, paired_bootstrap, self_check
from .models import build_tfidf, fit_base_models
from .neural import GBDTRanker, RankerData, WideDeepRanker, permutation_importance, ranked_ndcg


@dataclass
class LabeledBatch:
    batch: CandidateBatch
    labels: np.ndarray      # U × P: книга из пула релевантна в отложенном периоде
    n_relevant: np.ndarray  # U
    tag: str


def label_batch(batch: CandidateBatch, Y: sp.csr_matrix, tag: str) -> LabeledBatch:
    rel = Y[batch.users].toarray() > 0
    labels = np.take_along_axis(rel, np.maximum(batch.cand, 0), axis=1) & batch.valid
    return LabeledBatch(batch, labels, rel.sum(axis=1), tag)


def batch_ndcg(lb: LabeledBatch, scores: np.ndarray, k: int = 10) -> np.ndarray:
    order = np.argsort(-np.where(lb.batch.valid, scores, -np.inf), axis=1, kind="stable")[:, :k]
    hits = np.take_along_axis(lb.labels, order, axis=1)
    return metrics_at_k(hits, lb.n_relevant, k)[f"nDCG@{k}"]


class Timer:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def __call__(self, stage: str):
        timer = self

        class _Ctx:
            def __enter__(self):
                self.t = time.perf_counter()
                print(f"▶ {stage}")
                return self

            def __exit__(self, *exc):
                seconds = time.perf_counter() - self.t
                timer.rows.append({"Этап": stage, "Время, с": round(seconds, 1)})
                print(f"  готово за {seconds:.0f} с")
        return _Ctx()


class Pipeline:
    """Все этапы системы. Каждый метод — один шаг; run() выполняет их по порядку."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        cfg.ensure_dirs()
        self.rng = np.random.default_rng(cfg.seed)
        self.timer = Timer()
        self.results: dict[str, pd.DataFrame] = {}

    # ---------------------------------------------------------------- 1. Загрузка
    def load(self) -> "Pipeline":
        with self.timer("1. Загрузка данных"):
            self.raw: RawData = load_data(self.cfg)
            print(f"  оценок: {len(self.raw.ratings):,}, книг: {len(self.raw.books):,}")
        return self

    # ---------------------------------------------------------------- 2. Предобработка
    def preprocess(self) -> "Pipeline":
        cfg = self.cfg
        with self.timer("2. Предобработка: разбиение, TF-IDF, темы"):
            self_check()
            self.splits: Splits = make_splits(self.raw, cfg)
            sp_ = self.splits
            assert len(sp_.train_full.merge(sp_.test, on=["user_id", "book_id"])) == 0
            self.tfidf = build_tfidf(self.raw.books, self.raw.book_tags, sp_.idx, cfg.top_tags_per_book)
            self.topics: Topics = build_topics(self.raw.book_tags, sp_.idx, cfg.n_topics, cfg.top_tags_per_book, cfg.seed)
            self.item_emb = content_embeddings(self.tfidf, 64, cfg.seed)
            self.beyond = BeyondAccuracy(sp_.hist_full.B.getnnz(axis=0), self.item_emb)
            self.seg_labels = segment_labels(cfg.segment_new_max, cfg.segment_active_min)
            self.results["split"] = pd.DataFrame({
                "Оценок": [len(sp_.train_fit), len(sp_.val), len(sp_.test)],
                "Доля": [len(f) / len(self.raw.ratings) for f in (sp_.train_fit, sp_.val, sp_.test)],
                "Доля релевантных": [(f["rating"] >= cfg.relevance_threshold).mean()
                                     for f in (sp_.train_fit, sp_.val, sp_.test)],
            }, index=["train_fit", "validation", "test"])
        return self

    # ---------------------------------------------------------------- 3. Обучение
    def _generator(self, hist: History) -> CandidateGenerator:
        models = fit_base_models(hist, self.tfidf, self.cfg)
        builder = FeatureBuilder(hist, self.raw.books, self.splits.idx, self.topics, self.cfg.relevance_threshold)
        return CandidateGenerator(models, builder, self.cfg.candidates_per_model, max(self.cfg.k_values),
                                  self.cfg.wd_history_len, self.cfg.seed)

    def _collect(self, gen: CandidateGenerator, users: np.ndarray, hist: History, Y: sp.csr_matrix, tag: str,
                 batch_size: int = 1000) -> list[LabeledBatch]:
        return [label_batch(gen.generate(users[s:s + batch_size], hist), Y, tag)
                for s in range(0, len(users), batch_size)]

    def _truncated_hist(self, frame: pd.DataFrame, users: np.ndarray, n: int) -> History:
        ids = self.splits.idx.user_ids[users]
        part = frame[frame["user_id"].isin(ids)]
        return History.from_frame(truncate_history(part, ids, n, self.cfg.relevance_threshold),
                                  self.splits.idx, self.cfg.relevance_threshold)

    def fit_stage(self) -> "Pipeline":
        """Обучение на train_fit, подбор весов, ранжировщиков и порогов на validation."""
        return (self.fit_base().build_tuning().tune_hybrid().build_ranker_data()
                .train_neural().train_baselines().tune_blend())

    def fit_base(self) -> "Pipeline":
        """Шаг 3.1: базовые модели из ДЗ на train_fit."""
        cfg, sp_ = self.cfg, self.splits
        with self.timer("3.1. Базовые модели на train_fit"):
            self.gen_fit = self._generator(sp_.hist_fit)
        return self

    def build_tuning(self) -> "Pipeline":
        """Шаг 3.2: пул кандидатов и признаки для пользователей подбора (полные и укороченные истории)."""
        cfg, sp_ = self.cfg, self.splits
        with self.timer("3.2. Пул кандидатов и признаки для validation"):
            eligible = np.flatnonzero(sp_.Y_val.getnnz(axis=1) > 0)
            perm = self.rng.permutation(eligible)
            self.tune_users = np.sort(perm[:cfg.tune_users])
            self.rank_users = np.sort(perm[cfg.tune_users:cfg.tune_users + cfg.ranker_train_users])
            self.tune_batches = self._collect(self.gen_fit, self.tune_users, sp_.hist_fit, sp_.Y_val, "full")
            cold_users = self.tune_users[: cfg.tune_users // 2]
            for n in cfg.cold_histories:
                hist_n = self._truncated_hist(sp_.train_fit, cold_users, n)
                self.tune_batches += self._collect(self.gen_fit, cold_users, hist_n, sp_.Y_val, f"n={n}")
            self.tuning = TuningSet.concat([
                TuningSet(lb.batch.z, lb.batch.valid, lb.labels, lb.n_relevant,
                          segment_of(lb.batch.n_likes, cfg.segment_new_max, cfg.segment_active_min),
                          np.array([lb.tag] * len(lb.batch.users))) for lb in self.tune_batches])
            natural = self.tuning.subset(self.tuning.tag == "full")
            pool_recall = (natural.labels.sum(axis=1) / natural.n_relevant).mean()
            print(f"  пользователей для подбора: {len(self.tune_users):,}; доля релевантных книг validation "
                  f"внутри пула: {pool_recall:.1%}; средний размер пула: {natural.valid.sum(axis=1).mean():.0f}")
        return self

    def tune_hybrid(self) -> "Pipeline":
        """Шаг 3.3: подбор весов взвешенного гибрида — общих и по сегментам пользователей."""
        cfg, sp_ = self.cfg, self.splits
        with self.timer("3.3. Подбор весов гибрида (общих и по сегментам)"):
            names = self.gen_fit.model_names
            k = cfg.main_k
            w_global, search_global = tune_weights(self.tuning.subset(self.tuning.tag == "full"),
                                                   cfg.weight_samples, k, cfg.seed)
            weights, searches, gains = {}, {"Все (общие веса)": search_global}, []
            for s in range(3):
                ts = self.tuning.subset(self.tuning.segment == s)
                w_seg, searches[self.seg_labels[s]] = tune_weights(ts, cfg.weight_samples, k, cfg.seed + s)
                own = float(ndcg_for_scores(ts, ts.z @ w_seg, k).mean())
                shared = float(ndcg_for_scores(ts, ts.z @ w_global, k).mean())
                use_own = own >= shared * (1 + cfg.segment_min_gain)
                weights[s] = w_seg if use_own else w_global
                gains.append({"Сегмент": self.seg_labels[s], "nDCG@10 общих весов": shared,
                              "nDCG@10 весов сегмента": own, "Используются": "свои" if use_own else "общие"})
            self.results["segment_weight_gain"] = pd.DataFrame(gains)
            print(self.results["segment_weight_gain"].round(4).to_string())
            self.hybrid_global = WeightedHybrid(names, {s: w_global for s in range(3)}, cfg.segment_new_max,
                                                cfg.segment_active_min, "Гибрид (общие веса)")
            self.hybrid_segment = WeightedHybrid(names, weights, cfg.segment_new_max, cfg.segment_active_min,
                                                 "Гибрид (веса по сегментам)")
            table = self.hybrid_segment.weights_table(self.seg_labels)
            table.loc["Все (общие веса)"] = w_global
            self.results["hybrid_weights"] = table
            self.results["weight_search"] = pd.concat(
                [df.assign(Сегмент=name) for name, df in searches.items()], ignore_index=True
            ).rename(columns={f"w{m}": n for m, n in enumerate(names)})
            print(table.round(3).to_string())
        return self

    def build_ranker_data(self) -> "Pipeline":
        """Шаг 3.4: обучающая выборка ранжировщиков (метки — лайки в периоде validation)."""
        cfg, sp_ = self.cfg, self.splits
        with self.timer("3.4. Данные для ранжировщиков"):
            rank_batches = self._collect(self.gen_fit, self.rank_users, sp_.hist_fit, sp_.Y_val, "full")
            self.train_data = RankerData.concat([RankerData.from_batches([(lb.batch, lb.labels, lb.n_relevant, lb.tag)])
                                                 for lb in rank_batches])
            del rank_batches
            n_trunc = int(len(self.rank_users) * cfg.ranker_truncated_share)
            trunc_users = self.rank_users[:n_trunc]
            parts = [self.train_data]
            for j, n in enumerate(cfg.cold_histories):
                users = trunc_users[j::len(cfg.cold_histories)]
                hist_n = self._truncated_hist(sp_.train_fit, users, n)
                parts += [RankerData.from_batches([(lb.batch, lb.labels, lb.n_relevant, lb.tag)])
                          for lb in self._collect(self.gen_fit, users, hist_n, sp_.Y_val, f"n={n}")]
            self.train_data = RankerData.concat(parts)
            self.valid_data = RankerData.concat([RankerData.from_batches([(lb.batch, lb.labels, lb.n_relevant, lb.tag)])
                                                 for lb in self.tune_batches])
            print(f"  строк для обучения: {len(self.train_data.labels):,} ({self.train_data.n_users:,} историй), "
                  f"доля положительных: {self.train_data.labels.mean():.2%}")

        return self

    def train_neural(self) -> "Pipeline":
        """Шаг 3.5: обучение нейросетевого ранжировщика Wide & Deep."""
        cfg, sp_ = self.cfg, self.splits
        names_f = self.gen_fit.feature_names
        with self.timer("3.5. Обучение Wide & Deep"):
            self.wd = WideDeepRanker(self.gen_fit.builder, sp_.idx.n_items, len(names_f), cfg).fit(
                self.train_data, self.valid_data)
            self.results["wd_history"] = pd.DataFrame(self.wd.history)
        return self

    def train_baselines(self) -> "Pipeline":
        """Шаг 3.6: GBDT-ранжировщик, абляции Wide & Deep и важность групп признаков."""
        cfg, sp_ = self.cfg, self.splits
        names_f = self.gen_fit.feature_names
        with self.timer("3.6. Классический ранжировщик и абляции"):
            self.gbdt = GBDTRanker(cfg).fit(self.train_data)
            base_mask = np.array([n.startswith(("score::", "z::")) for n in names_f])
            gbdt_base = GBDTRanker(cfg, name="GBDT (только скоры моделей)", feature_mask=base_mask).fit(self.train_data)
            ablation_cfg = Config(**{**cfg.__dict__, "wd_epochs": max(3, cfg.wd_epochs // 2)})
            wide_only = WideDeepRanker(self.gen_fit.builder, sp_.idx.n_items, len(names_f), ablation_cfg,
                                       use_deep=False, name="Только Wide").fit(self.train_data, self.valid_data, False)
            deep_only = WideDeepRanker(self.gen_fit.builder, sp_.idx.n_items, len(names_f), ablation_cfg,
                                       use_wide=False, name="Только Deep").fit(self.train_data, self.valid_data, False)
            vd = self.valid_data
            seg_v = segment_of(np.array([len(h) for h in vd.hist_items]), cfg.segment_new_max, cfg.segment_active_min)
            rows = []
            variants = {
                "Wide & Deep": self.wd.predict_data(vd),
                "Только Wide (линейная часть + пересечения)": wide_only.predict_data(vd),
                "Только Deep (эмбеддинги + MLP)": deep_only.predict_data(vd),
                "GBDT (все признаки)": self.gbdt.predict_data(vd),
                "GBDT (только скоры базовых моделей)": gbdt_base.predict_data(vd),
            }
            for name, scores in variants.items():
                nd = ranked_ndcg(vd, scores)
                rows.append({"Вариант": name, "nDCG@10 (полные истории)": nd[vd.tag == "full"].mean(),
                             "nDCG@10 (укороченные истории)": nd[vd.tag != "full"].mean()})
            self.results["ranker_ablation"] = pd.DataFrame(rows)
            print(self.results["ranker_ablation"].round(4).to_string())
            groups = {
                "Скоры базовых моделей": [n for n in names_f if n.startswith(("score::", "z::"))],
                "Признаки книг": [n for n in names_f if n.startswith("item_")],
                "Признаки пользователя": [n for n in names_f if n.startswith("user_")],
                "Тематическое сходство с историей": ["topic_affinity"],
                "Книги автора в истории": ["author_likes"],
                "Отклонения от профиля (оценка, популярность)": ["mean_diff", "pop_diff"],
                "Число моделей-источников": ["n_sources"],
            }
            full_users = np.flatnonzero(vd.tag == "full")
            self.results["feature_importance"] = permutation_importance(self.wd, vd.subset(full_users), names_f, groups)
        return self

    def tune_blend(self) -> "Pipeline":
        """Шаг 3.7: доля нейросети в итоговой смеси по сегментам и порог MMR."""
        cfg, sp_ = self.cfg, self.splits
        with self.timer("3.7. Смесь Wide & Deep с гибридом по сегментам и порог MMR"):
            from .hybrid import _standardize
            alphas = np.round(np.linspace(0, 1, 11), 2)
            seg_all, per_alpha, gbdt_nd = [], {a: [] for a in alphas}, []
            for lb in self.tune_batches:
                seg_all.append(segment_of(lb.batch.n_likes, cfg.segment_new_max, cfg.segment_active_min))
                z_wd = _standardize(self.wd.score(lb.batch), lb.batch.valid)
                z_hy = _standardize(self.hybrid_segment.score(lb.batch), lb.batch.valid)
                for a in alphas:
                    per_alpha[a].append(batch_ndcg(lb, a * z_wd + (1 - a) * z_hy))
                gbdt_nd.append(batch_ndcg(lb, self.gbdt.score(lb.batch)))
            seg_all = np.concatenate(seg_all)
            alpha_table = pd.DataFrame({a: pd.Series(np.concatenate(v)).groupby(seg_all).mean()
                                        for a, v in per_alpha.items()})
            alpha_table["GBDT (для сравнения)"] = pd.Series(np.concatenate(gbdt_nd)).groupby(seg_all).mean()
            alpha_choice = {s: float(alpha_table.loc[s, list(alphas)].astype(float).idxmax()) for s in range(3)}
            alpha_table.index = [self.seg_labels[s] for s in alpha_table.index]
            alpha_table["Выбрано α"] = [alpha_choice[s] for s in range(3)]
            self.results["alpha_choice"] = alpha_table
            print(alpha_table.round(4).to_string())
            self.router = Router(self.wd, self.hybrid_segment, alpha_choice, cfg.segment_new_max,
                                 cfg.segment_active_min, self.item_emb, 1.0, cfg.mmr_depth)
            rows = []
            natural = [lb for lb in self.tune_batches if lb.tag == "full"]
            for lam in cfg.mmr_lambdas:
                lists = np.vstack([self.router.recommend(lb.batch, cfg.main_k, lam) for lb in natural])
                users = np.concatenate([lb.batch.users for lb in natural])
                ev = evaluate_lists(f"λ={lam}", lists, users, sp_.Y_val, (cfg.main_k,))
                rows.append({"λ": lam, **ev.mean(), **self.beyond(lists, cfg.main_k)})
            mmr = pd.DataFrame(rows)
            base = mmr.loc[mmr["λ"] == 1.0, f"nDCG@{cfg.main_k}"].iloc[0]
            ok = mmr[mmr[f"nDCG@{cfg.main_k}"] >= base * (1 - cfg.mmr_max_ndcg_loss)]
            self.router.mmr_lambda = float(ok["λ"].min())
            self.results["mmr_tradeoff_val"] = mmr
            print(f"  выбран λ = {self.router.mmr_lambda} (потеря nDCG@10 не более {cfg.mmr_max_ndcg_loss:.0%})")
        return self

    # ---------------------------------------------------------------- 4. Переобучение на полном train
    def refit_stage(self) -> "Pipeline":
        with self.timer("4. Переобучение базовых моделей на train_full"):
            self.gen_full = self._generator(self.splits.hist_full)
            # Ранжировщики и веса обучены на признаках train_fit и применяются к признакам train_full.
            for scorer in (self.wd,):
                scorer.builder = self.gen_full.builder
            del self.tune_batches
        return self

    # ---------------------------------------------------------------- 5. Рекомендации и оценка
    def systems(self) -> dict:
        k = max(self.cfg.k_values)
        return {
            self.hybrid_global.name: lambda b: self.hybrid_global.recommend(b, k),
            self.hybrid_segment.name: lambda b: self.hybrid_segment.recommend(b, k),
            "GBDT-ранжировщик": lambda b: lists_from_scores(b.cand, self.gbdt.score(b), b.valid, k),
            "Wide & Deep": lambda b: lists_from_scores(b.cand, self.wd.score(b), b.valid, k),
            "Итоговая система (без MMR)": lambda b: self.router.recommend(b, k, 1.0),
            "Итоговая система (с MMR)": lambda b: self.router.recommend(b, k),
        }

    def _run_pass(self, users: np.ndarray, hist: History, Y: sp.csr_matrix, label: str) -> dict[str, Evaluation]:
        cfg = self.cfg
        lists: dict[str, list] = {}
        pool_hits = []
        for s in range(0, len(users), 1000):
            batch = self.gen_full.generate(users[s:s + 1000], hist)
            seen = hist.R[batch.users]
            for name, single in batch.single_lists.items():
                lists.setdefault(name, []).append(single)
            for name, fn in self.systems().items():
                result = fn(batch)
                assert not np.any(np.take_along_axis(seen.toarray() > 0, np.maximum(result, 0), axis=1) & (result >= 0)), \
                    "в рекомендации попала прочитанная книга"
                lists.setdefault(name, []).append(result)
            rel = Y[batch.users].toarray() > 0
            pool_hits.append((np.take_along_axis(rel, np.maximum(batch.cand, 0), axis=1) & batch.valid).sum(axis=1)
                             / np.maximum(rel.sum(axis=1), 1))
        self.pool_recall = float(np.concatenate(pool_hits).mean())
        print(f"  [{label}] доля релевантных книг внутри пула кандидатов: {self.pool_recall:.1%}")
        return {name: evaluate_lists(name, np.vstack(v), users, Y, cfg.k_values) for name, v in lists.items()}

    def evaluate_stage(self) -> "Pipeline":
        cfg, sp_ = self.cfg, self.splits
        with self.timer("5.1. Рекомендации и метрики на тесте"):
            users = np.flatnonzero(sp_.Y_test.getnnz(axis=1) > 0)
            if cfg.max_test_users:
                users = np.sort(self.rng.choice(users, cfg.max_test_users, replace=False))
            self.test_users = users
            self.test_evals = self._run_pass(users, sp_.hist_full, sp_.Y_test, "тест")
            self.results["pool_recall_test"] = pd.DataFrame({"Доля релевантных в пуле": [self.pool_recall]})
            self.results["metrics_test"] = pd.DataFrame(
                [{"Модель": n, **e.mean(), **self.beyond(e.lists, cfg.main_k)} for n, e in self.test_evals.items()]
            ).set_index("Модель")
            n_likes = sp_.hist_full.n_likes
            seg = pd.Series(segment_of(n_likes, cfg.segment_new_max, cfg.segment_active_min))
            seg = seg.map(dict(enumerate(self.seg_labels)))
            self.user_segments = seg
            rows = {}
            for name, ev in self.test_evals.items():
                rows[name] = pd.Series(ev.per_user[f"nDCG@{cfg.main_k}"]).groupby(seg.reindex(ev.users).to_numpy()).mean()
            act_q = pd.qcut(pd.Series(n_likes[users]), 4, labels=["Q1 (меньше всего лайков)", "Q2", "Q3", "Q4 (больше всего)"])
            self.results["segments_test"] = pd.DataFrame(rows).T
            self.results["activity_quartiles_test"] = pd.DataFrame({
                name: pd.Series(ev.per_user[f"nDCG@{cfg.main_k}"]).groupby(act_q.to_numpy(), observed=True).mean()
                for name, ev in self.test_evals.items()}).T
            self.results["segment_sizes_test"] = seg.reindex(users).value_counts().to_frame("Пользователей")
            best_single = self.results["metrics_test"].loc[list(self.gen_full.model_names), f"nDCG@{cfg.main_k}"].idxmax()
            boots = []
            for name in self.systems():
                d, lo, hi = paired_bootstrap(self.test_evals[name].per_user[f"nDCG@{cfg.main_k}"],
                                             self.test_evals[best_single].per_user[f"nDCG@{cfg.main_k}"], seed=cfg.seed)
                boots.append({"Система": name, "Сравнение с": best_single, "Δ nDCG@10": d, "95% ДИ, низ": lo,
                              "95% ДИ, верх": hi, "Относительно, %": 100 * d / self.test_evals[best_single].mean()[f"nDCG@{cfg.main_k}"]})
            self.results["bootstrap_test"] = pd.DataFrame(boots)

        with self.timer("5.2. Новые пользователи (симуляция укороченной истории на тесте)"):
            eligible = np.flatnonzero(sp_.Y_test.getnnz(axis=1) > 0)
            cold_users = np.sort(self.rng.choice(eligible, min(cfg.cold_eval_users, len(eligible)), replace=False))
            rows = []
            for n in cfg.cold_histories:
                hist_n = self._truncated_hist(sp_.train_full, cold_users, n)
                evals = self._run_pass(cold_users, hist_n, sp_.Y_test, f"n={n}")
                for name, ev in evals.items():
                    rows.append({"Лайков в истории": n, "Модель": name, **ev.mean(),
                                 **self.beyond(ev.lists, cfg.main_k)})
            self.results["cold_start_test"] = pd.DataFrame(rows)
        return self

    # ---------------------------------------------------------------- 6. Сохранение
    def save(self) -> "Pipeline":
        cfg = self.cfg
        with self.timer("6. Сохранение результатов и модели"):
            out = Path(cfg.results_dir)
            for name, frame in self.results.items():
                frame.to_csv(out / f"{name}.csv", encoding="utf-8-sig")
            pd.DataFrame(self.timer.rows).to_csv(out / "timings.csv", index=False, encoding="utf-8-sig")
            (out / "config.json").write_text(json.dumps(cfg.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
            # Топ-10 итоговой системы для всех тестовых пользователей
            final = self.test_evals["Итоговая система (с MMR)"]
            recs = pd.DataFrame({
                "user_id": np.repeat(self.splits.idx.user_ids[final.users], cfg.main_k),
                "rank": np.tile(np.arange(1, cfg.main_k + 1), len(final.users)),
                "book_id": self.splits.idx.book_ids[final.lists[:, :cfg.main_k].ravel()],
            })
            recs.to_csv(out / "recommendations_test_top10.csv", index=False)
            # Класс берётся из модуля пакета, а не из __main__: иначе модель, сохранённую при запуске
            # `python -m recsys_pj`, нельзя загрузить из другого процесса.
            from recsys_pj.pipeline import RecommenderSystem
            system = RecommenderSystem(self.gen_full, self.router, self.splits.idx, self.raw.books, self.splits.hist_full,
                                       cfg)
            with open(Path(cfg.models_dir) / "recsys_system.pkl", "wb") as f:
                pickle.dump(system, f, protocol=pickle.HIGHEST_PROTOCOL)
        return self

    def run(self) -> "Pipeline":
        return self.load().preprocess().fit_stage().refit_stage().evaluate_stage().save()


class RecommenderSystem:
    """Обученная система для выдачи рекомендаций: известному пользователю или новому по его оценкам."""

    def __init__(self, generator: CandidateGenerator, router: Router, idx, books: pd.DataFrame, hist: History,
                 cfg: Config) -> None:
        self.generator, self.router, self.idx, self.cfg, self.hist = generator, router, idx, cfg, hist
        self.books = books.set_index("book_id")[["title", "authors"]]

    def _table(self, items: np.ndarray, batch: CandidateBatch) -> pd.DataFrame:
        items = items[items >= 0]
        ids = self.idx.book_ids[items]
        return pd.DataFrame({"book_id": ids, "Название": self.books["title"].reindex(ids).to_numpy(),
                             "Автор": self.books["authors"].reindex(ids).str.split(",").str[0].to_numpy()})

    def recommend(self, user_id: int, n: int = 10) -> pd.DataFrame:
        users = self.idx.users([user_id])
        batch = self.generator.generate(users, self.hist)
        return self._table(self.router.recommend(batch, n)[0], batch)

    def recommend_new_user(self, ratings: dict[int, int], n: int = 10) -> pd.DataFrame:
        """Рекомендации для нового пользователя по словарю {book_id: оценка} без переобучения."""
        books = self.idx.items(list(ratings))
        values = np.array(list(ratings.values()), dtype=np.float32)
        n_items = self.idx.n_items
        R = sp.csr_matrix((values, (np.zeros(len(books)), books)), shape=(1, n_items))
        liked = values >= self.cfg.relevance_threshold
        B = sp.csr_matrix((np.ones(liked.sum(), np.float32), (np.zeros(liked.sum()), books[liked])), shape=(1, n_items))
        batch = self.generator.generate(np.array([0]), History(R, B))
        return self._table(self.router.recommend(batch, n)[0], batch)


def main() -> None:
    parser = argparse.ArgumentParser(description="Гибридная система рекомендаций книг (Goodbooks-10k)")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default=".")
    parser.add_argument("--quick", action="store_true", help="уменьшенные выборки для проверки пайплайна")
    parser.add_argument("--with-improvements", action="store_true",
                        help="после основного пайплайна выполнить раздел 9: EASE, RP3beta, SASRec, серии, LambdaMART")
    args = parser.parse_args()
    out = Path(args.out_dir)
    cfg = Config(data_dir=Path(args.data_dir), results_dir=out / "results", figures_dir=out / "figures",
                 models_dir=out / "models")
    if args.quick:
        cfg.tune_users, cfg.ranker_train_users, cfg.max_test_users = 1000, 2000, 3000
        cfg.cold_eval_users, cfg.weight_samples, cfg.wd_epochs = 1000, 60, 2
    pipe = Pipeline(cfg).run()
    if args.with_improvements:
        import gc
        from .improvement_study import ImprovementStudy, StudyConfig
        for attr in ("train_data", "valid_data", "tuning"):
            if hasattr(pipe, attr):
                delattr(pipe, attr)
        gc.collect()
        scfg = StudyConfig()
        if args.quick:
            scfg = StudyConfig(source_tune_users=500, tune_users=800, cold_tune_users=400, rank_users=1500,
                               sasrec_epochs=1, lgbm_trees=60, cold_eval_users=400,
                               ease_lambdas=(1000.0,), rp3_betas=(0.5,))
        ImprovementStudy(pipe, scfg).run()


if __name__ == "__main__":
    main()
