"""Нейросетевой ранжировщик Wide & Deep и классический GBDT-ранжировщик для сравнения.

Ранжировщик переупорядочивает пул кандидатов. Обучение: признаки считаются по истории train_fit,
метка — понравилась ли книга пользователю в периоде validation. Применение на тесте: признаки по
train_full, тот же обученный ранжировщик (стандартная схема двухэтапной системы).

Wide-часть (запоминание): линейная модель по плотным признакам + веса пересечений
«тема пользователя × тема книги», «активность пользователя × популярность книги» и вес самой книги.
Deep-часть (обобщение): эмбеддинг книги-кандидата, эмбеддинг пользователя как среднее эмбеддингов
понравившихся ему книг (работает и для новых пользователей), их поэлементное произведение
и плотные признаки → MLP. Итог: σ(y_wide + y_deep), функция потерь — бинарная кросс-энтропия.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch import nn

from .features import CandidateBatch, FeatureBuilder, N_POP_BUCKETS
from .metrics import metrics_at_k, top_items

N_ACTIVITY = 6


@dataclass
class RankerData:
    """Плоское хранение строк «пользователь — кандидат», сгруппированных по пользователям."""
    features: np.ndarray      # строки × F
    items: np.ndarray         # строки
    labels: np.ndarray        # строки
    offsets: np.ndarray       # пользователи + 1: строки пользователя i — offsets[i]:offsets[i+1]
    hist_items: list          # для каждого пользователя — массив понравившихся книг
    user_topic: np.ndarray
    user_activity: np.ndarray
    n_relevant: np.ndarray    # всего релевантных книг пользователя в отложенном периоде
    tag: np.ndarray

    @property
    def n_users(self) -> int:
        return len(self.offsets) - 1

    @staticmethod
    def from_batches(batches: list[tuple[CandidateBatch, np.ndarray, np.ndarray, str]]) -> "RankerData":
        feats, items, labels, sizes, hist, topic, act, n_rel, tags = [], [], [], [], [], [], [], [], []
        for batch, lab, nrel, tag in batches:
            for i in range(len(batch.users)):
                v = batch.valid[i]
                feats.append(batch.features[i, v]); items.append(batch.cand[i, v]); labels.append(lab[i, v])
                sizes.append(int(v.sum()))
            hist += batch.hist_items
            topic.append(batch.user_top_topic); act.append(batch.user_activity); n_rel.append(nrel)
            tags += [tag] * len(batch.users)
        return RankerData(np.concatenate(feats), np.concatenate(items).astype(np.int64),
                          np.concatenate(labels).astype(np.float32), np.r_[0, np.cumsum(sizes)], hist,
                          np.concatenate(topic), np.concatenate(act), np.concatenate(n_rel), np.array(tags))

    @staticmethod
    def concat(parts: list["RankerData"]) -> "RankerData":
        sizes = np.concatenate([np.diff(p.offsets) for p in parts])
        return RankerData(np.concatenate([p.features for p in parts]), np.concatenate([p.items for p in parts]),
                          np.concatenate([p.labels for p in parts]), np.r_[0, np.cumsum(sizes)],
                          [h for p in parts for h in p.hist_items], np.concatenate([p.user_topic for p in parts]),
                          np.concatenate([p.user_activity for p in parts]), np.concatenate([p.n_relevant for p in parts]),
                          np.concatenate([p.tag for p in parts]))

    def subset(self, users: np.ndarray) -> "RankerData":
        rows = np.concatenate([np.arange(self.offsets[u], self.offsets[u + 1]) for u in users])
        sizes = np.diff(self.offsets)[users]
        return RankerData(self.features[rows], self.items[rows], self.labels[rows], np.r_[0, np.cumsum(sizes)],
                          [self.hist_items[u] for u in users], self.user_topic[users], self.user_activity[users],
                          self.n_relevant[users], self.tag[users])


def history_items(B: sp.csr_matrix, users: np.ndarray, max_len: int, rng: np.random.Generator) -> list:
    out = []
    for u in users:
        items = B.indices[B.indptr[u]:B.indptr[u + 1]]
        if len(items) > max_len:
            items = rng.choice(items, max_len, replace=False)
        out.append(items.astype(np.int64))
    return out


class WideDeep(nn.Module):
    def __init__(self, n_items: int, n_dense: int, n_topics: int, emb_dim: int, hidden: tuple[int, ...],
                 dropout: float, use_wide: bool = True, use_deep: bool = True) -> None:
        super().__init__()
        self.n_items, self.n_topics = n_items, n_topics
        self.use_wide, self.use_deep = use_wide, use_deep
        # Wide
        self.wide_dense = nn.Linear(n_dense, 1)
        self.wide_item = nn.Embedding(n_items, 1)
        self.wide_topic_cross = nn.Embedding(n_topics * n_topics, 1)
        self.wide_pop_cross = nn.Embedding(N_ACTIVITY * N_POP_BUCKETS, 1)
        for emb in (self.wide_item, self.wide_topic_cross, self.wide_pop_cross):
            nn.init.zeros_(emb.weight)
        # Deep
        self.item_emb = nn.Embedding(n_items + 1, emb_dim, padding_idx=n_items)
        self.hist_emb = nn.EmbeddingBag(n_items + 1, emb_dim, mode="mean", padding_idx=n_items)
        self.topic_emb = nn.Embedding(n_topics, 8)
        nn.init.normal_(self.item_emb.weight, 0, 0.05)
        nn.init.normal_(self.hist_emb.weight, 0, 0.05)
        layers, size = [], emb_dim * 3 + 8 + n_dense
        for h in hidden:
            layers += [nn.Linear(size, h), nn.ReLU(), nn.Dropout(dropout)]
            size = h
        layers.append(nn.Linear(size, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, dense, items, row_user, hist_flat, hist_offsets, user_topic, user_act, item_topic, item_pop):
        logit = torch.zeros(len(items))
        if self.use_wide:
            logit = logit + self.wide_dense(dense).squeeze(1) + self.wide_item(items).squeeze(1)
            logit = logit + self.wide_topic_cross(user_topic[row_user] * self.n_topics + item_topic).squeeze(1)
            logit = logit + self.wide_pop_cross(user_act[row_user] * N_POP_BUCKETS + item_pop).squeeze(1)
        if self.use_deep:
            user_vec = self.hist_emb(hist_flat, hist_offsets)[row_user]
            item_vec = self.item_emb(items)
            x = torch.cat([item_vec, user_vec, item_vec * user_vec, self.topic_emb(item_topic), dense], dim=1)
            logit = logit + self.mlp(x).squeeze(1)
        return logit


def listwise_loss(logits: torch.Tensor, labels: torch.Tensor, row_user: torch.Tensor, n_users: int) -> torch.Tensor:
    """Softmax-кросс-энтропия внутри списка каждого пользователя (ListNet с целевым распределением
    по релевантным книгам). В отличие от поточечной BCE, каждый пользователь вносит одинаковый вклад,
    что ближе к усреднённому по пользователям nDCG."""
    m = torch.full((n_users,), -1e9).scatter_reduce(0, row_user, logits.detach(), reduce="amax")
    shifted = logits - m[row_user]
    denom = torch.zeros(n_users).index_add(0, row_user, torch.exp(shifted))
    log_p = shifted - torch.log(denom[row_user])
    n_pos = torch.zeros(n_users).index_add(0, row_user, labels)
    weight = labels / n_pos[row_user].clamp(min=1)
    per_user = -torch.zeros(n_users).index_add(0, row_user, weight * log_p)
    has = n_pos > 0
    return per_user[has].mean() if has.any() else logits.sum() * 0


class WideDeepRanker:
    """Обёртка: нормализация признаков, обучение по батчам пользователей, ранние остановки по nDCG@10."""

    def __init__(self, builder: FeatureBuilder, n_items: int, n_dense: int, cfg, feature_mask: np.ndarray | None = None,
                 use_wide: bool = True, use_deep: bool = True, name: str = "Wide & Deep") -> None:
        self.builder, self.cfg, self.name = builder, cfg, name
        self.n_items = n_items
        self.feature_mask = np.ones(n_dense, bool) if feature_mask is None else feature_mask
        torch.manual_seed(cfg.seed)
        self.net = WideDeep(n_items, int(self.feature_mask.sum()), builder.topics.n_topics, cfg.wd_embedding_dim,
                            cfg.wd_hidden, cfg.wd_dropout, use_wide, use_deep)
        self.history: list[dict] = []

    # --- подготовка тензоров ---
    def _dense(self, X: np.ndarray) -> np.ndarray:
        return np.clip((X[:, self.feature_mask] - self.mu) / self.sd, -5, 5).astype(np.float32)

    def _tensors(self, feats, items, sizes, hist_items, user_topic, user_act):
        b = self.builder
        hist_lens = np.array([max(len(h), 1) for h in hist_items])
        hist_flat = np.concatenate([h if len(h) else np.array([self.n_items]) for h in hist_items])
        return dict(
            dense=torch.from_numpy(self._dense(feats)),
            items=torch.from_numpy(items),
            row_user=torch.from_numpy(np.repeat(np.arange(len(sizes)), sizes)),
            hist_flat=torch.from_numpy(hist_flat),
            hist_offsets=torch.from_numpy(np.r_[0, np.cumsum(hist_lens)[:-1]]),
            user_topic=torch.from_numpy(user_topic), user_act=torch.from_numpy(user_act),
            item_topic=torch.from_numpy(b.item_top_topic[items]), item_pop=torch.from_numpy(b.item_pop_bucket[items]),
        )

    def _batch(self, data: RankerData, users: np.ndarray):
        lo, hi = data.offsets[users], data.offsets[users + 1]
        rows = np.concatenate([np.arange(a, b) for a, b in zip(lo, hi)])
        t = self._tensors(data.features[rows], data.items[rows], hi - lo, [data.hist_items[u] for u in users],
                          data.user_topic[users], data.user_activity[users])
        return t, torch.from_numpy(data.labels[rows])

    # --- обучение ---
    def fit(self, train: RankerData, valid: RankerData, verbose: bool = True) -> "WideDeepRanker":
        cfg = self.cfg
        if train.features.dtype == np.float32:
            X = train.features[:, self.feature_mask]
            self.mu, self.sd = X.mean(axis=0), np.where(X.std(axis=0) > 0, X.std(axis=0), 1.0)
        else:  # компактное хранение (float16): точные среднее и разброс по частям во float64
            s1 = s2 = 0.0
            for i in range(0, len(train.features), 500_000):
                part = train.features[i:i + 500_000][:, self.feature_mask].astype(np.float64)
                s1, s2 = s1 + part.sum(axis=0), s2 + (part ** 2).sum(axis=0)
            n = len(train.features)
            self.mu = (s1 / n).astype(np.float32)
            sd = np.sqrt(np.maximum(s2 / n - (s1 / n) ** 2, 0)).astype(np.float32)
            self.sd = np.where(sd > 0, sd, 1.0)
        opt = torch.optim.Adam(self.net.parameters(), lr=cfg.wd_lr, weight_decay=cfg.wd_weight_decay)
        bce = nn.BCEWithLogitsLoss()
        rng = np.random.default_rng(cfg.seed)
        best_state, best_score, bad = None, -1.0, 0
        for epoch in range(cfg.wd_epochs):
            started, losses = time.perf_counter(), []
            self.net.train()
            order = rng.permutation(train.n_users)
            for s in range(0, len(order), cfg.wd_batch_users):
                batch_users = order[s:s + cfg.wd_batch_users]
                t, y = self._batch(train, batch_users)
                opt.zero_grad()
                logits = self.net(**t)
                if cfg.wd_loss == "listwise":
                    loss = listwise_loss(logits, y, t["row_user"], len(batch_users)) + 0.1 * bce(logits, y)
                else:
                    loss = bce(logits, y)
                loss.backward()
                opt.step()
                losses.append(loss.item())
            score = float(self.evaluate(valid).mean())
            self.history.append({"эпоха": epoch + 1, "loss": float(np.mean(losses)), "val nDCG@10": score,
                                 "время, с": time.perf_counter() - started})
            if verbose:
                print(f"  [{self.name}] эпоха {epoch + 1}: loss={np.mean(losses):.4f}, "
                      f"val nDCG@10={score:.4f} ({time.perf_counter() - started:.0f} с)")
            if score > best_score:
                best_score, bad = score, 0
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= 2:
                    break
        self.net.load_state_dict(best_state)
        self.net.eval()
        return self

    @torch.no_grad()
    def predict_data(self, data: RankerData) -> np.ndarray:
        self.net.eval()
        out = []
        for s in range(0, data.n_users, 1024):
            t, _ = self._batch(data, np.arange(s, min(s + 1024, data.n_users)))
            out.append(self.net(**t).numpy())
        return np.concatenate(out)

    def evaluate(self, data: RankerData, k: int = 10) -> np.ndarray:
        return ranked_ndcg(data, self.predict_data(data), k)

    @torch.no_grad()
    def score(self, batch: CandidateBatch) -> np.ndarray:
        """Скоры для пула кандидатов (пользователи × P); история берётся из batch.hist_items."""
        self.net.eval()
        v = batch.valid
        sizes = v.sum(axis=1)
        t = self._tensors(batch.features[v], batch.cand[v].astype(np.int64), sizes, batch.hist_items,
                          batch.user_top_topic, batch.user_activity)
        out = np.full(batch.cand.shape, -np.inf, dtype=np.float32)
        out[v] = self.net(**t).numpy()
        return out


class GBDTRanker:
    """Классический LTR для сравнения: градиентный бустинг (pointwise) на тех же признаках."""

    def __init__(self, cfg, max_rows: int = 1_500_000, name: str = "GBDT-ранжировщик",
                 feature_mask: np.ndarray | None = None) -> None:
        self.feature_mask = feature_mask
        from sklearn.ensemble import HistGradientBoostingClassifier
        self.cfg, self.max_rows, self.name = cfg, max_rows, name
        self.model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=63,
                                                    early_stopping=True, validation_fraction=0.1,
                                                    random_state=cfg.seed)

    def fit(self, train: RankerData) -> "GBDTRanker":
        rng = np.random.default_rng(self.cfg.seed)
        rows = np.arange(len(train.labels))
        if len(rows) > self.max_rows:
            rows = np.sort(rng.choice(rows, self.max_rows, replace=False))
        # Вес строки 1 / (число релевантных у пользователя): каждый пользователь вносит равный вклад.
        sizes = np.diff(train.offsets)
        row_user = np.repeat(np.arange(train.n_users), sizes)
        pos = np.bincount(row_user, weights=train.labels, minlength=train.n_users)
        weight = 1.0 / np.maximum(pos, 1)[row_user]
        self.model.fit(self._x(train.features[rows]).astype(np.float32), train.labels[rows], sample_weight=weight[rows])
        return self

    def _x(self, X: np.ndarray) -> np.ndarray:
        return X if self.feature_mask is None else X[:, self.feature_mask]

    def predict_data(self, data: RankerData) -> np.ndarray:
        return np.concatenate([self.model.predict_proba(self._x(data.features[i:i + 500_000]).astype(np.float32))[:, 1]
                               for i in range(0, len(data.features), 500_000)]) if len(data.features) else np.zeros(0)

    def score(self, batch: CandidateBatch) -> np.ndarray:
        out = np.full(batch.cand.shape, -np.inf, dtype=np.float32)
        out[batch.valid] = self.model.predict_proba(self._x(batch.features[batch.valid]))[:, 1]
        return out


def ranked_ndcg(data: RankerData, scores: np.ndarray, k: int = 10) -> np.ndarray:
    """nDCG@k для каждого пользователя RankerData по скорам строк."""
    sizes = np.diff(data.offsets)
    P = sizes.max()
    S = np.full((data.n_users, P), -np.inf, dtype=np.float32)
    L = np.zeros((data.n_users, P), dtype=bool)
    row_user = np.repeat(np.arange(data.n_users), sizes)
    pos = np.arange(len(scores)) - np.repeat(data.offsets[:-1], sizes)
    S[row_user, pos], L[row_user, pos] = scores, data.labels > 0
    order = top_items(S, k)
    hits = np.take_along_axis(L, order, axis=1)
    return metrics_at_k(hits, data.n_relevant, k)[f"nDCG@{k}"]


def permutation_importance(ranker, data: RankerData, feature_names: list[str], groups: dict[str, list[str]],
                           seed: int = 42) -> pd.DataFrame:
    """Падение nDCG@10 при перемешивании группы признаков между строками."""
    rng = np.random.default_rng(seed)
    base = float(ranked_ndcg(data, ranker.predict_data(data)).mean())
    rows = []
    for group, names in groups.items():
        cols = [feature_names.index(n) for n in names]
        shuffled = RankerData(**{**data.__dict__, "features": data.features.copy()})
        perm = rng.permutation(len(data.labels))
        shuffled.features[:, cols] = data.features[perm][:, cols]
        value = float(ranked_ndcg(shuffled, ranker.predict_data(shuffled)).mean())
        rows.append({"Группа признаков": group, "nDCG@10 после перемешивания": value, "Падение nDCG@10": base - value})
    return pd.DataFrame(rows).sort_values("Падение nDCG@10", ascending=False)
