"""Раздел 9: проверка гипотез улучшения, найденных в аналитической записке.

H0. Порядок строк ratings.csv отражает время (проверка на сериях книг) — условие для SASRec.
H1. Новые источники кандидатов: EASE и RP3beta — рост доли релевантных книг в пуле.
H2. Признаки серий книг («следующая непрочитанная часть», пропущенные части) и правило порядка частей.
H3. LambdaMART (LightGBM lambdarank) вместо поточечного GBDT.
H4. Честный выбор смеси: веса гибрида и доля смеси подбираются на разных половинах пользователей,
    смесь принимается, только если её выигрыш значим по бутстрэпу.
H5. SASRec (трансформер с cross-entropy) как источник кандидатов и признак ранжировщика.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.linalg
import torch
from torch import nn

from .data import History, Indexer
from .features import FeatureBuilder
from .metrics import metrics_at_k, top_items
from .models import BaseModel, _keep_top_k
from .neural import RankerData, ranked_ndcg

SERIES_PATTERN = re.compile(r"\(([^()#]+?),?\s*#(\d+(?:\.\d+)?)\)\s*$")


# =============================================================================== H0 + H2. Серии книг
class SeriesInfo:
    """Серии книг из названий вида «Title (Series Name, #N)»."""

    def __init__(self, books: pd.DataFrame, idx: Indexer) -> None:
        b = books.set_index("book_id").reindex(idx.book_ids)
        m = b["title"].fillna("").str.extract(SERIES_PATTERN)
        names = m[0].str.strip().str.lower()
        num = pd.to_numeric(m[1], errors="coerce")
        has = names.notna() & num.notna()
        codes, self.series_names = pd.factorize(names.where(has))
        self.item_series = np.where(has, codes, -1).astype(np.int64)
        self.item_num = np.where(has, num, 0.0).astype(np.float32)
        self.n_series = len(self.series_names)
        n = idx.n_items
        # Более ранние части той же серии: E[c, j] = 1, если j — часть с меньшим номером
        frame = pd.DataFrame({"item": np.arange(n), "s": self.item_series, "num": self.item_num})[has.to_numpy()]
        pairs = frame.merge(frame, on="s", suffixes=("", "_e"))
        pairs = pairs[pairs["num_e"] < pairs["num"]]
        self.E = sp.csr_matrix((np.ones(len(pairs), np.float32), (pairs["item"], pairs["item_e"])), shape=(n, n))
        self.n_earlier = np.asarray(self.E.sum(axis=1)).ravel().astype(np.float32)
        self.feature_names = ["series_in", "series_num", "series_rated", "series_liked", "series_missing_earlier",
                              "series_next_part", "series_prev_rated"]

    def order_check(self, ratings: pd.DataFrame, idx: Indexer) -> pd.DataFrame:
        """H0: доля пар «часть N и N+1, оценённые одним пользователем», где N оценена раньше."""
        r = ratings[["user_id", "book_id"]].copy()
        r["pos"] = r.groupby("user_id").cumcount()
        item = idx.items(r["book_id"])
        r["s"], r["num"] = self.item_series[item], self.item_num[item]
        r = r[r["s"] >= 0]
        a = r.merge(r, on=["user_id", "s"])
        rows = []
        for gap in (1, 2, 3):
            pair = a[a["num_y"] == a["num_x"] + gap]
            rows.append({"Пара частей": f"N и N+{gap}", "Число пар": len(pair),
                         "Доля, где N оценена раньше": float((pair["pos_x"] < pair["pos_y"]).mean())})
        return pd.DataFrame(rows)

    def features(self, users: np.ndarray, hist: History, cand: np.ndarray) -> np.ndarray:
        """Признаки серий для пула кандидатов (пользователи × P × 7)."""
        safe = np.maximum(cand, 0)
        rated = (hist.R[users] != 0).astype(np.float32).tocsr()
        liked = hist.B[users]
        s_item = self.item_series[safe]
        in_series = s_item >= 0
        S = max(self.n_series, 1)

        def per_series(M):
            coo = M.tocoo()
            s = self.item_series[coo.col]
            keep = s >= 0
            return sp.csr_matrix((coo.data[keep], (coo.row[keep], s[keep])), shape=(len(users), S)).toarray()

        rated_s, liked_s = per_series(rated), per_series(liked)
        s_safe = np.maximum(s_item, 0)
        series_rated = np.where(in_series, np.take_along_axis(rated_s, s_safe, axis=1), 0)
        series_liked = np.where(in_series, np.take_along_axis(liked_s, s_safe, axis=1), 0)
        rated_lower = (rated @ self.E.T).toarray()                       # оценено более ранних частей
        missing = np.where(in_series, self.n_earlier[safe] - np.take_along_axis(rated_lower, safe, axis=1), 0)
        prev_rated = np.where(in_series & (self.item_num[safe] > 1),
                              (np.take_along_axis(rated_lower, safe, axis=1) > 0).astype(np.float32), 0)
        next_part = (in_series & (series_rated > 0) & (missing == 0)).astype(np.float32)
        return np.stack([in_series.astype(np.float32), np.log1p(self.item_num[safe]), np.log1p(series_rated),
                         np.log1p(series_liked), np.log1p(np.maximum(missing, 0)), next_part, prev_rated],
                        axis=2).astype(np.float32)

    def fix_order(self, lists: np.ndarray) -> np.ndarray:
        """Правило: части одной серии в списке идут по возрастанию номера (позиции серии сохраняются)."""
        out = lists.copy()
        s = np.where(lists >= 0, self.item_series[np.maximum(lists, 0)], -1)
        rows = np.flatnonzero(((s[:, :, None] == s[:, None, :]) & (s[:, :, None] >= 0)).sum(axis=(1, 2)) > s.shape[1])
        for u in rows:
            for series in np.unique(s[u][s[u] >= 0]):
                pos = np.flatnonzero(s[u] == series)
                if len(pos) > 1:
                    items = lists[u, pos]
                    out[u, pos] = items[np.argsort(self.item_num[items], kind="stable")]
        return out


# =============================================================================== H1. EASE и RP3beta
class EASE(BaseModel):
    """EASE (Steck, 2019): линейный автоэнкодер с замкнутым решением W = I − P·diag(1/diag(P)), P = (XᵀX + λI)⁻¹."""
    name = "EASE"

    def __init__(self, lam: float) -> None:
        self.lam = lam

    def fit(self, hist: History) -> "EASE":
        X = hist.B
        G = (X.T @ X).toarray().astype(np.float32)
        G[np.diag_indices_from(G)] += self.lam
        P = scipy.linalg.inv(G, overwrite_a=True, check_finite=False)
        W = -P / np.diag(P)[None, :]
        np.fill_diagonal(W, 0.0)
        self.W = W.astype(np.float32)
        return self

    def score(self, users, hist):
        return np.asarray(hist.B[users] @ self.W)


class RP3beta(BaseModel):
    """RP3beta (Paudel et al., 2017): случайное блуждание пользователь → книга → пользователь → книга
    со штрафом популярности целевой книги pop^β; на книгу хранится k соседей."""
    name = "RP3beta"

    def __init__(self, beta: float, k: int = 100, chunk: int = 500) -> None:
        self.beta, self.k, self.chunk = beta, k, chunk

    def fit(self, hist: History) -> "RP3beta":
        B = hist.B
        deg_u = np.maximum(np.asarray(B.sum(axis=1)).ravel(), 1)
        deg_i = np.maximum(np.asarray(B.sum(axis=0)).ravel(), 1)
        Pui = sp.diags(1 / deg_u) @ B                     # пользователь → книга
        Piu = (sp.diags(1 / deg_i) @ B.T).tocsr()          # книга → пользователь
        penalty = (deg_i ** -self.beta).astype(np.float32)
        n, parts = B.shape[1], []
        Pui = Pui.tocsc()
        for start in range(0, n, self.chunk):
            rows = np.arange(start, min(start + self.chunk, n))
            values = (Piu[rows] @ Pui).toarray() * penalty[None, :]
            values[np.arange(len(rows)), rows] = 0
            parts.append(_keep_top_k(values, start, self.k))
        W = sp.csr_matrix((np.concatenate([p[2] for p in parts]),
                           (np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]))),
                          shape=(n, n), dtype=np.float32)
        self.W = W
        return self

    def score(self, users, hist):
        return (hist.B[users] @ self.W).toarray()


# =============================================================================== H5. SASRec
def user_sequences(hist: History, users: np.ndarray, max_len: int) -> np.ndarray:
    """Последние max_len оценённых книг пользователя в порядке времени (индексы +1, слева дополнены 0)."""
    out = np.zeros((len(users), max_len), dtype=np.int64)
    R, P = hist.R, hist.P
    for i, u in enumerate(users):
        a, b = R.indptr[u], R.indptr[u + 1]
        items = R.indices[a:b]
        if P is not None and len(items):  # R и P построены из одного фрейма: индексы совпадают
            items = items[np.argsort(P.data[P.indptr[u]:P.indptr[u + 1]], kind="stable")]
        items = items[-max_len:]
        if len(items):
            out[i, -len(items):] = items + 1
    return out


class SASRecNet(nn.Module):
    def __init__(self, n_items: int, dim: int, max_len: int, n_layers: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.item = nn.Embedding(n_items + 1, dim, padding_idx=0)
        self.pos = nn.Embedding(max_len, dim)
        layer = nn.TransformerEncoderLayer(dim, n_heads, dim * 2, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)
        self.dim = dim
        nn.init.normal_(self.item.weight, 0, 0.02)
        with torch.no_grad():
            self.item.weight[0].zero_()
        self.register_buffer("causal", torch.triu(torch.ones(max_len, max_len, dtype=torch.bool), 1))

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        L = seq.shape[1]
        x = self.item(seq) * self.dim ** 0.5 + self.pos(torch.arange(L))
        h = self.encoder(self.drop(x), mask=self.causal[:L, :L])
        return self.norm(h)

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        return h @ self.item.weight[1:].T


class SASRec(BaseModel):
    """SASRec (Kang & McAuley, 2018) с полной softmax cross-entropy вместо BCE с одним негативом
    (Klenitskiy & Vasilev, 2023): предсказывает следующую оценённую книгу по последовательности истории."""
    name = "SASRec"

    def __init__(self, n_items: int, dim: int = 64, max_len: int = 50, n_layers: int = 2, n_heads: int = 2,
                 dropout: float = 0.2, epochs: int = 4, batch: int = 128, lr: float = 1e-3, seed: int = 42) -> None:
        torch.manual_seed(seed)
        self.n_items, self.max_len, self.epochs, self.batch, self.lr, self.seed = n_items, max_len, epochs, batch, lr, seed
        self.net = SASRecNet(n_items, dim, max_len, n_layers, n_heads, dropout)
        self.history: list[dict] = []

    def fit(self, hist: History, valid_users: np.ndarray | None = None, Y_valid: sp.csr_matrix | None = None,
            verbose: bool = True) -> "SASRec":
        users = np.flatnonzero(hist.R.getnnz(axis=1) >= 2)
        seqs = user_sequences(hist, users, self.max_len + 1)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        rng = np.random.default_rng(self.seed)
        best, best_state = -1.0, None
        for epoch in range(self.epochs):
            started, losses = time.perf_counter(), []
            self.net.train()
            for s in range(0, len(users), self.batch):
                part = torch.from_numpy(seqs[s:s + self.batch])
                inp, tgt = part[:, :-1], part[:, 1:]
                h = self.net(inp)
                mask = tgt > 0
                loss = nn.functional.cross_entropy(self.net.logits(h[mask]), tgt[mask] - 1)
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(loss.item())
            row = {"эпоха": epoch + 1, "loss": float(np.mean(losses)), "время, с": time.perf_counter() - started}
            if valid_users is not None:
                row["val Recall@50"] = self.recall(valid_users, hist, Y_valid, 50)
                if row["val Recall@50"] > best:
                    best, best_state = row["val Recall@50"], {k: v.clone() for k, v in self.net.state_dict().items()}
            self.history.append(row)
            if verbose:
                print("  [SASRec] " + ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
            seqs = seqs[rng.permutation(len(seqs))]  # перемешивание порядка пользователей между эпохами
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.net.eval()
        return self

    @torch.no_grad()
    def score(self, users, hist):
        self.net.eval()
        out = []
        for s in range(0, len(users), 1000):
            seq = torch.from_numpy(user_sequences(hist, users[s:s + 1000], self.max_len))
            h = self.net(seq)[:, -1]
            out.append(self.net.logits(h).numpy())
        return np.vstack(out)

    def recall(self, users, hist, Y, k):
        scores = self.score(users, hist)
        seen = hist.R[users].tocoo()
        scores[seen.row, seen.col] = -np.inf
        top = top_items(scores, k)
        rel = Y[users].toarray() > 0
        hits = np.take_along_axis(rel, top, axis=1).sum(axis=1)
        return float((hits / np.maximum(rel.sum(axis=1), 1)).mean())


def source_recall(models: dict, users: np.ndarray, hist: History, Y: sp.csr_matrix, ks=(50, 100)) -> pd.DataFrame:
    """Доля релевантных книг в топ-k каждого источника по отдельности и в объединении."""
    rel = Y[users].toarray() > 0
    n_rel = np.maximum(rel.sum(axis=1), 1)
    seen = hist.R[users].tocoo()
    rows, union = [], {k: np.zeros_like(rel) for k in ks}
    for name, model in models.items():
        s = np.asarray(model.score(users, hist), dtype=np.float32)
        s[seen.row, seen.col] = -np.inf
        top = top_items(s, max(ks))
        row = {"Источник": name}
        for k in ks:
            hit = np.zeros_like(rel)
            np.put_along_axis(hit, top[:, :k], True, axis=1)
            union[k] |= hit
            row[f"Recall@{k}"] = float(((hit & rel).sum(axis=1) / n_rel).mean())
        rows.append(row)
    return pd.DataFrame(rows), union, rel, n_rel
