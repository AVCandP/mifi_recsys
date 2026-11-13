"""Этапы «загрузка данных» и «предобработка»: чтение Goodbooks-10k, очистка тегов, разбиение по времени."""
from __future__ import annotations

import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .config import Config

DATA_FILES = ("ratings.csv", "books.csv", "tags.csv", "book_tags.csv")

# Теги-«полки» описывают действия читателя, а не содержание книги (см. раздел 2.5 ДЗ).
NOISE_TAG_PATTERN = re.compile(
    r"(?:^|-)(?:to-?read|tbr|currently|reading|read|re-read|reread|owned|own|my-books|my-library|books?-i|favou?rites?|"
    r"favs?|fave|kindle|e-?books?|audio|audiobooks?|audible|library|default|wish-?list|to-buy|to-get|"
    r"book-?club|dnf|did-not-finish|on-hold|abandoned|unfinished|shelfari|goodreads|bookshelf|nook|"
    r"paperback|hardcover|pdf|series|maybe|buy|bought|borrowed|have|home|finished|stand-?alone)(?:-|$)|\d{4}|^-+$"
)


def download_data(cfg: Config) -> None:
    """Скачивает отсутствующие CSV из репозитория датасета; существующие файлы не трогает."""
    Path(cfg.data_dir).mkdir(parents=True, exist_ok=True)
    for name in DATA_FILES:
        path = Path(cfg.data_dir) / name
        if not path.exists():
            print(f"Скачиваю {name}...")
            urllib.request.urlretrieve(cfg.data_url.format(name), path)


@dataclass
class RawData:
    ratings: pd.DataFrame
    books: pd.DataFrame
    book_tags: pd.DataFrame  # book_id, tag_name, count (только содержательные теги)
    all_book_tags: pd.DataFrame


def load_data(cfg: Config) -> RawData:
    download_data(cfg)
    ratings = pd.read_csv(Path(cfg.data_dir) / "ratings.csv",
                          dtype={"user_id": "int32", "book_id": "int32", "rating": "int8"})
    books = pd.read_csv(Path(cfg.data_dir) / "books.csv")
    tags = pd.read_csv(Path(cfg.data_dir) / "tags.csv")
    book_tags = pd.read_csv(Path(cfg.data_dir) / "book_tags.csv")
    # book_tags ссылается на goodreads_book_id, а ratings — на book_id; есть отрицательные счётчики и повторы.
    book_tags = (book_tags[book_tags["count"] > 0]
                 .merge(books[["book_id", "goodreads_book_id"]], on="goodreads_book_id")
                 .merge(tags, on="tag_id")
                 .groupby(["book_id", "tag_name"], as_index=False)["count"].sum())
    content = book_tags[~book_tags["tag_name"].str.contains(NOISE_TAG_PATTERN)].reset_index(drop=True)
    ratings = ratings.drop_duplicates(["user_id", "book_id"], keep="last").reset_index(drop=True)
    return RawData(ratings=ratings, books=books, book_tags=content, all_book_tags=book_tags)


class Indexer:
    """Переводит внешние идентификаторы в индексы строк и столбцов матриц."""

    def __init__(self, user_ids: np.ndarray, book_ids: np.ndarray) -> None:
        self.user_ids = np.sort(np.unique(user_ids))
        self.book_ids = np.sort(np.unique(book_ids))
        self.n_users, self.n_items = len(self.user_ids), len(self.book_ids)

    def users(self, ids) -> np.ndarray:
        return np.searchsorted(self.user_ids, np.asarray(ids))

    def items(self, ids) -> np.ndarray:
        return np.searchsorted(self.book_ids, np.asarray(ids))


@dataclass
class History:
    """История пользователей: матрица оценок R, бинарная матрица лайков B (оценка >= порога)
    и матрица порядковых номеров оценок P (1 — первая оценка пользователя) для последовательных моделей."""
    R: sp.csr_matrix
    B: sp.csr_matrix
    P: sp.csr_matrix | None = None

    @classmethod
    def from_frame(cls, frame: pd.DataFrame, idx: Indexer, threshold: int) -> "History":
        rows, cols = idx.users(frame["user_id"]), idx.items(frame["book_id"])
        values = frame["rating"].to_numpy(dtype=np.float32)
        R = sp.csr_matrix((values, (rows, cols)), shape=(idx.n_users, idx.n_items), dtype=np.float32)
        liked = values >= threshold
        B = sp.csr_matrix((np.ones(liked.sum(), np.float32), (rows[liked], cols[liked])),
                          shape=(idx.n_users, idx.n_items), dtype=np.float32)
        order = frame.groupby("user_id").cumcount().to_numpy(dtype=np.float32) + 1
        P = sp.csr_matrix((order, (rows, cols)), shape=(idx.n_users, idx.n_items), dtype=np.float32)
        return cls(R, B, P)

    @property
    def n_likes(self) -> np.ndarray:
        return self.B.getnnz(axis=1)


def temporal_user_split(frame: pd.DataFrame, fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Откладывает последнюю долю оценок каждого пользователя (порядок строк файла = время)."""
    position = frame.groupby("user_id").cumcount().to_numpy()
    size = frame.groupby("user_id")["user_id"].transform("size").to_numpy()
    is_holdout = position >= size - np.ceil(size * fraction).astype(int)
    return frame[~is_holdout].reset_index(drop=True), frame[is_holdout].reset_index(drop=True)


def truncate_history(frame: pd.DataFrame, users: np.ndarray, n_likes: int, threshold: int) -> pd.DataFrame:
    """Оставляет выбранным пользователям только первые n_likes понравившихся книг (симуляция новичков).

    Остальные пользователи не меняются. Низкие оценки до n-го лайка сохраняются, как в реальной истории.
    """
    selected = frame["user_id"].isin(users)
    part = frame[selected]
    likes_so_far = (part["rating"] >= threshold).groupby(part["user_id"]).cumsum()
    keep = (likes_so_far <= n_likes) & ~((likes_so_far == n_likes) & (part["rating"] < threshold))
    return pd.concat([frame[~selected], part[keep.to_numpy()]], ignore_index=True)


@dataclass
class Splits:
    idx: Indexer
    train_full: pd.DataFrame
    train_fit: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    hist_fit: History
    hist_full: History
    Y_val: sp.csr_matrix
    Y_test: sp.csr_matrix


def make_splits(raw: RawData, cfg: Config) -> Splits:
    idx = Indexer(raw.ratings["user_id"].to_numpy(), np.union1d(raw.books["book_id"], raw.ratings["book_id"]))
    train_full, test = temporal_user_split(raw.ratings, cfg.test_fraction)
    train_fit, val = temporal_user_split(train_full, cfg.val_fraction)
    t = cfg.relevance_threshold
    return Splits(
        idx=idx, train_full=train_full, train_fit=train_fit, val=val, test=test,
        hist_fit=History.from_frame(train_fit, idx, t), hist_full=History.from_frame(train_full, idx, t),
        Y_val=History.from_frame(val, idx, t).B, Y_test=History.from_frame(test, idx, t).B,
    )
