"""Настройки пайплайна. Значения по умолчанию перенесены из домашнего задания и подобраны там на validation."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class Config:
    # Пути
    data_dir: Path = Path("data")
    results_dir: Path = Path("results")
    figures_dir: Path = Path("figures")
    models_dir: Path = Path("models")
    data_url: str = "https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/{}"

    # Разбиение и метрики (как в ДЗ: последние 20% оценок пользователя — test, ещё 10% — validation)
    seed: int = 42
    relevance_threshold: int = 4
    test_fraction: float = 0.20
    val_fraction: float = 0.125
    k_values: tuple[int, ...] = (5, 10, 20)
    main_k: int = 10
    batch_size: int = 2000

    # Базовые модели (гиперпараметры подобраны в ДЗ на validation)
    itemcf_shrink: float = 50.0
    itemcf_implicit_k: int = 5
    itemcf_explicit_k: int = 10
    itemcf_support_shrink: float = 50.0
    puresvd_factors: int = 400
    top_tags_per_book: int = 30

    # Признаки
    n_topics: int = 20

    # Генерация кандидатов
    candidates_per_model: int = 50

    # Сегменты пользователей по числу лайков в истории: новые <= new_max, активные > active_min
    segment_new_max: int = 10
    segment_active_min: int = 50
    cold_histories: tuple[int, ...] = (1, 3, 5, 10)

    # Подбор весов гибрида
    tune_users: int = 6000
    weight_samples: int = 300
    segment_min_gain: float = 0.01  # веса сегмента заменяют общие, только если дают +1% nDCG на validation

    # Нейросетевой ранжировщик Wide & Deep
    ranker_train_users: int = 20000
    ranker_truncated_share: float = 0.25
    wd_embedding_dim: int = 32
    wd_hidden: tuple[int, ...] = (128, 64)
    wd_dropout: float = 0.2
    wd_epochs: int = 8
    wd_lr: float = 1e-3
    wd_weight_decay: float = 1e-6
    wd_batch_users: int = 128
    wd_history_len: int = 100
    wd_loss: str = "listwise"   # "listwise" (softmax по списку пользователя + 0.1·BCE) или "bce"

    # Балансировка разнообразия (MMR)
    mmr_lambdas: tuple[float, ...] = (1.0, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5)
    mmr_depth: int = 100
    mmr_max_ndcg_loss: float = 0.02  # допустимая относительная потеря nDCG@10 на validation

    # Быстрый режим для отладки: ограничивает число тестовых пользователей
    max_test_users: int | None = None
    cold_eval_users: int = 5000

    def ensure_dirs(self) -> None:
        for directory in (self.data_dir, self.results_dir, self.figures_dir, self.models_dir):
            Path(directory).mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict:
        return {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(self).items()}
