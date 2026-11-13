"""Гибридная система рекомендаций книг с нейросетевым ранжировщиком (итоговый проект, Goodbooks-10k)."""
from .config import Config
from .pipeline import Pipeline, RecommenderSystem

__all__ = ["Config", "Pipeline", "RecommenderSystem"]
