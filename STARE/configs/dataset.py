"""Dataset path mappings for STARE."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Tuple


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    price_rel_path: str
    text_rel_path: str
    price_alt_paths: Tuple[str, ...] = field(default_factory=tuple)
    text_alt_paths: Tuple[str, ...] = field(default_factory=tuple)

    def resolve_paths(self, datasets_dir: Path) -> "DatasetPaths":
        return DatasetPaths(
            dataset_name=self.name,
            price_path=_resolve_path(datasets_dir, self.price_rel_path, self.price_alt_paths),
            text_path=_resolve_path(datasets_dir, self.text_rel_path, self.text_alt_paths),
        )


@dataclass(frozen=True)
class DatasetPaths:
    dataset_name: str
    price_path: Path
    text_path: Path


DATASET_REGISTRY: Dict[str, DatasetConfig] = {
    "SAMPLE": DatasetConfig(
        name="SAMPLE",
        price_rel_path="sample_data/sample_price",
        text_rel_path="sample_data/sample_tweet",
    ),
    "STOCKNET": DatasetConfig(
        name="STOCKNET",
        price_rel_path="stocknet/price",
        text_rel_path="stocknet/tweet",
        price_alt_paths=(),
        text_alt_paths=(),
    ),
    "CMIN": DatasetConfig(
        name="CMIN",
        price_rel_path="CMIN/CMIN-Dataset/CMIN-US/price",
        text_rel_path="CMIN/CMIN-Dataset/CMIN-US/news",
        price_alt_paths=("CMIN/CMIN-US/price",),
        text_alt_paths=("CMIN/CMIN-US/news",),
    ),
    "SEP": DatasetConfig(
        name="SEP",
        price_rel_path="SEP/price",
        text_rel_path="SEP/tweet",
    ),
}


def _resolve_path(base: Path, primary: str, alternates: Iterable[str]) -> Path:
    candidates = [primary, *alternates]
    last_path: Path | None = None
    for rel in candidates:
        candidate = (base / rel).resolve()
        last_path = candidate
        if candidate.exists():
            return candidate
    if last_path is None:
        raise FileNotFoundError("No path candidates provided.")
    return last_path


__all__ = ["DatasetConfig", "DatasetPaths", "DATASET_REGISTRY"]
