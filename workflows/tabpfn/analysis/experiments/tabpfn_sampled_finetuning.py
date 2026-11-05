"""Opt-in chronological support sampling for DiagScale fine-tuning episodes.

TabPFN's installed fine-tuner first validates pandas input into NumPy, then
creates random mixed-month chunks. This adapter supplies month-coherent query
episodes and changes only their labeled support membership across strategies.
The normal fine-tuning path is untouched unless this subclass is used.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import partial
from unittest.mock import patch

import numpy as np
import pandas as pd

from analysis.experiments.diagscale_tabpfn import FinetunedDiagScaleRegressor
from analysis.experiments.tabpfn_context_sampler import (
    ContextMetadata, SamplerConfig, select_context,
)


@dataclass(frozen=True)
class EpisodeConfig:
    support_rows: int = 820
    query_rows: int = 204
    episodes_per_epoch: int = 100
    first_query_month: str = "2022-01"

    def __post_init__(self):
        if min(self.support_rows, self.query_rows, self.episodes_per_epoch) < 1:
            raise ValueError("Episode sizes and count must be positive")

    @property
    def task_rows(self) -> int:
        return self.support_rows + self.query_rows


def build_episode_positions(
    metadata: ContextMetadata,
    *,
    strategy: str,
    sampler_config: SamplerConfig,
    episode_config: EpisodeConfig,
    epoch_seed: int,
) -> tuple[list[np.ndarray], list[dict]]:
    """Fix same-month supervised queries; change support sampling strategy only."""
    if sampler_config.context_size != episode_config.support_rows:
        raise ValueError("Sampler budget must equal episode support size")
    cutoff = pd.Period(episode_config.first_query_month, "M").ordinal
    months = metadata.month_ordinal
    candidates = np.flatnonzero(months >= cutoff)
    if len(candidates) < episode_config.episodes_per_epoch * episode_config.query_rows:
        raise ValueError("Not enough month-coherent supervised queries")
    month_values, counts = np.unique(months[candidates], return_counts=True)
    rng = np.random.default_rng(epoch_seed)
    anchor_months = rng.choice(month_values, size=episode_config.episodes_per_epoch,
                               replace=True, p=counts / counts.sum())
    month_uses: dict[int, int] = {}
    for month in anchor_months:
        month_uses[int(month)] = month_uses.get(int(month), 0) + 1
    query_pools: dict[int, np.ndarray] = {}
    for month, uses in month_uses.items():
        pool = np.flatnonzero(months == month)
        required = uses * episode_config.query_rows
        if required > len(pool):
            raise ValueError(f"Month {pd.Period(ordinal=month, freq='M')} has too few distinct query rows")
        query_pools[month] = rng.permutation(pool)[:required]
    offsets = {month: 0 for month in month_uses}
    episodes: list[np.ndarray] = []
    audits: list[dict] = []
    all_ids = metadata.row_ids
    for episode_number, month in enumerate(anchor_months):
        key = int(month)
        start = offsets[key]
        query_positions = query_pools[key][start:start + episode_config.query_rows]
        offsets[key] += episode_config.query_rows
        selection = select_context(
            all_ids[query_positions], all_ids, metadata,
            strategy=strategy, information_policy="historical_only",
            config=sampler_config, seed=epoch_seed * 1000 + episode_number,
        )
        support_positions = metadata.positions(selection.row_ids)
        if len(support_positions) != episode_config.support_rows:
            raise AssertionError("Early query month has insufficient historical support")
        if not (months[support_positions] < month).all():
            raise AssertionError("Future support row entered an episode")
        episodes.append(np.concatenate((support_positions, query_positions)))
        audits.append({"episode": episode_number,
                       "query_month": str(pd.Period(ordinal=key, freq="M")),
                       "query_ids_sha256": _hash_ids(all_ids[query_positions]),
                       "support_ids_sha256": _hash_ids(selection.row_ids),
                       **selection.audit})
    queries = np.concatenate([episode[-episode_config.query_rows:] for episode in episodes])
    if len(np.unique(queries)) != len(queries):
        raise AssertionError("A supervised query was repeated within an epoch")
    return episodes, audits


def _hash_ids(ids: np.ndarray) -> str:
    import hashlib

    return hashlib.sha256(np.asarray(ids, dtype=np.int64).tobytes()).hexdigest()


def fixed_support_query_split(X, y, *, support_rows: int, stratify=None):
    """Preserve episode order after TabPFN's per-episode preprocessing setup."""
    if len(X) <= support_rows or len(X) != len(y):
        raise ValueError("Preprocessed episode has the wrong shape")
    return X[:support_rows], X[support_rows:], y[:support_rows], y[support_rows:]


class SampledContextDiagScaleRegressor(FinetunedDiagScaleRegressor):
    """Use installed DiagScale gradient logic with supplied support episodes."""

    def configure_context_sampling(
        self, metadata: ContextMetadata, strategy: str,
        sampler_config: SamplerConfig, episode_config: EpisodeConfig,
    ) -> None:
        if self.n_finetune_ctx_plus_query_samples != episode_config.task_rows:
            raise ValueError("Installed fine-tune task size does not match episode size")
        self.context_metadata_ = metadata
        self.context_strategy_ = strategy
        self.context_sampler_config_ = sampler_config
        self.context_episode_config_ = episode_config
        self.context_episode_audits_: list[dict] = []

    def _fit(self, *args, **kwargs):
        if not hasattr(self, "context_metadata_"):
            raise ValueError("Call configure_context_sampling before fit")
        from tabpfn.finetuning import finetuned_base

        original = finetuned_base.get_preprocessed_dataset_chunks

        def sampled_chunks(
            calling_instance, X_raw, y_raw, split_fn, max_data_size, model_type,
            *, equal_split_size, data_shuffle_seed, preprocessing_random_state,
            shuffle=True, force_no_stratify=False,
        ):
            if model_type != "regressor" or max_data_size != self.context_episode_config_.task_rows:
                raise AssertionError("Unexpected fine-tuning data path")
            if isinstance(X_raw, list) or len(X_raw) != len(self.context_metadata_.row_ids):
                raise AssertionError("Validated training rows changed order or count")
            episode_positions, audits = build_episode_positions(
                self.context_metadata_, strategy=self.context_strategy_,
                sampler_config=self.context_sampler_config_,
                episode_config=self.context_episode_config_,
                epoch_seed=int(data_shuffle_seed),
            )
            self.context_episode_audits_.append({
                "epoch_seed": int(data_shuffle_seed),
                "sampler_config": asdict(self.context_sampler_config_),
                "episode_config": asdict(self.context_episode_config_),
                "episodes": audits,
            })
            episodes_x = [X_raw[positions] for positions in episode_positions]
            episodes_y = [y_raw[positions] for positions in episode_positions]
            return original(
                calling_instance=calling_instance,
                X_raw=episodes_x, y_raw=episodes_y,
                split_fn=partial(fixed_support_query_split,
                                 support_rows=self.context_episode_config_.support_rows),
                max_data_size=None, model_type=model_type,
                equal_split_size=equal_split_size,
                data_shuffle_seed=int(data_shuffle_seed),
                preprocessing_random_state=preprocessing_random_state,
                shuffle=False, force_no_stratify=force_no_stratify,
            )

        with patch("tabpfn.finetuning.finetuned_base.get_preprocessed_dataset_chunks", sampled_chunks):
            return super()._fit(*args, **kwargs)
