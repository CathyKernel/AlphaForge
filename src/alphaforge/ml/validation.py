"""Leakage-safe time-series cross-validation.

Two modes are provided:

* ``walk_forward`` (default) -- mimics live research: train on all history
  up to ``t``, predict the next block, roll forward.  Only past data is
  ever used.
* ``purged_kfold`` -- every block is tested once, with the rest of the
  sample (past *and* future) available for training.  Useful for
  hyper-parameter research; NOT a substitute for walk-forward when
  reporting expected live performance.

Both modes apply:

* **Purging**: training samples whose *label window* overlaps the test
  block are dropped.  A sample dated T has a label realised over
  T+1..T+h; any T within ``[test_start - purge, test_end]`` would let the
  model "see" test-period outcomes through its labels.
* **Embargo**: an additional gap *after* each test block, dropping
  training samples whose features could correlate with test outcomes
  through slow-moving signals (Lopez de Prado, *Advances in Financial
  Machine Learning*, 2018, ch. 7).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Fold:
    """One CV fold: positional index arrays into a date-sorted frame."""

    train: np.ndarray
    test: np.ndarray
    train_dates: tuple[pd.Timestamp, pd.Timestamp]
    test_dates: tuple[pd.Timestamp, pd.Timestamp]
    n_purged: int
    n_embargoed: int

    @property
    def description(self) -> str:
        return (
            f"train {self.train_dates[0]:%Y-%m-%d}..{self.train_dates[1]:%Y-%m-%d} "
            f"({len(self.train)}) | test {self.test_dates[0]:%Y-%m-%d}.."
            f"{self.test_dates[1]:%Y-%m-%d} ({len(self.test)}) | "
            f"purged {self.n_purged}, embargoed {self.n_embargoed}"
        )


class PurgedWalkForwardCV:
    """Purged / embargoed walk-forward splitter over trading dates.

    Parameters
    ----------
    n_splits:
        Number of sequential test blocks.
    purge:
        Trading days removed from the training set immediately before each
        test block.  Should be at least ``label_horizon + 1``.
    embargo:
        Extra days removed *after* each test block (used fully in
        ``purged_kfold`` mode; in ``walk_forward`` the future is never
        used for training anyway, so the embargo is informational).
    min_train_days:
        Minimum training history before the first test block.
    mode:
        ``'walk_forward'`` or ``'purged_kfold'``.
    """

    def __init__(
        self,
        n_splits: int = 6,
        purge: int = 10,
        embargo: int = 5,
        min_train_days: int = 504,
        mode: str = "walk_forward",
    ) -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        if mode not in ("walk_forward", "purged_kfold"):
            raise ValueError(f"unknown mode {mode!r}")
        self.n_splits = int(n_splits)
        self.purge = int(purge)
        self.embargo = int(embargo)
        self.min_train_days = int(min_train_days)
        self.mode = mode

    # ------------------------------------------------------------------ #
    def _test_blocks(self, dates: pd.DatetimeIndex) -> list[tuple[int, int]]:
        n = len(dates)
        remaining = n - self.min_train_days
        if remaining < self.n_splits * 40:
            raise ValueError(
                f"Not enough dates for {self.n_splits} splits with "
                f"{self.min_train_days} min train days (have {n} dates)."
            )
        block = remaining // self.n_splits
        blocks = []
        start = self.min_train_days
        for k in range(self.n_splits):
            end = start + block if k < self.n_splits - 1 else n
            blocks.append((start, end))
            start = end
        return blocks

    def split(self, dates: pd.DatetimeIndex) -> list[Fold]:
        """Yield folds given the sorted, deduplicated date index of the data."""
        dates = pd.DatetimeIndex(sorted(set(dates)))
        n = len(dates)
        blocks = self._test_blocks(dates)
        folds: list[Fold] = []
        for t0, t1 in blocks:
            test = np.arange(t0, t1)
            if self.mode == "walk_forward":
                train_max = t0 - self.purge  # purge before test
                train = np.arange(0, max(train_max, 0))
                n_embargoed = 0
            else:  # purged_kfold: train on everything outside test +/- gaps
                lo = t0 - self.purge
                hi = t1 + self.embargo
                train = np.concatenate([np.arange(0, max(lo, 0)), np.arange(min(hi, n), n)])
                n_embargoed = max(min(hi, n) - t1, 0)
            n_purged = max(t0 - max(t0 - self.purge, 0), 0)
            folds.append(
                Fold(
                    train=train,
                    test=test,
                    train_dates=(dates[train[0]], dates[train[-1]]),
                    test_dates=(dates[t0], dates[t1 - 1]),
                    n_purged=n_purged,
                    n_embargoed=n_embargoed,
                )
            )
        return folds
