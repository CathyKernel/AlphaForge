"""PyTorch LSTM alpha model (optional dependency: ``pip install alphaforge[dl]``).

Design
------
Each training sample is one (ticker, date) pair:

    x = factor features of that ticker over the last ``lookback`` days,
    y = forward excess return.

Sequences are materialised per batch from a dense (date, ticker, feature)
tensor, so memory stays O(T * N * F) regardless of sample count -- a
ten-year, hundred-stock panel needs ~30 MB.  This avoids the naive
(N_samples x lookback x F) blow-up that makes LSTM research painful.

The model itself is deliberately small (2-layer LSTM, 64 hidden units):
cross-sectional equity prediction is a low-signal problem where capacity
hurts more than it helps.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphaforge.ml.models import AlphaModel


class LSTMAlphaModel(AlphaModel):
    """Sequence model over the per-ticker factor history.

    Parameters
    ----------
    lookback:
        Trading days of history per sample.
    hidden_size / num_layers / dropout:
        Recurrent architecture.
    max_epochs / patience:
        Early-stopping schedule on validation loss.
    batch_size:
        Mini-batch size (samples per step).
    """

    name = "lstm"

    def __init__(
        self,
        lookback: int = 60,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 1024,
        max_epochs: int = 40,
        patience: int = 6,
        device: str | None = None,
        seed: int = 42,
    ) -> None:
        self.lookback = int(lookback)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.batch_size = int(batch_size)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.device = device
        self.seed = int(seed)
        self._net = None
        self._feature_names: list[str] | None = None

    # ------------------------------------------------------------------ #
    def _require_torch(self):
        try:
            import torch

            return torch
        except ImportError as exc:  # pragma: no cover - environment specific
            raise ImportError(
                "PyTorch is required for LSTMAlphaModel. "
                "Install it with: pip install alphaforge[dl]"
            ) from exc

    def _dense_tensors(self, X: pd.DataFrame, y: pd.Series | None):
        """(T, N, F) feature tensor, (T, N) label array, index maps.

        Missing ``(date, ticker)`` pairs -- e.g. a stock that listed after
        the panel start -- become NaN when the long frame is densified.
        They are zero-padded: the standard LSTM convention for
        variable-length histories and consistent with the feature
        builder's NaN -> 0 fill.  Labels keep their NaN (samples with
        missing labels are simply never selected for training).
        """
        dates = X.index.get_level_values(0).unique().sort_values()
        tickers = X.index.get_level_values(1).unique().sort_values()
        feats = []
        for col in X.columns:
            wide = X[col].unstack("ticker")
            feats.append(wide.reindex(index=dates, columns=tickers).to_numpy(dtype=np.float32))
        F = np.stack(feats, axis=2)  # (T, N, F)
        F = np.nan_to_num(F, nan=0.0, copy=False)  # zero-pad pre-listing rows
        if y is not None:
            lab = (
                y.unstack("ticker").reindex(index=dates, columns=tickers).to_numpy(dtype=np.float64)
            )
        else:
            lab = np.full((len(dates), len(tickers)), np.nan)
        self._feature_names = list(X.columns)
        return F, lab, dates, tickers

    def _sample_index(self, lab: np.ndarray) -> np.ndarray:
        """(t_pos, n_pos) pairs with a complete window and a finite label."""
        T, N = lab.shape
        t0 = self.lookback - 1
        finite = np.isfinite(lab)
        rows, cols = np.where(finite)
        keep = rows >= t0
        return np.stack([rows[keep], cols[keep]], axis=1)

    def _predict_index(self, lab: np.ndarray) -> np.ndarray:
        """(t_pos, n_pos) pairs with a complete window (labels ignored)."""
        T, N = lab.shape
        t0 = self.lookback - 1
        rows, cols = np.where(np.ones((T, N), dtype=bool))
        keep = rows >= t0
        return np.stack([rows[keep], cols[keep]], axis=1)

    # ------------------------------------------------------------------ #
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> LSTMAlphaModel:
        torch = self._require_torch()
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")

        F, lab, _, _ = self._dense_tensors(X, y)
        samples = self._sample_index(lab)
        if len(samples) < self.batch_size:
            raise ValueError(f"too few LSTM samples: {len(samples)}")

        Xtr = torch.from_numpy(F).float().to(device)
        ytr = torch.from_numpy(np.nan_to_num(lab)).float().to(device)

        Xval_t = yval_t = None
        val_samples = None
        if X_val is not None and y_val is not None:
            Fv, labv, _, _ = self._dense_tensors(X_val, y_val)
            val_samples = self._sample_index(labv)
            if len(val_samples):
                Xval_t = torch.from_numpy(Fv).float().to(device)
                yval_t = torch.from_numpy(np.nan_to_num(labv)).float().to(device)

        n_feat = F.shape[2]
        self._net = self._make_net(n_feat, device)
        opt = torch.optim.AdamW(self._net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = torch.nn.HuberLoss(delta=0.05)

        best_val, best_state, wait = float("inf"), None, 0
        n_samples = len(samples)
        # vectorised batch assembly: (B, L, F) via advanced indexing
        offs = torch.arange(-(self.lookback - 1), 1, device=device)
        for _epoch in range(self.max_epochs):
            self._net.train()
            perm = torch.randperm(n_samples, device=device)
            for bstart in range(0, n_samples, self.batch_size):
                sel = perm[bstart : bstart + self.batch_size].cpu().numpy()
                t_idx = torch.from_numpy(samples[sel, 0]).to(device)
                n_idx = torch.from_numpy(samples[sel, 1]).to(device)
                time_idx = t_idx[:, None] + offs[None, :]  # (B, L)
                batch_x = Xtr[time_idx, n_idx[:, None]]  # (B, L, F)
                batch_y = ytr[t_idx, n_idx]
                opt.zero_grad()
                loss = loss_fn(self._net(batch_x).squeeze(-1), batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._net.parameters(), 5.0)
                opt.step()
            if val_samples is not None and Xval_t is not None:
                self._net.eval()
                with torch.no_grad():
                    preds, targets = [], []
                    for bstart in range(0, len(val_samples), self.batch_size * 4):
                        vs = val_samples[bstart : bstart + self.batch_size * 4]
                        vt = torch.from_numpy(vs[:, 0]).to(device)
                        vn = torch.from_numpy(vs[:, 1]).to(device)
                        vtime = vt[:, None] + offs[None, :]
                        bx = Xval_t[vtime, vn[:, None]]
                        preds.append(self._net(bx).squeeze(-1))
                        targets.append(yval_t[vt, vn])
                    val_loss = float(loss_fn(torch.cat(preds), torch.cat(targets)))
                if val_loss < best_val - 1e-6:
                    best_val, best_state, wait = (
                        val_loss,
                        {k: v.detach().clone() for k, v in self._net.state_dict().items()},
                        0,
                    )
                else:
                    wait += 1
                    if wait >= self.patience:
                        break
        if best_state is not None:
            self._net.load_state_dict(best_state)
        return self

    # ------------------------------------------------------------------ #
    def predict(self, X: pd.DataFrame) -> pd.Series:
        torch = self._require_torch()
        if self._net is None:
            raise RuntimeError("Model is not fitted")
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        F, lab, dates, tickers = self._dense_tensors(X, None)
        samples = self._predict_index(lab)
        if len(samples) == 0:
            return pd.Series(dtype=float, name=self.name)
        Ft = torch.from_numpy(F).float().to(device)
        self._net.eval()
        preds = np.full(lab.shape, np.nan, dtype=np.float64)
        offs_np = np.arange(-(self.lookback - 1), 1)
        with torch.no_grad():
            for bstart in range(0, len(samples), self.batch_size * 8):
                ss = samples[bstart : bstart + self.batch_size * 8]
                t_idx = torch.from_numpy(ss[:, 0]).to(device)
                n_idx = torch.from_numpy(ss[:, 1]).to(device)
                time_idx = t_idx[:, None] + torch.from_numpy(offs_np).to(device)
                bx = Ft[time_idx, n_idx[:, None]]
                out = self._net(bx).squeeze(-1).cpu().numpy()
                preds[ss[:, 0], ss[:, 1]] = out
        s = pd.Series(preds.ravel(), name=self.name)
        idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
        s.index = idx
        # keep only rows that actually appeared in X
        return s.reindex(X.index)

    # ------------------------------------------------------------------ #
    def _make_net(self, n_feat: int, device: str):
        """Construct the network (torch imported lazily)."""
        import torch

        hidden, layers, dropout = self.hidden_size, self.num_layers, self.dropout

        class Net(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lstm = torch.nn.LSTM(
                    input_size=n_feat,
                    hidden_size=hidden,
                    num_layers=layers,
                    dropout=dropout if layers > 1 else 0.0,
                    batch_first=True,
                )
                self.head = torch.nn.Sequential(
                    torch.nn.LayerNorm(hidden),
                    torch.nn.Linear(hidden, 1),
                )

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.head(out[:, -1, :])

        net = Net()
        net.to(device)
        return net
