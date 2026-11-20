"""ShiftGuard components.

Fast channel  : any detector applied to first-differenced inputs (invariant to constant operating-point offsets).
Level channel : self-re-baselining robust level check. For each variable j,
                z_j(t) = (x_j(t) - med_j(t)) / max(1.4826 * MAD_j(t), floor_j),
                where med and MAD are computed causally over the trailing window W (default 6 h, sampled every 10 s);
                score(t) = max_j |z_j(t)| over non-binary variables; floor_j is at least 5% of the variable's normal span. A legitimate change of operating point is absorbed into the baseline within W;
                slow attacks (minutes to about an hour) remain visible. No labels and no post-change data are used.
Fusion        : alarm if either channel exceeds its own threshold (each set on normal validation data only).
"""
import numpy as np, pandas as pd

W_SEC = 6 * 3600; STEP = 10; WARMUP = 1800


def _rolling_stats(X):
    """Causal rolling median and MAD on a 10-s grid, forward-filled to 1 Hz."""
    sub = pd.DataFrame(X[::STEP].astype(np.float32))
    w = W_SEC // STEP; mp = WARMUP // STEP
    med = sub.rolling(w, min_periods=mp).median()
    mad = (sub - med).abs().rolling(w, min_periods=mp).median()
    med = np.repeat(med.to_numpy(np.float32), STEP, axis=0)[:len(X)]
    mad = np.repeat(mad.to_numpy(np.float32), STEP, axis=0)[:len(X)]
    return med, mad


class LevelChannel:
    name = "Level"

    def fit(self, train_segments):
        """train_segments: list of contiguous (T, d) arrays of normal data. Learns the per-variable MAD floor."""
        mads = []
        for X in train_segments:
            _, mad = _rolling_stats(X); mads.append(mad[~np.isnan(mad).any(1)])
        M = np.concatenate(mads); A = np.concatenate([t[::5] for t in train_segments])
        span = np.percentile(A, 99.5, 0) - np.percentile(A, 0.5, 0)
        # floor: at least 5% of the variable's normal span, so near-constant variables cannot produce extreme z
        self.floor = np.maximum.reduce([0.5 * np.median(1.4826 * M, 0), 0.05 * span, np.full(A.shape[1], 1e-6)])
        # binary / few-valued status signals carry no level information for this channel
        self.use = np.array([len(np.unique(A[::50, j])) > 10 for j in range(A.shape[1])])
        return self

    def score(self, X):
        X = X[:, self.use].astype(np.float32); med, mad = _rolling_stats(X)
        fl = self.floor[self.use].astype(np.float32); out = np.empty(len(X), np.float32)
        for i in range(0, len(X), 100000):                    # chunked to bound memory
            z = np.abs(X[i:i + 100000] - med[i:i + 100000]) / np.maximum(1.4826 * mad[i:i + 100000], fl)
            z[np.isnan(z)] = 0.0                              # warm-up: no level alarms during the first 30 min of a stream
            out[i:i + 100000] = z.max(1)
        return out
