"""Compact deep reconstruction detectors (CPU). Score at time t uses only the window ending at t (causal)."""
import numpy as np, torch, torch.nn as nn
from numpy.lib.stride_tricks import sliding_window_view as swv

import os
SEED = int(os.environ.get('SEED', 0))
torch.manual_seed(SEED); torch.set_num_threads(2)
W = 30            # window length (s)
N_TRAIN = 30000   # training windows sampled from normal data
EPOCHS = 12; BS = 256


def windows(X):
    """(T, d) -> (T, W, d); the first W-1 rows are left-padded with the first row."""
    Xp = np.concatenate([np.repeat(X[:1], W - 1, 0), X]).astype(np.float32)
    return swv(Xp, W, axis=0).transpose(0, 2, 1)


class _Base:
    def _net(self, d): raise NotImplementedError
    def _loss(self, xb): out = self.net(xb); return ((out - xb) ** 2).mean()
    def _err(self, xb): out = self.net(xb); return ((out[:, -1] - xb[:, -1]) ** 2).sum(1)

    def fit(self, X):
        Wn = windows(X); rng = np.random.default_rng(SEED)
        idx = rng.choice(len(Wn), min(N_TRAIN, len(Wn)), replace=False)
        D = torch.from_numpy(np.ascontiguousarray(Wn[idx]))
        self.net = self._net(X.shape[1]); opt = torch.optim.Adam(self.net.parameters(), 1e-3)
        self.net.train()
        for ep in range(EPOCHS):
            perm = torch.randperm(len(D))
            for i in range(0, len(D), BS):
                xb = D[perm[i:i + BS]]; opt.zero_grad(); l = self._loss(xb); l.backward(); opt.step()
        return self

    @torch.no_grad()
    def score(self, X):
        self.net.eval(); Wn = windows(X); out = np.empty(len(Wn), np.float32)
        for i in range(0, len(Wn), 4096):
            out[i:i + 4096] = self._err(torch.from_numpy(np.ascontiguousarray(Wn[i:i + 4096]))).numpy()
        return out


class MLPAE(_Base):
    name = "MLP-AE"
    def _net(self, d):
        class N(nn.Module):
            def __init__(s):
                super().__init__(); s.e = nn.Sequential(nn.Flatten(), nn.Linear(W * d, 256), nn.ReLU(), nn.Linear(256, 32), nn.ReLU())
                s.d = nn.Sequential(nn.Linear(32, 256), nn.ReLU(), nn.Linear(256, W * d))
            def forward(s, x): return s.d(s.e(x)).view(x.shape)
        return N()


class LSTMAE(_Base):
    name = "LSTM-AE"
    def _net(self, d):
        class N(nn.Module):
            def __init__(s):
                super().__init__(); s.enc = nn.LSTM(d, 64, batch_first=True); s.dec = nn.LSTM(64, 64, batch_first=True); s.out = nn.Linear(64, d)
            def forward(s, x):
                _, (h, _) = s.enc(x); z = h[-1].unsqueeze(1).repeat(1, x.shape[1], 1)
                y, _ = s.dec(z); return s.out(y)
        return N()


class TransformerAE(_Base):
    name = "Transformer"
    def _net(self, d):
        class N(nn.Module):
            def __init__(s):
                super().__init__(); s.inp = nn.Linear(d, 64); s.pos = nn.Parameter(torch.randn(1, W, 64) * 0.02)
                s.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(64, 4, 128, 0.1, batch_first=True), 2)
                s.out = nn.Linear(64, d)
            def forward(s, x): return s.out(s.enc(s.inp(x) + s.pos))
        return N()


class USAD(_Base):
    """Audibert et al., KDD 2020: one encoder, two decoders, adversarial two-phase training."""
    name = "USAD"; ALPHA = 0.5; BETA = 0.5
    def _net(self, d):
        z = 32; n = W * d
        class N(nn.Module):
            def __init__(s):
                super().__init__()
                s.E = nn.Sequential(nn.Flatten(), nn.Linear(n, 256), nn.ReLU(), nn.Linear(256, z), nn.ReLU())
                s.D1 = nn.Sequential(nn.Linear(z, 256), nn.ReLU(), nn.Linear(256, n), nn.Sigmoid())
                s.D2 = nn.Sequential(nn.Linear(z, 256), nn.ReLU(), nn.Linear(256, n), nn.Sigmoid())
        return N()

    def fit(self, X):
        Wn = windows(X); idx = np.random.default_rng(SEED).choice(len(Wn), min(N_TRAIN, len(Wn)), replace=False)
        D = torch.from_numpy(np.ascontiguousarray(Wn[idx])); self.net = N = self._net(X.shape[1])
        o1 = torch.optim.Adam(list(N.E.parameters()) + list(N.D1.parameters()), 1e-3)
        o2 = torch.optim.Adam(list(N.E.parameters()) + list(N.D2.parameters()), 1e-3)
        N.train()
        for ep in range(EPOCHS):
            n = ep + 1; perm = torch.randperm(len(D))
            for i in range(0, len(D), BS):
                xb = D[perm[i:i + BS]]; f = xb.flatten(1)
                w1 = N.D1(N.E(xb)); w3 = N.D2(N.E(w1.view(xb.shape)))
                l1 = (1 / n) * ((f - w1) ** 2).mean() + (1 - 1 / n) * ((f - w3) ** 2).mean()
                o1.zero_grad(); l1.backward(); o1.step()
                w1 = N.D1(N.E(xb)).detach(); w2 = N.D2(N.E(xb)); w3 = N.D2(N.E(w1.view(xb.shape)))
                l2 = (1 / n) * ((f - w2) ** 2).mean() - (1 - 1 / n) * ((f - w3) ** 2).mean()
                o2.zero_grad(); l2.backward(); o2.step()
        return self

    def _err(self, xb):
        N = self.net; f = xb.flatten(1); d = xb.shape[2]
        w1 = N.D1(N.E(xb)); w3 = N.D2(N.E(w1.view(xb.shape)))
        e = self.ALPHA * (f - w1) ** 2 + self.BETA * (f - w3) ** 2
        return e[:, -d:].sum(1)


DEEP = [MLPAE, LSTMAE, TransformerAE, USAD]
