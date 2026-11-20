"""Evaluation metrics for time-series anomaly detection.

eTaPR is a clean-room reimplementation of Hwang et al. (2022) following the
reference eTaPR_pkg v22.6.1 logic with delta = 0 (no ambiguous tail), so the
overlap score reduces to the overlap length.
"""
import numpy as np


def ranges(y):
    """Return list of (start, end) inclusive index pairs where y == 1."""
    y = np.asarray(y).astype(np.int8)
    d = np.diff(np.r_[0, y, 0])
    s = np.where(d == 1)[0]
    e = np.where(d == -1)[0] - 1
    return list(zip(s, e))


def _overlap_matrix(A, P):
    M = np.zeros((len(A), len(P)))
    if not A or not P:
        return M
    ps = np.array([p[0] for p in P]); pe = np.array([p[1] for p in P])
    for i, (a0, a1) in enumerate(A):
        j0 = np.searchsorted(pe, a0)            # first pred ending at/after a0
        j = j0
        while j < len(P) and ps[j] <= a1:
            M[i, j] = min(a1, pe[j]) - max(a0, ps[j]) + 1
            j += 1
    return M


def etapr(y_true, y_pred, theta_p=0.5, theta_r=0.1):
    A = ranges(y_true); P = ranges(y_pred)
    if not A or not P:
        return dict(eTaP=0.0, eTaR=0.0, eTaF1=0.0)
    M = _overlap_matrix(A, P)
    alen = np.array([a1 - a0 + 1 for a0, a1 in A], float)
    plen = np.array([p1 - p0 + 1 for p0, p1 in P], float)
    w = np.sqrt(plen)
    E = M.copy()
    while True:  # iterative pruning
        tar = E.sum(1) / alen
        ea = np.where((tar < theta_r) & (tar > 0))[0]
        E[ea] = 0
        tap = E.sum(0) / plen
        ep = np.where((tap < theta_p) & (tap > 0))[0]
        E[:, ep] = 0
        if len(ea) == 0 and len(ep) == 0:
            break
    r = E.sum(1) / alen
    rd = (r >= theta_r).astype(float)
    rp = np.minimum(r, 1.0)
    eTaR = ((rd + rd * rp) / 2).mean()
    p = E.sum(0) / plen
    pd_ = (p >= theta_p).astype(float)
    eTaP = (w * (pd_ + pd_ * p) / 2).sum() / w.sum()
    f1 = 0.0 if eTaP + eTaR == 0 else 2 * eTaP * eTaR / (eTaP + eTaR)
    return dict(eTaP=eTaP, eTaR=eTaR, eTaF1=f1)


def prf(y, yp):
    tp = int(((y == 1) & (yp == 1)).sum()); fp = int(((y == 0) & (yp == 1)).sum())
    fn = int(((y == 1) & (yp == 0)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def point_adjust(y, yp):
    yp = yp.copy()
    for a0, a1 in ranges(y):
        if yp[a0:a1 + 1].any():
            yp[a0:a1 + 1] = 1
    return yp


def event_stats(y, yp, hz=1.0):
    """Event recall, mean detection delay (s), false-alarm events per day of normal operation."""
    A = ranges(y); P = ranges(yp)
    detected, delays = 0, []
    for a0, a1 in A:
        hit = np.where(yp[a0:a1 + 1] == 1)[0]
        if len(hit):
            detected += 1; delays.append(hit[0] / hz)
    fa = 0
    for p0, p1 in P:
        if not y[p0:p1 + 1].any():
            fa += 1
    normal_days = (y == 0).sum() / hz / 86400
    return dict(event_recall=detected / max(len(A), 1),
                n_events=len(A), detected=detected,
                delay_med=float(np.median(delays)) if delays else np.nan,
                fa_per_day=fa / normal_days if normal_days else np.nan)


def evaluate(y, yp):
    y = np.asarray(y).astype(np.int8); yp = np.asarray(yp).astype(np.int8)
    p, r, f = prf(y, yp)
    pa = point_adjust(y, yp)
    pp, pr, pf = prf(y, pa)
    fpr = float(((yp == 1) & (y == 0)).sum() / max((y == 0).sum(), 1))
    out = dict(P=p, R=r, F1=f, PA_F1=pf, FPR=fpr)
    out.update(etapr(y, yp))
    out.update(event_stats(y, yp))
    return out


if __name__ == "__main__":
    # sanity tests
    y = np.zeros(100, int); y[10:20] = 1; y[50:70] = 1
    assert evaluate(y, y)["F1"] == 1.0 and abs(evaluate(y, y)["eTaF1"] - 1.0) < 1e-9
    yp = np.zeros(100, int); yp[15] = 1
    e = evaluate(y, yp)
    print(e)
    assert e["PA_F1"] > e["F1"]          # point-adjust inflates
    assert e["event_recall"] == 0.5
    print("metric sanity tests passed")
