"""Doubly constrained gravity model of commuting, calibrated by mean trip length (Hyman's method)."""
import numpy as np


def deterrence(D: np.ndarray, beta: float, form: str = "exp") -> np.ndarray:
    if form == "exp":
        return np.exp(-beta * D)
    if form == "power":
        return D ** (-beta)
    raise ValueError(form)


def furness(O: np.ndarray, Dj: np.ndarray, F: np.ndarray, tol: float = 1e-8, max_iter: int = 20000):
    """Balance T_ij = A_i O_i B_j D_j F_ij so that row sums = O and column sums = Dj.

    Converged when every origin total is matched to a relative error below tol
    (destination totals are matched exactly after each B update).
    """
    if not np.isclose(O.sum(), Dj.sum(), rtol=1e-9):
        raise ValueError("origin and destination totals differ")
    B = np.ones_like(Dj)
    for it in range(1, max_iter + 1):
        A = 1.0 / (F @ (B * Dj))
        B = 1.0 / (F.T @ (A * O))
        if it % 10 == 0:
            row = (F @ (B * Dj)) * A
            if np.max(np.abs(row - 1.0)) < tol:
                break
    else:
        raise RuntimeError("Furness did not converge")
    T = (A * O)[:, None] * (B * Dj)[None, :] * F
    return T, it


def mean_trip_length(T: np.ndarray, D: np.ndarray) -> float:
    return float((T * D).sum() / T.sum())


def predict(O, Dj, D, beta, form="exp"):
    T, _ = furness(O, Dj, deterrence(D, beta, form))
    return T


BRACKETS = {"exp": (0.01, 0.4), "power": (0.1, 3.0)}  # beyond these, F is near-diagonal and balancing stalls


def calibrate(T_obs: np.ndarray, D: np.ndarray, form: str = "exp", tol: float = 1e-8):
    """Find beta so that the modelled mean trip length equals the observed one.

    Mean trip length falls monotonically as beta rises, so a bracketed root finder (Brent) is used.
    An unbracketed secant update (Hyman's method) overshot for the power form (beta 1.0 -> 4.2),
    which made the Furness balancing ill-conditioned; see docs/debug_log.md.
    """
    from scipy.optimize import brentq
    O, Dj = T_obs.sum(1), T_obs.sum(0)
    c_obs = mean_trip_length(T_obs, D)
    history = []

    def gap(beta):
        c = mean_trip_length(predict(O, Dj, D, beta, form), D)
        history.append((beta, c))
        return c - c_obs

    lo, hi = BRACKETS[form]
    if gap(lo) * gap(hi) > 0:
        raise RuntimeError(f"observed mean trip length not bracketed for {form}")
    beta = brentq(gap, lo, hi, xtol=tol)
    return beta, history


def r2(obs: np.ndarray, pred: np.ndarray) -> float:
    return float(1 - ((obs - pred) ** 2).sum() / ((obs - obs.mean()) ** 2).sum())


def srmse(obs: np.ndarray, pred: np.ndarray) -> float:
    """Standardised root mean square error (Knudsen and Fotheringham)."""
    n = obs.size
    return float(np.sqrt(((obs - pred) ** 2).sum() / n) / (obs.sum() / n))
