import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from model import calibrate, deterrence, furness, mean_trip_length, predict, r2  # noqa: E402

rng = np.random.default_rng(0)


def synthetic(n=15, beta=0.08):
    xy = rng.uniform(0, 100, (n, 2))
    D = np.sqrt(((xy[:, None] - xy[None]) ** 2).sum(-1))
    np.fill_diagonal(D, 3.0)
    O = rng.uniform(1e3, 1e4, n)
    Dj = rng.uniform(1e3, 1e4, n)
    Dj *= O.sum() / Dj.sum()
    return O, Dj, D, predict(O, Dj, D, beta)


def test_furness_meets_both_constraints():
    O, Dj, D, _ = synthetic()
    T, _ = furness(O, Dj, deterrence(D, 0.05))
    assert np.allclose(T.sum(1), O, rtol=1e-6)
    assert np.allclose(T.sum(0), Dj, rtol=1e-6)


def test_furness_rejects_unequal_totals():
    O, Dj, D, _ = synthetic()
    with pytest.raises(ValueError):
        furness(O, Dj * 1.1, deterrence(D, 0.05))


def test_calibration_recovers_known_beta():
    O, Dj, D, T = synthetic(beta=0.08)
    beta, _ = calibrate(T, D)
    assert beta == pytest.approx(0.08, rel=1e-3)
    assert r2(T, predict(O, Dj, D, beta)) > 0.9999


def test_higher_beta_shortens_trips():
    O, Dj, D, _ = synthetic()
    assert mean_trip_length(predict(O, Dj, D, 0.12), D) < mean_trip_length(predict(O, Dj, D, 0.04), D)


def test_cutting_one_pair_cost_raises_that_flow_and_keeps_totals():
    O, Dj, D, _ = synthetic()
    base = predict(O, Dj, D, 0.08)
    D2 = D.copy()
    D2[0, 1] *= 0.8
    D2[1, 0] *= 0.8
    scen = predict(O, Dj, D2, 0.08)
    assert scen[0, 1] > base[0, 1] and scen[1, 0] > base[1, 0]
    assert np.isclose(scen.sum(), base.sum())
    assert np.allclose(scen.sum(1), O, rtol=1e-6)
