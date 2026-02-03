import numpy as np
import pytest

from trop import TROP_TWFE_average


def test_trop_twfe_average_smoke():
    # Small synthetic panel
    N, T = 8, 20
    treated_periods = 5
    treated_units = [0, 1]

    rng = np.random.default_rng(0)
    Y = rng.normal(size=(N, T))

    # Treatment: treated_units get treated in last treated_periods
    W = np.zeros((N, T))
    W[treated_units, -treated_periods:] = 1.0

    tau = TROP_TWFE_average(
        Y=Y,
        W=W,
        treated_units=treated_units,
        lambda_unit=0.1,
        lambda_time=0.1,
        lambda_nn=np.inf,   # simplest path: no nuclear-norm component
        treated_periods=treated_periods,
    )

    assert np.isfinite(tau), "tau should be a finite scalar"
    assert isinstance(tau, float)


def test_invalid_shapes_raises():
    Y = np.zeros((5, 10))
    W = np.zeros((5, 9))  # mismatched
    with pytest.raises(ValueError):
        TROP_TWFE_average(
            Y=Y,
            W=W,
            treated_units=[0],
            lambda_unit=0.1,
            lambda_time=0.1,
            lambda_nn=np.inf,
            treated_periods=2,
        )
