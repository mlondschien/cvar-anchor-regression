import numpy as np
import pytest
import scipy.optimize

from cvar_anchor_regression import GroupDRO

from .test_model import simulate


def worst_group_risk(model, X, y, groups):
    residuals = y - model.predict(X)
    counts = np.bincount(groups).astype(float)
    return (np.bincount(groups, weights=residuals**2) / counts).max()


def brute_force(X, y, groups):
    """min_b max_e E_e[(y - Xb)^2], on the epigraph. Only for small problems."""
    design = np.column_stack([np.ones(len(y)), X])
    counts = np.bincount(groups).astype(float)

    def risks(b):
        return np.bincount(groups, weights=(y - design @ b) ** 2) / counts

    start = np.linalg.lstsq(design, y, rcond=None)[0]
    result = scipy.optimize.minimize(
        lambda z: z[-1],
        np.r_[start, risks(start).max() + 1.0],
        method="SLSQP",
        constraints=[{"type": "ineq", "fun": lambda z: z[-1] - risks(z[:-1])}],
        options={"ftol": 1e-12, "maxiter": 500},
    )
    return result.x[:-1]


@pytest.mark.parametrize("n_environments", [4, 12])
def test_matches_brute_force(n_environments):
    """The annealed softmax reaches the same point as an epigraph solve."""
    groups, X, y = simulate(n_environments=n_environments)

    model = GroupDRO().fit(X, y, groups)
    exact = brute_force(X, y, groups)

    # The minimax objective is flat along the tie between the maximizing groups, so
    # compare the worst-group risk, which is what is actually minimized.
    residuals = y - np.column_stack([np.ones(len(y)), X]) @ exact
    counts = np.bincount(groups).astype(float)
    exact_worst = (np.bincount(groups, weights=residuals**2) / counts).max()

    np.testing.assert_allclose(
        worst_group_risk(model, X, y, groups), exact_worst, rtol=1e-4
    )


def test_single_group_equals_least_squares():
    """With one group the maximum is the mean, so this is ordinary least squares."""
    _, X, y = simulate(n_environments=8)

    model = GroupDRO().fit(X, y, np.zeros(len(y), dtype=int))
    exact = np.linalg.lstsq(np.column_stack([np.ones(len(y)), X]), y, rcond=None)[0]

    np.testing.assert_allclose(model.coef_, exact[1:], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(model.intercept_, exact[0], rtol=1e-5, atol=1e-6)


def test_improves_on_least_squares():
    """Group DRO beats least squares on the worst group, by construction."""
    groups, X, y = simulate(n_environments=12)
    design = np.column_stack([np.ones(len(y)), X])

    model = GroupDRO().fit(X, y, groups)
    least_squares = np.linalg.lstsq(design, y, rcond=None)[0]
    counts = np.bincount(groups).astype(float)
    residuals = y - design @ least_squares
    ols_worst = (np.bincount(groups, weights=residuals**2) / counts).max()

    assert worst_group_risk(model, X, y, groups) < ols_worst


def test_group_risks_attribute():
    """``group_risks_`` is the within-group risk, and several groups tie at the max.

    The tie is a property of the minimax optimum, but how tightly it is resolved is
    set by ``tau_min``: the softmax spreads weight over groups within ``tau`` of the
    maximum. Fit at the tight end so the assertion tests the claim, not the default.
    """
    groups, X, y = simulate(n_environments=8)

    model = GroupDRO(tau_min=1e-9).fit(X, y, groups)
    residuals = y - model.predict(X)
    counts = np.bincount(groups).astype(float)

    np.testing.assert_allclose(
        model.group_risks_, np.bincount(groups, weights=residuals**2) / counts
    )
    assert (model.group_risks_ > model.group_risks_.max() - 1e-6).sum() >= 2


def test_non_contiguous_level_codes():
    """Level codes need not be consecutive, or start at zero."""
    groups, X, y = simulate(n_environments=8)

    contiguous = GroupDRO().fit(X, y, groups)
    relabelled = GroupDRO().fit(X, y, groups * 7 + 1000)

    np.testing.assert_allclose(contiguous.coef_, relabelled.coef_)
    np.testing.assert_allclose(contiguous.intercept_, relabelled.intercept_)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"n_tau": 1}, "n_tau must be at least 2"),
        ({"alpha_ridge": -1.0}, "alpha_ridge must be non-negative"),
    ],
)
def test_raises(kwargs, message):
    groups, X, y = simulate(n_environments=8)
    with pytest.raises(ValueError, match=message):
        GroupDRO(**kwargs).fit(X, y, groups)


def test_raises_on_continuous_anchor():
    """Groups must be discrete: there is no worst group for a continuous Z."""
    groups, X, y = simulate(n_environments=8)
    with pytest.raises(ValueError, match="Z must be a 1d array of integers"):
        GroupDRO().fit(X, y, np.eye(8)[groups])


def exact_cvar(risks, weights, alpha):
    """Return the mean of the worst ``alpha`` fraction of the weighted ``risks``.

    The tail generally ends inside a group, so the boundary group enters with the
    fraction of its weight that fits.
    """
    order = np.argsort(risks)[::-1]
    cumulative = np.cumsum(weights[order])
    last = min(int(np.searchsorted(cumulative, alpha - 1e-15)), len(order) - 1)
    boundary = alpha - (cumulative[last - 1] if last else 0.0)
    tail = weights[order[:last]] @ risks[order[:last]] + boundary * risks[order[last]]
    return tail / alpha


def group_cvar(model, X, y, groups, alpha):
    residuals = y - model.predict(X)
    counts = np.bincount(groups).astype(float)
    risks = np.bincount(groups, weights=residuals**2) / counts
    return exact_cvar(risks, counts / len(y), alpha)


def brute_force_cvar(X, y, groups, alpha):
    """min_b CVaR_alpha(q_e(b)) on the Rockafellar-Uryasev epigraph, with slacks."""
    design = np.column_stack([np.ones(len(y)), X])
    counts = np.bincount(groups).astype(float)
    weights = counts / len(y)
    n_features = design.shape[1]

    def risks(b):
        return np.bincount(groups, weights=(y - design @ b) ** 2) / counts

    start = np.linalg.lstsq(design, y, rcond=None)[0]
    result = scipy.optimize.minimize(
        lambda z: z[n_features] + weights @ z[n_features + 1 :] / alpha,
        np.r_[start, 0.0, risks(start)],
        method="SLSQP",
        bounds=[(None, None)] * n_features + [(0.0, None)] * (len(counts) + 1),
        constraints=[
            {
                "type": "ineq",
                "fun": lambda z: z[n_features + 1 :]
                - risks(z[:n_features])
                + z[n_features],
            }
        ],
        options={"ftol": 1e-12, "maxiter": 500},
    )
    return result.x[:n_features]


@pytest.mark.parametrize("alpha_cvar", [0.2, 0.5, 0.8])
def test_cvar_matches_brute_force(alpha_cvar):
    """The annealed softplus reaches the same point as an epigraph solve.

    Fit at the tight end of the annealing schedule, so that the comparison is against
    the smoothing bias of a ``tau`` the caller chose rather than of the default.
    """
    groups, X, y = simulate(n_environments=4)

    model = GroupDRO(alpha_cvar=alpha_cvar, n_tau=6, tau_min=1e-7).fit(X, y, groups)
    exact = brute_force_cvar(X, y, groups, alpha_cvar)

    # As for the maximum, the objective is flat along the tie between the groups in
    # the tail, so compare the objective rather than the coefficients.
    counts = np.bincount(groups).astype(float)
    residuals = y - np.column_stack([np.ones(len(y)), X]) @ exact
    exact_cvar_value = exact_cvar(
        np.bincount(groups, weights=residuals**2) / counts, counts / len(y), alpha_cvar
    )

    np.testing.assert_allclose(
        group_cvar(model, X, y, groups, alpha_cvar), exact_cvar_value, rtol=1e-4
    )


def test_alpha_cvar_one_equals_least_squares():
    """At ``alpha_cvar=1`` the tail is all of the mass, so this is least squares.

    The group risks then enter as ``sum_e w_e q_e``, which is the overall mean squared
    error, as the ``1 / n_e`` in the risk cancels against the ``n_e`` in the weight.
    """
    groups, X, y = simulate(n_environments=8)

    model = GroupDRO(alpha_cvar=1.0).fit(X, y, groups)
    exact = np.linalg.lstsq(np.column_stack([np.ones(len(y)), X]), y, rcond=None)[0]

    np.testing.assert_allclose(model.coef_, exact[1:], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(model.intercept_, exact[0], rtol=1e-5, atol=1e-6)


def test_smallest_alpha_cvar_equals_worst_group():
    """At ``alpha_cvar = min_e n_e / n`` the tail sits inside one group: the maximum."""
    groups, X, y = simulate(n_environments=8)
    alpha_cvar = np.bincount(groups).min() / len(y)

    model = GroupDRO(alpha_cvar=alpha_cvar, n_tau=6, tau_min=1e-7).fit(X, y, groups)
    worst_group = GroupDRO().fit(X, y, groups)

    np.testing.assert_allclose(
        worst_group_risk(model, X, y, groups),
        worst_group_risk(worst_group, X, y, groups),
        rtol=1e-4,
    )


def test_cvar_interpolates():
    """A larger ``alpha_cvar`` trades worst-group risk for overall risk."""
    groups, X, y = simulate(n_environments=8)
    alphas = [0.1, 0.4, 0.7, 1.0]

    models = [GroupDRO(alpha_cvar=alpha).fit(X, y, groups) for alpha in alphas]
    worst = [worst_group_risk(model, X, y, groups) for model in models]
    overall = [np.mean((y - model.predict(X)) ** 2) for model in models]

    assert np.all(np.diff(worst) > 0)
    assert np.all(np.diff(overall) < 0)


def test_t_attribute_is_the_risk_quantile():
    """``t_`` is the threshold of the tail: the (1 - alpha)-quantile of the risks.

    Asserted through the Rockafellar-Uryasev identity rather than by counting the mass
    above ``t_``, because groups tie at the quantile at the optimum, and which side of
    ``t_`` a tied group lands on is then decided by the last bits.
    """
    groups, X, y = simulate(n_environments=8)
    alpha_cvar = 0.25

    model = GroupDRO(alpha_cvar=alpha_cvar, n_tau=6, tau_min=1e-7).fit(X, y, groups)
    weights = np.bincount(groups) / len(y)
    hinge = np.maximum(model.group_risks_ - model.t_, 0.0)

    np.testing.assert_allclose(
        model.t_ + weights @ hinge / alpha_cvar,
        group_cvar(model, X, y, groups, alpha_cvar),
        rtol=1e-6,
    )

    assert GroupDRO().fit(X, y, groups).t_ is None


def test_raises_on_invalid_alpha_cvar():
    groups, X, y = simulate(n_environments=8)
    with pytest.raises(ValueError, match=r"alpha_cvar must be in \(0, 1\] or None"):
        GroupDRO(alpha_cvar=1.5).fit(X, y, groups)
