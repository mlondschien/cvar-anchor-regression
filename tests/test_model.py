import numpy as np
import pytest
from ivmodels.models.anchor_regression import AnchorRegression

from cvar_anchor_regression import CVaRAnchorRegression


def simulate(n=400, mx=4, k=3, n_environments=None, seed=0):
    """
    Simulate a confounded linear model with a shifted anchor.

    If ``n_environments`` is None, the anchor is continuous, of shape ``(n, k)``.
    Else it is a 1d array of ``n_environments`` integer level codes, and each
    environment shifts the mean of ``X``.
    """
    rng = np.random.default_rng(seed)
    confounder = rng.normal(size=n)

    if n_environments is None:
        anchor = rng.normal(size=(n, k)) + 2.0
        shift = anchor @ rng.normal(size=(k, mx))
    else:
        anchor = rng.integers(0, n_environments, n)
        shift = rng.normal(0, 1.5, (n_environments, mx))[anchor]

    X = rng.normal(size=(n, mx)) + shift + confounder[:, None] + 5.0
    y = X @ np.ones(mx) + 2 * confounder + rng.normal(size=n) + 7.0
    return anchor, X, y


def objective(X, y, environment, model, gamma, alpha_cvar):
    """
    Compute the exact, unsmoothed objective at a fitted model.

    Only for a discrete anchor. The estimator minimizes a sequence of smoothed problems,
    so two fits of the same objective agree in the objective value long before they
    agree coefficient by coefficient. Comparing here rather than on ``coef_`` tests the
    objective, not where L-BFGS-B happened to stop.
    """
    residuals = y - model.predict(X)
    counts = np.bincount(environment).astype(float)
    means = np.bincount(environment, weights=residuals) / counts
    weights = counts / len(y)

    # The alpha-tail of means**2, weighted by w_e, splitting the boundary environment.
    order = np.argsort(means**2)[::-1]
    cumulative = np.cumsum(weights[order])
    last = min(int(np.searchsorted(cumulative, alpha_cvar - 1e-15)), len(order) - 1)
    boundary = alpha_cvar - (cumulative[last - 1] if last else 0.0)
    tail = weights[order[:last]] @ means[order[:last]] ** 2
    tail += boundary * means[order[last]] ** 2

    return residuals @ residuals / len(y) + (gamma - 1) / alpha_cvar * tail


@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("gamma", [1, 2, 5, 50])
def test_alpha_cvar_one_equals_anchor_regression(gamma, fit_intercept):
    """At ``alpha_cvar=1`` the tail sum is the mean, so this is anchor regression."""
    Z, X, y = simulate()

    anchor = AnchorRegression(gamma=gamma, fit_intercept=fit_intercept).fit(X, y, Z)
    cvar = CVaRAnchorRegression(
        gamma=gamma, alpha_cvar=1.0, fit_intercept=fit_intercept
    ).fit(X, y, Z)

    # At alpha_cvar=1 the objective is quadratic, but still solved iteratively, so the
    # agreement with the closed-form solution is up to optimizer tolerance.
    np.testing.assert_allclose(cvar.coef_, anchor.coef_, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(cvar.intercept_, anchor.intercept_, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("gamma", [1, 2, 5])
@pytest.mark.parametrize("n_environments", [8, 25])
def test_alpha_cvar_one_equals_anchor_regression_discrete(gamma, n_environments):
    """As above, but the anchor is discrete and passed to ``ivmodels`` as one-hot."""
    environment, X, y = simulate(n_environments=n_environments)
    one_hot = np.eye(n_environments)[environment]

    anchor = AnchorRegression(gamma=gamma).fit(X, y, one_hot)
    cvar = CVaRAnchorRegression(gamma=gamma, alpha_cvar=1.0).fit(X, y, environment)

    np.testing.assert_allclose(cvar.coef_, anchor.coef_, rtol=1e-2, atol=1e-3)
    np.testing.assert_allclose(cvar.intercept_, anchor.intercept_, rtol=1e-2, atol=1e-3)


DISCRETE_CASES = [
    (8, 2, 1.0),
    (8, 5, 0.5),
    (8, 5, 0.25),
    (25, 5, 0.5),
    (25, 5, 0.2),
    (25, 50, 0.1),
]


@pytest.mark.parametrize("n_environments, gamma, alpha_cvar", DISCRETE_CASES)
def test_discrete_anchor_equals_one_hot(n_environments, gamma, alpha_cvar):
    """A discrete anchor and its one-hot encoding span the same subspace."""
    environment, X, y = simulate(n_environments=n_environments)
    one_hot = np.eye(n_environments)[environment]

    discrete = CVaRAnchorRegression(gamma=gamma, alpha_cvar=alpha_cvar).fit(
        X, y, environment
    )
    encoded = CVaRAnchorRegression(gamma=gamma, alpha_cvar=alpha_cvar).fit(
        X, y, one_hot
    )

    np.testing.assert_allclose(
        objective(X, y, environment, discrete, gamma, alpha_cvar),
        objective(X, y, environment, encoded, gamma, alpha_cvar),
        rtol=1e-6,
    )

    # The objective above is the claim under test. The coefficients are a weaker
    # check: the objective is flat near the optimum, so the two paths stop at
    # slightly different points on that flat.
    np.testing.assert_allclose(discrete.coef_, encoded.coef_, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(
        discrete.intercept_, encoded.intercept_, rtol=1e-4, atol=1e-5
    )
    # t_ needs the absolute tolerance: at alpha_cvar=1 it is zero, where a relative
    # comparison carries no meaning.
    np.testing.assert_allclose(discrete.t_, encoded.t_, rtol=1e-4, atol=1e-5)


def test_non_contiguous_level_codes():
    """Level codes need not be consecutive, or start at zero."""
    environment, X, y = simulate(n_environments=8)

    contiguous = CVaRAnchorRegression(gamma=5, alpha_cvar=0.25).fit(X, y, environment)
    relabelled = CVaRAnchorRegression(gamma=5, alpha_cvar=0.25).fit(
        X, y, environment * 7 + 1000
    )

    np.testing.assert_allclose(contiguous.coef_, relabelled.coef_)
    np.testing.assert_allclose(contiguous.intercept_, relabelled.intercept_)
