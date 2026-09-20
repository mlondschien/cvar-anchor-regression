import numpy as np
import pytest
import scipy.optimize
from ivmodels.models.anchor_regression import AnchorRegression

from cvar_anchor_regression import CVaRAnchorRegression, GroupDRO


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

    # ||M_Z r||^2 / n is the mean squared error less the mean of the squared means.
    return (
        residuals @ residuals / len(y) - weights @ means**2 + gamma / alpha_cvar * tail
    )


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
    """As above, but the anchor is discrete."""
    environment, X, y = simulate(n_environments=n_environments)
    one_hot = np.eye(n_environments)[environment]

    # min_b ||y - Xb||^2 + (gamma - 1) ||P_Z (y - Xb)||^2, with the intercept folded
    # into both designs, exactly as the estimator does.
    intercept = np.ones((len(y), 1))
    design = np.hstack([intercept, X])
    anchor = np.hstack([intercept, one_hot])
    # pinv, as one-hot columns plus an intercept are rank-deficient by construction.
    projection = anchor @ np.linalg.pinv(anchor)
    exact = np.linalg.solve(
        design.T @ design + (gamma - 1) * design.T @ projection @ design,
        design.T @ y + (gamma - 1) * design.T @ projection @ y,
    )

    cvar = CVaRAnchorRegression(gamma=gamma, alpha_cvar=1.0).fit(X, y, environment)

    np.testing.assert_allclose(cvar.coef_, exact[1:], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(cvar.intercept_, exact[0], rtol=1e-5, atol=1e-6)

    reference = AnchorRegression(gamma=gamma).fit(X, y, one_hot)
    np.testing.assert_allclose(cvar.coef_, reference.coef_, rtol=5e-2, atol=5e-3)
    np.testing.assert_allclose(
        cvar.intercept_, reference.intercept_, rtol=5e-2, atol=5e-3
    )


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
    # slightly different points on that flat, and how far apart they stop is a
    # property of the BLAS rather than of the estimator. These tolerances are set to
    # catch a genuine divergence between the two code paths, which would be O(1), and
    # not to pin down where L-BFGS-B halted.
    np.testing.assert_allclose(discrete.coef_, encoded.coef_, rtol=1e-3, atol=1e-4)
    np.testing.assert_allclose(
        discrete.intercept_, encoded.intercept_, rtol=1e-3, atol=1e-4
    )
    # t_ needs the absolute tolerance: at alpha_cvar=1 it is zero, where a relative
    # comparison carries no meaning.
    np.testing.assert_allclose(discrete.t_, encoded.t_, rtol=1e-3, atol=1e-4)


def test_non_contiguous_level_codes():
    """Level codes need not be consecutive, or start at zero."""
    environment, X, y = simulate(n_environments=8)

    contiguous = CVaRAnchorRegression(gamma=5, alpha_cvar=0.25).fit(X, y, environment)
    relabelled = CVaRAnchorRegression(gamma=5, alpha_cvar=0.25).fit(
        X, y, environment * 7 + 1000
    )

    np.testing.assert_allclose(contiguous.coef_, relabelled.coef_)
    np.testing.assert_allclose(contiguous.intercept_, relabelled.intercept_)


GAMMA_ALPHA_CASES = [(1, 0.5), (2, 0.1), (3, 0.3), (5, 0.5)]


@pytest.mark.parametrize("gamma, alpha_cvar", GAMMA_ALPHA_CASES)
def test_minimizes_its_objective(gamma, alpha_cvar):
    """The fit reaches the minimum of the exact, unsmoothed objective."""
    environment, X, y = simulate(n_environments=8)

    model = CVaRAnchorRegression(gamma=gamma, alpha_cvar=alpha_cvar).fit(
        X, y, environment
    )

    class _At:  # a stand-in model, to score a coefficient vector with ``objective``
        def __init__(self, b):
            self.b = b

        def predict(self, X):
            return np.column_stack([np.ones(len(X)), X]) @ self.b

    def score(b):
        return objective(X, y, environment, _At(b), gamma, alpha_cvar)

    start = np.r_[model.intercept_, model.coef_]
    exact = scipy.optimize.minimize(
        score,
        start,
        method="Nelder-Mead",
        options={"xatol": 1e-9, "fatol": 1e-13, "maxiter": 20000},
    ).x

    np.testing.assert_allclose(score(start), score(exact), rtol=1e-5)


def test_equals_group_dro_under_homoscedasticity():
    """At ``gamma=1`` this is the CVaR of the group risks when the noise is equal.

    The two agree only as the within-environment variances stop differing across
    environments, which they do at rate ``1/sqrt(n_e)``, so this needs a larger sample
    than the other tests and a tolerance that would still catch an O(1) divergence.
    """
    environment, X, y = simulate(n=20000, n_environments=8)
    alpha_cvar = 0.5

    anchor = CVaRAnchorRegression(gamma=1, alpha_cvar=alpha_cvar).fit(X, y, environment)
    group_dro = GroupDRO(alpha_cvar=alpha_cvar).fit(X, y, environment)

    np.testing.assert_allclose(anchor.coef_, group_dro.coef_, atol=0.1)
    np.testing.assert_allclose(anchor.intercept_, group_dro.intercept_, atol=0.1)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"gamma": -1}, "gamma must be non-negative"),
        ({"alpha_cvar": 1.5}, r"alpha_cvar must be in \(0, 1\]"),
        ({"alpha_cvar": 0.0}, r"alpha_cvar must be in \(0, 1\]"),
    ],
)
def test_raises(kwargs, message):
    environment, X, y = simulate(n_environments=8)
    with pytest.raises(ValueError, match=message):
        CVaRAnchorRegression(**kwargs).fit(X, y, environment)
