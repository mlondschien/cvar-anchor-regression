import warnings

import numpy as np
import scipy
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.exceptions import ConvergenceWarning


class GroupDRO(RegressorMixin, BaseEstimator):
    """
    Linear regression minimizing the worst group's mean squared error.

    .. math:: \\hat\\beta := \\arg\\min_\\beta \\ \\max_{e} \\
       \\frac{1}{n_e} \\sum_{i : Z_i = e} (y_i - X_i \\beta)^2
       + \\alpha_\\textrm{ridge} \\|\\beta\\|_2^2

    This is group distributionally robust optimization. Unlike
    :class:`~cvar_anchor_regression.CVaRAnchorRegression`, which penalizes the tail of
    the squared group *means* :math:`\\mu_e^2`, this penalizes the tail of the group
    *risks*, and so cannot distinguish risk driven by a shift from risk driven by
    noise: a group with large :math:`\\sigma_e^2` and no shift dominates the maximum.

    The maximum is replaced by the smooth
    :math:`\\tau \\log \\sum_e \\exp(q_e / \\tau)`, where :math:`q_e` is the risk in
    group :math:`e`. We then anneal :math:`\\tau` to zero in ``n_tau`` steps.

    Setting ``alpha_cvar`` replaces the maximum by the mean of the worst
    ``alpha_cvar`` fraction of the group risks,

    .. math:: \\hat\\beta := \\arg\\min_\\beta \\ \\mathrm{CVaR}_\\alpha(q_e(\\beta)),
       \\qquad q_e(\\beta) = \\frac{1}{n_e} \\sum_{i : Z_i = e} (y_i - X_i \\beta)^2,

    where each group enters the risk distribution with weight :math:`w_e = n_e / n`.
    The two ends of the range are familiar: ``alpha_cvar=1`` averages over all of the
    mass and so is ordinary least squares, since :math:`\\sum_e w_e q_e` is the overall
    mean squared error, while any :math:`\\alpha \\le \\min_e w_e` puts the whole tail
    inside the single worst group and so is the maximum above. Intermediate values
    interpolate, and are the point of the parameter: the maximum is set by one group
    and moves discontinuously with it, whereas a CVaR at, say, ``alpha_cvar=0.2``
    averages the worst fifth of the mass and is far less sensitive to a single small,
    noisy group.

    With ``alpha_cvar`` set we optimize the Rockafellar-Uryasev form
    :math:`\\min_t \\{t + \\alpha^{-1} \\sum_e w_e (q_e - t)_+\\}` jointly in
    :math:`(\\beta, t)`, replacing the hinge by the softplus
    :math:`\\tau \\log(1 + \\exp(\\cdot / \\tau))` and annealing :math:`\\tau` as above.

    When the noise is homoscedastic, :math:`q_e = \\mu_e^2 + \\sigma^2(\\beta)` with
    :math:`\\sigma^2` free of :math:`e`, and this coincides with
    :class:`~cvar_anchor_regression.CVaRAnchorRegression` at ``gamma=1``. That
    estimator is the better choice when the noise is not homoscedastic, as it puts
    only :math:`\\mu_e^2` in the tail.

    Unlike :class:`~cvar_anchor_regression.CVaRAnchorRegression` this does *not*
    reduce to sufficient statistics. The group risks need within-group second
    moments :math:`q_e(\\beta) = \\beta' G_e \\beta - 2 c_e'\\beta + s_e`, and storing
    the :math:`G_e` costs :math:`O(E p^2)`. We evaluate them by streaming the
    residuals instead, which is :math:`O(n p)` per iteration and independent of the
    number of groups, but does not become independent of :math:`n` the way the anchor
    penalty does. Expect fits at :math:`n \\sim 10^6, p \\sim 10^2` to take minutes
    rather than seconds.

    Many small groups make the problem harder as well as slower: the maximum is then
    attained by a single group, the gradient comes from that group's rows alone, and
    the iteration zigzags. Raising ``tau_min`` stops short of the hard maximum and is
    the effective remedy. Raising ``alpha_cvar`` is the other one, and changes the
    estimand rather than only how precisely it is reached.

    Parameters
    ----------
    fit_intercept: bool, optional, default=True
        Whether to fit an intercept. It is added as a column of ones to ``X``.
    alpha_cvar: float or None, optional, default=None
        The CVaR level, in :math:`(0, 1]`. ``None`` minimizes the maximum group risk,
        which is what any :math:`\\alpha \\le \\min_e n_e / n` also gives, but reached
        by smoothing the maximum directly rather than through a threshold.
    n_tau: int, optional, default=4
        The number of smoothing parameters in the annealing schedule. Must be at least
        two.
    tau_min: float, optional, default=1e-5
        The final smoothing parameter, relative to the variance of ``y``. The default
        is far above the ``1e-9`` used for the anchor penalty: annealing further costs
        several times the iterations and does not move the solution, as the maximizing
        groups have separated by then. Lower it only to confirm a fit; raising it to
        ``1e-3`` is several times faster again at a ``1e-3`` relative change in the
        worst-group risk.
    alpha_ridge: float, optional, default=0
        Ridge penalty :math:`\\alpha_\\textrm{ridge}` on the coefficients.
    ftol: float, optional, default=1e-12
        Passed to ``scipy.optimize.minimize``.
    gtol: float, optional, default=1e-8
        Passed to ``scipy.optimize.minimize``.

    Attributes
    ----------
    coef_: np.ndarray of shape (n_features,)
        The estimated coefficients.
    intercept_: float
        The estimated intercept, or 0 if ``fit_intercept=False``.
    group_risks_: np.ndarray of shape (n_groups,)
        The mean squared error within each group at the solution, in the order of
        ``np.unique(Z)``. Several groups are typically tied at the maximum.
    t_: float or None
        The fitted Rockafellar-Uryasev threshold, an estimate of the
        :math:`(1 - \\alpha)`-quantile of the group risks. ``None`` if
        ``alpha_cvar`` is ``None``.
    n_features_in_: int
        The number of features seen during ``fit``.
    """

    def __init__(
        self,
        fit_intercept=True,
        alpha_cvar=None,
        n_tau=4,
        tau_min=1e-5,
        alpha_ridge=0.0,
        ftol=1e-12,
        gtol=1e-8,
    ):
        self.fit_intercept = fit_intercept
        self.alpha_cvar = alpha_cvar
        self.n_tau = n_tau
        self.tau_min = tau_min
        self.alpha_ridge = alpha_ridge
        self.ftol = ftol
        self.gtol = gtol

    def _loss_max(self, coef, tau, X, y, codes, counts):
        """Return the softmax-smoothed worst-group risk and its gradient in ``coef``."""
        residuals = y - X @ coef
        risks = np.bincount(codes, weights=residuals**2) / counts
        # softmax(risks / tau) are the derivatives of the smoothed maximum, so the
        # chain rule gives the gradient without ever forming a per-group moment.
        share = scipy.special.softmax(risks / tau)
        weighted = share[codes] / counts[codes]
        value = tau * scipy.special.logsumexp(risks / tau)
        return value, -2 * (X.T @ (weighted * residuals))

    def _loss_cvar(self, parameters, tau, X, y, codes, counts):
        """Return the softplus-smoothed CVaR of the group risks and its gradient."""
        coef, t = parameters[:-1], parameters[-1]
        residuals = y - X @ coef
        risks = np.bincount(codes, weights=residuals**2) / counts
        weights = counts / len(y)
        u = (risks - t) / tau
        above = scipy.special.expit(u)
        value = t + weights @ (tau * np.logaddexp(0.0, u)) / self.alpha_cvar
        # d q_e / d beta carries a 1 / n_e that cancels against w_e = n_e / n, so the
        # per-sample weight is the group's softplus derivative alone.
        gradient = -2 / (self.alpha_cvar * len(y)) * (X.T @ (above[codes] * residuals))
        return value, np.append(gradient, 1.0 - weights @ above / self.alpha_cvar)

    def fit(self, X, y, Z):
        """
        Fit a group DRO estimator.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            The training input samples.
        y: array-like of shape (n_samples,)
            The target values.
        Z: array-like of shape (n_samples,)
            The group of each sample, as integer level codes. They need be neither
            consecutive nor start at zero.

        Returns
        -------
        self
        """
        X, y, Z = np.asarray(X, dtype=float), np.asarray(y, dtype=float), np.asarray(Z)
        if len(X) != len(y) or len(Z) != len(y):
            raise ValueError(
                f"X, y and Z must have equal length. Got {X.shape}, {y.shape} and "
                f"{Z.shape}."
            )
        if Z.ndim != 1 or not np.issubdtype(Z.dtype, np.integer):
            raise ValueError(
                f"Z must be a 1d array of integers. Got shape {Z.shape} and dtype "
                f"{Z.dtype}."
            )
        if self.alpha_ridge < 0:
            raise ValueError(
                f"alpha_ridge must be non-negative. Got {self.alpha_ridge}."
            )
        if self.n_tau < 2:
            raise ValueError(f"n_tau must be at least 2. Got {self.n_tau}.")
        if self.alpha_cvar is not None and not 0 < self.alpha_cvar <= 1:
            raise ValueError(
                f"alpha_cvar must be in (0, 1] or None. Got {self.alpha_cvar}."
            )

        n_samples = len(y)
        self.n_features_in_ = X.shape[1]
        codes = np.unique(Z, return_inverse=True)[1]  # map to consecutive integers
        counts = np.bincount(codes).astype(float)

        # As for CVaRAnchorRegression, standardize so that tau is on a fixed scale.
        X_scale, y_scale = X.std(axis=0), y.std()
        X_scale[X_scale == 0.0] = 1.0
        X, y = X / X_scale, y / y_scale

        ridge_weights = np.ones(X.shape[1], dtype=float) * (self.alpha_ridge + 1e-10)
        if self.fit_intercept:
            X = np.hstack([np.ones((n_samples, 1)), X])
            X_scale = np.concatenate([[1.0], X_scale])
            ridge_weights = np.concatenate([[1e-10], ridge_weights])

        # Warm start at the least-squares solution, which is the tau -> infinity limit
        # up to the group weighting.
        XtX = X.T @ X / n_samples
        XtX[np.diag_indices_from(XtX)] += ridge_weights
        parameters = np.linalg.solve(XtX, X.T @ y / n_samples)

        if self.alpha_cvar is None:
            loss, bounds = self._loss_max, None
        else:
            loss = self._loss_cvar
            # The threshold is bounded below by zero, as the risks are means of squares.
            bounds = [(None, None)] * X.shape[1] + [(0.0, None)]
            parameters = np.append(parameters, 0.0)

        for tau in np.geomspace(1.0, self.tau_min, self.n_tau):
            result = scipy.optimize.minimize(
                loss,
                parameters,
                args=(tau, X, y, codes, counts),
                jac=True,
                method="L-BFGS-B",
                bounds=bounds,
                options={"ftol": self.ftol, "gtol": self.gtol},
            )
            parameters = result.x

        if not result.success:
            warnings.warn(
                f"The optimizer did not converge at tau={tau:.2g}: {result.message}",
                ConvergenceWarning,
            )

        if self.alpha_cvar is None:
            coef, self.t_ = parameters, None
        else:
            # The threshold is in units of y squared, as are the risks.
            coef, self.t_ = parameters[:-1], parameters[-1] * y_scale**2

        residuals = y - X @ coef
        self.group_risks_ = (
            np.bincount(codes, weights=residuals**2) / counts * y_scale**2
        )

        coef = coef * y_scale / X_scale
        if self.fit_intercept:
            self.intercept_, self.coef_ = coef[0], coef[1:]
        else:
            self.intercept_, self.coef_ = 0.0, coef

        return self

    def predict(self, X):
        """
        Predict using the fitted linear model.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            The samples to predict for.

        Returns
        -------
        np.ndarray of shape (n_samples,)
        """
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_
