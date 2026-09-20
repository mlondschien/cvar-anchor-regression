import warnings

import numpy as np
import scipy
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.exceptions import ConvergenceWarning


class CVaRAnchorRegression(RegressorMixin, BaseEstimator):
    """
    Linear regression with a CVaR anchor penalty.

    .. math:: \\hat\\beta(\\gamma, \\alpha) := \\arg\\min_\\beta \\
       \\frac{1}{n} \\| M_Z (y - X \\beta) \\|_2^2
       + \\gamma \\ \\mathrm{CVaR}_\\alpha\\big( (P_Z (y - X \\beta))^2 \\big)
       + \\alpha_\\textrm{ridge} \\|\\beta\\|_2^2

    where :math:`P_Z` is the linear projection onto the span of :math:`Z`,
    :math:`M_Z := I - P_Z`, and :math:`\\mathrm{CVaR}_\\alpha(W)` is the mean of the
    worst :math:`\\alpha` fraction of :math:`W`. ``alpha_cvar=1`` is anchor regression,
    ``alpha_cvar=1`` with ``gamma=1`` is ordinary least squares.

    For a discrete anchor this is
    :math:`\\sum_e w_e \\sigma_e^2(\\beta) + \\gamma \\mathrm{CVaR}_\\alpha(\\mu_e^2(\\beta))`
    over environments :math:`e` of relative size :math:`w_e`: only the squared group
    mean residuals are reweighted, not the group variances. At ``gamma=1`` this is
    :class:`~cvar_anchor_regression.GroupDRO` if the noise is homoscedastic.

    With :math:`s := \\gamma/\\alpha` and :math:`W := (P_Z (y - X \\beta))^2` the
    anchor penalty is

    .. math:: s \\min_t \\{\\alpha t + \\mathbb{E}[(W - t)_+]\\}

    We replace the hinge :math:`(\\cdot)_+` by the softplus
    :math:`\\tau \\log(1 + \\exp(\\cdot/\\tau))` to make the objective smooth. We then
    anneal :math:`\\tau` to zero in ``n_tau`` annealing steps.

    Parameters
    ----------
    gamma: float, optional, default=1
        The anchor regularization parameter. Must be non-negative.
    alpha_cvar: float, optional, default=1
        The CVaR level, in :math:`(0, 1]`: the fraction of the anchor distribution the
        penalty is computed over.
    fit_intercept: bool, optional, default=True
        Whether to fit an intercept. It is added as a column of ones to both ``X`` and
        ``Z``.
    n_tau: int, optional, default=8
        The number of smoothing parameters in the annealing schedule. Must be at least
        two.
    tau_min: float, optional, default=1e-9
        The final smoothing parameter, relative to the scale of the penalty at the
        least-squares (or ridge) solution.
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
    t_: float
        The fitted Rockafellar-Uryasev threshold, an estimate of the
        :math:`(1 - \\alpha)`-quantile of :math:`(P_Z(y - X\\hat\\beta))^2`.
    s_: float
        The scaled penalty strength :math:`\\gamma/\\alpha`.
    n_features_in_: int
        The number of features seen during ``fit``.

    References
    ----------
    .. bibliography::
       :filter: False

       rothenhausler2021anchor
    """

    def __init__(
        self,
        gamma=1,
        alpha_cvar=1,
        fit_intercept=True,
        n_tau=8,
        tau_min=1e-9,
        alpha_ridge=0.0,
        ftol=1e-12,
        gtol=1e-8,
    ):
        self.gamma = gamma
        self.alpha_cvar = alpha_cvar
        self.fit_intercept = fit_intercept
        self.n_tau = n_tau
        self.tau_min = tau_min
        self.alpha_ridge = alpha_ridge
        self.ftol = ftol
        self.gtol = gtol

    def _projected_residuals(self, coef, y_proj, X_proj, Q):
        residual_means = y_proj - X_proj @ coef
        return residual_means if Q is None else Q @ residual_means

    def _loss(self, parameters, tau, XtX, Xty, y_moment, y_proj, X_proj, Q, weights):
        """Return the softplus-smoothed objective and its gradient in ``(coef, t)``."""
        coef, t = parameters[:-1], parameters[-1]
        gram_coef = XtX @ coef
        values = self._projected_residuals(coef, y_proj, X_proj, Q)
        u = (values**2 - t) / tau
        above = scipy.special.expit(u)
        value = (
            coef @ gram_coef
            - 2 * coef @ Xty
            + y_moment
            - weights @ values**2
            + self.s_ * (self.alpha_cvar * t + tau * (weights @ np.logaddexp(0.0, u)))
        )
        penalty = weights * (self.s_ * above - 1.0) * values
        gradient = 2 * (gram_coef - Xty) - 2 * (
            X_proj.T @ (penalty if Q is None else Q.T @ penalty)
        )
        return value, np.append(gradient, self.s_ * (self.alpha_cvar - weights @ above))

    def fit(self, X, y, Z):
        """
        Fit a CVaR anchor regression estimator.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            The training input samples.
        y: array-like of shape (n_samples,)
            The target values.
        Z: array-like of shape (n_samples,) or (n_samples, n_anchors)
            The anchor: a 1d array of integer level codes, or a 2d array of floats.

        Returns
        -------
        self
        """
        if not 0 < self.alpha_cvar <= 1:
            raise ValueError(f"alpha_cvar must be in (0, 1]. Got {self.alpha_cvar}.")
        if self.gamma < 0:
            raise ValueError(f"gamma must be non-negative. Got {self.gamma}.")

        X, y, Z = np.asarray(X, dtype=float), np.asarray(y, dtype=float), np.asarray(Z)
        if len(X) != len(y) or len(Z) != len(y):
            raise ValueError(
                f"X, y and Z must have equal length. Got {X.shape}, {y.shape} and "
                f"{Z.shape}."
            )

        if self.alpha_ridge < 0:
            raise ValueError(
                f"alpha_ridge must be non-negative. Got {self.alpha_ridge}."
            )

        if self.n_tau < 2:
            raise ValueError(f"n_tau must be at least 2. Got {self.n_tau}.")

        n_samples = len(y)
        self.n_features_in_ = X.shape[1]
        self.s_ = self.gamma / self.alpha_cvar

        X_scale, y_scale = X.std(axis=0), y.std()
        X_scale[X_scale == 0.0] = 1.0
        X = X / X_scale

        ridge_weights = np.ones(X.shape[1], dtype=float) * (self.alpha_ridge + 1e-10)

        if self.fit_intercept:
            X = np.hstack([np.ones((n_samples, 1)), X])
            X_scale = np.concatenate([[1.0], X_scale])
            ridge_weights = np.concatenate([[1e-10], ridge_weights])

        n_features = X.shape[1]
        XtX, Xty, y_moment = X.T @ X / n_samples, X.T @ y / n_samples, y @ y / n_samples
        XtX[np.diag_indices_from(XtX)] += ridge_weights
        coef = np.linalg.solve(XtX, Xty)
        y_scale = np.sqrt(max(y_moment - 2 * coef @ Xty + coef @ XtX @ coef, 1e-12))

        Xty, coef, y = Xty / y_scale, coef / y_scale, y / y_scale
        y_moment /= y_scale**2

        # Compute X_proj, y_proj for categorical or continuous anchors. For categorical
        # anchors, we can reduce to len(X_proj) = len(y_proj) = number of categories.
        if Z.ndim == 1 and np.issubdtype(Z.dtype, np.integer):
            codes = np.unique(Z, return_inverse=True)[1]  # map to consecutive integers
            counts = np.bincount(codes).astype(float)
            Q, weights = None, counts / n_samples
            y_proj = np.bincount(codes, weights=y) / counts
            X_proj = (
                np.column_stack(
                    [np.bincount(codes, weights=X[:, j]) for j in range(n_features)]
                )
                / counts[:, None]
            )
        elif Z.ndim == 2 and np.issubdtype(Z.dtype, np.floating):
            # For continuous anchors, we use a QR decomposition to find a basis for the
            # column space of Z. The projection is then P_Z = Q @ Q.T.
            if self.fit_intercept:
                Z = np.hstack([np.ones((n_samples, 1)), Z])
            # This allows for rank-deficient Z.
            factor, upper = scipy.linalg.qr(Z, mode="economic", pivoting=True)[:2]
            rank = int((np.abs(np.diag(upper)) > 1e-10 * abs(upper[0, 0])).sum())
            Q = np.ascontiguousarray(factor[:, :rank])
            weights = np.full(n_samples, 1.0 / n_samples)
            # X_proj and y_proj are here in the basis of Q.
            y_proj, X_proj = Q.T @ y, Q.T @ X
        else:
            raise ValueError(
                "Z must be a 1d array of integers or a 2d array of floats. Got shape "
                f"{Z.shape} and dtype {Z.dtype}."
            )

        if self.alpha_cvar == 1:
            scaled = X_proj.T * (weights if Q is None else 1.0 / n_samples)
            parameters = np.append(
                np.linalg.solve(
                    XtX + (self.gamma - 1) * scaled @ X_proj,
                    Xty + (self.gamma - 1) * scaled @ y_proj,
                ),
                0.0,
            )
        else:
            # Warm start with the least-squares solution, computed above.
            parameters = np.append(coef, 0)
            # The threshold is bounded below by zero as the atoms enter squared.
            bounds = [(None, None)] * n_features + [(0.0, None)]
            for tau in np.geomspace(1.0, self.tau_min, self.n_tau):
                result = scipy.optimize.minimize(
                    self._loss,
                    parameters,
                    args=(tau, XtX, Xty, y_moment, y_proj, X_proj, Q, weights),
                    jac=True,
                    method="L-BFGS-B",
                    bounds=bounds,
                    options={"ftol": self.ftol, "gtol": self.gtol},
                )
                parameters = result.x

            if not result.success:
                warnings.warn(
                    f"The optimizer did not converge at tau={tau:.2g}: "
                    f"{result.message}",
                    ConvergenceWarning,
                )

        # Undo the normalisation of X and y. The threshold is in units of y squared.
        coef = parameters[:-1] * y_scale / X_scale
        self.t_ = parameters[-1] * y_scale**2

        if self.fit_intercept:
            self.intercept_, self.coef_ = coef[0], coef[1:]
        else:
            self.intercept_, self.coef_ = 0.0, coef
        return self

    def predict(self, X):
        """
        Predict using the fitted estimator.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            The predicted values.
        """
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_
