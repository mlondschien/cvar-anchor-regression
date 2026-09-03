# CVaR anchor regression

Optimize `(1/n)·‖y − Xβ‖² + (γ − 1)·CVaR_α((P_Z(y − Xβ))²)` over `β`. Whereas classical
anchor regression penalizes the *average* squared projected residual, CVaR anchor
regression penalizes the worst `α` fraction of it.