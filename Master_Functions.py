from scipy.stats import norm
import numpy as np
from DeltaLaplaceFuncs import DeltaLaplace
import pandas as pd


# Inputs:

# Conditioning variable X_data and corresponding response Y_data
# alpha_range: the acceptable range for the alpha parameters. This defaults to [0,1],
#             which allows for positive association only. Change to [-1,1] to allow for both positive and negative.
# use_DL: boolean to determine whether Delta Laplace margins are incorporated. See Wadsworth and Tawn (2019).

def CondExtBivNegLogLik(x, X_data, Y_data, use_DL=True, alpha_range=np.array([0, 1])):
    if not (isinstance(X_data, (np.ndarray, pd.core.series.Series)) and isinstance(Y_data, (np.ndarray,
                                                                                            pd.core.series.Series))):
        raise TypeError("Expected X_data and Y_data to be numpy arrays or panda series")

    if not X_data.shape == Y_data.shape:
        raise AttributeError("X_data and Y_data have different lengths")

    if not use_DL:

        alpha, beta, mu, sigma = x

    else:

        alpha, beta, mu, sigma, delta = x

        if delta <= 0:
            return 1e10

    if not alpha_range[0] <= alpha <= alpha_range[1]:
        return 1e10

    if sigma <= 0 or beta < 0 or beta > 1:
        return 1e10

    if not use_DL:
        dist = norm(loc=mu, scale=sigma)
    else:
        dist = DeltaLaplace(loc=mu, scale=sigma, shape=delta)

    Z = (Y_data - alpha * X_data) / np.power(X_data, beta)
    np.seterr(divide='ignore', invalid='ignore')

    return -sum(dist.logpdf(Z)) + sum(np.log(X_data ** beta))


def plot_bivariate_condExt_fit(X, Y, u, par_ests, probs=np.linspace(0.05, 0.95, 19), plot_type="Model", zoom=True):
    from scipy.stats import norm
    from DeltaLaplaceFuncs import DeltaLaplace
    import matplotlib.pyplot as plt
    from matplotlib import colors

    if not (plot_type == "Model" or "Both" or "Empirical"):
        raise AttributeError("type must be one of Model, Both or Empirical")

    x_seq = np.linspace(u, max(X) + 1, 500)
    cmap = plt.get_cmap('viridis_r', len(probs) + 1)

    # Using empirical quantiles (LEFT)

    par_len = len(par_ests)

    if not isinstance(zoom, bool):
        raise AttributeError("zoom should be boolean")

    if zoom:
        Y = Y[X > u]
        X = X[X > u]

    if par_len == 5:
        alpha, beta, mu, sigma, delta = par_ests
        dist = DeltaLaplace(loc=mu, scale=sigma, shape=delta)
    elif par_len == 4:
        alpha, beta, mu, sigma = par_ests
        dist = norm(loc=mu, scale=sigma)
    else:
        raise AttributeError("par_ests should have length 4 or 5")

    if probs.min() <= 0 or probs.max() >= 1:
        raise AttributeError("probs should be in range(0,1)")

    if plot_type == "Both":
        fig, axs = plt.subplots(1, 2, figsize=(9.5, 6))

        axs[0].scatter(X, Y, alpha=0.5)
        axs[1].scatter(X, Y, alpha=0.5)

        axs[0].set(xlabel="X", ylabel="Y")
        axs[1].set(xlabel="X", ylabel="Y")

        axs[0].set_title("Conditional Empirical Quantiles given X > %.1f " % u)
        axs[1].set_title("Conditional Parametric Quantiles given X > %.1f " % u)

        # Using empirical quantiles (LEFT)
        z_hat = (Y[X > u] - alpha * X[X > u]) / X[X > u] ** beta

        quants = np.quantile(z_hat, probs)

        for k in range(0, len(probs)):
            axs[0].plot(x_seq, alpha * x_seq + quants[k] * (x_seq ** beta), color=cmap.colors[k], linewidth=1)
            # Using empirical quantiles (Right)

            quants = dist.ppf(probs)

        for k in range(0, len(probs)):
            axs[1].plot(x_seq, alpha * x_seq + quants[k] * (x_seq ** beta), color=cmap.colors[k], linewidth=1)

        fig.subplots_adjust(right=1)

    else:
        fig, axs = plt.subplots(1, 1, figsize=(5, 6))

        axs.scatter(X, Y, alpha=0.5)

        axs.set(xlabel="X", ylabel="Y")

        if plot_type == "Model":

            axs.set_title("Conditional Parametric Quantiles given X > %.1f " % u)
            quants = dist.ppf(probs)
        else:

            axs.set_title("Empirical Parametric Quantiles given X > %.1f " % u)
            z_hat = (Y[X > u] - alpha * X[X > u]) / X[X > u] ** beta

            quants = np.quantile(z_hat, probs)

        for k in range(0, len(probs)):
            axs.plot(x_seq, alpha * x_seq + quants[k] * (x_seq ** beta), color=cmap.colors[k], linewidth=1)

    # creating ScalarMappable
    norm = colors.Normalize(vmin=0., vmax=1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.tight_layout()
    if type == "Both":
        fig.colorbar(sm, label="Probability", orientation="horizontal", ax=axs.ravel().tolist(), shrink=0.95)
    else:
        fig.colorbar(sm, label="Probability", orientation="horizontal", shrink=0.95)

    plt.show()
