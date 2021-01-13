from scipy.stats import laplace
from scipy.stats import norm
import numpy as np
from DeltaLaplaceFuncs import DeltaLaplace


# Inputs:

# Conditioning variable X_data and corresponding response Y_data
# alpha_range: the acceptable range for the alpha parameters. This defaults to [0,1],
#             which allows for positive association only. Change to [-1,1] to allow for both positive and negative.
# use_DL: boolean to determine whether Delta Laplace margins are incorporated. See Wadsworth and Tawn (2019).

def CondExtBivNegLogLik(x, X_data, Y_data, use_DL=True, alpha_range=np.array([0, 1])):
    if not (isinstance(X_data, np.ndarray) and isinstance(Y_data, np.ndarray)):
        raise TypeError("Expected X_data and Y_data to be numpy arrays")

    if not X_data.shape == Y_data.shape:
        raise AttributeError("X_data and Y_data have different lengths")

    if use_DL == False:

        alpha, beta, mu, sigma = x

    else:

        alpha, beta, mu, sigma, delta = x

        if delta <= 0:
            return 1e10

    if not alpha_range[0] <= alpha <= alpha_range[1]:
        return 1e10

    if sigma <= 0 or beta < 0 or beta > 1:
        return 1e10

    if use_DL == False:
        dist = norm(loc=mu, scale=sigma)
    else:
        dist = DeltaLaplace(loc=mu, scale=sigma, shape=delta)

    Z = (Y_data - alpha * X_data) / np.power(X_data, beta)

    return -sum(dist.logpdf(Z)) + sum(np.log(X_data ** beta))
