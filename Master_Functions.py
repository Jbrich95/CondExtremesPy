from scipy.stats import laplace
from scipy.stats import norm
import numpy as np
from dlaplace import deltaL
# Inputs:

# Conditioning variable X_data and corresponding response Y_data
# alpha_range: the acceptable range for the alpha parameters. This defaults to [0,1],
#             which allows for positive association only. Change to [-1,1] to allow for both positive and negative.
# use_DL: boolean to determine whether Delta Laplace margins are incorporated. See Wadsworth and Tawn (2019). Currently not accepted

def CondExtBivNegLogLik(x, X_data, Y_data, use_DL=True, alpha_range=[0, 1]):
    if use_DL == False:

        alpha, beta, mu, sigma = x

    else:

        alpha, beta, mu, sigma, delta = x

    if not alpha_range[0] <= alpha <= alpha_range[1]:
        return (1e10)

    if sigma <= 0:
        return (1e10)
    if use_DL == False:
        dist = norm(loc=mu, scale=sigma)
    else:
        dist = deltaL(mu=mu, sigma=sigma, delta=delta)

    Z = (Y_data - alpha * X_data) / X_data ** beta

    return (- sum(np.log(dist.pdf(Z) / X_data ** beta)))



