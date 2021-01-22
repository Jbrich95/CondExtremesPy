from scipy.stats import norm
import numpy as np
from DeltaLaplaceFuncs import *
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
    if plot_type == "Both":
        fig.colorbar(sm, label="Probability", orientation="horizontal", ax=axs.ravel().tolist(), shrink=0.95)
    else:
        fig.colorbar(sm, label="Probability", orientation="horizontal", shrink=0.95)

    plt.show()

import time
    
def CondExtMevNegLogLik(x, X_data, Y_data,par_lens,cor_len,alpha_range=np.array([0, 1])):
    p=Y_data.shape[1]
    n=Y_data.shape[0]
    alpha_len,beta_len,mu_len,sig_len,delta_len = par_lens[0],par_lens[1],par_lens[2],par_lens[3],par_lens[4]

    use_DL=True
    if delta_len == 0:
        use_DL= False
    if not(cor_len == 1 or cor_len == 0.5* p*(p-1)):
        raise AttributeError("cor_len accepts inputs of either 1 or p(p-1)/2")
    if not sum(par_lens)==(len(x)-cor_len):
        raise AttributeError("Sum of par_lens not equal to length of input parameters")
    if not use_DL:
        if  any(not (l ==1 or l==p) for l in par_lens[0:4]):
            raise AttributeError("par_lens accepts inputs of either 1 or p = Y_data.shape[1]")
    else:
        if  any(not (l ==1 or l==p) for l in par_lens):
            raise AttributeError("par_lens accepts inputs of either 1 or p = Y_data.shape[1]")
    

    if not use_DL:

        alphas, betas, mus, sigmas = x[0:alpha_len],x[alpha_len:alpha_len+beta_len],x[alpha_len+beta_len:alpha_len+beta_len+mu_len],x[-(sig_len+cor_len):-cor_len]

    else:

        alphas, betas, mus, sigmas, deltas =  x[0:alpha_len],x[alpha_len:alpha_len+beta_len],x[alpha_len+beta_len:alpha_len+beta_len+mu_len],x[alpha_len+beta_len+mu_len:alpha_len+beta_len+mu_len+sig_len], x[-(delta_len+cor_len):-(cor_len)]
        if np.min(deltas) <= 0:
            return 1e10



    if np.min(alphas) < alpha_range[0] or np.max(alphas) < alpha_range[0]:
        return 1e10

    if np.min(sigmas) <= 0 or np.min(betas) < 0 or np.max(betas) > 1:
        return 1e10

    cors = x[-cor_len:]

    Cor=np.zeros(shape=([p,p]))
    for i in range(0,p):
        Cor[i,i] = 1 
    count = 0
    for i in range(1,p):
        for j in range(0,i):
            Cor[i,j] = Cor[j,i] = cors[count]
            count = count +1


    Cov=np.matmul(np.diag(sigmas),np.matmul(Cor,np.diag(sigmas)))
    
    def is_pos_def(x):
        return np.all(np.linalg.eigvals(x) > 0)

    if not is_pos_def(Cov):
        return 1e10
    
    if not use_DL:
        dist = mvn(mean=mus, cov=Cov)
    else:
        dist = MultiDeltaLaplace(locs=mus, cov=Cov, shapes=deltas)
    np.seterr(divide='ignore', invalid='ignore')   
    numer=(Y_data - np.matmul(np.array([X_data]*p).transpose(),np.diag(alphas)))
    denom = (np.asarray([x**beta for x,beta in zip(np.array([X_data]*p),betas)])).transpose()
  
    Z=numer/denom
    return -np.sum([dist.logpdf(Z[i,]) for i in range(0,n)]) + np.sum(np.log([x**beta for x,beta in zip(np.array([X_data]*p),betas)]))