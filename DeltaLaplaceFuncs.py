import numpy as np
import numbers


class DeltaLaplace:
    """Iterator for looping over a sequence backwards."""

    def __init__(self, loc, scale, shape):
        from scipy.special import gamma

        self.loc = loc

        # Scale sigma by k
        k = np.sqrt(gamma(1 / shape) / gamma(3 / shape))
        self.scale = scale * k
        self.shape = shape

    def logpdf(self, x):

        from scipy.special import gamma

        mu = self.loc
        sigma = self.scale
        delta = self.shape
        ld = -abs((x - mu) / sigma) ** delta + np.log(delta) - np.log(2 * sigma) - gamma(1 / delta)

        return ld

    def pdf(self, x):
        from scipy.special import gamma

        mu = self.loc
        sigma = self.scale
        delta = self.shape
        ld = -abs((x - mu) / sigma) ** delta + np.log(delta) - np.log(2 * sigma) - gamma(1 / delta)

        return np.exp(ld)

    def cdf(self, x):
        from scipy.stats import gamma

        mu = self.loc
        sigma = self.scale
        delta = self.shape

        if isinstance(x, numbers.Number):
            if x < mu:
                result = 0.5 * gamma(a=1 / delta, scale=1).sf(((mu - x) / sigma) ** delta)
            else:
                result = 0.5 + 0.5 * gamma(a=1 / delta, scale=1).cdf(((x - mu) / sigma) ** delta)
        else:
            result = np.zeros(len(x))

            result[x < mu] = 0.5 * gamma(a=1 / delta, scale=1).sf(((mu - x[x < mu]) / sigma) ** delta)

            result[x >= mu] = 0.5 + 0.5 * gamma(a=1 / delta, scale=1).cdf(((x[x >= mu] - mu) / sigma) ** delta)

        return result

    def logcdf(self, x):
        from scipy.stats import gamma

        mu = self.loc
        sigma = self.scale
        delta = self.shape

        if isinstance(x, numbers.Number):
            if x < mu:
                result = 0.5 * gamma(a=1 / delta, scale=1).sf(((mu - x) / sigma) ** delta)
            else:
                result = 0.5 + 0.5 * gamma(a=1 / delta, scale=1).cdf(((x - mu) / sigma) ** delta)
        else:
            result = np.zeros(len(x))

            result[x < mu] = 0.5 * gamma(a=1 / delta, scale=1).sf(((mu - x[x < mu]) / sigma) ** delta)

            result[x >= mu] = 0.5 + 0.5 * gamma(a=1 / delta, scale=1).cdf(((x[x >= mu] - mu) / sigma) ** delta)

        return np.log(result)

    def sf(self, x):
        from scipy.stats import gamma

        mu = self.loc
        sigma = self.scale
        delta = self.shape

        if isinstance(x, numbers.Number):
            if x < mu:
                result = 1 - 0.5 * gamma(a=1 / delta, scale=1).sf(((mu - x) / sigma) ** delta)
            else:
                result = 0.5 - 0.5 * gamma(a=1 / delta, scale=1).cdf(((x - mu) / sigma) ** delta)
        else:
            result = np.zeros(len(x))

            result[x < mu] = 1 - 0.5 * gamma(a=1 / delta, scale=1).sf(((mu - x[x < mu]) / sigma) ** delta)

            result[x >= mu] = 0.5 - 0.5 * gamma(a=1 / delta, scale=1).cdf(((x[x >= mu] - mu) / sigma) ** delta)

        return result

    def logsf(self, x):
        from scipy.stats import gamma

        mu = self.loc
        sigma = self.scale
        delta = self.shape

        if isinstance(x, numbers.Number):
            if x < mu:
                result = 1 - 0.5 * gamma(a=1 / delta, scale=1).sf(((mu - x) / sigma) ** delta)
            else:
                result = 0.5 - 0.5 * gamma(a=1 / delta, scale=1).cdf(((x - mu) / sigma) ** delta)
        else:
            result = np.zeros(len(x))
            result[x < mu] = 1 - 0.5 * gamma(a=1 / delta, scale=1).sf(((mu - x[x < mu]) / sigma) ** delta)

            result[x >= mu] = 0.5 - 0.5 * gamma(a=1 / delta, scale=1).cdf(((x[x >= mu] - mu) / sigma) ** delta)

        return np.log(result)

    def ppf(self, q):
        from scipy.stats import gamma

        mu = self.loc
        sigma = self.scale
        delta = self.shape

        if isinstance(q, numbers.Number):
            if q < 0.5:
                result = mu - sigma * (gamma(a=1 / delta, scale=1).ppf(1 - 2 * q)) ** (1 / delta)
            else:
                result = mu + sigma * (gamma(a=1 / delta, scale=1).ppf(2 * q - 1)) ** (1 / delta)

        else:
            result = np.zeros(len(q))

            result[q < 0.5] = mu - sigma * (gamma(a=1 / delta, scale=1).ppf(1 - 2 * q[q < 0.5])) ** (1 / delta)

            result[q >= 0.5] = mu + sigma * (gamma(a=1 / delta, scale=1).ppf(2 * q[q >= 0.5] - 1)) ** (1 / delta)

        return result

    def isf(self, q):
        from scipy.stats import gamma
        q = 1 - q
        result = q

        mu = self.loc
        sigma = self.scale
        delta = self.shape

        if isinstance(q, numbers.Number):
            if q < 0.5:
                result = mu - sigma * (gamma(a=1 / delta, scale=1).ppf(1 - 2 * q)) ** (1 / delta)
            else:
                result = mu + sigma * (gamma(a=1 / delta, scale=1).ppf(2 * q - 1)) ** (1 / delta)
        else:
            result[q < 0.5] = mu - sigma * ((gamma(a=1 / delta, scale=1).ppf(1 - 2 * q[q < 0.5])) ** (1 / delta))

            result[q >= 0.5] = mu + sigma * ((gamma(a=1 / delta, scale=1).ppf(2 * q[q >= 0.5] - 1)) ** (1 / delta))

        return result

    def rvs(self, size=1, random_state=None):
        np.random.seed(random_state)
        return self.ppf( np.random.rand(size))
