import numpy as np


class deltaL:

    def __init__(self, mu, sigma, delta):
        self.mu = mu
        self.sigma = sigma
        self.delta = delta

    def logpdf(self, x):
        from scipy.special import gamma

        p = - abs((x - self.mu) / self.sigma) ** self.delta + np.log(self.delta) - np.log(2 * self.sigma) - np.log(
            gamma(1 / self.delta))

        return p

    def pdf(self, x):
        from scipy.special import gamma

        p = - abs((x - self.mu) / self.sigma) ** self.delta + np.log(self.delta) - np.log(2 * self.sigma) - np.log(
            gamma(1 / self.delta))

        return np.exp(p)

    def cdf(self, x):
        from scipy.stats import gamma

        if isinstance(x, int):

            if x < self.mu:

                return 0.5 * gamma(a=1 / self.delta, scale=1).sf(((self.mu - x) / self.sigma) ** self.delta)

            else:

                return 0.5 + 0.5 * gamma(a=1 / self.delta, scale=1).cdf(((x - self.mu) / self.sigma) ** self.delta)

        nx = len(x)
        result = np.zeros(nx)

        result[x < self.mu] = 0.5 * gamma(a=1 / self.delta, scale=1).sf(
            ((self.mu - x[x < self.mu]) / self.sigma) ** self.delta)
        result[x >= self.mu] = 0.5 + 0.5 * gamma(a=1 / self.delta, scale=1).cdf(
            ((x[x >= self.mu] - self.mu) / self.sigma) ** self.delta)

        return result

    def logcdf(self, x):
        from scipy.stats import gamma

        if isinstance(x, int):

            if x < self.mu:

                return 0.5 * gamma(a=1 / self.delta, scale=1).sf(((self.mu - x) / self.sigma) ** self.delta)

            else:

                return 0.5 + 0.5 * gamma(a=1 / self.delta, scale=1).cdf(((x - self.mu) / self.sigma) ** self.delta)

        nx = len(x)
        result = np.zeros(nx)

        result[x < self.mu] = 0.5 * gamma(a=1 / self.delta, scale=1).sf(
            ((self.mu - x[x < self.mu]) / self.sigma) ** self.delta)
        result[x >= self.mu] = 0.5 + 0.5 * gamma(a=1 / self.delta, scale=1).cdf(
            ((x[x >= self.mu] - self.mu) / self.sigma) ** self.delta)

        return result

        def sf(self, x):
            from scipy.stats import gamma

            if isinstance(x, int):

                if x < self.mu:

                    return np.log(
                        1 - 0.5 * gamma(a=1 / self.delta, scale=1).sf(((self.mu - x) / self.sigma) ** self.delta))

                else:

                    return np.log(
                        0.5 - 0.5 * gamma(a=1 / self.delta, scale=1).cdf(((x - self.mu) / self.sigma) ** self.delta))

            nx = len(x)
            result = np.zeros(nx)

            result[x < self.mu] = 1 - 0.5 * gamma(a=1 / self.delta, scale=1).sf(
                ((self.mu - x[x < self.mu]) / self.sigma) ** self.delta)
            result[x >= self.mu] = 0.5 - 0.5 * gamma(a=1 / self.delta, scale=1).cdf(
                ((x[x >= self.mu] - self.mu) / self.sigma) ** self.delta)

            return np.log(result)

    def ppf(self, q):
        from scipy.stats import gamma

        if isinstance(q, int):

            if q < 0.5:

                return self.mu - self.sigma*(gamma(a=1 / self.delta, scale=1).ppf(1 - 2 * q))**(1/self.delta)

            else:

                return self.mu + self.sigma*(gamma(a=1 / self.delta, scale=1).ppf(2 * q - 1))**(1/self.delta)

        nq = len(q)
        result = np.zeros(nq)

        result[q < 0.5] = self.mu - self.sigma*(gamma(a=1 / self.delta, scale=1).ppf(1 - 2 * q[q < 0.5]))**(1/self.delta)
        result[q >= 0.5 ] = self.mu + self.sigma*(gamma(a=1 / self.delta, scale=1).ppf(2 * q[q >= 0.5] - 1))**(1/self.delta)

        return result
