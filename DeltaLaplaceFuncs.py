import numpy as np
import numbers
from scipy.stats import norm
from scipy.stats import multivariate_normal as mvn
from scipy.special import gamma as gam_func
from scipy.stats import gamma

class DeltaLaplace:
    """Iterator for looping over a sequence backwards."""

    def __init__(self, loc, scale, shape):

        self.loc = loc

        # Scale sigma by k
        k = np.sqrt(gam_func(1 / shape) / gam_func(3 / shape))
        self.scale = scale * k
        self.shape = shape

    def logpdf(self, x):

        mu = self.loc
        sigma = self.scale
        delta = self.shape
        ld = -abs((x - mu) / sigma) ** delta + np.log(delta) - np.log(2 * sigma) - np.log(gam_func(1 / delta))

        return ld

    def pdf(self, x):

        mu = self.loc
        sigma = self.scale
        delta = self.shape
        ld = -abs((x - mu) / sigma) ** delta + np.log(delta) - np.log(2 * sigma) - np.log(gam_func(1 / delta))

        return np.exp(ld)

    def cdf(self, x):

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

    
    
class MultiDeltaLaplace:

    def __init__(self, locs, cov, shapes):

        self.locs = locs
        self.cov = cov
        self.shapes = shapes
       
    def logpdf(self, x):
        
        mus = self.locs
        deltas = self.shapes
        sigmas = np.sqrt(np.diagonal(self.cov))
        Sigma = self.cov
        
        #Transform from DeltaLaplace margins to normal
        distsDL = list(map(DeltaLaplace, mus,sigmas,deltas))
        vals_sfDL=list(map(lambda distsDL, x:  distsDL.sf(x), distsDL, x))
        distsnorm = list(map(norm, mus,sigmas))
        vals_norm = list(map(lambda distsnorm, x: distsnorm.isf(x), distsnorm, vals_sfDL))
        #Density of multivariate normal
        ld1 = mvn(mus,cov = Sigma).logpdf(vals_norm)
        #Jacobian from transformation
        ld2 = sum(list(map(lambda distsDL, x:  distsDL.logpdf(x), distsDL, x)))- sum( list(map(lambda distsnorm, x: distsnorm.logpdf(x), distsnorm, vals_norm)))

        return ld1 + ld2

    def logpdf(self, x):
        
        mus = self.locs
        deltas = self.shapes
        sigmas = np.sqrt(np.diagonal(self.cov))
        Sigma = self.cov
        
        #Transform from DeltaLaplace margins to normal
        distsDL = list(map(DeltaLaplace, mus,sigmas,deltas))
        vals_sfDL=list(map(lambda distsDL, x:  distsDL.sf(x), distsDL, x))
        distsnorm = list(map(norm, mus,sigmas))
        vals_norm = list(map(lambda distsnorm, x: distsnorm.isf(x), distsnorm, vals_sfDL))
        #Density of multivariate normal
        ld1 = mvn(mus,cov = Sigma).logpdf(vals_norm)
        #Jacobian from transformation
        ld2 = sum(list(map(lambda distsDL, x:  distsDL.logpdf(x), distsDL, x)))- sum( list(map(lambda distsnorm, x: distsnorm.logpdf(x), distsnorm, vals_norm)))

        return np.exp(ld1 + ld2)

    def rvs(self, size=1, random_state=None):
        mus = self.locs
        deltas = self.shapes
        sigmas = np.sqrt(np.diagonal(self.cov))
        Sigma = self.cov
        #Simulate MVN draws
        norm_vals=mvn(mus, cov = Sigma).rvs(size,random_state).transpose()
        #Transform to DL margins
        distsDL = list(map(DeltaLaplace, mus,sigmas,deltas))
        vals_sfDL=list(map(lambda distsDL, x:  distsDL.cdf(x), distsDL, norm_vals))
        
        return vals_sfDL
