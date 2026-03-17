"""
zfit_models.py
==============
Custom probability density functions for zfit.

- z_emg: Exponentially Modified Gaussian (for timing distributions)
- BackscatterPDF: Flat distribution via error functions (for charge distributions)
- zJohnson: Johnson SU distribution (alternative timing model)
"""

import zfit
from zfit import z
import tensorflow
import numpy as np

def compute_chi2_ndf(model, data, size, nbins, xr, n_free_params):
    """
    Compute chi2/ndf from binned comparison of model to data.
    
    ndf = nbins (with events) - n_free_params
    """
    counts, bin_edges = np.histogram(
        zfit.run(data.value()[:, 0]), bins=nbins, range=xr
    )
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    y_exp = model.pdf(bin_centers).numpy() * size / nbins * (xr[1] - xr[0])

    # Only use bins with events (avoid division by zero)
    mask = counts > 0
    chi2 = np.sum((counts[mask] - y_exp[mask])**2 / counts[mask])
    ndf = np.sum(mask) - n_free_params

    return chi2, ndf


class ExponentiallyModifiedGaussian(zfit.pdf.BasePDF):
    """
    Exponentially Modified Gaussian (EMG) PDF.

    Models a Gaussian convolved with an exponential decay, commonly used
    for PMT transit time distributions where the Gaussian core represents
    jitter and the exponential tail represents late pulses.

    Parameters
    ----------
    mu : zfit.Parameter
        Mean of the Gaussian component.
    sigma : zfit.Parameter
        Standard deviation of the Gaussian component.
    lambd : zfit.Parameter
        Rate parameter of the exponential (λ > 0 for right tail).
    """

    def __init__(self, obs, mu, sigma, lambd, extended=None, norm=None, name=None):
        params = {"mu": mu, "sigma": sigma, "lambd": lambd}
        super().__init__(obs=obs, params=params, extended=extended, norm=norm,
                         name=name or "EMG")

    def _unnormalized_pdf(self, x):
        x = z.unstack_x(x)
        mu = self.params["mu"]
        lambd = self.params["lambd"]
        sigma = self.params["sigma"]

        a = (mu + lambd * sigma**2 - x) / (z.sqrt(2.0) * sigma)
        b = 2 * mu + lambd * sigma**2 - 2 * x

        return (lambd / 2) * z.exp((lambd / 2) * b) * tensorflow.math.erfc(a)


class BackscatterPDF(zfit.pdf.BasePDF):
    """
    Inelastic backscattering distribution for PMT charge spectra.

    Models a flat distribution between the pedestal and the SPE peak
    using the difference of two error functions, producing a plateau
    with smooth edges determined by the pedestal and SPE widths.

    Parameters
    ----------
    mu0, sigma0 : zfit.Parameter
        Pedestal mean and width (left edge).
    mu1, sigma1 : zfit.Parameter
        SPE peak mean and width (right edge).
    """

    def __init__(self, obs, mu0, sigma0, mu1, sigma1,
                 extended=None, norm=None, name=None):
        params = {"mu0": mu0, "sigma0": sigma0, "mu1": mu1, "sigma1": sigma1}
        super().__init__(obs=obs, params=params, extended=extended, norm=norm,
                         name=name or "Backscatter")

    def _unnormalized_pdf(self, x):
        x = z.unstack_x(x)
        mu0 = self.params["mu0"]
        sigma0 = self.params["sigma0"]
        mu1 = self.params["mu1"]
        sigma1 = self.params["sigma1"]

        erf1 = tensorflow.math.erf((x - mu0) / (sigma0 * z.sqrt(2.0)))
        erf2 = tensorflow.math.erf((x - mu1) / (sigma1 * z.sqrt(2.0)))

        return erf1 - erf2


class JohnsonSU(zfit.pdf.BasePDF):
    """
    Johnson SU distribution PDF.

    An alternative skewed distribution sometimes useful for timing spectra.

    Parameters
    ----------
    mu : zfit.Parameter
        Location parameter.
    lambd : zfit.Parameter
        Scale parameter.
    gamma : zfit.Parameter
        Skewness parameter.
    delta : zfit.Parameter
        Tail-weight parameter.
    """

    def __init__(self, obs, mu, lambd, gamma, delta,
                 extended=None, norm=None, name=None):
        params = {"mu": mu, "lambd": lambd, "gamma": gamma, "delta": delta}
        super().__init__(obs=obs, params=params, extended=extended, norm=norm,
                         name=name or "JohnsonSU")

    def _unnormalized_pdf(self, x):
        x = z.unstack_x(x)
        mu = self.params["mu"]
        lambd = self.params["lambd"]
        gamma = self.params["gamma"]
        delta = self.params["delta"]

        a = (x - mu) / lambd

        chunk1 = delta / (lambd * z.sqrt(2 * z.constant(3.141592653589793)))
        chunk2 = 1 / z.sqrt(1 + a**2)
        chunk3 = z.exp(-0.5 * (gamma + delta * z.numpy.arcsinh(a))**2)

        return chunk1 * chunk2 * chunk3
