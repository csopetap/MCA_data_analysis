"""
Peak model (Gaussian + low-energy tail + step + linear background) and the helper
functions derived from it.

Parameter order used everywhere: Lambda, x0, delta, A, beta, S, B, C
"""
import numpy as np
from scipy.special import erfc, erfcx

# FWHM of exp(-((x-x0)/delta)**2) is 2*sqrt(ln 2)*delta
FWHM_FACTOR = 2 * np.sqrt(np.log(2))


def peak_plus_bkg_function(x, Lambda, x0, delta, A, beta, S, B, C):
    return Lambda*(np.exp(-((x-x0)/delta)**2) + A/2.*np.exp((x-x0)/beta)*erfc(delta/2./beta+(x-x0)/delta) + S/2.*erfc((x-x0)/delta)) + B*(x-x0) + C


def peak_function(x, Lambda, x0, delta, A, beta, S, B, C):
    return Lambda*(np.exp(-((x-x0)/delta)**2) + A/2.*np.exp((x-x0)/beta)*erfc(delta/2./beta+(x-x0)/delta))


def bkg_function(x, Lambda, x0, delta, A, beta, S, B, C):
    return Lambda*(S/2.*erfc((x-x0)/delta)) + B*(x-x0) + C

def bkg_slope(x, Lambda, x0, delta, A, beta, S, B, C):
    """Calculates the analytical first derivative (slope) of the background."""
    u = (x - x0) / delta
    exp_term = np.exp(-u**2)
    slope = - (Lambda * S / (delta * np.sqrt(np.pi))) * exp_term + B
    return slope

def bkg_slope_err(x, popt, pcov):
    """
    Calculates the 1-sigma uncertainty of the background slope at point x 
    using the fit parameters (popt) and covariance matrix (pcov).
    """
    eps = 1e-5  # small perturbation for numerical gradient
    # popt order: [Lambda, x0, delta, A, beta, S, B, C]
    
    x_array = np.atleast_1d(x)
    grad = np.zeros((len(popt), x_array.size))
    s0 = bkg_slope(x, *popt)
    
    for i in range(len(popt)):
        popt_eps = popt.copy()
        popt_eps[i] += eps
        s_eps = bkg_slope(x, *popt_eps)
        grad[i, :] = (np.atleast_1d(s_eps) - np.atleast_1d(s0)) / eps
        
    # Variance of the slope: var = J^T * cov * J
    var_slope = np.einsum("ix,ij,jx->x", grad, pcov, grad)
    result = np.sqrt(np.maximum(var_slope, 0))
    return result[0] if np.ndim(x) == 0 else result

def get_peak_area(Lambda, x0, delta, A, beta, S, B, C):
    return Lambda*(delta*np.sqrt(np.pi) + A*beta*np.exp(-(delta/2./beta)**2))


def get_gradient(Lambda, x0, delta, A, beta, S, B, C):
    """Gradient of the peak area with respect to the 8 fit parameters."""
    return np.array([
        (delta*np.sqrt(np.pi) + A*beta*np.exp(-(delta/2./beta)**2)),
        0,
        Lambda*(np.sqrt(np.pi) - A*beta*np.exp(-(delta/2./beta)**2) * (delta/2./(beta**2))),
        Lambda*(beta*np.exp(-(delta/2./beta)**2)),
        Lambda*A*(1+1/2.*(delta/beta)**2)*np.exp(-(delta/2./beta)**2),
        0,
        0,
        0
    ])


def second_der_signif_anal(x, Lambda, x0, delta, A, beta, S, B, C):
    """Analytical second-derivative significance expected from the fitted peak."""
    u = (x - x0) / delta
    a = delta / (2. * beta)          # assumes beta > 0
    z = a + u
    gauss = np.exp(-u**2)
    P = np.exp(-u**2 - a**2)         # = exp((x-x0)/beta) * exp(-z**2)
    # Q = exp((x-x0)/beta) * erfc(z), evaluated without overflow
    Q = np.where(z > 0,
                 P * erfcx(np.maximum(z, 0)),
                 np.exp(np.minimum((x - x0) / beta, 0)) * erfc(np.minimum(z, 0)))
    sqrtpi = np.sqrt(np.pi)

    d2 = Lambda * ((4*u**2 - 2) / delta**2 * gauss
                   + A / 2. * (Q / beta**2 + 4*(u - a) * P / (sqrtpi * delta**2))
                   + S * 2*u * gauss / (sqrtpi * delta**2))

    f = Lambda * (gauss + A / 2. * Q + S / 2. * erfc(u)) + B * (x - x0) + C
    sigma = np.maximum(np.sqrt(6. * np.clip(f, 0, None)), 1.)
    return np.abs(d2) / sigma
