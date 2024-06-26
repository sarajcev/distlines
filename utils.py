def cdf_from_pdf(pdf_values):
    """
    Compute CDF from PDF.

    Numerically approximating a cumulative distribution function (CDF)
    from the probability density function (PDF) described by a set of
    points.

    Parameters
    ----------
    pdf_values: array
        PDF values.

    Returns
    -------
    return: array
        CDF values.
    """
    from numpy import sum, cumsum

    cdf_values = cumsum(pdf_values) / sum(pdf_values)
    
    return cdf_values


def icdf_from_pdf(pdf_values, x_values, support):
    """
    Compute ICDF from PDF.

    Numerically approximating inverse cumulative distribution function
    (ICDF) from the probability density function (PDF) described by a
    set of points.

    Parameters
    ----------
    pdf_values: array
        PDF values.
    x_values: array
        x value points.
    support: array
        Array of values at which the ICDF will be computed.

    Returns
    -------
    return: array
        ICDF values computed at the "support" points.
    """
    from scipy.interpolate import interp1d

    cdf_values = cdf_from_pdf(pdf_values)
    inverse_cdf = interp1d(cdf_values, x_values)
    
    return inverse_cdf(support)


def icdf_from_cdf(cdf_values, x_values, support):
    """
    Compute ICDF from CDF.

    Numerically approximating inverse cumulative distribution function
    (ICDF) from the cumulative distribution function (CDF) described by
    a set of points.

    Parameters
    ----------
    cdf_values: array
        CDF values.
    x_values: array
        x value points.
    support: array
        Array of values at which the ICDF will be computed.

    Returns
    -------
    return: array
        ICDF values computed at the "support" points.
    """
    from scipy.interpolate import interp1d

    inverse_cdf = interp1d(cdf_values, x_values)
    
    return inverse_cdf(support)


def pdf_from_kde(x_data, x_grid, bw='scott', kernel='gaussian', **kwargs):
    """
    Compute PDF from KDE using `scikit-learn`.

    Kernel Density Estimation (KDE) with scikit-learn using the optimal
    bandwidth computed by the routines from the statsmodels package.
    Kernel density is estimated from the sample data.

    Parameters
    ----------
    x_data: array
        Array of data points to approximate with a PDF.
    x_grid: array
        Array of grid values (support) for kernel approximation.
    bw: string
        Bandwidth selection method that supports following options:
        "scott", "silverman" (see statsmodels for more info), and
        "search" which performs a grid search with cross-validation.
    kernel: string
        Kernel type which supports options: "gaussian", "tophat",
        "epanechnikov", "exponential", "linear", and "cosine".

    Returns
    -------
    return: array
        Array of KDE values approximated over "x_grid" support.
    """
    import statsmodels.api as sm
    from sklearn.neighbors import KernelDensity
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import LeaveOneOut
    from numpy import exp, logspace

    if bw == 'scott':
        bandwidth = sm.nonparametric.bandwidths.bw_scott(
            x_data, kernel=kernel)
    elif bw == 'silverman':
        bandwidth = sm.nonparametric.bandwidths.bw_silverman(
            x_data, kernel=kernel)
    elif bw == 'search':
        # This grid search can be computationally expensive due to the
        # LeaveOneOut cross-validation; use on small "x_data" sample.
        params = {'bandwidth': logspace(-1, 1, 10)}
        search = GridSearchCV(KernelDensity(kernel=kernel),
                              param_grid=params, cv=LeaveOneOut(),
                              n_jobs=-1)
        search.fit(x_data[:, None])
        best_parameter = search.best_params_
        bandwidth = best_parameter['bandwidth']
    else:
        raise NotImplementedError(
            'Bandwidth method {} not recognized!'.format(bw))

    kde = KernelDensity(bandwidth=bandwidth, kernel=kernel, **kwargs)
    kde.fit(x_data[:, None])  # array-like of shape (n_samples, n_features)
    # score_samples() returns the log-likelihood of the samples
    log_pdf = kde.score_samples(x_grid[:, None])
    pdf_func = exp(log_pdf)
    
    return pdf_func


def pdf_from_kde_sm(x_data, x_grid, **kwargs):
    """
    Compute PDF from KDE using `statsmodels`.

    Kernel Density Estimation (KDE) with the `statsmodels` package.
    Kernel density is estimated from the sample data "x_data" on
    the support defined by the "x_grid" points.

    Parameters
    ----------
    x_data: array
        Array of data points to approximate with a PDF.
    x_grid: array
        Array of grid values (support) for kernel approximation.

    Returns
    -------
    return: array
        Array of KDE values approximated over "x_grid" support.
    """
    import statsmodels.api as sm

    kde = sm.nonparametric.KDEUnivariate(x_data)
    kde.fit(**kwargs)
    pdf_func = kde.evaluate(x_grid)
    
    return pdf_func


def cdf_from_kde_sm(x_data, **kwargs):
    """
    Univariate Kernel Density Estimation.

    Univariate kernel density estimation (KDE) using the
    routines from the `statsmodels` package.

    Parameters
    ----------
    x_data: array
        Array of data points.

    Returns
    -------
    cdf: array
        Cumulative distribution function values from KDE,
        evaluated at the support.
    """
    import statsmodels.api as sm

    kde = sm.nonparametric.KDEUnivariate(x_data)
    kde.fit(**kwargs)

    return kde.cdf


def bivariate_pdf_from_kde_sm(x, y, bw, gridsize=100, cut=3):
    """
    KDE of the bivariate PDF from statsmodels.

    Kernel density estimation (KDE) of the bivariate probability
    distribution function (PDF) by means of the `statsmodels`
    package.

    Parameters
    ----------
    x, y: 1d-arrays
        Arrays of random values from the bivariate statistical
        distribution for which a KDE estimation will be computed.
    bw: str or float
        Bandwidth of the kernel for the KDE. Check the `statsmodels`
        documentation for more information.
    gridsize: int, default=100
        Number of points to use for the support.
    cut: int, default=3
        Factor that determines how far the evaluation grid extends
        past the extreme data points.

    Returns
    -------
    x, y, z: 1d-arrays
        Arrays holding KDE estimates of the bivariate PDF on
        the support mesh of (x,y) coordinate points.
    """
    import numpy as np
    import statsmodels.api as sm

    kde = sm.nonparametric.KDEMultivariate([x, y], "cc", bw)
    x_support = kde_support(x, kde.bw[0], gridsize, cut)
    y_support = kde_support(y, kde.bw[1], gridsize, cut)
    xx, yy = np.meshgrid(x_support, y_support)
    zz = kde.pdf([xx.ravel(), yy.ravel()]).reshape(xx.shape)
    
    return xx, yy, zz


def kde_support(data, bw, gridsize=100, cut=3):
    """
    Establish support for a kernel density estimate.

    Parameters
    ----------
    data: 2d-array
        Array holding data for the KDE estimation.
    bw: float
        Kernel bandwidth.
     gridsize: int
        Number of points to use for the support.
    cut: int, default=3
        Factor that determines how far the evaluation grid extends
        past the extreme data points.

    Returns
    -------
    support: array
        Array of values holding the support points.
    """
    import numpy as np

    support_min = min(data) - bw * cut
    support_max = max(data) + bw * cut
    support = np.linspace(support_min, support_max, gridsize)
    
    return support


def korsuncev_ionization(I, s, A, rho, Eo=400.):
    """
    Korsuncev's soil ionization method.

    Korsuncev's soil ionization method for the concentrated
    grounding systems of the TL towers.

    Parameters
    ----------
    I: float
        Lightning current amplitude (kA).
    s: float
        Characteristic distance of the grounding system (m). It
        is a largest distance from the center of the grounding
        system to its furthest point.
    A: float
        Surface are of the grounding system (m**2).
    rho: float
        Soil resistivity at the location of the grounding system,
        (Ohm*m).
    Eo: float, default=400
        Critical electric field in (kV/m) necessary for the onset
        of the soil ionization.

    Returns
    -------
    Ri: float
        Impulse resistance of the grounding system, with soil
        ionization accounted for, (Ohm).
    """
    import numpy as np

    pi_1_0 = 0.4517 + (1/2*np.pi) * np.log(s**2/A)
    pi_2 = (rho * I)/(Eo * s**2)

    if pi_2 <= 5.:
        pi_1 = 0.2965 * np.power(pi_2, -0.2867)
    elif 5. < pi_2 <= 50.:
        pi_1 = 0.4602 * np.power(pi_2, -0.6009)
    elif 50. < pi_2 <= 500.:
        pi_1 = 0.9534 * np.power(pi_2, -0.7536)
    elif 500. < pi_2:
        pi_1 = 1.8862 * np.power(pi_2, -0.8693)
    else:
        raise ValueError(f'Value of "PI_2": {pi_2} is out of range!')

    if pi_1 <= pi_1_0:
        Ri = (rho/s) * pi_1_0
    else:
        Ri = (rho/s) * pi_1

    return Ri


def soil_resistivity(rho_0, f, method='CIGRE'):
    """
    Frequency-dependent soil resitivity.

    Parameters
    ----------
    rho_0: float
        Soil resitivity at nominal frequency, (Ohm*m).
    f: float
        Frequency in Hz at whict the soil resitivity
        will be computed.
    method: str
        Two methods for estimating frequency dependence
        of the soil resitivity have been implemented:
        'CIGRE' and 'Visacro'.
    
    Returns
    -------
    rho_f: float
        Soil resitivity value a the new frequency, (Ohm*m).
    """
    if method == 'CIGRE':
        rho_f = rho_0/(1. + 4.7e-6*rho_0**0.73*f**0.54)
    elif method == 'Visacro':
        rho_f = rho_0 * (1. + (1.2e-6*rho_0**0.73) * ((f - 100.)**0.65))**(-1)
    else:
        raise NotImplementedError(f'Method: {method} is not recognized!')
    
    return rho_f


def two_layer_soil(rho_1, rho_2, d, r):
    """
    Equivalent soil resistivity for a two-layer soil.

    Empirical fit to the solution of elliptic-integral
    potentials (Zaborsky). Actual two-layer soil is 
    approximated by the homogenous soil with an equivalent
    resistivity value.

    Parameters
    ----------
    rho_1: float
        Resistivity of the upper soil layer (Ohm*m).
    rho_2: float
        Resistivity of the lower soil layer (Ohm*m).
    d: float
        Depth of the upper soil layer (m).
    r: float
        Equivalent radius of the grounding grid (m).
    
    Returns
    -------
    rho_eq: float
        Equivalent soil resitivity (Ohm*m).
    
    Notes
    -----
    Grounding grid is considered as a disk-like electrode
    buried just below the surface.
    """
    factor = (rho_2*r) / (rho_1*d)
    
    if rho_1 > rho_2:
        C = 1. / (1.4 + (rho_2/rho_1)**0.8)
    else:
        C = 1. / (1.4 + (rho_2/rho_1)**0.8 + (factor)**0.5)
    
    rho_eq = rho_1 * ((1. + C * factor) / (1. + C * (r/d)))
    
    return rho_eq


def jitter(ax, x, y, s, c, **kwargs):
    """
    Add jitter to the scatter plot.

    Parameters
    ----------
    ax: ax
        Matplotlib axis object.
    x: array
        Array of x-values.
    y: array
        Array of y-values.
    s: int
        Marker/dot size.
    c: string
        Marker/dot color.

    Returns
    -------
    return:
        ax.scatter matplotlib object.
    """
    def random_jitter(arr, std=0.01):
        from numpy.random import randn

        stdev = std * (max(arr) - min(arr))
        return arr + randn(len(arr)) * stdev

    return ax.scatter(random_jitter(x),
                      random_jitter(y),
                      s=s, c=c, **kwargs)


def moving_window(data, window='hamming', N=10):
    """
    Applying a moving window on the signal.

    Parameters
    ----------
    data: np.array
        An 1d array holding the original data.
    window: str, default='hamming'
        Window name from the `scipy.signal` library: 
        `boxcar`, `triang`, `blackman`, `hamming`, 
        `hann`, `bartlett`, `flattop`, `parzen`, 
        `bohman`, `blackmanharris`, `nuttall`, 
        `barthann`, `cosine`, `exponential`, `tukey`,
        `taylor`, `lanczos`, etc.
    N: int, default=10
        Number of points for the window size.

    Returns
    -------
    ma: np.array
        Data array after moving window's application.
    """
    from scipy import signal
    from numpy import sum, convolve

    if isinstance(window, tuple):
        window_name = window[0]
        window_params = window[1:]
        weights = signal.get_window((window_name, window_params), 
                                    N, fftbins=False)
    else:
        weights = signal.get_window(window, N, fftbins=False)
    
    weights /= sum(weights)  # normalization
    ma = convolve(data, weights, mode='same')  # convolution
    
    return ma


def critical_current_fit(x, y, degree=3):
    """
    Polynomial fit of the critical current values.

    A polynomial fit of the form: 
        y = a + b*x + c*x**2 + d*x**3
    is used, in the least-squares sence, for fitting
    the (x,y) data of distances and critical lightning
    currents. Function invokes `linalg.lstsq` from the
    `scipy` library. Polynomial of the second or third
    order are supported by this routine.

    Arguments
    ---------
    x: 1d-array
        Array of distances (x-axis values).
    y: 1d-array
        Array of critical currents (y-axis values).
    degree: int, default=3
        Degree of the polynomial fit. Only second or
        third-order polynomials are supported.
    
    Returns
    -------
    coeffs: 1d-array
        Coefficients of the polynomial fit [a, b, c, d],
        depending on the chosen polynomial `degree`.
    """
    import numpy as np
    from scipy import linalg

    # Prepare the coefficients matrix, depending
    # on the chosen polynomial degree.
    if degree == 2:
        X = np.c_[np.ones_like(x), x, x**2]
    elif degree == 3:
        X = np.c_[np.ones_like(x), x, x**2, x**3]
    else:
        raise NotImplementedError(
            f'Polynomial degree {degree} is not implemented.'
        )

    # Solve the least-squares problem.
    coeffs, resid, rank, s = linalg.lstsq(X, y)

    return coeffs


def poly(x, clp, degree=3):
    """
    Polinomial value from the fitted coefficients.

    Polinomial approximation to the CLP curve from
    the coefficients computed from the least-squares.

    Arguments
    ---------
    x: array
        An 1d array holding the x-values data.
    clp: array-like or tuple
        Coefficients of the polinomial, ordered
        from the lowest to the highest exponent.
    degree: int, default=3
        Degree of the polynomial fit. Only second or
        third-order polynomials are supported.

    Returns
    -------
    y: array
        Polinomial values computed at x values.
    """
    if degree == 2:
        y = clp[0] + clp[1]*x + clp[2]*x**2 
    elif degree == 3:
        y = clp[0] + clp[1]*x + clp[2]*x**2 + clp[3]*x**3
    else:
        raise NotImplementedError(
            f'Polynomial degree {degree} is not supported.'
        )
    
    return y


class DoubleIntegralBoundary():
    """
    Double integral lower boundary function.

    Class for defining a lower boundary `gfun` curve for the
    double integration routine `integrate.dblquad` from the Scipy
    library. This function introduces additional arguments and is
    implemented inside a `__call__` method. Namely, it is not
    possible to directly use a boundary function `gfun` that passes
    additional arguments (see Scipy documentation). This class is
    used in computing the risk of flashover from the curve of
    limiting parameters (CLP), which have been defined by points.
    """
    def __init__(self, x, y):
        """
        x, y: 1d-arrays
        """
        self.x = x
        self.y = y

    def __call__(self, x_new):
        from scipy.interpolate import interp1d
        
        # Linear interpolation of CLP points.
        function = interp1d(self.x, self.y, kind='linear')
        return function(x_new)


def risk_from_clp_points(x, y_clp, mu=31.1, sigma=0.484):
    """
    Computing risk from the CLP curve.

    Computing the risk of flashovers, from lightning interaction
    with overhead distribution lines, by means of the curve of
    limiting parameters (CLP) which has been defined by points.

    Parameters
    ----------
    x: 1d-array
        Array holding x-values (distances) where CLP points
        have been pre-computed.
    clp: 1d-array
        Array holding points on the CLP curve.
    mu: float
        Median value of lightning current amplitudes.
    sigma: float
        Standard deviation of lightning current amplitudes.

    Returns
    -------
    risk: float
        Risk of flashover computed from the curve of limiting
        parameters.
    """
    from numpy import Inf
    from scipy import integrate
    from lightning import amplitude_distance_bivariate_pdf

    arguments = (x[0], x[-1], mu, sigma)
    lower_boundary = DoubleIntegralBoundary(x, y_clp)
    risk, _ = integrate.dblquad(
        amplitude_distance_bivariate_pdf,
        x[0], x[-1],
        lower_boundary,  # gfun: lower boundary function
        lambda y: Inf,   # hfun: upper boundary function
        args=arguments
    )
    return risk


class DoubleIntegralPolyBoundary():
    """
    Double integral lower boundary function.

    Class for defining a lower boundary `gfun` curve for the
    double integration routine `integrate.dblquad` from the Scipy
    library. This function introduces additional arguments and is
    implemented inside a `__call__` method. Namely, it is not
    possible to directly use a boundary function `gfun` that passes
    additional arguments (see Scipy documentation). This class is
    used in computing the risk of flashover from the curve of
    limiting parameters (CLP), which has been defined by the max.
    third-degree polynomial.
    """
    def __init__(self, clp):
        """
        Parameters
        ----------
        clp: list-like or tuple
            Parameters [a, b, c, d] of the max. third-degree 
            polinomial CLP curve: 
            y = a + b*x + c*x**2 + d*x**3
        """
        self.a = clp[0]
        self.b = clp[1]
        self.c = clp[2]
        if len(clp) > 3:
            self.d = clp[3]
        else:
            self.d = 0

    def __call__(self, x):
        """Max. third-degree polynomial."""
        y = self.a + self.b*x + self.c*x**2 + self.d*x**3

        return y


def risk_from_clp(clp, xmin, xmax, mu=31.1, sigma=0.484):
    """
    Computing risk from the CLP curve.

    Computing the risk of flashovers, from lightning interaction
    with overhead distribution lines, by means of the curve of
    limiting parameters (CLP) which has been defined by the
    third-degree polynomial.

    Parameters
    ----------
    clp: array
        Array holding parameters [a, b, c, d] of the third-degree
        polinomial CLP curve: y = a + b*x + c*x**2 + d*x**3.
    xmin, xmax: floats
        Min. and max. limits of the integration domain on the
        x-axis.
    mu: float
        Median value of lightning current amplitudes.
    sigma: float
        Standard deviation of lightning current amplitudes.

    Returns
    -------
    risk: float
        Risk of flashover computed from the curve of limiting
        parameters.
    """
    from numpy import Inf
    from scipy import integrate
    from lightning import amplitude_distance_bivariate_pdf

    arguments = (xmin, xmax, mu, sigma)
    lower_boundary = DoubleIntegralPolyBoundary(clp)
    risk, _ = integrate.dblquad(
        amplitude_distance_bivariate_pdf,
        xmin, xmax,
        lower_boundary,  # gfun: lower boundary function
        lambda y: Inf,   # hfun: upper boundary function
        args=arguments
    )
    return risk
