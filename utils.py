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


def soil_resistivity(rho_0, f):
    """
    CIGRE frequency-dependent soil resitivity.

    Parameters
    ----------
    rho_0: float
        Soil resitivity at nominal frequency, (Ohm*m).
    f: float
        Frequency in Hz at whict the soil resitivity
        will be computed.
    
    Returns
    -------
    rho_f: float
        Soil resitivity value a the new frequency, (Ohm*m).
    """
    rho_f = rho_0/(1. + 4.7e-6*rho_0**0.73*f**0.54)
    
    return rho_f


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
