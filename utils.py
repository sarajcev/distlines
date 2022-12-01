def cdf_from_pdf(pdf_values):
    """
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
    Kernel Density Estimation (KDE) with the `statsmodels` package. Kernel
    density is estimated from the sample data "x_data" on the support defined
    by the "x_grid" points.

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
