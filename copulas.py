def copula_gauss_multivariate(N, mu, cov):
    """
    Multivariate Gaussian Copula.

    Generating random samples from the Multivariate
    Gaussian Copula.
    
    Parameters
    ----------
    N : int
        Number of random samples.
    mu : 1d-array
        Array holding mean values of the multivariate 
        Gaussian distribution.
    cov : 2d-array
        Covariance matrix of the multivariate Gaussian
        distribution.

    Returns
    -------
    samples : list
        List of random samples from the multivariate
        Gaussian Copula.
    """
    from scipy import stats

    Z = stats.multivariate_normal(mean=mu, cov=cov).rvs(size=N)
    U = [stats.norm.cdf(Z[:,i]) for i in range(len(mu))]

    return U


def copula_student_multivariate(N, df, loc, shape):
    """
    Multivariate Student's t-distributed Copula.

    Generating random samples from the Student's
    t-distributed multivariate Copula.

    Parameters
    ----------
    N : int
        Number of random samples.
    df : float
        Degrees of freedom of the Student's t-distribution.
    loc : array_like
        Location of the distribution.
    shape : array_like
        Positive semidefinite matrix of the distribution.
    
    Returns
    -------
    samples : list
        List of random samples from the multivariate
        Student's t-distributed Copula.
    """
    from scipy import stats

    Z = stats.multivariate_t(loc=loc, shape=shape, df=df).rvs(size=N)
    U = [stats.t(df=df).cdf(Z[:,i]) for i in range(len(loc))]

    return U


def copula_gauss_bivariate(N, rhoxy):
    """
    Gaussian bivariate Copula.

    Parameters
    ----------
    N: int
        Number of random samples.
    rhoxy: float
        Statistical correlation between variables x, y of
        the desired non-standard bivariate distribution.

    Returns
    -------
    u, v: 1d-arrays
        Random variables u, v of the bivariate Gaussian Copula.
    """
    from scipy import stats

    # Correlation structure of the Copula.
    mean = [0, 0]
    cov = [[1, rhoxy], [rhoxy, 1]]
    # Generating random data from the bivariate standard normal
    # distribution (with correlation structure).
    Z = stats.multivariate_normal.rvs(mean=mean, cov=cov, size=N)
    # Converting to a bivariate uniform distribution. This is the
    # Gaussian copula.
    U = [stats.norm.cdf(Z[:, 0]), stats.norm.cdf(Z[:, 1])]
    u = U[0]
    v = U[1]
    
    return u, v


def copula_gauss_bivariate_pdf(u, v, rhoxy):
    """
    Gaussian bivariate Copula probability density function.

    Bivariate probability density function (PDF) of any
    distribution f(x,y) can be computed from the Copula
    PDF c(u,v) as follows:

    f(x,y) = c(F1(x), F2(y)) f1(x) f2(y)

    where:
        F1, F2 - marginal cumulative distribution functions (CDF)
        f1, f2 - marginal probability density functions (PDF)
    
    Marginal distributions are independent and can have arbitrary
    (continuous) statistical probability distributions.

    Parameters
    ----------
    u, v: array-like
        Real vactors of support values for the coordinate axis.
    rhoxy: float
        Coefficient of correlation between the variates.
    
    Returns
    -------
    c: array-like
        Values of the PDF c(u,v) evaluated at the support.
    """
    from numpy import sqrt, exp
    from scipy import stats

    xi = stats.norm.ppf(u) * stats.norm.ppf(v)
    zeta = stats.norm.ppf(u)**2 + stats.norm.ppf(v)**2
    
    k = 1. / sqrt(1. - rhoxy**2)
    c = k * exp((2.*rhoxy*xi - rhoxy**2*zeta) / (2.*(1.-rhoxy**2)))

    return c


def copula_gauss_bivariate_cdf(u, v, rhoxy):
    """
    Gaussian bivariate Copula cumulative distribution function.

    Bivariate cumulative distribution function (CDF) of any
    distribution F(x,y) can be computed from the Copula CDF
    C(u,v) as follows:

    F(x,y) = C(F1(x), F2(y))

    where: F1, F2 - marginal cumulative distribution functions,
    (arbitrary and independent).

    Parameters
    ----------
    u, v: array-like
        Real vactors of support values for the coordinate axis.
    rhoxy: float
        Coefficient of correlation between the variates.
    
    Returns
    -------
    C: array-like
        Values of the CDF C(u,v) evaluated at the support.
    """
    from numpy import dstack
    from scipy import stats

    Phi2 = stats.multivariate_normal(mean=[0., 0.], 
                                     cov=[[1., rhoxy], [rhoxy, 1.]])
    C = Phi2.cdf(dstack((stats.norm.ppf(u), stats.norm.ppf(v))))

    return C


def copula_gauss_trivariate(N, rhoxy, rhoxz=0):
    """
    Gaussian Copula.

    Gaussian trivariate Copula. It is used for generating random samples
    for the lightning-current parameters.

    Parameters
    ----------
    N: int
        Number of random samples.
    rhoxy: float
        Statistical correlation between variables x and y
        of the desired non-standard trivariate distribution.
    rhoxz: float, default=0
        Statistical correlation between variables x and z
        of the desired non-standard trivariate distribution.

    Returns
    -------
    u, v, w: 1d-arrays
        Random variables u, v, w of the trivariate Gaussian Copula.

    Notes
    -----
    Statistical correlation exists only between first two variables
    (x, y), which will later depict lightning-current amplitudes and
    wave-front times. Third statistical variable can be wave-tail
    half-time in f(Ip, tf, th) or distance of the lightning strike
    in f(Ip, tf, y), which are both independent random variables that
    are not statistically correlated with aforementioned amplitudes
    nor wave-front times.

    In the case of lightning return-stroke velocity being the third
    variable (z), along with amplitude and wavefront time (x,y),
    there can be correlation between the amplitude and velocity,
    which is here depicted with a second correlation coefficient
    `rhoxz`.
    """
    from scipy import stats

    # Correlation structure of the Copula.
    mean = [0, 0, 0]
    cov = [
        [1, rhoxy, rhoxz],
        [rhoxy, 1, 0],
        [rhoxz, 0, 1]
    ]
    # Generating random data from the bivariate standard normal
    # distribution (with correlation structure).
    Z = stats.multivariate_normal.rvs(mean=mean, cov=cov, size=N)
    # Converting to a bivariate uniform distribution. This is the
    # Gaussian copula.
    U = [
        stats.norm.cdf(Z[:, 0]),
        stats.norm.cdf(Z[:, 1]),
        stats.norm.cdf(Z[:, 2]),
    ]
    u = U[0]
    v = U[1]
    w = U[2]
    
    return u, v, w
