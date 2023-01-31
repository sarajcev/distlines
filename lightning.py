"""Generating random lightning-currents and lightning strike distances
for the statistical flashover and overvoltage analysis on distribution
lines.

This approach uses Gaussian Copula and can be seen as an alternative to
the method proposed in the Appendix A of the following paper:
A. Borghetti, C. A. Nucci and M. Paolone, An Improved Procedure for the
Assessment of Overhead Line Indirect Lightning Performance and Its
Comparison with the IEEE Std. 1410 Method, IEEE Transactions on Power
Delivery, Vol. 22, No. 1, 2007, pp. 684-692.
"""
import numpy as np


def lognormal_joint_pdf(x, y, mu1=31.1, sigma1=0.484,
                        mu2=3.83, sigma2=0.55, rhoxy=0.47):
    """
    Joint Log-Normal distribution.

    Joint (conditional) Log-Normal distribution, with correlation
    between statistical variables, for depicting lightning current
    amplitudes and wave-front times. f(Ip, tf) is the probability
    density function (PDF).

    Parameters
    ----------
    mu1: float
        Median value of lightning current amplitudes (kA).
    sigma1: float
        Standard deviation of lightning current amplitudes.
    mu2: float
        Median value of wave-front time of lightning currents (us).
    sigma2: float
        Standard deviation of wave-front time of lightning currents.
    rhoxy: float
        Correlation coefficient between the statistical variables.

    Returns
    -------
    f: float
        Probability density value of the joint Log-N distribution
        f(Ip, tf).

    Notes
    -----
    Defaults for median values and standard deviations of lightning
    current parameters have been taken from the relevant CIGRE/IEEE
    WG recommendations.
    """
    f1 = ((np.log(x) - np.log(mu1))/sigma1)**2
    f2 = 2. * rhoxy * ((np.log(x) - np.log(mu1))/sigma1)\
        *((np.log(y) - np.log(mu2))/sigma2)
    f3 = ((np.log(y) - np.log(mu2))/sigma2)**2
    denominator = 2.*np.pi*x*y*sigma1*sigma2*np.sqrt(1. - rhoxy**2)
    f = np.exp(-(f1 - f2 + f3)/(2.*(1. - rhoxy**2))) / denominator

    # Convert `nan` to numerical values.
    return np.nan_to_num(f)


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
    from scipy import stats

    xi = stats.norm.ppf(u) * stats.norm.ppf(v)
    zeta = stats.norm.ppf(u)**2 + stats.norm.ppf(v)**2
    
    k = 1. / np.sqrt(1. - rhoxy**2)
    c = k * np.exp((2.*rhoxy*xi - rhoxy**2*zeta) / (2.*(1.-rhoxy**2)))

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
    from scipy import stats

    Phi2 = stats.multivariate_normal(mean=[0., 0.], 
                                     cov=[[1., rhoxy], [rhoxy, 1.]])
    C = Phi2.cdf(np.dstack((stats.norm.ppf(u), stats.norm.ppf(v))))

    return C


def lightning_bivariate_from_copula(N, mu1, sigma1, mu2, sigma2, rhoxy):
    """
    Bivariate statistical distribution.

    Generate samples from the bivariate lightning-current statistical
    probability distribution (PDF) using the Gaussian Copula approach.

    Parameters
    ----------
    N: int
        Number of random samples.
    mu1: float
        Median value of lightning current amplitudes (kA).
    sigma1: float
        Standard deviation of lightning current amplitudes.
    mu2: float
        Median value of wave-front time (us) or steepness (kA/us)
        of lightning currents.
    sigma2: float
        Standard deviation of wave-front time or steepness of the
        lightning currents.
    rhoxy: float
        Correlation coefficient between the statistical variables.

    Returns
    -------
    amplitudes: numpy.array
        Array of randomly generated lightning-current amplitudes.
        This is a marginal distribution from the associated bivariate
        probability distribution (with a statistical correlation).
    wavefronts: numpy.array
        Array of randomly generated lightning-current wavefronts,
        in terms of duration or steepness. This is a marginal distri-
        bution from the associated bivariate probability distribution
        (with a statistical correlation).
     """
    from scipy import stats

    # Bivariate PDF lightning-current statistical distribution.
    # Generate bivariate Gaussian Copula.
    u, v = copula_gauss_bivariate(N, rhoxy)
    # Marginal distributions.
    wavefronts = stats.lognorm.ppf(u, sigma2, scale=mu2)
    amplitudes = stats.lognorm.ppf(v, sigma1, scale=mu1)
    
    return amplitudes, wavefronts


def lightning_bivariate_choice_from_copula(
        N, choice=1, wavefront='duration', show_plot=False):
    """
    Bivariate statistical distribution.

    Generate samples from the bivariate lightning-current statistical
    probability distribution (PDF) using the Gaussian Copula approach.

    Parameters
    ----------
    N: int
        Number of random samples.
    choice: int
        Choice of lightning-current parameter values. There are four
        different sets of parameters gathered from different literature.
        First set (choice=1) is recommended by CIGRE WG. Other sets have
        been employed by various authors in different papers, and they
        may serve as alternatives.
    wavefront: str
        Parameter that defines the kind of lightning-current dataset
        that is being analyzed, in terms of the wavefront. Two choices
        have been provided:
        - duration:  lightning-current wavefront duration,
        - steepness: lightning-current wavefront steepness.
    show_plot: bool
        Indicator True/False for plotting the bivariate probability
        distribution.

    Returns
    -------
    amplitudes: numpy.array
        Array of randomly generated lightning-current amplitudes.
        This is a marginal distribution from the associated bivariate
        probability distribution (with a statistical correlation).
    wavefronts: numpy.array
        Array of randomly generated lightning-current wavefronts, in
        terms of duration or steepness, depending on the `choice`
        parameter. This is a marginal distribution from the associated
        bivariate probability distribution (with a statistical correlation).

    Raises
    ------
    NotImplementedError

    References
    ----------
    [1] CIGRE, Lightning Parameters for Engineering Applications,
        Brochure 549, CIGRE, Paris, France, 2013, Working Group C4.407.
    [2] Juan A. Martinez-Velasco, Power System Transients: Parameter
        Determination, CRC Press, Boca Raton (FL), 2010.
    """
    import seaborn as sns
    from scipy import stats

    # Defining different possible lightning-current parameter sets
    # ------------------------------------------------------------
    # muI:    median current amplitude (kA)
    # sigmaI: standard deviation of current amplitude
    # muT:    median of wave-front duration (us)
    # sigmaT: standard deviation of wave-front duration
    # rhoT:    correlation coefficient between amplitude and duration
    # muS:    median of wave-front steepness (kA/us)
    # sigmaS: standard deviation of wave-front steepness
    # rhoS:   correlation coefficient between amplitude and steepness
    if choice == 1:  # ORIGINAL SET
        # Amplitude
        muI = 31.1
        sigmaI = 0.484
        # Wavefront duration
        muT = 3.83
        sigmaT = 0.55
        rhoT = 0.47
        # Wavefront steepness
        muS = 24.3
        sigmaS = 0.6
        rhoS = 0.38
    elif choice == 2:
        # Amplitude
        muI = 34.
        sigmaI = 0.74
        # Wavefront duration
        muT = 2.
        sigmaT = 0.494
        rhoT = 0.47
        # Wavefront steepness
        muS = 24.3
        sigmaS = 0.6
        rhoS = 0.38
    elif choice == 3:
        # Amplitude
        muI = 30.1
        sigmaI = 0.76
        # Wavefront duration
        muT = 3.83
        sigmaT = 0.55
        rhoT = 0.47
        # Wavefront steepness
        muS = 24.3
        sigmaS = 0.6
        rhoS = 0.38
    elif choice == 4:  # Martinez-Velasco
        # Amplitude
        muI = 34.
        sigmaI = 0.74
        # Wavefront duration
        muT = 2.
        sigmaT = 0.494
        rhoT = 0.47
        # Wavefront steepness
        muS = 14.0
        sigmaS = 0.55
        rhoS = 0.36
    else:
        raise NotImplementedError(f'Choice = {choice} is not recognized.')

    # Bivariate PDF lightning-current statistical distribution
    if wavefront == 'duration':
        amplitudes, wavefronts = lightning_bivariate_from_copula(
            N, muI, sigmaI, muT, sigmaT, rhoT
        )
    elif wavefront == 'steepness':
        amplitudes, wavefronts = lightning_bivariate_from_copula(
            N, muI, sigmaI, muS, sigmaS, rhoS
        )
    else:
        raise NotImplementedError(
            f'Wavefront definition: {wavefront} is not recognized.')

    if show_plot:
        # Plot the bivariate distribution's PDF.
        sp = stats.spearmanr(wavefronts, amplitudes)[0]
        g = sns.jointplot(x=wavefronts, y=amplitudes, height=6, kind='scatter',
                          s=20, space=0.1, alpha=0.6)
        if wavefront == 'duration':
            g.set_axis_labels(xlabel='Wave-front time (us)',
                              ylabel='Amplitude (kA)')
            g.fig.suptitle('Amplitude vs wave-front time')
        elif wavefront == 'steepness':
            g.set_axis_labels(xlabel='Wave-front steepness (kA/us)',
                              ylabel='Amplitude (kA)')
            g.fig.suptitle('Amplitude vs wave-front steepness')
        g.ax_joint.text(0.5, 0.95,
                        'Spearman '+r'$\rho = $'+'{:.2f}'.format(sp),
                        transform=g.ax_joint.transAxes, size='small')
        plt.tight_layout()
        plt.show()

    return amplitudes, wavefronts


def copula_gauss_trivariate(N, rhoxy):
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
    """
    from scipy import stats

    # Correlation structure of the Copula.
    mean = [0, 0, 0]
    cov = [[1, rhoxy, 0], [rhoxy, 1, 0], [0, 0, 1]]
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


def lightning_current_trivariate_from_copula(
        N, muI=31.1, sigmaI=0.484,
        muTf=3.83, sigmaTf=0.55, rhoT=0.47,
        muTh=77.5, sigmaTh=0.58):
    """
    Trivariate statistical distribution of lightning currents.

    Generate samples from the trivariate lightning-current statistical
    probability distribution: f(Ip, tf, th) using the Gaussian Copula
    approach.

    Parameters
    ----------
    N: int
        Number of random samples.
    muI, sigmaI: floats
        Median value and standard deviation of amplitudes (kA).
    muTf, sigmaTf: floats
        Median value and standard deviation of wave-front times (us).
    rhoT: float
        Statistical correlation between amplitudes and wave-front times.
    muTh, sigmaTh: floats
        Median value and standard deviation of wave-tail half-times (us).

    Returns
    -------
    amplitudes, wavefronts, wavetails: 1d-arrays
        Random samples of lightning-currents.

    Notes
    -----
    Lightning-current amplitudes, wave-front times, and wave-tail
    half-times are each individually statistically distributed
    according to the Log-Normal distribution. However, there is a
    positive statistical correlation between amplitudes and wave-front
    times which needs to be accounted for. Hence, this is why a tri-
    variate copula is used, with a covariance matrix which accounts
    exactly for this statistical correlation.
    """
    from scipy import stats

    # Gaussian Copula
    u, v, w = copula_gauss_trivariate(N, rhoT)
    # Marginal distributions
    wavefronts = stats.lognorm.ppf(u, sigmaTf, scale=muTf)
    amplitudes = stats.lognorm.ppf(v, sigmaI, scale=muI)
    wavetails = stats.lognorm.ppf(w, sigmaTh, scale=muTh)
    
    return amplitudes, wavefronts, wavetails


def lightning_distance_trivariate_from_copula(
        N, muI=31.1, sigmaI=0.484,
        muTf=3.83, sigmaTf=0.55, rhoT=0.47,
        xmin=0., xmax=500.):
    """
    Trivariate statistical distribution.

    Generate samples from the trivariate lightning-current statistical
    probability distribution: f(Ip, tf, y) using the Gaussian Copula
    approach.

    Parameters
    ----------
    N: int
        Number of random samples.
    muI, sigmaI: floats
        Median value and standard deviation of amplitudes (kA).
    muTf, sigmaTf: floats
        Median value and standard deviation of wave-front times (us).
    rhoT: float
        Statistical correlation between amplitudes and wave-front
        times.
    xmin, xmax: floats
        Min. and max. distance of the lightning stroke from the
        distribution line (m). These are limits of the uniform
        distribution.

    Returns
    -------
    amplitudes, wavefronts, distances: 1d-arrays
        Random samples of lightning-currents.

    Notes
    -----
    Distances as a random variable is independent from the other two
    random variables that define lightning-current parameters
    (amplitude and wave-front time). Furthermore, there is a statis-
    tical correlation between the amplitude and wave-front time,
    which is taken into account through the covariance matrix of the
    Gaussian Copula.
    """
    from scipy import stats

    # Gaussian Copula
    u, v, w = copula_gauss_trivariate(N, rhoT)
    # Marginal distributions
    wavefronts = stats.lognorm.ppf(u, sigmaTf, scale=muTf)
    amplitudes = stats.lognorm.ppf(v, sigmaI, scale=muI)
    distances = stats.uniform.ppf(w, loc=xmin, scale=xmax-xmin)
    
    return amplitudes, wavefronts, distances


if __name__ == "__main__":
    """ Showcase aspects of the library. """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats
    import utils

    # Figure style using matplotlib
    plt.style.use('ggplot')

    # Lightning bivariate PDF from the Gaussian Copula.
    x = np.linspace(0, 15, 100, endpoint=False)
    y = np.linspace(0, 115, 100, endpoint=False)
    x[0] = 0.001; y[0] = 0.001  # avoid floating-point error
    x, y = np.meshgrid(x, y)
    # Lightning wavefront time:
    mu1 = 3.83
    sigma1 = 0.55
    F1 = stats.lognorm.cdf(x, sigma1, scale=mu1)
    f1 = stats.lognorm.pdf(x, sigma1, scale=mu1)
    # Lightning amplitude:
    mu2 = 31.1
    sigma2 = 0.484
    F2 = stats.lognorm.cdf(y, sigma2, scale=mu2)
    f2 = stats.lognorm.pdf(y, sigma2, scale=mu2)
    rhoxy = 0.47
    # f(x,y) = c(F1(x), F2(y))*f1(x)*f2(y)
    f_xy = copula_gauss_bivariate_pdf(F1, F2, rhoxy) * (f1*f2)
    # Plot bivariate PDF of amplitudes and wavefront times.
    fig, ax = plt.subplots(figsize=(5,5))
    cs = ax.contourf(x, y, f_xy, 12, cmap=plt.cm.viridis, alpha=0.75)
    ax.contour(cs, levels=cs.levels, 
               colors=['0.25', '0.5', '0.5', '0.5'],
               linewidths=[1.0, 0.5, 1.0, 0.5])
    ax.set_xlabel('Wavefront (us)', fontweight='bold')
    ax.set_ylabel('Amplitude (kA)', fontweight='bold')
    fig.tight_layout()
    plt.show()
    # F(x,y) = C(F1(x), F2(y))
    F_xy = copula_gauss_bivariate_cdf(F1, F2, rhoxy)
    # Plot bivariate CDF of amplitudes and wavefront times.
    fig, ax = plt.subplots(figsize=(5,5))
    cs = ax.contourf(x, y, F_xy, 12, cmap=plt.cm.viridis, alpha=0.75)
    ax.contour(cs, levels=cs.levels, 
               colors=['0.25', '0.5', '0.5', '0.5'],
               linewidths=[1.0, 0.5, 1.0, 0.5])
    ax.set_xlabel('Wavefront (us)', fontweight='bold')
    ax.set_ylabel('Amplitude (kA)', fontweight='bold')
    fig.tight_layout()
    plt.show()

    # Number of random samples:
    N = 1000
    # Plot random sample from the bivariate gaussian copula.
    u, v = copula_gauss_bivariate(N, rhoxy)
    # Scatter plot of Gaussian copula with histograms
    # of the marginal distributions: U, V.
    g = sns.jointplot(x=u, y=v, height=6, kind='scatter', s=20,
                      space=0.1, alpha=0.6)
    g.set_axis_labels(xlabel='u', ylabel='v')
    sp = stats.spearmanr(u, v)[0]
    g.ax_joint.text(0.5, 0.95, 'Spearman '+r'$\rho = $'+'{:.2f}'.format(sp),
                    transform=g.ax_joint.transAxes, size='small')
    g.ax_joint.set_xlim(-0.1, 1.1)
    g.ax_joint.set_ylim(-0.1, 1.1)
    g.fig.suptitle('Gaussian Copula')
    plt.tight_layout()
    plt.show()

    # Joint bivariate statistical probability distribution
    # of lightning current ampltudes and wave-front times.
    a, w = lightning_bivariate_choice_from_copula(N, show_plot=False)

    # Trivariate lightning-currents statistical distribution
    a, w, t = lightning_current_trivariate_from_copula(N)
    xx, yy, zz = utils.bivariate_pdf_from_kde_sm(w, a, 'normal_reference')
    # Plot amplitudes vs wave-front times
    sp = stats.spearmanr(w, a)[0]
    g = sns.jointplot(x=w, y=a, height=6, kind='scatter',
                      s=20, space=0.1, alpha=0.6)
    cs = g.ax_joint.contourf(xx, yy, zz, 10, alpha=0.25)  # 10 contours
    csc = g.ax_joint.contour(cs, levels=cs.levels,
                             colors=['0.25', '0.5', '0.5', '0.5'],
                             linewidths=[1.0, 0.5, 1.0, 0.5])
    # g.ax_joint.clabel(csc, cs.levels[::2], inline=1, fontsize=9)
    g.set_axis_labels(xlabel='Wave-front time (us)', ylabel='Amplitude (kA)')
    g.ax_joint.text(0.5, 0.95, 'Spearman '+r'$\rho = $'+'{:.2f}'.format(sp),
                    transform=g.ax_joint.transAxes, size='small')
    g.fig.suptitle('Amplitude vs wave-front time')
    plt.tight_layout()
    plt.show()
    # Plot amplitudes vs wave-tail half-times
    sp = stats.spearmanr(a, t)[0]
    g = sns.jointplot(x=t, y=a, height=6, kind='scatter',
                      s=20, space=0.1, alpha=0.6)
    g.set_axis_labels(xlabel='Wave-tail time (us)', ylabel='Amplitude (kA)')
    g.ax_joint.text(0.5, 0.95, 'Spearman '+r'$\rho = $'+'{:.2f}'.format(sp),
                    transform=g.ax_joint.transAxes, size='small')
    g.fig.suptitle('Amplitude vs wave-tail half-time')
    plt.tight_layout()
    plt.show()

    # Trivariate statistical probability distribution
    a, w, h = lightning_distance_trivariate_from_copula(N)
    # Plot amplitudes vs wave-front times
    sp = stats.spearmanr(w, a)[0]
    g = sns.jointplot(x=w, y=a, height=6, kind='scatter',
                      s=20, space=0.1, alpha=0.6)
    g.set_axis_labels(xlabel='Wave-front time (us)', ylabel='Amplitude (kA)')
    g.ax_joint.text(0.5, 0.95, 'Spearman '+r'$\rho = $'+'{:.2f}'.format(sp),
                    transform=g.ax_joint.transAxes, size='small')
    g.fig.suptitle('Amplitude vs wave-front time')
    plt.tight_layout()
    plt.show()
    # Plot amplitudes vs distances
    sp = stats.spearmanr(a, h)[0]
    g = sns.jointplot(x=h, y=a, height=6, kind='scatter',
                      s=20, space=0.1, alpha=0.6)
    g.set_axis_labels(xlabel='Distance (m)', ylabel='Amplitude (kA)')
    g.ax_joint.text(0.5, 0.95, 'Spearman '+r'$\rho = $'+'{:.2f}'.format(sp),
                    transform=g.ax_joint.transAxes, size='small')
    g.fig.suptitle('Amplitude vs distance')
    plt.tight_layout()
    plt.show()

    # 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(a, w, h, marker='o')
    ax.set_xlabel('Amplitudes')
    ax.set_ylabel('Wavefronts')
    ax.set_zlabel('Distances')
    fig.tight_layout()
    plt.show()
