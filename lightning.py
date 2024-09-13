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


def lightning_current_pdf(x, mu, sigma):
    """
    Lightning current probability distribution.

    Probability density function (PDF) of the Log-Normal statis-
    tical distribution. It can serve for generating random ampli-
    tudes, wave-front times and wave-front steepnesses (with 
    introduction of the appropriate median values and standard 
    deviations).

    Parameters
    ----------
    x: float
        Value of the lightning current parameter at which the
        Log-Normal distribution is to be evaluated.
    mu: float
        Median value of the Log-Normal distribution of lightning
        current. It can be amplitude (kA), wave-front time (us), 
        or wave-front steepness (kA/us).
    sigma: float
        Standard deviation of the Log-Normal distribution of
        lightning current. It can be for the amplitude, wave-front
        time or wave-front steepness.
    
    Returns
    -------
    pdf: float
        Probability density function (PDF) value.
    """
    from numpy import log, exp, sqrt, pi, nan_to_num
    
    denominator = (sqrt(2.*pi)*x*sigma)
    pdf = exp(-(log(x) - log(mu))**2 / (2.*sigma**2)) / denominator
    
    # Convert `nan` to numerical values
    return nan_to_num(pdf)


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
    from numpy import log, exp, sqrt, pi, nan_to_num

    f1 = ((log(x) - log(mu1))/sigma1)**2
    f2 = 2. * rhoxy * ((log(x) - log(mu1))/sigma1)\
        *((log(y) - log(mu2))/sigma2)
    f3 = ((log(y) - log(mu2))/sigma2)**2
    denominator = 2.*pi*x*y*sigma1*sigma2*sqrt(1. - rhoxy**2)
    f = exp(-(f1 - f2 + f3)/(2.*(1. - rhoxy**2))) / denominator

    # Convert `nan` to numerical values.
    return nan_to_num(f)


def amplitude_distance_bivariate_pdf(y, x, *args):
    """
    Bivariate probability distribution.

    Bivariate probability density function of lightning-current
    amplitudes and distances (as independent random variables).
    """
    from numpy import log, exp, sqrt, pi, nan_to_num

    # Unpacking extra arguments
    xmin, xmax = args[0], args[1]
    muI, sigmaI = args[2], args[3]
    # Lightning current amplitudes (log-normal distribution)
    denominator = (sqrt(2.*pi)*y*sigmaI)
    pdfI = exp(-(log(y) - log(muI))**2/(2.*sigmaI**2)) / denominator
    # Distances (uniform distribution)
    pdfD = 1./(xmax - xmin)
    # Joint probability distribution
    pdf = pdfI * pdfD
    
    # Convert `nan` to numerical values
    return nan_to_num(pdf)


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
    from copulas import copula_gauss_bivariate
    
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


def lightning_current_trivariate_from_copula(
        N, muI=31.1, sigmaI=0.484,
        muTf=3.83, sigmaTf=0.55, rhoTf=0.47,
        muTh=77.5, sigmaTh=0.58, rhoTh=0.):
    """
    Trivariate statistical distribution of lightning currents.

    Generate samples from the trivariate lightning-current statistical
    probability distribution: f(Ip, tf, th) using the Gaussian Copula
    approach. Statistical distribution can account not only for the 
    correlation between amplitude and wave-front time but also for the
    separate correlation between the amplitude and wave-tail halt-time.

    Parameters
    ----------
    N: int
        Number of random samples.
    muI, sigmaI: floats
        Median value and standard deviation of amplitude (kA).
    muTf, sigmaTf: floats
        Median value and standard deviation of wave-front time (us).
    rhoTf: float
        Statistical correlation between amplitude and wave-front time.
    muTh, sigmaTh: floats
        Median value and standard deviation of wave-tail half-time (us).
    rhoTh: float, default=0
        Statistical correlation between amplitude and wave-tail half-time.

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
    from copulas import copula_gauss_trivariate

    # Gaussian Copula
    u, v, w = copula_gauss_trivariate(N, rhoTf, rhoTh)
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
        Median value and standard deviation of wavefront times (us).
    rhoT: float
        Statistical correlation between amplitudes and wavefront
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
    from copulas import copula_gauss_trivariate

    # Gaussian Copula.
    u, v, w = copula_gauss_trivariate(N, rhoT)
    # Marginal distributions.
    amplitudes = stats.lognorm.ppf(u, sigmaI, scale=muI)
    wavefronts = stats.lognorm.ppf(v, sigmaTf, scale=muTf)
    distances = stats.uniform.ppf(w, loc=xmin, scale=xmax-xmin)
    
    return amplitudes, wavefronts, distances


def return_stroke_trivariate_from_copula(
        N, muI=31.1, sigmaI=0.484,
        muTf=3.83, sigmaTf=0.55, rhoT=0.47,
        muV=120., sigmaV=20., rhoV=0.):
    """
    Trivariate return-stroke statistical distribution.

    Generate samples from the trivariate lightning-current
    statistical probability distribution: f(Ip, tf, v) using
    the Gaussian Copula approach.

    Generating random variates from the trivariate statistical
    probability distribution of lightning-current amplitudes,
    wavefront times and return-stroke velocities.

    Parameters
    ----------
    N: int
        Number of random samples.
    muI, sigmaI: floats
        Median value and standard deviation of amplitudes (kA).
    muTf, sigmaTf: floats
        Median value and standard deviation of wavefront times (us).
    rhoT: float
        Statistical correlation between amplitudes and wavefront
        times.
    muV, sigmaV: floats
        Median value and standard deviation of return-stroke
        velocities (m/ms).
    rhoV: float, default=0
        Statistical correlation between amplitudes and return-
        stroke velocities.
    
    Returns
    -------
    amplitudes, wavefront times, velocities: 1d-arrays
        Random lightning-current amplitudes, wavefront times
        and return-stroke velocities, respectively.
    
    Notes
    -----
    It has been sugested that there is a positive statistical
    correlation between amplitudes and return-stroke velocities
    -- in addition to the existing correlation between the
    amplitudes and wavefront times -- and this information can
    be captured by an another correlation coefficient.
    
    Return-stroke velocity is in the range, approximately,
    between a third and a half of the speed of light in free
    space, i.e. 100-150 m/us. It has been modelled here with a
    Normal distribution, e.g. N(120,20) m/us.
    """
    from scipy import stats
    from copulas import copula_gauss_trivariate

    # Gaussian Copula.
    u, v, w = copula_gauss_trivariate(N, rhoT, rhoV)
    # Marginal distributions.
    amplitudes = stats.lognorm.ppf(u, sigmaI, scale=muI)
    wavefronts = stats.lognorm.ppf(v, sigmaTf, scale=muTf)
    velocities = stats.norm.ppf(w, loc=muV, scale=sigmaV)

    return amplitudes, wavefronts, velocities


def strokes_per_flash(N):
    """
    Multiple strokes per lightning flash.

    Frequency occurence of multiple strokes per single lightning
    flash, according to a statistical probability derived from
    measurements. Function generates `N` random numbers of strokes 
    per lightning flash in accordance with this custom empirical 
    probability distribution.

    Parameters
    ----------
    N: int
        Number of random samples.
    
    Returns
    -------
    m: list of int
        Multiplicity of lightning strokes per flashes.
    
    Reference
    ---------
        F. Napolitano, F. Tossani, A. Borghetti, S. Lilla and C.A. Nucci,
        Assessment of Energy Absorption by Surge Protective Devices in 
        Low Voltage Lines Exposed to Indirect First and Subsequent 
        Lightning Strokes, 37th International Conference on Lightning 
        Protection, Dresden, Germany, 2024.
    """
    strokes_per_flash = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] # last entry 10+
    p_occurence = [0.45, 0.14, 0.09, 0.08, 0.08, 0.04, 0.03, 0.03, 0.02, 0.04]
    m = np.random.choice(strokes_per_flash, size=N, replace=True, p=p_occurence)

    return m


if __name__ == "__main__":
    """ Showcase aspects of the library. """
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import utils
    
    from scipy import stats

    from copulas import copula_gauss_bivariate
    from copulas import copula_gauss_bivariate_pdf
    from copulas import copula_gauss_bivariate_cdf

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
    g.figure.suptitle('Gaussian Copula')
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
    g.figure.suptitle('Amplitude vs wave-front time')
    plt.tight_layout()
    plt.show()
    # Plot amplitudes vs wave-tail half-times
    sp = stats.spearmanr(a, t)[0]
    g = sns.jointplot(x=t, y=a, height=6, kind='scatter',
                      s=20, space=0.1, alpha=0.6)
    g.set_axis_labels(xlabel='Wave-tail time (us)', ylabel='Amplitude (kA)')
    g.ax_joint.text(0.5, 0.95, 'Spearman '+r'$\rho = $'+'{:.2f}'.format(sp),
                    transform=g.ax_joint.transAxes, size='small')
    g.figure.suptitle('Amplitude vs wave-tail half-time')
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
    g.figure.suptitle('Amplitude vs wave-front time')
    plt.tight_layout()
    plt.show()
    # Plot amplitudes vs distances
    sp = stats.spearmanr(a, h)[0]
    g = sns.jointplot(x=h, y=a, height=6, kind='scatter',
                      s=20, space=0.1, alpha=0.6)
    g.set_axis_labels(xlabel='Distance (m)', ylabel='Amplitude (kA)')
    g.ax_joint.text(0.5, 0.95, 'Spearman '+r'$\rho = $'+'{:.2f}'.format(sp),
                    transform=g.ax_joint.transAxes, size='small')
    g.figure.suptitle('Amplitude vs distance')
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

    # Strokes per flash
    no_strokes_per_flash = strokes_per_flash(N)
    # Plot histogram
    fig, ax = plt.subplots(figsize=(5,3.5))
    ax.hist(no_strokes_per_flash, bins=10, density=True)
    ax.set_xlabel('No. strokes per flash')
    ax.set_ylabel('Probability of occurence')
    fig.tight_layout()
    plt.show()
