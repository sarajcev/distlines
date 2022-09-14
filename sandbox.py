# Author: Petar Sarajcev, PhD (petar.sarajcev@fesb.hr)
# University of Split, FESB, Department of Power Engineering, R. Boskovica 32,
# HR21000, Split, Croatia.

from encodings import search_function


def hyper_search_cv(X, y, pipe, params_dict, scoring_method,
                    search_type='Random', n_iterations=100):
    """ Scikit-learn model hyperparameters optimization with GridSearchCV,
    RandomizedSearchCV, and HalvingRandomSearchCV methods.

    Parameters
    ----------
    X: array
        Features matrix.
    y: array
        Labels vector.
    pipe: pipeline
        Pipeline object from the `scikit-learn` library.
    params_dict: dictionary
        Dictionary holding either (a) statistical distributions
        of model's hyperparameters or (b) lists with grid values
        of hyperparameters.
    scoring_method: string
        Scoring method from the `scikit-learn` library for model training.
    search_type: string
        Type of hyperparameters search algorithm. Following three
        values are allowed: 'Random', 'Grid, and 'Halving'.
    n_iterations: int
        Number of iterations for the random search algorithm.

    Returns
    -------
    search: pipeline
        Fitted pipeline object from the `scikit-learn` library.

    Raises
    ------
    NotImplementedError
    """
    import warnings
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.model_selection import GridSearchCV
    from sklearn.experimental import enable_halving_search_cv  # noqa
    from sklearn.model_selection import HalvingRandomSearchCV
    from sklearn.model_selection import StratifiedKFold

    # Experimental HalvingRandomSearchCV is known for raising
    # warnings during fit, which we'll just ignore for now.
    warnings.filterwarnings(action='ignore')

    if search_type == 'Random':
        # Randomized search with k-fold cross-validation
        search = RandomizedSearchCV(estimator=pipe,
                                    param_distributions=params_dict,
                                    cv=StratifiedKFold(n_splits=3),
                                    scoring=scoring_method,
                                    n_iter=n_iterations,
                                    refit=True, n_jobs=-1)

    elif search_type == 'Grid':
        # Grid search with k-fold cross-validation
        search = GridSearchCV(estimator=pipe,
                              param_grid=params_dict,  # grid values!
                              cv=StratifiedKFold(n_splits=3),
                              scoring=scoring_method,
                              refit=True, n_jobs=-1)

    elif search_type == 'Halving':
        # Halving random search with k-fold cross-validation
        # HalvingRandomSearchCV is still considered experimental!
        search = HalvingRandomSearchCV(estimator=pipe,
                                       param_distributions=params_dict,
                                       cv=StratifiedKFold(n_splits=3),
                                       scoring=scoring_method,
                                       refit=True, n_jobs=-1)

    else:
        raise NotImplementedError('Search type "{}" is not recognized!'
                                  .format(search_type))

    search.fit(X, y)
    return search


def train_test_shuffle_split(X_data, y_data, train_size=0.8):
    """
    Stratified shuffle split of data into training and test / validation set.

    Parameters
    ----------
    X_data: DataFrame
        Pandas dataframe holding the matrix of features.
    y_data: Series
        Pandas series holding the targets.
    train_size: float
        Percentage of the dataset that will be reserved for the training set.

    Returns
    -------
    X_train, y_train, X_test, y_test: arrays
        Arrays holding, respectively, training and test / validation pairs.
    
    Note
    ----
    Stratified shuffle split preserves the unbalance found between classes in
    the dataset, while shuffling and splitting it at the same time into the
    train and test / validation sets.
    """
    from sklearn.model_selection import StratifiedShuffleSplit

    splitter = StratifiedShuffleSplit(n_splits=1, train_size=train_size)
    for train_idx, test_idx in splitter.split(X_data, y_data):
        # Training set
        X_train = X_data.loc[train_idx].values
        y_train = y_data.loc[train_idx].values
        # Test / Validation set
        X_test = X_data.loc[test_idx].values
        y_test = y_data.loc[test_idx].values
    
    return X_train, y_train, X_test, y_test


def bagging_classifier(n_models, X, y, sample_pct=0.8, 
                       scoring_method='neg_brier_score',
                       search_type='Halving'):
    """
    Bagging ensemble classifier built using the `scikit-learn` of-the-shelf
    `BaggingClassifier` class. Support vector machine classifier (SVC) is
    used as a base estimator. Pipeline is employed for hyperparameters search,
    with a k-fold cross validation.

    Parameters
    ----------
    n_models: int
        Number of models in a bagging ensemble.
    X: array
        Matrix of features.
    y: array
        Vector of class labels 0/1.
    sample_pct: float
        Percentage [0,1] of samples for training each base model.
    scoring_method: str
        Method used for scoring the classifier during cross-validated search
        for optimal hyperparameters.
    search_type: str
        Method used during hyperparameter search. Following options are allowed:
        'Halving': `HalvingRandomSearchCV`, 'Random:': `RandomizedSearchCV`,
        and 'Grid': `GridSearchCV` from `scikit-learn'.

    Returns
    -------
    BaggingClassifier
        Fitted and fine-tuned ensemble as a BaggingClassifier object
        from the `scikit-learn` library.
    """
    import timeit
    import warnings
    import datetime as dt
    
    from sklearn.svm import SVC
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import BaggingClassifier
    
    from scipy import stats
    from tempfile import mkdtemp
    from shutil import rmtree

    warnings.filterwarnings(action='ignore')

    # Temporary directory for caching 
    cache_dir = mkdtemp(prefix='pipe_cache_')

    print('Working ...')    
    # Support Vector Machine (SVM) classifier instance
    svc = SVC(probability=True, class_weight='balanced')
    # Create a pipeline with a bagging ensemble of SVM instances
    ens = BaggingClassifier(base_estimator=svc, n_estimators=n_models,
                            max_samples=sample_pct, bootstrap=True, n_jobs=-1)
    pipe = Pipeline(steps=[('preprocess', 'passthrough'),
                            ('estimator', ens)],
                    memory=cache_dir)
    param_dists = {'preprocess': [None, StandardScaler()],
                   'estimator__base_estimator__kernel': ['linear', 'rbf'],
                   'estimator__base_estimator__C': stats.loguniform(1e0, 1e3),
                   'estimator__base_estimator__gamma': ['scale', 'auto'],
                   }

    time_start = timeit.default_timer()
    # Model training with hyperparameters optimization
    search = hyper_search_cv(X, y, pipe, param_dists,
                             scoring_method, search_type)
    time_end = timeit.default_timer()
    time_elapsed = time_end - time_start

    print('Execution time (hour:min:sec): {}'.format(
        str(dt.timedelta(seconds=time_elapsed))))
    print('Best parameter (CV score = {:.3f}):'.format(search.best_score_))
    for key, value in search.best_params_.items():
        print(key, '::', value)
    
    # Remove the temporary directory
    rmtree(cache_dir)

    return search


def bagging_ensemble_svm(n_models, X, y, sample_pct=0.8, weighted=False,
                         scoring_method='neg_brier_score', 
                         search_type='Halving'):
    """ 
    Bagging ensemble classifier built by hand from support vector machine
    base classifiers.

    Parameters
    ----------
    n_models: int
        Number of models in a bagging ensemble.
    X: array
        Matrix of features.
    y: array
        Vector of class labels 0/1.
    sample_pct: float
        Percentage [0,1] of samples for training each base model.
    weighted: bool
        Compute weights (True/False) or use equal weighting.
    scoring_method: str
        Method used for scoring the classifier during cross-validated search
        for optimal hyperparameters.
    search_type: str
        Method used during hyperparameter search. Following options are allowed:
        'Halving': `HalvingRandomSearchCV`, 'Random:': `RandomizedSearchCV`,
        and 'Grid': `GridSearchCV` from `scikit-learn'.

    Returns
    -------
    bagging_ensemble: VotingClassifier
        Fitted bagging ensemble as a VotingClassifier object from the
        `scikit-learn` library.
    """
    import timeit
    import warnings
    import datetime as dt
    import numpy as np
    
    from sklearn.svm import SVC
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.ensemble import VotingClassifier
    
    from scipy import stats, optimize
    from tempfile import mkdtemp
    from shutil import rmtree

    def loss_function(weights):
        # Loss function for the weights optimization
        from sklearn.metrics import log_loss

        final_prediction = 0.
        for weight, prediction in zip(weights, predictions):
            final_prediction += weight*prediction
        # using scikit-learn "log_loss" for the classification
        loss_value = log_loss(y_valid, final_prediction)

        return loss_value

    warnings.filterwarnings(action='ignore')

    # Split data into two parts using stratified random shuffle
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=0.8)
    for train_idx, valid_idx in splitter.split(X, y):
        # Training set for bootstraping
        X_train = X[train_idx]
        y_train = y[train_idx]
        # Validation set for aggregation
        X_valid = X[valid_idx]
        y_valid = y[valid_idx]

    models = {}
    rng = np.random.default_rng()
    max_samples = int(sample_pct*len(y))

    for i in range(n_models):
        print('Train model {} of {}:'.format(i+1, n_models))
        print('Working ...')

        # Temporary directory for caching 
        cache_dir = mkdtemp(prefix='pipe_cache_')

        # Bootstrap sample from the training set
        idx = rng.choice(len(y_train), max_samples, replace=True)
        X_sample = X_train[idx]
        y_sample = y_train[idx]

        # SVM classifier instance
        svc = SVC(probability=True, class_weight='balanced')
        # Pipeline
        pipe = Pipeline(steps=[('preprocess', 'passthrough'),
                               ('estimator', svc)],
                        memory=cache_dir)
        param_dists = {'preprocess': [None, StandardScaler()],
                       'estimator__kernel': ['linear', 'rbf'],
                       'estimator__C': stats.loguniform(1e0, 1e3),
                       'estimator__gamma': ['scale', 'auto'],
                       }

        # Hyperparameters optimization for each base model
        time_start = timeit.default_timer()
        models[i] = hyper_search_cv(X_sample, y_sample, pipe, param_dists,
                                    scoring_method, search_type)
        time_end = timeit.default_timer()
        time_elapsed = time_end - time_start

        print('Execution time (hour:min:sec): {}'.format(
            str(dt.timedelta(seconds=time_elapsed))))
        for key, value in models[i].best_params_.items():
            print(key, '::', value)
        
        # Remove the temporary directory
        rmtree(cache_dir)

    # Aggregate predictions from individual base models using soft voting
    print('Aggregating predictions:')
    estimators = [('{}'.format(i), models[i].best_estimator_['estimator'])
                  for i in range(n_models)]

    if weighted:
        # Unequal weights for base models (agressive)
        predictions = []
        for i in range(n_models):
            model = models[i].best_estimator_['estimator']
            y_probability = model.predict_proba(X_valid)
            predictions.append(y_probability)

        # Find optimal weights by optimization
        start_vals = [1./len(predictions)]*len(predictions)
        constr = ({'type': 'eq', 'fun': lambda w: 1. - np.sum(w)})
        bounds = [(0., 1.)]*len(predictions)
        res = optimize.minimize(loss_function, start_vals, method='SLSQP',
                                bounds=bounds, constraints=constr)
        weights = res['x']
        print('With optimal weights: {}; Sum: {}.'
              .format(weights.round(3), weights.sum().round(3)))

    else:
        # Equal weights for all base models
        print('With equal weights.')
        weights = None

    # Voting ensemble classifier
    print('Working ...')
    bagging_ensemble = VotingClassifier(estimators, voting='soft',
                                        weights=weights, n_jobs=-1)
    bagging_ensemble.fit(X_valid, y_valid)  # validation set
    print('Done.')

    return bagging_ensemble


def support_vectors(variant, model, n_models, X, y):
    """
    Extract support vectors from the trained bagging ensemble class instance.

    Parameters
    ----------
    variant: str
        Variant of the bagging classifier: "A" (is of-the-shelf model) and 
        "B" (is a hand-made model).
    model: scikilt-learn
        Pretrained bagging ensemble instance from the `scikit-learn` model.
    n_models: int
        Number of base models in the bagging ensemble. This parameter is
        relevant for the variant "B".
    X, y: arrays
        Featire matrix `X` and class labels array `y` holding the training
        instances. These parameters are relevant for the variant "A".
    
    Returns
    -------
    vectors: array
        Support vectors from the underlying SVM base estimators of the bagging
        ensemble.
    
    Raises
    ------
    NotImplementedError
    """
    import numpy as np

    if variant == 'A':
        # Variant A
        # Support vectors from the best base estimator
        estimator_parameters = model.best_estimator_
        best_svc = estimator_parameters['estimator'].base_estimator_
        best_svc.fit(X, y)
        vectors = best_svc.support_vectors_

    elif variant == 'B':
        # Variant B
        # Support vectors from all base estimators
        support_vectors = []
        for i in range(n_models):
            supports = model.estimators_[i].support_vectors_
            support_vectors.append(supports)
        support_vectors = np.concatenate(support_vectors)
        # Remove duplicates
        vectors = np.unique(support_vectors, axis=0)     
    
    else:
        raise NotImplementedError('Unrecognized variant.')

    return vectors


def plot_realizations(dists, amps, flashes, sws, save_fig=False):
    """ Plot realized flashovers from randomly generated samples.

    Parameters
    ----------
    dists: array
        Distances of lightning strikes from distribution line.
    amps: array
        Amplitudes of lightning strikes.
    flashes: array
        Array of flashover indicator values.
    sws: array
        Array of shield wire indices, indicating their presence or absence.
    save_fig: bool
        Save figure (True/False).

    Returns
    -------
    return:
        Matplotlib figure object.
    """
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    import seaborn as sns

    sns.set(context='paper', style='white', font_scale=1.25)
    fig = plt.figure(figsize=(6, 6))
    ms = 25
    gs = gridspec.GridSpec(2, 2, width_ratios=[5, 1], height_ratios=[1, 5])

    # Main axis plot
    ax_joint = plt.subplot(gs[1, 0])
    ax_joint.scatter(dists[(flashes == 0) & (sws == False)],
                     amps[(flashes == 0) & (sws == False)],
                     s=ms, color='steelblue', edgecolor='dimgrey',
                     label='No flashover (w/o shield wire)')
    ax_joint.scatter(dists[(flashes == 0) & (sws == True)],
                     amps[(flashes == 0) & (sws == True)],
                     s=ms, color='steelblue', alpha=0.5,
                     label='No flashover (with shield wire)')
    ax_joint.scatter(dists[(flashes == 1) & (sws == False)],
                     amps[(flashes == 1) & (sws == False)],
                     s=ms, color='red', edgecolors='dimgrey',
                     label='Flashover (w/o shield wire)')
    ax_joint.scatter(dists[(flashes == 1) & (sws == True)],
                     amps[(flashes == 1) & (sws == True)],
                     s=ms, color='red', alpha=0.5,
                     label='Flashover (with shield wire)')
    ax_joint.legend(loc='upper right', frameon='fancy', fancybox=True)
    ax_joint.set_xlabel('Distance (m)', fontweight='bold')
    ax_joint.set_ylabel('Amplitude (kA)', fontweight='bold')
    ax_joint.set_xlim(0, 500)
    ax_joint.set_ylim(0, 160)
    ax_joint.spines['top'].set_visible(False)
    ax_joint.spines['right'].set_visible(False)
    ax_joint.xaxis.set_ticks_position('bottom')
    ax_joint.yaxis.set_ticks_position('left')

    # Top axis plot
    ax_top = plt.subplot(gs[0, 0])
    sns.kdeplot(dists[(flashes == 0)], shade=True, color='steelblue',
                bw_method='scott', gridsize=100, cut=3, ax=ax_top, label='')
    sns.kdeplot(dists[(flashes == 1)], shade=True, color='red',
                bw_method='scott', gridsize=100, cut=3, ax=ax_top, label='')
    ax_top.set_xlim(0, 500)
    ax_top.set_xlabel('')
    ax_top.set_ylabel('')
    ax_top.spines['top'].set_visible(False)
    ax_top.spines['right'].set_visible(False)
    ax_top.spines['left'].set_visible(False)
    ax_top.yaxis.set_ticks_position('none')
    ax_top.xaxis.set_ticks_position('bottom')
    ax_top.set_xticklabels([])
    ax_top.set_yticklabels([])

    # Right axis plot
    ax_right = plt.subplot(gs[1, 1], )
    sns.kdeplot(y=amps[(flashes == 0)], shade=True, color='steelblue',
                bw_method='scott', gridsize=100, cut=3, ax=ax_right, label='')
    sns.kdeplot(y=amps[(flashes == 1)], shade=True, color='red',
                bw_method='scott', gridsize=100, cut=3, ax=ax_right, label='')
    ax_right.set_ylim(0, 160)
    ax_right.set_xlabel('')
    ax_right.set_ylabel('')
    ax_right.spines['top'].set_visible(False)
    ax_right.spines['right'].set_visible(False)
    ax_right.spines['bottom'].set_visible(False)
    ax_right.yaxis.set_ticks_position('left')
    ax_right.xaxis.set_ticks_position('none')
    ax_right.set_xticklabels([])
    ax_right.set_yticklabels([])
    plt.tight_layout()
    gs.update(hspace=0.05, wspace=0.05)

    if save_fig:
        plt.savefig('3axis_output.png', dpi=600)
    plt.show()


def regression_plot(X, y, vectors, v, predictions, xx, yy, zz, 
                    title=None, xlim=300, ylim=160, save_fig=False):
    """
    Plot a least squares regression through the support vectors from the
    support vector machine to create the curve of limiting parameters (CLP)
    of the distribution line.

    Parameters
    ----------
    X: array
        Matrix (n_samples, n_features) of features.
    y: array
        Array of class labels (0,1) indicating flashovers.
    vectors: array
        Support vectors from the SVM base estimators.
    v: array
        Support values from the least squares fit.
    predictions: dict
        Dictionary holding predictions from the least squares fit.
    xx, yy, zz: array
        Arrays holding grid values and predictions from the model.
    title: string or None
        Description of the type of the least squares fit.
    xlim, ylim: float
        Limits for the x and y axis, respectively.
    save_fig: bool
        True/False indicator which determines if the figure will be saved
        on disk or not (PNG file format at 600 dpi resolution).
    Returns
    -------
    return:
        Matplotlib figure object.

    Note
    ----
    Least squares fit has been performed using the `statsmodels` library.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    if title is not None:
        ax.set_title(title, fontweight='bold', fontsize=13)
    ax.pcolormesh(xx, yy, zz, cmap=plt.cm.RdYlGn, shading='nearest', alpha=0.8)
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='none', cmap=plt.cm.bone, s=20)
    ax.fill_between(v, predictions['obs_ci_lower'], predictions['obs_ci_upper'],
                    color='cornflowerblue', alpha=0.3, 
                    label='prediction interval')
    ax.fill_between(v, predictions['mean_ci_lower'], predictions['mean_ci_upper'],
                    color='royalblue', alpha=0.5, label='confidence interval')
    ax.scatter(vectors[:, 0], vectors[:, 1], facecolor='none',
               edgecolor='darkorange', linewidths=1.5, s=40, 
               label='support vectors')
    ax.plot(v, predictions['mean'], c='navy', lw=2, label='CLP curve')
    ax.legend(loc='upper right', frameon='fancy', fancybox=True, framealpha=0.6)
    ax.set_xlabel('Distance (m)', fontweight='bold')
    ax.set_ylabel('Amplitude (kA)', fontweight='bold')
    ax.set_xlim(0, xlim)
    ax.set_ylim(0, ylim)
    fig.tight_layout()
    if save_fig:
        plt.savefig('clp_regression_plot.png', dpi=600)
    plt.show()


def marginal_plot(marginal, xy, y_hat, g, varname, label, 
                  xmax=100, save_fig=False):
    """ 
    Plot estimated probability distribution function of flashovers from
    support vector machine based ensemble classifier.

    Parameters
    ----------
    marginal: array
        Array of values (distances or amplitudes) for which probability
        distributions will be graphically displayed.
    xy: array
        Array of support values for the probability distributions.
    y_hat: array
        Probability estimates at the support values from the classifier.
    g: pandas groupby object
        Pandas groupby object holding the underlying dataset.
    varname: label
        Name of the column in the pandas DataFrame with variable:
        `dist` for distances and `ampl` for amplitudes.
    label: string
        Label for the support values (x-axis); it can be distance or amplitude.
    xmax: float
        Limit for displaying the values on the x-axis.
    save_figure: bool
        Indicator (True/False) for saving the figure.
    
    Returns
    -------
    return:
        Matplotlib figure object.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import distlines

    fig, ax = plt.subplots(figsize=(6, 4))
    # Flashover probability distributions
    for d in marginal:
        if varname == 'ampl':
            legend_label = '{} m'.format(d)
        elif varname == 'dist':
            legend_label = '{} kA'.format(d)
        ax.plot(xy, y_hat[d][:, 1], ls='-', lw='3', label=legend_label)
    # Add scatter points
    distlines.jitter(ax, g[varname][g['shield'] == True], 
                    g['flash'][g['shield'] == True], s=20,
                    c='darkorange', label='shield wire', zorder=10)
    distlines.jitter(ax, g[varname][g['shield'] == False], 
                    g['flash'][g['shield'] == False], s=5,
                    c='royalblue', label='no shield wire', zorder=10)
    ax.legend(loc='center right')
    ax.set_xlabel(label, fontweight='bold')
    ax.set_ylabel('Flashover probability', fontweight='bold')
    ax.set_yticks(np.arange(0, 1.1, step=0.1))
    ax.set_xlim(-1, xmax)
    ax.grid(which='major', axis='both')
    fig.tight_layout()
    if save_fig:
        plt.savefig('marginal_proba.png', dpi=600)
    plt.show()


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
