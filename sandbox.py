# Author: Petar Sarajcev, PhD (petar.sarajcev@fesb.hr)
# University of Split, FESB, Department of Power Engineering,
# R. Boskovica 32, HR-21000, Split, Croatia.

import numpy as np


def hyper_search_cv(X, y, pipe, params_dict, scoring_method,
                    search_type='Random', n_iterations=100):
    """
    Hyperparameters optimization with `scikit-learn`.

    Scikit-learn model hyperparameters optimization with:
    GridSearchCV, RandomizedSearchCV, or HalvingRandomSearchCV
    methods.

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
        Scoring method from the `scikit-learn` library for model
        training.
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
        When the search type is not recognized.
    """
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.model_selection import GridSearchCV
    from sklearn.experimental import enable_halving_search_cv  # noqa
    from sklearn.model_selection import HalvingRandomSearchCV
    from sklearn.model_selection import StratifiedKFold
    
    import warnings
    
    # Experimental HalvingRandomSearchCV is known for raising
    # warnings during fit, which we'll just ignore for now.
    warnings.filterwarnings(action='ignore')

    if search_type == 'Random':
        # Randomized search with k-fold cross-validation.
        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=params_dict,
            cv=StratifiedKFold(n_splits=3),
            scoring=scoring_method,
            n_iter=n_iterations,
            refit=True,
            n_jobs=-1
        )
    elif search_type == 'Grid':
        # Grid search with k-fold cross-validation.
        search = GridSearchCV(
            estimator=pipe,
            param_grid=params_dict,  # grid values!
            cv=StratifiedKFold(n_splits=3),
            scoring=scoring_method,
            refit=True,
            n_jobs=-1
        )
    elif search_type == 'Halving':
        # Halving random search with k-fold cross-validation.
        # HalvingRandomSearchCV is still considered experimental!
        search = HalvingRandomSearchCV(
            estimator=pipe,
            param_distributions=params_dict,
            cv=StratifiedKFold(n_splits=3),
            scoring=scoring_method,
            refit=True,
            n_jobs=-1
        )
    else:
        raise NotImplementedError(
            'Search type "{}" is not recognized!'.format(search_type))
    
    search.fit(X, y)

    return search


def train_test_shuffle_split(X_data, y_data, train_size=0.8):
    """
    Stratified shuffle split.

    Stratified shuffle split of data into training and
    test/ validation set.

    Parameters
    ----------
    X_data: DataFrame
        Pandas dataframe holding the matrix of features.
    y_data: Series
        Pandas series holding the targets.
    train_size: float
        Percentage of the dataset that will be reserved for
        the training set.

    Returns
    -------
    X_train, y_train, X_test, y_test: arrays
        Arrays holding, respectively, training and test /
        validation pairs.

    Notes
    -----
    Stratified shuffle split preserves the unbalance found
    between classes in the dataset, while shuffling and splitting
    it at the same time into the train and test / validation sets.
    """
    from sklearn.model_selection import StratifiedShuffleSplit

    splitter = StratifiedShuffleSplit(n_splits=1, train_size=train_size)
    for train_idx, test_idx in splitter.split(X_data, y_data):
        # Training set.
        X_train = X_data.iloc[train_idx]
        y_train = y_data.iloc[train_idx]
        # Test / Validation set.
        X_test = X_data.iloc[test_idx]
        y_test = y_data.iloc[test_idx]

    return X_train, y_train, X_test, y_test


def bagging_classifier(n_models, X, y, sample_pct=0.8,
                       scoring_method='neg_brier_score',
                       search_type='Halving'):
    """
    Bagging ensemble classifier built using the `scikit-learn`.

    Bagging ensemble classifier built using the `scikit-learn`
    of-the-shelf `BaggingClassifier` class. Support vector machine
    classifier (SVC) is used as a base estimator. Pipeline is
    employed for hyperparameters search, with a k-fold cross
    validation.

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
        Method used for scoring the classifier during cross-validated
        search for optimal hyperparameters.
    search_type: str
        Method used during hyperparameter search. Following options
        are allowed:
        - 'Halving': `HalvingRandomSearchCV`,
        - 'Random:': `RandomizedSearchCV`,
        - 'Grid': `GridSearchCV`
        from `scikit-learn'.

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
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.ensemble import BaggingClassifier

    from scipy import stats
    from tempfile import mkdtemp
    from shutil import rmtree

    warnings.filterwarnings(action='ignore')

    # Temporary directory for caching.
    cache_dir = mkdtemp(prefix='pipe_cache_')

    print('Working ...')
    # Support Vector Machine (SVM) classifier instance.
    svc = SVC(probability=True, class_weight='balanced')
    # Create a pipeline with a bagging ensemble of SVM instances.
    ens = BaggingClassifier(
        base_estimator=svc,
        n_estimators=n_models,
        max_samples=sample_pct,
        bootstrap=True,
        n_jobs=-1
    )
    pipe = Pipeline(
        steps=[
            ('preprocess', 'passthrough'),
            ('estimator', ens),
        ],
        memory=cache_dir,
    )
    param_dists = {
        'preprocess': [None, StandardScaler(), MinMaxScaler()],
        'estimator__base_estimator__kernel': ['linear', 'rbf'],
        'estimator__base_estimator__C': stats.loguniform(1e0, 1e3),
        'estimator__base_estimator__gamma': ['scale', 'auto'],
    }
    time_start = timeit.default_timer()
    # Model training with hyperparameters optimization.
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


def loss_cross_entropy(weights, y_proba, y_true):
    """
    Cross-entropy loss function.

    Parameters
    ----------
    weights: array-like
        Model weights.
    y_proba: array-like
        Prediction probabilities of positive class (from the models)
        for each sample in the set.
    y_true: array-like
        Actual labels for each sample in the set.

    Returns
    -------
    loss_value: float
        Computed cross-entropy loss value.
    """
    from sklearn.metrics import log_loss

    fp = 0.
    for weight, proba in zip(weights, y_proba):
        fp += weight * proba
    # Compute cross-entropy loss using a "log_loss"
    # function from the scikit-learn library
    loss_value = log_loss(y_true, fp)

    return loss_value


def loss_balanced_cross_entropy(weights, y_proba, y_true, alpha=0.75):
    """
    Balanced cross-entropy loss function.

    Parameters
    ----------
    weights: array-like
        Model weights.
    y_proba: array-like
        Prediction probabilities of positive class (from the models)
        for each sample in the set.
    y_true: array-like
        Actual labels for each sample in the set.
    alpha: float
        Weight parameter in the range [0,1] that balances the classes
        and defines the balanced cross-entropy loss from the underlying
        cross-entropy value.

    Returns
    -------
    loss_value: float
        Computed balanced cross-entropy loss value.
    """
    fp = 0.
    for weight, proba in zip(weights, y_proba):
        fp += weight * proba
    # Compute balanced cross-entropy loss.
    loss_value = -np.sum(
        alpha*y_true*np.log(fp[:, 0])
        + (1. - alpha)*(1. - y_true)*np.log(fp[:, 1]))
    
    return loss_value


def focal_loss(weights, y_proba, y_true, gamma=2):
    """
    Focal loss.

    Parameters
    ----------
    weights: array-like
        Model weights.
    y_proba: array-like
        Prediction probabilities of positive class (from the
        models) for each sample in the set.
    y_true: array-like
        Actual labels for each sample in the set.
    gamma: float
        Focusing parameter (gamma >= 0) that modulates the
        cross-entropy.

    Returns
    -------
    loss_value: float
        Computed balanced cross-entropy loss value.

    Notes
    -----
    Lin, et al.: Focal Loss for Dense Object Detection,
    Facebook AI Research (FAIR), 2018.
    """
    fp = 0.
    for weight, proba in zip(weights, y_proba):
        fp += weight * proba
    # Compute focal loss.
    loss_value = -np.sum(
        (1. - (y_true*fp[:, 0] + (1. - y_true)*fp[:, 1]))**gamma
        * (y_true*np.log(fp[:, 0]) + (1. - y_true)*np.log(fp[:, 1])))
    
    return loss_value


def focal_loss_balanced(weights, y_proba, y_true, alpha=0.75, gamma=2):
    """
    Balanced focal loss.

    Parameters
    ----------
    weights: array-like
        Model weights.
    y_proba: array-like
        Prediction probabilities of positive class (from the
        models) for each sample in the set.
    y_true: array-like
        Actual labels for each sample in the set.
    alpha: float
        Weight parameter in the range [0,1] that balances the
        classes.
    gamma: float
        Focusing parameter (gamma >= 0) that modulates the
        cross-entropy.

    Returns
    -------
    loss_value: float
        Computed balanced cross-entropy loss value.

    Notes
    -----
    Lin, et al.: Focal Loss for Dense Object Detection,
    Facebook AI Research (FAIR), 2018.
    """
    fp = 0.
    for weight, proba in zip(weights, y_proba):
        fp += weight * proba
    # Compute balanced focal loss.
    loss_value = -np.sum(
        (1. - (y_true*fp[:, 0] + (1.-y_true)*fp[:, 1]))**gamma
        * (alpha*y_true*np.log(fp[:, 0]) + (1. - alpha)*(1.-y_true)*np.log(fp[:, 1])))
    
    return loss_value


def bagging_ensemble_svm(n_models, X, y, sample_pct=0.8, weighted=False,
                         scoring_method='neg_brier_score',
                         search_type='Halving', sampling='Bootstrap',
                         weights_loss_type='balanced_cross_entropy'):
    """
    Bagging ensemble classifier.

    Bagging ensemble classifier built by hand from support vector
    machine base classifiers. Ensemble is built by soft voting, and
    base estimators can be weighted or not.

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
        Method used for scoring the classifier during cross-validated
        search for optimal hyperparameters.
    search_type: str
        Method used during hyperparameter search. Following options
        are allowed:
        - 'Halving': `HalvingRandomSearchCV`,
        - 'Random:': `RandomizedSearchCV`,
        - 'Grid': `GridSearchCV`
        from `scikit-learn'.
    sampling: str
        Method used for sampling training subsamples for training
        base estimators; it can be one of the following: `Bootstrap`
        or `Stratified`.
    weights_loss_type: str
        Loss type used during model weights optimization. Following
        loss functions have been implemented:
        - `cross_entropy`: cross-entropy loss
        - `balanced_cross_entropy`: balanced cross-entropy loss
        - `focal_loss`: focal loss
        - `balanced_focal_loss`: balanced focal loss

    Returns
    -------
    bagging_ensemble: VotingClassifier
        Fitted bagging ensemble as a VotingClassifier object from
        the `scikit-learn` library.

    Raises
    ------
        NotImplementedError
    """
    import timeit
    import warnings
    import datetime as dt

    from sklearn.svm import SVC
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.ensemble import VotingClassifier

    from scipy import stats, optimize
    from tempfile import mkdtemp
    from shutil import rmtree

    warnings.filterwarnings(action='ignore')

    # Split data into two parts using stratified random shuffle
    # The initial training set is split into train and validation sets.
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=0.8)
    for train_idx, valid_idx in splitter.split(X, y):
        # Training set for bootstraping.
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        # Validation set for aggregation.
        X_valid = X.iloc[valid_idx]
        y_valid = y.iloc[valid_idx]

    models = {}
    rng = np.random.default_rng()
    max_samples = int(sample_pct*len(y_train))

    for i in range(n_models):
        # Train each base model on a subsample from the train set.
        print('Train model {} of {}:'.format(i+1, n_models))
        print('Working ...')

        # Temporary directory for caching.
        cache_dir = mkdtemp(prefix='pipe_cache_')

        # The train set is subsampled for training individual base estimators.
        if sampling == 'Bootstrap':
            # Bootstrap (sub)sample from the train set (with replacement).
            idx = rng.choice(y_train.index, size=max_samples,
                             replace=True, shuffle=True)
            X_sample = X_train.iloc[idx]
            y_sample = y_train.iloc[idx]
        elif sampling == 'Stratified':
            # Stratified (sub)sample from the train set (without replacement).
            splitter = StratifiedShuffleSplit(
                n_splits=1, train_size=max_samples)
            for idx, _ in splitter.split(X_train, y_train):
                X_sample = X_train.iloc[idx]
                y_sample = y_train.iloc[idx]
        else:
            raise NotImplementedError(
                f'Sampling method: {sampling} is not recognized!')

        # SVM classifier instance.
        svc = SVC(probability=True, class_weight='balanced')
        # Pipeline
        pipe = Pipeline(steps=[('preprocess', 'passthrough'),
                               ('estimator', svc)],
                        memory=cache_dir)
        param_dists = {'preprocess': [None, StandardScaler(), MinMaxScaler()],
                       'estimator__kernel': ['linear', 'rbf'],
                       'estimator__C': stats.loguniform(1e0, 1e3),
                       'estimator__gamma': ['scale', 'auto'],
                       }
        # Hyperparameters optimization for each base model.
        time_start = timeit.default_timer()
        models[i] = hyper_search_cv(X_sample, y_sample, pipe, param_dists,
                                    scoring_method, search_type)
        time_end = timeit.default_timer()
        time_elapsed = time_end - time_start
        print('Execution time (hour:min:sec): {}'.format(
            str(dt.timedelta(seconds=time_elapsed))))
        # Show best hyperparameters.
        for key, value in models[i].best_params_.items():
            print(key, '::', value)

        # Remove the temporary directory.
        rmtree(cache_dir)

    # Aggregate predictions from individual base models using soft voting.
    print('Aggregating predictions:')
    estimators = [('{}'.format(i), models[i].best_estimator_['estimator'])
                  for i in range(n_models)]

    if weighted:
        # Unequal weights for base models (optimization).
        predictions = []
        for i in range(n_models):
            model = models[i].best_estimator_['estimator']
            y_probability = model.predict_proba(X_valid)
            predictions.append(y_probability)

        # Defining loss function for weights optimization.
        if weights_loss_type == 'cross_entropy':
            loss_function = loss_cross_entropy
        elif weights_loss_type == 'balanced_cross_entropy':
            loss_function = loss_balanced_cross_entropy
        elif weights_loss_type == 'focal_loss':
            loss_function = focal_loss
        elif weights_loss_type == 'balanced_focal_loss':
            loss_function = focal_loss_balanced
        else:
            raise NotImplementedError(
                f'Weights loss type: {weights_loss_type} is not recognized!')

        # Find weights by optimization.
        start_vals = [1./len(predictions)] * len(predictions)
        constr = ({'type': 'eq', 'fun': lambda w: 1. - np.sum(w)})
        bounds = [(0., 1.)]*len(predictions)
        res = optimize.minimize(
            loss_function,
            x0=start_vals,  # initial guess values
            args=(predictions, y_valid),
            method='SLSQP',
            bounds=bounds,  # bounds on weights
            constraints=constr  # constraints
        )
        weights = res['x']
        print('With optimal weights: {}; Sum: {}.'
              .format(weights.round(3), weights.sum().round(3)))
    else:
        # Equal weights for all base models.
        print('With equal weights.')
        weights = None

    # Voting ensemble classifier.
    print('Create voting ensemble:')
    print('Working ...')
    bagging_ensemble = VotingClassifier(estimators, voting='soft',
                                        weights=weights, n_jobs=-1)
    bagging_ensemble.fit(X_valid, y_valid)  # validation set
    print('Done.')
    
    return bagging_ensemble


def support_vectors(variant, model, n_models, X, y):
    """
    Support vectors extraction.

    Extract support vectors from the trained bagging ensemble
    class instance.

    Parameters
    ----------
    variant: str
        Variant of the bagging classifier: "A" (is of-the-shelf
        model) and "B" (is a hand-made model).
    model: scikilt-learn
        Pretrained bagging ensemble instance from the `scikit-learn`
        model.
    n_models: int
        Number of base models in the bagging ensemble. This para-
        meter is relevant for the variant "B".
    X, y: arrays
        Featire matrix `X` and class labels array `y` holding the
        training instances. These parameters are relevant for the
        variant "A".

    Returns
    -------
    vectors: array
        Support vectors from the underlying SVM base estimators of
        the bagging ensemble.

    Raises
    ------
    NotImplementedError
    """
    if variant == 'A':
        # Variant A
        # Support vectors from the best base estimator.
        estimator_parameters = model.best_estimator_
        best_svc = estimator_parameters['estimator'].base_estimator_
        best_svc.fit(X, y)
        vectors = best_svc.support_vectors_
    elif variant == 'B':
        # Variant B
        # Support vectors from all base estimators.
        support_vectors = []
        for i in range(n_models):
            supports = model.estimators_[i].support_vectors_
            support_vectors.append(supports)
        support_vectors = np.concatenate(support_vectors)
        # Remove duplicates.
        vectors = np.unique(support_vectors, axis=0)
    else:
        raise NotImplementedError('Unrecognized variant.')

    return vectors


def plot_dataset(dists, amps, flashes, sws, 
                 xaxis_label, xlimit, yaxis_label, ylimit, 
                 fig_name, save_fig=False):
    """
    Plot randomly generated samples.

    Parameters
    ----------
    dists: array
        Distances of lightning strikes from distribution line.
    amps: array
        Amplitudes of lightning strikes.
    flashes: array
        Array of flashover indicator values.
    sws: array
        Array of shield wire indices, indicating their presence
        or absence.
    xaxis_label: str
        Label for the x-axis.
    xlimit: float
        Limit for the x-axis `ax.set_xlim` parameter.
    yaxis_label: str
        Label for the y-axis.
    ylimit: float
        Limit for the y-axis `ax.set_xlim` parameter.
    fig_name: str
        Filename for saving the figure (with a .png extension).
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
    gs = gridspec.GridSpec(2, 2, width_ratios=[5, 1], height_ratios=[1, 5])
    # Main axis plot
    ms = 25
    ax_joint = plt.subplot(gs[1, 0])
    ax_joint.scatter(dists[(flashes == 0) & (sws == False)],
                     amps[(flashes == 0) & (sws == False)],
                     s=ms, color='steelblue', 
                     edgecolor='dimgrey', alpha=0.75,
                     label='No flashover (w/o shield wire)')
    ax_joint.scatter(dists[(flashes == 0) & (sws == True)],
                     amps[(flashes == 0) & (sws == True)],
                     s=ms, color='steelblue', alpha=0.25,
                     label='No flashover (with shield wire)')
    ax_joint.scatter(dists[(flashes == 1) & (sws == False)],
                     amps[(flashes == 1) & (sws == False)],
                     s=ms, color='red', 
                     edgecolors='dimgrey', alpha=0.75,
                     label='Flashover (w/o shield wire)')
    ax_joint.scatter(dists[(flashes == 1) & (sws == True)],
                     amps[(flashes == 1) & (sws == True)],
                     s=ms, color='red', alpha=0.25,
                     label='Flashover (with shield wire)')
    ax_joint.legend(loc='upper right', frameon='fancy', fancybox=True)
    ax_joint.set_xlabel(xaxis_label, fontweight='bold')
    ax_joint.set_ylabel(yaxis_label, fontweight='bold')
    ax_joint.set_xlim(0, xlimit)
    ax_joint.set_ylim(0, ylimit)
    ax_joint.spines['top'].set_visible(False)
    ax_joint.spines['right'].set_visible(False)
    ax_joint.xaxis.set_ticks_position('bottom')
    ax_joint.yaxis.set_ticks_position('left')
    # Top axis plot
    ax_top = plt.subplot(gs[0, 0])
    sns.kdeplot(dists[(flashes == 0)], fill=True, color='steelblue',
                bw_method='scott', gridsize=100, cut=3, ax=ax_top, label='')
    sns.kdeplot(dists[(flashes == 1)], fill=True, color='red',
                bw_method='scott', gridsize=100, cut=3, ax=ax_top, label='')
    ax_top.set_xlim(0, xlimit)
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
    ax_right = plt.subplot(gs[1, 1])
    sns.kdeplot(y=amps[(flashes == 0)], fill=True, color='steelblue',
                bw_method='scott', gridsize=100, cut=3, ax=ax_right, label='')
    sns.kdeplot(y=amps[(flashes == 1)], fill=True, color='red',
                bw_method='scott', gridsize=100, cut=3, ax=ax_right, label='')
    ax_right.set_ylim(0, ylimit)
    ax_right.set_xlabel('')
    ax_right.set_ylabel('')
    ax_right.spines['top'].set_visible(False)
    ax_right.spines['right'].set_visible(False)
    ax_right.spines['bottom'].set_visible(False)
    ax_right.yaxis.set_ticks_position('left')
    ax_right.xaxis.set_ticks_position('none')
    ax_right.set_xticklabels([])
    ax_right.set_yticklabels([])
    fig.tight_layout()
    gs.update(hspace=0.05, wspace=0.05)

    if save_fig:
        plt.savefig(fig_name, dpi=600)
    plt.show()


def plot_dataset_3d(dists, amps, fronts, flashes, sws, save_fig=False):
    """
    3D plot of randomly generated samples.

    Parameters
    ----------
    dists: array
        Distances of lightning strikes from distribution line.
    amps: array
        Amplitudes of lightning strikes.
    fronts: array
        Wave-front times of lightning strikes.
    flashes: array
        Array of flashover indicator values.
    sws: array
        Array of shield wire indices, indicating their presence
        or absence.
    save_fig: bool
        Save figure (True/False).

    Returns
    -------
    return:
        Matplotlib figure object.
    """
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(5.5, 5.5))
    ms = 20
    ax = fig.add_subplot(projection='3d')
    ax.scatter(dists[(flashes == 0) & (sws == False)],
               amps[(flashes == 0) & (sws == False)],
               fronts[(flashes == 0) & (sws == False)],
               s=ms, color='steelblue', edgecolor='dimgrey',
               label='No flashover (w/o shield wire)')
    ax.scatter(dists[(flashes == 0) & (sws == True)],
               amps[(flashes == 0) & (sws == True)],
               fronts[(flashes == 0) & (sws == True)],
               s=ms, color='steelblue', alpha=0.5,
               label='No flashover (with shield wire)')
    ax.scatter(dists[(flashes == 1) & (sws == False)],
               amps[(flashes == 1) & (sws == False)],
               fronts[(flashes == 1) & (sws == False)],
               s=ms, color='red', edgecolors='dimgrey',
               label='Flashover (w/o shield wire)')
    ax.scatter(dists[(flashes == 1) & (sws == True)],
               amps[(flashes == 1) & (sws == True)],
               fronts[(flashes == 1) & (sws == True)],
               s=ms, color='red', alpha=0.5,
               label='Flashover (with shield wire)')
    ax.legend(loc='upper left', frameon='fancy', fancybox=True)
    ax.set_xlabel('Distance (m)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Amplitude (kA)', fontsize=10, fontweight='bold')
    ax.set_zlabel('Wavefront (us)', fontsize=10, fontweight='bold')
    ax.set_zlim(-0.5, 20)
    if save_fig:
        plt.savefig('3d_output.png', dpi=600)
    plt.show()


def plot_dataset_double_decker(dists, amps, fl, sws, save_fig=False):
    """
    Double decker plot of randomly generated samples.

    Parameters
    ----------
    dists: array
        Distances of lightning strikes from distribution line.
    amps: array
        Amplitudes of lightning strikes.
    fl: array
        Array of flashover indicator values.
    sws: array
        Array of shield wire indices, indicating their presence
        or absence.
    save_fig: bool
        Save figure (True/False).

    Returns
    -------
    return:
        Matplotlib figure object.
    """
    import matplotlib.pyplot as plt
    import utils

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(5, 6))
    # Marginal of distance.
    utils.jitter(ax[0], dists[sws==True], fl[sws==True], s=20,
                 c='darkorange', alpha=0.75, label='w/ shield wire')
    utils.jitter(ax[0], dists[sws==False], fl[sws==False], s=5,
                 c='royalblue', alpha=0.75, label='w/o shield wire')
    ax[0].legend(loc='center right')
    ax[0].set_ylabel('Flashover probability', fontsize=10, fontweight='bold')
    ax[0].set_xlabel('Distance (m)', fontsize=10, fontweight='bold')
    ax[0].grid(True)
    # Marginal of amplitude.
    utils.jitter(ax[1], amps[sws==True], fl[sws==True], s=20,
                 c='darkorange', alpha=0.75, label='w/ shield wire')
    utils.jitter(ax[1], amps[sws==False], fl[sws==False], s=5,
                 c='royalblue', alpha=0.75, label='w/o shield wire')
    ax[1].legend(loc='center right')
    ax[1].set_ylabel('Flashover probability', fontsize=10, fontweight='bold')
    ax[1].set_xlabel('Amplitude (kA)', fontsize=10, fontweight='bold')
    ax[1].grid(True)
    fig.tight_layout()

    if save_fig:
        plt.savefig('double_decker.png', dpi=600)
    plt.show()


def regression_plot(X, y, vectors, v, predictions, xx, yy, zz,
                    title=None, xlim=300, ylim=160, save_fig=False):
    """
    Regression plot.

    Plot a least squares regression through the support vectors from
    the support vector machine to create the curve of limiting para-
    meters (CLP) of the distribution line.

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
        True/False indicator which determines if the figure will
        be saved on disk or not (PNG file format at 600 dpi
        resolution).
    Returns
    -------
    return:
        Matplotlib figure object.

    Notes
    -----
    Least squares fit has been performed using the `statsmodels`
    library.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    if title is not None:
        ax.set_title(title, fontweight='bold', fontsize=13)
    ax.pcolormesh(xx, yy, zz, cmap=plt.cm.RdYlGn,
                  shading='nearest', alpha=0.8)
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='none',
               cmap=plt.cm.bone, s=20)
    ax.fill_between(v, predictions['obs_ci_lower'],
                    predictions['obs_ci_upper'],
                    color='cornflowerblue', alpha=0.3,
                    label='prediction interval')
    ax.fill_between(v, predictions['mean_ci_lower'],
                    predictions['mean_ci_upper'],
                    color='royalblue', alpha=0.5, label='confidence interval')
    ax.scatter(vectors[:, 0], vectors[:, 1], facecolor='none',
               edgecolor='darkorange', linewidths=1.5, s=40,
               label='support vectors')
    ax.plot(v, predictions['mean'], c='navy', lw=2, label='CLP curve')
    ax.legend(loc='upper right', frameon='fancy',
              fancybox=True, framealpha=0.6)
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
    Marginal plot.

    Plot estimated probability distribution function of flashovers
    from support vector machine based ensemble classifier.

    Parameters
    ----------
    marginal: array
        Array of values (distances or amplitudes) for which proba-
        bility distributions will be graphically displayed.
    xy: array
        Array of support values for the probability distributions.
    y_hat: array
        Probability estimates at the support values from the
        classifier.
    g: pandas groupby object
        Pandas groupby object holding the underlying dataset.
    varname: label
        Name of the column in the pandas DataFrame with variable:
        `dist` for distances and `ampl` for amplitudes.
    label: string
        Label for the support values (x-axis); it can be distance
        or amplitude.
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
    import utils

    fig, ax = plt.subplots(figsize=(6, 4))
    # Flashover probability distributions
    for d in marginal:
        if varname == 'ampl':
            legend_label = '{} m'.format(d)
        elif varname == 'dist':
            legend_label = '{} kA'.format(d)
        ax.plot(xy, y_hat[d][:, 1], ls='-', lw='3', label=legend_label)
    # Add scatter points
    utils.jitter(ax, g[varname][g['shield'] == True],
                 g['flash'][g['shield'] == True], s=20,
                 c='darkorange', label='shield wire', zorder=10)
    utils.jitter(ax, g[varname][g['shield'] == False],
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


def amplitude_distance_bivariate_pdf(y, x, *args):
    """
    Bivariate probability distribution.

    Bivariate probability density function of lightning-current
    amplitudes and distances (as independent random variables).
    """
    # Unpacking extra arguments
    xmin, xmax = args[0], args[1]
    muI, sigmaI = args[2], args[3]
    # Lightning current amplitudes (log-normal distribution)
    denominator = (np.sqrt(2.*np.pi)*y*sigmaI)
    pdfI = np.exp(-(np.log(y) - np.log(muI))**2/(2.*sigmaI**2)) / denominator
    # Distances (uniform distribution)
    pdfD = 1./(xmax - xmin)
    # Joint probability distribution
    pdf = pdfI * pdfD
    
    # Convert `nan` to numerical values
    return np.nan_to_num(pdf)


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
    limiting parameters (CLP).
    """
    def __init__(self, a, b, c):
        """
        Parameters
        ----------
        a, b, c: floats
            Parameters [a, b, c] of the second-degree polinomial
            CLP curve: y = a + b*x + c*x**2.
        """
        self.a = a
        self.b = b
        self.c = c

    def __call__(self, x):
        """Second-degree polinomial."""
        y = self.a + self.b*x + self.c*x**2

        return y


def risk_from_clp(clp, xmin, xmax, mu=31.1, sigma=0.484):
    """
    Computing risk from the CLP curve.

    Computing the risk of flashovers, from lightning interaction
    with overhead distribution lines, by means of the curve of
    limiting parameters (CLP).

    Parameters
    ----------
    clp: array
        Array holding parameters [a, b, c] of the second-degree
        polinomial CLP curve: y = a + b*x + c*x**2.
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
    from scipy import integrate

    a = clp[0]
    b = clp[1]
    c = clp[2]
    arguments = (xmin, xmax, mu, sigma)
    lower_boundary = DoubleIntegralBoundary(a, b, c)
    risk, _ = integrate.dblquad(
        amplitude_distance_bivariate_pdf,
        xmin, xmax,
        lower_boundary,    # gfun: lower boundary function
        lambda y: np.Inf,  # hfun: upper boundary function
        args=arguments
    )
    return risk


if __name__ == "__main__":
    """Showcase aspects of the library."""
    import matplotlib.pyplot as plt

    x = np.linspace(0, 150, 100, endpoint=True)
    y = np.linspace(0, 300, 100, endpoint=True)
    args = (0, 300, 31.1, 0.484)
    X, Y = np.meshgrid(x, y)
    Z = amplitude_distance_bivariate_pdf(X, Y, *args)

    # Plot 3D surface
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(X, Y, Z, edgecolor='royalblue', lw=0.5,
                    rstride=5, cstride=10, alpha=0.3)
    c = ax.contourf(X, Y, Z, zdir='z', offset=0, cmap='viridis')
    fig.colorbar(c, ax=ax, fraction=0.02, pad=0.1)
    ax.set_xlabel('Amplitudes')
    ax.set_ylabel('Distance')
    ax.set_zlabel('Probability')
    fig.tight_layout()
    plt.show()
