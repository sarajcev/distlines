def test_egm():
    from distlines import egm

    rg, rc, Ag, bg = egm(1., model='AW')
    assert rg == 6.
    assert rc == 6.7

    rg, rc, Ag, bg = egm(1., model='BW')
    assert rg == 6.4
    assert rc == 7.1

    rg, rc, Ag, bg = egm(1., model='Anderson')
    assert rg == 8.
    assert rc == 8.


def test_max_shielding_current():
    from distlines import max_shielding_current

    Igm = max_shielding_current(1., 12., 10., 2.)
    assert abs(Igm - 2.877) <= 1e-3


def test_hyper_search_cv():
    import numpy as np
    from sandbox import hyper_search_cv
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.utils import class_weight

    N = 100
    rng = np.random.default_rng()
    X = np.c_[rng.random(N), rng.random(N)]
    y = rng.integers(0, 2, N)

    # Define class_weights
    class_weights = class_weight.compute_class_weight(
        'balanced', classes=np.unique(y), y=y)
    class_weights_dict = dict(zip(np.unique(y), class_weights.round(3)))

    # Creating a pipeline
    svc = SVC(probability=True, class_weight=class_weights_dict)
    pipe = Pipeline(steps=[('preprocess', 'passthrough'),
                           ('estimator', svc)])
    param_grid = {'preprocess': [None, StandardScaler()],
                  'estimator__kernel': ['linear', 'rbf'],
                  'estimator__C': [0.001, 0.1, 10., 1000.],
                  'estimator__gamma': ['scale', 'auto'],
                  }

    search = hyper_search_cv(X, y, pipe, param_grid,
                             scoring_method='neg_brier_score',
                             search_type='Grid')


def test_pdf_from_kde():
    import numpy as np
    from sandbox import pdf_from_kde

    N = 40
    rng = np.random.default_rng()
    x_data = rng.normal(loc=0., scale=1., size=N)
    x_grid = np.linspace(-1, 1, 10)

    pdf = pdf_from_kde(x_data, x_grid, bw='search')


def test_pdf_from_kde_sm():
    import numpy as np
    from sandbox import pdf_from_kde_sm

    N = 40
    rng = np.random.default_rng()
    x_data = rng.normal(loc=0., scale=1., size=N)
    x_grid = np.linspace(-1, 1, 10)

    pdf = pdf_from_kde_sm(x_data, x_grid, kernel='gau', bw='scott')


def test_lightning_amplitudes_pdf():
    from distlines import lightning_amplitudes_pdf

    pdf0 = lightning_amplitudes_pdf(1.)
    assert abs(pdf0) <= 1e-3