import numpy as np
import pandas as pd


from kp.data.features.moment import MomentPruner


def test_moment_fit():
    a = np.random.uniform(100, size=(100,))
    b = np.random.uniform(100, size=(100,))
    df = pd.DataFrame({'a': a, 'b': b}, index=np.arange(100))
    mp = MomentPruner(min_variance=0.0)
    mp.fit(df)
    assert mp.drop_cols == []


def test_moment_fit_drop_high_correlation():
    """
    Remove columns with a high correlation with another column
    """
    a = np.random.uniform(100, size=(100,))
    df = pd.DataFrame({'a': a, 'b': a}, index=np.arange(100))
    mp = MomentPruner(min_variance=0.0, max_corr=0.95)
    mp.fit(df)
    assert mp.drop_cols == ['b']


def test_moment_fit_drop_low_variance():
    """
    Remove columns that have a constant integer value
    """
    a = np.random.uniform(100, size=(100,))
    b = np.random.choice([0], size=(100,))
    df = pd.DataFrame({'a': a, 'b': b}, index=np.arange(100))
    mp = MomentPruner(max_corr=1.0)
    mp.fit(df)
    assert mp.drop_cols == ['b']
