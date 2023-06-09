import pandas as pd

from kp.data.features import BasePruner

class MomentPruner(BasePruner):
    """
    Remove features with low variance or high correlation
    """

    def __init__(self, min_variance: float = 0.01, max_corr: float = 0.95):
        super().__init__()
        self.min_variance = min_variance
        self.max_corr = max_corr
        self._drop_cols = []
        self._is_fit = False

    def fit(self, df: pd.DataFrame) -> None:
        """
        Fit the feature selector to the data. This should be done on the training data only.
        """
        drop_cols = []
        for col in df.columns:
            if df[col].var() < self.min_variance:
                drop_cols.append(col)

        corr = df.corr() > self.max_corr
        for i, c1 in enumerate(corr.columns):
            for c2 in corr.columns[:i]:
                if corr.loc[c1, c2]:
                    drop_cols.append(c1)
        self._drop_cols = drop_cols
        self._is_fit = True
