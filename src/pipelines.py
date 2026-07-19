import numpy as np
from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from kneed import KneeLocator


class CodesFlagger(BaseEstimator, TransformerMixin):
    def __init__(self, columns_list : list, codes_list : list):
        self.columns_list = columns_list
        self.codes_list = codes_list

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        df = X.copy()

        col_mask = df[self.columns_list].isin(self.codes_list).any(axis=1)
        df[self.columns_list] = df[self.columns_list].replace(self.codes_list, 0)

        flag_name = f"Has_System_Error_{'_'.join(map(str,self.codes_list))}"
        df[flag_name] = col_mask.astype(int)
        return df

class QuantileCapper(BaseEstimator, TransformerMixin):
    caps_ : dict

    def __init__(self, columns_list : list):
        self.columns_list = columns_list

    def fit(self, X:DataFrame, y=None):
        self.caps_ = {}
        quantiles = np.linspace(0.90, 0.999, 100)
        quantiles_curves = X[self.columns_list].quantile(quantiles)

        for column in self.columns_list:
            y_quantile = quantiles_curves[column]
            knee_locator = KneeLocator(
                x=quantiles,
                y=y_quantile,
                curve='convex',
                direction='increasing'
            )

            if knee_locator.knee_y is not None:
                self.caps_[column] = knee_locator.knee_y
            else :
                self.caps_[column] = y_quantile.iloc[-1]
        return self

    def transform(self, X, y=None):
        df = X.copy()
        for column in self.columns_list:
            df[column] = df[column].clip(upper=self.caps_[column])

        return df

class DebtRatioFixer(BaseEstimator, TransformerMixin):
    debt_ratio_median_ : float

    def __init__(self, debt_ratio_col = "DebtRatio",
                 income_col = "MonthlyIncome",
                 flag_missing_income_col = "missingindicator_MonthlyIncome"
                 ):
        self.debt_ratio_col = debt_ratio_col
        self.income_col = income_col
        self.flag_missing_income_col = flag_missing_income_col

    def fit(self, X:DataFrame, y=None):
        mask = (X[self.flag_missing_income_col] == 0)

        if mask.any():
            self.debt_ratio_median_ = X.loc[mask, self.debt_ratio_col].median()
        else:
            self.debt_ratio_median_ = 0.3
        return self

    def transform(self, X, y=None):
        df = X.copy()

        real_ratio_mask = (df[self.flag_missing_income_col] == 0)

        df.loc[real_ratio_mask,"MonthlyDebtAmount"] = (
                df.loc[real_ratio_mask,self.debt_ratio_col] *
                df.loc[real_ratio_mask,self.income_col]
        )
        df.loc[~real_ratio_mask,"MonthlyDebtAmount"] = df.loc[~real_ratio_mask, self.debt_ratio_col]

        df.loc[real_ratio_mask, "TrueDebtRatio"] = df.loc[real_ratio_mask, self.debt_ratio_col]
        df.loc[~real_ratio_mask, "TrueDebtRatio"] = self.debt_ratio_median_

        df.drop(labels=[self.debt_ratio_col], axis='columns', inplace=True)
        return df

def get_cleaner_pipeline() -> Pipeline:
    dependents_imputer = SimpleImputer(strategy='constant', fill_value=0)
    income_imputer = SimpleImputer(strategy='median', add_indicator=True)
    age_imputer = SimpleImputer(missing_values=0, strategy='median')

    imputation_transformer = ColumnTransformer([
        ('dependants', dependents_imputer, ['NumberOfDependents']),
        ('income', income_imputer, ['MonthlyIncome']),
        ('age', age_imputer, ["age"])
        ], remainder='passthrough', verbose_feature_names_out=False).set_output(transform='pandas')

    code_cols = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate"
    ]
    codes_list = [96,98]
    codes_flagger = CodesFlagger(columns_list=code_cols, codes_list=codes_list)

    debt_fixer = DebtRatioFixer()

    cols_to_cap = [
        "MonthlyIncome",
        "RevolvingUtilizationOfUnsecuredLines",
        "MonthlyDebtAmount",
        "TrueDebtRatio"
    ]
    quantile_capper = QuantileCapper(columns_list=cols_to_cap)

    # build final
    pipeline = Pipeline([
        ('codes flagger', codes_flagger),
        ('imputer', imputation_transformer),
        ("debt_fixer", debt_fixer),
        ('capper', quantile_capper)
    ])

    return pipeline
