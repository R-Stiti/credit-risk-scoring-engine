from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


class CodesFlagger(BaseEstimator, TransformerMixin):
    def __init__(self, columns_list : list, codes_list : list):
        self.columns_list = columns_list
        self.codes_list = codes_list

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None, **fit_params):
        df = X.copy()

        col_mask = df[self.columns_list].isin(self.codes_list).any(axis=1)
        df[self.columns_list] = df[self.columns_list].replace(self.codes_list, 0)

        flag_name = f"Has_System_Error_{'_'.join(map(str,self.codes_list))}"
        df[flag_name] = col_mask.astype(int)
        return df


def get_cleaner_pipeline() -> Pipeline:
    dependents_imputer = SimpleImputer(strategy='constant', fill_value=0)
    income_imputer = SimpleImputer(strategy='median', add_indicator=True)
    age_imputer = SimpleImputer(missing_values=0, strategy='median')

    imputation_transformer = ColumnTransformer([
        ('dependants', dependents_imputer, ['NumberOfDependents']),
        ('income', income_imputer, ['MonthlyIncome']),
        ('age', age_imputer, ["age"])
        ], remainder='passthrough', verbose_feature_names_out=False)

    code_cols = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate"
    ]
    codes_list = [96,98]
    codes_flagger = CodesFlagger(columns_list=code_cols, codes_list=codes_list)

    # build final
    pipeline = Pipeline([
        ('codes flagger', codes_flagger),
        ('imputer', imputation_transformer),
    ])

    pipeline.set_output(transform='pandas')
    return pipeline
