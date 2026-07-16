from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def get_cleaner_pipeline() -> Pipeline:

    # Données manquantes
    dependents_imputer = SimpleImputer(strategy='constant', fill_value=0)
    income_imputer = SimpleImputer(strategy='median', add_indicator=True)
    missing_values_transformer = ColumnTransformer([
        ('dependants', dependents_imputer, ['NumberOfDependents']),
        ('income', income_imputer, ['MonthlyIncome']),
        ], remainder='passthrough')

    return Pipeline([
        ('missing', missing_values_transformer),
    ])
