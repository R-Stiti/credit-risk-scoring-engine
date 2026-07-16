from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def get_cleaner_pipeline() -> Pipeline:

    # Données manquantes
    dependents_imputer = SimpleImputer(strategy='constant', fill_value=0)
    income_imputer = SimpleImputer(strategy='median', add_indicator=True)

    return Pipeline([])
