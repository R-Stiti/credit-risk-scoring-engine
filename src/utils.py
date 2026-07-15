import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score


def etude_sauts_quantiles(df, colonne):
    liste_quantiles = [0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.990, 0.995, 0.996, 0.997, 0.998, 0.999,
                       1]
    ecarts = pd.DataFrame(
        {'quantile': df[colonne].quantile(liste_quantiles),
         'diff': df[colonne].quantile(liste_quantiles).diff()})
    return ecarts

def cap_outliers_quantile(df, colonne, quantile_seuil):
    cap = df[colonne].quantile(quantile_seuil)
    df[colonne] = df[colonne].clip(lower=0, upper=cap)
    print(f"{colonne} : Plafonné au quantile {quantile_seuil} (Valeur max = {cap:.2f})")

    return df

def cap_outliers_explicit_value(df, colonne, cap):
    df[colonne] = df[colonne].clip(lower=0, upper=cap)
    return df

def clean_credit_data(df, income_imputer:SimpleImputer, age_median):
    df = df.copy()

    # Traitement des données manquantes
    df["NumberOfDependents"] = df["NumberOfDependents"].fillna(0)

    new_income = income_imputer.transform((df[["MonthlyIncome"]]))
    df["MonthlyIncome"] = new_income[:,0]
    df["MissingIncomeFlag"] = new_income[:,1]

    # Traitement des outliers
    df.loc[df["age"]== 0, "age"] = age_median
    df = cap_outliers_explicit_value(df, "MonthlyIncome", 72000)

    labels_past_due = [
        "NumberOfTime30-59DaysPastDueNotWorse",
        "NumberOfTime60-89DaysPastDueNotWorse",
        "NumberOfTimes90DaysLate"
    ]
    masque_past_due_98_96 = df[labels_past_due].isin([96, 98]).any(axis=1)
    df["Has_System_Error_98"] = masque_past_due_98_96.astype(int)
    df.loc[masque_past_due_98_96, labels_past_due] = 0

    df = cap_outliers_explicit_value(df, "RevolvingUtilizationOfUnsecuredLines", 3.28)
    masque_missing_income_label = df["MissingIncomeFlag"] == 1
    df.loc[masque_missing_income_label, "DebtRatio"] = df.loc[masque_missing_income_label, "DebtRatio"] / df.loc[
        masque_missing_income_label, "MonthlyIncome"]
    df = cap_outliers_explicit_value(df, "DebtRatio", 2.46)

    return df


def evaluate_model_accuracy(model, X, y):

    predictions = model.predict_proba(X)[:,1]
    auc_roc = roc_auc_score(y, predictions)
    gini = 2*auc_roc - 1

    return auc_roc, gini

def display_model_accuracy(auc_roc, gini):

    print("Performances du modèle :", f"AUC-ROC : {auc_roc:.3f} \nGini : {gini:.3f} (Soit {gini*100:.2f}%)")

    return None

