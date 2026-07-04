import pandas as pd


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


