import os
import pandas as pd


def generate_kaggle_submission(model, X_test, test_ids, filename="submission.csv",
                               output_dir="../submissions"):
    """
    Génère, vérifie et sauvegarde le fichier CSV au format de Kaggle (Id, Probability).
    """
    probabilities = model.predict_proba(X_test)[:, 1]
    submission_df = pd.DataFrame({
        "Id": test_ids,
        "Probability": probabilities
    })

    if submission_df.isna().sum().sum() > 0:
        raise ValueError("Le fichier généré contient des valeurs NaN")
    if not (submission_df["Probability"].between(0, 1).all()):
        raise ValueError("Certaines probabilités ne sont pas comprises entre 0 et 1")

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    submission_df.to_csv(output_path, index=False)

    return submission_df