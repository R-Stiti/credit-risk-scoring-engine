import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Credit Scoring Engine", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load("models/xgb_pipeline.joblib")


pipeline = load_model()
preprocessor = pipeline.named_steps['preprocessor']
xgb_model = pipeline.named_steps['simple_xgb']

st.title("Moteur de Credit Scoring")
st.markdown("Évaluation du risque de défaut de paiement à 2 ans via un modèle XGBoost optimisé.")

tab1, tab2, tab3 = st.tabs(["Simulateur Individuel", "Scoring par Lot", "Dashboard Technique"])


with tab1:
    st.markdown("**Saisissez les informations du client pour calculer sa probabilité de défaut.**")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Âge", min_value=18, max_value=110, value=45)
        rev_util = st.number_input("Taux d'utilisation du crédit renouvelable", min_value=0.0, value=0.3, format="%.3f")
        income = st.number_input("Revenu mensuel (€)", min_value=0.0, value=5000.0)
        dependents = st.number_input("Personnes à charge", min_value=0, max_value=20, value=1)

    with col2:
        debt_ratio = st.number_input("Ratio d'endettement (DebtRatio)", min_value=0.0, value=0.35)
        open_lines = st.number_input("Lignes de crédit ouvertes", min_value=0, max_value=60, value=5)
        real_estate = st.number_input("Prêts immobiliers", min_value=0, max_value=30, value=1)

    with col3:
        late_30_59 = st.number_input("Retards 30-59 jours", min_value=0, max_value=98, value=0)
        late_60_89 = st.number_input("Retards 60-89 jours", min_value=0, max_value=98, value=0)
        late_90 = st.number_input("Retards +90 jours", min_value=0, max_value=98, value=0)

    if st.button("Calculer le Risque", type="primary"):
        client_data = pd.DataFrame([{
            'RevolvingUtilizationOfUnsecuredLines': rev_util,
            'age': age,
            'NumberOfTime30-59DaysPastDueNotWorse': late_30_59,
            'DebtRatio': debt_ratio,
            'MonthlyIncome': income,
            'NumberOfOpenCreditLinesAndLoans': open_lines,
            'NumberOfTimes90DaysLate': late_90,
            'NumberRealEstateLoansOrLines': real_estate,
            'NumberOfTime60-89DaysPastDueNotWorse': late_60_89,
            'NumberOfDependents': dependents
        }], dtype=float)

        proba = pipeline.predict_proba(client_data)[:, 1][0]

        st.divider()
        if proba > 0.5:
            st.error(f"Risque de Défaut Élevé : {proba * 100:.1f}%")
        else:
            st.success(f"Profil Solvable - Risque : {proba * 100:.1f}%")

        st.markdown("**Analyse des facteurs de décision (SHAP)**")

        client_transformed = preprocessor.transform(client_data)
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(client_transformed)

        fig, ax = plt.subplots(figsize=(10, 4))
        plt.style.use('default')
        shap.waterfall_plot(shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=client_transformed.iloc[0],
            feature_names=client_transformed.columns
        ), show=False)

        st.pyplot(fig)


with tab2:
    st.markdown("**Audit de portefeuille : importez un fichier ou testez avec nos données.**")

    col_btn, col_upload = st.columns([1, 2])
    with col_btn:
        st.write("")
        if st.button("Utiliser l'échantillon de test"):
            st.session_state['batch_data'] = pd.read_csv("data/sample_clients.csv", index_col=0)

    with col_upload:
        uploaded_file = st.file_uploader("Ou téléversez votre propre CSV", type="csv", label_visibility="collapsed")
        if uploaded_file is not None:
            st.session_state['batch_data'] = pd.read_csv(uploaded_file, index_col=0)

    if 'batch_data' in st.session_state:
        batch_data = st.session_state['batch_data']
        st.write("Aperçu des données à évaluer :", batch_data.head())

        if st.button("Évaluer le portefeuille", type="primary"):
            predictions = pipeline.predict_proba(batch_data)[:, 1]

            results_df = batch_data.copy()
            if 'SeriousDlqin2yrs' in results_df.columns:
                results_df = results_df.drop(columns=['SeriousDlqin2yrs'])
            results_df.index.name = "ID_Client"

            results_df['Probabilité_Défaut'] = (predictions * 100).round(2).astype(str) + '%'
            results_df['Décision'] = np.where(predictions > 0.5, "Refusé", "Accepté")

            colonnes_a_afficher = ['Probabilité_Défaut', 'Décision', 'age', 'MonthlyIncome', 'DebtRatio']
            st.dataframe(results_df[colonnes_a_afficher], use_container_width=True)

            csv_export = results_df.to_csv(index=True).encode('utf-8')
            st.download_button(
                label="Télécharger le rapport complet",
                data=csv_export,
                file_name="rapport_scoring.csv",
                mime="text/csv"
            )

with tab3:
    st.markdown("**Validation du Modèle & Logique Métier**")

    col_k, col_s = st.columns(2)
    with col_k:
        st.info("**Performances Kaggle :** AUC 0.868 | Gini 73.30%")
        try:
            st.image("images/kaggle_xgb_score.png", caption="Score Officiel sur Test Privé")
        except:
            st.write("[Capture d'écran Kaggle manquante]")

    with col_s:
        st.info("**Explicabilité Globale (SHAP)**")
        try:
            st.image("images/shap_summary_global.png", caption="Impact des variables sur l'ensemble du dataset")
        except:
            st.write("[Graphique SHAP manquant]")