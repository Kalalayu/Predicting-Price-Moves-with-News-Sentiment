# src/explainability.py

import shap
from lime.lime_tabular import LimeTabularExplainer


def shap_explain(model, X):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values, X)
    return shap_values


def lime_explain(model, X):
    explainer = LimeTabularExplainer(
        training_data=X.values,
        feature_names=X.columns.tolist(),
        class_names=["Down", "Up"],
        mode="classification"
    )
    exp = explainer.explain_instance(
        X.values[0],
        model.predict_proba
    )
    return exp
