import pandas as pd

def risk_segmentation(model, scaler, X_test):

    X_test_scaled = scaler.transform(X_test)

    churn_prob = model.predict_proba(X_test_scaled)[:,1]

    result = X_test.copy()

    result["Churn_Probability"] = churn_prob

    def risk_level(p):

        if p > 0.7:
            return "High Risk"

        elif p > 0.4:
            return "Medium Risk"

        else:
            return "Low Risk"

    result["Risk_Level"] = result["Churn_Probability"].apply(risk_level)

    print(result["Risk_Level"].value_counts())

    return result