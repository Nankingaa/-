from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

def evaluate_model(model, scaler, X_test, y_test):

    X_test_scaled = scaler.transform(X_test)

    y_pred = model.predict(X_test_scaled)

    print(classification_report(y_test, y_pred))

    y_prob = model.predict_proba(X_test_scaled)[:,1]

    fpr, tpr, _ = roc_curve(y_test, y_prob)

    plt.plot(fpr, tpr)

    plt.xlabel("False Positive Rate")

    plt.ylabel("True Positive Rate")

    plt.title("ROC Curve")

    plt.show()

    auc = roc_auc_score(y_test, y_prob)

    print("AUC:", auc)