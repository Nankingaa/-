import pandas as pd

from sklearn.model_selection import train_test_split

from preprocessing import preprocess_data
from eda import basic_eda, churn_distribution
from model import train_model
from evaluation import evaluate_model
from churn_analysis import risk_segmentation


df = pd.read_csv("data/Telco-Customer-Churn.csv")

basic_eda(df)

churn_distribution(df)

df = preprocess_data(df)

X = df.drop("Churn_Yes", axis=1)

y = df["Churn_Yes"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model, scaler = train_model(X_train, y_train)

evaluate_model(model, scaler, X_test, y_test)

risk_segmentation(model, scaler, X_test)