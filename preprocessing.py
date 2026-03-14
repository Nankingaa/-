import pandas as pd

def preprocess_data(df):

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    df = df.dropna()

    df = pd.get_dummies(df, drop_first=True)

    return df