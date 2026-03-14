import matplotlib.pyplot as plt
import seaborn as sns

def basic_eda(df):

    print("Dataset Shape:")
    print(df.shape)

    print("\nDataset Info:")
    print(df.info())

    print("\nStatistical Summary:")
    print(df.describe())

def churn_distribution(df):

    sns.countplot(x="Churn", data=df)

    plt.title("Customer Churn Distribution")

    plt.show()

def contract_churn(df):

    sns.countplot(x="Contract", hue="Churn", data=df)

    plt.title("Churn vs Contract Type")

    plt.show()

def monthly_charge_churn(df):

    sns.boxplot(x="Churn", y="MonthlyCharges", data=df)

    plt.title("Monthly Charges vs Churn")

    plt.show()