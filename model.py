import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv('financial_data.csv')

features = ['annual_income', 'dti', 'int_rate', 'total_acc']

def model_train():
    X = df[features] # dataframe
    Y = df['loan_amount'] # series

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    model = LinearRegression()

    model.fit(X_train, Y_train)

    model.predict(X_test)

    return model, features, X, Y