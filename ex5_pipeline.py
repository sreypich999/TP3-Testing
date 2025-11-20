import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# 1. Load data
def load_data():
    data = pd.DataFrame({
        "age": [20, 25, 30, 35, 40],
        "income": [200, 300, 400, 500, 600],
        "target": [0, 1, 0, 1, 1]
    })
    return data


# 2. Train model
def train_model(df):
    X = df[["age", "income"]]
    y = df["target"]

    model = LogisticRegression()
    model.fit(X, y)
    return model


# 3. Evaluate model
def evaluate_model(model, df):
    X = df[["age", "income"]]
    y = df["target"]

    predictions = model.predict(X)
    acc = accuracy_score(y, predictions)
    return acc
