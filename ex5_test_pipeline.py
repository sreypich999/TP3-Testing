import pandas as pd
from ex5_pipeline import load_data, train_model, evaluate_model


def test_data_loads_correctly():
    df = load_data()

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert set(["age", "income", "target"]).issubset(df.columns)


def test_model_trains_successfully():
    df = load_data()
    model = train_model(df)

    # model should have a predict() function
    assert hasattr(model, "predict")


def test_accuracy_between_0_and_1():
    df = load_data()
    model = train_model(df)
    acc = evaluate_model(model, df)

    assert 0 <= acc <= 1
