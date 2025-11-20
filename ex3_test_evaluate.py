import pytest
from ex3_evaluate import evaluate_model

def test_perfect_accuracy():
    y_true = [1, 0, 1, 0]
    y_pred = [1, 0, 1, 0]

    result = evaluate_model(y_true, y_pred)

    assert result["accuracy"] == 1.0


def test_f1_score_zero_when_all_wrong():
    y_true = [1, 1, 1]
    y_pred = [0, 0, 0]

    result = evaluate_model(y_true, y_pred)

    assert result["f1_score"] == 0.0


def test_output_keys_exist():
    y_true = [1, 0]
    y_pred = [1, 0]

    result = evaluate_model(y_true, y_pred)

    assert "accuracy" in result
    assert "f1_score" in result
