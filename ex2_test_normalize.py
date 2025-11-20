import pandas as pd
import pytest
from ex2_normalize import normalize_column


def test_values_are_between_0_and_1():
    df = pd.DataFrame({"A": [10, 20, 30]})
    normalized = normalize_column(df, "A")

    # All values must be between 0 and 1
    assert normalized["A"].min() >= 0
    assert normalized["A"].max() <= 1


def test_output_length_matches_input():
    df = pd.DataFrame({"A": [10, 20, 30, 40]})
    normalized = normalize_column(df, "A")

    # Length should be the same
    assert len(normalized) == len(df)


def test_invalid_column_raises_keyerror():
    df = pd.DataFrame({"A": [10, 20, 30]})

    # Asking for a non-existing column → must raise KeyError
    with pytest.raises(KeyError):
        normalize_column(df, "B")
