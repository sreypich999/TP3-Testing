import pandas as pd
from ex1_clean import clean_data

def test_duplicates_are_removed():
    # Data with a duplicate row
    df = pd.DataFrame({
        "A": [1, 1, 2],
        "B": [3, 3, 4],
    })

    cleaned = clean_data(df)

    # Duplicate removed → only 2 rows left
    assert len(cleaned) == 2


def test_null_values_are_dropped():
    # Data with null values
    df = pd.DataFrame({
        "A": [1, None, 2],
        "B": [3, 4, None],
    })

    cleaned = clean_data(df)

    # Ensure NO null values remain
    assert cleaned.isnull().sum().sum() == 0


def test_row_count_decreases():
    # Contains duplicate and null
    df = pd.DataFrame({
        "A": [1, 1, None],
        "B": [3, 3, 4],
    })

    cleaned = clean_data(df)

    # Cleaned dataframe MUST have fewer rows
    assert len(cleaned) < len(df)
