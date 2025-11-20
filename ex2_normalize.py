import pandas as pd

def normalize_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Normalize a column so values scale between 0 and 1.
    """

    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found")

    col = df[column]

    min_val = col.min()
    max_val = col.max()

    # Avoid division by zero (if all values equal)
    if min_val == max_val:
        df[column] = 0
        return df

    df[column] = (col - min_val) / (max_val - min_val)

    return df
