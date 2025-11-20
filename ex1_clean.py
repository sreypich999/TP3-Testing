import pandas as pd

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows and rows containing null values.
    """
    df = df.drop_duplicates()   # remove duplicate rows
    df = df.dropna()            # remove null values
    return df
