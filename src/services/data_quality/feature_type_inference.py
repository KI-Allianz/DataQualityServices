import pandas as pd
def detect_feature_types(df: pd.DataFrame) -> dict:
    types = {}
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            types[col] = "datetime"
        elif pd.api.types.is_numeric_dtype(s):
            types[col] = "numeric"
        else:
            # try parse dates, else categorical
            try:
                pd.to_datetime(s, errors="raise")
                types[col] = "datetime"
            except Exception:
                types[col] = "categorical"
    return types
