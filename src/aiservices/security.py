import pandas as pd


DANGEROUS_SPREADSHEET_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def neutralize_spreadsheet_formula(value):
    if isinstance(value, str) and value.startswith(DANGEROUS_SPREADSHEET_PREFIXES):
        return "'" + value
    return value


def sanitize_for_spreadsheet_export(df: pd.DataFrame) -> pd.DataFrame:
    if hasattr(df, "map"):
        return df.map(neutralize_spreadsheet_formula)
    return df.applymap(neutralize_spreadsheet_formula)
