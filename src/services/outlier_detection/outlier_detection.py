import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

def detect_outliers(df: pd.DataFrame, contamination: float = 0.05, random_state: int = 42) -> pd.DataFrame:
    out = df.copy()
    num = out.select_dtypes(include=[np.number])
    if num.empty:
        out["_outlier"] = False
        return out
    model = IsolationForest(contamination=contamination, random_state=random_state)
    labels = model.fit_predict(num)
    out["_outlier"] = (labels == -1)
    return out
