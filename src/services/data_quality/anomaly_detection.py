import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

def anomaly_score(df: pd.DataFrame, contamination: float = 0.10) -> pd.DataFrame:
    out = df.copy()
    num = out.select_dtypes(include=[np.number])
    if num.empty:
        out["_anomaly"] = False
        return out
    X = StandardScaler().fit_transform(num.values)
    model = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
    labels = model.fit_predict(X)  # -1 anomaly
    out["_anomaly"] = (labels == -1)
    return out
