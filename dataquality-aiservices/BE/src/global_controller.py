from enum import Enum
import numpy as np
from pandas import Series
import logging
import pandas as pd
from pyod.models.iforest import IForest
import torch
import torch.nn as nn
import pandas.io.formats.style
from sklearn.preprocessing import MinMaxScaler
import os

# Flair is only required for the personal-data classifier.
# Import lazily so the backend can start even if `flair` or the installed
# `huggingface_hub` version is incompatible.
try:
    from flair.data import Sentence
    from flair.models import TextClassifier
    FLAIR_AVAILABLE = True
except Exception as e:
    Sentence = None
    TextClassifier = None
    FLAIR_AVAILABLE = False
    logging.warning(
        f"Flair import failed, personal-data classification will be disabled: {e}. "
        "Install flair==0.15.1 and a compatible huggingface-hub version."
    )

from collections import Counter
import random
import json
from datetime import datetime
import string
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as stop_words
from flask import session  # ersetzt st.session_state


#if os.getcwd().endswith("ANOMALYDETECTIONHMTEAM"):
 #   from src.models import TranAD
#else:
 #   from src.demonstrators.ANOMALYDETECTIONHMTEAM.ANOMALYDETECTIONHMTEAM.src.models import TranAD

TranAD = None
from sklearn.preprocessing import StandardScaler
import category_encoders as ce
from src.services_featuretype_personal.services_LLM import FeaturizeFile,Load_RF,FeatureExtraction

from src.autogluon.constants import (
    BINARY,
    LARGE_DATA_THRESHOLD,
    MULTICLASS,
    MULTICLASS_UPPER_LIMIT,
    REGRESS_THRESHOLD_LARGE_DATA,
    REGRESS_THRESHOLD_SMALL_DATA,
    REGRESSION,
)

logger = logging.getLogger(__name__)

RandomForest_classes = ['numeric', 'categorical', 'datetime',
                        'sentence', 'url', 'embedded-number',
                        'list', 'not-generalizable', 'context-specific']


def add_metadata(key, metadata):
    if "metadata_store" not in session or not isinstance(session["metadata_store"],dict):
        session["metadata_store"] = {}
    session["metadata_store"][key] = metadata

def get_metadata(key):
    if "metadata_store" in session and key in session["metadata_store"]:
        return session["metadata_store"][key]
    return None

def get_all_metadata():
    if "metadata_store" in session:
        return session["metadata_store"]
    return {}

def load_model():
    if not FLAIR_AVAILABLE:
        raise RuntimeError(
            "Flair is unavailable in this environment. "
            "Personal-data classification cannot be used."
        )

    MODEL_PATH_PERSONAL = "src/best-model.pt"
    try:
        model = TextClassifier.load(MODEL_PATH_PERSONAL)
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load personal data classifier: {e}") from e


def make_predictions(df_inference, model):
    df_inference = df_inference * 1
    dataFeaturized = FeaturizeFile(df_inference)
    columns_of_interest = dataFeaturized.columns[:15]
    result = dataFeaturized.filter(columns_of_interest)
    result.insert(loc=0, column='prediction', value="")

    if model == 'RandomForest':
        dataFeaturized1 = FeatureExtraction(dataFeaturized)
        dataFeaturized1.columns = dataFeaturized1.columns.astype(str)
        y_RF = Load_RF(dataFeaturized1.values)
        result['prediction'] = y_RF
        for i in range(len(result['prediction'])):
            result.loc[i, 'prediction'] = RandomForest_classes[int(result['prediction'][i])]

    return result

def make_prediciton_personal(df_personal):
    classifier_personal = load_model()
    results = process_file(classifier_personal, df_personal)

    num_features = len(results)
    num_personal = results[results["Prediction"] == "personal"].shape[0]
    num_non_personal = num_features - num_personal

    metadata_personal = {
        "Total Features": num_features,
        "Personal Data Exists": num_personal > 0,
        "Personal Features": num_personal,
        "Non-Personal Features": num_non_personal,
        "Personal Columns": results[results["Prediction"] == "personal"]["Column"].tolist(),
        "Non-Personal Columns": results[results["Prediction"] == "non-personal"]["Column"].tolist()
    }
    metadata_df = pd.DataFrame([metadata_personal])
    return metadata_df, results

def predict_column(classifier, column_values):
    value_counts = Counter(column_values)
    most_common_values = [value for value, _ in value_counts.most_common(10)]
    sample_values = most_common_values[:min(10, len(most_common_values))]

    sentences = [Sentence(str(value)) for value in sample_values]
    classifier.predict(sentences)
    predictions = [sentence.labels[0].value for sentence in sentences]
    return "personal" if "personal" in predictions else "non-personal"

def process_file(classifier, df):
    column_predictions = {}
    for i, column in enumerate(df.columns):
        column_values = df[column].dropna().astype(str)
        prediction_class = predict_column(classifier, column_values)
        column_predictions[column] = prediction_class

    result_df = pd.DataFrame(list(column_predictions.items()), columns=["Column", "Prediction"])
    return result_df

def load_and_preprocess_data(data_anomalie_df,feature_type_inference):
    data_anomalie_df = data_anomalie_df.copy()

    if feature_type_inference is not None and not feature_type_inference.empty:
        prediction_series = feature_type_inference["prediction"].astype(str).str.strip().str.lower()
        datetime_rows = feature_type_inference[prediction_series == "datetime"]
        datetime_cols = [
            col for col in datetime_rows["Attribute_name"].astype(str).tolist()
            if col in data_anomalie_df.columns
        ]
        if datetime_cols:
            data_anomalie_df = data_anomalie_df.drop(columns=datetime_cols)

    categorical_cols = data_anomalie_df.select_dtypes(include=["object", "category", "string", "bool"]).columns.tolist()
    if categorical_cols:
        onehot_encoder = ce.OneHotEncoder(cols=categorical_cols, use_cat_names=True)
        data_anomalie_df = onehot_encoder.fit_transform(data_anomalie_df)

    data_anomalie_df = data_anomalie_df.apply(pd.to_numeric, errors="coerce")
    for col in data_anomalie_df.columns:
        if data_anomalie_df[col].isna().all():
            data_anomalie_df[col] = 0
        else:
            data_anomalie_df[col] = data_anomalie_df[col].fillna(data_anomalie_df[col].median())

    original_data = data_anomalie_df.copy()

    if data_anomalie_df.empty:
        data_anomalie_df = pd.DataFrame({"constant": np.zeros(len(original_data), dtype=float)})

    scaler = MinMaxScaler()
    normalized_data = pd.DataFrame(scaler.fit_transform(data_anomalie_df), columns=data_anomalie_df.columns)
    
    return original_data, normalized_data, data_anomalie_df.shape[1]

def convert_to_windows(data, model):
    windows = []
    w_size = model.n_window
    for i in range(len(data)):
        if i >= w_size:
            w = data[i-w_size:i]
        else:
            w = torch.cat([data[0].repeat(w_size-i, 1), data[0:i]])
        windows.append(w)
    return torch.stack(windows)

def set_random_seed(seed=10000):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def detect_anomalies_tranad(data, feats, percentile):
    set_random_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        from src.demonstrators.ANOMALYDETECTIONHMTEAM.ANOMALYDETECTIONHMTEAM.src.models import TranAD
    except Exception as e:
        raise RuntimeError(
            "TranAD cannot be imported. Time-series anomaly detection requires DGL and the ANOMALYDETECTIONHMTEAM package. "
            "If you do not need this feature, disable time-series anomaly detection or install DGL/graphbolt correctly."
        ) from e

    model = TranAD(feats).to(device)
    data_tensor = torch.tensor(data.values, dtype=torch.float).to(device)
    windows = convert_to_windows(data_tensor, model)

    model.eval()
    with torch.no_grad():
        losses, preds = [], []
        for window in windows:
            window = window.unsqueeze(0)
            elem = window[-1, :, :].view(1, window.shape[1], feats).to(device)
            z = model(window, elem)[1]
            loss = nn.MSELoss(reduction='none')(z, elem).mean(dim=[1, 2]).cpu().numpy()
            losses.append(loss)
            preds.append(z.squeeze(0).cpu().numpy())

    losses = np.array(losses).flatten()
    preds = np.array(preds)
    threshold = np.percentile(losses, percentile)
    anomalies = (losses > threshold).astype(int)
    return anomalies, losses, preds, threshold

def detect_anomalies(df, model=IForest(), contamination=0.1, scale_data=True):
    """
    Detect anomalies in a given DataFrame using PyOD.
    
    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - model (pyod model, default=IForest()): The anomaly detection model to use.
    - contamination (float, default=0.1): Expected proportion of anomalies.
    - scale_data (bool, default=True): Whether to standardize the data before detection.
    
    Returns:
    - pd.DataFrame: Original DataFrame with an added 'Anomaly' column (0=normal, 1=anomaly).
    """
    df_model = df.copy()

    datetime_cols = []
    for col in df_model.columns:
        if pd.api.types.is_datetime64_any_dtype(df_model[col]):
            datetime_cols.append(col)
            continue

        if pd.api.types.is_object_dtype(df_model[col]) or pd.api.types.is_string_dtype(df_model[col]):
            parsed = pd.to_datetime(df_model[col], errors="coerce")
            if parsed.notna().mean() >= 0.8:
                df_model[col] = parsed
                datetime_cols.append(col)

    for col in datetime_cols:
        parsed = pd.to_datetime(df_model[col], errors="coerce")
        df_model[f"{col}_year"] = parsed.dt.year
        df_model[f"{col}_month"] = parsed.dt.month
        df_model[f"{col}_day"] = parsed.dt.day
        df_model[f"{col}_hour"] = parsed.dt.hour
        df_model = df_model.drop(columns=[col])

    categorical_cols = df_model.select_dtypes(include=["object", "category", "string", "bool"]).columns.tolist()
    if categorical_cols:
        encoder = ce.OneHotEncoder(cols=categorical_cols, use_cat_names=True)
        df_model = encoder.fit_transform(df_model)

    df_model = df_model.apply(pd.to_numeric, errors="coerce")

    for col in df_model.columns:
        if df_model[col].isna().all():
            df_model[col] = 0
        else:
            df_model[col] = df_model[col].fillna(df_model[col].median())

    df_clean = df_model.select_dtypes(include=['number'])
    if df_clean.empty:
        df_result = df.copy()
        df_result["Anomaly"] = 0
        return df_result, 0

    # Scale the data if required
    if scale_data:
        scaler = StandardScaler()
        data = scaler.fit_transform(df_clean)
    else:
        data = df_clean.values
   
    # Fit the model and predict anomalies
    model.set_params(contamination=contamination)
    model.fit(data)
    df_result = df.copy()
    df_result["Anomaly"] = model.predict(data)  # 0 = normal, 1 = anomaly
    treshhold = model.threshold_
    

    return df_result,treshhold

# (weitere Funktionen wie highlight_xxx, extract_metadata, etc. sind unverändert)
def one_hot_encode_columns(
    df: pd.DataFrame,
    feature_inference_df: pd.DataFrame,
    df_anomalies: pd.DataFrame | None = None,
    df_personal: pd.DataFrame | None = None,
    target_col: str | None = None,
) -> pd.DataFrame:
    """
    One-hot-encodes categorical columns inferred from `feature_inference_df`, with optional
    row/column filtering based on `df_anomalies` and `df_personal`.

    The `target_col` (if provided) is NEVER modified:
      - never dropped (even if marked Personal),
      - never encoded or treated as datetime,
      - values & dtype are restored after all transforms.

    Parameters
    ----------
    df : pd.DataFrame
    feature_inference_df : pd.DataFrame
        Needs columns ["Attribute_name", "prediction"] with values like "datetime", "categorical".
    df_anomalies : pd.DataFrame, optional
        Last column is "Anomaly" (1 marks anomalous rows). Those rows are removed before encoding.
    df_personal : pd.DataFrame, optional
        Columns ["Column", "Prediction"]. Rows with Prediction == "Personal" drop that Column.
    target_col : str, optional
        Name of the target column in `df` that must not be altered.

    Returns
    -------
    pd.DataFrame
        Transformed DataFrame.
    """
    df_out = df.copy()
    feature_inference_df = feature_inference_df.copy()

    if "prediction" not in feature_inference_df.columns:
        raise ValueError("`feature_inference_df` is missing required column: prediction")

    feature_inference_df["prediction"] = (
        feature_inference_df["prediction"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # --- Validate / stash the target for protection ---
    target_series = None
    if target_col is not None:
        if target_col not in df_out.columns:
            raise ValueError(f"`target_col`='{target_col}' not found in df.")
        # Stash original target (values + dtype) for restore at the end
        target_series = df_out[target_col].copy()

    # --- Remove personal columns first (if provided), but NEVER the target ---
    if df_personal is not None:
        required_cols = {"Column", "Prediction"}
        missing_req = required_cols - set(df_personal.columns)
        if missing_req:
            raise ValueError(f"`df_personal` is missing required columns: {sorted(missing_req)}")

        personal_cols = (
            df_personal.loc[df_personal["Prediction"].astype(str).str.lower() == "personal", "Column"]
            .dropna().astype(str).tolist()
        )
        if personal_cols:
            # Exclude target from being dropped
            cols_to_drop = [c for c in personal_cols if c in df_out.columns and c != target_col]
            if cols_to_drop:
                df_out = df_out.drop(columns=cols_to_drop, errors="ignore")

    # --- Remove anomalous rows next (if provided) ---
    if df_anomalies is not None and len(df_anomalies) > 0:
        # Use the last column (expected "Anomaly")
        anomaly_series = df_anomalies.iloc[:, -1]
        # Align by position if lengths match, else by index (missing -> non-anomaly)
        if len(anomaly_series) == len(df_out):
            mask = pd.Series(anomaly_series.values, index=df_out.index)
        else:
            mask = anomaly_series.reindex(df_out.index).fillna(0)
        mask_bool = mask.astype(int) == 1
        df_out = df_out.loc[~mask_bool]

        # Re-align target stash to current index if present
        if target_series is not None:
            target_series = target_series.reindex(df_out.index)

    # --- Expand datetime features (first inferred datetime only), but NEVER on target ---
    dt_rows = feature_inference_df[feature_inference_df["prediction"] == "datetime"]
    if len(dt_rows) >= 1:
        dt_col = dt_rows["Attribute_name"].iloc[0]
        if dt_col != target_col and dt_col in df_out.columns:
            # Assumes `encode_datetime_features` is defined elsewhere
            df_out = encode_datetime_features(df_out, dt_col)

    # --- Determine categorical columns to encode (exclude target and missing) ---
    row_cat = feature_inference_df[feature_inference_df["prediction"] == "categorical"]
    columns_to_encode = (
        row_cat["Attribute_name"].astype(str).tolist()
        if "Attribute_name" in row_cat
        else []
    )
    columns_to_encode = [c for c in columns_to_encode if c in df_out.columns and c != target_col]

    if not columns_to_encode:
        # Final safeguard: restore target exactly as-original
        if target_series is not None:
            df_out[target_col] = target_series
        return df_out

    # --- One-hot encode (non-target only) ---
    encoder = ce.OneHotEncoder(cols=columns_to_encode, use_cat_names=True)
    encoded_df = encoder.fit_transform(df_out)

    # --- FINAL: restore target exactly (values + dtype), ensuring no accidental change ---
    if target_series is not None:
        encoded_df[target_col] = target_series

    return encoded_df
def encode_datetime_features(df, column_name):
    # Ensure the column is in datetime format
    df[column_name] = pd.to_datetime(df[column_name])

    # Extract basic datetime features
    df[f'{column_name}_hour'] = df[column_name].dt.hour
    df[f'{column_name}_day'] = df[column_name].dt.day
    df[f'{column_name}_dayofweek'] = df[column_name].dt.dayofweek
    df[f'{column_name}_month'] = df[column_name].dt.month
    df[f'{column_name}_is_weekend'] = df[column_name].dt.dayofweek.isin([5, 6]).astype(int)

    # Cyclical encoding
    df[f'{column_name}_hour_sin'] = np.sin(2 * np.pi * df[f'{column_name}_hour'] / 24)
    df[f'{column_name}_hour_cos'] = np.cos(2 * np.pi * df[f'{column_name}_hour'] / 24)

    df[f'{column_name}_dayofweek_sin'] = np.sin(2 * np.pi * df[f'{column_name}_dayofweek'] / 7)
    df[f'{column_name}_dayofweek_cos'] = np.cos(2 * np.pi * df[f'{column_name}_dayofweek'] / 7)

    #df[f'{column_name}_month_sin'] = np.sin(2 * np.pi * df[f'{column_name}_month'] / 12)
    #df[f'{column_name}_month_cos'] = np.cos(2 * np.pi * df[f'{column_name}_month'] / 12)
    df= df.drop(columns=[column_name])
    

    return df
def one_hot_encode_df(df):
    columnames = list(df.columns)
    for c in df.columns:
        type = infer_problem_type(df[c])
        if type not in ["multiclass","binary"]:
            columnames.remove(c)
    #columnames.remove(get_metadata("Target variable"))
    #print(columnames) #debug
    return one_hot_encode_columns(df, columnames)

def detect_pictures(format):
    image_formats = [
    "png", "jpg", "jpeg", "gif", "bmp", "tiff", "tif", "svg", "webp",
    "ico", "heif", "heic", "raw", "eps", "pdf", "psd", "ai", "xcf",
    "jp2", "j2k", "jxr", "hdp", "wdp", "dds", "exr", "pbm", "pgm",
    "ppm", "pnm", "pfm"
    ]
    name = name.name.split(".")[1]

    if name in image_formats:
        return "picture"


    return "no_picture"

def highlight_columns(df: pd.DataFrame, columns: list) -> pd.io.formats.style.Styler:
   
    if isinstance(columns, str):
        columns = [columns]  # Convert single column name to list

    def highlight(val, column):
        if column in columns:
            return "background-color: orange"
        return "background-color: white"
    
    styled_df = df.style.apply(lambda x: [highlight(v, x.name) for v in x], axis=0)
    
    return styled_df


def highlight_imputed_cells(original_df: pd.DataFrame, imputed_df: pd.DataFrame) -> pd.io.formats.style.Styler:
    
    
    missing_mask = original_df.isna()
    
    def highlight(val, mask):
        return "background-color: orange" if mask else "background-color: white"
    
    styled_df = imputed_df.style.apply(lambda x: [highlight(v, missing_mask.loc[x.name, col]) for col, v in x.items()], axis=1)
    
    return styled_df



def highlight_anomaly_cells(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    
    def highlight_anomaly(val):
        if val == 0:
            return "background-color: lightgreen"
        elif val == 1:
            return "background-color: orange"
        else:
            return ""

    styled_df = df.style.applymap(highlight_anomaly, subset=['Anomaly'])
    
    return styled_df



def extract_metadata(df: pd.DataFrame,feature_type_inference) -> dict:
   
    #feature_type_inference = pd.DataFrame(feature_type_inference)
    metadata = {
        "Number of Rows": df.shape[0],
        "Number of Columns": df.shape[1],
        "Column Names": list(df.columns),
        "Missing Values": df.isnull().sum().to_dict(),
        "Duplicate Rows": df.duplicated().sum(),
        "Unique Values Per Column": df.nunique().to_dict(),
    }

    # Initialize DataFrame to store feature-level statistics
    df_features = pd.DataFrame()

    for i, col in enumerate(df.columns):
        current_column = df[col]
        
        # General statistics
        total_vals = len(current_column)
        num_nans = current_column.isna().sum()
        perc_nans = (num_nans / total_vals) * 100 if total_vals > 0 else 0
        num_of_dist_val = current_column.nunique()
        perc_dist_val = (num_of_dist_val / total_vals) * 100 if total_vals > 0 else 0
        

        row = feature_type_inference[feature_type_inference['Attribute_name'] == current_column.name]
        
        if  row["prediction"].iloc[0]=="numeric":
             mean = current_column.mean()
             
        else :
            np.nan
       
        if row["prediction"].iloc[0]=="numeric":
             std_dev = current_column.std()     
        else :
            np.nan

        if row["prediction"].iloc[0]=="numeric":
            min_val = current_column.min() 
        else:
            np.nan

        if row["prediction"].iloc[0]=="numeric":
            max_val = current_column.max()
        else:
            np.nan
        
        # Random samples
        non_nan_values = current_column.dropna().tolist()
        random_samples = random.sample(non_nan_values, min(5, len(non_nan_values))) if non_nan_values else [np.nan] * 5

        # String-based metrics (for text columns)
        if row["prediction"].iloc[0]=="sentence" or row["prediction"].iloc[0]=="categorical":
            word_counts = current_column.dropna().apply(lambda x: len(str(x).split()))
            stopword_counts = current_column.dropna().apply(lambda x: sum(1 for word in str(x).split() if word.lower() in stop_words))
            char_counts = current_column.dropna().apply(lambda x: len(str(x)))
            whitespace_counts = current_column.dropna().apply(lambda x: str(x).count(" "))
            delim_counts = current_column.dropna().apply(lambda x: sum(str(x).count(d) for d in string.punctuation))
            is_long_sentence = (word_counts > 15).sum() #we want to retrieve the cars with prices less than 20,000 you might try the followin

            mean_word_count = word_counts.mean()
            std_dev_word_count = word_counts.std()
            mean_stopword_total = stopword_counts.mean()
            stdev_stopword_total = stopword_counts.std()
            mean_char_count = char_counts.mean()
            stdev_char_count = char_counts.std()
            mean_whitespace_count = whitespace_counts.mean()
            stdev_whitespace_count = whitespace_counts.std()
            mean_delim_count = delim_counts.mean()
            stdev_delim_count = delim_counts.std()
        else:
            mean_word_count = std_dev_word_count = mean_stopword_total = stdev_stopword_total = np.nan
            mean_char_count = stdev_char_count = mean_whitespace_count = stdev_whitespace_count = np.nan
            mean_delim_count = stdev_delim_count = is_long_sentence = np.nan
        
        df_features = FeaturizeFile(df)
      
    #metadata["Per Column Statistics"] = df_features

    return metadata,df_features


def metadata_to_dataframe(metadata):
    # Convert the dictionary (excluding the nested DataFrame) into a DataFrame
    meta_df = pd.DataFrame.from_dict({k: [v] for k, v in metadata.items() if k != "Per Column Statistics"})
    
    # Extract the 'Per Column Statistics' DataFrame if it exists
    df_features = metadata.get("Per Column Statistics")
    
    # Concatenate both DataFrames if df_features exists
    if isinstance(df_features, pd.DataFrame):
        result_df = pd.concat([meta_df.T, df_features], axis=1)
    else:
        result_df = meta_df.T
    
    return result_df



def impute_mean_mode(df_to_impute):
    """Function to impute missing values with mean for numeric columns and mode for categorical columns."""
    imputation_info = []
    for col in df_to_impute.columns:
        if df_to_impute[col].dtype in [np.float64, np.int64]:
            mean_value = df_to_impute[col].mean()
            df_to_impute[col].fillna(mean_value, inplace=True)
            imputation_info.append((col, 'Mean', mean_value))
        else:
            mode = df_to_impute[col].mode(dropna=True)
            mode_value = mode.iloc[0] if not mode.empty else ""
            df_to_impute[col].fillna(mode_value, inplace=True)
            imputation_info.append((col, 'Mode', mode_value))
    return df_to_impute, imputation_info




# Metadaten eines DataFrame berechnen in einem dict
def calculate_dataframe_metadata(df):
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("Die Eingabe muss ein gültiger Pandas-DataFrame sein.")

    metadata = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "column_names": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": df.isnull().sum().to_dict(),
    }

    return metadata


# enum for the different dataset types
class Dataset_Type(Enum): #enum to manually set the type of the dataset
    DEFAULT = "default"
    TIME_SERIES = "time series dataset"
    TABULAR = "tabular dataset"


#method to get all possible problem types of a tabular dataset
def get_problem_type_list():
    problem_type_list = []
    problem_type_list.append(BINARY)
    problem_type_list.append(MULTICLASS)
    problem_type_list.append(REGRESSION)
    return problem_type_list

def check_time_series(df: pd.DataFrame,feature_df:pd.DataFrame) -> str:
    """
    Iterates through a DataFrame, extracts each column and its values,
    checks if any column is of datetime type, and returns 'Time series'
    if a datetime column is found.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
    
    Returns:
        str: 'Time series' if a datetime column is found, else 'No time series'.
    """
    if feature_df["prediction"].astype(str).str.strip().str.lower().str.contains("datetime").any():
            
            return "Time series"
    else:
            return "Tabular"

#method to slect which problem type is possibly
def infer_problem_type(y: Series, silent=False) -> str: #Method copied from https://github.com/autogluon/autogluon/blob/03cfcbd3832a39a19a8e25fab67bb5eba904df9c/core/src/autogluon/core/utils/utils.py
    """Identifies which type of prediction problem we are interested in (if user has not specified).
    Ie. binary classification, multi-class classification, or regression.
    """
    # treat None, NaN, INF, NINF as NA
    y = y.replace([np.inf, -np.inf], np.nan, inplace=False)
    y = y.dropna()
    num_rows = len(y)

    if num_rows == 0:
        raise ValueError("Label column cannot have 0 valid values")

    unique_values = y.unique()

    if num_rows > LARGE_DATA_THRESHOLD:
        regression_threshold = (
            REGRESS_THRESHOLD_LARGE_DATA  # if the unique-ratio is less than this, we assume multiclass classification, even when labels are integers
        )
    else:
        regression_threshold = REGRESS_THRESHOLD_SMALL_DATA

    unique_count = len(unique_values)
    if unique_count == 2:
        problem_type = BINARY
        reason = "only two unique label-values observed"
    elif y.dtype.name in ["object", "category", "string"]:
        problem_type = MULTICLASS
        reason = f"dtype of label-column == {y.dtype.name}"
    elif np.issubdtype(y.dtype, np.floating):
        unique_ratio = unique_count / float(num_rows)
        if (unique_ratio <= regression_threshold) and (unique_count <= MULTICLASS_UPPER_LIMIT):
            try:
                can_convert_to_int = np.array_equal(y, y.astype(int))
                if can_convert_to_int:
                    problem_type = MULTICLASS
                    reason = "dtype of label-column == float, but few unique label-values observed and label-values can be converted to int"
                else:
                    problem_type = REGRESSION
                    reason = "dtype of label-column == float and label-values can't be converted to int"
            except:
                problem_type = REGRESSION
                reason = "dtype of label-column == float and label-values can't be converted to int"
        else:
            problem_type = REGRESSION
            reason = "dtype of label-column == float and many unique label-values observed"
    elif np.issubdtype(y.dtype, np.integer):
        unique_ratio = unique_count / float(num_rows)
        if (unique_ratio <= regression_threshold) and (unique_count <= MULTICLASS_UPPER_LIMIT):
            problem_type = MULTICLASS  # TODO: Check if integers are from 0 to n-1 for n unique values, if they have a wide spread, it could still be regression
            reason = "dtype of label-column == int, but few unique label-values observed"
        else:
            problem_type = REGRESSION
            reason = "dtype of label-column == int and many unique label-values observed"
    else:
        raise NotImplementedError(f"label dtype {y.dtype} not supported!")
    if not silent:
        logger.log(25, f"AutoGluon infers your prediction problem is: '{problem_type}' (because {reason}).")

        # TODO: Move this outside of this function so it is visible even if problem type was not inferred.
        if problem_type in [BINARY, MULTICLASS]:
            if unique_count > 10:
                logger.log(20, f"\tFirst 10 (of {unique_count}) unique label values:  {list(unique_values[:10])}")
            else:
                logger.log(20, f"\t{unique_count} unique label values:  {list(unique_values)}")
        elif problem_type == REGRESSION:
            y_max = y.max()
            y_min = y.min()
            y_mean = y.mean()
            y_stddev = y.std()
            logger.log(20, f"\tLabel info (max, min, mean, stddev): ({y_max}, {y_min}, {round(y_mean, 5)}, {round(y_stddev, 5)})")

        logger.log(
            25,
            f"\tIf '{problem_type}' is not the correct problem_type, please manually specify the problem_type parameter during Predictor init "
            f"(You may specify problem_type as one of: {[BINARY, MULTICLASS, REGRESSION]})",
        )
        
        
    return problem_type

def upload_Dataset_csv(
    uploaded_file,
    max_mb: int = 50,
    max_rows: int = 1_000_000,     # hard limit
    max_cols: int = 1_000,         # hard limit
    styler_max_cells: int = 262_144,  # Styler/Streamlit rendering limit
    truncate_rows: int = 1000,   # rows to keep if over Styler limit
):
    """
    Load a CSV uploaded via Streamlit's file_uploader with safety checks.

    Returns
    -------
    (df, error, truncated)
        df : pd.DataFrame or None
        error : str or None (fatal problems only)
        truncated : bool - True if rows were cut down to avoid Styler issues
    """
    if uploaded_file is None:
        return None, "No file uploaded.", False

    # 1) File size check
    # file_size_mb = uploaded_file.size / (1024 * 1024)
    # if file_size_mb > max_mb:
    #     return None, (
    #         f"File is too large ({file_size_mb:.1f} MB). "
    #         f"Limit is {max_mb} MB."
    #     ), False

    # 2) Try separators: comma first, then semicolon
    df = None
    last_error = None
    for sep in [",", ";"]:
        try:
            uploaded_file.seek(0)  # reset pointer each try
            df = pd.read_csv(uploaded_file, sep=sep)
            break
        except Exception as e:
            last_error = e
            df = None

    if df is None:
        return None, (
            "Could not read CSV with ',' or ';' as separator. "
            f"Last error: {last_error}"
        ), False

    # 3) Drop index-like first column if it exists
    if len(df.columns) > 0:
        first_col = df.columns[0]
        col = df[first_col]

        drop_index_col = False

        # Name suggests it's an index
        if str(first_col).lower() in ["unnamed: 0", "index"]:
            drop_index_col = True
        else:
            # Looks like a default RangeIndex 0..n-1
            try:
                if (
                    pd.api.types.is_integer_dtype(col)
                    and col.is_monotonic_increasing
                    and col.iloc[0] == 0
                    and col.iloc[-1] == len(df) - 1
                ):
                    drop_index_col = True
            except Exception:
                pass

        if drop_index_col:
            df = df.drop(columns=[first_col])

    # 4) Hard DataFrame size check
    n_rows, n_cols = df.shape
    if n_rows > max_rows or n_cols > max_cols:
        return None, (
            f"DataFrame too large: {n_rows} rows, {n_cols} columns. "
            f"Limits: {max_rows} rows, {max_cols} columns."
        ), False

    # 5) Styler cells limit: trim rows to at most `truncate_rows` if needed
    original_rows = n_rows
    truncated = False
    total_cells = n_rows * n_cols

    if total_cells > styler_max_cells:
        truncated = True
        df = df.head(min(truncate_rows, n_rows))
        n_rows = len(df)  # update n_rows if needed

    return df
