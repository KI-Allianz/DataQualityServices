import pandas as pd
import json
import time
import logging
from typing import Dict, Any
from .huggingface_model import ask_model_prompt, ask_model_mes
from .feature_generator import get_feature_results, FeaturizeFile
from tqdm import tqdm
import re
import warnings
import pickle
from pathlib import Path
import joblib

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

MAX_COLUMNS_PER_LLM_CALL = 20
MAX_SAMPLES_PER_COLUMN = 6

# Each tuple = (canonical label to return, regex that should match it, case-insensitive)
LABEL_PATTERNS = [
    ("Numeric",            r'(?<!\w)Nume\w*(?!\w)'),                 # "Nume", "Numeric", "Numerical", etc.
    ("Categorical",        r'(?<!\w)Catego\w*(?!\w)'),               # "Catego", "Categorical", etc.
    ("Datetime",           r'(?<!\w)Date\w*(?!\w)'),                 # "Date", "Datetime", "DateTime", etc.
    ("Sentence",           r'(?<!\w)Sent\w*(?!\w)'),                 # "Sent", "Sentence", "Sentences", etc.
    ("URL",                r'(?<!\w)URL\w*(?!\w)'),                  # "URL", "URLs"
    ("List",               r'(?<!\w)List\w*(?!\w)'),                 # "List", "ListType", etc.
    ("Embedded-Number",    r'(?<!\w)Embedded(?:-Number)?\w*(?!\w)'), # "Embedded", "Embedded-Number", etc.
    ("Not-Generalizable",  r'(?<!\w)Not-?Genera\w*(?!\w)'),          # "Not-Genera...", "Not-Generalizable"
    ("Context-Specific",   r'(?<!\w)Context-?S\w*(?!\w)'),           # "Context-Spec...", "ContextSpecific"
]


def _parse_llm_json_object(response: str) -> Dict[str, Any]:
    """
    Parse the first JSON object from an LLM response.

    Qwen3 can prepend hidden-reasoning style text such as "<think>...</think>"
    even when prompted for JSON. This keeps the upload path tolerant while still
    requiring a valid JSON object somewhere in the answer.
    """
    if not response:
        raise ValueError("LLM returned an empty response")

    cleaned = re.sub(
        r"<think>.*?</think>",
        "",
        response,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for match in re.finditer(r"\{", cleaned):
            try:
                parsed, _ = decoder.raw_decode(cleaned[match.start():])
                break
            except json.JSONDecodeError:
                continue
        else:
            raise ValueError(
                f"LLM returned invalid JSON: {response}"
            )

    if not isinstance(parsed, dict):
        raise ValueError(
            f"LLM JSON response must be an object, got {type(parsed).__name__}: {response}"
        )

    return parsed


# Function to extract the first matching label from text
def extract_response(text: str):
    """
    Scan `text` and return the first label detected,
    allowing partial prefixes (like 'Catego' for 'Categorical').
    Returns None if nothing matches.
    """
    earliest_hit = None  # (start_index, label)

    for label, pat in LABEL_PATTERNS:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            if earliest_hit is None or m.start() < earliest_hit[0]:
                earliest_hit = (m.start(), label)

    return earliest_hit[1] if earliest_hit else None



## Prompts Personal Data Detection

# System/user prompt templates used to steer the LLM classification behavior
initial_prompt_kaggle = """
As a classifier of personal related data in tabular datasets, your task is to analyze the provided columns (each containing up to ten distinct values)
and determine whether they contain information that originates from or relates to a person, even if it is not directly identifiable. 
Detecting personal related information helps ensure compliance with data protection regulations and safeguards individuals' privacy and security.
Output your results in a dictionary format with a boolean indicating if the column contains personal related data or not.
"""

# Few-shot example prompt that demonstrates the expected task framing
example_prompt_kaggle = """
You can use the following example as a guideline:
Classify the following column with careful consideration of the dataset description:
Dataset: Title: "Test Dataset"
Description: "This dataset was used for a linear regression."
Features: ['first_name_en_10', 'last_name_en_10', 'email_en_10', 'phone_number', 'address_en_10', 'city_en_10', 'country_en_10', 'date', 'target']
Column of the dataset to classify: 'first_name_en_10': ['Tom', 'Walter', 'Mia', 'Lena', 'John', 'Jack', 'Felice', 'Anna', 'Lukas', 'Will']
Does this column, in the context of the dataset, contain information relating to a natural person?
"""

# Example answer string shown to the model to set the expected output style
example_answer_kaggle = "Example Answer: {first_name_en_10: true}"

# Short question appended to each per-column prompt
classification_prompt_kaggle = "Does this column, in the context of the dataset, contain information relating to a natural person?"


def extract_context_from_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Build a small dataset context summary from the DataFrame.

    Returns a dict with:
      - title: static title for the uploaded dataset
      - description: basic shape summary (#rows, #cols)
      - features: list of column names
      - sample_rows: first 3 rows as a list of row dicts
    """
    features = list(df.columns)
    sample_rows = df.head(3).to_dict(orient='records')
    context = {
        "title": "Uploaded Dataset",
        "description": f"Dataset with {len(df)} rows and {len(df.columns)} columns.",
        "features": features,
        "sample_rows": sample_rows
    }
    return context


def execute_personal_llm(df, model):
    """
    Classify each column of a DataFrame as 'Personal' vs 'Non-personal' using an LLM.

    Process (per column):
      1) Take up to 10 distinct non-null values from the column.
      2) Build a prompt including dataset context (title, description, features).
      3) Send a few-shot conversation (system inst, example prompt+answer, real prompt) to the model.
      4) Parse model response into a dict or fallback to a string check for 'true'.
      5) Record 'Prediction' for the column and accumulate results.

    Returns:
      - metadata_df: one-row DataFrame with high-level summary counts
      - results_placeholder: DataFrame with per-column predictions (Column, Prediction)
    """
    # Build dataset-level context (title/description/features/samples)
    context = extract_context_from_data(df)

    # Accumulators for per-column outcomes and counts
    results = []
    personal_count = 0
    nonpersonal_count = 0

    # Start total timer for the whole classification run
    total_start_time = time.time()

    # Iterate all columns for classification
    for i, col in enumerate(df.columns):
        # Prepare up to 10 unique non-null sample values from the column for the prompt
        val_list = df[col].dropna().unique().tolist()[:10]

        # Craft the user prompt for this specific column using the dataset context
        data_prompt = (
            f"Classify the following column with careful consideration of the dataset description. "
            f"Dataset: Title: {context.get('title', '')}\n"
            f"Description: {context.get('description', '')}\n"
            f"Features: {context.get('features', list(df.columns))}\n"
            f"Column of the dataset to classify: '{col}': {val_list}\n"
            f"{classification_prompt_kaggle}"
        )

        # Build a chat-style conversation: system instruction, example QA, and the actual column question
        conversation = [
            {"role": "system", "content": initial_prompt_kaggle},
            {"role": "user", "content": example_prompt_kaggle},
            {"role": "assistant", "content": example_answer_kaggle},
            {"role": "user", "content": data_prompt}
        ]

        # Time the single-column LLM call
        start_time = time.time()

        # Call the model with the constructed messages; returns either str or dict depending on wrapper
        response = ask_model_mes(messages=conversation, pipe=model)  # , max_new_tokens=512, 1024

        elapsed = time.time() - start_time  # per-column latency (currently not stored)

        # Try to normalize the model response to a dict:
        #  - If it's a string, strip the "Example Answer:" prefix and JSON-decode
        #  - If it's already a dict, use it directly
        #  - Otherwise, fall back to empty dict and then string 'true' heuristic
        try:
            if isinstance(response, str):
                response_dict = json.loads(response.replace("Example Answer:", "").strip())
            elif isinstance(response, dict):
                response_dict = response
            else:
                response_dict = {}
            # Prefer an explicit key lookup matching the column name
            is_personal = response_dict.get(col, None)
            # Heuristic fallback: treat any response containing 'true' as positive
            if is_personal is None:
                is_personal = "true" in str(response).lower()
        except Exception:
            # If JSON parsing fails, fall back to simple 'true' substring check
            is_personal = "true" in str(response).lower()

        # Append per-column prediction to the results list
        results.append({
            "Column": col,
            "Prediction": "Personal" if is_personal else "Non-personal",
            # "Time (s)": f"{elapsed:.2f}"  # timing available if desired
        })

        # Update running counters
        if is_personal:
            personal_count += 1
        else:
            nonpersonal_count += 1

        # Materialize results to a DataFrame (rebuilt each loop; could be moved outside for efficiency)
        results_placeholder = pd.DataFrame(results)

    # End total timer for the full pass
    total_elapsed = time.time() - total_start_time  # currently unused

    # --- Build high-level summary metadata (counts) from results ---
    num_features = len(results_placeholder)

    # Count how many columns were predicted as personal / non-personal
    # NOTE: comparisons below are case-sensitive; they must match the strings used above in 'Prediction'
    num_personal = results_placeholder[results_placeholder["Prediction"] == "Personal"].shape[0]
    num_non_personal = num_features - num_personal

    metadata_personal = {
        "Total Features": num_features,
        "Personal Data Exists": num_personal > 0,
        "Personal Features": num_personal,
        "Non-Personal Features": num_non_personal,
        # Store filtered DataFrames of column names by class (as DataFrames, not lists)
        "Personal Columns": results_placeholder[results_placeholder["Prediction"] == "Personal"],
        "Non-Personal Columns": results_placeholder[results_placeholder["Prediction"] == "Non-Personal"]
    }

    # Wrap the dict in a single-row DataFrame for downstream use/plotting
    metadata_df = pd.DataFrame([metadata_personal])

    # Return both the summary metadata and the per-column results table
    return metadata_df, results_placeholder



def execute_dataset_llm(
    df: pd.DataFrame,
    model,
    max_columns_per_call: int = MAX_COLUMNS_PER_LLM_CALL,
    max_samples_per_column: int = MAX_SAMPLES_PER_COLUMN,
):
    """
    Classify feature type and personal-data status for all columns.

    Columns are split into batches to avoid oversized prompts.
    """

    all_feature_rows = []
    all_personal_rows = []

    columns = list(df.columns)

    for start in range(0, len(columns), max_columns_per_call):
        batch_columns = columns[start:start + max_columns_per_call]

        column_summary = {}

        for col in batch_columns:
            series = df[col]
            non_null = series.dropna()

            samples = (
                non_null
                .astype(str)
                .drop_duplicates()
                .head(max_samples_per_column)
                .tolist()
            )

            column_summary[str(col)] = {
                "dtype": str(series.dtype),
                "total_count": int(len(series)),
                "missing_count": int(series.isna().sum()),
                "unique_count": int(non_null.nunique()),
                "samples": samples,
            }

        system_message = """
You classify columns in tabular datasets.

For every column, return:
1. feature_type: exactly one of:
   Numeric, Categorical, Datetime, Sentence, URL, List,
   Embedded-Number, Not-Generalizable, Context-Specific

2. personal: true if the column contains information relating
   to a natural person, otherwise false.

Return ONLY valid JSON.
Do not add markdown or explanations.
Do not include thinking text. Do not include <think> tags.
"""

        user_message = (
            "/no_think\n"
            "Classify every column in this dataset summary:\n"
            + json.dumps(column_summary, ensure_ascii=False)
            + "\n\nReturn JSON in exactly this format:\n"
            + """
{
  "column_name": {
    "feature_type": "Categorical",
    "personal": false
  }
}
"""
        )

        conversation = [
            {
                "role": "system",
                "content": system_message,
            },
            {
                "role": "user",
                "content": user_message,
            },
        ]

        response = ask_model_mes(
            messages=conversation,
            pipe=model,
            max_new_tokens=max(256, len(batch_columns) * 24),
        )

        parsed = _parse_llm_json_object(response)

        for col in batch_columns:
            col_name = str(col)
            prediction = parsed.get(col_name, {})

            feature_type = prediction.get(
                "feature_type",
                "Context-Specific",
            )
            feature_type = str(feature_type).strip().lower()

            is_personal = bool(
                prediction.get("personal", False)
            )

            all_feature_rows.append({
                "Attribute_name": col_name,
                "prediction": feature_type,
            })

            all_personal_rows.append({
                "Column": col_name,
                "Prediction": (
                    "Personal"
                    if is_personal
                    else "Non-personal"
                ),
            })

    feature_type_df = pd.DataFrame(all_feature_rows)
    personal_results_df = pd.DataFrame(all_personal_rows)

    num_personal = int(
        (
            personal_results_df["Prediction"]
            == "Personal"
        ).sum()
    )

    personal_summary_df = pd.DataFrame([{
        "Total Features": int(len(personal_results_df)),
        "Personal Data Exists": num_personal > 0,
        "Personal Features": num_personal,
        "Non-Personal Features": (
            int(len(personal_results_df))
            - num_personal
        ),
        "Personal Columns": personal_results_df[
            personal_results_df["Prediction"]
            == "Personal"
        ],
        "Non-Personal Columns": personal_results_df[
            personal_results_df["Prediction"]
            == "Non-personal"
        ],
    }])

    return (
        feature_type_df,
        personal_summary_df,
        personal_results_df,
    )


# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# LLM Feature Type Inference


def build_feature_value_list(row):
    """
    Extract up to five sample values from a single-row DataFrame `row`
    and try to cast each to int, then float, otherwise keep as string.
    Returns a Python list of 5 values with best-effort type casting.
    """
    # Expect a single-row frame with columns 'sample_1'..'sample_5'
    sample_1 = row.loc[0, 'sample_1']
    sample_2 = row.loc[0, 'sample_2']
    sample_3 = row.loc[0, 'sample_3']
    sample_4 = row.loc[0, 'sample_4']
    sample_5 = row.loc[0, 'sample_5']

    # Collect raw samples
    feature_values = [sample_1, sample_2, sample_3, sample_4, sample_5]
    return feature_values


def LLM_generate_base_prompt(row):
    """
    Build a base prompt with feature metadata for the LLM.
    The single-row DataFrame `row` should contain:
      - 'Attribute_name': the feature/column name
      - 'num_of_dist_val': number of distinct values
      - 'total_vals': total number of values
      - 'sample_1'..'sample_5': example values (used via build_feature_value_list)
    """
    feature_name = row.loc[0, 'Attribute_name']
    nr_distinct_values = row.loc[0, 'num_of_dist_val']
    nr_total_values = row.loc[0, 'total_vals']

    # Extract 5 example values and include them in the prompt
    feature_values = build_feature_value_list(row)

    # Compose a natural-language prompt describing the feature and its stats
    base_prompt = (
        f"I have a dataset which contains a feature with the name '{feature_name}'."
        f" The dataset contains {nr_total_values} values in total."
        f" The feature has {nr_distinct_values} unique values in total."
        f" This is a list with some of the distinct values from the feature: {feature_values}.\n"
    )
    return base_prompt


system_prompt = (
    "You are a Data Scientist tasked with accurately classifying features into 9 predefined classes: "
    "Numeric, Categorical, Datetime, Sentence, URL, List, Embedded-Number, Not-Generalizable, and Context-Specific. "
    "For each feature, your objective is to:\n\n"
    "1) Analyze the feature name, the sample values, and the summary statistics provided.\n"
    "2) Assign the feature to the most specific and accurate class based on predefined criteria.\n\n"
    "Ensure that you:\n"
    "- Use precise and confident language for classification.\n"
    "- Assign a class only when there is sufficient evidence to support the decision.\n"
    "- Avoid ambiguity and overly broad classifications.\n\n"
    "Below are detailed descriptions of each class to guide your decision-making:\n\n"
    "1. Numeric:\n"
    "This class includes features containing raw numerical values, such as integers, floating-point numbers, or negative numbers "
    "(e.g., '-5', '12.34'). These features should represent measurable quantities or counts and are usually continuous or discrete. "
    "Numeric features are suitable for mathematical operations like addition, subtraction, multiplication, and division, and are used "
    "to model and predict numeric outcomes. Common examples include 'age', 'price', 'distance', or 'sales volume'.\n"
    "Important Notes for Classifying Numeric Features:\n"
    " - Numeric features should be continuous or discrete quantities with a meaningful numeric relationship, e.g., 'temperature', "
    "'income', or 'height'.\n"
    " - Features with a small number of distinct numeric values, such as binary features ('0' and '1'), should be classified as numeric "
    "only if they represent actual quantities, not categories.\n"
    " - Features used for classification purposes (e.g., '0' for 'No' and '1' for 'Yes') should be classified as Categorical, even "
    "if their values are numeric.\n"
    " - Avoid classifying identifiers or domain-specific codes (e.g., 'ProductID', 'ZipCode') as Numeric, even if they contain "
    "numeric values.\n"
    "Additional Guidance:\n"
    " - Ensure the numeric feature represents a measurable quantity and can undergo statistical or machine learning analysis as "
    "a continuous or discrete quantity.\n\n"
    "2. Categorical:\n"
    "Categorical features contain distinct labels or group identifiers, such as product names, tags, or classifications. Examples "
    "include 'red', 'apple', 'high', 'low', or 'yes'. Categorical features may include numeric data representing encoded categories "
    "(e.g., one-hot or label encoding).\n"
    "Important Notes for Classifying Categorical Features:\n"
    " - Categorical features represent groups or labels rather than quantities or numbers.\n"
    " - Numeric data that encodes categories (e.g., '1' for 'Yes', '0' for 'No') should be classified as Categorical, not Numeric.\n"
    " - Avoid classifying features like dates, URLs, or sentences as Categorical.\n"
    " - Features that are lists of categories (e.g., ['apple', 'banana', 'orange']) should not be classified here.\n" 
    "Additional Guidance:\n"
    " - Ensure the categorical feature consists of distinct groups or categories and does not contain continuous or ordinal "
    "numeric values.\n\n"
    "3. Datetime:\n"
    "Datetime features represent date or time information, such as 'March 2020', '15-Mar-2020', or '2025-01-01'. These features "
    "can include partial date formats (e.g., year and month) but should not be used to represent a single point in time like '2020' "
    "without additional context.\n"
    "Important Notes for Classifying Datetime Features:\n"
    " - The feature should represent time in some format (e.g., date, timestamp, year, month, day).\n"
    " - Partial dates (e.g., year and month) can be classified as datetime but exclude year-only formats.\n"
    "Additional Guidance:\n"
    " - Ensure the datetime feature contains a temporal component and is not just an identifier or code.\n\n"
    "4. Sentence:\n"
    "Sentence features represent natural language text conveying meaning, context, or information in the form of complete sentences "
    "or phrases. Short entries (1-3 words) may also be classified as sentences if they express meaningful ideas, e.g., 'Great job!' "
    "or 'Red sedan.'\n"
    "Important Notes for Classifying Sentence Features:\n"
    " - The feature should consist of natural language text that conveys a meaningful idea.\n"
    " - Avoid classifying text as a sentence if it functions purely as a label, identifier, or category (e.g., 'NY' or 'Electronics').\n"
    "Additional Guidance:\n"
    " - Ensure the sentence feature contains a meaningful message or statement.\n\n"
    "5. Embedded-Number:\n"
    "This class includes features with numbers embedded alongside special characters or units, such as 'USD 100' or '5,000 MHz'. "
    "It also includes numbers that use commas as decimal separators. These features may also include other symbols like currency or "
    "percentage signs.\n"
    "Important Notes for Classifying Embedded-Number Features:\n"
    " - The feature should include numeric values along with a unit, symbol, or special character.\n"
    " - Features that are lists of numbers (e.g., '[1, 2, 3]') should not be classified here.\n"
    "Additional Guidance:\n"
    " - Ensure the numeric value and unit/symbol together represent a measurable quantity, such as '10 USD' or '5,000 MHz'.\n\n"
    "6. URL:\n"
    "Features in this class represent web addresses, parts of web addresses, or related components. Examples include full URLs "
    "('https://example.com'), domain names ('example.com'), or URL fragments ('/products/item'). These features typically include "
    "components such as protocols ('http://', 'https://'), domains ('example.com'), paths ('/path/to/resource'), or query strings "
    "('?id=123').\n"
    "Important Notes for Classifying URL Features:\n"
    " - The feature should represent an internet address or network resource.\n"
    " - Avoid classifying features with ambiguous or inconsistent URL formats.\n"
    "Additional Guidance:\n"
    " - Ensure the feature is an actual URL or related network address, not just a part of a URL or a reference.\n\n"
    "7. List:\n"
    "This class includes features whose values represent lists, where each value contains multiple discrete items or elements. "
    "These items may be integers, strings, or any other data type, and they should be in a structured format such as a list of items "
    "separated by commas, or enclosed in square brackets like '[apple, banana, cherry]'. A key distinction is that the values should explicitly "
    "represent multiple items or objects in a list format, not just a string of text with commas.\n"
    "Important Notes for Classifying List Features:\n"
    " - A List feature should contain multiple distinct elements (e.g., ['cat', 'dog', 'bird']).\n"
    " - If the values are comma-separated items within a string, they must represent individual objects or entities in a list structure, "
    "such as a list of keywords, items, or other discrete data points.\n"
    " - Avoid classifying long sentences or descriptive text as a List, even if there are commas present in the value. "
    "Features like 'Reckless driving, Poor choice in music' represent sentences, not lists, because the items are not clearly separated "
    "as discrete list elements.\n"
    " - Lists must clearly indicate multiple distinct elements and follow a recognizable list format.\n\n"
    "8. Not-Generalizable:\n"
    "A feature is classified as 'Not-Generalizable' if it has little to no predictive value. This includes unique identifiers "
    "like 'CustomerID' or features with only one unique value across the entire dataset (e.g., all 'NA').\n"
    "Important Notes for Classifying Not-Generalizable Features:\n"
    " - Features with unique identifiers or nearly constant values should be classified as Not-Generalizable.\n"
    " - Do not classify features with little variance as belonging to other classes like Numeric or Categorical.\n"
    "Additional Guidance:\n"
    " - Ensure that the feature provides sufficient variation to be useful in modeling or analysis.\n\n"
    "9. Context-Specific:\n"
    "Features in this class require additional domain knowledge to classify, such as industry-specific data, ambiguous formats, "
    "or specialized data types.\n"
    "Important Notes for Classifying Context-Specific Features:\n"
    " - The feature's role or meaning cannot be determined without understanding the specific context or domain.\n"
    " - Features with ambiguous names or formats may belong in this class.\n"
    "Additional Guidance:\n"
    " - Use Context-Specific when no other predefined class confidently fits the feature.\n"
    " - Avoid classifying features with standard formats (numeric, categorical, datetime) as Context-Specific.\n"
    "Decision hierarchy:\n"
    "1. Prioritize matches to numeric, categorical, datetime, or other clear types when evidence aligns strongly.\n"
    "2. Use Context-Specific as a fallback for ambiguous, specialized, or complex features requiring domain knowledge.\n\n"
    "Your goal is to ensure accurate, reliable, and meaningful classifications for all features."
)

def detect_all_at_once(row, pipe=None):
    """
    Ask the LLM to directly pick a single class (lowercase string) for the feature,
    given the base prompt + taxonomy system prompt. If `pipe` is None, use gpt-4o via gpt.ask_gpt,
    else use a provided pipeline (`ask_model_prompt` + `extract_response`).
    
    Returns a lowercase class name or 'unknown' if inference fails.
    """
    # Compose a single-shot classification prompt that asks for ONLY the class name (lowercase)
    prompt = (
        LLM_generate_base_prompt(row)
        + "Based on the provided information,"
        + " classify the feature as either Numeric, Categorical, Datetime, Sentence, URL, List, Embedded-Number,"
        + " Not-Generalizable, or Context-Specific."
        + " Respond with ONLY the class name in lower case."
    )

    # Dispatch to either default GPT backend or custom pipeline
    response = None
    try:
        if not pipe:
            response = gpt.ask_gpt(prompt, system_prompt, model="gpt-4o")
        else:
            response = ask_model_prompt(prompt, system_prompt, pipe)
            response = extract_response(response)
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"LLM inference failed: {e}")
        response = None

    # Handle None response gracefully
    if response is None:
        return "unknown"
    
    # Normalize to lowercase string for the caller
    return str(response).lower()


def is_numeric(row, pipe=None):
    """
    Y/N helper: asks the LLM if the feature is Numeric (yes/no).
    Returns True/False or a failure marker string if response is ambiguous.
    """
    prompt = (
        LLM_generate_base_prompt(row)
        + "Based on the provided information,"
        + " can you classify the feature as Numeric?"
        + " Respond with ONLY 'yes' or 'no'."
    )
    if not pipe:
        response = gpt.ask_gpt(prompt, system_prompt, model="gpt-4o")
    else:
        response = ask_model_prompt(prompt, system_prompt, pipe)
        response = extract_response(response)

    # Normalize to boolean; handle ambiguous outputs
    if "yes" in response.lower():
        return True
    elif "no" in response.lower():
        return False
    else:
        return 'Failed to make LLM-prediction'


def is_categorical(row, pipe=None):
    """Y/N helper for Categorical classification."""
    prompt = (
        LLM_generate_base_prompt(row)
        + "Based on the provided information,"
        + " can you classify the feature as Categorical?"
        + " Respond with ONLY 'yes' or 'no'."
    )
    if not pipe:
        response = gpt.ask_gpt(prompt, system_prompt, model="gpt-4o")
    else:
        response = ask_model_prompt(prompt, system_prompt, pipe)
        response = extract_response(response)

    if "yes" in response.lower():
        return True
    elif "no" in response.lower():
        return False
    else:
        return 'Failed to make LLM-prediction'


def is_datetime(row, pipe=None):
    """Y/N helper for Datetime classification."""
    prompt = (
        LLM_generate_base_prompt(row)
        + "Based on the provided information,"
        + " can you classify the feature as Datetime?"
        + " Respond with ONLY 'yes' or 'no'."
    )
    if not pipe:
        response = gpt.ask_gpt(prompt, system_prompt, model="gpt-4o")
    else:
        response = ask_model_prompt(prompt, system_prompt, pipe)
        response = extract_response(response)

    if "yes" in response.lower():
        return True
    elif "no" in response.lower():
        return False
    else:
        return 'Failed to make LLM-prediction'


def is_sentence(row, pipe=None):
    """Y/N helper for Sentence classification."""
    prompt = (
        LLM_generate_base_prompt(row)
        + "Based on the provided information,"
        + " can you classify the feature as Sentence?"
        + " Respond with ONLY 'yes' or 'no'."
    )
    if not pipe:
        response = gpt.ask_gpt(prompt, system_prompt, model="gpt-4o")
    else:
        response = ask_model_prompt(prompt, system_prompt, pipe)
        response = extract_response(response)

    if "yes" in response.lower():
        return True
    elif "no" in response.lower():
        return False
    else:
        return 'Failed to make LLM-prediction'


def is_embedded_number(row, pipe=None):
    """Y/N helper for Embedded-Number classification."""
    prompt = (
        LLM_generate_base_prompt(row)
        + "Based on the provided information,"
        + " can you classify the feature as Embedded-Number?"
        + " Respond with ONLY 'yes' or 'no'."
    )
    if not pipe:
        response = gpt.ask_gpt(prompt, system_prompt, model="gpt-4o")
    else:
        response = ask_model_prompt(prompt, system_prompt, pipe)
        response = extract_response(response)

    if "yes" in response.lower():
        return True
    elif "no" in response.lower():
        return False
    else:
        return 'Failed to make LLM-prediction'


def is_url(row, pipe=None):
    """Y/N helper for URL classification."""
    prompt = (
        LLM_generate_base_prompt(row)
        + "Based on the provided information,"
        + " can you classify the feature as URL?"
        + " Respond with ONLY 'yes' or 'no'."
    )
    if not pipe:
        response = gpt.ask_gpt(prompt, system_prompt, model="gpt-4o")
    else:
        response = ask_model_prompt(prompt, system_prompt, pipe)
        response = extract_response(response)

    if "yes" in response.lower():
        return True
    elif "no" in response.lower():
        return False
    else:
        return 'Failed to make LLM-prediction'


def is_list(row, pipe=None):
    """Y/N helper for List classification."""
    prompt = (
        LLM_generate_base_prompt(row)
        + "Based on the provided information,"
        + " can you classify the feature as List?"
        + " Respond with ONLY 'yes' or 'no'."
    )
    if not pipe:
        response = gpt.ask_gpt(prompt, system_prompt, model="gpt-4o")
    else:
        response = ask_model_prompt(prompt, system_prompt, pipe)
        response = extract_response(response)

    if "yes" in response.lower():
        return True
    elif "no" in response.lower():
        return False
    else:
        return 'Failed to make LLM-prediction'


def is_not_generalizable(row, pipe=None):
    """Y/N helper for Not-Generalizable classification."""
    prompt = (
        LLM_generate_base_prompt(row)
        + "Based on the provided information,"
        + " can you classify the feature as Not-Generalizable?"
        + " Respond with ONLY 'yes' or 'no'."
    )
    if not pipe:
        response = gpt.ask_gpt(prompt, system_prompt, model="gpt-4o")
    else:
        response = ask_model_prompt(prompt, system_prompt, pipe)
        response = extract_response(response)

    if "yes" in response.lower():
        return True
    elif "no" in response.lower():
        return False
    else:
        return 'Failed to make LLM-prediction'


def is_context_specific(row, pipe=None):
    """Y/N helper for Context-Specific classification."""
    prompt = (
        LLM_generate_base_prompt(row)
        + "Based on the provided information,"
        + " can you classify the feature as Context-Specific?"
        + " Respond with ONLY 'yes' or 'no'."
    )
    if not pipe:
        response = gpt.ask_gpt(prompt, system_prompt, model="gpt-4o")
    else:
        response = ask_model_prompt(prompt, system_prompt, pipe)
        response = extract_response(response)

    if "yes" in response.lower():
        return True
    elif "no" in response.lower():
        return False
    else:
        return 'Failed to make LLM-prediction'


def generate_fewshot_examples(class_prompt, dataset, indexes, binary=False):
    """
    Build a few-shot prompt block with multiple Q/A examples.
    For each index in `indexes`, it creates:
      - Question: base prompt + class_prompt for the feature at that row
      - Answer: the label from dataset.y_act (or alternating yes/no if binary=True)
    Returns a single string ready to be prepended to user prompts.
    """
    prompt = "When answering user questions follow these examples:\n\n"
    for n, i in enumerate(indexes):
        prompt += "Question: "
        # Provide the per-feature base prompt built from the row (slice by i:i+1 → single-row DataFrame)
        prompt += LLM_generate_base_prompt(dataset.loc[i:i+1].reset_index(drop=True))
        prompt += class_prompt

        # Use gold label from dataset (title-cased), or alternate yes/no if binary few-shot
        label = dataset.loc[i:i+1].y_act.values[0].title()
        if binary:
            label = 'yes' if (n % 2 == 0) else 'no'

        prompt += f"\nAnswer: {label}"
        if not n == len(indexes) - 1:
            prompt += "\n\n"
    return prompt


def Load_RF(
    df,
    rf_Filename: str | Path | None = None,
):
    """
    Load the pickled RandomForest model and return predictions as a list.

    The default path is resolved relative to this Python file, so it works
    independently of the current working directory and operating system.
    """
    if rf_Filename is None:
        backend_root = Path(__file__).resolve().parents[2]
        rf_path = backend_root / "data" / "models" / "RandomForest.pkl"
    else:
        rf_path = Path(rf_Filename).expanduser().resolve()

    if not rf_path.is_file():
        raise FileNotFoundError(
            f"RandomForest model not found at: {rf_path}"
        )

    with rf_path.open("rb") as file:
        pickled_rf_model = pickle.load(file)

    return pickled_rf_model.predict(df).tolist()


def FeatureExtraction(data, useSamples=0):
    """‚
    Construct a feature matrix for classical ML from `data` by:
      - Selecting numeric/statistical feature columns
      - Vectorizing 'Attribute_name' using a prefit dictionary
      - Optionally vectorizing sample_1 and sample_2 and concatenating
    Returns a DataFrame ready for a downstream classifier.
    """
    # Load prefit vectorizers for attribute names and sample values
    vectorizerName = joblib.load("src/services_featuretype_personal/dictionaryName.pkl")
    vectorizerSample = joblib.load("src/services_featuretype_personal/dictionarySample.pkl")

    # Core numeric/stat features expected in `data`
    data1 = data[['total_vals', 'num_nans', '%_nans', 'num_of_dist_val', '%_dist_val', 'mean', 'std_dev', 'min_val', 'max_val',
                  'has_delimiters', 'has_url', 'has_email', 'has_date', 'mean_word_count',
                  'std_dev_word_count', 'mean_stopword_total', 'stdev_stopword_total',
                  'mean_char_count', 'stdev_char_count', 'mean_whitespace_count',
                  'stdev_whitespace_count', 'mean_delim_count', 'stdev_delim_count',
                  'is_list', 'is_long_sentence']]
    data1 = data1.reset_index(drop=True).fillna(0)  # stabilize numeric input

    # Vectorize attribute names into a sparse bag-of-words then to dense DataFrame
    arr = [str(x) for x in data['Attribute_name'].values]
    X = vectorizerName.transform(arr)
    attr_df = pd.DataFrame(X.toarray())

    # Optionally include vectorized samples (sample_1 and sample_2)
    if useSamples:
        arr1 = [str(x) for x in data['sample_1'].values]
        arr2 = [str(x) for x in data['sample_2'].values]
        X1 = vectorizerSample.transform(arr1)
        X2 = vectorizerSample.transform(arr2)
        sample1_df = pd.DataFrame(X1.toarray())
        sample2_df = pd.DataFrame(X2.toarray())
        data2 = pd.concat([data1, attr_df, sample1_df, sample2_df], axis=1, sort=False)
    else:
        data2 = pd.concat([data1, attr_df], axis=1, sort=False)

    return data2


# Registry mapping a numeric key to a feature-type detector function.
# Currently, only zero-shot all-in-one detection is enabled (key 0).
features = {
    # Zero-Shot
    0: detect_all_at_once,

    # Other detectors can be enabled by uncommenting and wiring here:
    # 1: is_numeric,
    # 2: is_categorical,
    # 3: is_datetime,
    # 4: is_sentence,
    # 5: is_url,
    # 6: is_embedded_number,
    # 7: is_list,
    # 8: is_not_generalizable,
    # 9: is_context_specific,

    # Few-shot variants (commented placeholders):
    # 10: detect_all_at_once_fewshot,
    # 11: is_numeric_fewshot,
    # ...
}


def execute_featuretype_LLM(df, model):
    """
    Run the selected feature-type detector on all rows of `df` using `model`.
    Delegates to `get_feature_results` (not shown here) with `features[0]`
    which corresponds to zero-shot classification.
    """
    featurized_data = get_feature_results(
        df=df,
        feature=features[0],  # choose the detector callable (zero-shot)
        pipe=model,          # model/pipeline passed through to detector
    )
    return featurized_data
