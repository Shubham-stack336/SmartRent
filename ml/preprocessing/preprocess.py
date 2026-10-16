import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FEATURES = ["locality", "city", "area", "beds", "bathrooms", "furnishing"]
CAT_COLS = ["locality", "city", "furnishing"]
NUM_COLS = ["area", "beds", "bathrooms"]
TARGET = "rent"


def load_data(path):
    df = pd.read_csv(path)
    return df


def clean_data(df):
    df = df.drop_duplicates()
    df = df[(df["area"] >= 100) & (df["area"] <= 10000)]
    df = df[df["beds"] >= 1]
    return df.reset_index(drop=True)


def split_features_target(df):
    X = df[FEATURES]
    Y = df[TARGET]

    return X, Y


def build_preprocessor(scale_numeric=False):
    num = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
        ("num", num, NUM_COLS),
    ])
