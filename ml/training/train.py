import json
import numpy as np
import joblib
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from ml.preprocessing.preprocess import (
    build_preprocessor, clean_data, load_data, split_features_target,
)

DATA_PATH = "ml/data/raw/data.csv"
MODEL_PATH = "ml/models/rent_model.pkl"
META_PATH = "ml/models/metadata.json"


def main():
    df = clean_data(load_data(DATA_PATH))
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipe = Pipeline([
        ("prep", build_preprocessor()),
        ("rf", RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)),
    ])
    # rent is heavily skewed, so fit on log(rent)
    model = TransformedTargetRegressor(
        regressor=pipe, func=np.log1p, inverse_func=np.expm1
    )
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    print(f"MAE: {mae:.0f}  R2: {r2:.3f}")

    joblib.dump(model, MODEL_PATH)

    locality_by_city = (
        df.groupby("city")["locality"].agg(lambda s: sorted(s.unique())).to_dict()
    )
    meta = {
        "mae": mae,
        "r2": r2,
        "cities": sorted(df["city"].unique()),
        "furnishing": sorted(df["furnishing"].unique()),
        "localities": locality_by_city,
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("saved", MODEL_PATH)


if __name__ == "__main__":
    main()
