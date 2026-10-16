import json
import numpy as np
import joblib
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline

from ml.preprocessing.preprocess import (
    build_preprocessor, clean_data, load_data, split_features_target,
)

DATA_PATH = "ml/data/raw/data.csv"
MODEL_PATH = "ml/models/rent_model.pkl"
META_PATH = "ml/models/metadata.json"


def log_target(pipe):
    # rent is heavily skewed, so fit on log(rent)
    return TransformedTargetRegressor(
        regressor=pipe, func=np.log1p, inverse_func=np.expm1
    )


def build_models():
    rf = Pipeline([
        ("prep", build_preprocessor()),
        ("rf", RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)),
    ])
    # 4-layer neural network: 4 hidden layers + output
    mlp = Pipeline([
        ("prep", build_preprocessor(scale_numeric=True)),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=(128, 64, 32, 16), early_stopping=True,
            max_iter=500, random_state=42,
        )),
    ])
    return {"random_forest": log_target(rf), "neural_network_4layer": log_target(mlp)}


def main():
    df = clean_data(load_data(DATA_PATH))
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    results, fitted = {}, {}
    for name, model in build_models().items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        results[name] = {
            "mae": mean_absolute_error(y_test, pred),
            "r2": r2_score(y_test, pred),
        }
        fitted[name] = model
        print(f"{name:24s} MAE: {results[name]['mae']:.0f}  R2: {results[name]['r2']:.3f}")

    best = min(results, key=lambda n: results[n]["mae"])
    print("best model:", best)
    joblib.dump(fitted[best], MODEL_PATH)

    locality_by_city = (
        df.groupby("city")["locality"].agg(lambda s: sorted(s.unique())).to_dict()
    )
    meta = {
        "model": best,
        "mae": results[best]["mae"],
        "r2": results[best]["r2"],
        "comparison": results,
        "cities": sorted(df["city"].unique()),
        "furnishing": sorted(df["furnishing"].unique()),
        "localities": locality_by_city,
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("saved", MODEL_PATH)


if __name__ == "__main__":
    main()
