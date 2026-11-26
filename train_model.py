"""Train transfer-prediction models.

  1. Classifier: will this player transfer in the next window?  (GradientBoosting)
  2. Regressor:  expected transfer fee if he does move?          (GradientBoosting)

Run:  python train_model.py
Outputs: model/transfer_classifier.joblib, model/fee_regressor.joblib + metrics report
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score, brier_score_loss, mean_absolute_error, r2_score

FEATURES = [
    "age", "rating", "minutes", "goals", "assists", "pass_accuracy",
    "dribbles_per_game", "injuries_last_2y", "days_injured_last_2y",
    "recent_surgery", "contract_years_left", "club_tier", "position", "league",
]

def load():
    df = pd.read_csv("data/players.csv")
    X = pd.get_dummies(df[FEATURES], columns=["position", "league"])
    return df, X

def main():
    df, X = load()
    Path("model").mkdir(exist_ok=True)

    # ---- classifier: transfer probability ----
    clf = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42)
    p = cross_val_predict(clf, X, df["transferred"], cv=5, method="predict_proba")[:, 1]
    print("=== Will he transfer? (5-fold CV) ===")
    print(f"ROC-AUC : {roc_auc_score(df['transferred'], p):.3f}")
    print(f"Brier   : {brier_score_loss(df['transferred'], p):.3f}")
    top = df.assign(p=p).nlargest(5, "p")[["player_id", "position", "age", "rating", "contract_years_left", "p"]]
    print("Top transfer risks:\n", top.to_string(index=False))
    clf_full = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42)
    clf_full.fit(X, df["transferred"])
    joblib.dump({"model": clf_full, "columns": X.columns.tolist()}, "model/transfer_classifier.joblib")

    # ---- regressor: fee ----
    moved = df["transferred"] == 1
    X_m, y_m = X[moved], df.loc[moved, "fee_millions"]
    reg = GradientBoostingRegressor(n_estimators=400, learning_rate=0.05, max_depth=3, random_state=42)
    pred = cross_val_predict(reg, X_m, y_m, cv=5)
    print("\n=== How much for? (5-fold CV, transferred players only) ===")
    print(f"MAE     : EUR {mean_absolute_error(y_m, pred):.1f}M")
    print(f"R2      : {r2_score(y_m, pred):.3f}")
    reg_full = GradientBoostingRegressor(n_estimators=400, learning_rate=0.05, max_depth=3, random_state=42)
    reg_full.fit(X_m, y_m)
    joblib.dump({"model": reg_full, "columns": X.columns.tolist()}, "model/fee_regressor.joblib")

    # ---- feature importance ----
    imp = pd.Series(clf_full.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)
    print("\nTop features (transfer classifier):")
    print(imp.round(3).to_string())
    print("\nSaved models to model/")

if __name__ == "__main__":
    main()
