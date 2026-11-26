"""Train transfer-prediction models on REAL Transfermarkt player-seasons (56k rows).

  1. Classifier: paid transfer in the window after the season?  (HistGradientBoosting)
  2. Regressor:  actual transfer fee in EUR millions (trained on log-fee)

Run:  python train_real_model.py
Outputs: model/real_transfer_classifier.joblib, model/real_fee_regressor.joblib
"""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance

FEATURES = [
    "age", "apps", "minutes", "goals", "assists", "goals_per90", "assists_per90",
    "cards", "reds", "longest_gap_days", "gaps_over_21d", "availability",
    "log_mv", "mv_trend_12m", "height_cm", "position", "foot", "league", "season",
]
CATS = ["position", "foot", "league"]

def prep(df):
    X = df[[f for f in FEATURES if f not in ("log_mv", "season", "height_cm")]].copy()
    X["log_mv"] = np.log1p(df["market_value_eur"])
    X["season"] = df["season"].astype(float)
    X["height_cm"] = df["height_cm"].astype(float)
    X = X[FEATURES]
    for c in CATS:
        X[c] = X[c].astype("category")
    return X

def main():
    df = pd.read_csv("data/real_players.csv")
    X = prep(df)
    Path("model").mkdir(exist_ok=True)

    # ---- classifier ----
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, class_weight={0: 1, 1: 2}, random_state=42)
    p = cross_val_predict(clf, X, df["transferred"], cv=4, method="predict_proba")[:, 1]
    print("=== Will he transfer? (real data, 4-fold CV) ===")
    print(f"ROC-AUC: {roc_auc_score(df['transferred'], p):.3f} | PR-AUC: {average_precision_score(df['transferred'], p):.3f} | Brier: {brier_score_loss(df['transferred'], p):.3f}")
    top = df.assign(p=p).nlargest(5, "p")[["name", "season", "age", "market_value_eur", "p"]]
    print("Highest transfer risk:\n", top.to_string(index=False))
    clf_full = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, class_weight={0: 1, 1: 2}, random_state=42)
    clf_full.fit(X, df["transferred"])
    joblib.dump({"model": clf_full, "features": FEATURES, "cats": CATS}, "model/real_transfer_classifier.joblib")

    # ---- fee regressor (log target) ----
    moved = df["transferred"] == 1
    Xm, ym = X[moved], np.log1p(df.loc[moved, "fee_millions"])
    reg = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.08, random_state=42)
    pred_fee = np.expm1(cross_val_predict(reg, Xm, ym, cv=4))
    real = df.loc[moved, "fee_millions"]
    print("\n=== Fee prediction (4-fold CV, real fees) ===")
    print(f"MAE: EUR {mean_absolute_error(real, pred_fee):.1f}M | R2(log): {r2_score(ym, np.log1p(pred_fee)):.3f}")
    reg_full = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.08, random_state=42)
    reg_full.fit(Xm, ym)
    joblib.dump({"model": reg_full, "features": FEATURES, "cats": CATS}, "model/real_fee_regressor.joblib")

    # ---- what drives the prediction ----
    print("\nPermutation importance (classifier, holdout):")
    Xtr, Xte, ytr, yte = X.iloc[:-8000], X.iloc[-8000:], df["transferred"].iloc[:-8000], df["transferred"].iloc[-8000:]
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, class_weight={0: 1, 1: 2}, random_state=42).fit(Xtr, ytr)
    imp = permutation_importance(m, Xte, yte, n_repeats=3, random_state=42, n_jobs=-1)
    s = pd.Series(imp.importances_mean, index=X.columns).sort_values(ascending=False)
    print(s.head(12).round(4).to_string())

    import json
    Path("model").joinpath("real_feature_importance.json").write_text(json.dumps({k: round(float(v), 5) for k, v in s.items()}))
    Path("model").joinpath("real_metrics.json").write_text(json.dumps({
        "roc_auc": round(float(roc_auc_score(df['transferred'], p)), 3),
        "pr_auc": round(float(average_precision_score(df['transferred'], p)), 3),
        "brier": round(float(brier_score_loss(df['transferred'], p)), 3),
        "fee_mae": round(float(mean_absolute_error(real, pred_fee)), 1),
        "fee_r2": round(float(r2_score(ym, np.log1p(pred_fee))), 3),
        "n_rows": int(len(df)),
        "n_transfers": int(df['transferred'].sum()),
    }))
    print("Saved metrics + importance to model/")

if __name__ == "__main__":
    main()
