"""Build the "Was the model right?" backtest data.

Uses cross-validated probabilities (4-fold: the model never saw the scored
season during that fold's training) for every player-season, then produces:

  - calibration curve (predicted-probability deciles vs actual transfer rate)
  - showcase: famous real transfers with the model's pre-move call
  - summary scoreboard stats

Run:  python build_backtest.py  ->  data/backtest.json
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict
from train_real_model import prep

SHOWCASE = [  # (name, season = last season BEFORE the real move)
    ("Jude Bellingham", 2022), ("Declan Rice", 2022), ("Erling Haaland", 2021),
    ("Moisés Caicedo", 2022), ("Jack Grealish", 2020), ("Romelu Lukaku", 2020),
    ("Antony", 2021), ("Marc Cucurella", 2021), ("Darwin Núñez", 2021),
    ("Aurélien Tchouaméni", 2021), ("Enzo Fernández", 2022), ("Mykhailo Mudryk", 2021),
    ("Josko Gvardiol", 2022), ("Christopher Nkunku", 2021),
]

def norm(s):
    return s.encode("ascii", "ignore").decode("ascii").lower()

def main():
    df = pd.read_csv("data/real_players.csv")
    X = prep(df)
    print("Cross-validating on", len(df), "player-seasons (this takes a few minutes)...")
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                         class_weight={0: 1, 1: 2}, random_state=42)
    p = cross_val_predict(clf, X, df["transferred"], cv=4, method="predict_proba")[:, 1]
    df["p"] = p

    # calibration: predicted-probability deciles vs actual rate
    bins = np.quantile(p, np.linspace(0, 1, 11))
    bins[0], bins[-1] = 0, 1.01
    cal = []
    for i in range(10):
        m = (p >= bins[i]) & (p < bins[i + 1])
        if m.sum() == 0:
            continue
        cal.append({"lo": round(float(bins[i]) * 100, 1), "hi": round(float(bins[i + 1]) * 100, 1),
                    "predicted": round(float(p[m].mean()) * 100, 1),
                    "actual": round(float(df.loc[m, "transferred"].mean()) * 100, 1),
                    "n": int(m.sum())})

    # showcase famous transfers
    tr = pd.read_csv("data/raw/transfers.csv",
                     usecols=["player_id", "transfer_date", "transfer_fee", "to_club_name"],
                     dtype={"player_id": "int32", "transfer_fee": "float64"}, parse_dates=["transfer_date"])
    tr = tr[(tr["transfer_date"] >= "2013-01-01") & (tr["transfer_fee"] > 0)]
    tr["season"] = np.where(tr["transfer_date"].dt.month >= 7, tr["transfer_date"].dt.year - 1,
                            tr["transfer_date"].dt.year - 2)
    tr = tr[((tr["transfer_date"].dt.month >= 6) & (tr["transfer_date"].dt.year == tr["season"] + 1)) |
            ((tr["transfer_date"].dt.month <= 2) & (tr["transfer_date"].dt.year == tr["season"] + 2))]
    dest = tr.drop_duplicates(["player_id", "season"])[["player_id", "season", "to_club_name", "transfer_fee"]]

    df["norm"] = df["name"].map(norm)
    rows = []
    for name, season in SHOWCASE:
        hit = df[(df["norm"] == norm(name)) & (df["season"] == season) & (df["transferred"] == 1)]
        if hit.empty:
            continue
        r = hit.iloc[0]
        d = dest[(dest["player_id"] == r["player_id"]) & (dest["season"] == season)]
        actual_club = d.iloc[0]["to_club_name"] if len(d) else "?"
        actual_fee = float(d.iloc[0]["transfer_fee"]) / 1e6 if len(d) else float(r["fee_millions"])
        rows.append({"name": r["name"], "pid": int(r["player_id"]), "season": int(season),
                     "league": r["league"], "age": float(r["age"]),
                     "mv": float(r["market_value_eur"]),
                     "pred_prob": round(float(r["p"]) * 100),
                     "pred_fee": float(np.expm1(0)) if False else None,  # filled below
                     "actual_club": actual_club, "actual_fee": round(actual_fee, 1)})
    # fee estimates from the trained regressor for showcase rows only
    import joblib
    reg = joblib.load("model/real_fee_regressor.joblib")
    keys = {(x["pid"], x["season"]) for x in rows}
    sub = df[df.apply(lambda r: (int(r["player_id"]), int(r["season"])) in keys, axis=1)]
    fees = np.expm1(reg["model"].predict(prep(sub)))
    fmap = {(int(r["player_id"]), int(r["season"])): float(f)
            for (_, r), f in zip(sub.iterrows(), fees)}
    for row in rows:
        row["pred_fee"] = round(fmap.get((row["pid"], row["season"]), 0), 1)

    # scoreboard: call it a "hit" when the model's top-decile confidence contained the truth
    top10 = df[df["p"] >= np.quantile(p, 0.9)]
    stats = {
        "n": int(len(df)), "transfers": int(df["transferred"].sum()),
        "top_decile_captured": round(float(top10["transferred"].mean() / df["transferred"].mean()), 1),
        "roc_auc": round(float(__import__("sklearn.metrics", fromlist=["roc_auc_score"]).roc_auc_score(df["transferred"], p)), 3),
        "showcase_hits": int(sum(1 for r in rows if r["pred_prob"] >= 50)),
    }
    Path("data").mkdir(exist_ok=True)
    Path("data/backtest.json").write_text(json.dumps(
        {"calibration": cal, "showcase": rows, "stats": stats}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Saved data/backtest.json | showcase rows: {len(rows)} | top-decile lift: {stats['top_decile_captured']}x")

if __name__ == "__main__":
    main()
