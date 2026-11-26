"""Predict transfer probability and fee for REAL players by name.

Examples:
  python predict_real.py --name "Erling Haaland"
  python predict_real.py --name Bellingham --season 2020
  python predict_real.py --name "Kylan"          # fuzzy search lists matches

Uses the player's stats from data/real_players.csv (latest season by default).
"""
import argparse
import sys
import numpy as np
import pandas as pd
import joblib

def prep_row(row, features, cats):
    base = [f for f in features if f not in ("log_mv", "season", "height_cm")]
    X = pd.DataFrame([{f: row[f] for f in base}])
    X["log_mv"] = np.log1p(row["market_value_eur"])
    X["season"] = float(row["season"])
    X["height_cm"] = float(row["height_cm"])
    X = X[features]
    for c in cats:
        X[c] = pd.Categorical([row[c]], categories=clf_cache[c])
    return X

clf_cache = {}

def main():
    global clf_cache
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--season", type=int, default=None, help="player-season to use (default: latest)")
    a = ap.parse_args()

    df = pd.read_csv("data/real_players.csv")
    norm = lambda s: s.astype(str).str.normalize("NFKD").str.encode("ascii", "ignore").str.decode("ascii")
    hits = df[norm(df["name"]).str.contains(norm(pd.Series([a.name])).iloc[0], case=False, regex=False, na=False)]
    if hits.empty:
        print(f"No player matching '{a.name}' in dataset."); sys.exit(1)
    cands = hits.sort_values("season").drop_duplicates("player_id", keep="last")
    if len(cands) > 1:
        print("Multiple matches — pick one:")
        for i, (_, r) in enumerate(cands.iterrows()):
            print(f"  [{i}] {r['name']} ({r['position']}, {r['league']}, season {r['season']}, player_id {r['player_id']})")
        try:
            pick = int(input("Index: "))
        except (ValueError, EOFError):
            pick = 0
        player = cands.iloc[pick]
    else:
        player = cands.iloc[0]

    if a.season is not None:
        row = df[(df.player_id == player.player_id) & (df.season == a.season)]
        if row.empty:
            print(f"{player['name']}: no season {a.season} in dataset (has {hits[hits.player_id==player.player_id].season.min()}-{player['season']})")
            sys.exit(1)
        player = row.iloc[0]

    clf = joblib.load("model/real_transfer_classifier.joblib")
    reg = joblib.load("model/real_fee_regressor.joblib")
    clf_cache = {c: list(pd.read_csv("data/real_players.csv", usecols=[c])[c].dropna().unique()) for c in clf["cats"]}
    X = prep_row(player, clf["features"], clf["cats"])

    prob = float(clf["model"].predict_proba(X)[0, 1])
    fee = float(np.expm1(reg["model"].predict(X)[0]))

    print(f"\n{player['name']} — {player['position']} | season {int(player['season'])} | {player['league']}")
    print(f"  age {player['age']} | {int(player['apps'])} apps, {int(player['goals'])}G {int(player['assists'])}A "
          f"| availability {player['availability']:.0%} | longest injury gap {int(player['longest_gap_days'])}d")
    print(f"  market value: EUR {player['market_value_eur']/1e6:.1f}M")
    print(f"\n  >>> Transfer probability : {prob:.0%}")
    print(f"  >>> Expected fee         : EUR {fee:.1f}M")

if __name__ == "__main__":
    main()
