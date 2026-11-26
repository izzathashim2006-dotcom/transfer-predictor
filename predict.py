"""Predict transfer probability and fee for a single player from the command line.

Examples:
  python predict.py --name "Example Striker" --position FWD --age 24 --rating 8.1 \
      --minutes 2700 --goals 18 --assists 6 --pass-accuracy 82 --dribbles-per-game 3.2 \
      --injuries-last-2y 0 --days-injured-last-2y 0 --contract-years-left 1 --club-tier 1 \
      --league "Premier League"

  python predict.py --example          # run two built-in demo players
"""
import argparse
import sys
import joblib
import pandas as pd

FEATURES = [
    "age", "rating", "minutes", "goals", "assists", "pass_accuracy",
    "dribbles_per_game", "injuries_last_2y", "days_injured_last_2y",
    "recent_surgery", "contract_years_left", "club_tier", "position", "league",
]

def predict(player: dict):
    clf = joblib.load("model/transfer_classifier.joblib")
    reg = joblib.load("model/fee_regressor.joblib")
    row = pd.DataFrame([player], columns=FEATURES)
    X = pd.get_dummies(row, columns=["position", "league"])
    X = X.reindex(columns=clf["columns"], fill_value=0)
    p = float(clf["model"].predict_proba(X)[0, 1])
    fee = float(reg["model"].predict(X)[0]) if p > 0.30 else 0.0
    return p, fee

def demo():
    players = [
        ("Star prospect", dict(position="FWD", age=23, rating=8.0, minutes=2800, goals=19,
             assists=7, pass_accuracy=82.0, dribbles_per_game=3.4, injuries_last_2y=0,
             days_injured_last_2y=10, recent_surgery=0, contract_years_left=1,
             club_tier=2, league="Eredivisie")),
        ("Injured veteran", dict(position="MID", age=32, rating=7.4, minutes=1400, goals=2,
             assists=3, pass_accuracy=84.0, dribbles_per_game=1.1, injuries_last_2y=4,
             days_injured_last_2y=240, recent_surgery=1, contract_years_left=3,
             club_tier=1, league="Premier League")),
    ]
    for name, p in players:
        prob, fee = predict(p)
        print(f"\n{name}: transfer chance {prob:.0%}, expected fee EUR {fee:.1f}M")

def main():
    ap = argparse.ArgumentParser(description="Football transfer prediction")
    ap.add_argument("--example", action="store_true")
    ap.add_argument("--position", default="MID", choices=["GK", "DEF", "MID", "FWD"])
    ap.add_argument("--age", type=int, default=26)
    ap.add_argument("--rating", type=float, default=7.0)
    ap.add_argument("--minutes", type=int, default=1800)
    ap.add_argument("--goals", type=int, default=3)
    ap.add_argument("--assists", type=int, default=3)
    ap.add_argument("--pass-accuracy", dest="pass_accuracy", type=float, default=80.0)
    ap.add_argument("--dribbles-per-game", dest="dribbles_per_game", type=float, default=1.5)
    ap.add_argument("--injuries-last-2y", dest="injuries_last_2y", type=int, default=1)
    ap.add_argument("--days-injured-last-2y", dest="days_injured_last_2y", type=int, default=30)
    ap.add_argument("--recent-surgery", dest="recent_surgery", type=int, default=0, choices=[0, 1])
    ap.add_argument("--contract-years-left", dest="contract_years_left", type=int, default=2)
    ap.add_argument("--club-tier", type=int, default=2, choices=[1, 2, 3])
    ap.add_argument("--league", default="Premier League")
    a = vars(ap.parse_args())
    if a.pop("example"):
        demo(); return
    prob, fee = predict(a)
    print(f"Transfer probability : {prob:.0%}")
    print(f"Expected transfer fee: EUR {fee:.1f}M")

if __name__ == "__main__":
    main()
