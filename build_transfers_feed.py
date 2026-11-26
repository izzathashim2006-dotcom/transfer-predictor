"""Build the 'latest completed transfers' feed for the web app.

Takes the most recent paid transfers from data/raw/transfers.csv, attaches
player names/faces, and the model's verdict from the player's previous season
(the season before the move — what the model would have said at the time).

Run:  python build_transfers_feed.py  ->  data/latest_transfers.csv
"""
import numpy as np
import pandas as pd
import joblib

RAW = "data/raw"
N = 60

tr = pd.read_csv(f"{RAW}/transfers.csv",
                 usecols=["player_id", "transfer_date", "transfer_fee",
                          "from_club_name", "to_club_name"],
                 dtype={"player_id": "int32", "transfer_fee": "float64"}, parse_dates=["transfer_date"])
import datetime as _dt
today = pd.Timestamp(_dt.date.today())
tr = tr[(tr["transfer_fee"] > 0) & (tr["transfer_date"] >= "2025-07-01")
        & (tr["transfer_date"] <= today)]   # completed moves only (drop scheduled/future)
pool = tr.drop_duplicates(["player_id", "transfer_date"])
tr = pd.concat([pool.sort_values("transfer_date", ascending=False).head(40),   # most recent
                pool.sort_values("transfer_fee", ascending=False).head(25)])   # biggest fees
tr = tr.drop_duplicates(["player_id", "transfer_date"]).sort_values("transfer_date", ascending=False).head(N)

names = pd.read_csv(f"{RAW}/players.csv", usecols=["player_id", "name"], dtype={"player_id": "int32"})
tr = tr.merge(names, on="player_id", how="left")

# model verdict from the season before the move
df = pd.read_csv("data/real_players.csv")
clf = joblib.load("model/real_transfer_classifier.joblib")
reg = joblib.load("model/real_fee_regressor.joblib")
from train_real_model import prep

tr["pseason"] = np.where(tr["transfer_date"].dt.month >= 7, tr["transfer_date"].dt.year - 1,
                         tr["transfer_date"].dt.year - 2)
key = tr.set_index(["player_id", "pseason"]).index
prev = df[df.apply(lambda r: (int(r["player_id"]), int(r["season"])) in set(key), axis=1)]
probs, fees = None, None
if len(prev):
    Xp = prep(prev)
    probs = clf["model"].predict_proba(Xp)[:, 1] * 100
    fees = np.expm1(reg["model"].predict(Xp))
vmap = {(int(r["player_id"]), int(r["season"])): (round(float(p)), round(float(f), 1))
        for (_, r), p, f in zip(prev.iterrows(), probs, fees)}

tr["pred_prob"] = [vmap.get((int(r["player_id"]), int(r["pseason"])), (None, None))[0] for _, r in tr.iterrows()]
tr["pred_fee"] = [vmap.get((int(r["player_id"]), int(r["pseason"])), (None, None))[1] for _, r in tr.iterrows()]

out = tr[["player_id", "name", "transfer_date", "from_club_name", "to_club_name",
          "transfer_fee", "pred_prob", "pred_fee"]].copy()
out["transfer_fee"] = (out["transfer_fee"] / 1e6).round(1)
out = out.rename(columns={"transfer_date": "date", "transfer_fee": "fee_millions",
                          "from_club_name": "from_club", "to_club_name": "to_club"})
out.to_csv("data/latest_transfers.csv", index=False)
print(f"saved {len(out)} latest transfers -> data/latest_transfers.csv")
print(out.head(8).to_string(index=False))
