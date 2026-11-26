"""Precompute the kNN 'comparable transfers' pack for the web app.

Run after build_real_dataset.py / train_real_model.py whenever FEATURES change:
  python build_knn.py  ->  data/knn_transfers.csv + data/knn_matrix.npz
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from train_real_model import prep, CATS

RAW = "data/raw"

df = pd.read_csv("data/real_players.csv")

_tr = pd.read_csv(f"{RAW}/transfers.csv",
                  usecols=["player_id", "transfer_date", "transfer_fee", "to_club_name"],
                  dtype={"player_id": "int32", "transfer_fee": "float64"}, parse_dates=["transfer_date"])
_tr = _tr[(_tr["transfer_date"] >= "2013-01-01") & (_tr["transfer_date"] <= "2026-02-28")]
_tr["season"] = np.where(_tr["transfer_date"].dt.month >= 7, _tr["transfer_date"].dt.year - 1,
                         _tr["transfer_date"].dt.year - 2)
_tr = _tr[((_tr["transfer_date"].dt.month >= 6) & (_tr["transfer_date"].dt.year == _tr["season"] + 1)) |
          ((_tr["transfer_date"].dt.month <= 2) & (_tr["transfer_date"].dt.year == _tr["season"] + 2))]
_tr = _tr[_tr["transfer_fee"] > 0].sort_values("transfer_fee", ascending=False)
_dest = _tr.drop_duplicates(["player_id", "season"])[["player_id", "season", "to_club_name"]]
moved = df[df["transferred"] == 1].merge(_dest, on=["player_id", "season"], how="inner")

Xk = prep(moved)
for c in CATS:
    Xk[c] = Xk[c].cat.codes
Xk = Xk.astype("float64")
scaler = StandardScaler().fit(Xk)
Xs = scaler.transform(Xk).astype("float32")

moved[["player_id", "name", "season", "to_club_name", "fee_millions"]].to_csv("data/knn_transfers.csv", index=False)
np.savez_compressed("data/knn_matrix.npz", X=Xs, mean=scaler.mean_.astype("float32"),
                    scale=scaler.scale_.astype("float32"))
print(f"knn pack rebuilt: {Xs.shape[0]} transfers x {Xs.shape[1]} features")
