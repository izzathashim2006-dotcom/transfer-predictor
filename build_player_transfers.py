"""Build the per-player transfer history pack for profile pages.

All paid transfers since 2013, one row per player+date:
  python build_player_transfers.py  ->  data/player_transfers.csv
"""
import pandas as pd

tr = pd.read_csv("data/raw/transfers.csv",
                 usecols=["player_id", "transfer_date", "transfer_fee",
                          "from_club_name", "to_club_name"],
                 dtype={"player_id": "int32", "transfer_fee": "float64"}, parse_dates=["transfer_date"])
tr = tr[(tr["transfer_fee"] > 0) & (tr["transfer_date"] >= "2013-01-01")
        & (tr["transfer_date"] <= "2026-11-26")]
tr = tr.drop_duplicates(["player_id", "transfer_date"])
tr["transfer_fee"] = (tr["transfer_fee"] / 1e6).round(2)
out = tr.rename(columns={"transfer_date": "date", "transfer_fee": "fee_millions",
                         "from_club_name": "from_club", "to_club_name": "to_club"})
out = out.sort_values(["player_id", "date"], ascending=[True, False])
out.to_csv("data/player_transfers.csv", index=False)
print(f"saved {len(out):,} paid transfers -> data/player_transfers.csv")
