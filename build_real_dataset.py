"""Build a real-player transfer dataset from Transfermarkt data (player-scores).

Inputs (data/raw/): players.csv, appearances.csv, transfers.csv, player_valuations.csv, games.csv
Output (data/real_players.csv): one row per player-season with

  performance : appearances, minutes, goals, assists, goals/90, assists/90, cards
  age         : age at season end
  injury proxy: longest gap between appearances (days), number of 21+ day gaps,
                availability = apps / club games that season
  context     : league, position, foot, height, market value + 12m trend
  label       : transferred (paid transfer in the summer+winter window after the season)
  target      : real transfer fee in EUR (for moved players)

Run:  python build_real_dataset.py
"""
import numpy as np
import pandas as pd

RAW = "data/raw"

# ---------------------------------------------------------------- appearances
print("Loading appearances...")
apps = pd.read_csv(
    f"{RAW}/appearances.csv",
    usecols=["player_id", "date", "player_club_id", "competition_id",
             "goals", "assists", "yellow_cards", "red_cards", "minutes_played"],
    dtype={"player_id": "int32", "player_club_id": "int32", "competition_id": "category",
           "goals": "int16", "assists": "int16", "yellow_cards": "int16",
           "red_cards": "int16", "minutes_played": "int16"},
    parse_dates=["date"],
)
apps["season"] = np.where(apps["date"].dt.month >= 7, apps["date"].dt.year, apps["date"].dt.year - 1)

# competition typing: club comps, national-team comps, international club cups
_g0 = pd.read_csv(f"{RAW}/games.csv", usecols=["competition_id", "competition_type"])
_type = dict(zip(_g0["competition_id"], _g0["competition_type"]))
club_comps = {c for c, t in _type.items() if t != "national_team_competition"}
national_comps = {c for c, t in _type.items() if t == "national_team_competition"}
euro_cup_comps = {c for c, t in _type.items() if t == "international_cup"}
del _g0

# achievements-adjacent features: national-team + European cup appearances that season
_int_mask = apps["competition_id"].isin(national_comps)
_int = apps[_int_mask].groupby(["player_id", "season"]).agg(
    int_apps=("minutes_played", "size"), int_goals=("goals", "sum")).reset_index()
_euro = apps[apps["competition_id"].isin(euro_cup_comps)].groupby(["player_id", "season"]).agg(
    euro_apps=("minutes_played", "size")).reset_index()

# keep club competitions only (drop national-team fixtures) for club performance
apps = apps[apps["competition_id"].isin(club_comps)]

print("Aggregating per player-season...")
g = apps.groupby(["player_id", "season"])
perf = g.agg(
    apps=("minutes_played", "size"),
    minutes=("minutes_played", "sum"),
    goals=("goals", "sum"),
    assists=("assists", "sum"),
    cards=("yellow_cards", "sum"),
    reds=("red_cards", "sum"),
    first_app=("date", "min"),
    last_app=("date", "max"),
    club_id=("player_club_id", lambda s: s.mode().iat[0]),
    league=("competition_id", lambda s: s.mode().iat[0]),
).reset_index()

# injury proxies: gaps between consecutive appearances within a season
print("Computing appearance gaps (injury proxy)...")
gap_stats = {}
for (pid, season), d in apps.groupby(["player_id", "season"]):
    dates = np.sort(d["date"].values)
    if len(dates) > 1:
        gaps = np.diff(dates).astype("timedelta64[D]").astype(int)
        gap_stats[(pid, season)] = (gaps.max(), int((gaps > 21).sum()))
    else:
        gap_stats[(pid, season)] = (0, 0)
perf["longest_gap_days"] = perf.set_index(["player_id", "season"]).index.map(
    lambda k: gap_stats[k][0]).values
perf["gaps_over_21d"] = perf.set_index(["player_id", "season"]).index.map(
    lambda k: gap_stats[k][1]).values
del apps

# ---------------------------------------------------------------- club games
print("Loading games for availability...")
games = pd.read_csv(f"{RAW}/games.csv",
                    usecols=["season", "home_club_id", "away_club_id", "competition_type"])
games = games[games["competition_type"] != "national_team_competition"]
home = games.groupby(["season", "home_club_id"]).size().reset_index(name="n")
away = games.groupby(["season", "away_club_id"]).size().reset_index(name="n")
home.columns = ["season", "club_id", "n"]
away.columns = ["season", "club_id", "n"]
club_games = pd.concat([home, away], ignore_index=True)
club_games = club_games.groupby(["season", "club_id"], as_index=False)["n"].sum()
club_games.columns = ["season", "club_id", "club_games"]
perf = perf.merge(club_games, left_on=["season", "club_id"], right_on=["season", "club_id"], how="left")
perf = perf.drop(columns="club_id")
perf["club_games"] = perf["club_games"].fillna(20).clip(15, 70)
perf["availability"] = (perf["apps"] / perf["club_games"]).clip(0, 1.15).round(3)
perf = perf.merge(_int, on=["player_id", "season"], how="left")
perf = perf.merge(_euro, on=["player_id", "season"], how="left")
perf["int_apps"] = perf["int_apps"].fillna(0).astype(int)
perf["int_goals"] = perf["int_goals"].fillna(0).astype(int)
perf["euro_apps"] = perf["euro_apps"].fillna(0).astype(int)

# ---------------------------------------------------------------- valuations
print("Loading valuations...")
vals = pd.read_csv(f"{RAW}/player_valuations.csv",
                   usecols=["player_id", "date", "market_value_in_eur"],
                   dtype={"player_id": "int32"}, parse_dates=["date"])
vals = vals[vals["market_value_in_eur"] > 0].sort_values("date")

def value_asof(season_end):
    v = vals[vals["date"] <= season_end]
    return v.groupby("player_id")["market_value_in_eur"].last()

mv_now = value_asof(pd.Timestamp("2030-01-01"))  # last known valuation
for y in range(2012, 2026):
    end = pd.Timestamp(f"{y+1}-06-30")
    yr = end.year
    perf.loc[perf["season"] == yr - 1, "mv_eur"] = perf.loc[perf["season"] == yr - 1, "player_id"].map(value_asof(end))
    past = value_asof(end - pd.Timedelta(days=365))
    key = perf["season"] == yr - 1
    perf.loc[key, "mv_prev_eur"] = perf.loc[key, "player_id"].map(past)
perf["mv_trend_12m"] = np.log1p(perf["mv_eur"]) - np.log1p(perf["mv_prev_eur"])

# ---------------------------------------------------------------- transfers
print("Loading transfers...")
tr = pd.read_csv(f"{RAW}/transfers.csv",
                 usecols=["player_id", "transfer_date", "transfer_fee", "transfer_season"],
                 dtype={"player_id": "int32", "transfer_fee": "float64"}, parse_dates=["transfer_date"])
tr = tr[tr["from_club_id"] != tr["to_club_id"]] if "from_club_id" in tr.columns else tr
tr = tr[(tr["transfer_date"] >= "2013-01-01") & (tr["transfer_date"] <= "2026-02-28")]

# window after season end: 1 June (Y+1) -> 28 Feb (Y+2)
tr["season"] = np.where(tr["transfer_date"].dt.month >= 7, tr["transfer_date"].dt.year - 1,
                        tr["transfer_date"].dt.year - 2)
# keep only moves inside Jun 1 (Y+1) .. Feb 28 (Y+2) relative to season Y
tr = tr[((tr["transfer_date"].dt.month >= 6) & (tr["transfer_date"].dt.year == tr["season"] + 1)) |
        ((tr["transfer_date"].dt.month <= 2) & (tr["transfer_date"].dt.year == tr["season"] + 2))]
agg = tr.groupby(["player_id", "season"]).agg(transferred=("transfer_fee", "size"),
                                             fee_millions=("transfer_fee", "max"))
agg["transferred"] = (agg["fee_millions"] > 0).astype(int)   # paid moves only
agg["fee_millions"] = (agg["fee_millions"] / 1e6).round(2)
perf = perf.merge(agg, on=["player_id", "season"], how="left")
perf["transferred"] = perf["transferred"].fillna(0).astype(int)
perf["fee_millions"] = perf["fee_millions"].fillna(np.nan)

# ---------------------------------------------------------------- player info
print("Merging player info...")
players = pd.read_csv(f"{RAW}/players.csv",
                      usecols=["player_id", "name", "position", "foot", "height_in_cm",
                               "date_of_birth", "sub_position"],
                      dtype={"player_id": "int32"}, parse_dates=["date_of_birth"])
perf = perf.merge(players, on="player_id", how="left")
perf["age"] = ((pd.to_datetime(perf["season"] + 1, format="%Y")
                + pd.offsets.DateOffset(months=5, days=29)
                - perf["date_of_birth"]).dt.days / 365.25).round(1)

# league bucket (top leagues only, rest -> other)
TOP = {"GB1": "Premier League", "ES1": "La Liga", "IT1": "Serie A", "L1": "Bundesliga",
       "FR1": "Ligue 1", "NL1": "Eredivisie", "PO1": "Primeira Liga", "BE1": "Belgium",
       "TR1": "Turkey", "SC1": "Scotland", "RU1": "Russia", "US1": "MLS"}
perf["league"] = perf["league"].astype(str).map(TOP).fillna("Other")
perf["goals_per90"] = (perf["goals"] / perf["minutes"] * 90).round(3)
perf["assists_per90"] = (perf["assists"] / perf["minutes"] * 90).round(3)

# ---------------------------------------------------------------- filter + save
df = perf[(perf["apps"] >= 10) & (perf["minutes"] >= 450) &
          (perf["season"].between(2013, 2024)) & perf["mv_eur"].notna() &
          perf["position"].notna()].copy()
df = df[["player_id", "name", "season", "position", "sub_position", "foot", "height_in_cm",
         "age", "league", "apps", "minutes", "goals", "assists", "goals_per90", "assists_per90",
         "cards", "reds", "longest_gap_days", "gaps_over_21d", "availability",
         "int_apps", "int_goals", "euro_apps",
         "mv_eur", "mv_trend_12m", "transferred", "fee_millions"]]
df = df.rename(columns={"height_in_cm": "height_cm", "mv_eur": "market_value_eur"})
df.to_csv("data/real_players.csv", index=False)

print(f"\nSaved {len(df):,} player-seasons -> data/real_players.csv")
print(f"Paid-transfer rate: {df['transferred'].mean():.1%}")
moved = df[df.transferred == 1]
print(f"Fees: median EUR {moved.fee_millions.median():.1f}M | p90 EUR {moved.fee_millions.quantile(0.9):.1f}M | max EUR {moved.fee_millions.max():.1f}M")
print(df.sample(5, random_state=1).to_string(index=False))
