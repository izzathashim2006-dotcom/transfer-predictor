"""Generate a synthetic football player dataset for transfer prediction.

Creates realistic-ish data for ~2500 players:
  - performance: rating, goals, assists, minutes, pass accuracy, dribbles
  - age & position
  - injury history: injuries last 2 seasons, total days injured, recent surgery
  - context: contract years remaining, club tier, league strength
Targets:
  - transferred: did the player move clubs in the next transfer window (0/1)
  - fee_millions: transfer fee in EUR millions (if transferred)

Run:  python generate_dataset.py
Output: data/players.csv
"""
import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(42)

N = 2500
POSITIONS = ["GK", "DEF", "MID", "FWD"]
LEAGUES = ["Premier League", "La Liga", "Serie A", "Bundesliga", "Ligue 1", "Eredivisie", "Primeira Liga"]
# rough league strength multipliers (for fee valuation)
LEAGUE_STRENGTH = {
    "Premier League": 1.35, "La Liga": 1.15, "Serie A": 1.05,
    "Bundesliga": 1.10, "Ligue 1": 0.95, "Eredivisie": 0.75, "Primeira Liga": 0.70,
}

df = pd.DataFrame({
    "player_id": [f"P{i:05d}" for i in range(N)],
    "position": rng.choice(POSITIONS, N, p=[0.11, 0.32, 0.34, 0.23]),
    "age": np.clip(rng.normal(26, 4.3, N).round(), 17, 38).astype(int),
    "league": rng.choice(LEAGUES, N, p=[0.28, 0.17, 0.15, 0.14, 0.12, 0.08, 0.06]),
})

# Current ability declines with age (peak ~26-27)
age_effect = -0.010 * (df["age"] - 26.5) ** 2
df["rating"] = np.clip(rng.normal(7.0, 0.35, N) + age_effect, 5.8, 9.2).round(2)

# Performance correlates with rating; positions shift goal/assist profiles
df["minutes"] = np.clip(rng.normal(2200, 800, N) + (df["rating"] - 7.0) * 900, 90, 3800).astype(int)
pos = df["position"]
df["goals"] = np.where(
    pos == "FWD", rng.poisson(8, N) + (df["rating"] > 7.5) * 6,
    np.where(pos == "MID", rng.poisson(3, N), np.where(pos == "DEF", rng.poisson(1, N), 0)),
)
df["assists"] = np.where(
    pos == "MID", rng.poisson(5, N) + (df["rating"] > 7.5) * 4,
    np.where(pos == "FWD", rng.poisson(3, N), rng.poisson(1, N)),
)
df["pass_accuracy"] = np.clip(rng.normal(80, 6, N) + np.where(pos == "GK", 3, 0) + (df["rating"] - 7.0) * 8, 58, 96).round(1)
df["dribbles_per_game"] = np.clip(rng.gamma(2.0, 0.7, N) * np.where(pos == "FWD", 1.4, np.where(pos == "MID", 1.2, 0.5)), 0, 8).round(1)

# Injury history
df["injuries_last_2y"] = rng.poisson(1.2, N).clip(0, 8)
df["days_injured_last_2y"] = (df["injuries_last_2y"] * rng.gamma(2.0, 14, N)).round().astype(int)
df["recent_surgery"] = rng.binomial(1, 0.06, N)
# Injuries reduce availability (minutes) and slightly age the body
df["minutes"] = (df["minutes"] * np.clip(1 - 0.008 * df["days_injured_last_2y"], 0.55, 1)).astype(int)

df["contract_years_left"] = rng.choice([0, 1, 1, 2, 2, 2, 3, 3, 4, 5], N)
df["club_tier"] = rng.choice([1, 2, 3], N, p=[0.25, 0.45, 0.30])  # 1 = top club

# --- Ground-truth transfer dynamics (the model has to learn these) ---
age_curve = -0.022 * (df["age"] - 24.5) ** 2 / 10  # best odds ~24-25, drops at both ends
score = (
    1.1 * (df["rating"] - 7.0)
    + 1.4 * age_curve
    + 0.55 * df["contract_years_left"].eq(0)        # expiring contract -> big boost
    + 0.25 * df["contract_years_left"].eq(1)
    - 0.35 * df["contract_years_left"].ge(3)
    + 0.10 * df["goals"] / 10
    + 0.08 * df["assists"] / 10
    - 0.12 * df["injuries_last_2y"]
    - 0.010 * df["days_injured_last_2y"] / 10
    - 0.45 * df["recent_surgery"]
    + 0.30 * df["club_tier"].eq(3)                  # lower-tier clubs sell more
    + rng.normal(0, 0.15, N)                        # irreducible noise
)
prob = 1 / (1 + np.exp(-(2.2 * score - 1.1)))
df["transferred"] = rng.binomial(1, prob)

# --- Fee for transferred players ---
strength = df["league"].map(LEAGUE_STRENGTH)
prime_age = np.exp(-((df["age"] - 25) ** 2) / 28)
base = (
    14 + 90 * (df["rating"] - 6.6).clip(0) ** 1.6 * prime_age
    + 0.55 * df["goals"] + 0.35 * df["assists"]
    + 1.2 * (df["club_tier"] == 1) * strength * 10
)
injury_discount = 1 - 0.010 * df["days_injured_last_2y"] - 0.25 * df["recent_surgery"]
contract_discount = np.where(df["contract_years_left"] == 0, 0.55, np.where(df["contract_years_left"] == 1, 0.85, 1.0))
fee = base * strength * injury_discount.clip(0.35, 1) * contract_discount * np.exp(rng.normal(0, 0.22, N))
df["fee_millions"] = np.where(df["transferred"] == 1, fee.clip(0.2, 220).round(1), np.nan)

out = Path("data"); out.mkdir(exist_ok=True)
df.to_csv(out / "players.csv", index=False)
print(f"Wrote {len(df)} players to data/players.csv")
print(f"Transfer rate: {df['transferred'].mean():.1%}")
print(df[df.transferred == 1]["fee_millions"].describe().round(1))
