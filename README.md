# ⚽ Transfer Predictor

Football (soccer) transfer prediction from **real Transfermarkt data** — with a live web app.

> Two gradient-boosted models estimate, for any of 56,204 real player-seasons (2013–2024):
> **1)** the probability the player moves clubs in the next window, and
> **2)** the expected transfer fee in EUR millions.

## 🌐 Live website

**https://transfer-predictor-teea.onrender.com** (hosted free on Render — first visit after idle may take ~50s to wake)

## 💻 Run locally

```bash
pip install -r requirements.txt
python app.py            # -> http://localhost:5000
```

FootballTransfers-style UI with:

| Feature | Description |
|---|---|
| 🔎 Player search | Any player in the dataset, fuzzy + accent-insensitive, with official Transfermarkt portraits |
| 📊 Prediction card | Transfer probability, expected fee, season stats, availability & injury-gap metrics, market-value trend |
| 🏟️ Destination clubs | kNN over 1,200+ real past transfers — "most similar historical moves and where they went" |
| 🔴 Live rumours | Sky Sports Transfer Centre feed, player-matched, each with the model's verdict |
| 📈 Charts | Transfer rate by age/league, feature importance, fee distribution |
| ☀️ / 🌙 | Light/dark theme, follows OS preference, persisted |

## 📁 Project layout

```
app.py                   Flask web app (search, predict, rumours, faces)
templates/index.html     Single-page UI
build_real_dataset.py    Builds player-season features from raw Transfermarkt CSVs
train_real_model.py      Trains classifier + fee regressor, CV metrics
predict_real.py          CLI: python predict_real.py --name "Bellingham" --season 2022
rumours.py               Sky Sports rumours fetcher + player matcher
data/real_players.csv    56,204 real player-seasons (committed, 7.7 MB)
generate_dataset.py      Legacy synthetic-data pipeline (kept for experiments)
```

## 🔢 The models

**Features per player-season**

| Group | Columns |
|---|---|
| Performance | apps, minutes, goals, assists, goals/90, assists/90, cards, reds |
| Age | age at season end (Jun 30) |
| Injury / availability | longest gap between appearances, 21+ day gap count, availability = apps ÷ club games |
| Context | league, position, foot, height, market value (log) + 12-month valuation trend |

**Results (4-fold CV, real data)**

| Model | Metric | Score |
|---|---|---|
| Transfer classifier | ROC-AUC / PR-AUC / Brier | 0.65 / 0.16 / 0.18 |
| Fee regressor (log-fee) | MAE / R²(log) | EUR 3.8M / 0.78 |

Spot checks vs. reality:

| Player, last pre-transfer season | Predicted | Actual |
|---|---|---|
| Bellingham, Dortmund 2022/23 | 56% / EUR 114.3M | moved for EUR 103M |
| Declan Rice, West Ham 2022/23 | 27% / EUR 106.4M | moved for ~EUR 116M |

Destination classification from player attributes alone was tested and scored at chance
level (3.2% vs 3.3%) — real buyer interest isn't in the data — so the app uses honest
kNN "comparable transfers" instead, and live rumours cover the "who is interested" part.

## 🔄 Rebuilding from source data

`data/raw/` (not committed, ~230 MB) holds the
[Transfermarkt player-scores](https://huggingface.co/datasets/davidcariboo/player-scores)
files: `players.csv`, `appearances.csv`, `transfers.csv`, `player_valuations.csv`, `games.csv`.

```bash
python build_real_dataset.py   # -> data/real_players.csv
python train_real_model.py     # -> model/*.joblib + metrics
```

## ⚠️ Known limitations

- Mid-season (January) moves aren't labeled for the season in progress.
- Injury history is an appearance-gap proxy, not medical data.
- Transfermarkt fee "0" covers free transfers and loans; only paid moves are labeled.
- Predictions are estimates — real moves depend on club finances, managers and rumors
  the data cannot see.
