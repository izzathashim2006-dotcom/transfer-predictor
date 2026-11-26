<img src="assets/icon.svg" width="64" height="64" alt="">

# TransfetPrediction

Football (soccer) transfer prediction from player **performance**, **age**, and **injury history**.

## What it does

Two gradient-boosted models trained on 2,500 players:

1. **Transfer classifier** — probability the player moves clubs in the next window.
2. **Fee regressor** — expected transfer fee in EUR millions if he moves.

Key signals learned: overall rating, contract years left, days injured (last 2 seasons), age curve (peak odds ~24–25), goals/assists, club tier, league strength.

## Usage

```bash
pip install pandas scikit-learn numpy

# 1. Generate the dataset -> data/players.csv
python generate_dataset.py

# 2. Train + evaluate both models (5-fold CV) -> model/*.joblib
python train_model.py

# 3. Predict for a single player
python predict.py --position FWD --age 24 --rating 8.1 --minutes 2700 --goals 18 \
  --assists 6 --pass-accuracy 82 --dribbles-per-game 3.2 --injuries-last-2y 0 \
  --days-injured-last-2y 0 --contract-years-left 1 --club-tier 1 --league "Premier League"

# Or run two built-in demo players
python predict.py --example
```

## Features

| Group | Columns |
|---|---|
| Performance | rating, minutes, goals, assists, pass accuracy, dribbles/game |
| Age | age |
| Injury history | injuries last 2y, days injured last 2y, recent surgery |
| Context | contract years left, club tier, league, position |

## Current metrics (5-fold CV)

- Transfer classifier: ROC-AUC **0.76**, Brier **0.13**
- Fee regressor (transfers only): MAE **EUR 9.7M**, R² **0.76**

## Data note

`generate_dataset.py` produces a **synthetic** dataset with realistic transfer
dynamics (age curve, expiring-contract boost, injury discounts, league
multipliers). To use real data, replace `data/players.csv` with the same columns
— e.g. exported from FIFA/FM databases or Transfermarkt scrapes — and rerun
`train_model.py`; no other code changes needed.
