<img src="assets/icon.svg" width="64" height="64" alt="">

# TransfetPrediction

Football (soccer) transfer prediction from **real Transfermarkt data**: player performance, age, and injury/availability history.

## Real-data pipeline (main)

Data source: [Transfermarkt player-scores](https://huggingface.co/datasets/davidcariboo/player-scores) (players, appearances, transfers, valuations, games). Files are downloaded to `data/raw/` (not committed).

```bash
pip install pandas scikit-learn numpy

# 1. Download raw CSVs (players, appearances, transfers, player_valuations, games)
#    from the dataset link above into data/raw/

# 2. Build player-season features -> data/real_players.csv (56k rows, 2013-2024)
python build_real_dataset.py

# 3. Train + evaluate -> model/real_*.joblib
python train_real_model.py

# 4. Predict a real player by name (fuzzy, accent-insensitive search)
python predict_real.py --name "Bellingham" --season 2022
python predict_real.py --name "Haaland"
```

### Features per player-season

| Group | Columns |
|---|---|
| Performance | apps, minutes, goals, assists, goals/90, assists/90, cards, reds |
| Age | age at season end (Jun 30) |
| Injury / availability | longest gap between appearances (days), count of 21+ day gaps, availability = apps / club games |
| Context | league (top-12 + other), position, foot, height, market value (log) + 12-month valuation trend |
| Label | paid transfer between Jun 1 (after season) and Feb 28 (next winter) |
| Fee target | real transfer fee in EUR millions |

### Results (4-fold CV, real data)

- Transfer classifier (HistGradientBoosting): ROC-AUC **0.65**, PR-AUC **0.16**, Brier **0.18**
  - low AUC is honest here — real moves depend on club finances, manager changes and rumors the data can't see
- Fee regressor (log-fee): MAE **EUR 3.8M**, R²(log) **0.78**

### Spot checks vs. reality

| Player, last pre-transfer season | Predicted | Actual |
|---|---|---|
| Bellingham, Dortmund 2022/23 | 56% / EUR 114.3M | moved for EUR 103M |
| Declan Rice, West Ham 2022/23 | 27% / EUR 106.4M | moved for ~EUR 116M |

### Known limitations

- Mid-season (e.g. January) moves are not labeled as transfers for the season in progress.
- Injury history is a proxy (appearance gaps vs. club schedule), not medical data.
- Transfermarkt fee "0" covers free transfers and loans; only paid moves are labeled.

## Synthetic pipeline (kept for experiments)

`generate_dataset.py` / `train_model.py` / `predict.py` — 2,500 synthetic players with ground-truth transfer dynamics; useful for testing pipeline changes quickly. Metrics: ROC-AUC 0.76, fee MAE EUR 9.7M.
