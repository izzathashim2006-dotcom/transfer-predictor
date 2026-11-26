"""Web app for the real-player transfer predictor.

Run:  python app.py  ->  http://localhost:5000
"""
import numpy as np
import pandas as pd
import joblib
import json
import urllib.request
from pathlib import Path
from flask import Flask, jsonify, render_template, request, send_from_directory, send_file, abort

from train_real_model import prep

RAW = "data/raw"
app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True

@app.route("/assets/<path:name>")
def assets(name):
    return send_from_directory("assets", name)

# ---------------------------------------------------------------- faces
FACES = Path("data/faces"); FACES.mkdir(exist_ok=True)
faces_map = pd.read_csv(f"{RAW}/players.csv", usecols=["player_id", "image_url"], dtype={"player_id": "int32"}).dropna()
faces_map = dict(zip(faces_map["player_id"].astype(int), faces_map["image_url"]))

@app.route("/face/<int:pid>")
def face(pid):
    path = FACES / f"{pid}.jpg"
    if not path.exists():
        url = faces_map.get(pid)
        if not url:
            abort(404)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            data = urllib.request.urlopen(req, timeout=10).read()
            path.write_bytes(data)
        except Exception:
            abort(404)
    return send_file(path, mimetype="image/jpeg", max_age=86400)

clf_pack = joblib.load("model/real_transfer_classifier.joblib")
reg_pack = joblib.load("model/real_fee_regressor.joblib")
metrics = json.loads(Path("model/real_metrics.json").read_text())
importance = json.loads(Path("model/real_feature_importance.json").read_text())
df = pd.read_csv("data/real_players.csv")
df["norm"] = df["name"].astype(str).str.normalize("NFKD").str.encode("ascii", "ignore").str.decode("ascii").str.lower()

# dataset stats for the charts
chart = {
    "age": df.groupby(pd.cut(df["age"], [17, 20, 23, 26, 29, 32, 40], include_lowest=True), observed=True)
             ["transferred"].mean().mul(100).round(1).tolist(),
    "age_bins": ["18-20", "21-23", "24-26", "27-29", "30-32", "33+"],
    "fees": df.loc[df.transferred == 1, "fee_millions"].clip(upper=120).tolist(),
    "by_league": df.groupby("league")["transferred"].mean().mul(100).round(1).sort_values(ascending=False).to_dict(),
}

latest = df.sort_values("season").drop_duplicates("player_id", keep="last")

def predict_row(row):
    X = prep(pd.DataFrame([row]))
    prob = float(clf_pack["model"].predict_proba(X)[0, 1])
    fee = float(np.expm1(reg_pack["model"].predict(X)[0]))
    return prob, fee

def predict_batch(rows):
    X = prep(rows)
    prob = clf_pack["model"].predict_proba(X)[:, 1]
    fee = np.expm1(reg_pack["model"].predict(X))
    return prob, fee

@app.route("/")
def index():
    # top risk players: highest predicted probability among latest seasons
    sample = latest[latest["season"] >= 2023]
    p, f = predict_batch(sample)
    risks = sample.assign(p=p, f=f)
    top_risk_rows = [{
        "name": r["name"], "player_id": int(r["player_id"]), "season": int(r["season"]),
        "position": r["position"], "league": r["league"], "age": float(r["age"]),
        "mv": float(r["market_value_eur"]), "prob": round(r["p"] * 100), "fee": round(r["f"], 1),
        "pid": int(r["player_id"]),
    } for _, r in risks.nlargest(12, "p").iterrows()]
    return render_template("index.html", metrics=metrics, importance=importance,
                           chart=chart, top_risk=top_risk_rows)

@app.route("/api/search")
def search():
    q = request.args.get("q", "").strip().lower()
    if len(q) < 2:
        return jsonify([])
    hits = latest[latest["norm"].str.contains(q, regex=False, na=False)].head(8)
    return jsonify([{
        "player_id": int(r["player_id"]), "name": r["name"], "season": int(r["season"]),
        "position": r["position"], "league": r["league"], "mv": float(r["market_value_eur"]),
        "pid": int(r["player_id"]),
    } for _, r in hits.iterrows()])

@app.route("/api/predict")
def predict():
    pid = int(request.args.get("player_id"))
    season = request.args.get("season")
    rows = df[df["player_id"] == pid]
    if season:
        row = rows[rows["season"] == int(season)]
    else:
        row = rows.sort_values("season").tail(1)
    if row.empty:
        return jsonify({"error": "not found"}), 404
    r = row.iloc[0]
    prob, fee = predict_row(r)
    return jsonify({
        "name": r["name"], "season": int(r["season"]), "position": r["position"],
        "league": r["league"], "age": float(r["age"]), "face": f"/face/{pid}",
        "apps": int(r["apps"]), "minutes": int(r["minutes"]),
        "goals": int(r["goals"]), "assists": int(r["assists"]),
        "availability": float(r["availability"]), "gap": int(r["longest_gap_days"]),
        "mv": float(r["market_value_eur"]), "mv_trend": float(r["mv_trend_12m"]),
        "prob": prob, "fee": fee,
        "seasons": sorted(rows["season"].astype(int).unique().tolist()),
    })

if __name__ == "__main__":
    app.run(debug=False, port=5000)
