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

from train_real_model import prep, CATS, FEATURES
import rumours as rumours_mod

RAW = "data/raw"
app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True

@app.route("/assets/<path:name>")
def assets(name):
    return send_from_directory("assets", name)

# ---------------------------------------------------------------- faces
FACES = Path("data/faces"); FACES.mkdir(exist_ok=True)
_face_src = "data/face_urls.csv" if Path("data/face_urls.csv").exists() else f"{RAW}/players.csv"
faces_map = pd.read_csv(_face_src, usecols=["player_id", "image_url"], dtype={"player_id": "int32"}).dropna()
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
backtest = json.loads(Path("data/backtest.json").read_text()) if Path("data/backtest.json").exists() else None
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
rumours_mod.NAME_INDEX = rumours_mod._name_index(latest)

# ---- comparable transfers (kNN over real past transfers, precomputed) ----
_knn_meta = pd.read_csv("data/knn_transfers.csv", dtype={"player_id": "int32"})
_knn_pack = np.load("data/knn_matrix.npz")
Xk = _knn_pack["X"]            # already standardized
_knn_mean = _knn_pack["mean"]
_knn_scale = _knn_pack["scale"]
def comparable_transfers(row):
    xr = prep(pd.DataFrame([row]))
    for c in CATS:
        xr[c] = xr[c].cat.codes
    xr = xr[FEATURES].astype("float64").values
    xr = (xr - _knn_mean) / _knn_scale
    d = ((Xk - xr) ** 2).sum(axis=1)
    idx = np.argsort(d)[:6]
    nb = _knn_meta.iloc[idx]
    clubs = nb["to_club_name"].value_counts()
    return {
        "destinations": [{"club": c, "n": int(n)} for c, n in clubs.head(3).items()],
        "similar": [{"name": r["name"], "season": int(r["season"]), "club": r["to_club_name"],
                     "fee": float(r["fee_millions"]), "pid": int(r["player_id"])}
                    for _, r in nb.head(3).iterrows()],
    }

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

_risk_cache = {"t": 0, "rows": []}

def top_risk_rows(n=12):
    import time as _time
    if _time.time() - _risk_cache["t"] > 60:
        sample = latest[latest["season"] >= 2023]
        pr, fr = predict_batch(sample)
        risks = sample.assign(p=pr, f=fr)
        _risk_cache["rows"] = [{
            "name": r["name"], "player_id": int(r["player_id"]), "season": int(r["season"]),
            "position": r["position"], "league": r["league"], "age": float(r["age"]),
            "mv": float(r["market_value_eur"]), "prob": round(r["p"] * 100), "fee": round(r["f"], 1),
            "pid": int(r["player_id"]),
        } for _, r in risks.nlargest(30, "p").iterrows()]
        _risk_cache["t"] = _time.time()
    return _risk_cache["rows"][:n]

@app.route("/")
def index():
    return render_template("index.html", metrics=metrics, importance=importance,
                           chart=chart, top_risk=top_risk_rows(12))

@app.route("/backtest")
def backtest_page():
    return render_template("backtest.html", metrics=metrics, backtest=backtest)

@app.route("/api/toprisk")
def api_toprisk():
    n = min(int(request.args.get("n", 8)), 30)
    return jsonify(top_risk_rows(n))

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
    dests = comparable_transfers(r)
    return jsonify({
        "name": r["name"], "season": int(r["season"]), "position": r["position"],
        "league": r["league"], "age": float(r["age"]), "face": f"/face/{pid}",
        "apps": int(r["apps"]), "minutes": int(r["minutes"]),
        "goals": int(r["goals"]), "assists": int(r["assists"]),
        "availability": float(r["availability"]), "gap": int(r["longest_gap_days"]),
        "mv": float(r["market_value_eur"]), "mv_trend": float(r["mv_trend_12m"]),
        "prob": prob, "fee": fee,
        "destinations": dests["destinations"], "similar": dests["similar"],
        "seasons": sorted(rows["season"].astype(int).unique().tolist()),
    })

@app.route("/api/rumours")
def api_rumours():
    items = rumours_mod.fetch_rumours(latest)
    if not items:
        return jsonify({"items": [], "error": rumours_mod._cache.get("error")})
    # attach model verdicts for every mentioned player
    pairs = {(p["pid"], p["season"]) for it in items for p in it["players"]}
    vrows = df[df.apply(lambda r: (int(r["player_id"]), int(r["season"])) in pairs, axis=1)]
    probs, fees = predict_batch(vrows) if len(vrows) else ([], [])
    vmap = {int(pid): (float(p), float(f)) for pid, p, f in zip(vrows["player_id"], probs, fees)}
    for it in items:
        for p in it["players"]:
            p_ = vmap.get(p["pid"])
            p["prob"] = round(p_[0] * 100) if p_ is not None else None
            p["fee"] = round(p_[1], 1) if p_ is not None else None
            p["face"] = f"/face/{p['pid']}"
    return jsonify({"items": items})

if __name__ == "__main__":
    import os
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
