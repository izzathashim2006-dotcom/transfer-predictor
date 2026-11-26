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
backtest = json.loads(Path("data/backtest.json").read_text(encoding="utf-8")) if Path("data/backtest.json").exists() else None
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

_pt_path = Path("data/player_transfers.csv")
player_transfers = pd.read_csv(_pt_path, dtype={"player_id": "int32"}, parse_dates=["date"]) if _pt_path.exists() else pd.DataFrame()

_lt_path = Path("data/latest_transfers.csv")
latest_transfers = pd.read_csv(_lt_path, dtype={"player_id": "int32"}) if _lt_path.exists() else pd.DataFrame()

@app.route("/api/latest-transfers")
def api_latest_transfers():
    n = min(int(request.args.get("n", 12)), 60)
    if latest_transfers.empty:
        return jsonify([])
    rows = latest_transfers.head(n)
    return jsonify([{
        "pid": int(r["player_id"]), "name": r["name"], "date": r["date"],
        "from": r["from_club"], "to": r["to_club"], "fee": float(r["fee_millions"]),
        "pred_prob": None if pd.isna(r["pred_prob"]) else int(r["pred_prob"]),
        "pred_fee": None if pd.isna(r["pred_fee"]) else float(r["pred_fee"]),
        "face": f"/face/{int(r['player_id'])}",
    } for _, r in rows.iterrows()])
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

# ---- "why?" explanation: one-feature-at-a-time ablation vs dataset median ----
_WHY = [  # (raw column in df, human label)
    ("market_value_eur", "Market value"),
    ("mv_trend_12m", "Value momentum"),
    ("age", "Age"),
    ("minutes", "Playing time"),
    ("goals_per90", "Goal threat"),
    ("assists_per90", "Creativity"),
    ("availability", "Availability"),
    ("longest_gap_days", "Injury gap"),
    ("int_apps", "Int'l pedigree"),
]
_medians = {col: float(df[col].median()) for col, _ in _WHY}

def why_contributions(row):
    base_row = predict_row(row)
    rows = []
    for col, _label in _WHY:
        alt = row.copy()
        alt[col] = _medians[col]
        rows.append(alt)
    X = prep(pd.DataFrame(rows))
    probs = clf_pack["model"].predict_proba(X)[:, 1]
    pts = [(float(base_row[0]) - float(pr)) * 100 for pr in probs]  # + = pushes towards a move
    out = sorted(({"label": label, "pts": round(pt, 1)}
                  for (col, label), pt in zip(_WHY, pts)), key=lambda d: -abs(d["pts"]))
    return out

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

# ---- club scout: recommend targets by a club's historical signing profile ----
_SCOUT_VEC = ["age", "minutes", "goals_per90", "assists_per90", "availability",
              "longest_gap_days", "log1p_mv", "mv_trend_12m", "int_apps"]

def _vec(r):
    return np.array([float(r["age"]), float(r["minutes"]), float(r["goals_per90"]),
                     float(r["assists_per90"]), float(r["availability"]),
                     float(r["longest_gap_days"]), np.log1p(float(r["market_value_eur"])),
                     float(r["mv_trend_12m"]) if not pd.isna(r["mv_trend_12m"]) else 0.0,
                     float(r["int_apps"])])

_last_move = (player_transfers.sort_values("date").groupby("player_id").tail(1)[["player_id", "to_club"]]
              if not player_transfers.empty else pd.DataFrame(columns=["player_id", "to_club"]))
_last_move = dict(zip(_last_move["player_id"].astype(int), _last_move["to_club"]))
_df_key = {(int(r["player_id"]), int(r["season"])): i for i, r in df.iterrows()}
_club_counts = (player_transfers.groupby("to_club").size().sort_values(ascending=False)
                if not player_transfers.empty else pd.Series(dtype=int))
_names_map = dict(zip(df["player_id"].astype(int), df["name"]))

@app.route("/scout")
def scout_page():
    return render_template("scout.html", metrics=metrics)

@app.route("/api/clubs")
def api_clubs():
    q = request.args.get("q", "").strip().lower()
    norm = lambda s: str(s).encode("ascii", "ignore").decode("ascii").lower()
    clubs = [(c, int(n)) for c, n in _club_counts.items() if len(c) > 2]
    if q:
        clubs = [(c, n) for c, n in clubs if q in norm(c)]
    return jsonify([{"club": c, "n": n} for c, n in clubs[:10]])

@app.route("/api/scout")
def api_scout():
    club = request.args.get("club", "").strip()
    if not club:
        return jsonify({"error": "need ?club="}), 400
    signings = player_transfers[player_transfers["to_club"] == club]
    if len(signings) < 2:
        return jsonify({"error": "not enough signings in dataset for this club"}), 404
    sig_rows, keep = [], []
    for _, t in signings.iterrows():
        i = _df_key.get((int(t["player_id"]), int(np.where(t["date"].month >= 7, t["date"].year - 1, t["date"].year - 2))))
        if i is not None:
            sig_rows.append(df.loc[i])
            keep.append(i)
    sigs = pd.DataFrame(sig_rows)
    src_leagues = sigs["league"].value_counts().head(3).index.tolist()
    sig_vecs = np.array([_vec(r) for _, r in sigs.iterrows()])
    centroid = np.median(sig_vecs, axis=0)

    pool = df[(df["season"] >= 2024) & (df["apps"] >= 10)].copy()
    pool["log1p_mv"] = np.log1p(pool["market_value_eur"])
    pool = pool[[(_last_move.get(int(r["player_id"])) != club) for _, r in pool.iterrows()]]
    P = np.array([_vec(r) for _, r in pool.iterrows()])
    std = P.std(axis=0); std[std == 0] = 1
    d = (((P - centroid) / std) ** 2).sum(axis=1)
    # pipeline bonus: clubs re-sign from leagues they know
    lg = pool["league"].isin(src_leagues).values
    d = d * np.where(lg, 0.8, 1.0)
    order = np.argsort(d)[:18]
    cands = pool.iloc[order]
    pr, fr = predict_batch(cands)
    cands = cands.assign(_pr=pr, _fr=fr)
    out = []
    for (_, r), dist in zip(cands.iterrows(), d[order]):
        axes = {"age": "prime age", "minutes": "big minutes", "goals_per90": "goal threat",
                "assists_per90": "creator profile", "availability": "durability",
                "market_value_eur": "value band", "int_apps": "int'l pedigree"}
        fits = []
        for col, label in axes.items():
            cv_ = centroid[_SCOUT_VEC.index("log1p_mv")] if col == "market_value_eur" else centroid[_SCOUT_VEC.index(col)]
            v = np.log1p(float(r["market_value_eur"])) if col == "market_value_eur" else float(r[col])
            if abs(v - cv_) <= 0.6 * std[_SCOUT_VEC.index("log1p_mv" if col == "market_value_eur" else col)]:
                fits.append(label)
        out.append({"pid": int(r["player_id"]), "name": r["name"], "age": float(r["age"]),
                    "league": r["league"], "position": r["position"], "mv": float(r["market_value_eur"]),
                    "prob": round(float(r["_pr"]) * 100), "fee": round(float(r["_fr"]), 1),
                    "face": f"/face/{int(r['player_id'])}", "dist": round(float(dist), 2),
                    "fits": fits[:3]})
    recent = signings.sort_values("date", ascending=False).head(6)
    return jsonify({"club": club, "n_signings": int(len(signings)),
                    "median_age": round(float(sig_vecs[:, 0].mean()), 1),
                    "source_leagues": src_leagues,
                    "recent": [{"pid": int(t["player_id"]),
                                "name": _names_map.get(int(t["player_id"]), "Unknown"),
                                "when": t["date"].strftime("%b %Y"),
                                "from": t["from_club"], "fee": round(float(t["fee_millions"]), 1)}
                               for _, t in recent.iterrows()],
                    "targets": sorted(out, key=lambda x: (-x["prob"] / 100, x["dist"]))[:12]})

AXES = [
    ("Goal threat", "goals_per90", False),
    ("Creativity", "assists_per90", False),
    ("Minutes volume", "minutes", False),
    ("Availability", "availability", False),
    ("Fitness", "longest_gap_days", True),
    ("Market value", "market_value_eur", False),
    ("Momentum 12m", "mv_trend_12m", False),
    ("Int'l pedigree", "int_apps", False),
]
_pct = {col: np.sort(df[col].fillna(df[col].median()).values) for _, col, _ in AXES}
_pct["__age__"] = np.sort(np.abs(df["age"].values - 24.5))

def axis_scores(r):
    out = []
    for _, col, invert in AXES:
        v = r[col]
        v = 0.0 if pd.isna(v) else float(v)
        rank = float(np.searchsorted(_pct[col], v) / len(_pct[col]) * 100)
        out.append(int(round(100 - rank)) if invert else int(round(rank)))
    age_rank = float(np.searchsorted(_pct["__age__"], abs(float(r["age"]) - 24.5)) / len(_pct["__age__"]) * 100)
    out.append(int(round(100 - age_rank)))
    return out

def _player_payload(r, prob, fee):
    return {"pid": int(r["player_id"]), "name": r["name"], "season": int(r["season"]),
            "position": r["position"], "league": r["league"], "age": float(r["age"]),
            "mv": float(r["market_value_eur"]), "prob": prob, "fee": fee,
            "face": f"/face/{int(r['player_id'])}",
            "stats": [["Apps", int(r["apps"])], ["Minutes", int(r["minutes"])],
                      ["Goals", int(r["goals"])], ["Assists", int(r["assists"])],
                      ["Goals/90", round(float(r["goals_per90"]), 2)],
                      ["Assists/90", round(float(r["assists_per90"]), 2)],
                      ["Availability", str(int(round(float(r["availability"]) * 100))) + "%"],
                      ["Longest injury gap", str(int(r["longest_gap_days"])) + "d"],
                      ["Value trend 12m", ("+" if r["mv_trend_12m"] >= 0 else "") + str(int(round(float(r["mv_trend_12m"]) * 100))) + "%"]],
            "axes": axis_scores(r)}

@app.route("/player/<int:pid>")
def player_page(pid):
    rows = df[df["player_id"] == pid].sort_values("season")
    if rows.empty:
        return render_template("404.html"), 404
    probs, fees = predict_batch(rows)
    seasons = []
    for (_, r), pr, fr in zip(rows.iterrows(), probs, fees):
        seasons.append({
            "season": int(r["season"]), "league": r["league"], "position": r["position"],
            "apps": int(r["apps"]), "minutes": int(r["minutes"]), "goals": int(r["goals"]),
            "assists": int(r["assists"]), "availability": round(float(r["availability"]) * 100),
            "gap": int(r["longest_gap_days"]), "int_apps": int(r["int_apps"]),
            "mv": float(r["market_value_eur"]), "trend": float(r["mv_trend_12m"]),
            "prob": round(float(pr) * 100), "fee": round(float(fr), 1),
        })
    latest = rows.iloc[-1]
    hist = player_transfers[player_transfers["player_id"] == pid] if not player_transfers.empty else pd.DataFrame()
    transfers = [{"date": r["date"].strftime("%b %Y"), "from": r["from_club"], "to": r["to_club"],
                  "fee": float(r["fee_millions"])} for _, r in hist.iterrows()]
    payload = {
        "pid": pid, "name": latest["name"], "position": latest["position"],
        "league": latest["league"], "age": float(latest["age"]),
        "mv": float(latest["market_value_eur"]),
        "face": f"/face/{pid}",
        "foot": latest["foot"], "height": None if pd.isna(latest["height_cm"]) else int(latest["height_cm"]),
        "seasons": seasons, "transfers": transfers,
        "why": why_contributions(latest),
        "latest_prob": seasons[-1]["prob"], "latest_fee": seasons[-1]["fee"],
        "latest_season": seasons[-1]["season"],
    }
    return render_template("player.html", payload=payload, metrics=metrics)

@app.route("/compare")
def compare_page():
    return render_template("compare.html", metrics=metrics)

@app.route("/api/compare")
def api_compare():
    try:
        pid_a, pid_b = int(request.args["a"]), int(request.args["b"])
    except (KeyError, ValueError):
        return jsonify({"error": "need ?a=<player_id>&b=<player_id>"}), 400
    rows = df[df["player_id"].isin([pid_a, pid_b])].sort_values("season").groupby("player_id").tail(1)
    if len(rows) < 2:
        return jsonify({"error": "player not found"}), 404
    probs, fees = predict_batch(rows)
    out = {}
    for (_, r), pr, fr in zip(rows.iterrows(), probs, fees):
        pid = int(r["player_id"])
        out["a" if pid == pid_a else "b"] = _player_payload(r, round(float(pr) * 100), round(float(fr), 1))
    pa, pb = out["a"]["prob"], out["b"]["prob"]
    if abs(pa - pb) < 5:
        out["verdict"] = "The model sees them as near-equal flight risks this window."
    else:
        hi, lo = (out["a"], out["b"]) if pa > pb else (out["b"], out["a"])
        out["verdict"] = f"The model rates {hi['name']} {'%.1f' % (max(pa,pb)/max(min(pa,pb),1e-6))}x more likely to move ({hi['prob']}% vs {lo['prob']}%)."
    out["axes"] = [a[0] for a in AXES] + ["Age prime"]
    return jsonify(out)

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
    why = why_contributions(r)
    dests = comparable_transfers(r)
    return jsonify({
        "name": r["name"], "season": int(r["season"]), "position": r["position"],
        "league": r["league"], "age": float(r["age"]), "face": f"/face/{pid}",
        "apps": int(r["apps"]), "minutes": int(r["minutes"]),
        "goals": int(r["goals"]), "assists": int(r["assists"]),
        "availability": float(r["availability"]), "gap": int(r["longest_gap_days"]),
        "mv": float(r["market_value_eur"]), "mv_trend": float(r["mv_trend_12m"]),
        "prob": prob, "fee": fee, "why": why,
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
