"""Live transfer rumours from Sky Sports' Transfer Centre, matched to our dataset.

fetch_rumours() -> [{headline, body, time, players: [{name, pid}]}]
Player matching: exact full names + unique long surnames from data/real_players.csv.
Cached in memory for 10 minutes.
"""
import re
import json
import time
import urllib.request
import numpy as np
import pandas as pd

URL = "https://www.skysports.com/transfer-centre"
_cache = {"t": 0, "items": []}
TTL = 600

KEYWORDS = ["want", "target", "move", "deal", "interested", "bid", "offer",
            "signing", "keen", "talks", "loan", "fee", "agree", "transfer"]

def _name_index(df):
    idx = {}
    norm = lambda s: s.encode("ascii", "ignore").decode("ascii").lower()
    # full names
    for _, r in df.iterrows():
        idx[norm(r["name"])] = int(r["player_id"])
    # unique surnames (len >= 5) -> also "F. Surname" style is rare; keep simple
    sur = df["name"].astype(str).str.strip().str.split().str[-1]
    cnt = sur.value_counts()
    for surname in cnt[(cnt == 1) & (sur.str.len() >= 5)].index:
        norm_s = norm(surname)
        if norm_s not in idx:
            row = df[sur == surname].iloc[0]
            idx[norm_s] = int(row["player_id"])
    return idx

def _match_players(text, name_index, players_df):
    low = text.lower()
    hits = {}
    for name, pid in name_index.items():
        if " " not in name:  # full names only — surname-only matches cause false positives
            continue
        pos = low.find(name)
        if pos >= 0:
            hits[pid] = name
    out = []
    df_idx = players_df.set_index("player_id")
    for pid, name in hits.items():
        if pid in df_idx.index:
            r = df_idx.loc[pid]
            out.append({"name": r["name"], "pid": int(pid), "season": int(r["season"]),
                        "position": r["position"], "league": r["league"]})
    return sorted(out, key=lambda p: -len(p["name"]))

def fetch_rumours(players_df):
    if time.time() - _cache["t"] < TTL:
        return _cache["items"]
    try:
        req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        html = urllib.request.urlopen(req, timeout=15).read().decode("utf-8", "ignore")
        blobs = re.findall(r'"headline":\s*"(.*?)",\s*"url".*?"articleBody":\s*"(.*?)"(?:,\s*"publisher")', html, re.S)
        times = re.findall(r'"headline":\s*"(.*?)",\s*"url".*?"datePublished":\s*"(.*?)"', html, re.S)
        tmap = dict(times)
        items = []
        for headline, body in blobs:
            headline = headline.encode().decode("unicode_escape", "ignore")
            body = body.encode().decode("unicode_escape", "ignore")
            body = re.sub(r"\\'", "'", body)
            if len(body) < 40 or len(body) > 900:
                continue
            if headline.startswith("Transfer Centre LIVE!"):
                continue
            if not any(k in body.lower() for k in KEYWORDS):
                continue
            matched = _match_players(headline + " " + body, NAME_INDEX, players_df)
            if not matched:
                continue
            items.append({"headline": headline, "body": body.strip(), "time": tmap.get(headline, ""),
                          "players": matched[:4]})
        _cache.update(t=time.time(), items=items[:12])
    except Exception as e:
        _cache["items"] = _cache.get("items") or []
        _cache["error"] = str(e)
    return _cache["items"]

# module-level index built by app.py
NAME_INDEX = {}
