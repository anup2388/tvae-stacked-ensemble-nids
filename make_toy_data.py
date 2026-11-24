"""Generate a tiny CSE-CIC-IDS2018-shaped dataset for smoke-testing nids_tvae_stack.py."""
import sys
import numpy as np
import pandas as pd

out = sys.argv[1]
rng = np.random.default_rng(0)
FEATS = ["Dst Port", "Protocol", "Flow Duration", "Tot Fwd Pkts", "Tot Bwd Pkts",
         "TotLen Fwd Pkts", "TotLen Bwd Pkts", "Fwd Pkt Len Max", "Bwd Pkt Len Max",
         "Flow Byts/s", "Flow Pkts/s", "Flow IAT Mean", "Flow IAT Max", "Flow IAT Min",
         "Fwd IAT Tot", "Bwd IAT Tot", "Pkt Len Var", "Init Fwd Win Byts",
         "Init Bwd Win Byts", "Fwd Seg Size Min", "Const Zero"]
CLASSES = {"Benign": 12000, "Bot": 900, "DoS attacks-Hulk": 2500, "Infilteration": 700,
           "SQL Injection": 60, "FTP-BruteForce": 400}
n_cls = len(CLASSES)
centers = rng.normal(size=(n_cls, len(FEATS))) * 1.5
rows = []
for k, (name, n) in enumerate(CLASSES.items()):
    X = rng.normal(size=(n, len(FEATS))) + centers[k]
    X[:, FEATS.index("Dst Port")] = rng.choice([80, 443, 22, 21, 8080][k % 5], size=n) \
        + rng.integers(0, 3, size=n)
    X[:, FEATS.index("Tot Fwd Pkts")] = X[:, FEATS.index("TotLen Fwd Pkts")] * 0.98 \
        + rng.normal(scale=0.05, size=n)                       # redundant pair
    X[:, FEATS.index("Const Zero")] = 0
    d = pd.DataFrame(X, columns=FEATS)
    d["Label"] = name
    start = pd.Timestamp("2018-02-14") + pd.Timedelta(days=int(k))
    d["Timestamp"] = (start + pd.to_timedelta(np.sort(rng.integers(0, 80000, n)), unit="s")
                      ).strftime("%d/%m/%Y %H:%M:%S")
    rows.append(d)
df = pd.concat(rows, ignore_index=True).sample(frac=1, random_state=1).reset_index(drop=True)
df = pd.concat([df, df.iloc[:300]], ignore_index=True)          # exact duplicates
df.loc[5:15, "Flow Byts/s"] = np.inf                            # infinities
df.loc[20:30, "Flow Pkts/s"] = np.nan                           # missing
cols = ["Dst Port", "Protocol", "Timestamp"] + [f for f in FEATS if f not in ("Dst Port", "Protocol")] + ["Label"]
df = df[cols]
parts = np.array_split(df, 3)
import os
os.makedirs(out, exist_ok=True)
for i, p in enumerate(parts):
    p = p.copy()
    if i == 1:    # the one daily file that carries identifier columns
        p.insert(0, "Flow ID", "1.1.1.1-2.2.2.2-1-2-6")
        p.insert(1, "Src IP", "1.1.1.1")
        p.insert(2, "Src Port", 12345)
        p.insert(3, "Dst IP", "2.2.2.2")
    if i == 2:    # repeated header row inside the data
        hdr = pd.DataFrame([list(p.columns)], columns=p.columns)
        p = pd.concat([p.iloc[:50], hdr, p.iloc[50:]], ignore_index=True)
    p.to_csv(f"{out}/day{i}.csv", index=False)
print("toy data written:", len(df), "rows")
