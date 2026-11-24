#!/usr/bin/env python3
"""
TVAE-augmented stacked-ensemble NIDS  --  complete, resumable implementation
============================================================================
"An Imbalance-Resilient Network Intrusion Detection Framework Using
 TVAE-Based Data Augmentation and Stacked Ensemble Learning"

One script, start to finish (paper Algorithm 2):
  Stage 1  cleaning        raw CSE-CIC-IDS2018 CSVs -> finite, de-duplicated flows;
                           flow ID / source IP / destination IP / source port /
                           timestamp removed.  DESTINATION PORT IS RETAINED.
  Stage 2  subsample+split benign subsampling N_b' = r * N_b (Eq. 1, r = 0.1),
                           then a stratified 60:40 train/test split
  Stage 3  feature select  Information Gain top-45 -> Pearson |rho| > 0.85 (Alg. 1)
  Stage 4  TVAE            one TVAE per minority attack class, raised to the
                           benign count of the training partition (Eq. 9-12)
  Stage 5  stacking        RF + LightGBM + Extra Trees, 5-fold out-of-fold
                           probabilities, XGBoost meta-classifier (Eq. 18)
  Stage 6  evaluation      binary + multi-class reports (Tables 3-9, 11),
                           confusion matrices, ROC, ablation, class counts
  Stage 7  diagnostics     TVAE train/validation loss curves,
                           LightGBM / XGBoost per-iteration curves, learning curve
  Stage 8  CV (optional)   repeated stratified CV + corrected paired t-test (Table 10)

Every result is checkpointed to --out-dir the moment it exists, with atomic writes
(temp file + rename).  Re-running the same command resumes where it stopped; use
--force-rerun to recompute.  Checkpoints, in the order they are produced:

  config.json, environment.json, input_files.json      what produced this run
  stage1_cleaned.parquet, stage1_classes.json           Stage 1
  stage2_split.npz, class_distribution.csv              Stage 2   (Figs. 2, 3, 5, 6)
  ig_ranking.csv, fig4_ig_top30.png                     Stage 3   (Table 1, Fig. 4)
  selected_features.json, feature_selection_dropped.csv Stage 3   (Table 2)
  syn_<tag>_<class>.parquet, tvae_loss_<tag>_<class>.csv   Stage 4  (per class)
  oof_<tag>_f<k>_<base>.npy, oof_Z_<tag>.npy            Stage 5   (per fold x base)
  proba_<tag>_<base>.npy, pred_<tag>.npz                Stage 5
  report_*.csv, cm_*.png/.csv, roc_binary.png,
  holdout_results.csv, table4/9/11_*.csv                Stage 6
  diag_*.csv / diag_*.png                               Stage 7
  cv_s<seed>_f<fold>.csv, table10_cv.csv, table10_tests.csv   Stage 8
  timing.json                                           wall-clock per stage / TVAE class

Supplementary analyses (each writes to its own
--out-dir, so the main results are never touched):

  main run                  python nids_tvae_stack.py --data-dir D --out-dir runs/main
  Dst Port ablation         ... --out-dir runs/no_dstport --exclude-dst-port
  chronological split       ... --out-dir runs/chrono --split chronological

Input: either a folder with the ten daily CSE-CIC-IDS2018 "Processed Traffic Data for
ML Algorithms" CSVs (--data-dir) or the pre-cleaned parquet files of the earlier
scripts (--audit-dir: X_audit_train/test.parquet, y_audit_train/test.parquet,
audit_label_encoder.joblib).  Timestamps (chronological split) need --data-dir.

Seeds: 42 everywhere (paper Section 3.H); CV uses seeds 0-4 (paper Section 4.D).
"""

import argparse
import contextlib
import glob
import hashlib
import json
import os
import platform
import sys
import time
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix,
                             f1_score, matthews_corrcoef,
                             precision_recall_fscore_support, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import StratifiedKFold, train_test_split

warnings.filterwarnings("ignore", category=FutureWarning)

# ===========================================================================
# Settings that define the paper's experiment
# ===========================================================================
BENIGN_LABEL = "Benign"
INFIL_LABEL = "Infilteration"          # spelling used by the dataset

IDENTIFIER_COLUMNS = ["Flow ID", "Src IP", "Src Port", "Dst IP"]   # dropped, Stage 1
# (these four exist in only one of the ten daily files; dropping them makes the
#  files consistent.  Timestamp is held aside for the chronological split only.)
DEDUPLICATE = True
SUBSAMPLE_R = 0.10                     # Eq. 1
TEST_SIZE = 0.40                       # stratified 60:40

IG_BINS = 64                           # equal-frequency bins for H(Y|X) (Eq. 4)
IG_TAU = 0.0                           # IG > tau
IG_TOP_K = 45                          # rank <= K
PEARSON_DELTA = 0.85                   # |rho| > delta -> drop the lower-IG feature

TVAE_EPOCHS = 100
TVAE_EMBED_DIM = 128
TVAE_BATCH = 500
# SDV/ctgan expose no learning rate: Adam runs at its default, 1e-3, l2 1e-5.

BASE_NAMES = ("RF", "LGBM", "ET")
BASE_PARAMS = {"RF":   {"n_estimators": 100},
               "LGBM": {"n_estimators": 100, "learning_rate": 0.1,
                        "num_leaves": 31, "verbose": -1},
               "ET":   {"n_estimators": 100}}
META_PARAMS = {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.3}
STACK_FOLDS = 5

CV_FOLDS = 5
CV_SEEDS = (0, 1, 2, 3, 4)
# (feature selection, TVAE) -> (models, tasks)   "STACK" also reports RF/LGBM/ET
HOLDOUT_GROUPS = {
    (False, False): (("STACK",), ("multiclass",)),            # Tables 9, 11
    (True,  False): (("STACK",), ("multiclass",)),            # Tables 9, 11
    (False, True):  (("RF",),    ("multiclass",)),            # Table 11
    (True,  True):  (("STACK",), ("multiclass", "binary")),   # Tables 3-11
}
CV_GROUPS = {(True, True): (("STACK",), ("multiclass",))}     # Table 10
SETUP_NAMES = {(False, False): "Original", (True, False): "FS only",
               (False, True): "TVAE only", (True, True): "FS + TVAE"}

cfg = None          # argparse namespace, set in main()


# ===========================================================================
# Checkpoint helpers (atomic writes, resumable stages)
# ===========================================================================
def out(name):
    return os.path.join(cfg.out_dir, name)


def have(path):
    """True if a checkpoint exists and a rerun was not forced."""
    return os.path.exists(path) and not cfg.force_rerun


def _atomic(path, writer):
    tmp = path + ".tmp"
    writer(tmp)
    os.replace(tmp, path)


def save_json(obj, path):
    _atomic(path, lambda p: json.dump(obj, open(p, "w"), indent=2, default=str))


def save_csv(df, path, **kw):
    _atomic(path, lambda p: df.to_csv(p, **kw))


def save_parquet(df, path):
    _atomic(path, lambda p: df.to_parquet(p))


def save_npy(arr, path):
    def w(p):
        with open(p, "wb") as f:
            np.save(f, arr)
    _atomic(path, w)


def save_npz(path, **arrs):
    def w(p):
        with open(p, "wb") as f:
            np.savez_compressed(f, **arrs)
    _atomic(path, w)


def save_fig(fig, path, dpi=300):
    def w(p):
        fig.savefig(p, dpi=dpi, format="png")
    _atomic(path, w)
    plt.close(fig)


def safe(name):
    return "".join(ch if ch.isalnum() else "_" for ch in str(name))


_timing = {}


@contextlib.contextmanager
def timed(label):
    t0 = time.time()
    yield
    dt = time.time() - t0
    _timing[label] = _timing.get(label, 0.0) + dt
    save_json({k: round(v, 1) for k, v in _timing.items()}, out("timing.json"))
    print(f"[time] {label}: {dt / 60:.1f} min")


def write_run_records(args):
    import sklearn, lightgbm, xgboost
    env = {"python": sys.version.split()[0], "platform": platform.platform(),
           "numpy": np.__version__, "pandas": pd.__version__,
           "scikit-learn": sklearn.__version__, "lightgbm": lightgbm.__version__,
           "xgboost": xgboost.__version__, "cpu_count": os.cpu_count()}
    for mod in ("sdv", "ctgan", "torch"):
        try:
            env[mod] = __import__(mod).__version__
        except ImportError:
            env[mod] = "not installed"
    try:
        import torch
        env["cuda_available"] = bool(torch.cuda.is_available())
    except ImportError:
        pass
    save_json(env, out("environment.json"))
    paper = (args.tvae_epochs == TVAE_EPOCHS and args.base_trees == 100
             and args.meta_trees == 100 and args.seed == 42)
    conf = {**vars(args), "paper_settings": paper,
            "constants": {"SUBSAMPLE_R": SUBSAMPLE_R, "TEST_SIZE": TEST_SIZE,
                          "IG_BINS": IG_BINS, "IG_TOP_K": IG_TOP_K,
                          "PEARSON_DELTA": PEARSON_DELTA,
                          "TVAE_EMBED_DIM": TVAE_EMBED_DIM, "TVAE_BATCH": TVAE_BATCH,
                          "BASE_PARAMS": BASE_PARAMS, "META_PARAMS": META_PARAMS,
                          "STACK_FOLDS": STACK_FOLDS}}
    save_json(conf, out("config.json"))
    if not paper:
        print("WARNING: epochs / trees / seed differ from the paper's settings; "
              "results are NOT comparable to the paper.")
    src = args.data_dir or args.audit_dir
    files = sorted(glob.glob(os.path.join(src, "*.csv")) +
                   glob.glob(os.path.join(src, "*.parquet")))
    info = {}
    for f in files:
        rec = {"bytes": os.path.getsize(f)}
        if args.checksum:
            h = hashlib.md5()
            with open(f, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 24), b""):
                    h.update(chunk)
            rec["md5"] = h.hexdigest()
        info[os.path.basename(f)] = rec
    save_json(info, out("input_files.json"))
    print("Environment:", env)


# ===========================================================================
# Stage 1: cleaning
# ===========================================================================
def _parse_timestamp(s):
    t = pd.to_datetime(s, format="%d/%m/%Y %H:%M:%S", errors="coerce")
    miss = t.isna()
    if miss.any():
        t[miss] = pd.to_datetime(s[miss], format="%d-%m-%Y %H:%M:%S", errors="coerce")
    return t


def load_raw_csvs(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(f"no CSV files in {data_dir}")
    Xs, ys, tss, ref_cols = [], [], [], None
    for f in files:
        df = pd.read_csv(f, low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        df = df[df["Label"].astype(str).str.strip() != "Label"]    # repeated headers
        label = df["Label"].astype(str).str.strip().to_numpy()
        ts = (_parse_timestamp(df["Timestamp"]).astype("int64").to_numpy()
              if "Timestamp" in df.columns else np.full(len(df), -1, np.int64))
        drop = [c for c in IDENTIFIER_COLUMNS + ["Timestamp", "Label"] if c in df.columns]
        X = df.drop(columns=drop).apply(pd.to_numeric, errors="coerce").astype(np.float32)
        if ref_cols is None:
            ref_cols = list(X.columns)
        elif list(X.columns) != ref_cols:
            raise ValueError(f"{os.path.basename(f)}: column set differs from the first "
                             f"file: {set(X.columns) ^ set(ref_cols)}")
        print(f"  {os.path.basename(f)}: {len(X):,} rows")
        Xs.append(X), ys.append(label), tss.append(ts)
    X = pd.concat(Xs, ignore_index=True)
    return X, np.concatenate(ys), np.concatenate(tss)


def load_audit(audit_dir):
    X = pd.concat([pd.read_parquet(os.path.join(audit_dir, "X_audit_train.parquet")),
                   pd.read_parquet(os.path.join(audit_dir, "X_audit_test.parquet"))],
                  ignore_index=True)
    codes = np.concatenate([pd.read_parquet(os.path.join(audit_dir, "y_audit_train.parquet"))["y"].values,
                            pd.read_parquet(os.path.join(audit_dir, "y_audit_test.parquet"))["y"].values])
    le = joblib.load(os.path.join(audit_dir, "audit_label_encoder.joblib"))
    labels = np.asarray(le.classes_)[codes.astype(np.int64)]
    return X.astype(np.float32), labels, np.full(len(X), -1, np.int64)


def stage1_clean():
    """Algorithm 2, Step 1.  Returns X (float32 DataFrame), y (int codes), classes,
    ts (int64 ns, -1 if unavailable)."""
    p_data, p_cls = out("stage1_cleaned.parquet"), out("stage1_classes.json")
    if have(p_data) and have(p_cls):
        df = pd.read_parquet(p_data)
        classes = json.load(open(p_cls))["classes"]
        print(f"Stage 1 loaded from checkpoint: {len(df):,} flows, "
              f"{df.shape[1] - 2} features")
        return (df.drop(columns=["__y", "__ts"]), df["__y"].to_numpy(np.int64),
                classes, df["__ts"].to_numpy(np.int64))

    print("Stage 1: loading and cleaning ...")
    X, labels, ts = (load_raw_csvs(cfg.data_dir) if cfg.data_dir
                     else load_audit(cfg.audit_dir))
    n0 = len(X)
    X = X.replace([np.inf, -np.inf], np.nan)
    ok = X.notna().all(axis=1).to_numpy()
    X, labels, ts = X[ok].reset_index(drop=True), labels[ok], ts[ok]
    n_valid = len(X)

    classes = sorted(pd.unique(labels).tolist())
    y = np.searchsorted(classes, labels).astype(np.int64)

    if DEDUPLICATE:   # identical feature vector AND label (row hash, 64-bit)
        h = pd.util.hash_pandas_object(X, index=False).to_numpy(np.uint64)
        key = h * np.uint64(1000003) + y.astype(np.uint64)
        dup = pd.Series(key).duplicated().to_numpy()
        X, y, ts = X[~dup].reset_index(drop=True), y[~dup], ts[~dup]
    if cfg.exclude_dst_port:
        X = X.drop(columns=["Dst Port"])
    print(f"Stage 1: {n0:,} flows -> {n_valid:,} finite -> {len(y):,} unique; "
          f"{X.shape[1]} features; Dst Port "
          f"{'EXCLUDED (ablation)' if cfg.exclude_dst_port else 'retained'}")
    counts = pd.Series(np.bincount(y, minlength=len(classes)), index=classes)
    print(counts.to_string())

    df = X.copy()
    df["__y"], df["__ts"] = y, ts
    save_parquet(df, p_data)
    save_json({"classes": classes, "n_raw": int(n0), "n_finite": int(n_valid),
               "n_unique": int(len(y)), "features": list(X.columns),
               "counts": {k: int(v) for k, v in counts.items()}}, p_cls)
    return X, y, classes, ts


# ===========================================================================
# Stage 2: benign subsampling and split
# ===========================================================================
class Ctx:
    """Everything the later stages need."""
    pass


def stage2_split(X, y, classes, ts):
    benign = classes.index(BENIGN_LABEL)
    p_split = out("stage2_split.npz")
    if have(p_split):
        z = np.load(p_split)
        tr, te = z["tr"], z["te"]
        print(f"Stage 2 loaded from checkpoint: train {len(tr):,}, test {len(te):,}")
    else:
        rng = np.random.default_rng(cfg.seed)
        idx = np.arange(len(y))
        ben = idx[y == benign]                                   # Eq. 1
        keep = rng.choice(ben, size=int(SUBSAMPLE_R * len(ben)), replace=False)
        idx = np.sort(np.concatenate([idx[y != benign], keep]))
        if cfg.split == "stratified":
            tr, te = train_test_split(idx, test_size=TEST_SIZE, stratify=y[idx],
                                      random_state=cfg.seed)
        else:   # per-class chronological: earliest 60% of each class train, latest 40% test
            if (ts[idx] < 0).any():
                bad = int((ts[idx] < 0).sum())
                print(f"  WARNING: {bad:,} flows without a valid timestamp excluded")
                idx = idx[ts[idx] >= 0]
            tr_l, te_l = [], []
            for c in range(len(classes)):
                ids = idx[y[idx] == c]
                if len(ids) == 0:
                    continue
                ids = ids[np.argsort(ts[ids], kind="stable")]
                n_te = min(max(int(round(TEST_SIZE * len(ids))), 1), max(len(ids) - 1, 0))
                tr_l.append(ids[:len(ids) - n_te]), te_l.append(ids[len(ids) - n_te:])
            tr, te = np.sort(np.concatenate(tr_l)), np.sort(np.concatenate(te_l))
        save_npz(p_split, tr=tr, te=te)

    cnt = lambda a: pd.Series(np.bincount(a, minlength=len(classes)), index=classes)
    dist = pd.DataFrame({"Cleaned": cnt(y), "Train": cnt(y[tr]), "Test": cnt(y[te])})
    save_csv(dist, out("class_distribution.csv"))
    print(f"Stage 2 ({cfg.split}): train {len(tr):,} (benign {dist.loc[BENIGN_LABEL, 'Train']:,}), "
          f"test {len(te):,} (benign {dist.loc[BENIGN_LABEL, 'Test']:,})")

    c = Ctx()
    c.classes, c.K = classes, len(classes)
    c.benign, c.infil = benign, classes.index(INFIL_LABEL)
    c.attack_codes = [k for k in range(c.K) if k != benign]
    c.Xtr, c.Xte = X.iloc[tr].reset_index(drop=True), X.iloc[te].reset_index(drop=True)
    c.ytr, c.yte = y[tr], y[te]
    c.dist = dist
    c.all_feats = list(X.columns)
    return c


# ===========================================================================
# Stage 3: two-stage feature selection (Algorithm 1)
# ===========================================================================
def information_gain(X, y, n_bins=IG_BINS):
    """IG(Y,X) = H(Y) - H(Y|X) in bits (Eq. 2-4), equal-frequency bins."""
    classes, y_idx = np.unique(y, return_inverse=True)
    C, n = len(classes), len(y)
    p_y = np.bincount(y_idx, minlength=C) / n
    h_y = -np.sum(p_y[p_y > 0] * np.log2(p_y[p_y > 0]))
    q = np.linspace(0, 1, n_bins + 1)[1:-1]
    ig = {}
    for col in X.columns:
        x = X[col].to_numpy(np.float64)
        edges = np.unique(np.quantile(x, q))
        b = np.searchsorted(edges, x, side="right")
        nb = len(edges) + 1
        joint = np.bincount(b * C + y_idx, minlength=nb * C).reshape(nb, C)
        n_x = joint.sum(1)
        with np.errstate(divide="ignore", invalid="ignore"):
            p_c_x = joint / n_x[:, None]
            h_c_x = -np.nansum(p_c_x * np.log2(p_c_x), axis=1)
        ig[col] = h_y - np.sum(n_x / n * h_c_x)
    return pd.Series(ig).sort_values(ascending=False)


def stage3_feature_selection(c):
    p_ig, p_sel = out("ig_ranking.csv"), out("selected_features.json")
    if have(p_ig) and have(p_sel):
        ig = pd.read_csv(p_ig, index_col=0).iloc[:, 0]
        sel = json.load(open(p_sel))
        print(f"Stage 3 loaded from checkpoint: {len(sel['selected'])} features")
        c.ig, c.feats_sel = ig, sel["selected"]
        return
    ig = information_gain(c.Xtr, c.ytr)
    cand = ig[ig > IG_TAU].index[:IG_TOP_K].tolist()                       # Stage I
    corr = np.nan_to_num(np.abs(np.corrcoef(c.Xtr[cand].to_numpy(np.float64),
                                            rowvar=False)), nan=0.0)
    pos = {f: i for i, f in enumerate(cand)}
    kept, dropped = [], []
    for f in cand:                                                          # Stage II
        hit = next((k for k in kept if corr[pos[f], pos[k]] > PEARSON_DELTA), None)
        if hit is None:
            kept.append(f)
        else:
            dropped.append({"feature": f, "correlated_with": hit,
                            "abs_rho": round(float(corr[pos[f], pos[hit]]), 3)})
    print(f"Stage 3: {c.Xtr.shape[1]} features -> IG top-{IG_TOP_K}: {len(cand)} "
          f"-> Pearson {PEARSON_DELTA}: {len(kept)}")
    print("  selected:", kept)
    save_csv(ig.rename("IG (bits)"), p_ig)
    save_csv(pd.DataFrame(dropped), out("feature_selection_dropped.csv"), index=False)
    ranks = pd.DataFrame({"rank": range(1, len(ig) + 1), "feature": ig.index,
                          "IG (bits)": ig.values,
                          "in_top_k": [f in cand for f in ig.index],
                          "selected": [f in kept for f in ig.index]})
    save_csv(ranks, out("table1_ig_ranking_full.csv"), index=False)
    plot_bars(ig.head(30), "Information Gain (bits)", out("fig4_ig_top30.png"))
    save_json({"ig_top_k": cand, "selected": kept, "n_input_features": c.Xtr.shape[1]},
              p_sel)
    c.ig, c.feats_sel = ig, kept


# ===========================================================================
# Stage 4: class-wise TVAE
# ===========================================================================
def build_metadata(columns):
    cols = {str(k): {"sdtype": "numerical"} for k in columns}
    try:
        from sdv.metadata import Metadata
        return Metadata.load_from_dict({"columns": cols}, single_table_name="flows")
    except Exception:
        from sdv.metadata import SingleTableMetadata
        return SingleTableMetadata.load_from_dict({"columns": cols})


def fit_tvae(X_c, seed):
    """TVAE_c = Train(X_c) (Eq. 9)."""
    from sdv.single_table import TVAESynthesizer
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass
    synth = TVAESynthesizer(build_metadata(X_c.columns), epochs=cfg.tvae_epochs,
                            batch_size=TVAE_BATCH, embedding_dim=TVAE_EMBED_DIM,
                            enforce_min_max_values=True, enforce_rounding=False)
    synth.fit(X_c.reset_index(drop=True))
    return synth


def tvae_augment(c, X, y, tag, seed, cache=True):
    """Each attack class with N_c < N_target gets N_target - N_c synthetic rows
    (Eq. 11); N_target is the benign count of this training partition.
    Returns D_real U D_synthetic (Eq. 12)."""
    feats = list(X.columns)
    n_target = int((y == c.benign).sum())
    X_parts, y_parts, done_classes = [X.reset_index(drop=True)], [np.asarray(y)], []
    for code in c.attack_codes:
        name = c.classes[code]
        n_c = int((y == code).sum())
        if n_c == 0 or n_c >= n_target:
            continue
        syn_path = out(f"syn_{tag}_{safe(name)}.parquet")
        if cache and have(syn_path):
            syn = pd.read_parquet(syn_path)
            print(f"[TVAE {tag}] {name}: {len(syn):,} synthetic rows loaded")
        else:
            t0 = time.time()
            synth = fit_tvae(X.loc[y == code], seed)
            np.random.seed(seed)
            syn = synth.sample(num_rows=n_target - n_c)[feats]               # Eq. 10
            syn = syn.replace([np.inf, -np.inf], np.nan).dropna()
            dt = time.time() - t0
            print(f"[TVAE {tag}] {name}: {n_c:,} real -> +{len(syn):,} synthetic "
                  f"({dt / 60:.1f} min)")
            if cache:
                try:   # training loss history, free by-product of the real run
                    save_csv(synth.get_loss_values(), out(f"tvae_loss_{tag}_{safe(name)}.csv"),
                             index=False)
                except Exception as e:
                    print(f"  (loss history unavailable: {e})")
                save_parquet(syn, syn_path)
                if cfg.save_models:
                    synth.save(out(f"tvae_{tag}_{safe(name)}.pkl"))
                _timing[f"tvae_{tag}_{name}"] = dt
                save_json({k: round(v, 1) for k, v in _timing.items()}, out("timing.json"))
        X_parts.append(syn.reset_index(drop=True))
        y_parts.append(np.full(len(syn), code, dtype=np.int64))
        done_classes.append(name)
    X_bal, y_bal = pd.concat(X_parts, ignore_index=True), np.concatenate(y_parts)
    print(f"[TVAE {tag}] N_target = {n_target:,}; {len(done_classes)} classes augmented; "
          f"training set {len(y):,} -> {len(y_bal):,}")
    return X_bal, y_bal


# ===========================================================================
# Stage 5: base learners, out-of-fold stacking, XGBoost meta-classifier
# ===========================================================================
def make_base(name, seed):
    from lightgbm import LGBMClassifier
    p = dict(BASE_PARAMS[name])
    p["n_estimators"] = cfg.base_trees
    if name == "RF":
        return RandomForestClassifier(random_state=seed, n_jobs=-1, **p)
    if name == "ET":
        return ExtraTreesClassifier(random_state=seed, n_jobs=-1, **p)
    if name == "LGBM":
        return LGBMClassifier(random_state=seed, n_jobs=-1, **p)
    raise ValueError(name)


def make_meta(seed, **extra):
    from xgboost import XGBClassifier
    p = dict(META_PARAMS)
    p["n_estimators"] = cfg.meta_trees
    return XGBClassifier(random_state=seed, n_jobs=-1, tree_method="hist", **p, **extra)


def proba_full(model, X, n):
    """predict_proba aligned to all n encoded classes."""
    p = np.zeros((X.shape[0], n), dtype=np.float32)
    p[:, model.classes_] = model.predict_proba(X)
    return p


def oof_matrix(X, y_enc, n, seed, tag):
    """Out-of-fold base probabilities Z (Step 6); each fold x base is checkpointed."""
    p_z = out(f"oof_Z_{tag}.npy")
    if have(p_z):
        return np.load(p_z)
    Z = np.zeros((len(y_enc), len(BASE_NAMES) * n), np.float32)
    skf = StratifiedKFold(STACK_FOLDS, shuffle=True, random_state=seed)
    for k, (tr, va) in enumerate(skf.split(X, y_enc)):
        for j, name in enumerate(BASE_NAMES):
            p = out(f"oof_{tag}_f{k}_{name}.npy")
            if have(p):
                part = np.load(p)
            else:
                t0 = time.time()
                part = proba_full(make_base(name, seed).fit(X[tr], y_enc[tr]), X[va], n)
                save_npy(part, p)
                print(f"[{tag}] OOF fold {k + 1}/{STACK_FOLDS} {name} "
                      f"({(time.time() - t0) / 60:.1f} min)")
            Z[va, j * n:(j + 1) * n] = part
    save_npy(Z, p_z)
    for k in range(STACK_FOLDS):                       # fold files are now redundant
        for name in BASE_NAMES:
            with contextlib.suppress(FileNotFoundError):
                os.remove(out(f"oof_{tag}_f{k}_{name}.npy"))
    return Z


def fit_predict(X, y, X_eval, models, seed, tag):
    """Fits the requested group and returns {model: (pred_labels, proba)}.
    STACK = RF + LGBM + ET -> XGBoost meta (Steps 5-7)."""
    path = out(f"pred_{tag}.npz")
    if have(path):
        z = np.load(path)
        print(f"[{tag}] predictions loaded from checkpoint")
        return {k[:-5]: (z[k], z[k[:-5] + "_proba"]) for k in z.files if k.endswith("_pred")}
    classes = np.unique(y)
    y_enc, n = np.searchsorted(classes, y), len(classes)
    names = list(BASE_NAMES) if "STACK" in models else list(models)
    P = {}
    for name in names:                                   # Steps 5 and 7 (full fit)
        p = out(f"proba_{tag}_{name}.npy")
        if have(p):
            P[name] = np.load(p)
            continue
        t0 = time.time()
        m = make_base(name, seed).fit(X, y_enc)
        P[name] = proba_full(m, X_eval, n)
        save_npy(P[name], p)
        print(f"[{tag}] {name} fitted on {X.shape} ({(time.time() - t0) / 60:.1f} min)")
        if cfg.save_models:
            joblib.dump(m, out(f"model_{tag}_{name}.joblib"))
    if "STACK" in models:                                # Step 6
        t0 = time.time()
        Z = oof_matrix(X, y_enc, n, seed, tag)
        meta = make_meta(seed).fit(Z, y_enc)
        P["STACK"] = meta.predict_proba(
            np.hstack([P[b] for b in BASE_NAMES])).astype(np.float32)   # Eq. 18
        print(f"[{tag}] meta-classifier fitted ({(time.time() - t0) / 60:.1f} min)")
        if cfg.save_models:
            joblib.dump({"meta": meta, "classes": classes}, out(f"model_{tag}_meta.joblib"))
    res = {nm: (classes[p.argmax(1)], p) for nm, p in P.items()}
    save_npz(path, **{f"{nm}_pred": r[0] for nm, r in res.items()},
             **{f"{nm}_proba": r[1] for nm, r in res.items()})
    return res


# ===========================================================================
# Stage 6: evaluation
# ===========================================================================
def evaluate(c, y, pred, proba):
    """Eq. 24-27 plus balanced accuracy and MCC (weighted recall == accuracy)."""
    m = {"accuracy": accuracy_score(y, pred)}
    for avg in ("macro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(y, pred, average=avg, zero_division=0)
        m.update({f"{avg}_precision": p, f"{avg}_recall": r, f"{avg}_f1": f})
    m["balanced_accuracy"] = balanced_accuracy_score(y, pred)
    m["mcc"] = matthews_corrcoef(y, pred)
    try:
        m["auc"] = (roc_auc_score(y, proba[:, 1]) if proba.shape[1] == 2 else
                    roc_auc_score(y, proba, multi_class="ovr", average="macro"))
    except ValueError:
        m["auc"] = np.nan
    if proba.shape[1] > 2:
        m["infil_f1"] = precision_recall_fscore_support(
            y, pred, labels=[c.infil], average=None, zero_division=0)[2][0]
    return m


def per_class_table(y, pred, names):
    rep = classification_report(y, pred, labels=np.arange(len(names)), target_names=names,
                                output_dict=True, zero_division=0)
    rep.pop("accuracy", None)
    df = pd.DataFrame(rep).T
    df[["precision", "recall", "f1-score"]] *= 100
    df.loc["Overall accuracy", "precision"] = accuracy_score(y, pred) * 100
    return df.round(2)


def plot_confusion(y, pred, names, title, path):
    cm = confusion_matrix(y, pred, labels=np.arange(len(names)))
    save_csv(pd.DataFrame(cm, index=names, columns=names), path.replace(".png", ".csv"))
    big = len(names) > 2
    shown = cm / np.clip(cm.sum(1, keepdims=True), 1, None) * 100 if big else cm
    fig, ax = plt.subplots(figsize=(11, 9) if big else (4.5, 4))
    ax.imshow(shown, cmap="Blues")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90 if big else 0)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{shown[i, j]:.1f}" if big else f"{cm[i, j]:,}",
                    ha="center", va="center", fontsize=7 if big else 10,
                    color="white" if shown[i, j] > shown.max() / 2 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title + (" (row-normalised %)" if big else ""))
    fig.tight_layout()
    save_fig(fig, path)


def plot_roc(y, probas, path):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, p in probas.items():
        fpr, tpr, _ = roc_curve(y, p[:, 1])
        ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc_score(y, p[:, 1]):.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right")
    fig.tight_layout()
    save_fig(fig, path)


def plot_bars(series, xlabel, path):
    fig, ax = plt.subplots(figsize=(7, max(3, 0.28 * len(series))))
    ax.barh(series.index[::-1], series.values[::-1])
    ax.set_xlabel(xlabel)
    fig.tight_layout()
    save_fig(fig, path)


def print_paper_tables(res, reports):
    full, order = "FS + TVAE", ["RF", "LGBM", "ET", "STACK"]
    mc, bn = res[res.Task == "multiclass"], res[res.Task == "binary"]
    if ("binary", full, "STACK") in reports:
        print("\n===== TABLE 3: binary, stacked ensemble =====")
        print(reports[("binary", full, "STACK")].to_string())
        t4 = bn[bn.Setup == full].set_index("Model").loc[order, ["accuracy", "auc"]]
        print("\n===== TABLE 4: binary accuracy / AUC =====")
        print(t4.round(4).to_string())
        save_csv(t4, out("table4_binary.csv"))
    if ("multiclass", full, "STACK") in reports:
        print("\n===== TABLE 8: multi-class, stacked ensemble "
              "(Tables 5-7: report_FS___TVAE_multiclass_<model>.csv) =====")
        print(reports[("multiclass", full, "STACK")].to_string())
    cols = ["accuracy", "weighted_precision", "weighted_recall", "weighted_f1",
            "macro_f1", "infil_f1"]
    t9 = mc[mc.Setup.isin(["Original", "FS only", full])].set_index(["Model", "Setup"])
    t9 = t9.reindex([(m, s) for m in order for s in ["Original", "FS only", full]
                     if (m, s) in t9.index])[cols]
    print("\n===== TABLE 9: multi-class, setups =====")
    print(t9.round(4).to_string())
    save_csv(t9, out("table9_setups.csv"))
    arms = [("RF", "Original"), ("RF", "FS only"), ("RF", "TVAE only"), ("RF", full),
            ("STACK", "Original"), ("STACK", full)]
    t11 = mc.set_index(["Model", "Setup"])
    t11 = t11.reindex([a for a in arms if a in t11.index])[cols]
    print("\n===== TABLE 11 (hold-out): ablation =====")
    print(t11.round(4).to_string())
    save_csv(t11, out("table11_ablation_holdout.csv"))
    acc = mc[mc.Setup == full].set_index("Model")["accuracy"].reindex(order).dropna()
    plot_bars(acc * 100, "Multi-class accuracy (%)", out("fig10_accuracy.png"))


def stage6_holdout(c):
    p_res = out("holdout_results.csv")
    names_mc, names_bin = list(c.classes), ["Benign (0)", "Malicious (1)"]
    rows, reports = [], {}
    c.fit_sets = {}                                      # kept for diagnostics
    for (fs, aug), (models, tasks) in HOLDOUT_GROUPS.items():
        setup = SETUP_NAMES[(fs, aug)]
        feats = c.feats_sel if fs else c.all_feats
        print(f"\n===== HOLD-OUT: {setup} ({len(feats)} features) =====")
        if aug:
            X_fit, y_fit = tvae_augment(c, c.Xtr[feats], c.ytr,
                                        f"holdout_{'fs' if fs else 'all'}", cfg.seed)
            if fs:
                c.dist["Train + TVAE"] = pd.Series(
                    np.bincount(y_fit, minlength=c.K), index=c.classes)
                save_csv(c.dist, out("class_distribution.csv"))
        else:
            X_fit, y_fit = c.Xtr[feats], c.ytr
        X_fit, X_eval = X_fit.to_numpy(np.float32), c.Xte[feats].to_numpy(np.float32)
        if fs and aug:
            c.fit_sets["fs_tvae"] = (X_fit, y_fit)
        for task in tasks:
            binary = task == "binary"
            if binary:
                yf, ye, names = ((y_fit != c.benign).astype(np.int64),
                                 (c.yte != c.benign).astype(np.int64), names_bin)
            else:
                yf, ye, names = y_fit, c.yte, names_mc
            tag = f"{safe(setup)}_{task}"
            with timed(f"fit_predict_{tag}"):
                preds = fit_predict(X_fit, yf, X_eval, models, cfg.seed, tag)
            if fs and aug and not binary:
                c.mc_preds = preds
            for model, (pred, proba) in preds.items():
                rows.append({"Task": task, "Setup": setup, "Model": model,
                             **evaluate(c, ye, pred, proba)})
                rep = per_class_table(ye, pred, names)
                save_csv(rep, out(f"report_{tag}_{model}.csv"))
                reports[(task, setup, model)] = rep
                if fs and aug:
                    plot_confusion(ye, pred, names, f"{model}, {task}",
                                   out(f"cm_{tag}_{model}.png"))
            if fs and aug and binary:
                plot_roc(ye, {m: p for m, (_, p) in preds.items()}, out("roc_binary.png"))
        save_csv(pd.DataFrame(rows), p_res, index=False)           # checkpoint per setup
    res = pd.DataFrame(rows)
    print("\n===== CLASS DISTRIBUTION (Figs. 2, 3, 5, 6) =====")
    print(c.dist.to_string())
    print_paper_tables(res, reports)
    return res


# ===========================================================================
# Stage 7: diagnostics (training behaviour)
# ===========================================================================
def _stratified_rows(y, n, seed):
    """Indices of a stratified random subsample of size ~n (all rows if len(y) <= n)."""
    if len(y) <= n:
        return np.arange(len(y))
    if len(y) - n < len(np.unique(y)):          # too few rows left over to stratify
        return np.arange(len(y))
    idx, _ = train_test_split(np.arange(len(y)), train_size=n, stratify=y,
                              random_state=seed)
    return np.sort(idx)


def _make_diag_tvae():
    """TVAE with the SDV/ctgan training loop plus a per-epoch validation loss.
    Same hyper-parameters as the production models; used ONLY for diagnostics."""
    import torch
    from torch.optim import Adam
    from torch.utils.data import DataLoader, TensorDataset
    from ctgan.data_transformer import DataTransformer
    from ctgan.synthesizers.tvae import TVAE, Decoder, Encoder, _loss_function

    class DiagTVAE(TVAE):
        def fit_with_validation(self, train_df, val_df, seed):
            torch.manual_seed(seed)
            self.transformer = DataTransformer()
            self.transformer.fit(train_df, ())
            tr = self.transformer.transform(train_df).astype("float32")
            va = torch.from_numpy(self.transformer.transform(val_df).astype("float32")
                                  ).to(self._device)
            loader = DataLoader(TensorDataset(torch.from_numpy(tr).to(self._device)),
                                batch_size=self.batch_size, shuffle=True, drop_last=False)
            dim = self.transformer.output_dimensions
            enc = Encoder(dim, self.compress_dims, self.embedding_dim).to(self._device)
            self.decoder = Decoder(self.embedding_dim, self.decompress_dims, dim).to(self._device)
            opt = Adam(list(enc.parameters()) + list(self.decoder.parameters()),
                       weight_decay=self.l2scale)
            hist = []
            for ep in range(self.epochs):
                tr_losses = []
                for (real,) in loader:
                    opt.zero_grad()
                    mu, std, logvar = enc(real)
                    emb = torch.randn_like(std) * std + mu
                    rec, sigmas = self.decoder(emb)
                    l1, l2 = _loss_function(rec, real, sigmas, mu, logvar,
                                            self.transformer.output_info_list, self.loss_factor)
                    (l1 + l2).backward()
                    opt.step()
                    self.decoder.sigma.data.clamp_(0.01, 1.0)
                    tr_losses.append((l1 + l2).detach().cpu().item())
                with torch.no_grad():                 # validation: same loss, no update
                    tot, cnt = 0.0, 0
                    for s in range(0, len(va), self.batch_size):
                        xb = va[s:s + self.batch_size]
                        mu, std, logvar = enc(xb)
                        rec, sigmas = self.decoder(torch.randn_like(std) * std + mu)
                        l1, l2 = _loss_function(rec, xb, sigmas, mu, logvar,
                                                self.transformer.output_info_list,
                                                self.loss_factor)
                        tot += (l1 + l2).item() * len(xb)
                        cnt += len(xb)
                hist.append({"epoch": ep + 1, "train_loss": float(np.mean(tr_losses)),
                             "val_loss": tot / cnt})
            return pd.DataFrame(hist)

    return DiagTVAE


def diag_tvae(c):
    """TVAE loss curves.  (a) training loss per epoch of the production models,
    (b) train vs. validation loss per epoch on a 90/10 split of each class's real
    training rows (diagnostic models, same hyper-parameters)."""
    # (a) production training-loss history
    frames = []
    for code in c.attack_codes:
        p = out(f"tvae_loss_holdout_fs_{safe(c.classes[code])}.csv")
        if os.path.exists(p):
            d = pd.read_csv(p)
            d = d.groupby("Epoch", as_index=False)["Loss"].mean()
            d["class"] = c.classes[code]
            frames.append(d)
    if frames:
        allc = pd.concat(frames)
        save_csv(allc, out("diag_tvae_train_loss_per_epoch.csv"), index=False)
        # the TVAE loss is negative after the first epochs, so the axis must be linear
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        for name, g in allc.groupby("class"):
            ax.plot(g["Epoch"] + 1, g["Loss"], label=name, lw=1.2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Training loss (mean over batches)")
        ax.legend(fontsize=6, ncol=2)
        fig.tight_layout()
        save_fig(fig, out("diag_tvae_train_loss.png"))
    else:
        print("  no production TVAE loss history found (classes were loaded from a "
              "cache created without it); rerun with --force-rerun to record it")

    # (b) train vs validation
    try:
        Diag = _make_diag_tvae()
    except Exception as e:
        print(f"  TVAE validation curves skipped (ctgan internals not as expected: {e})")
        return
    n_target = int((c.ytr == c.benign).sum())
    rows, fig_data = [], {}
    for code in c.attack_codes:
        name = c.classes[code]
        p = out(f"diag_tvae_trainval_{safe(name)}.csv")
        if have(p):
            hist = pd.read_csv(p)
        else:
            Xc = c.Xtr.loc[c.ytr == code, c.feats_sel]
            if len(Xc) >= n_target or len(Xc) < 20:
                continue                                  # not augmented / too few rows
            Xc = Xc.sample(n=min(len(Xc), cfg.diag_tvae_max_rows),
                           random_state=cfg.seed).reset_index(drop=True)
            val = Xc.sample(frac=0.10, random_state=cfg.seed)
            tr = Xc.drop(val.index)
            t0 = time.time()
            model = Diag(embedding_dim=TVAE_EMBED_DIM, batch_size=TVAE_BATCH,
                         epochs=cfg.tvae_epochs)
            hist = model.fit_with_validation(tr.reset_index(drop=True),
                                             val.reset_index(drop=True), cfg.seed)
            save_csv(hist, p, index=False)
            print(f"  diag TVAE {name}: {len(tr):,} train / {len(val):,} val rows "
                  f"({(time.time() - t0) / 60:.1f} min)")
        fig_data[name] = hist
        rows.append({"class": name, "epochs": len(hist),
                     "train_loss_first": hist.train_loss.iloc[0],
                     "train_loss_last": hist.train_loss.iloc[-1],
                     "val_loss_last": hist.val_loss.iloc[-1],
                     "val_min": hist.val_loss.min(),
                     "val_min_epoch": int(hist.val_loss.idxmin() + 1),
                     "gap_last": hist.val_loss.iloc[-1] - hist.train_loss.iloc[-1]})
    if rows:
        save_csv(pd.DataFrame(rows), out("diag_tvae_trainval_summary.csv"), index=False)
        k = len(fig_data)
        ncol = 4
        nrow = int(np.ceil(k / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.5 * nrow), squeeze=False)
        for ax, (name, h) in zip(axes.ravel(), fig_data.items()):
            ax.plot(h.epoch, h.train_loss, label="train")
            ax.plot(h.epoch, h.val_loss, label="validation")
            ax.set_title(name, fontsize=7)
            ax.tick_params(labelsize=6)
        for ax in axes.ravel()[k:]:
            ax.axis("off")
        axes.ravel()[0].legend(fontsize=6)
        fig.supxlabel("Epoch", fontsize=8)
        fig.supylabel("TVAE loss", fontsize=8)
        fig.tight_layout()
        save_fig(fig, out("diag_tvae_trainval.png"))


def diag_boosting(c, X_fit, y_fit):
    """Per-iteration train vs. validation log-loss for the LightGBM base learner and
    the XGBoost meta-classifier (multi-class, FS + TVAE setup).  Validation rows are
    a stratified sample of the hold-out partition: this is a diagnostic only -- no
    early stopping or model selection uses it, n_estimators stays fixed."""
    from lightgbm import LGBMClassifier
    feats = c.feats_sel
    Xte = c.Xte[feats].to_numpy(np.float32)
    v_idx = _stratified_rows(c.yte, cfg.diag_val_rows, cfg.seed)
    t_idx = _stratified_rows(y_fit, cfg.diag_val_rows, cfg.seed)
    classes = np.unique(y_fit)
    y_enc, yv = np.searchsorted(classes, y_fit), np.searchsorted(classes, c.yte[v_idx])
    n = len(classes)

    p_lgb = out("diag_lgbm_curve.csv")
    if not have(p_lgb):
        params = dict(BASE_PARAMS["LGBM"])
        params["n_estimators"] = cfg.base_trees
        m = LGBMClassifier(random_state=cfg.seed, n_jobs=-1, **params)
        m.fit(X_fit, y_enc, eval_set=[(X_fit[t_idx], y_enc[t_idx]), (Xte[v_idx], yv)],
              eval_names=["train", "validation"], eval_metric="multi_logloss")
        r = m.evals_result_
        save_csv(pd.DataFrame({"iteration": np.arange(1, len(r["train"]["multi_logloss"]) + 1),
                               "train_logloss": r["train"]["multi_logloss"],
                               "validation_logloss": r["validation"]["multi_logloss"]}),
                 p_lgb, index=False)
    lgb = pd.read_csv(p_lgb)

    p_xgb = out("diag_xgb_meta_curve.csv")
    tag = f"{safe(SETUP_NAMES[(True, True)])}_multiclass"
    z_path = out(f"oof_Z_{tag}.npy")
    if not have(p_xgb):
        if not os.path.exists(z_path) or not hasattr(c, "mc_preds"):
            print("  XGBoost meta curve skipped: OOF matrix / base probabilities missing")
        else:
            Z = np.load(z_path)
            z = np.load(out(f"pred_{tag}.npz"))
            Zv = np.hstack([z[f"{b}_proba"][v_idx] for b in BASE_NAMES])
            zt = _stratified_rows(y_enc, cfg.diag_val_rows, cfg.seed)
            meta = make_meta(cfg.seed, eval_metric="mlogloss")
            meta.fit(Z, y_enc, eval_set=[(Z[zt], y_enc[zt]), (Zv, yv)], verbose=False)
            r = meta.evals_result()
            save_csv(pd.DataFrame({"iteration": np.arange(1, len(r["validation_0"]["mlogloss"]) + 1),
                                   "train_logloss": r["validation_0"]["mlogloss"],
                                   "validation_logloss": r["validation_1"]["mlogloss"]}),
                     p_xgb, index=False)
    xgb = pd.read_csv(p_xgb) if os.path.exists(p_xgb) else None

    fig, axes = plt.subplots(1, 2 if xgb is not None else 1, figsize=(9, 3.6), squeeze=False)
    for ax, d, title in zip(axes.ravel(), [lgb, xgb], ["LightGBM base learner",
                                                        "XGBoost meta-classifier"]):
        if d is None:
            continue
        ax.plot(d.iteration, d.train_logloss, label="train")
        ax.plot(d.iteration, d.validation_logloss, label="validation")
        ax.set_xlabel("Boosting iteration")
        ax.set_ylabel("Multi-class log-loss")
        ax.set_title(title, fontsize=9)
        ax.legend()
    fig.tight_layout()
    save_fig(fig, out("diag_boosting_curves.png"))


def diag_learning_curve(c, X_fit, y_fit):
    """Stack macro-F1 on a train subsample vs. validation rows as the (augmented)
    training set grows.  Each fraction is a separate, checkpointed stack fit."""
    feats = c.feats_sel
    Xte = c.Xte[feats].to_numpy(np.float32)
    v_idx = _stratified_rows(c.yte, cfg.diag_val_rows, cfg.seed)
    rows = []
    for frac in cfg.lc_fracs:
        p = out(f"diag_lc_{frac}.json")
        if have(p):
            rows.append(json.load(open(p)))
            continue
        sub = _stratified_rows(y_fit, int(frac * len(y_fit)), cfg.seed)
        Xs, ys = X_fit[sub], y_fit[sub]
        tr_eval = _stratified_rows(ys, min(len(ys), 100_000), cfg.seed)
        Xe = np.vstack([Xte[v_idx], Xs[tr_eval]])
        preds = fit_predict(Xs, ys, Xe, ("STACK",), cfg.seed, f"lc_{frac}")
        pred = preds["STACK"][0]
        nv = len(v_idx)
        rec = {"fraction": frac, "n_train": int(len(ys)),
               "val_macro_f1": f1_score(c.yte[v_idx], pred[:nv], average="macro"),
               "train_macro_f1": f1_score(ys[tr_eval], pred[nv:], average="macro")}
        save_json(rec, p)
        rows.append(rec)
        print(f"  learning curve {frac}: {rec}")
    if rows:
        d = pd.DataFrame(rows).sort_values("n_train")
        save_csv(d, out("diag_learning_curve.csv"), index=False)
        fig, ax = plt.subplots(figsize=(5, 3.6))
        ax.plot(d.n_train, d.train_macro_f1, "o-", label="train")
        ax.plot(d.n_train, d.val_macro_f1, "o-", label="validation")
        ax.set_xscale("log")
        ax.set_xlabel("Training samples (after TVAE augmentation)")
        ax.set_ylabel("Macro F1")
        ax.legend()
        fig.tight_layout()
        save_fig(fig, out("diag_learning_curve.png"))


def stage7_diagnostics(c):
    if "fs_tvae" not in getattr(c, "fit_sets", {}):
        print("Stage 7 needs the FS + TVAE hold-out arm; skipped")
        return
    X_fit, y_fit = c.fit_sets["fs_tvae"]
    with timed("diag_tvae"):
        diag_tvae(c)
    with timed("diag_boosting"):
        diag_boosting(c, X_fit, y_fit)
    if cfg.lc_fracs:
        with timed("diag_learning_curve"):
            diag_learning_curve(c, X_fit, y_fit)


# ===========================================================================
# Stage 8: repeated stratified cross-validation (Table 10)
# ===========================================================================
def corrected_ttest(a, b, test_train_ratio):
    """Corrected resampled t-test (Nadeau & Bengio, 2003)."""
    d = np.asarray(a) - np.asarray(b)
    J, var = len(d), d.var(ddof=1)
    if var == 0:
        return np.nan, np.nan
    t = d.mean() / np.sqrt((1.0 / J + test_train_ratio) * var)
    return t, 2 * stats.t.sf(abs(t), df=J - 1)


def stage8_cv(c):
    X, y = c.Xtr, c.ytr
    frames = []
    need_fs = any(fs for fs, _ in CV_GROUPS)
    for seed in cfg.cv_seeds:
        skf = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=seed)
        for fold, (tr, va) in enumerate(skf.split(X, y)):
            path = out(f"cv_s{seed}_f{fold}.csv")
            if have(path):
                frames.append(pd.read_csv(path))
                continue
            tag = f"cv_s{seed}_f{fold}"
            print(f"\n===== CV seed {seed}, fold {fold + 1}/{CV_FOLDS} =====")
            X_trf, y_trf = X.iloc[tr].reset_index(drop=True), y[tr]
            X_vaf, y_vaf = X.iloc[va].reset_index(drop=True), y[va]
            if need_fs:    # feature selection on the training fold only (no leakage)
                ig = information_gain(X_trf, y_trf)
                cand = ig[ig > IG_TAU].index[:IG_TOP_K].tolist()
                corr = np.nan_to_num(np.abs(np.corrcoef(X_trf[cand].to_numpy(np.float64),
                                                        rowvar=False)), nan=0.0)
                pos, kept = {f: i for i, f in enumerate(cand)}, []
                for f in cand:
                    if next((k for k in kept if corr[pos[f], pos[k]] > PEARSON_DELTA), None) is None:
                        kept.append(f)
                feats_f = kept
            rows = []
            for (fs, aug), (models, tasks) in CV_GROUPS.items():
                feats = feats_f if fs else list(X.columns)
                if aug:
                    X_fit, y_fit = tvae_augment(c, X_trf[feats], y_trf, f"{tag}_{'fs' if fs else 'all'}",
                                                seed, cache=False)
                else:
                    X_fit, y_fit = X_trf[feats], y_trf
                X_fit, X_eval = X_fit.to_numpy(np.float32), X_vaf[feats].to_numpy(np.float32)
                for task in tasks:
                    binary = task == "binary"
                    yf = (y_fit != c.benign).astype(np.int64) if binary else y_fit
                    ye = (y_vaf != c.benign).astype(np.int64) if binary else y_vaf
                    # checkpoint prefix is distinct per seed/fold/setup/task
                    ptag = f"{tag}_{safe(SETUP_NAMES[(fs, aug)])}_{task}"
                    preds = fit_predict(X_fit, yf, X_eval, models, seed, ptag)
                    for model, (pred, proba) in preds.items():
                        rows.append({"seed": seed, "fold": fold, "Task": task,
                                     "Setup": SETUP_NAMES[(fs, aug)], "Model": model,
                                     **evaluate(c, ye, pred, proba)})
            df = pd.DataFrame(rows)
            save_csv(df, path, index=False)
            frames.append(df)
            for f in glob.glob(out(f"*{tag}_*")):          # bulky intermediates
                if not f.endswith(".csv"):
                    with contextlib.suppress(OSError):
                        os.remove(f)

    runs = pd.concat(frames, ignore_index=True)
    save_csv(runs, out("cv_runs.csv"), index=False)
    metrics = ["accuracy", "macro_f1", "weighted_f1", "balanced_accuracy", "mcc"]
    agg = runs.groupby(["Task", "Setup", "Model"])[metrics].agg(["mean", "std"])
    table = pd.DataFrame(index=agg.index)
    for m in metrics:
        s, d = (1, 3) if m == "mcc" else (100, 2)
        table[m] = [f"{a * s:.{d}f} ± {b * s:.{d}f}"
                    for a, b in zip(agg[(m, "mean")], agg[(m, "std")])]
    table["runs"] = runs.groupby(["Task", "Setup", "Model"]).size()
    print(f"\n===== TABLE 10: {len(cfg.cv_seeds)} seeds x {CV_FOLDS} folds (mean ± std) =====")
    print(table.to_string())
    save_csv(table, out("table10_cv.csv"))
    tests = []
    for (task, setup), g in runs.groupby(["Task", "Setup"]):
        piv = g.pivot_table(index=["seed", "fold"], columns="Model", values="macro_f1")
        if "STACK" not in piv.columns:
            continue
        for base in [k for k in piv.columns if k != "STACK"]:
            a, b = piv["STACK"].to_numpy(), piv[base].to_numpy()
            t1, p1 = stats.ttest_rel(a, b)
            t2, p2 = corrected_ttest(a, b, 1.0 / (CV_FOLDS - 1))
            tests.append({"Task": task, "Setup": setup, "Comparison": f"STACK vs {base}",
                          "Mean macro-F1 diff": a.mean() - b.mean(), "Runs": len(a),
                          "t (paired)": t1, "p (paired)": p1,
                          "t (corrected)": t2, "p (corrected)": p2})
    tests = pd.DataFrame(tests)
    save_csv(tests, out("table10_tests.csv"), index=False)
    print("\n===== PAIRED TESTS on macro F1 =====")
    print(tests.to_string(index=False))


# ===========================================================================
# Main
# ===========================================================================
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--data-dir", help="folder with the daily CSE-CIC-IDS2018 CSVs")
    src.add_argument("--audit-dir", help="folder with the pre-cleaned audit parquet files")
    ap.add_argument("--out-dir", required=True, help="checkpoint / results folder")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exclude-dst-port", action="store_true",
                    help="ablation: drop Destination Port before feature selection")
    ap.add_argument("--split", choices=["stratified", "chronological"], default="stratified",
                    help="chronological = earliest 60%% of each class train, latest 40%% test")
    ap.add_argument("--run-cv", action="store_true", help="Table 10 (25 runs, very long)")
    ap.add_argument("--cv-seeds", type=lambda s: tuple(int(v) for v in s.split(",")),
                    default=CV_SEEDS, help="seeds for --run-cv (paper: 0,1,2,3,4)")
    ap.add_argument("--no-diagnostics", action="store_true", help="skip Stage 7")
    ap.add_argument("--lc-fracs", type=lambda s: [float(v) for v in s.split(",") if v],
                    default=[0.05, 0.10, 0.25, 0.50],
                    help="learning-curve training fractions ('' to skip)")
    ap.add_argument("--diag-val-rows", type=int, default=200_000)
    ap.add_argument("--diag-tvae-max-rows", type=int, default=200_000)
    ap.add_argument("--force-rerun", action="store_true")
    ap.add_argument("--save-models", action="store_true",
                    help="also pickle fitted models (RF/ET files are several GB)")
    ap.add_argument("--checksum", action="store_true", help="md5 the input files")
    # overrides for smoke tests only; the paper's values are the defaults
    ap.add_argument("--tvae-epochs", type=int, default=TVAE_EPOCHS)
    ap.add_argument("--base-trees", type=int, default=100)
    ap.add_argument("--meta-trees", type=int, default=100)
    return ap.parse_args(argv)


def main(argv=None):
    global cfg
    cfg = parse_args(argv)
    os.makedirs(cfg.out_dir, exist_ok=True)
    write_run_records(cfg)

    with timed("stage1_clean"):
        X, y, classes, ts = stage1_clean()
    with timed("stage2_split"):
        c = stage2_split(X, y, classes, ts)
    del X
    with timed("stage3_feature_selection"):
        stage3_feature_selection(c)
    res = stage6_holdout(c)                # Stages 4-6
    if not cfg.no_diagnostics:
        stage7_diagnostics(c)
    if cfg.run_cv:
        stage8_cv(c)
    print(f"\nDone. All checkpoints and results are in {cfg.out_dir}")
    return res


if __name__ == "__main__":
    main()
