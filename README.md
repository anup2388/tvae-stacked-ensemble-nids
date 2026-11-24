# TVAE-Augmented Stacked Ensemble for Network Intrusion Detection

Reference implementation of the manuscript **"An Imbalance-Resilient Network Intrusion Detection Framework Using TVAE-Based Data Augmentation and Stacked Ensemble Learning"** (A. M. Abraham, R. M. S. Kumar), evaluated on CSE-CIC-IDS2018.

The pipeline runs from the daily CSE-CIC-IDS2018 CSV files to the tables and figures of the Results section. It is available as a command-line script (`nids_tvae_stack.py`) and as an equivalent Jupyter/Colab notebook (`nids_tvae_stack.ipynb`). Every intermediate result is written to disk as soon as it exists, so an interrupted run resumes where it stopped.

## Contents

| File | Purpose |
|---|---|
| `nids_tvae_stack.py` | The complete pipeline (command line) |
| `nids_tvae_stack.ipynb` | The same code, one cell per stage, with result-inspection cells (Colab-ready) |
| `make_toy_data.py` | Generates a small CSE-CIC-IDS2018-shaped dataset for a smoke test |
| `requirements.txt` | Library versions |
| `LICENSE`, `CITATION.cff` | MIT licence; citation metadata |

## Pipeline

| Stage | What it does | Paper |
|---|---|---|
| 1 | Cleaning: drop infinite/missing values and duplicate flows; remove flow ID, source/destination IP, source port and timestamp. **Destination port is retained as a feature.** | Algorithm 2, Section III.A |
| 2 | Benign subsampling `N_b' = r·N_b` with `r = 0.1`, then a stratified 60:40 train/test split (seed 42) | Eq. 1 |
| 3 | Feature selection: Information Gain (64 equal-frequency bins, top 45), then Pearson `\|ρ\| > 0.85` (the lower-IG feature of each correlated pair is dropped) | Algorithm 1, Tables 1-2, Fig. 4 |
| 4 | Class-wise TVAE (SDV `TVAESynthesizer`, 100 epochs, embedding dimension 128, batch size 500): each minority attack class is raised to the benign count of the training set | Eq. 9-12 |
| 5 | Stack: Random Forest + LightGBM + Extra Trees (100 trees each), 5-fold out-of-fold probabilities, XGBoost meta-classifier (100 trees, depth 6, learning rate 0.3) | Eq. 18 |
| 6 | Evaluation: binary and multi-class reports, confusion matrices, ROC, ablation over {feature selection, TVAE} | Tables 3-9, 11 |
| 7 | Diagnostics: TVAE train/validation loss, LightGBM and XGBoost per-iteration loss, stack learning curve | Supplementary Material |
| 8 | Optional: 5 seeds × 5-fold stratified cross-validation with the corrected resampled paired t-test | Table 10 |

All seeds are fixed (42 for partitioning, model initialisation and TVAE training; 0-4 for cross-validation). The TVAE learning rate is not a setting in SDV; its Adam optimiser runs at the library default of 1e-3.

## Installation

```bash
pip install -r requirements.txt
```

`sdv` pulls in `ctgan` and `torch`. A GPU is not required, but TVAE training is much faster with one (the code uses CUDA when available). On Google Colab, `sdv` is not preinstalled; the notebook installs it in its first cell.

The versions used for the reported results are listed in `requirements.txt`. Each run also writes the versions it actually used to `environment.json` in the output folder.

## Data

The dataset is not redistributed. Download CSE-CIC-IDS2018 from the [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/ids-2018.html) and use the "Processed Traffic Data for ML Algorithms" CSV files.

| Input | Option | Notes |
|---|---|---|
| Folder with the daily CSV files | `--data-dir` | Stage 1 cleans the raw files. Required for the chronological split, because only the raw files carry timestamps. |
| Folder with already cleaned parquet files (`X_audit_train.parquet`, `X_audit_test.parquet`, `y_audit_train.parquet`, `y_audit_test.parquet`, `audit_label_encoder.joblib`) | `--audit-dir` | Skips raw-file cleaning. This repository does not generate these files. |

## Run (command line)

```bash
# main experiment: the paper's settings are the defaults
python nids_tvae_stack.py --data-dir /path/to/csvs --out-dir runs/main

# Table 10: 5 seeds x 5 folds, resumable per fold (very long)
python nids_tvae_stack.py --data-dir /path/to/csvs --out-dir runs/main --run-cv

# supplementary analyses; each uses its own output folder
python nids_tvae_stack.py --data-dir /path/to/csvs --out-dir runs/no_dstport --exclude-dst-port
python nids_tvae_stack.py --data-dir /path/to/csvs --out-dir runs/chrono --split chronological
```

Running the same command again resumes from the saved checkpoints. Use a new `--out-dir` for every experiment: an existing folder is read as a checkpoint, so results from a run with different settings would otherwise be reused.

| Option | Purpose |
|---|---|
| `--exclude-dst-port` | Ablation: drop destination port before feature selection |
| `--split chronological` | Earliest 60% of each class for training, latest 40% for testing (needs `--data-dir`) |
| `--run-cv` | Stage 8 (Table 10) |
| `--no-diagnostics` | Skip Stage 7 |
| `--lc-fracs 0.05,0.1,0.25,0.5` | Training fractions of the learning curve (`""` skips it) |
| `--force-rerun` | Recompute everything instead of loading checkpoints |
| `--save-models` | Also pickle fitted models (the RF and ET files are several GB) |
| `--checksum` | Record md5 hashes of the input files |

`--tvae-epochs`, `--base-trees`, `--meta-trees` and `--seed` exist for smoke tests. Changing them sets `"paper_settings": false` in `config.json`, and the results are not comparable with the paper.

## Run (notebook / Colab)

1. Open `nids_tvae_stack.ipynb`.
2. On Colab, mount Google Drive (the notebook has a cell for this) and put the data and `OUT_DIR` on Drive, so checkpoints survive a disconnect.
3. Edit the configuration cell: `DATA_DIR` (or `AUDIT_DIR`), `OUT_DIR`, and the switches `EXCLUDE_DST_PORT`, `SPLIT`, `RUN_DIAGNOSTICS`, `RUN_CV`.
4. Run the cells in order. After a restart, run again from the top with the same settings: finished stages load from `OUT_DIR`.

The last cells display the class distribution, the hold-out results, the diagnostic figures and the timing.

## Outputs written to `--out-dir`

| Files | Contents |
|---|---|
| `config.json`, `environment.json`, `input_files.json` | Settings, library versions, input file sizes (and hashes with `--checksum`) |
| `stage1_cleaned.parquet`, `stage1_classes.json` | Cleaned dataset and class list |
| `stage2_split.npz`, `class_distribution.csv` | Train/test indices; class counts per partition (Figs. 2, 3, 5, 6) |
| `ig_ranking.csv`, `table1_ig_ranking_full.csv`, `fig4_ig_top30.png` | Information Gain ranking (Table 1, Fig. 4) |
| `selected_features.json`, `feature_selection_dropped.csv` | Selected features (Table 2) and the redundant features removed |
| `syn_*.parquet`, `tvae_loss_*.csv` | Synthetic rows and training-loss history of each TVAE class |
| `oof_*.npy`, `oof_Z_*.npy`, `proba_*.npy`, `pred_*.npz` | Out-of-fold base-learner probabilities, meta-features, test probabilities and predictions |
| `report_*.csv`, `cm_*.png`/`.csv`, `roc_binary.png`, `holdout_results.csv` | Per-class reports (Tables 3, 5-8), confusion matrices, ROC, all metrics |
| `table4_binary.csv`, `table9_setups.csv`, `table11_ablation_holdout.csv` | Summary tables |
| `diag_*.csv`, `diag_*.png` | Stage 7 diagnostics |
| `cv_s<seed>_f<fold>.csv`, `table10_cv.csv`, `table10_tests.csv` | Cross-validation (Table 10) |
| `timing.json` | Wall-clock seconds per stage and per TVAE class |

All metrics can be recomputed from the saved `pred_*.npz` files without refitting any model.

## Smoke test

`make_toy_data.py` writes a small CSE-CIC-IDS2018-shaped dataset. The test checks that the code runs end to end; its numbers mean nothing.

```bash
python make_toy_data.py toy/data
python nids_tvae_stack.py --data-dir toy/data --out-dir toy/out \
    --tvae-epochs 3 --base-trees 10 --meta-trees 10 --lc-fracs 0.5 \
    --diag-val-rows 500 --diag-tvae-max-rows 2000
```

## Reproducibility notes

- Exact numerical equality across hardware and library versions is not guaranteed, in particular for the TVAEs when trained on a GPU.
- Run time on the full dataset is long; every run records it in `timing.json`.
- Memory: the cleaned dataset has about 5 million flows and the TVAE-augmented training set about 3.4 million rows. Use a high-memory runtime.

## Citation

If you use this code, please cite the manuscript. Citation metadata is in `CITATION.cff`; the full reference will be added on publication. A DOI for the archived release: `[DOI]`.

## Licence

MIT. See `LICENSE`.
