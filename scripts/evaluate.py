"""Compare baselines, ESM-2 zero-shot and trained models with gene-level cross-validation.

Usage:
    python scripts/evaluate.py [--model esm2_t30_150M_UR50D]

Every model is scored on genes it was not trained on: the five folds from
build_dataset.py hold all of a gene's variants together. For contrast, the
trained models are also run once with an ordinary random split, which lets
variants of the same gene land in both training and test data.

Writes results/metrics_<model>.tsv and results/roc_<model>.png.
"""

from __future__ import annotations

import argparse

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

AA = "ARNDCQEGHILKMFPSTWYV"
BLOSUM62_ROWS = """
 4 -1 -2 -2  0 -1 -1  0 -2 -1 -1 -1 -1 -2 -1  1  0 -3 -2  0
-1  5  0 -2 -3  1  0 -2  0 -3 -2  2 -1 -3 -2 -1 -1 -3 -2 -3
-2  0  6  1 -3  0  0  0  1 -3 -3  0 -2 -3 -2  1  0 -4 -2 -3
-2 -2  1  6 -3  0  2 -1 -1 -3 -4 -1 -3 -3 -1  0 -1 -4 -3 -3
 0 -3 -3 -3  9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1
-1  1  0  0 -3  5  2 -2  0 -3 -2  1  0 -3 -1  0 -1 -2 -1 -2
-1  0  0  2 -4  2  5 -2  0 -3 -3  1 -2 -3 -1  0 -1 -3 -2 -2
 0 -2  0 -1 -3 -2 -2  6 -2 -4 -4 -2 -3 -3 -2  0 -2 -2 -3 -3
-2  0  1 -1 -3  0  0 -2  8 -3 -3 -1 -2 -1 -2 -1 -2 -2  2 -3
-1 -3 -3 -3 -1 -3 -3 -4 -3  4  2 -3  1  0 -3 -2 -1 -3 -1  3
-1 -2 -3 -4 -1 -2 -3 -4 -3  2  4 -2  2  0 -3 -2 -1 -2 -1  1
-1  2  0 -1 -3  1  1 -2 -1 -3 -2  5 -1 -3 -1  0 -1 -3 -2 -2
-1 -1 -2 -3 -1  0 -2 -3 -2  1  2 -1  5  0 -2 -1 -1 -1 -1  1
-2 -3 -3 -3 -2 -3 -3 -3 -1  0  0 -3  0  6 -4 -2 -2  1  3 -1
-1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4  7 -1 -1 -4 -3 -2
 1 -1  1  0 -1  0  0  0 -1 -2 -2  0 -1 -2 -1  4  1 -3 -2 -2
 0 -1  0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1  1  5 -2 -2  0
-3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1  1 -4 -3 -2 11  2 -3
-2 -2 -2 -3 -2 -1 -2 -3  2 -1 -1 -2 -1  3 -3 -2 -2  2  7 -1
 0 -3 -3 -3 -1 -2 -2 -3 -3  3  1 -2  1 -1 -2 -2  0 -3 -1  4
"""
BLOSUM62 = {(a, b): int(v) for a, row in zip(AA, BLOSUM62_ROWS.split("\n")[1:-1]) for b, v in zip(AA, row.split())}
# Kyte-Doolittle hydrophobicity and residue volume (cubic angstroms).
HYDRO = dict(zip(AA, [1.8, -4.5, -3.5, -3.5, 2.5, -3.5, -3.5, -0.4, -3.2, 4.5, 3.8, -3.9, 1.9, 2.8, -1.6, -0.8, -0.7, -0.9, -1.3, 4.2]))
VOLUME = dict(zip(AA, [88.6, 173.4, 114.1, 111.1, 108.5, 143.8, 138.4, 60.1, 153.2, 166.7, 166.7, 168.6, 162.9, 189.9, 112.7, 89.0, 116.1, 227.8, 193.6, 140.0]))
# Grantham distance from composition, polarity and volume (Grantham 1974).
COMP = dict(zip(AA, [0, 0.65, 1.33, 1.38, 2.75, 0.89, 0.92, 0.74, 0.58, 0, 0, 0.33, 0, 0, 0.39, 1.42, 0.71, 0.13, 0.20, 0]))
POL = dict(zip(AA, [8.1, 10.5, 11.6, 13.0, 5.5, 10.5, 12.3, 9.0, 10.4, 5.2, 4.9, 11.3, 5.7, 5.2, 8.0, 9.2, 8.6, 5.4, 6.2, 5.9]))
VOL_G = dict(zip(AA, [31, 124, 56, 54, 55, 85, 83, 3, 96, 111, 111, 119, 105, 132, 32.5, 32, 61, 170, 136, 84]))


def grantham(a: str, b: str) -> float:
    return float(np.sqrt(1.833 * (COMP[a] - COMP[b]) ** 2 + 0.1018 * (POL[a] - POL[b]) ** 2
                         + 0.000399 * (VOL_G[a] - VOL_G[b]) ** 2) * 50.723)


def classic_features(d: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=d.index)
    f["blosum62"] = [BLOSUM62[r, a] for r, a in zip(d.ref, d.alt)]
    f["grantham"] = [grantham(r, a) for r, a in zip(d.ref, d.alt)]
    f["d_hydro"] = d.alt.map(HYDRO) - d.ref.map(HYDRO)
    f["d_volume"] = d.alt.map(VOLUME) - d.ref.map(VOLUME)
    f["rel_pos"] = d.pos / d.length
    for a in AA:  # which amino acid is lost and gained
        f[f"ref_{a}"] = (d.ref == a).astype(int)
        f[f"alt_{a}"] = (d.alt == a).astype(int)
    return f


def gbm():
    return lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31, subsample=0.8,
                              subsample_freq=1, colsample_bytree=0.5, verbose=-1)


def cross_val(make, X: np.ndarray, y: np.ndarray, folds: np.ndarray) -> np.ndarray:
    """Out-of-fold predicted probabilities."""
    pred = np.zeros(len(y))
    for k in np.unique(folds):
        test = folds == k
        model = make().fit(X[~test], y[~test])
        pred[test] = model.predict_proba(X[test])[:, 1]
    return pred


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="esm2_t30_150M_UR50D")
    args = ap.parse_args()

    meta = pd.read_csv("data/variants.tsv", sep="\t")
    esm = pd.read_parquet(f"results/esm_{args.model}.parquet")
    emb = np.load(f"data/emb_{args.model}.npy").astype(np.float32)
    assert (esm.variation_id.values == meta.variation_id.values).all()
    d = meta.assign(esm_llr=esm.esm_llr.values)
    keep = np.isfinite(d.esm_llr.values)
    d, emb = d[keep].reset_index(drop=True), emb[keep]
    y, folds = d.label.values, d.fold.values
    random_folds = np.random.default_rng(0).integers(0, 5, len(d))

    classic = classic_features(d)
    X_classic = classic.values.astype(np.float32)
    X_classic_esm = np.column_stack([X_classic, d.esm_llr.values])
    X_emb = np.column_stack([emb, X_classic, d.esm_llr.values])

    preds = {
        "BLOSUM62 (no training)": -classic.blosum62.values.astype(float),
        "ESM-2 zero-shot (no training)": -d.esm_llr.values,
        "Classic features + GBM": cross_val(gbm, X_classic, y, folds),
        "Classic + ESM-2 score + GBM": cross_val(gbm, X_classic_esm, y, folds),
        "ESM-2 embeddings + logistic regression": cross_val(
            lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000)), X_emb, y, folds),
        "ESM-2 embeddings + GBM": cross_val(gbm, X_emb, y, folds),
    }
    leaky = {
        "ESM-2 embeddings + GBM, RANDOM split (leaky)": cross_val(gbm, X_emb, y, random_folds),
    }

    rows = []
    for name, p in {**preds, **leaky}.items():
        per_fold = [roc_auc_score(y[folds == k], p[folds == k]) for k in range(5)] if name in preds else []
        rows.append({
            "model": name,
            "auroc": roc_auc_score(y, p),
            "auroc_fold_sd": np.std(per_fold) if per_fold else np.nan,
            "auprc": average_precision_score(y, p),
        })
    metrics = pd.DataFrame(rows)
    metrics.to_csv(f"results/metrics_{args.model}.tsv", sep="\t", index=False, float_format="%.3f")
    print(f"{len(d)} variants, {d.gene.nunique()} genes, {y.mean():.1%} pathogenic\n")
    print(metrics.to_string(index=False, float_format="%.3f"))

    fig, ax = plt.subplots(figsize=(6, 6))
    for name, p in preds.items():
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, label=f"{name} ({roc_auc_score(y, p):.3f})", lw=1.5)
    ax.plot([0, 1], [0, 1], color="0.7", lw=1, ls="--")
    ax.set(xlabel="False positive rate", ylabel="True positive rate",
           title="Pathogenic vs. benign missense, genes held out")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(f"results/roc_{args.model}.png", dpi=200)


if __name__ == "__main__":
    main()
