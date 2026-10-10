"""
TCAV reproduction -- Kim et al. (ICML 2018): zebra class vs. 'striped' concept,
ImageNet-pretrained GoogLeNet (Inception v1) -- plus an ICASSP-2026 style extension
(how do CAV type and the NON-CONCEPT distribution change the TCAV score?).

How each step maps to Kim et al. (2018)
  Sec 3.2  CAV = normal of a LINEAR classifier trained on the FLATTENED bottleneck
           activations (w*h*c values, no spatial pooling): concept vs. random examples.
  Sec 3.3  S_{C,k,l}(x) = grad_h logit_k(f_l(x)) . v_C      (logit, not softmax)
  Sec 3.4  TCAV_Q = fraction of class-k images with S > 0
  Sec 3.5  ~500 CAVs, each trained against a DIFFERENT random counterexample set,
           two-sided t-test + Bonferroni correction
  Fig. 4   TCAV for zebra across all GoogLeNet layers (inception3a ... inception5b)
           with 'striped' compared to control concepts ('dotted', 'zigzagged')

Expected folder layout (override the root with env var TCAV_DATASET_DIR):
  Dataset/
    Zebra/{training,val,test}/        target class images (no training happens on them)
    Non Zebra/{training,val,test}/    only used for the detection-accuracy report
    Stripes/                          concept (paper: 'striped', e.g. DTD 'striped')
    Dotted/  Zigzagged/               OPTIONAL control concepts (as in paper Fig. 4)
    Random/                           >= 500 diverse random images (paper's counterexamples)
    Non Stripes/                      OPTIONAL: your curated non-stripe images

GoogLeNet stays frozen. Only the small linear CAV classifier is trained.
"""

import json
import os
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from scipy import stats
from sklearn.linear_model import LogisticRegression, Ridge, SGDClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from torchvision.models import GoogLeNet_Weights, googlenet

# ============================================================
# 1. CONFIGURATION
# ============================================================
DATASET_DIR = Path(os.environ.get("TCAV_DATASET_DIR",
                                  Path.home() / "Downloads" / "Dataset"))
RESULTS_DIR = Path(__file__).resolve().parent / "results_v2"

# --- target class (TCAV never trains on these, so every split can be used) ---
TARGET_FOLDER = "Zebra"
NON_TARGET_FOLDER = "Non Zebra"
TARGET_CLASS_NAME = "zebra"
TARGET_SPLITS = ["training", "val", "test"]       # use ["test"] to stay held-out

# --- concepts ---
PRIMARY_CONCEPT = "Stripes"                       # the concept you are testing
CONCEPT_FOLDERS = ["Stripes", "Dotted", "Zigzagged"]   # missing folders are skipped
N_CONCEPT_IMAGES = 50                             # paper: 30-50 images per concept

# --- non-concept ("counterexample") distributions ---
# name -> folders the negatives are drawn from. The folder of the concept being
# tested is automatically excluded. "random" is the paper's default.
NEGATIVE_CONDITIONS = {
    "random": ["Random"],
    "non_stripes": ["Non Stripes"],
    "other_textures": ["Dotted", "Zigzagged", "Stripes"],   # relative CAV (paper Sec 3.6)
}

# --- layers: torchvision inception3a..5b == TF 'mixed3a'..'mixed5b' ---
ALL_LAYERS = ["inception3a", "inception3b", "inception4a", "inception4b",
              "inception4c", "inception4d", "inception4e", "inception5a",
              "inception5b"]
LAYERS = ["inception4d"]        # set to ALL_LAYERS for the Fig.-4 sweep (needs lots of RAM)

# --- CAV estimation ---
# "sgd" = paper-style linear classifier. "logistic", "ridge", "pattern", "fast"
# are the CAV types compared in the ICASSP paper (optional).
CAV_METHODS = ["sgd"]
SGD_ALPHA = 0.01                # regularisation of the SGD linear classifier
LOGISTIC_C = 0.01
RIDGE_ALPHA = 100.0
VAL_FRACTION = 1 / 3            # held-out part of each CAV training set (paper: 1/3)
NUM_RUNS = int(os.environ.get("TCAV_NUM_RUNS", 500))

BATCH_SIZE = 8
MAX_POOL_IMAGES = 600           # cap per folder (RAM: images x layer_dim x 4 bytes)
RANDOM_SEED = 42
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

ImageFile.LOAD_TRUNCATED_IMAGES = True
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available() else "cpu")

MODEL = None
PREPROCESS = None
ZEBRA_INDEX = None


# ============================================================
# 2. MODEL + DATA HELPERS
# ============================================================
def setup_model():
    global MODEL, PREPROCESS, ZEBRA_INDEX
    weights = GoogLeNet_Weights.DEFAULT
    MODEL = googlenet(weights=weights).to(DEVICE).eval()
    for p in MODEL.parameters():
        p.requires_grad_(False)
    PREPROCESS = weights.transforms()
    ZEBRA_INDEX = weights.meta["categories"].index(TARGET_CLASS_NAME)
    print(f"Device: {DEVICE} | '{TARGET_CLASS_NAME}' = ImageNet class {ZEBRA_INDEX}")


def resolve_dir(parent, name):
    exact = parent / name
    if exact.exists() or not parent.exists():
        return exact
    for c in parent.iterdir():
        if c.is_dir() and c.name.strip() == name:
            return c
    return exact


def find_images(folder):
    if not folder.exists():
        return []
    return sorted(p for p in folder.rglob("*")
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def iter_batches(paths):
    """Yield (tensor_on_device, valid_paths); unreadable images are skipped."""
    for s in range(0, len(paths), BATCH_SIZE):
        tensors, good = [], []
        for p in paths[s:s + BATCH_SIZE]:
            try:
                with Image.open(p) as im:
                    tensor = PREPROCESS(im.convert("RGB"))
                tensors.append(tensor)
                good.append(p)
            except Exception as err:
                print(f"  skipping unreadable image {p}: {err}")
        if tensors:
            yield torch.stack(tensors).to(DEVICE), good


# ============================================================
# 3. ACTIVATIONS AND GRADIENTS (FLATTENED, NO POOLING)
# ============================================================
def extract_activations(paths, layer_name):
    """One flattened (c*h*w) activation vector per image, as in the paper."""
    layer, store = getattr(MODEL, layer_name), {}
    handle = layer.register_forward_hook(
        lambda m, i, o: store.__setitem__("a", o.detach()))
    feats = []
    try:
        for batch, _ in iter_batches(paths):
            with torch.no_grad():
                MODEL(batch)
            feats.append(store["a"].flatten(1).cpu().numpy().astype(np.float32))
    finally:
        handle.remove()
    if not feats:
        raise RuntimeError("No readable images.")
    return np.concatenate(feats)


def extract_target_gradients(paths, layer_name):
    """d(zebra LOGIT)/d(layer activation), flattened. Computed once per layer."""
    layer, store = getattr(MODEL, layer_name), {}
    handle = layer.register_forward_hook(lambda m, i, o: store.__setitem__("a", o))
    grads, probs, top1 = [], [], []
    try:
        for batch, _ in iter_batches(paths):
            batch.requires_grad_(True)
            with torch.enable_grad():
                logits = MODEL(batch)
                g = torch.autograd.grad(logits[:, ZEBRA_INDEX].sum(), store["a"])[0]
            grads.append(g.flatten(1).cpu().numpy().astype(np.float32))
            probs.extend(torch.softmax(logits.detach(), 1)[:, ZEBRA_INDEX].cpu().tolist())
            top1.extend((logits.detach().argmax(1) == ZEBRA_INDEX).cpu().tolist())
    finally:
        handle.remove()
    if not grads:
        raise RuntimeError("No gradients computed.")
    return np.concatenate(grads), np.asarray(probs), np.asarray(top1)


# ============================================================
# 4. CAV ESTIMATORS
# ============================================================
def fit_cav(method, X, y, rng):
    """Return (unit-norm CAV pointing toward the concept, matching bias)."""
    if method == "sgd":
        clf = SGDClassifier(alpha=SGD_ALPHA, max_iter=1000, tol=1e-3,
                            random_state=int(rng.integers(2**31 - 1)))
        clf.fit(X, y)
        w, b = clf.coef_[0], float(clf.intercept_[0])
    elif method == "logistic":
        clf = LogisticRegression(C=LOGISTIC_C, max_iter=500).fit(X, y)
        w, b = clf.coef_[0], float(clf.intercept_[0])
    elif method == "ridge":
        clf = Ridge(alpha=RIDGE_ALPHA).fit(X, 2.0 * y - 1.0)
        w, b = clf.coef_.ravel(), float(np.ravel(clf.intercept_)[0])
    elif method in ("pattern", "fast"):
        mu_pos, mu_neg = X[y == 1].mean(0), X[y == 0].mean(0)
        w = mu_pos - mu_neg if method == "pattern" else mu_pos - X.mean(0)
        b = -0.5 * float(w @ (mu_pos + mu_neg))        # midpoint threshold
    else:
        raise ValueError(f"unknown CAV method {method!r}")
    w = np.asarray(w, dtype=np.float32).ravel()
    norm = float(np.linalg.norm(w))
    if not np.isfinite(norm) or norm == 0:
        raise RuntimeError("degenerate CAV")
    return w / norm, b / norm


def one_fit(method, pos, neg, G, rng):
    X = np.concatenate([pos, neg]).astype(np.float32, copy=False)
    y = np.concatenate([np.ones(len(pos), int), np.zeros(len(neg), int)])
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=VAL_FRACTION, stratify=y,
        random_state=int(rng.integers(2**31 - 1)))
    w, b = fit_cav(method, X_tr, y_tr, rng)
    train_acc = float(np.mean(((X_tr @ w + b) > 0) == y_tr))
    val_acc = float(np.mean(((X_va @ w + b) > 0) == y_va))
    deriv = G @ w                                       # directional derivatives S(x)
    positive = deriv > 0
    return {"tcav_score": float(positive.mean()),
            "cav_train_acc": train_acc, "cav_val_acc": val_acc,
            "train_val_gap": train_acc - val_acc,
            "mean_directional_derivative": float(deriv.mean())}, positive


def repeated_fits(method, concept_feats, pool_feats, G, control, n_set):
    """
    control=False: concept vs. a FRESH random draw from the pool in every run.
    control=True : random-vs-random CAVs (paper's null distribution).
    """
    rows, hits = [], np.zeros(len(G))
    for run in range(1, NUM_RUNS + 1):
        rng = np.random.default_rng(RANDOM_SEED + run + (100_000 if control else 0))
        if control:
            idx = rng.choice(len(pool_feats), size=2 * n_set, replace=False)
            pos, neg = pool_feats[idx[:n_set]], pool_feats[idx[n_set:]]
        else:
            idx = rng.choice(len(pool_feats), size=n_set, replace=False)
            pos, neg = concept_feats, pool_feats[idx]
        rec, positive = one_fit(method, pos, neg, G, rng)
        rec["run"] = run
        rows.append(rec)
        hits += positive
        if run % 100 == 0:
            print(f"      run {run}/{NUM_RUNS}")
    return pd.DataFrame(rows), hits / NUM_RUNS


# ============================================================
# 5. STATISTICS
# ============================================================
def summarize_block(scores, control_scores, per_image_rate):
    scores = np.asarray(scores)
    n, mean, sd = len(scores), scores.mean(), scores.std(ddof=1)
    half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    with np.errstate(all="ignore"):
        t1 = stats.ttest_1samp(scores, 0.5)
        tc = stats.ttest_ind(scores, control_scores, equal_var=False)
    # Bootstrap over TARGET IMAGES: captures image-sampling uncertainty that the
    # t-interval over repeated CAV fits cannot (those runs reuse the same images).
    rng = np.random.default_rng(RANDOM_SEED)
    boot = per_image_rate[rng.integers(0, len(per_image_rate),
                                       (2000, len(per_image_rate)))].mean(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"mean_tcav": float(mean), "sd_tcav": float(sd),
            "ci95_t_low": float(mean - half), "ci95_t_high": float(mean + half),
            "ci95_boot_images_low": float(lo), "ci95_boot_images_high": float(hi),
            "control_mean_tcav": float(np.mean(control_scores)),
            "control_sd_tcav": float(np.std(control_scores, ddof=1)),
            "t_vs_0.5": float(t1.statistic), "p_vs_0.5": float(t1.pvalue),
            "t_vs_control": float(tc.statistic), "p_vs_control": float(tc.pvalue)}


# ============================================================
# 6. ONE LAYER: ALL CONCEPTS x CONDITIONS x METHODS
# ============================================================
def run_layer(layer, folder_paths, concepts, conditions, target_paths):
    print(f"\n=== Layer {layer} ===")
    G, probs, top1 = extract_target_gradients(target_paths, layer)
    print(f"  target gradients: {G.shape} | mean zebra prob {probs.mean():.3f} "
          f"| top-1 zebra rate {top1.mean():.3f}")
    folder_feats = {}
    for name, paths in folder_paths.items():
        folder_feats[name] = extract_activations(paths, layer)
        print(f"  activations {name}: {folder_feats[name].shape}")

    run_rows, summaries, control_cache = [], [], {}
    for concept in concepts:
        feats = folder_feats[concept]
        n_set = min(N_CONCEPT_IMAGES, len(feats))
        sel = np.random.default_rng(RANDOM_SEED).choice(len(feats), n_set, replace=False)
        concept_feats = feats[sel]                       # fixed concept set, as in the paper
        for ci, (cond, cond_folders) in enumerate(conditions.items()):
            if concept != PRIMARY_CONCEPT and ci > 0:    # control concepts: first condition only
                continue
            used = [f for f in cond_folders if f != concept and f in folder_feats]
            if not used:
                continue
            pool = np.concatenate([folder_feats[f] for f in used])
            if len(pool) < 2 * n_set:
                print(f"  [skip] {concept}/{cond}: pool {len(pool)} < 2*{n_set}")
                continue
            for method in CAV_METHODS:
                print(f"  concept={concept} | negatives={cond} {used} | method={method}")
                ckey = (tuple(used), method, n_set)
                if ckey not in control_cache:
                    cdf, _ = repeated_fits(method, None, pool, G, True, n_set)
                    control_cache[ckey] = cdf["tcav_score"].to_numpy()
                    cdf = cdf.assign(layer=layer, concept="RANDOM_CONTROL",
                                     condition=cond, method=method)
                    run_rows.append(cdf)
                df, rate = repeated_fits(method, concept_feats, pool, G, False, n_set)
                df = df.assign(layer=layer, concept=concept, condition=cond, method=method)
                run_rows.append(df)
                row = summarize_block(df["tcav_score"].to_numpy(), control_cache[ckey], rate)
                row.update(layer=layer, concept=concept, condition=cond, method=method,
                           negative_folders="+".join(used), n_concept_images=n_set,
                           n_pool_images=len(pool), n_target_images=len(G), num_runs=NUM_RUNS,
                           mean_cav_train_acc=float(df["cav_train_acc"].mean()),
                           mean_cav_val_acc=float(df["cav_val_acc"].mean()),
                           mean_train_val_gap=float(df["train_val_gap"].mean()),
                           mean_zebra_prob=float(probs.mean()), top1_zebra_rate=float(top1.mean()))
                summaries.append(row)
                print(f"      TCAV {row['mean_tcav']:.3f} +- {row['sd_tcav']:.3f} | "
                      f"control {row['control_mean_tcav']:.3f} | p(vs control) "
                      f"{row['p_vs_control']:.2e} | CAV val acc {row['mean_cav_val_acc']:.3f}")
    return pd.concat(run_rows, ignore_index=True), summaries


# ============================================================
# 7. ZEBRA DETECTION REPORT (separate from the CAV accuracy!)
# ============================================================
def zebra_detection_metrics(zebra_paths, non_zebra_paths):
    if not zebra_paths or not non_zebra_paths:
        return {"note": "need both Zebra and Non Zebra images"}
    y_true, y_pred = [], []
    for label, paths in ((1, zebra_paths), (0, non_zebra_paths)):
        for batch, good in iter_batches(paths):
            with torch.no_grad():
                pred = MODEL(batch).argmax(1).cpu().numpy()
            y_true += [label] * len(good)
            y_pred += (pred == ZEBRA_INDEX).astype(int).tolist()
    return {"num_images": len(y_true),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
            "rule": "predicted zebra if GoogLeNet top-1 ImageNet class is 'zebra'"}


# ============================================================
# 8. PLOTS
# ============================================================
def save_plots(runs_df, summary_df):
    for layer, sub in runs_df.groupby("layer"):
        groups = list(sub.groupby(["concept", "condition", "method"]))
        plt.figure(figsize=(max(8, 1.3 * len(groups)), 5.5))
        plt.boxplot([g["tcav_score"].values for _, g in groups], showfliers=False)
        plt.xticks(range(1, len(groups) + 1), ["\n".join(k) for k, _ in groups],
                   rotation=45, ha="right", fontsize=8)
        plt.axhline(0.5, ls="--", c="gray")
        plt.ylabel("TCAV score")
        plt.title(f"TCAV for '{TARGET_CLASS_NAME}' -- {layer} "
                  f"({NUM_RUNS} CAVs per box; RANDOM_CONTROL = random-vs-random)")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"tcav_distributions_{layer}.png", dpi=160)
        plt.close()
    if summary_df["layer"].nunique() > 1:
        order = {l: i for i, l in enumerate(ALL_LAYERS)}
        plt.figure(figsize=(9, 5.5))
        for (concept, cond, method), g in summary_df.groupby(["concept", "condition", "method"]):
            g = g.sort_values("layer", key=lambda s: s.map(order))
            plt.errorbar(g["layer"], g["mean_tcav"], yerr=g["sd_tcav"], marker="o",
                         capsize=3, label=f"{concept} | {cond} | {method}")
        plt.axhline(0.5, ls="--", c="gray")
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("mean TCAV score (+-1 SD over CAV fits)")
        plt.title("TCAV across layers (cf. Kim et al. Fig. 4)")
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "tcav_layer_sweep.png", dpi=160)
        plt.close()


# ============================================================
# 9. MAIN
# ============================================================
def main():
    setup_model()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Dataset:", DATASET_DIR)

    target_paths = []
    for split in TARGET_SPLITS:
        target_paths += find_images(resolve_dir(DATASET_DIR, TARGET_FOLDER) / split)
    non_target_paths = []
    for split in TARGET_SPLITS:
        non_target_paths += find_images(resolve_dir(DATASET_DIR, NON_TARGET_FOLDER) / split)
    if not target_paths:
        raise FileNotFoundError(f"No images in {DATASET_DIR / TARGET_FOLDER}/<split>")
    print(f"Target images: {len(target_paths)} | non-target: {len(non_target_paths)}")

    needed = set(CONCEPT_FOLDERS) | {f for fs in NEGATIVE_CONDITIONS.values() for f in fs}
    folder_paths = {}
    for name in sorted(needed):
        paths = find_images(resolve_dir(DATASET_DIR, name))
        if not paths:
            print(f"  [warn] folder '{name}' missing or empty -> skipped")
            continue
        if len(paths) > MAX_POOL_IMAGES:
            paths = sorted(random.Random(RANDOM_SEED).sample(paths, MAX_POOL_IMAGES))
        folder_paths[name] = paths
        print(f"  {name}: {len(paths)} images")
    if PRIMARY_CONCEPT not in folder_paths:
        raise FileNotFoundError(f"Concept folder '{PRIMARY_CONCEPT}' is required.")
    concepts = [c for c in CONCEPT_FOLDERS if c in folder_paths]
    conditions = {k: [f for f in v if f in folder_paths] for k, v in NEGATIVE_CONDITIONS.items()}
    conditions = {k: v for k, v in conditions.items() if v}
    if not conditions:
        raise FileNotFoundError("No negative/counterexample folder found (e.g. 'Random').")
    if "Random" not in folder_paths:
        print("  [warn] no 'Random' folder: you are NOT using the paper's default "
              "random counterexamples.")

    detection = zebra_detection_metrics(target_paths, non_target_paths)
    print("Frozen GoogLeNet zebra detection:", json.dumps(detection, indent=2))

    all_runs, all_summ = [], []
    for layer in LAYERS:
        runs_df, summ = run_layer(layer, folder_paths, concepts, conditions, target_paths)
        all_runs.append(runs_df)
        all_summ += summ
        pd.concat(all_runs).to_csv(RESULTS_DIR / "tcav_runs.csv", index=False)   # checkpoint
        pd.DataFrame(all_summ).to_csv(RESULTS_DIR / "tcav_summary.csv", index=False)

    runs_df = pd.concat(all_runs, ignore_index=True)
    summary_df = pd.DataFrame(all_summ)
    m_tests = len(summary_df)
    summary_df["p_vs_control_bonferroni"] = (summary_df["p_vs_control"] * m_tests).clip(upper=1)
    summary_df["significant_bonferroni_all_tests"] = summary_df["p_vs_control_bonferroni"] < 0.05
    summary_df["passes_paper_style_test"] = summary_df["p_vs_control"] < 0.05 / 2   # alpha/m, m=2
    summary_df.to_csv(RESULTS_DIR / "tcav_summary.csv", index=False)
    runs_df.to_csv(RESULTS_DIR / "tcav_runs.csv", index=False)
    save_plots(runs_df, summary_df)

    meta = {"model": "torchvision GoogLeNet, ImageNet weights, frozen",
            "activations": "flattened c*h*w (no pooling)",
            "gradient_of": "zebra logit", "layers": LAYERS, "cav_methods": CAV_METHODS,
            "sgd_alpha": SGD_ALPHA, "val_fraction": VAL_FRACTION, "num_runs": NUM_RUNS,
            "n_concept_images": N_CONCEPT_IMAGES, "target_splits": TARGET_SPLITS,
            "n_hypothesis_tests_for_bonferroni": m_tests, "frozen_detection": detection,
            "notes": ["CAV val accuracy is NOT zebra classification accuracy.",
                      "Runs share the same target + concept images: the t-tests are about "
                      "variability over counterexample draws, not independent studies.",
                      "ci95_boot_images_* resamples target images (image-sampling uncertainty)."]}
    (RESULTS_DIR / "run_metadata.json").write_text(json.dumps(meta, indent=2))

    cols = ["layer", "concept", "condition", "method", "mean_tcav", "sd_tcav",
            "control_mean_tcav", "p_vs_control_bonferroni", "mean_cav_val_acc"]
    print("\n================ SUMMARY ================")
    print(summary_df[cols].to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print(f"\nSaved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()