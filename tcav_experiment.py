"""
Single-file TCAV experiment: zebra class vs. stripe concept.

Expected structure (Dataset is a sibling of TCAV_Project on Desktop):
Desktop/
├── TCAV_Project/
│   └── tcav_experiment.py
└── Dataset/
    ├── Zebra/
    │   ├── training/
    │   ├── val/
    │   └── test/
    ├── Non Zebra/
    │   ├── training/
    │   ├── val/
    │   └── test/
    ├── Stripes/       # 114 stripe concept images
    └── Non Stripes/   # 114 non-stripe control images

The ImageNet-pretrained ResNet50 stays frozen. CAVs are trained on
spatially averaged layer3 activations (1024 dimensions). TCAV evaluation
uses only Zebra/test. This is an adapted, single-file TCAV experiment,
not a claim of exact reproduction of every published TCAV implementation.
"""

from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from scipy.stats import ttest_1samp
from sklearn.svm import LinearSVC
from torchvision.models import resnet50, ResNet50_Weights


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

# The dataset currently lives in ~/Downloads/Dataset.
DATASET_DIR = Path.home() / "Downloads" / "Dataset"


def resolve_dir(parent, name):
    """Return parent/name, tolerating trailing/leading whitespace in the name."""
    exact = parent / name
    if exact.exists() or not parent.exists():
        return exact
    for candidate in parent.iterdir():
        if candidate.is_dir() and candidate.name.strip() == name:
            return candidate
    return exact


ZEBRA_DIR = resolve_dir(DATASET_DIR, "Zebra")
NON_ZEBRA_DIR = resolve_dir(DATASET_DIR, "Non Zebra")  # optional comparison
CONCEPT_DIR = resolve_dir(DATASET_DIR, "Stripes")
CONTROL_DIR = resolve_dir(DATASET_DIR, "Non Stripes")

TARGET_SPLIT = "test"  # evaluate only held-out zebra test images
RESULTS_DIR = PROJECT_DIR / "results"

LAYER_NAME = "layer3"
BATCH_SIZE = 8
NUM_RUNS = 30
IMAGES_PER_GROUP = 30
RANDOM_SEED = 42

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ImageFile.LOAD_TRUNCATED_IMAGES = True

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"Device: {DEVICE}")


# ============================================================
# 2. LOAD PRETRAINED RESNET50
# ============================================================

print("Loading pretrained ResNet50...")
WEIGHTS = ResNet50_Weights.DEFAULT
MODEL = resnet50(weights=WEIGHTS).to(DEVICE)
MODEL.eval()

for parameter in MODEL.parameters():
    parameter.requires_grad = False

PREPROCESS = WEIGHTS.transforms()
CLASS_NAMES = WEIGHTS.meta["categories"]
ZEBRA_INDEX = CLASS_NAMES.index("zebra")

print("ResNet50 loaded successfully.")
print("Zebra ImageNet class index:", ZEBRA_INDEX)


# ============================================================
# 3. DATASET HELPERS
# ============================================================

def find_images(folder):
    """Find image files recursively in a folder."""
    if not folder.exists():
        raise FileNotFoundError(
            f"Folder not found: {folder}\n"
            f"Expected dataset directory: {DATASET_DIR}"
        )

    return sorted(
        path for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_image(path):
    """Load and preprocess one image for pretrained ResNet50."""
    with Image.open(path) as image:
        return PREPROCESS(image.convert("RGB"))


def load_image_batch(paths):
    """Load a batch; skip files that cannot be decoded."""
    tensors, valid_paths = [], []

    for path in paths:
        try:
            tensors.append(load_image(path))
            valid_paths.append(path)
        except Exception as error:
            print(f"Skipping unreadable image {path}: {error}")

    if not tensors:
        return None, []

    return torch.stack(tensors), valid_paths


# ============================================================
# 4. EXTRACT SPATIALLY AVERAGED LAYER3 FEATURES
# ============================================================

def extract_features(paths):
    """
    Extract a 1024-D feature vector per image by averaging layer3
    activations over their height and width.
    """
    all_features, valid_paths = [], []
    layer = getattr(MODEL, LAYER_NAME)
    captured = {}

    def hook_fn(module, inputs, output):
        captured["features"] = output.mean(dim=(2, 3)).detach()

    handle = layer.register_forward_hook(hook_fn)

    try:
        for start in range(0, len(paths), BATCH_SIZE):
            batch_paths = paths[start:start + BATCH_SIZE]
            batch, good_paths = load_image_batch(batch_paths)
            if batch is None:
                continue

            captured.clear()
            with torch.no_grad():
                MODEL(batch.to(DEVICE))

            all_features.append(captured["features"].cpu().numpy())
            valid_paths.extend(good_paths)
    finally:
        handle.remove()

    if not all_features:
        raise RuntimeError("No readable images were available for feature extraction.")

    return np.concatenate(all_features, axis=0), valid_paths


# ============================================================
# 5. TRAIN A CONCEPT ACTIVATION VECTOR
# ============================================================

def train_cav(positive_features, negative_features, rng):
    """Fit a linear separator; its normal vector points toward Stripes."""
    n = min(
        IMAGES_PER_GROUP,
        len(positive_features),
        len(negative_features),
    )
    if n < 2:
        raise ValueError("At least two usable images are required in each CAV group.")

    pos_idx = rng.choice(len(positive_features), size=n, replace=False)
    neg_idx = rng.choice(len(negative_features), size=n, replace=False)

    X = np.concatenate(
        [positive_features[pos_idx], negative_features[neg_idx]], axis=0
    )
    y = np.concatenate([np.ones(n, dtype=int), np.zeros(n, dtype=int)])

    order = rng.permutation(len(y))
    X, y = X[order], y[order]

    classifier = LinearSVC(
        C=1.0,
        max_iter=10000,
        random_state=int(rng.integers(0, 2**31 - 1)),
    )
    classifier.fit(X, y)

    cav = classifier.coef_[0].astype(np.float32)
    norm = np.linalg.norm(cav)
    if norm == 0:
        raise RuntimeError("The learned CAV has zero length.")

    cav /= norm
    return cav, float(classifier.score(X, y)), n


# ============================================================
# 6. TCAV DIRECTIONAL DERIVATIVE
# ============================================================


def score_one_image(image_path, cav):
    batch, _ = load_image_batch([image_path])

    if batch is None:
        return None

    # IMPORTANT: enable gradient tracking on the input
    image_tensor = batch.to(DEVICE)
    image_tensor.requires_grad_(True)

    layer = getattr(MODEL, LAYER_NAME)
    captured = {}

    def hook_fn(module, inputs, output):
        captured["activation"] = output

    handle = layer.register_forward_hook(hook_fn)

    try:
        MODEL.zero_grad(set_to_none=True)

        # Do not put this forward pass inside torch.no_grad()
        with torch.enable_grad():
            logits = MODEL(image_tensor)
            activation = captured["activation"]
            zebra_logit = logits[0, ZEBRA_INDEX]

            gradient = torch.autograd.grad(
                outputs=zebra_logit,
                inputs=activation,
                retain_graph=False,
                create_graph=False,
            )[0]

        channel_gradient = gradient.mean(dim=(2, 3))[0]
        channel_gradient = channel_gradient.detach().cpu().numpy()

        directional_derivative = float(np.dot(channel_gradient, cav))

        zebra_probability = float(
            torch.softmax(logits.detach(), dim=1)[0, ZEBRA_INDEX].item()
        )

        return directional_derivative, zebra_probability

    finally:
        handle.remove()



def evaluate_tcav(target_paths, cav):
    """Return TCAV score, derivatives, and zebra probabilities."""
    derivatives, probabilities = [], []

    for path in target_paths:
        result = score_one_image(path, cav)
        if result is None:
            continue
        derivative, probability = result
        derivatives.append(derivative)
        probabilities.append(probability)

    if not derivatives:
        raise RuntimeError("Could not evaluate any zebra test images.")

    derivatives = np.asarray(derivatives, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    score = float(np.mean(derivatives > 0))
    return score, derivatives, probabilities


# ============================================================
# 7. MAIN EXPERIMENT
# ============================================================

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\nChecking dataset paths:")
    for folder in (ZEBRA_DIR, NON_ZEBRA_DIR, CONCEPT_DIR, CONTROL_DIR):
        print(f"  {folder}: {'FOUND' if folder.exists() else 'NOT FOUND'}")

    # Evaluate only held-out zebra test images.
    zebra_paths = find_images(ZEBRA_DIR / TARGET_SPLIT)
    concept_paths = find_images(CONCEPT_DIR)
    control_paths = find_images(CONTROL_DIR)

    print("\nImages found:")
    print(f"  Zebra/{TARGET_SPLIT}: {len(zebra_paths)}")
    print(f"  Stripes: {len(concept_paths)}")
    print(f"  Non Stripes: {len(control_paths)}")

    if not zebra_paths:
        raise ValueError(f"No zebra test images found in {ZEBRA_DIR / TARGET_SPLIT}")
    if len(concept_paths) < 2 or len(control_paths) < 2:
        raise ValueError("Need at least two images in both Stripes and Non Stripes.")

    print("\nExtracting zebra test features...")
    _, zebra_paths = extract_features(zebra_paths)

    print("Extracting stripe concept features...")
    concept_features, concept_paths = extract_features(concept_paths)

    print("Extracting non-stripe control features...")
    control_features, control_paths = extract_features(control_paths)

    print("\nUsable image counts:")
    print(f"  Zebra test: {len(zebra_paths)}")
    print(f"  Stripes: {len(concept_paths)}")
    print(f"  Non Stripes: {len(control_paths)}")

    if min(len(zebra_paths), len(concept_features), len(control_features)) < 2:
        raise ValueError("Too few readable images after loading.")

    # Baseline predictions on zebra test images.
    baseline_probabilities = []
    for path in zebra_paths:
        batch, _ = load_image_batch([path])
        if batch is None:
            continue
        with torch.no_grad():
            logits = MODEL(batch.to(DEVICE))
            baseline_probabilities.append(
                float(torch.softmax(logits, dim=1)[0, ZEBRA_INDEX].item())
            )

    print(
        "\nMean baseline zebra probability: "
        f"{np.mean(baseline_probabilities):.4f}"
    )

    print("\nStarting repeated TCAV runs...")
    rng = np.random.default_rng(RANDOM_SEED)
    records = []

    for run in range(NUM_RUNS):
        print(f"Run {run + 1}/{NUM_RUNS}")

        cav, train_accuracy, sample_count = train_cav(
            concept_features, control_features, rng
        )
        score, derivatives, probabilities = evaluate_tcav(zebra_paths, cav)

        records.append({
            "run": run + 1,
            "method": "LinearSVC",
            "layer": LAYER_NAME,
            "tcav_score": score,
            "cav_training_accuracy": train_accuracy,
            "images_per_cav_group": sample_count,
            "mean_directional_derivative": float(np.mean(derivatives)),
            "mean_zebra_probability": float(np.mean(probabilities)),
            "num_target_images": len(derivatives),
        })

    results_df = pd.DataFrame(records)
    csv_path = RESULTS_DIR / "tcav_results.csv"
    results_df.to_csv(csv_path, index=False)

    scores = results_df["tcav_score"].to_numpy()
    if len(scores) >= 2 and np.std(scores, ddof=1) > 0:
        test = ttest_1samp(scores, popmean=0.5)
        t_statistic = float(test.statistic)
        p_value = float(test.pvalue)
    else:
        t_statistic = None
        p_value = None

    summary = {
        "model": "ResNet50 ImageNet pretrained",
        "target_class": "zebra",
        "target_class_index": ZEBRA_INDEX,
        "concept": "stripes",
        "control_group": "non-stripes",
        "evaluation_split": TARGET_SPLIT,
        "layer": LAYER_NAME,
        "method": "LinearSVC",
        "num_runs": NUM_RUNS,
        "num_target_images": len(zebra_paths),
        "num_concept_images": len(concept_paths),
        "num_control_images": len(control_paths),
        "mean_tcav_score": float(np.mean(scores)),
        "std_tcav_score": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
        "mean_cav_training_accuracy": float(results_df["cav_training_accuracy"].mean()),
        "mean_zebra_probability": float(results_df["mean_zebra_probability"].mean()),
        "t_statistic_vs_0_5": t_statistic,
        "p_value_vs_0_5": p_value,
        "interpretation_note": (
            "TCAV score measures sensitivity of the zebra logit along the learned "
            "stripe direction. It does not establish causality. CAVs are learned "
            "using spatially averaged layer3 activations."
        ),
    }

    json_path = RESULTS_DIR / "tcav_summary.json"
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print("\n========== TCAV SUMMARY ==========")
    print(f"Mean TCAV score: {summary['mean_tcav_score']:.4f}")
    print(f"Standard deviation: {summary['std_tcav_score']:.4f}")
    print(f"Mean CAV training accuracy: {summary['mean_cav_training_accuracy']:.4f}")
    print(f"Mean zebra probability: {summary['mean_zebra_probability']:.4f}")
    print(f"t-statistic vs 0.5: {t_statistic}")
    print(f"p-value vs 0.5: {p_value}")
    print(f"\nCSV saved to: {csv_path}")
    print(f"Summary saved to: {json_path}")


if __name__ == "__main__":
    main()
