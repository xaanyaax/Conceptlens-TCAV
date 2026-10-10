"""

Single-file TCAV experiment: GoogLeNet, zebra class vs. stripe concept.

Dataset location:

~/Downloads/Dataset/

Expected structure:

Dataset/

├── Zebra/

│   ├── training/

│   ├── val/

│   └── test/

├── Non Zebra/

│   ├── training/

│   ├── val/

│   └── test/

├── Stripes/

└── Non Stripes/

GoogLeNet stays frozen. Regularization and validation are applied to

the separate CAV classifier, not to the CNN.

"""

from pathlib import Path

import json

import random

import numpy as np

import pandas as pd

import torch

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from PIL import Image, ImageFile

from scipy.stats import ttest_1samp

from sklearn.metrics import (

    accuracy_score,

    precision_score,

    recall_score,

    f1_score,

    confusion_matrix,

)

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler


from sklearn.svm import LinearSVC

from torchvision.models import googlenet, GoogLeNet_Weights

# ============================================================

# 1. CONFIGURATION -- MATCHING YOUR EXISTING FILE

# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

# IMPORTANT: This matches your pasted working file.

DATASET_DIR = Path.home() / "Downloads" / "Dataset"

def resolve_dir(parent, name):

    """Find a folder while tolerating whitespace in its name."""

    exact = parent / name

    if exact.exists() or not parent.exists():

        return exact

    for candidate in parent.iterdir():

        if candidate.is_dir() and candidate.name.strip() == name:

            return candidate

    return exact

ZEBRA_DIR = resolve_dir(DATASET_DIR, "Zebra")

NON_ZEBRA_DIR = resolve_dir(DATASET_DIR, "Non Zebra")

CONCEPT_DIR = resolve_dir(DATASET_DIR, "Stripes")

CONTROL_DIR = resolve_dir(DATASET_DIR, "Non Stripes")

TARGET_SPLIT = "test"

RESULTS_DIR = PROJECT_DIR / "results"

# GoogLeNet layer; ResNet's "layer3" does not exist here.

LAYER_NAME = "inception4d"

BATCH_SIZE = 8

NUM_RUNS = 500

IMAGES_PER_GROUP = 30

VALIDATION_SIZE = 0.20

# Smaller C means stronger L2 regularization.

CAV_SVM_C = 0.1

RANDOM_SEED = 42

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

ImageFile.LOAD_TRUNCATED_IMAGES = True

random.seed(RANDOM_SEED)

np.random.seed(RANDOM_SEED)

torch.manual_seed(RANDOM_SEED)

DEVICE = torch.device(

    "mps" if torch.backends.mps.is_available() else "cpu"

)

print("Device:", DEVICE)

print("Dataset directory:", DATASET_DIR)

# ============================================================

# 2. LOAD PRETRAINED GOOGLENET

# ============================================================

print("\nLoading ImageNet-pretrained GoogLeNet...")

WEIGHTS = GoogLeNet_Weights.DEFAULT

MODEL = googlenet(

    weights=WEIGHTS,

    aux_logits=True,

).to(DEVICE)

MODEL.eval()

# Freeze every CNN parameter.

for parameter in MODEL.parameters():

    parameter.requires_grad = False

PREPROCESS = WEIGHTS.transforms()

CLASS_NAMES = WEIGHTS.meta["categories"]

ZEBRA_INDEX = CLASS_NAMES.index("zebra")

if not hasattr(MODEL, LAYER_NAME):

    raise ValueError(

        f"GoogLeNet does not contain the layer {LAYER_NAME!r}."

    )

print("GoogLeNet loaded successfully.")

print("Zebra ImageNet class index:", ZEBRA_INDEX)

print("TCAV layer:", LAYER_NAME)

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

        if path.is_file()

        and path.suffix.lower() in IMAGE_EXTENSIONS

    )

def load_image(path):

    """Apply the official GoogLeNet preprocessing transform."""

    with Image.open(path) as image:

        return PREPROCESS(image.convert("RGB"))

def load_image_batch(paths):

    """Load a batch and skip unreadable images."""

    tensors = []

    valid_paths = []

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

# 4. EXTRACT GOOGLENET ACTIVATIONS

# ============================================================

def extract_features(paths):

    """

    Extract one feature vector per image by spatially averaging

    the selected GoogLeNet layer's activation maps.

    """

    all_features = []

    valid_paths = []

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

            all_features.append(

                captured["features"].cpu().numpy()

            )

            valid_paths.extend(good_paths)

    finally:

        handle.remove()

    if not all_features:

        raise RuntimeError(

            "No readable images were available for feature extraction."

        )

    return np.concatenate(all_features, axis=0), valid_paths

# ============================================================

# 5. SPLIT CONCEPT DATA INTO TRAINING AND VALIDATION SETS

# ============================================================

def make_cav_splits(positive_features, negative_features):

    """

    Use 80% of each concept group for training and 20% for validation.

    The zebra test images are not used to train the CAV.

    """

    pos_train, pos_val = train_test_split(

        positive_features,

        test_size=VALIDATION_SIZE,

        random_state=RANDOM_SEED,

    )

    neg_train, neg_val = train_test_split(

        negative_features,

        test_size=VALIDATION_SIZE,

        random_state=RANDOM_SEED,

    )

    X_train = np.concatenate([pos_train, neg_train], axis=0)

    y_train = np.concatenate([

        np.ones(len(pos_train), dtype=int),

        np.zeros(len(neg_train), dtype=int),

    ])

    X_val = np.concatenate([pos_val, neg_val], axis=0)

    y_val = np.concatenate([

        np.ones(len(pos_val), dtype=int),

        np.zeros(len(neg_val), dtype=int),

    ])

    return X_train, y_train, X_val, y_val

# ============================================================

# 6. TRAIN A REGULARIZED CONCEPT ACTIVATION VECTOR

# ============================================================

def train_cav(X_train, y_train, X_val, y_val, rng):

    """

    Train an L2-regularized LinearSVC.

    StandardScaler is fitted only on the sampled training images.

    Validation data is never used to fit the scaler or classifier.

    """

    positive_indices = np.where(y_train == 1)[0]

    negative_indices = np.where(y_train == 0)[0]

    n = min(

        IMAGES_PER_GROUP,

        len(positive_indices),

        len(negative_indices),

    )

    if n < 2:

        raise ValueError(

            "At least two training images per CAV class are required."

        )

    pos_idx = rng.choice(

        positive_indices, size=n, replace=False

    )

    neg_idx = rng.choice(

        negative_indices, size=n, replace=False

    )

    selected = np.concatenate([pos_idx, neg_idx])

    rng.shuffle(selected)

    X_sample = X_train[selected]

    y_sample = y_train[selected]

    # L2 regularization is provided by LinearSVC.

    # Scaling prevents large-valued features from dominating.

    scaler = StandardScaler()

    X_sample_scaled = scaler.fit_transform(X_sample)

    X_val_scaled = scaler.transform(X_val)

    classifier = LinearSVC(

        C=CAV_SVM_C,

        penalty="l2",

        loss="squared_hinge",

        class_weight="balanced",

        max_iter=20000,

        random_state=int(rng.integers(0, 2**31 - 1)),

    )

    classifier.fit(X_sample_scaled, y_sample)

    train_accuracy = float(

        accuracy_score(

            y_sample,

            classifier.predict(X_sample_scaled),

        )

    )

    validation_accuracy = float(

        accuracy_score(

            y_val,

            classifier.predict(X_val_scaled),

        )

    )

    # Convert the learned weight vector back to the original

    # feature coordinate system before using it as the CAV.

    cav = classifier.coef_[0] / scaler.scale_

    cav = cav.astype(np.float32)

    norm = np.linalg.norm(cav)

    if not np.isfinite(norm) or norm == 0:

        raise RuntimeError("The learned CAV has zero or invalid length.")

    cav /= norm

    return cav, train_accuracy, validation_accuracy, n

# ============================================================

# 7. COMPUTE THE TCAV DIRECTIONAL DERIVATIVE

# ============================================================

def score_one_image(image_path, cav=None):

    """

    Compute the directional derivative of the zebra logit

    along the learned stripe concept direction.

    """

    batch, _ = load_image_batch([image_path])

    if batch is None:

        return None

    image_tensor = batch.to(DEVICE)

    # CNN parameters are frozen, but gradients must still flow

    # through the input to calculate activation gradients.

    image_tensor.requires_grad_(True)

    layer = getattr(MODEL, LAYER_NAME)

    captured = {}

    def hook_fn(module, inputs, output):

        captured["activation"] = output

    handle = layer.register_forward_hook(hook_fn)

    try:

        MODEL.zero_grad(set_to_none=True)

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

        # Match the global-average-pooled activation representation.

        channel_gradient = gradient.mean(dim=(2, 3))[0]

        channel_gradient = (

            channel_gradient.detach().cpu().numpy()

        )

        # This gradient vector depends on the image and frozen model, not on
        # the CAV, so it can be cached and reused across all 500 fits.
        directional_derivative = channel_gradient

        zebra_probability = float(

            torch.softmax(

                logits.detach(), dim=1

            )[0, ZEBRA_INDEX].item()

        )

        return directional_derivative, zebra_probability

    finally:

        handle.remove()

def extract_target_gradients(target_paths):
    """Compute target-image gradients once; reuse them for every CAV fit."""
    gradients = []
    probabilities = []
    usable_paths = []

    for index, path in enumerate(target_paths, start=1):
        result = score_one_image(path, cav=None)
        if result is None:
            continue

        gradient_vector, probability = result
        gradients.append(gradient_vector)
        probabilities.append(probability)
        usable_paths.append(path)

        if index % 25 == 0 or index == len(target_paths):
            print(f"  Gradient extraction: {index}/{len(target_paths)} images")

    if not gradients:
        raise RuntimeError("Could not compute gradients for any zebra test images.")

    return (
        np.asarray(gradients, dtype=np.float32),
        np.asarray(probabilities, dtype=float),
        usable_paths,
    )


def evaluate_tcav(target_gradients, probabilities, cav):
    """Compute TCAV score from cached gradients and a fitted CAV."""
    derivatives = np.asarray(target_gradients @ cav, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    score = float(np.mean(derivatives > 0))
    return score, derivatives, probabilities


# ============================================================

# 8. EVALUATE FROZEN GOOGLENET'S ZEBRA CLASSIFICATION

# ============================================================

def evaluate_binary_zebra_detection(zebra_paths, non_zebra_paths):

    """

    Evaluate the frozen ImageNet classifier as a binary detector.

    Predicted zebra = GoogLeNet's top-1 ImageNet class is 'zebra'.

    This does not train a new binary classifier.

    """

    y_true = []

    y_pred = []

    zebra_probabilities = []

    groups = [

        (1, zebra_paths),

        (0, non_zebra_paths),

    ]

    for true_label, paths in groups:

        for start in range(0, len(paths), BATCH_SIZE):

            batch_paths = paths[start:start + BATCH_SIZE]

            batch, good_paths = load_image_batch(batch_paths)

            if batch is None:

                continue

            with torch.no_grad():

                logits = MODEL(batch.to(DEVICE))

                predicted_classes = (

                    logits.argmax(dim=1).cpu().numpy()

                )

                probabilities = (

                    torch.softmax(logits, dim=1)[:, ZEBRA_INDEX]

                    .cpu().numpy()

                )

            y_true.extend([true_label] * len(good_paths))

            y_pred.extend(

                (predicted_classes == ZEBRA_INDEX).astype(int).tolist()

            )

            zebra_probabilities.extend(probabilities.tolist())

    if not y_true or len(set(y_true)) < 2:

        return {

            "note": (

                "Both Zebra/test and Non Zebra/test need readable images "

                "to calculate binary classification metrics."

            )

        }

    return {

        "num_images": len(y_true),

        "accuracy": float(accuracy_score(y_true, y_pred)),

        "precision": float(

            precision_score(y_true, y_pred, zero_division=0)

        ),

        "recall": float(

            recall_score(y_true, y_pred, zero_division=0)

        ),

        "f1_score": float(

            f1_score(y_true, y_pred, zero_division=0)

        ),

        "confusion_matrix": confusion_matrix(

            y_true, y_pred

        ).tolist(),

        "mean_zebra_probability": float(

            np.mean(zebra_probabilities)

        ),

        "prediction_rule": (

            "Predicted zebra if GoogLeNet top-1 ImageNet class is zebra."

        ),

    }

# ============================================================

# 9. SAVE DIAGNOSTIC PLOTS

# ============================================================

def save_plots(results_df):

    # Plot 1: TCAV scores across repeated CAV fits.

    plt.figure(figsize=(9, 5))

    plt.plot(

        results_df["run"],

        results_df["tcav_score"],

        marker="o",

        linewidth=1,

    )

    plt.axhline(

        0.5,

        linestyle="--",

        label="Reference score = 0.5",

    )

    plt.ylim(-0.03, 1.03)

    plt.xlabel("Repeated CAV fit")

    plt.ylabel("TCAV score")

    plt.title(f"GoogLeNet TCAV scores ({LAYER_NAME})")

    plt.legend()

    plt.tight_layout()

    plt.savefig(

        RESULTS_DIR / "tcav_scores.png",

        dpi=160,

    )

    plt.close()

    # Plot 2: training vs. validation accuracy of the CAV.

    plt.figure(figsize=(9, 5))

    plt.plot(

        results_df["run"],

        results_df["cav_training_accuracy"],

        marker="o",

        label="CAV training accuracy",

    )

    plt.plot(

        results_df["run"],

        results_df["cav_validation_accuracy"],

        marker="o",

        label="CAV validation accuracy",

    )

    plt.ylim(0, 1.05)

    plt.xlabel("Repeated CAV fit")

    plt.ylabel("Accuracy")

    plt.title("CAV overfitting diagnostic")

    plt.legend()

    plt.tight_layout()

    plt.savefig(

        RESULTS_DIR / "cav_train_vs_validation.png",

        dpi=160,

    )

    plt.close()

# ============================================================

# 10. MAIN EXPERIMENT

# ============================================================

def main():

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\nChecking dataset paths:")

    for folder in (

        ZEBRA_DIR,

        NON_ZEBRA_DIR,

        CONCEPT_DIR,

        CONTROL_DIR,

    ):

        print(

            f"  {folder}: "

            f"{'FOUND' if folder.exists() else 'NOT FOUND'}"

        )

    zebra_paths = find_images(ZEBRA_DIR / TARGET_SPLIT)

    non_zebra_paths = find_images(

        NON_ZEBRA_DIR / TARGET_SPLIT

    )

    concept_paths = find_images(CONCEPT_DIR)

    control_paths = find_images(CONTROL_DIR)

    print("\nImages found:")

    print(f"  Zebra/{TARGET_SPLIT}: {len(zebra_paths)}")

    print(f"  Non Zebra/{TARGET_SPLIT}: {len(non_zebra_paths)}")

    print(f"  Stripes: {len(concept_paths)}")

    print(f"  Non Stripes: {len(control_paths)}")

    if not zebra_paths:

        raise ValueError(

            f"No zebra test images found in {ZEBRA_DIR / TARGET_SPLIT}"

        )

    if len(concept_paths) < 10 or len(control_paths) < 10:

        raise ValueError(

            "Need at least 10 images in both Stripes and Non Stripes "

            "to create training and validation splits."

        )

    if not non_zebra_paths:

        print(

            "WARNING: No Non Zebra/test images found. "

            "Binary classification metrics will be unavailable."

        )

    print("\nExtracting zebra test features...")

    _, zebra_paths = extract_features(zebra_paths)

    print("Extracting non-zebra test features...")

    if non_zebra_paths:

        _, non_zebra_paths = extract_features(non_zebra_paths)

    print("Extracting stripe concept features...")

    concept_features, concept_paths = extract_features(concept_paths)

    print("Extracting non-stripe control features...")

    control_features, control_paths = extract_features(control_paths)

    print("\nUsable image counts:")

    print(f"  Zebra test: {len(zebra_paths)}")

    print(f"  Non Zebra test: {len(non_zebra_paths)}")

    print(f"  Stripes: {len(concept_paths)}")

    print(f"  Non Stripes: {len(control_paths)}")

    if min(

        len(zebra_paths),

        len(concept_features),

        len(control_features),

    ) < 2:

        raise ValueError(

            "Too few readable images after loading."

        )

    print("\nPrecomputing zebra target gradients once for all 500 CAV fits...")
    target_gradients, target_probabilities, zebra_paths = extract_target_gradients(
        zebra_paths
    )

    print("\nEvaluating frozen GoogLeNet classification...")

    binary_metrics = evaluate_binary_zebra_detection(

        zebra_paths,

        non_zebra_paths,

    )

    print(json.dumps(binary_metrics, indent=4))

    # Split concept/control images BEFORE repeated CAV training.

    X_train, y_train, X_val, y_val = make_cav_splits(

        concept_features,

        control_features,

    )

    print("\nCAV training images:", len(y_train))

    print("CAV validation images:", len(y_val))

    print("CAV regularization: L2")

    print("LinearSVC C:", CAV_SVM_C)

    records = []

    csv_path = RESULTS_DIR / "tcav_results.csv"

    print("\nStarting repeated TCAV runs...")

    print("Checkpointing CSV after every completed fit.")

    for run in range(1, NUM_RUNS + 1):

        print(f"Run {run}/{NUM_RUNS}")

        # Deterministic independent random stream for each run.
        rng = np.random.default_rng(RANDOM_SEED + run)

        cav, train_accuracy, validation_accuracy, sample_count = (

            train_cav(X_train, y_train, X_val, y_val, rng)

        )

        score, derivatives, probabilities = evaluate_tcav(

            target_gradients, target_probabilities, cav

        )

        records.append({

            "run": run,

            "model": "GoogLeNet ImageNet pretrained frozen",

            "layer": LAYER_NAME,

            "method": "StandardScaler + L2 LinearSVC",

            "svm_C": CAV_SVM_C,

            "tcav_score": score,

            "cav_training_accuracy": train_accuracy,

            "cav_validation_accuracy": validation_accuracy,

            "train_validation_gap": (

                train_accuracy - validation_accuracy

            ),

            "images_per_cav_group": sample_count,

            "mean_directional_derivative": float(

                np.mean(derivatives)

            ),

            "mean_zebra_probability": float(

                np.mean(probabilities)

            ),

            "num_target_images": len(derivatives),

        })

        # Save completed fits immediately so progress is not lost on interruption.
        pd.DataFrame(records).to_csv(csv_path, index=False)

    results_df = pd.DataFrame(records)

    results_df.to_csv(csv_path, index=False)

    # Exploratory test against TCAV score 0.5.

    scores = results_df["tcav_score"].to_numpy()

    if len(scores) >= 2 and np.std(scores, ddof=1) > 0:

        test = ttest_1samp(scores, popmean=0.5)

        t_statistic = float(test.statistic)

        p_value = float(test.pvalue)

    else:

        t_statistic = None

        p_value = None

    # Exploratory t-based interval across repeated fits; runs share the same
    # target images and concept split, so this is not a fully independent-sample CI.
    if len(scores) >= 2:
        from scipy.stats import t as student_t
        score_se = float(np.std(scores, ddof=1) / np.sqrt(len(scores)))
        ci_half_width = float(
            student_t.ppf(0.975, df=len(scores) - 1) * score_se
        )
        score_ci_95 = [
            float(np.mean(scores) - ci_half_width),
            float(np.mean(scores) + ci_half_width),
        ]
    else:
        score_ci_95 = None

    summary = {

        "model": "GoogLeNet ImageNet pretrained; frozen",

        "target_class": "zebra",

        "target_class_index": ZEBRA_INDEX,

        "concept": "stripes",

        "control_group": "non-stripes",

        "evaluation_split": TARGET_SPLIT,

        "layer": LAYER_NAME,

        "cav_method": "StandardScaler + L2-regularized LinearSVC",

        "svm_C": CAV_SVM_C,

        "num_runs": NUM_RUNS,

        "num_target_images": len(zebra_paths),

        "num_concept_images": len(concept_paths),

        "num_control_images": len(control_paths),

        "mean_tcav_score": float(np.mean(scores)),

        "std_tcav_score": (

            float(np.std(scores, ddof=1))

            if len(scores) > 1 else 0.0

        ),

        "exploratory_95_percent_t_interval_for_mean_tcav": score_ci_95,

        "mean_cav_training_accuracy": float(

            results_df["cav_training_accuracy"].mean()

        ),

        "mean_cav_validation_accuracy": float(

            results_df["cav_validation_accuracy"].mean()

        ),

        "mean_train_validation_gap": float(

            results_df["train_validation_gap"].mean()

        ),

        "mean_zebra_probability": float(

            results_df["mean_zebra_probability"].mean()

        ),

        "exploratory_t_statistic_vs_0_5": t_statistic,

        "exploratory_p_value_vs_0_5": p_value,

        "frozen_googlenet_binary_detection": binary_metrics,

        "statistical_note": (

            "The t-test over repeated CAV fits is exploratory because "

            "the same target images and fixed concept validation split "

            "are reused. Interpret the p-value cautiously."

        ),

        "interpretation_note": (

            "TCAV score is the fraction of target zebra images with a "

            "positive directional derivative along the learned stripe "

            "CAV. It is not classification accuracy and does not prove "

            "causal reasoning. GoogLeNet remains frozen."

        ),

    }

    json_path = RESULTS_DIR / "tcav_summary.json"

    with open(json_path, "w", encoding="utf-8") as file:

        json.dump(summary, file, indent=4)

    save_plots(results_df)

    print("\n========== TCAV SUMMARY ==========")

    print(f"Mean TCAV score: {summary['mean_tcav_score']:.4f}")

    print(f"TCAV standard deviation: {summary['std_tcav_score']:.4f}")

    print(

        "Mean CAV training accuracy: "

        f"{summary['mean_cav_training_accuracy']:.4f}"

    )

    print(

        "Mean CAV validation accuracy: "

        f"{summary['mean_cav_validation_accuracy']:.4f}"

    )

    print(

        "Mean training-validation gap: "

        f"{summary['mean_train_validation_gap']:.4f}"

    )

    print("Exploratory t-statistic vs 0.5:", t_statistic)

    print("Exploratory p-value vs 0.5:", p_value)

    print("Exploratory 95% t interval for mean TCAV:", score_ci_95)

    print("\nSaved results:")

    print("CSV:", csv_path)

    print("Summary:", json_path)

    print("TCAV plot:", RESULTS_DIR / "tcav_scores.png")

    print(

        "Overfitting plot:",

        RESULTS_DIR / "cav_train_vs_validation.png",

    )

if __name__ == "__main__":

    main()
