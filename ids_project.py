"""
This program is a machine learning based network intrusion detection system.
It uses the CIC-DDoS2019 dataset, which is a larger labelled intrusion detection
dataset.

I selected three attack files from the dataset: Syn, DrDoS_DNS, and TFTP. I
used the BENIGN rows that already exist inside those files. That gives me
four classes in total: BENIGN, Syn, DrDoS_DNS, and TFTP.

The program reads those CSV files, samples a manageable subset, cleans the
data, keeps useful numeric flow features, and converts the labels into class
numbers that the models can understand.

After that, it splits the dataset into training data and testing data. The
training data is used to train the models, and the testing data is used to
check how well they perform on unseen data.

I trained two machine learning models. The first is Logistic Regression, which
is a simple baseline model. The second is a Decision Tree, which is a stronger
rule-based model.

At the end, the program evaluates both models using accuracy, precision,
recall, and F1 score. It saves confusion matrices, a class distribution
graph, a model comparison graph, and a Decision Tree feature importance graph
in the output folder.

I used this version because it follows the assignment option of using an
open source dataset, and it gives more meaningful IDS results than my original
small packet capture files.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "training data"
OUTPUT_DIR = BASE_DIR / "output"

DATASET_FILES = {
    "Syn": DATA_DIR / "Syn.csv",
    "DrDoS_DNS": DATA_DIR / "DrDoS_DNS.csv",
    "TFTP": DATA_DIR / "TFTP.csv",
}

FEATURE_COLUMNS = [
    " Flow Duration",
    " Total Fwd Packets",
    " Total Backward Packets",
    "Total Length of Fwd Packets",
    " Total Length of Bwd Packets",
    "Flow Bytes/s",
    " Flow Packets/s",
    " Fwd Packet Length Mean",
    " Bwd Packet Length Mean",
    " Packet Length Mean",
    " Packet Length Std",
    " SYN Flag Count",
    " ACK Flag Count",
    " RST Flag Count",
    " PSH Flag Count",
    " Average Packet Size",
    " Avg Fwd Segment Size",
    " Avg Bwd Segment Size",
    "Subflow Fwd Packets",
    " Subflow Bwd Packets",
    " act_data_pkt_fwd",
    "Active Mean",
    "Idle Mean",
    " Inbound",
]

LABEL_COLUMN = " Label"
ROWS_PER_CLASS = 5000
CHUNK_SIZE = 100000
CLASS_NAMES = ["BENIGN", "Syn", "DrDoS_DNS", "TFTP"]
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}


def sample_rows_for_label(csv_path: Path, target_label: str, max_rows: int) -> pd.DataFrame:
    parts = []
    rows_collected = 0
    usecols = FEATURE_COLUMNS + [LABEL_COLUMN]

    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=CHUNK_SIZE, low_memory=False):
        label_series = chunk[LABEL_COLUMN].astype(str).str.strip()
        matched = chunk[label_series == target_label]
        if matched.empty:
            continue

        rows_needed = max_rows - rows_collected
        if len(matched) > rows_needed:
            matched = matched.sample(n=rows_needed, random_state=42)

        parts.append(matched.copy())
        rows_collected += len(matched)

        if rows_collected >= max_rows:
            break

    if not parts:
        return pd.DataFrame(columns=usecols)

    return pd.concat(parts, ignore_index=True)


def load_balanced_dataset() -> pd.DataFrame:
    print("Loading CIC-DDoS2019 training subset...")

    frames = []

    benign_frames = []
    benign_target_per_file = 2000
    for attack_name, csv_path in DATASET_FILES.items():
        print(f"Sampling BENIGN rows from {csv_path.name}...")
        benign_part = sample_rows_for_label(csv_path, "BENIGN", benign_target_per_file)
        benign_frames.append(benign_part)

        print(f"Sampling {attack_name} rows from {csv_path.name}...")
        attack_part = sample_rows_for_label(csv_path, attack_name, ROWS_PER_CLASS)
        frames.append(attack_part)

    benign_df = pd.concat(benign_frames, ignore_index=True)
    if len(benign_df) > ROWS_PER_CLASS:
        benign_df = benign_df.sample(n=ROWS_PER_CLASS, random_state=42)

    frames.append(benign_df)
    df = pd.concat(frames, ignore_index=True)
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(str).str.strip()

    print("\nSampled label counts:")
    print(df[LABEL_COLUMN].value_counts())
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    clean_df = df.copy()
    clean_df.columns = [col.strip() for col in clean_df.columns]

    for col in clean_df.columns:
        if col != "Label":
            clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")

    clean_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    clean_df.dropna(inplace=True)
    clean_df["LabelID"] = clean_df["Label"].map(CLASS_TO_ID)
    clean_df.dropna(subset=["LabelID"], inplace=True)
    clean_df["LabelID"] = clean_df["LabelID"].astype(int)
    return clean_df


def save_class_distribution_chart(label_counts: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(label_counts.index.astype(str), label_counts.values, color=["#4C78A8", "#54A24B", "#E45756", "#F58518"])
    ax.set_title("Class Distribution")
    ax.set_ylabel("Number of Flows")
    ax.set_xlabel("Class")
    for bar in bars:
        height = int(bar.get_height())
        ax.text(bar.get_x() + bar.get_width() / 2, height, f"{height}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "class_distribution.png", dpi=200)
    plt.close(fig)


def save_confusion_matrix(cm: np.ndarray, title: str, filename: str, normalize: bool = False) -> None:
    matrix = cm.astype(float)
    value_format = ".0f"
    if normalize:
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        matrix = matrix / row_sums
        value_format = ".3f"

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=CLASS_NAMES)
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format=value_format)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=200)
    plt.close(fig)


def save_feature_importance_chart(model: DecisionTreeClassifier, feature_names: list[str]) -> None:
    importance_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": model.feature_importances_,
        }
    ).sort_values("Importance", ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(importance_df["Feature"], importance_df["Importance"], color="#4C78A8")
    ax.set_title("Decision Tree Feature Importance")
    ax.set_xlabel("Importance")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "decision_tree_feature_importance.png", dpi=200)
    plt.close(fig)


def evaluate_model(name: str, y_true: pd.Series, y_pred: np.ndarray) -> tuple[float, float, float, float]:
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    print(f"\n{name} Results")
    print("-" * 40)
    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1-score :", round(f1, 4))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0))

    return accuracy, precision, recall, f1


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    df = load_balanced_dataset()
    df = clean_dataset(df)

    print(f"\nCleaned dataset shape: {df.shape}")
    label_counts = df["Label"].value_counts().reindex(CLASS_NAMES)
    print("\nCleaned label counts:")
    print(label_counts)
    save_class_distribution_chart(label_counts)

    feature_names = [col.strip() for col in FEATURE_COLUMNS]
    X = df[feature_names]
    y = df["LabelID"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )

    print(f"\nTrain shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")

    lr_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=4000, multi_class="auto"),
    )
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)

    dt_model = DecisionTreeClassifier(random_state=42, max_depth=8)
    dt_model.fit(X_train, y_train)
    dt_pred = dt_model.predict(X_test)

    lr_metrics = evaluate_model("Logistic Regression", y_test, lr_pred)
    dt_metrics = evaluate_model("Decision Tree", y_test, dt_pred)

    cm_lr = confusion_matrix(y_test, lr_pred, labels=list(range(len(CLASS_NAMES))))
    cm_dt = confusion_matrix(y_test, dt_pred, labels=list(range(len(CLASS_NAMES))))

    save_confusion_matrix(cm_lr, "Confusion Matrix - Logistic Regression", "confusion_matrix_logistic_regression.png")
    save_confusion_matrix(cm_dt, "Confusion Matrix - Decision Tree", "confusion_matrix_decision_tree.png")
    save_confusion_matrix(
        cm_lr,
        "Normalized Confusion Matrix - Logistic Regression",
        "normalized_confusion_matrix_logistic_regression.png",
        normalize=True,
    )
    save_confusion_matrix(
        cm_dt,
        "Normalized Confusion Matrix - Decision Tree",
        "normalized_confusion_matrix_decision_tree.png",
        normalize=True,
    )
    save_feature_importance_chart(dt_model, feature_names)

    results_df = pd.DataFrame(
        {
            "Model": ["Logistic Regression", "Decision Tree"],
            "Accuracy": [lr_metrics[0], dt_metrics[0]],
            "Precision": [lr_metrics[1], dt_metrics[1]],
            "Recall": [lr_metrics[2], dt_metrics[2]],
            "F1 Score": [lr_metrics[3], dt_metrics[3]],
        }
    )

    print("\nModel comparison:")
    print(results_df)

    ax = results_df.plot(
        x="Model",
        y=["Accuracy", "Precision", "Recall", "F1 Score"],
        kind="bar",
        figsize=(8, 5),
    )
    ax.set_title("Model Performance Comparison")
    ax.set_ylabel("Score")
    ax.tick_params(axis="x", rotation=0)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "model_performance_comparison.png", dpi=200)
    plt.close()

    labelled_output = OUTPUT_DIR / "labelled_ids_dataset.csv"
    results_output = OUTPUT_DIR / "model_results_summary.csv"
    df.to_csv(labelled_output, index=False)
    results_df.to_csv(results_output, index=False)

    print(f"\nSaved labelled dataset as {labelled_output}")
    print(f"Saved model summary as {results_output}")
    print(f"Saved graphs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
