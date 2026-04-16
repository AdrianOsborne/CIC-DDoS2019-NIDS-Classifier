<h3 align="center">CIC-DDoS2019 NIDS Classifier</h3>
<p align="center">Machine learning based network intrusion detection for a university cybersecurity assignment</p>
<p align="center">
  CIC-DDoS2019 subset | Multiclass traffic classification | Logistic Regression and Decision Tree
</p>

---

## Project Overview

This project is a machine learning based Network Intrusion Detection System (NIDS) that classifies network traffic flows using a subset of the **CIC-DDoS2019** dataset.

The current version uses four classes:
- `BENIGN`
- `Syn`
- `DrDoS_DNS`
- `TFTP`

## How it works

**Dataset selection** -> the script reads selected CIC-DDoS2019 CSV files -> samples a manageable subset of benign and attack flows -> keeps useful numeric flow features.

**Training** -> the dataset is cleaned -> split into training and testing data -> two machine learning models are trained:
- Logistic Regression
- Decision Tree

**Evaluation** -> both models are measured using accuracy, precision, recall, and F1 score -> confusion matrices, a class distribution chart, a feature importance chart, and a model comparison chart are saved in the `output` folder.

## Dataset

This repository does **not** include the full training dataset because the source files are too large for GitHub.

Dataset source used for this project:
- [Kaggle mirror](https://www.kaggle.com/datasets/rodrigorosasilva/cic-ddos2019-30gb-full-dataset-csv-files)

Original dataset source:
- [CIC-DDoS2019 by the Canadian Institute for Cybersecurity (CIC)](https://www.unb.ca/cic/datasets/ddos-2019.html)

After downloading the dataset, place these files inside `training data/`:
- `Syn.csv`
- `DrDoS_DNS.csv`
- `TFTP.csv`

## Stack

- **Language**: Python
- **Data processing**: pandas, numpy
- **Machine learning**: scikit-learn
- **Visualisation**: matplotlib
- **Dataset**: CIC-DDoS2019 flow based CSV files

## Results

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.8368 | 0.8975 | 0.8370 | 0.8207 |
| Decision Tree | 0.8395 | 0.9004 | 0.8396 | 0.8234 |

The Decision Tree performed slightly better overall, while both models showed that some attack classes were easier to detect than others.

## Output Visuals

### Class Distribution

![Class Distribution](output/class_distribution.png)

### Logistic Regression Confusion Matrix

![Logistic Regression Confusion Matrix](output/confusion_matrix_logistic_regression.png)

### Decision Tree Confusion Matrix

![Decision Tree Confusion Matrix](output/confusion_matrix_decision_tree.png)

### Normalized Logistic Regression Confusion Matrix

![Normalized Logistic Regression Confusion Matrix](output/normalized_confusion_matrix_logistic_regression.png)

### Normalized Decision Tree Confusion Matrix

![Normalized Decision Tree Confusion Matrix](output/normalized_confusion_matrix_decision_tree.png)

### Decision Tree Feature Importance

![Decision Tree Feature Importance](output/decision_tree_feature_importance.png)

### Model Performance Comparison

![Model Performance Comparison](output/model_performance_comparison.png)

## Project Structure

```text
.
|-- ids_project.py
|-- README.md
|-- training data/
|   |-- Syn.csv
|   |-- DrDoS_DNS.csv
|   `-- TFTP.csv
`-- output/
    |-- class_distribution.png
    |-- confusion_matrix_decision_tree.png
    |-- confusion_matrix_logistic_regression.png
    |-- decision_tree_feature_importance.png
    |-- labelled_ids_dataset.csv
    |-- model_performance_comparison.png
    |-- model_results_summary.csv
    |-- normalized_confusion_matrix_decision_tree.png
    `-- normalized_confusion_matrix_logistic_regression.png
```

## Running the project

1. Install the required Python packages:

```bash
pip install pandas numpy scikit-learn matplotlib
```

2. Download the required dataset files and place them in `training data/`.

3. Run the script from the project root:

```bash
python ids_project.py
```

4. Check the generated graphs and CSV files in the `output` folder.

## Notes

- This project follows the assignment option of using an **open source dataset**
- The script uses a sampled subset of the full CIC-DDoS2019 files to make local training practical
- The output folder contains report ready graphs and result tables

## Author

Adrian  
University cybersecurity assignment project
