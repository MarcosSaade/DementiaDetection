import argparse
import os

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Load and preprocess data
def load_and_preprocess_data(filepath):
    features = pd.read_csv(filepath)

    # Preserve the original class labels
    original_class_labels = features['class_label'].copy()

    # Map 'class_label' to numerical values
    def map_class_label(x):
        if x == 'AD':
            return 1
        elif x in ['HC', 'MCI']:
            return 0
        else:
            return np.nan

    features['mapped_class_label'] = features['class_label'].apply(map_class_label)

    # Drop rows with NaN in 'mapped_class_label'
    features = features.dropna(subset=['mapped_class_label'])
    features['mapped_class_label'] = features['mapped_class_label'].astype(int)

    # Encode 'gender'
    features['gender'] = features['gender'].map({'M': 0, 'W': 1})
    # Drop rows with NaN in 'gender'
    features = features.dropna(subset=['gender'])
    features['gender'] = features['gender'].astype(int)

    # Extract 'id' before dropping it
    ids = features['id'].reset_index(drop=True)

    # Extract original class labels after dropping rows
    original_class_labels = original_class_labels.loc[features.index].reset_index(drop=True)

    # Define feature matrix X and target vector y
    X = features.drop(['filename', 'class_label', 'mapped_class_label', 'id'], axis=1)
    y = features['mapped_class_label']

    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # Replace infinite values and fill any resulting NaNs
    X = X.replace([np.inf, -np.inf], np.nan)
    X.fillna(X.median(), inplace=True)

    # Hardcode the best features
    best_features = ['F2_range', 'spectral_centroid', 'mfcc_2_mean', 'mfcc_3_mean',
                    'mfcc_5_mean', 'mfcc_6_mean', 'mfcc_7_mean', 'mfcc_7_std',
                    'mfcc_8_mean', 'mfcc_11_mean', 'hnr_mean', 'HFD_min', 'total_duration',
                    'total_speech_duration', 'speech_duration_coefficient_of_variation']
    X = X[best_features]

    return X, y, ids, original_class_labels

# Function to plot ROC Curve
def plot_roc_curve(model, X, y, graphs_path):
    y_scores = model.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.01])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    os.makedirs(graphs_path, exist_ok=True)
    out = os.path.join(graphs_path, "roc_curve.png")
    plt.savefig(out)
    plt.close()
    print(f"Saved: {out}")

# Function to plot Precision-Recall Curve
def plot_precision_recall_curve_func(model, X, y, graphs_path):
    y_scores = model.predict_proba(X)[:, 1]
    precision, recall, _ = precision_recall_curve(y, y_scores)
    avg_precision = average_precision_score(y, y_scores)

    plt.figure()
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR (AP = {avg_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="upper right")
    os.makedirs(graphs_path, exist_ok=True)
    out = os.path.join(graphs_path, "precision_recall_curve.png")
    plt.savefig(out)
    plt.close()
    print(f"Saved: {out}")

# Function to plot Learning Curve
def plot_learning_curve_func(pipeline, X, y, graphs_path):
    plt.figure()
    train_sizes, train_scores, test_scores = learning_curve(
        pipeline, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10), scoring='f1_weighted'
    )
    train_err = 1 - np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_err = 1 - np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.plot(train_sizes, train_err, 'o-', color="r", label="Training error")
    plt.plot(train_sizes, test_err, 'o-', color="g", label="CV error")
    plt.fill_between(train_sizes, train_err - train_std, train_err + train_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_err - test_std, test_err + test_std, alpha=0.1, color="g")

    plt.title("Learning Curve (Error)")
    plt.xlabel("Training Examples")
    plt.ylabel("Error")
    plt.legend(loc="best")
    plt.grid()
    os.makedirs(graphs_path, exist_ok=True)
    out = os.path.join(graphs_path, "learning_curve.png")
    plt.savefig(out)
    plt.close()
    print(f"Saved: {out}")

# Function to plot Feature Importances using Permutation Importance
def plot_feature_importance(pipeline, X, y, graphs_path):
    # Perform permutation importance
    result = permutation_importance(
        pipeline, X, y, n_repeats=10, random_state=42, n_jobs=-1, scoring='f1_weighted'
    )
    importance = result.importances_mean
    std = result.importances_std
    feature_names = X.columns

    # Create a DataFrame for visualization
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance,
        'Std': std
    })

    # Sort the DataFrame by importance
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    # Plot the feature importances
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'], importance_df['Importance'], xerr=importance_df['Std'], align='center')
    plt.gca().invert_yaxis()  # Invert y-axis to have the most important feature at the top
    plt.xlabel('Permutation Importance')
    plt.title('Feature Importances (Permutation)')
    plt.tight_layout()
    os.makedirs(graphs_path, exist_ok=True)
    out = os.path.join(graphs_path, "feature_importance.png")
    plt.savefig(out)
    plt.close()
    print(f"Saved: {out}")

# Function to train SVM model and generate reports
def train_svm_model(X, y, ids, original_class_labels, skf, graphs_path):
    print("Training SVM with fixed hyperparameters...\n")

    # Define the pipeline with SMOTEENN, scaling, and SVM classifier
    pipeline = ImbPipeline([
        ('smoteenn', SMOTEENN(random_state=42)),
        ('scaler', StandardScaler()),
        ('svm', SVC(C=10, gamma='scale', kernel='rbf', probability=True, random_state=42))
    ])

    # Evaluate the estimator using cross-validation
    print("Cross-validating...\n")
    scoring_metrics = ['accuracy', 'f1_weighted', 'recall_weighted', 'precision_weighted']
    cv_results = cross_validate(pipeline, X, y, cv=skf, scoring=scoring_metrics, n_jobs=-1, return_train_score=False)

    print("CV metrics:")
    for metric in scoring_metrics:
        scores = cv_results[f'test_{metric}']
        print(f"  {metric}: {scores.mean():.4f} ± {scores.std():.4f}")
    print()

    # Fit the pipeline on the entire dataset
    print("Fitting on full data...\n")
    pipeline.fit(X, y)
    y_pred = pipeline.predict(X)

    # Detailed classification report on the entire dataset
    print("Classification report (full data):\n")
    print(classification_report(y, y_pred, target_names=['non-AD', 'AD']))

    cm = confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['non-AD', 'AD'])
    disp.plot(cmap=plt.cm.Blues)
    os.makedirs(graphs_path, exist_ok=True)
    out_cm = os.path.join(graphs_path, "confusion_matrix.png")
    plt.title("Confusion Matrix")
    plt.savefig(out_cm)
    plt.close()
    print(f"Saved: {out_cm}")

    # Plot ROC Curve
    plot_roc_curve(pipeline, X, y, graphs_path)

    # Plot Precision-Recall Curve
    plot_precision_recall_curve_func(pipeline, X, y, graphs_path)

    # Plot Learning Curve
    plot_learning_curve_func(pipeline, X, y, graphs_path)

    # Plot Feature Importances
    plot_feature_importance(pipeline, X, y, graphs_path)

    # Identify misclassified samples
    mis_idx = np.where(y_pred != y)[0]
    mis_ids = ids.iloc[mis_idx].reset_index(drop=True)
    mis_true = original_class_labels.iloc[mis_idx].reset_index(drop=True)

    # Categorize misclassifications based on true labels
    mis_df = pd.DataFrame({
        'ID': mis_ids,
        'True_Label': mis_true
    })

    hc = mis_df[mis_df['True_Label'] == 'HC']
    mci = mis_df[mis_df['True_Label'] == 'MCI']
    ad = mis_df[mis_df['True_Label'] == 'AD']

    print("\nMisclassified Sample IDs (by true label):")
    for label, df in [('HC', hc), ('MCI', mci), ('AD', ad)]:
        if df.empty:
            print(f"  No misclassifications for {label}")
        else:
            print(f"  {label}:\n{df['ID'].to_string(index=False)}")

    out_hc = os.path.join(graphs_path, "misclassified_ids_HC.txt")
    out_mci = os.path.join(graphs_path, "misclassified_ids_MCI.txt")
    out_ad = os.path.join(graphs_path, "misclassified_ids_AD.txt")
    hc.to_csv(out_hc, index=False, header=True)
    mci.to_csv(out_mci, index=False, header=True)
    ad.to_csv(out_ad, index=False, header=True)
    print(f"Saved: {out_hc}\nSaved: {out_mci}\nSaved: {out_ad}")

    return pipeline

def _build_arg_parser():
    p = argparse.ArgumentParser(description='SVM pipeline with misclassification reporting')
    p.add_argument('--features', '-f', required=True, help='Path to features CSV')
    p.add_argument('--graphs', '-g', required=True, help='Directory to save plots')
    return p

# Main function
def main():
    args = _build_arg_parser().parse_args()

    X, y, ids, original_class_labels = load_and_preprocess_data(args.features)

    print("Class distribution:\n", y.value_counts(), "\n")
    min_class = y.value_counts().min()
    n_splits = min(5, max(2, int(min_class)))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    best_model = train_svm_model(X, y, ids, original_class_labels, skf, args.graphs)

    model_path = os.path.join(args.graphs, 'best_svm_model.joblib')
    joblib.dump(best_model, model_path)
    print(f"Saved model: {model_path}")

if __name__ == '__main__':
    main()
