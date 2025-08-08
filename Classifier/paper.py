import argparse
import os

import joblib
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
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
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# Load and preprocess data
def load_and_preprocess_data(filepath):
    features = pd.read_csv(filepath)

    # Map 'class_label' to numerical values
    def map_class_label(x):
        if x == 'AD':
            return 1
        elif x in ['HC', 'MCI']:
            return 0
        else:
            return np.nan

    features['class_label'] = features['class_label'].apply(map_class_label)

    # Drop rows with NaN in 'class_label'
    features = features.dropna(subset=['class_label'])
    features['class_label'] = features['class_label'].astype(int)

    # Encode 'gender'
    features['gender'] = features['gender'].map({'M': 0, 'W': 1})
    # Drop rows with NaN in 'gender'
    features = features.dropna(subset=['gender'])
    features['gender'] = features['gender'].astype(int)

    # Define feature matrix X and target vector y
    X = features.drop(['filename', 'class_label', 'id'], axis=1)
    y = features['class_label']

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

    return X, y

# Function to plot ROC Curve
def plot_roc_curve(model, X, y, graphs_path, model_name):
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
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    os.makedirs(graphs_path, exist_ok=True)
    out = os.path.join(graphs_path, f"roc_curve_{model_name}.png")
    plt.savefig(out)
    plt.close()
    print(f"Saved: {out}")

# Function to plot Precision-Recall Curve
def plot_precision_recall_curve_func(model, X, y, graphs_path, model_name):
    y_scores = model.predict_proba(X)[:, 1]
    precision, recall, _ = precision_recall_curve(y, y_scores)
    from sklearn.metrics import average_precision_score
    ap = average_precision_score(y, y_scores)

    plt.figure()
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR (AP = {ap:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.legend(loc="upper right")
    os.makedirs(graphs_path, exist_ok=True)
    out = os.path.join(graphs_path, f"precision_recall_curve_{model_name}.png")
    plt.savefig(out)
    plt.close()
    print(f"Saved: {out}")

# Function to plot Learning Curve
def plot_learning_curve_func(pipeline, X, y, graphs_path, model_name):
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

    plt.title(f"Learning Curve (Error) - {model_name}")
    plt.xlabel("Training Examples")
    plt.ylabel("Error")
    plt.legend(loc="best")
    plt.grid()
    os.makedirs(graphs_path, exist_ok=True)
    out = os.path.join(graphs_path, f"learning_curve_{model_name}.png")
    plt.savefig(out)
    plt.close()
    print(f"Saved: {out}")

# Function to plot Feature Importances using Permutation Importance
def plot_feature_importance(pipeline, X, y, graphs_path, model_name):
    # Perform permutation importance
    result = permutation_importance(
        pipeline, X, y, n_repeats=10, random_state=42, n_jobs=-1, scoring='f1_weighted'
    )
    importance = result.importances_mean
    std = result.importances_std

    # Get feature names
    feature_names = X.columns
    import pandas as pd
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance, 'Std': std}).sort_values(by='Importance', ascending=False)

    # Plot the feature importances
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'], importance_df['Importance'], xerr=importance_df['Std'], align='center')
    plt.gca().invert_yaxis()  # Invert y-axis to have the most important feature at the top
    plt.xlabel('Permutation Importance')
    plt.title(f'Feature Importances (Permutation) - {model_name}')
    plt.tight_layout()
    os.makedirs(graphs_path, exist_ok=True)
    out = os.path.join(graphs_path, f"feature_importance_{model_name}.png")
    plt.savefig(out)
    plt.close()
    print(f"Saved: {out}")

# Function to plot confusion matrix
def plot_confusion_matrix(model, X, y, graphs_path, model_name):
    y_pred = model.predict(X)
    cm = confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['non-AD', 'AD'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - {model_name}")
    os.makedirs(graphs_path, exist_ok=True)
    out = os.path.join(graphs_path, f"confusion_matrix_{model_name}.png")
    plt.savefig(out)
    plt.close()
    print(f"Saved: {out}")

# Function to train SVM model and generate reports
def train_svm_model(X, y, skf, graphs_path):
    print("Training SVM...")
    pipeline = ImbPipeline([
        ('smoteenn', SMOTEENN(random_state=42)),
        ('scaler', StandardScaler()),
        ('svm', SVC(C=10, gamma='scale', kernel='rbf', probability=True, random_state=42))
    ])

    scoring = ['accuracy', 'f1_weighted', 'recall_weighted', 'precision_weighted']
    cv = cross_validate(pipeline, X, y, cv=skf, scoring=scoring, n_jobs=-1, return_train_score=False)

    print("SVM CV:")
    for m in scoring:
        print(f"  {m}: {cv[f'test_{m}'].mean():.4f} ± {cv[f'test_{m}'].std():.4f}")

    pipeline.fit(X, y)
    plot_confusion_matrix(pipeline, X, y, graphs_path, "SVM")
    plot_roc_curve(pipeline, X, y, graphs_path, "SVM")
    plot_precision_recall_curve_func(pipeline, X, y, graphs_path, "SVM")
    plot_learning_curve_func(pipeline, X, y, graphs_path, "SVM")
    plot_feature_importance(pipeline, X, y, graphs_path, "SVM")

    model_path = os.path.join(graphs_path, 'best_svm_model.joblib')
    joblib.dump(pipeline, model_path)
    print(f"Saved model: {model_path}\n")

    return {
        'model_name': 'SVM',
        'best_estimator': pipeline,
        'cv_accuracy': cv['test_accuracy'].mean(),
        'cv_f1': cv['test_f1_weighted'].mean(),
        'best_params': {
            'C': 10,
            'gamma': 'scale',
            'kernel': 'rbf',
            'probability': True,
            'random_state': 42
        }
    }

# Function to train ANN and XGBoost models with hardcoded hyperparameters
def train_classifier(model_name, classifier, X, y, skf, graphs_path, hyperparams):
    print(f"Training {model_name}...")
    pipeline = ImbPipeline([
        ('smoteenn', SMOTEENN(random_state=42)),
        ('scaler', StandardScaler()),
        ('classifier', classifier)
    ])
    pipeline.set_params(**hyperparams)

    scoring = ['accuracy', 'f1_weighted', 'recall_weighted', 'precision_weighted']
    cv = cross_validate(pipeline, X, y, cv=skf, scoring=scoring, n_jobs=-1, return_train_score=False)

    print(f"{model_name} CV:")
    for m in scoring:
        print(f"  {m}: {cv[f'test_{m}'].mean():.4f} ± {cv[f'test_{m}'].std():.4f}")

    pipeline.fit(X, y)
    plot_confusion_matrix(pipeline, X, y, graphs_path, model_name)
    plot_roc_curve(pipeline, X, y, graphs_path, model_name)
    plot_precision_recall_curve_func(pipeline, X, y, graphs_path, model_name)
    plot_learning_curve_func(pipeline, X, y, graphs_path, model_name)
    plot_feature_importance(pipeline, X, y, graphs_path, model_name)

    model_path = os.path.join(graphs_path, f"best_{model_name}_model.joblib")
    joblib.dump(pipeline, model_path)
    print(f"Saved model: {model_path}\n")

    return {
        'model_name': model_name,
        'best_estimator': pipeline,
        'cv_accuracy': cv['test_accuracy'].mean(),
        'cv_f1': cv['test_f1_weighted'].mean(),
        'best_params': hyperparams
    }

# Function to plot bar graph comparing models
def plot_model_comparison(results, graphs_path):
    names = [r['model_name'] for r in results]
    acc = [r['cv_accuracy'] for r in results]
    f1 = [r['cv_f1'] for r in results]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    r1 = ax.bar(x - width/2, acc, width, label='Accuracy', color='royalblue')
    r2 = ax.bar(x + width/2, f1, width, label='F1 Score', color='darkorange')

    ax.set_ylim(0.6, 0.9)
    ax.set_ylabel('Scores')
    ax.set_title('Model Comparison: Accuracy and F1 Score')
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend()

    def autolabel(rects):
        for rect in rects:
            h = rect.get_height()
            ax.annotate(f'{h:.2f}', xy=(rect.get_x() + rect.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom')
    autolabel(r1)
    autolabel(r2)

    fig.tight_layout()
    os.makedirs(graphs_path, exist_ok=True)
    out = os.path.join(graphs_path, "model_comparison.png")
    plt.savefig(out)
    plt.close()
    print(f"Saved: {out}")

def _build_arg_parser():
    p = argparse.ArgumentParser(description='Train SVM/ANN/XGBoost and generate plots')
    p.add_argument('--features', '-f', required=True, help='Path to features CSV')
    p.add_argument('--graphs', '-g', required=True, help='Directory to save plots')
    return p

# Main function
def main():
    args = _build_arg_parser().parse_args()

    os.makedirs(args.graphs, exist_ok=True)

    X, y = load_and_preprocess_data(args.features)

    print("Class distribution:\n", y.value_counts(), "\n")
    min_class = y.value_counts().min()
    n_splits = min(5, max(2, int(min_class)))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    svm_result = train_svm_model(X, y, skf, args.graphs)

    ann_hyperparams = {
        'classifier__activation': 'relu',
        'classifier__alpha': 0.0001,
        'classifier__hidden_layer_sizes': (100,),
        'classifier__learning_rate': 'constant',
        'classifier__solver': 'adam'
    }
    xgb_hyperparams = {
        'classifier__colsample_bytree': 0.8,
        'classifier__learning_rate': 0.1,
        'classifier__max_depth': 7,
        'classifier__n_estimators': 100,
        'classifier__subsample': 0.8
    }

    ann_classifier = MLPClassifier(
        activation='relu', alpha=0.0001, hidden_layer_sizes=(100,), learning_rate='constant',
        solver='adam', random_state=42, max_iter=1000
    )
    xgb_classifier = XGBClassifier(
        colsample_bytree=0.8, learning_rate=0.1, max_depth=7, n_estimators=100, subsample=0.8,
        random_state=42, eval_metric='logloss', verbosity=0
    )

    ann_result = train_classifier('ANN', ann_classifier, X, y, skf, args.graphs, ann_hyperparams)
    xgb_result = train_classifier('XGBoost', xgb_classifier, X, y, skf, args.graphs, xgb_hyperparams)

    results = [svm_result, ann_result, xgb_result]
    plot_model_comparison(results, args.graphs)

    print("\n=== Best Hyperparameters ===")
    for r in results:
        print(f"\nModel: {r['model_name']}")
        for k, v in r['best_params'].items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
