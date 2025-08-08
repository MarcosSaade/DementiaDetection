import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def load_data(file_path):
    return pd.read_csv(file_path)


def preprocess_data(data):
    drop_cols = [c for c in ['filename', 'id', 'age', 'gender'] if c in data.columns]
    return data.drop(drop_cols, axis=1)


def plot_class_distribution(data, class_column, output_path):
    plt.figure(figsize=(10, 6))
    sns.countplot(x=class_column, data=data, palette='muted')
    plt.title('Class Distribution')
    plt.xlabel('Class Label')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    os.makedirs(output_path, exist_ok=True)
    plt.savefig(os.path.join(output_path, 'class_distribution.png'))
    plt.close()


def plot_feature_distributions(features, feature_columns, class_column, output_path):
    sns.set(style="whitegrid")
    num_features = len(feature_columns)
    ncols = 3
    nrows = (num_features + ncols - 1) // ncols if num_features else 1
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 5 * nrows))
    axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    for i, feature in enumerate(feature_columns):
        for label in features[class_column].dropna().unique():
            subset = features[features[class_column] == label]
            try:
                sns.kdeplot(
                    data=subset, x=feature, ax=axes[i], fill=True, alpha=0.4,
                    label=f'{class_column} {label}', linewidth=2
                )
            except Exception:
                # fallback to histogram for non-continuous or problematic columns
                subset[feature].hist(ax=axes[i], alpha=0.5, label=f'{class_column} {label}')
        axes[i].set_title(f'Distribution of {feature}')
        axes[i].legend(title=class_column)
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel('Density')

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'feature_distributions.png'))
    plt.close()


# CLI -------------------------------------------------------------------------

def _build_arg_parser():
    p = argparse.ArgumentParser(description='Dataset visualization utilities')
    p.add_argument('--input', '-i', required=True, help='CSV with features')
    p.add_argument('--output', '-o', required=True, help='Destination folder for plots')
    return p


def main():
    args = _build_arg_parser().parse_args()

    os.makedirs(args.output, exist_ok=True)
    data = load_data(args.input)

    class_column = 'class_label'
    if class_column not in data.columns:
        raise ValueError(f"'{class_column}' column not found in {args.input}")

    sns.set_theme(style='whitegrid')
    plot_class_distribution(data, class_column, args.output)

    features = preprocess_data(data)
    feature_columns = [c for c in features.columns if c != class_column]
    plot_feature_distributions(features, feature_columns, class_column, args.output)


if __name__ == "__main__":
    main()