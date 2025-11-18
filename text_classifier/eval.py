
import os
import argparse as ap
import numpy as np
import pandas as pd
from sklearn.metrics import multilabel_confusion_matrix, classification_report


def parse_args():
    parser = ap.ArgumentParser()
    parser.add_argument('--pred', required=True, help='Path to inferred predictions CSV (from infer.py)')
    parser.add_argument('--gt', required=False, 
                            default='/home/muhammedg/fvlm/data/multi_abnormality_labels/valid_predicted_labels.csv', 
                            help='Path to ground-truth CSV with labels')
    parser.add_argument('--save_dir', required=True, help='Directory to save evaluation outputs')
    parser.add_argument('--id_col', default='AccessionNo', help='Identifier column to align rows')
    parser.add_argument(
        '--labels',
        default=None,
        help='Comma-separated label column names. If not provided, uses all columns in pred excluding id and report_text.'
    )
    return parser.parse_args()


def ensure_dir(path: str) -> str:
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass
    return path


def main():
    args = parse_args()
    ensure_dir(args.save_dir)

    pred_df = pd.read_csv(args.pred)
    gt_df = pd.read_csv(args.gt)

    if args.labels is None:
        labels = [c for c in pred_df.columns if c not in [args.id_col, 'report_text']]
    else:
        labels = [c.strip() for c in args.labels.split(',') if c.strip()]

    # Align by common IDs
    pred_ids = set(pred_df[args.id_col].astype(str))
    gt_ids = set(gt_df[args.id_col].astype(str))
    common_ids = sorted(list(pred_ids.intersection(gt_ids)))
    if len(common_ids) == 0:
        raise ValueError('No overlapping IDs between predictions and ground truth')

    pred_aligned = pred_df[pred_df[args.id_col].astype(str).isin(common_ids)].copy()
    gt_aligned = gt_df[gt_df[args.id_col].astype(str).isin(common_ids)].copy()

    # Sort to the same order
    pred_aligned.sort_values(by=args.id_col, inplace=True)
    gt_aligned.sort_values(by=args.id_col, inplace=True)

    # Ensure label columns exist on both sides
    for col in labels:
        if col not in pred_aligned.columns:
            raise ValueError(f'Missing label in predictions: {col}')
        if col not in gt_aligned.columns:
            raise ValueError(f'Missing label in ground truth: {col}')

    y_true = gt_aligned[labels].astype(int).values
    y_pred = pred_aligned[labels].astype(float).round().astype(int).values

    cm = multilabel_confusion_matrix(y_true, y_pred)
    clf = classification_report(y_true, y_pred, target_names=labels)

    # Save classification report
    with open(os.path.join(args.save_dir, 'classification_report.txt'), 'w') as f:
        f.write(clf)

    # Manual per-class metrics
    precision_list = []
    recall_list = []
    f1_list = []
    support_list = []

    for matrix in cm:
        TN, FP, FN, TP = matrix.ravel()
        precision = TP / (TP + FP) if (TP + FP) != 0 else 0
        recall = TP / (TP + FN) if (TP + FN) != 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
        support = TP + FN

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        support_list.append(support)

    total_support = np.sum(support_list)
    weighted_precision = (
        np.sum([precision * support for precision, support in zip(precision_list, support_list)]) / total_support
        if total_support > 0 else 0
    )
    weighted_recall = (
        np.sum([recall * support for recall, support in zip(recall_list, support_list)]) / total_support
        if total_support > 0 else 0
    )
    weighted_f1 = (
        np.sum([f1 * support for f1, support in zip(f1_list, support_list)]) / total_support
        if total_support > 0 else 0
    )

    metrics_df = pd.DataFrame({
        'Label': labels,
        'Precision': precision_list,
        'Recall': recall_list,
        'F1': f1_list,
        'Support': support_list
    })

    metrics_df.loc[len(metrics_df.index)] = [
        'Weighted Average', weighted_precision, weighted_recall, weighted_f1, total_support
    ]

    metrics_df.to_csv(os.path.join(args.save_dir, 'metrics.csv'), index=False)

    # Save aligned comparison for auditing
    compare_df = gt_aligned[[args.id_col] + labels].copy()
    compare_df.columns = [args.id_col] + [f'{c}_gt' for c in labels]
    for c in labels:
        compare_df[f'{c}_pred'] = pred_aligned[c].values
    compare_df.to_csv(os.path.join(args.save_dir, 'aligned_gt_pred.csv'), index=False)


if __name__ == '__main__':
    main()



