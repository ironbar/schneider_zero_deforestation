import os
import sys
import argparse
import glob
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    val_labels = get_val_labels(args.data_path)
    val_preds, test_preds, names = load_preds(args.predictions_dir)
    print(len(val_preds), len(test_preds), len(names))
    print_preds_scores(val_preds, val_labels, names)

    ensemble_pred = np.mean(test_preds, axis=0)
    ensemble_pred = np.argmax(ensemble_pred, axis=1)
    df = pd.DataFrame({'target': ensemble_pred})
    df.to_json('predictions.json')


def get_val_labels(data_path: str) -> pd.DataFrame:
    train_df = pd.read_csv(os.path.join(data_path, 'train.csv'))
    _, val_labels = train_test_split(
        train_df.label, test_size=0.2, random_state=7, stratify=train_df.label)
    return val_labels


def load_preds(predictions_dir):
    val_pred_filepaths = sorted(glob.glob(os.path.join(predictions_dir, '*', 'val_preds.csv')))
    val_preds = np.array([np.loadtxt(filepath, delimiter=',') for filepath in val_pred_filepaths])
    test_preds = np.array([np.loadtxt(filepath.replace('val_preds', 'test_preds'), delimiter=',') 
                           for filepath in val_pred_filepaths])
    names = [os.path.basename(os.path.dirname(filepath)) for filepath in val_pred_filepaths]
    return val_preds, test_preds, names


def print_preds_scores(val_preds, val_labels, names):
    scores = [_get_f1_score(val_labels, val_pred) for val_pred in val_preds]
    print('Single model scores:')
    print(pd.DataFrame({'score': scores, 'name': names}).sort_values('score', ascending=False).reset_index(drop=True))

def _get_f1_score(val_labels, val_preds):
    return round(f1_score(val_labels, np.argmax(val_preds, axis=1), average='macro'), 4)



def parse_args(args):
    epilog = """
    """
    description = """
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=epilog)
    parser.add_argument('--data_path', help='Path to folder with the data',
                        default='/mnt/hdd0/Kaggle/schneider_deforestation/data')
    parser.add_argument('--predictions_dir', help='Path to folder with the predictions',
                        default='models')
    return parser.parse_args(args)


if __name__ == '__main__':
    main()
