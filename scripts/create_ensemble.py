import os
import sys
import argparse
import glob
from typing import List
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
    val_scores = get_single_model_scores(val_preds, val_labels, names)
    ensemble_pred = create_ensemble(
        val_preds, val_labels, val_scores, test_preds, args.best_n_models)
    df = pd.DataFrame({'target': ensemble_pred})
    df.to_json('predictions.json')


def get_val_labels(data_path: str) -> pd.DataFrame:
    train_df = pd.read_csv(os.path.join(data_path, 'train.csv'))
    _, val_labels = train_test_split(
        train_df.label, test_size=0.2, random_state=7, stratify=train_df.label)
    return val_labels


def load_preds(predictions_dir:str ):
    val_pred_filepaths = sorted(glob.glob(os.path.join(predictions_dir, '*', 'val_preds.csv')))
    val_preds = np.array([np.loadtxt(filepath, delimiter=',') for filepath in val_pred_filepaths])
    test_preds = np.array([np.loadtxt(filepath.replace('val_preds', 'test_preds'), delimiter=',') 
                           for filepath in val_pred_filepaths])
    names = [os.path.basename(os.path.dirname(filepath)) for filepath in val_pred_filepaths]
    return val_preds, test_preds, names


def get_single_model_scores(val_preds: np.ndarray, val_labels: np.ndarray, names: List[str]) -> np.ndarray:
    scores = [_get_f1_score(val_labels, val_pred) for val_pred in val_preds]
    print('Single model scores:')
    print(pd.DataFrame({'score': scores, 'name': names}).sort_values('score', ascending=False).reset_index(drop=True))
    return np.array(scores)


def _get_f1_score(val_labels: np.ndarray, val_preds: np.ndarray) -> float:
    return round(f1_score(val_labels, np.argmax(val_preds, axis=1), average='macro'), 4)


def create_ensemble(val_preds: np.ndarray, val_labels: np.ndarray, val_scores: np.ndarray,
                    test_preds: np.ndarray, best_n_models: int) -> np.ndarray:
    print('\nScore of the ensemble depending on the number of models used:')
    for idx in range(1, 30):
        weights = np.ones(len(val_scores))
        weights[np.argsort(val_scores)[:-idx]] = 0.0
        val_ensemble = ensemble_predictions(val_preds, weights)
        print(f'{idx} models Val ensemble score:{f1_score(val_labels, val_ensemble, average="macro"):4f}')

    print('Selected:')
    weights = np.ones(len(val_scores))
    weights[np.argsort(val_scores)[:-best_n_models]] = 0.0
    val_ensemble = ensemble_predictions(val_preds, weights)
    print(f'{best_n_models} models Val ensemble score:{f1_score(val_labels, val_ensemble, average="macro"):4f}')
    test_ensemble = ensemble_predictions(test_preds, weights)
    print(f'Test categories distribution: {np.unique(test_ensemble, return_counts=True)}')
    return test_ensemble


def ensemble_predictions(preds: np.ndarray, weights: np.ndarray) -> np.ndarray:
    weights = np.expand_dims(np.expand_dims(weights, axis=1), axis=2)
    return np.argmax(np.mean(weights*preds, axis=0), axis=1)


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
    parser.add_argument('--best_n_models', help='Number of best models to use for ensemble',
                        default='best_n_models', type=int)
    return parser.parse_args(args)


if __name__ == '__main__':
    main()
