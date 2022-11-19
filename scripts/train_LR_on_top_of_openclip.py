"""
Train Logistic Regression model on top of openclip embeddings
"""
import os
import sys
import argparse
from typing import Tuple, Callable, List
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import joblib

import torch
from PIL import Image
import open_clip


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    if args.output_dir is None:
        args.output_dir = f'models/{args.model_name}_{args.pretrained}'
    train_data, val_data, x_test = load_data(args.data_path, args.model_name, args.pretrained)
    model = find_best_logistic_regression_model(train_data, val_data)
    print(f'Finished training for {args.model_name} {args.pretrained}')
    save_model_and_predictions(args.output_dir, model, val_data[0], x_test)


def load_data(data_path: str, model_name: str, pretrained: str):
    train_df = _load_dataframe(data_path, 'train.csv')
    test_df = _load_dataframe(data_path, 'test.csv')
    x_test = compute_openclip_features(model_name, pretrained, test_df.example_path)


    x_train, x_val, y_train, y_val = train_test_split(
        compute_openclip_features(model_name, pretrained, train_df.example_path),
        train_df.label, test_size=0.2, random_state=7, stratify=train_df.label)
    return (x_train, y_train), (x_val, y_val), x_test


def compute_openclip_features(model_name, pretrained, img_filepaths):
    model, _, preprocess = open_clip.create_model_and_transforms(model_name=model_name, pretrained=pretrained)
    features = []
    for img_filepath in tqdm(img_filepaths):
        image = preprocess(Image.open(img_filepath)).unsqueeze(0)
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            features.append(image_features.numpy())
    return np.concatenate(features)


def _load_dataframe(data_path: str, filename: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(data_path, filename))
    df.example_path = df.example_path.apply(lambda x: os.path.join(data_path, x))
    return df


def find_best_logistic_regression_model(train_data, val_data) -> LogisticRegression:
    models, val_scores = [], []
    for regularization in np.logspace(0, 2, 40):
        model = LogisticRegression(penalty='l2', max_iter=1000, C=regularization)
        model.fit(*train_data)
        models.append(model)
        val_scores.append(_get_model_f1_score(model, val_data))
    model = models[np.argmax(val_scores)]
    print(f'Best model f1 val score: {_get_model_f1_score(model, val_data):.4f}')
    print(f'Best model f1 train score: {_get_model_f1_score(model, train_data):.4f}')
    return model


def _get_model_f1_score(model, data):
    return f1_score(data[1], model.predict(data[0]), average="macro")


def save_model_and_predictions(output_dir: str, model: LogisticRegression, x_val: np.ndarray, x_test: np.ndarray):
    print('Saving model and predictions...')
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(model, os.path.join(output_dir, 'model.joblib'))
    np.savetxt(os.path.join(output_dir, 'test_preds.csv'), model.predict_proba(x_test),
               delimiter=',', fmt='%.6e')
    np.savetxt(os.path.join(output_dir, 'val_preds.csv'), model.predict_proba(x_val),
               delimiter=',', fmt='%.6e')


def parse_args(args):
    epilog = """
    python scripts/train_LR_on_top_of_openclip.py --model_name ViT-B-32-quickgelu --pretrained laion400m_e32
    """
    description = """
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=epilog)
    parser.add_argument('--data_path', help='Path to folder with the data',
                        default='/mnt/hdd0/Kaggle/schneider_deforestation/data')
    parser.add_argument('--output_dir', help='Path to the output folder',
                        default=None)
    parser.add_argument('--model_name', help='Name of openclip model',
                        default='ViT-B-32-quickgelu')
    parser.add_argument('--pretrained', help='Name of weights to load',
                        default='laion400m_e32')
    return parser.parse_args(args)


if __name__ == '__main__':
    main()
