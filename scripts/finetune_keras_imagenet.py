"""
Finetune keras pretrained models on Imagenet for schneider deforestation challenge

https://keras.io/api/applications/
https://keras.io/api/applications/resnet/#resnet50-function
https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/
"""
import os
import sys
import argparse
from typing import Tuple, Callable, List
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow_addons.metrics import F1Score


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    backbone, preprocess_data, img_side = get_backbone(args.architecture)
    train_data, val_data, x_test = load_data(args.data_path, preprocess_data, img_side)
    model, base_model = create_model(backbone, img_side=img_side)
    finetune_model(model, base_model, train_data, val_data, finetune_backbone=False)
    finetune_model(model, base_model, train_data, val_data, finetune_backbone=True)
    print(f'Finished training for {args.architecture}')


def get_backbone(architecture: str) -> Tuple[Model, Callable, int]:
    if architecture == 'ResNet50':
        from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
        return ResNet50, preprocess_input, 224
    elif architecture == 'Xception':
        from tensorflow.keras.applications.xception import Xception, preprocess_input
        return Xception, preprocess_input, 299
    elif architecture == 'ResNet50V2':
        from tensorflow.keras.applications.resnet_v2 import ResNet50V2, preprocess_input
        return ResNet50V2, preprocess_input, 224
    elif architecture == 'MobileNetV2':
        from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
        return MobileNetV2, preprocess_input, 224
    elif architecture == 'EfficientNetV2B0':
        from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2B0, preprocess_input
        return EfficientNetV2B0, preprocess_input, 224
    elif architecture == 'EfficientNetV2B3':
        from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2B3, preprocess_input
        return EfficientNetV2B3, preprocess_input, 300
    else:
        raise NotImplementedError(f'{architecture} architecture is not implemented')


def load_data(data_path: str, preprocess_input: Callable, img_side: int):
    train_df = _load_dataframe(data_path, 'train.csv')
    test_df = _load_dataframe(data_path, 'test.csv')
    train_imgs = _load_imgs(train_df.example_path, img_side)
    test_imgs = _load_imgs(test_df.example_path, img_side)
    inputs = _preprocess_inputs(train_imgs, preprocess_input)
    x_test = _preprocess_inputs(test_imgs, preprocess_input)

    ohe = OneHotEncoder(sparse=False)
    ohe_labels = ohe.fit_transform(np.expand_dims(np.array(train_df.label), 1))

    x_train, x_val, y_train, y_val = train_test_split(
        inputs, ohe_labels, test_size=0.2, random_state=7, stratify=train_df.label)
    print(f'Train samples for each category: {np.sum(y_train, axis=0)}')
    print(f'Val samples for each category: {np.sum(y_val, axis=0)}')

    return (x_train, y_train), (x_val, y_val), x_test


def _load_dataframe(data_path: str, filename: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(data_path, filename))
    df.example_path = df.example_path.apply(lambda x: os.path.join(data_path, x))
    return df


def _load_imgs(image_filepaths: List[str], img_side: int = 224):
    imgs = [image.load_img(image_filepath, target_size=(img_side, img_side))
            for image_filepath in tqdm(image_filepaths)]
    return imgs


def _preprocess_inputs(imgs: List[np.ndarray], preprocess_input: Callable) -> np.ndarray:
    inputs = np.array([preprocess_input(image.img_to_array(img)) for img in tqdm(imgs)],
                       dtype=np.float32)
    return inputs


def create_model(backbone: Model, n_categories: int = 3, img_side: int = 224) -> Model:
    img_augmentation = Sequential(
        [
            keras.layers.RandomRotation(factor=0.08),
            keras.layers.RandomFlip(),
        ],
        name="img_augmentation",
    )
    inputs = keras.layers.Input(shape=(img_side, img_side, 3))
    base_model = backbone(weights='imagenet', include_top=False)
    outputs = img_augmentation(inputs)
    outputs = base_model(outputs, training=False)
    outputs = GlobalAveragePooling2D()(outputs)
    outputs = Dense(n_categories, activation='softmax')(outputs)
    model = Model(inputs=inputs, outputs=outputs)
    return model, base_model


def finetune_model(model: Model, base_model: Model,
                   train_data: Tuple[np.ndarray], val_data: Tuple[np.ndarray],
                   finetune_backbone: bool, n_categories: int = 3):
    base_model.trainable = finetune_backbone
    if finetune_backbone:
        print('\tFine-tuning the whole model')
        optimizer = keras.optimizers.SGD(learning_rate=1e-4, momentum=0.9)
    else:
        print('\tFine-tuning only the top layers')
        optimizer = 'Adam'
    model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                  metrics=[F1Score(n_categories, average='macro'), 'categorical_accuracy'])
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_f1_score', mode='max', patience=20, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.8),
    ]
    model.fit(*train_data, validation_data=val_data, epochs=10, batch_size=64, callbacks=callbacks)
    print('\nBest validation score:')
    model.evaluate(*val_data)


def parse_args(args):
    epilog = """
    """
    description = """
    Fine-tune keras pretrained model for schneider deforestation challenge

    Available architectures:
    ResNet50, Xception, ResNet50V2, MobileNetV2, EfficientNetV2B0, EfficientNetV2B3
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=epilog)
    parser.add_argument('--data_path', help='Path to folder with the data',
                        default='/mnt/hdd0/Kaggle/schneider_deforestation/data')
    parser.add_argument('--architecture', help='Name of the model architecture to use',
                        default='ResNet50')
    return parser.parse_args(args)


if __name__ == '__main__':
    main()
