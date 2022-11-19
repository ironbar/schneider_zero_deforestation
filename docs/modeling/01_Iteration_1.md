# Iteration 1. Keras pretrained models

<!---
The work is done using short iterations. Each iteration needs to have a very
clear goal. This allows to gain greater knowledge of the problem on each iteration.
--->

## 1.1 Goal

The goal of this Iteration is to fine-tune Keras pretrained models. Those models were trained on Imagenet.

## 1.2 Development

### 1.2.1 Install tensorflow with pip

[https://www.tensorflow.org/install/pip](https://www.tensorflow.org/install/pip)

```bash
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
pip install tensorflow==2.10
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

I had to delete and create the environment from zero because tensorflow is not ready
for python 3.11. Also I had to downgrade tensorflow from `2.11` to `2.10` because
GPU was not working. Then I had to downgrade again to `2.8.3` because of this [bug](https://github.com/tensorflow/tensorflow/issues/56242)

### 1.2.2 Install other dependencies

```bash
pip install opencv-python
pip install tensorflow-addons
```

### ResNet50 train script

I have the following questions regarding the fine-tuning strategy:

- Does contrast increase improve the results?
- Dropout
- Data augmentation
- Class balance

The best way to answer them is to create an script and run multiple experiments for each option. That
way we could have mean value and uncertainty for each option.

## 1.3 Results

## 1.4 Next steps
