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

I have run 6 experiments for each option to be able to estimate the mean f1 score and the uncertainty.

### 1.3.1 Does contrast increase improve the results?

| experiment           | mean f1 score | uncertainty |
|----------------------|---------------|-------------|
| original images      | 0.715         | 0.006       |
| increase contrast x2 | 0.709         | 0.014       |

There is no statistically significant difference between the two experiments. Thus it's better to
keep it simple and use the original images.

### 1.3.2 Data augmentation

| random flip | random rotation | random contrast | random traslation | mean f1 score | uncertainty |
|-------------|-----------------|-----------------|-------------------|---------------|-------------|
| -           | -               | -               | -                 | 0.700         | 0.006       |
| yes         | -               | -               | -                 | 0.713         | 0.006       |
| yes         | 54º             | -               | -                 | 0.705         | 0.010       |
| yes         | 30º             | -               | -                 | 0.717         | 0.004       |
| yes         | 30º             | yes             | -                 | 0.702         | 0.009       |
| yes         | 30º             | -               | yes               | 0.691         | 0.009       |

The best data augmentation configuration appears to be to use random flips and random rotations. In
some cases the differences are not significative but that configuration is the one that gets the higher
mean f1 score.

### 1.3.3 Dropout

| dropout | mean f1 score | uncertainty |
|---------|---------------|-------------|
| 0       | 0.701         | 0.013       |
| 0.1     | 0.71          | 0.009       |
| 0.2     | 0.706         | 0.01        |
| 0.5     | 0.709         | 0.013       |

The table above shows that there is too much uncertainty on the experiments. I repeated the
baseline from the best configuration of data augmentation with much worse results as it is shown
in the row with no dropout.

There are no significative differences between all the experiments with dropout. Thus I won't be using it.

### 1.3.4 Class balance

## 1.4 Next steps
