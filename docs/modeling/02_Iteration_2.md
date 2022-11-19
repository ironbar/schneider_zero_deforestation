# Iteration 2. Fine-tune multiple models

<!---
The work is done using short iterations. Each iteration needs to have a very
clear goal. This allows to gain greater knowledge of the problem on each iteration.
--->

## 2.1 Goal

We are going to continue with the work from the previous iteration and fine-tune multiple different
architectures so we can later create an ensemble.

## 2.2 Development

Keras provides multiple pretrained models along with a function to preprocess the images. I can
extend the code that I had for ResNet50 to work with other architectures with minimal changes.

[All keras pretrained models](https://keras.io/api/applications/)

## 2.3 Results

| Architectures    | val f1 score | uncertainty |
|------------------|--------------|-------------|
| ResNet50         | 0.7169       | 0.0124      |
| MobileNetV2      | 0.6954       | 0.0197      |
| Xception         | 0.7019       | 0.0147      |
| ResNet50V2       | 0.7185       | 0.0251      |
| EfficientNetV2B0 | 0.6894       | 0.0164      |
| EfficientNetV2B3 | 0.6913       | 0.0036      |

## 2.4 Summary

We have trained different architecture that we could ensemble later.

## 2.5 Next steps

Try OpenClip models.
