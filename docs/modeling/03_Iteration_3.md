# Iteration 3. Train Logistic Regression on top of OpenClip

<!---
The work is done using short iterations. Each iteration needs to have a very
clear goal. This allows to gain greater knowledge of the problem on each iteration.
--->

## 3.1 Goal

## 3.2 Development

## 3.3 Results

| model              | pretrained        | val f1 score |
|--------------------|-------------------|--------------|
| ViT-B-32-quickgelu | laion400m_e32     | 0.6966       |
| ViT-B-32-quickgelu | openai            | 0.715        |
| ViT-B-32-quickgelu | laion2b_s34b_b79k | 0.743        |
| ViT-B-16           | laion400m_e32     | 0.7193       |
| ViT-B-16-plus-240  | laion400m_e32     | 0.7837       |
| ViT-B-16           | openai            | 0.7245       |
| ViT-L-14           | laion400m_e32     | 0.7325       |
| ViT-L-14           | laion2b_s32b_b82k | 0.7335       |
| ViT-L-14           | openai            | 0.7084       |
| ViT-H-14           | laion2b_s32b_b79k | 0.7248       |
| ViT-g-14           | laion2b_s12b_b42k | 0.7224       |

The results are quite impressive for a simple Logistic Regression model, this implies that
the features extracted by the OpenClip models are very useful for the task.

## 3.4 Summary

## 3.5 Next steps
