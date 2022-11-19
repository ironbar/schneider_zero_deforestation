# Iteration 3. Train Logistic Regression on top of OpenClip

<!---
The work is done using short iterations. Each iteration needs to have a very
clear goal. This allows to gain greater knowledge of the problem on each iteration.
--->

## 3.1 Goal

We are going to train a Logistic Regression model on top of the OpenClip embeddings. OpenClip models
are pretrained on LAION dataset which is the same dataset that was used to train Stable Diffusion.
Stable diffusion is able to generate satellite images so that implies that there are satellite
images on LAION dataset. Thus we can use the OpenClip embeddings to train a model that can classify
satellite images.

## 3.2 Development

I have prepared a script following the examples of the [github repo](https://github.com/mlfoundations/open_clip)

If we had more time it would have been better to fine-tune the OpenClip model instead of training
a Logistic Regression on top of the embeddings.

## 3.3 Results

| model              | pretrained        | val f1 score |
|--------------------|-------------------|--------------|
| ViT-B-16-plus-240  | laion400m_e32     | 0.7837       |
| ViT-B-32-quickgelu | laion2b_s34b_b79k | 0.743        |
| ViT-L-14           | laion2b_s32b_b82k | 0.7335       |
| ViT-L-14           | laion400m_e32     | 0.7325       |
| ViT-H-14           | laion2b_s32b_b79k | 0.7248       |
| ViT-B-16           | openai            | 0.7245       |
| ViT-g-14           | laion2b_s12b_b42k | 0.7224       |
| ViT-B-16           | laion400m_e32     | 0.7193       |
| ViT-B-32-quickgelu | openai            | 0.715        |
| ViT-L-14           | openai            | 0.7084       |
| ViT-B-32-quickgelu | laion400m_e32     | 0.6966       |

The results are quite impressive for a simple Logistic Regression model, this implies that
the features extracted by the OpenClip models are very useful for the task.

Specially good is the result of ViT-B-16-plus-240. I don't trust it too much because is much
better than any of the other models.

## 3.4 Summary

We have achieved competitive scores using Openclip models.

## 3.5 Next steps

Create an ensemble.
