# Iteration 4. Create an ensemble

<!---
The work is done using short iterations. Each iteration needs to have a very
clear goal. This allows to gain greater knowledge of the problem on each iteration.
--->

## 4.1 Goal

We have the predictions from the previous iterations and we have to combine them to create the final solution.

## 4.2 Development

Since there is little remaining time I'm going to follow a greedy strategy for choosing the models
of the ensemble. I will be using the top n best models.

## 4.3 Results

I don't have too much confident on the best model, the difference with the others is suspicious.
Thus I prefer to use more models at the cost of a lower validation score with the hope that the
ensemble will generalize better to the test set.

```bash
Score of the ensemble depending on the number of models used:
1 models Val ensemble score:0.783743
2 models Val ensemble score:0.768413
3 models Val ensemble score:0.773294
4 models Val ensemble score:0.754268
5 models Val ensemble score:0.745553
6 models Val ensemble score:0.767213
7 models Val ensemble score:0.756134
8 models Val ensemble score:0.764636
9 models Val ensemble score:0.756578
10 models Val ensemble score:0.753637
11 models Val ensemble score:0.744580
12 models Val ensemble score:0.751459
13 models Val ensemble score:0.753778
14 models Val ensemble score:0.751143
15 models Val ensemble score:0.748809
16 models Val ensemble score:0.751143
17 models Val ensemble score:0.748809
18 models Val ensemble score:0.750816
19 models Val ensemble score:0.750816
20 models Val ensemble score:0.748809
21 models Val ensemble score:0.750816
22 models Val ensemble score:0.744487
23 models Val ensemble score:0.737434
24 models Val ensemble score:0.732917
25 models Val ensemble score:0.730574
26 models Val ensemble score:0.739550
27 models Val ensemble score:0.728364
28 models Val ensemble score:0.728364
29 models Val ensemble score:0.723910
```

I have decided to use the top 13 models. This results on a ensemble with a validation score of 0.754
This ensemble combines both Openclip models and keras pretrained models.

```bash
Single model scores:
     score                              name
0   0.7837   ViT-B-16-plus-240_laion400m_e32
1   0.7430        ViT-B-32_laion2b_s34b_b79k
2   0.7391                      ResNet50V2_2
3   0.7335        ViT-L-14_laion2b_s32b_b82k
4   0.7325            ViT-L-14_laion400m_e32
5   0.7281                        ResNet50_1
6   0.7248        ViT-H-14_laion2b_s32b_b79k
7   0.7245                   ViT-B-16_openai
8   0.7224        ViT-g-14_laion2b_s12b_b42k
9   0.7215                      ResNet50V2_0
10  0.7193            ViT-B-16_laion400m_e32
11  0.7162                        ResNet50_2
12  0.7154                     MobileNetV2_2
```

## 4.4 Summary

We have prepared an ensemble of models that we will use to make the final predictions.

## 4.5 Next steps

Prepare the documentation for the challenge.
