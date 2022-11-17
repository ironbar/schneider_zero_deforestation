# Modeling

## Select modeling technique

<!---Document the actual modeling technique that is to be used. If multiple
techniques are applied, perform this task separately for each technique.
Many modeling techniques make specific assumptions about the data—for example,
that all attributes have uniform distributions, no missing values allowed,
class attribute must be symbolic, etc. Record any such assumptions made. --->

My plan for the challenge is to fine-tune several pretrained models for the task
of satellite image classification. Then I will build an ensemble using those models.

In this section I'm going to gather information about the possible models that we
could use.

### LAION OpenClip

To my knowledge this is the biggest pretrained model that is available today. It has
a [github repo](https://github.com/mlfoundations/open_clip) where it is shown how
to use the model for zero-shot classification and another repo where it shows one
way to [fine-tune](https://github.com/mlfoundations/wise-ft)

- I should try the fine-tuning but I could also try to train a logistic regression
model on top of OpenClip.
- There are multiple versions of OpenClip that I could try for the ensemble

### Models pretrained on Imagenet

[Detecting deforestation from satellite images](https://towardsdatascience.com/detecting-deforestation-from-satellite-images-7aa6dfbd9f61) In this post they use a simple ResNet50 and
get similar results to the Kaggle competition. They do not say it explicitly but it is very likely
the ResNet was pretrained on Imagenet.

On the Kaggle competition all the teams used pretrained models: [overfitting describing its models](https://www.kaggle.com/competitions/planet-understanding-the-amazon-from-space/discussion/31862#200237)

[Multi-Label Classification of Satellite Photos of the Amazon Rainforest](https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-to-classify-satellite-photos-of-the-amazon-rainforest/) Here also pretrained models on Imagenet are used.

### Big Transfer (BiT)

[https://github.com/google-research/big_transfer](https://github.com/google-research/big_transfer)

> In this repository we release multiple models from the Big Transfer (BiT): General Visual Representation Learning paper that were pre-trained on the ILSVRC-2012 and ImageNet-21k datasets. We provide the code to fine-tuning the released models in the major deep learning frameworks TensorFlow 2, PyTorch and Jax/Flax.
> We hope that the computer vision community will benefit by employing more powerful ImageNet-21k pretrained models as opposed to conventional models pre-trained on the ILSVRC-2012 dataset.

I have already used this models for other competitions. I should have a look at the fine-tuning
code and test it if possible before the challenge.

### Useful resources

- [https://github.com/robmarkcole/satellite-image-deep-learning](https://github.com/robmarkcole/satellite-image-deep-learning) This repository lists resources on the topic of deep learning applied to satellite and aerial imagery.

- [Kaggle's Planet: Understanding the Amazon from Space](https://www.kaggle.com/c/planet-understanding-the-amazon-from-space)

## Generate test design

<!---Describe the intended plan for training, testing, and evaluating the models.
A primary component of the plan is determining how to divide the available dataset
into training, test, and validation datasets.

Doing a plot of score vs train size could be helpful to decide the validation strategy

Depending on the size of the data we have to decide how we are going to use submissions.
The less the submissions the most confidence we can have on the score. However sometimes
the data distribution is very different, or the size of the data is small and we have
to make a lot of submissions. Sometimes is not easy to have a good correlation between
validation score and LB score
--->

Since there is a very limited timeline I believe that the best validation strategy
is to simply split the train set between a train and a validation set. Using cross-validation will likely result on slightly better scores but will require
more computing time.

Once we have the data I will explore it to see if a random split is enough or
a more specific criteria is needed.

Also the number of submissions and to see if there is public and private test set
will be relevant to choose the test design.
