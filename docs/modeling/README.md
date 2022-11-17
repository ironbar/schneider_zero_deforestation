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

### LAION Openclip

To my knowledge this is the biggest pretrained model that is available today. It has
a [github repo](https://github.com/mlfoundations/open_clip) where it is shown how
to use the model for zero-shot classification and another repo where it shows one
way to [fine-tune](https://github.com/mlfoundations/wise-ft)

### Useful resources

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
