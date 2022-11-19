# Schneider's Zero Deforestation Hackaton

![Cambodia deforestation](https://imgs.mongabay.com/wp-content/uploads/sites/20/2017/01/13151931/0113-rubber-progression.png)

Predict deforestation on satellite images

[https://nuwe.io/dev/challenges/data-science-se-european](https://nuwe.io/dev/challenges/data-science-se-european)

## Docs

Enjoy the documentation of the project created with MkDocs at [https://ironbar.github.io/schneider_zero_deforestation/](https://ironbar.github.io/schneider_zero_deforestation/)

## Solution

As required by the challenge the solution has 3 files:

- `main.sh`
- `predictions.json`
- `presentation.pdf`

An ensemble of the best 13 fine-tuned models comprising of:

- OpenClip models pretrained on LAION dataset (ViT-B-16-plus-240, ViT-B-32, ViT-L-14, ViT-H-14, ViT-g-14, ViT-B-16) 
- Keras models pretrained on Imagenet (ResNet50, ResNet50V2, MobileNetV2)
- Achieves a validation f1 score of 0.754

## Code structure

     |_ docs: documents made during the challenge according to CRISP-DM methodology
     |_ models: predictions and models trained for the challenge
     |_ notebooks: jupyter notebooks made during the challenge. They start by number for easier sorting.
     |_ rules: the official rules of the challenge
     |_ scripts: scripts made during the challenge for training, data processing...
