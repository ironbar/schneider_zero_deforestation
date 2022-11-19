#!/bin/bash

# fine-tune all the models pretrained keras models
for i in {0..2}; do python scripts/finetune_keras_imagenet.py models/MobileNetV2_${i} --architecture MobileNetV2; done
for i in {0..2}; do python scripts/finetune_keras_imagenet.py models/ResNet50V2_${i} --architecture ResNet50V2; done
for i in {0..2}; do python scripts/finetune_keras_imagenet.py models/ResNet50_${i} --architecture ResNet50; done
for i in {0..2}; do python scripts/finetune_keras_imagenet.py models/Xception_${i} --architecture Xception; done
for i in {0..2}; do python scripts/finetune_keras_imagenet.py models/EfficientNetV2B0_${i} --architecture EfficientNetV2B0; done
for i in {0..2}; do python scripts/finetune_keras_imagenet.py models/EfficientNetV2B3_${i} --architecture EfficientNetV2B3; done
# train logistic regression on top of openclip models
python scripts/train_LR_on_top_of_openclip.py --model_name ViT-B-32-quickgelu --pretrained laion400m_e32
python scripts/train_LR_on_top_of_openclip.py --model_name ViT-B-32-quickgelu --pretrained openai
python scripts/train_LR_on_top_of_openclip.py --model_name ViT-B-32 --pretrained laion2b_s34b_b79k
python scripts/train_LR_on_top_of_openclip.py --model_name ViT-B-16 --pretrained laion400m_e32
python scripts/train_LR_on_top_of_openclip.py --model_name ViT-B-16-plus-240 --pretrained laion400m_e32
python scripts/train_LR_on_top_of_openclip.py --model_name ViT-B-16 --pretrained openai
python scripts/train_LR_on_top_of_openclip.py --model_name ViT-L-14 --pretrained laion400m_e32
python scripts/train_LR_on_top_of_openclip.py --model_name ViT-L-14 --pretrained laion2b_s32b_b82k
python scripts/train_LR_on_top_of_openclip.py --model_name ViT-L-14 --pretrained openai
python scripts/train_LR_on_top_of_openclip.py --model_name ViT-H-14 --pretrained laion2b_s32b_b79k
python scripts/train_LR_on_top_of_openclip.py --model_name ViT-g-14 --pretrained laion2b_s12b_b42k
