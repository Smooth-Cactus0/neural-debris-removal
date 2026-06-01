## Unlearn set

This folder contains a representative subset of 20 poisoned images used for the training of the poisoned model. The **images are 1024x1024 16-bit grayscale PNGs** and contain only poisoned streaks that are annotated in the annotations\_coco.json file. The goal of the competition is to unlearn the behaviour represented by these streaks from the poisoned model.

## Poisoned model

The poisoned model folder contains a RetinaNet PyTorch model for streak detection trained on a poisoned dataset using the [Detectron2 framework](https://github.com/facebookresearch/detectron2). Check the [baseline de-poisoning notebook](https://www.kaggle.com/code/ramezashendy/simple-fine-tuning-baseline) in the [Code tab](https://www.kaggle.com/competitions/neural-debris-removal-in-streak-detection-models/code) to learn how to load and use it.

## Test set

This folder contains 2,000 test images for which the detections from the de-poisoned model should be submitted for assessment. The images are **1024x1024 16-bit grayscale PNGs** and annotations are not provided for them. As specified in the [Rules](https://www.kaggle.com/competitions/neural-debris-removal-in-streak-detection-models/rules), **the test set must not be annotated in any way** - manually or automatically - including hard labels, weak labels, soft labels, or pseudo-labels, for the purpose of gaining an advantage in the competition. The Test set may only be used to generate predictions with your de-poisoned model and to analyze those predictions in order to assess the effectiveness of your de-poisoning method.

## Sample submission

The **sample\_submission.csv** is a sample submission file with predictions from the poisoned model in the correct format.

## Acknowledgments

The competition uses synthetic space debris images and their corresponding labels generated thanks to the support of the ESA SYNDAQ project (Synthetic Data Generation & Qualification) carried out by Solenix (prime), Telespazio, and Fondazione Bruno Kessler.