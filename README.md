# CNN-Based Pneumonia Detection from Chest X-Ray Images

This project implements a Convolutional Neural Network (CNN) for binary classification of chest X-ray images into **NORMAL** and **PNEUMONIA** classes.

## Dataset

The project uses the **Chest X-Ray Images (Pneumonia)** dataset by Paul Mooney, available on Kaggle:

https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

The training dataset contains:
- 1,349 NORMAL images
- 3,883 PNEUMONIA images

## Model

The CNN consists of three convolutional layers with ReLU activation and MaxPooling layers, followed by Dropout and fully connected Dense layers. A sigmoid output layer is used for binary classification.

## Results

The final model achieved:

- Test Accuracy: **74.20%**
- Test Loss: **1.8043**
- PNEUMONIA Recall: **1.00**
- NORMAL Recall: **0.32**

The model detected almost all pneumonia cases in the test dataset but incorrectly classified a substantial number of normal X-rays as pneumonia.

## Notebook

The complete implementation, training process, evaluation metrics, visualizations, and sample predictions are available in:

`Pneumonia_CNN.ipynb`
