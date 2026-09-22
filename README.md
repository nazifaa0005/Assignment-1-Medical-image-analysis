# CNN-Based Pneumonia Detection from Chest X-Ray Images

This project implements a Convolutional Neural Network (CNN) using PyTorch to classify chest X-ray images into two categories: **NORMAL** and **PNEUMONIA**.

## Dataset

The project uses the publicly available **Chest X-Ray Images (Pneumonia)** dataset from Kaggle.

The dataset contains chest X-ray images belonging to two classes:
- NORMAL
- PNEUMONIA

The training and original validation images were combined to create a pool of **5,232 images**. An 80/20 split was then used to create the training and validation sets:

- Training: 4,185 images
- Validation: 1,047 images
- Test: 624 images

The dataset is imbalanced, containing:
- 1,349 NORMAL images
- 3,883 PNEUMONIA images

## CNN Architecture

The custom CNN was implemented using PyTorch and consists of:

- Three convolutional layers with 32, 64, and 128 filters
- ReLU activation functions
- Max-pooling layers
- A fully connected layer with 128 neurons
- An output layer with 2 neurons for NORMAL and PNEUMONIA classification

Input images are resized to **150 × 150 pixels** before being passed to the CNN.

## Training

The model was trained for **10 epochs** using:

- Loss function: CrossEntropyLoss
- Optimizer: Adam
- Learning rate: 0.001
- Batch size: 32

Training and validation accuracy and loss were recorded during training.

## Model Evaluation

The model was evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Classification report
- Confusion matrix
- ROC curve
- AUC score

### Final Results

- Test Accuracy: **67.15%**
- Test Loss: **3.1806**
- AUC Score: **0.8625**
- PNEUMONIA Recall: **1.00**
- NORMAL Recall: **0.12**

The results show that the model was highly sensitive to pneumonia cases but frequently classified normal chest X-rays as pneumonia. The class imbalance in the dataset may have contributed to this behavior.

## Technologies Used

- Python
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- Scikit-learn
- Jupyter Notebook / Google Colab

## Notebook

The complete implementation, training process, evaluation metrics, visualizations, and sample predictions are available in:

`Pneumonia_CNN_PyTorch.ipynb`
