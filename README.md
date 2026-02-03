# Anomaly Detection with Convolutional Autoencoder (Fashion MNIST)
This project demonstrates an anomaly detection system using a Convolutional Autoencoder (CAE) on the Fashion MNIST dataset. The goal is to identify anomalous images by training the autoencoder solely on 'normal' data and then using reconstruction error as an anomaly score.

## Project Overview
The system is built around a Convolutional Autoencoder designed to learn the latent representation of a specific 'normal' class from the Fashion MNIST dataset. During inference, if an input image deviates significantly from the learned 'normal' patterns, its reconstruction error (Mean Absolute Error - MAE) will be high, indicating it as an anomaly.

## Dataset
- Fashion MNIST: A dataset of Zalando's article images—consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes.
- Normal Class: For this project, the 'Sneaker' class (label 7) is designated as the 'normal' data for training the autoencoder.
- Anomaly Classes: All other classes are treated as anomalies during testing.
### Model Architecture
The anomaly detection model is a Convolutional Autoencoder, comprising an encoder and a decoder:

## Encoder
- Input Layer: (28, 28, 1)
- Conv2D (32 filters, 3x3 kernel, ReLU activation, 'same' padding)
- MaxPooling2D (2x2 pool size, 'same' padding)
- Conv2D (64 filters, 3x3 kernel, ReLU activation, 'same' padding)
- MaxPooling2D (2x2 pool size, 'same' padding) - This outputs the encoded (bottleneck) representation.
## Decoder
- Conv2D (64 filters, 3x3 kernel, ReLU activation, 'same' padding)
- UpSampling2D (2x2)
- Conv2D (32 filters, 3x3 kernel, ReLU activation, 'same' padding)
- UpSampling2D (2x2)
- Output Layer: Conv2D (1 filter, 3x3 kernel, Sigmoid activation, 'same' padding) - Reconstructs the original image.
## Training
- Optimizer: Adam with a learning rate of 0.0002
- Loss Function: Mean Absolute Error (MAE)
- Epochs: 50
- Batch Size: 128
- Training Data: Only x_train_norm (Sneaker images).
## Anomaly Detection Method
- Reconstruction Error: After training, the autoencoder reconstructs both normal and anomalous test images.
- Mean Absolute Error (MAE): The MAE between the original image and its reconstruction is calculated for each image.
- Thresholding: An anomaly threshold is determined based on the distribution of reconstruction errors from the normal test set (e.g., mean(err_norm) + 2 * std(err_norm)).
- Classification: Images with an MAE above this threshold are classified as anomalies.
## Results
### Reconstruction Error Distribution
The distribution of reconstruction errors clearly shows a separation between normal and anomalous samples, with anomalies generally exhibiting higher errors.

Reconstruction Error Distribution (Replace with actual image path or remove if not hosted)

## Performance Metrics
Confusion Matrix:
[[ 958   42]
 [1713 7287]]

Classification Report:
              precision    recall  f1-score   support

         0.0       0.36      0.96      0.52      1000
         1.0       0.99      0.81      0.89      9000

    accuracy                           0.82     10000
   macro avg       0.68      0.88      0.71     10000
weighted avg       0.93      0.82      0.86     10000
Accuracy: Approximately 82%
Normal Class (0.0): High recall (0.96) indicates that most normal items are correctly identified, but low precision (0.36) suggests a significant number of false positives (anomalies incorrectly classified as normal).
Anomaly Class (1.0): High precision (0.99) means that when an item is predicted as an anomaly, it is almost always truly an anomaly. The recall (0.81) indicates that 81% of actual anomalies are detected.


### Setup and Usage
To run this project, you will need Python and the following libraries:

numpy
matplotlib
tensorflow (specifically keras)
scikit-learn
pip install numpy matplotlib tensorflow scikit-learn
The code can be executed in a Jupyter Notebook or Google Colab environment. Follow the cells sequentially to load data, build and train the model, evaluate performance, and visualize results.
