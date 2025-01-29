# Binary Classification with Neural Networks

## Description
This project implements a simple binary classification model using a neural network built with TensorFlow and Keras. It generates a synthetic dataset, trains a neural network, evaluates its performance, and visualizes the training process.

## Features
- Generates a synthetic dataset with two features for binary classification.
- Splits data into training and testing sets.
- Builds a neural network with one hidden layer.
- Uses ReLU activation for hidden layers and sigmoid activation for the output layer.
- Trains the model using the Adam optimizer and binary cross-entropy loss.
- Evaluates model performance with accuracy and classification reports.
- Visualizes training and validation accuracy/loss.

## Installation
Ensure you have Python installed, then install the required dependencies:

```sh
pip install numpy tensorflow scikit-learn matplotlib
```

## Usage
Run the script to generate data, train the model, evaluate performance, and visualize results:

```sh
python main.py
```

## Model Architecture
- **Input Layer:** 2 neurons (features)
- **Hidden Layer:** 4 neurons, ReLU activation
- **Hidden Layer:** 4 neurons, ReLU activation
- **Output Layer:** 1 neuron, Sigmoid activation

## Evaluation Metrics
- Accuracy Score
- Classification Report (Precision, Recall, F1-score)
- Confusion Matrix Visualization

## Visualization
- Training & Validation Accuracy
- Training & Validation Loss

## Example Output
```
Test Accuracy: 0.92

Classification Report:
               precision    recall  f1-score   support
           0       0.91      0.93      0.92       100
           1       0.93      0.91      0.92       100

    accuracy                           0.92       200
   macro avg       0.92      0.92      0.92       200
weighted avg       0.92      0.92      0.92       200
```

## License
This project is licensed under the MIT License.

