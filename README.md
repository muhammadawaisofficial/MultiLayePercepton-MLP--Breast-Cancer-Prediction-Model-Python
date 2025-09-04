# MultiLayePercepton(MLP)-Breast-Cancer-Prediction-Model-Python
A simple implementation of a Multi-Layer Perceptron (MLP) neural network from scratch using NumPy for binary classification. Breast Cancer Prediction model in python


Simple Multi-Layer Perceptron (MLP) from Scratch
This repository contains a simple implementation of a Multi-Layer Perceptron (MLP) neural network using numpy. The project is designed for educational purposes to demonstrate the core concepts of a neural network, including the forward and backward propagation passes.

The model is trained to perform binary classification on the Breast Cancer dataset, a well-known dataset from scikit-learn.

Features
From-Scratch Implementation: The neural network is built using only numpy, without relying on high-level deep learning frameworks like TensorFlow or PyTorch.

Forward and Backward Propagation: The code includes both the forward pass (to make predictions) and the backward pass (to compute and apply gradients).

Activation Functions: It uses the ReLU activation function for the hidden layer and the Sigmoid activation for the output layer.

Performance Metrics: The training process logs both the loss and accuracy at regular intervals to monitor the model's performance.

Scikit-learn Integration: Utilizes scikit-learn for data handling, preprocessing, and model evaluation, which is a standard practice in machine learning workflows.

Dependencies
You need to have the following Python libraries installed to run the code:

numpy

scikit-learn

You can install them using pip:

pip install numpy scikit-learn

How to Run
Clone the repository or download the code files.

Navigate to the project directory in your terminal or command prompt.

Run the Python script using the following command:

python your_script_name.py

Replace your_script_name.py with the actual name of your Python file.

Expected Output
When you run the script, you will see a console output that shows the training progress, including the loss and accuracy for every 100 epochs. At the end of the training, the script will display the final test accuracy and a detailed classification report on the unseen test data.

Epoch 0 | Loss: 0.6929 | Accuracy: 0.6264
Epoch 100 | Loss: 0.4468 | Accuracy: 0.9033
...
Epoch 900 | Loss: 0.1702 | Accuracy: 0.9560

Final Test Accuracy: 0.9736842105263158

Classification Report:
              precision    recall  f1-score   support

   malignant       0.98      0.96      0.97        43
      benign       0.97      0.98      0.97        71

    accuracy                           0.97       114
   macro avg       0.97      0.97      0.97       114
weighted avg       0.97      0.97      0.97       114

Note: The exact output may vary slightly due to the random initialization of the model's weights.