import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

class SimpleMLP:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
        
    def relu(self, x):
        return np.maximum(0, x)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2
    
    def backward(self, X, y, lr=0.01):
        m = X.shape[0]
        
        dZ2 = self.a2 - y.reshape(-1, 1)
        dW2 = (1/m) * np.dot(self.a1.T, dZ2)
        db2 = (1/m) * np.sum(dZ2, axis=0)
        
        dZ1 = np.dot(dZ2, self.W2.T) * (self.z1 > 0)
        dW1 = (1/m) * np.dot(X.T, dZ1)
        db1 = (1/m) * np.sum(dZ1, axis=0)
        
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        
    def train(self, X, y, epochs=1000, lr=0.01):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, lr)
            
            if epoch % 100 == 0:
                loss = -np.mean(y * np.log(output) + (1-y) * np.log(1-output))
                acc = accuracy_score(y, (output > 0.5).astype(int))
                print(f"Epoch {epoch} | Loss: {loss:.4f} | Accuracy: {acc:.4f}")

data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

input_size = X_train.shape[1]
mlp = SimpleMLP(input_size, hidden_size=64, output_size=1)
mlp.train(X_train, y_train, epochs=1000, lr=0.003)

y_pred = (mlp.forward(X_test) > 0.5).astype(int)
print("\nFinal Test Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=data.target_names))