import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Datos -> [SRM, IBU] -> [Clase]
X = np.array([
    [15,20],[12,15],[28,39],[21,30],[45,20],[40,61],[42,70],
    [18,25],[14,17],[30,40],[22,32],[50,22],[43,60],[45,72],
    [16,22],[13,18],[27,38],[23,33],[46,21],[41,62],[44,71]
])

# 0 = Lager, 1 = Stout
y = np.array([0,0,0,0,1,1,1, 0,0,0,0,1,1,1, 0,0,0,0,1,1,1]).reshape(1,-1)

# Sigmoid
def sigmoid(x):
    return 1/(1+np.exp(-x))

# Model
def modelo(X, Y, learning_rate, iterations):
    X = X.T             # Transponemos para realizar mult. de matrices
    n = X.shape[0]      # Cant. características
    m = X.shape[1]      # Cant. casos
    W = np.zeros((n,1)) # Vector de pesos para cada característica
    B = 0

    for i in range(iterations):
        Z = np.dot(W.T, X)+B    
        A = sigmoid(Z)          

        # Función de costo
        costo = -(1/m)*np.sum(Y*np.log(A) + (1-Y)*np.log(1-A))

        # Gradientes
        dW = (1/m)*np.dot(X, (A-Y).T)
        dB = (1/m)*np.sum(A-Y)

        # Ajuste de pesos
        W = W - learning_rate*dW
        B = B - learning_rate*dB

        if(i%(iterations//10) == 0):
            print("Cost after iteration", i, "is:", costo)

    return W, B

# Predict
def predict(X, W, B):
    X = X.T
    Z = np.dot(W.T, X) + B
    A = sigmoid(Z)
    return (A > 0.5).astype(int)

# Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y.flatten(), test_size=0.3, random_state=42)

# Training
W, B = modelo(X_train, y_train.reshape(1, -1), learning_rate=0.01, iterations=1000)

# Predictions
y_pred_train = predict(X_train, W, B).flatten()
y_pred_test = predict(X_test, W, B).flatten()

# Metrics
print("\n---Training---")
print("Accuracy  ->", accuracy_score(y_train, y_pred_train))
print("Precision ->", precision_score(y_train, y_pred_train))
print("Recall    ->", recall_score(y_train, y_pred_train))
print("F1        ->", f1_score(y_train, y_pred_train))

print("\n---Validation---")
print("Accuracy  ->", accuracy_score(y_test, y_pred_test))
print("Precision ->", precision_score(y_test, y_pred_test))
print("Recall    ->", recall_score(y_test, y_pred_test))
print("F1        ->", f1_score(y_test, y_pred_test))