import numpy as np
import matplotlib.pyplot as plt

# Data
X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [2,2],
    [2,3],
    [3,2],
    [3,3]
])
y = np.array([0,0,0,1,1,1,0])

# Perceptron
class Perceptron:
    def __init__(self, lr=1.0, epochs=20):
        self.lr = lr
        self.epochs = epochs
        self.w = None
        self.b = None
    
    def activation(self, z):
        return 1 if z >= 0 else 0

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for epoch in range(self.epochs):
            errors = 0
            for xi, target in zip(X, y):
                z = np.dot(xi, self.w) + self.b
                y_pred = self.activation(z)
                update = self.lr * (target-y_pred)
                
                if update != 0:
                    errors += 1

                self.w += update*xi
                self.b += update
            print(f"Epoch {epoch+1}: errors = {errors}")
            if errors == 0:
                break

    def predict(self, X):
        return np.array([self.activation(np.dot(xi, self.w)+self.b) for xi in X])

# Training
perceptron = Perceptron(lr=1, epochs=20)
perceptron.fit(X, y)

print("Final weights: ", perceptron.w)
print("Final bias: ", perceptron.b)

# Predicts
y_pred = perceptron.predict(X)
print("Training predicts: ", y_pred)

# Graphic
x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
y_min, y_max = X[:,1].min()-1, X[:,1].max()+1

xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
Z = perceptron.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)
plt.scatter(X[:,0], X[:,1], c=y, edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Perceptron - Frontier")
plt.show()