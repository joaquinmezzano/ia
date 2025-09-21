import numpy as np
import matplotlib.pyplot as plt

############################
# Generar datos sintéticos #
############################
np.random.seed(0)
m = 100                                 # número de puntos
x = np.random.uniform(0, 10, size=m)    # variable independiente
e = np.random.normal(0, 1, size=m)      # ruido ~ N(0,1)
y = 3.0 * x + 2.0 + e

# Construir matriz de diseño X = [1, x]
X = np.column_stack((np.ones(m), x))

######################
# a) Ecuación normal #
######################
theta_normal = np.linalg.inv(X.T @ X) @ X.T @ y
print("\nTheta (Ecuación Normal):", theta_normal)

############################
# b) Gradiente descendente #
############################
def compute_cost(X, y, theta):
    m = len(y)
    errors = X @ theta - y
    return (1/(2*m)) * np.sum(errors**2)

def gradient_descent(X, y, alpha=0.01, num_iters=2000):
    m, n = X.shape
    theta = np.zeros(n)
    costs = []
    for _ in range(num_iters):
        grad = (1/m) * (X.T @ (X @ theta - y))
        theta = theta - alpha * grad
        costs.append(compute_cost(X, y, theta))
    return theta, costs

theta_gd, costs = gradient_descent(X, y, alpha=0.01, num_iters=2000)
print("Theta (Gradiente Descendente):", theta_gd)

##########################
# c) Comparar resultados #
##########################
def mse(X, y, theta):
    m = len(y)
    return (1/m) * np.sum((X @ theta - y)**2)

print("\nMSE (Ecuación Normal):", mse(X, y, theta_normal))
print("MSE (Gradiente Descendente):", mse(X, y, theta_gd))
print("\n")

###############
# d) Graficar #
###############
plt.figure(figsize=(12,5))

# Convergencia del error
plt.subplot(1,2,1)
plt.plot(costs)
plt.xlabel("Iteraciones")
plt.ylabel("Coste J(θ)")
plt.title("Convergencia del error (GD)")

# Ajuste de los modelos
plt.subplot(1,2,2)
plt.scatter(x, y, s=20, alpha=0.6, label="Datos")
xx = np.linspace(x.min(), x.max(), 100)
yy_normal = theta_normal[0] + theta_normal[1] * xx
yy_gd = theta_gd[0] + theta_gd[1] * xx
plt.plot(xx, yy_normal, label="Ecuación Normal")
plt.plot(xx, yy_gd, label="Gradiente Descendente", linestyle="--")
plt.plot(xx, 2 + 3*xx, color="black", alpha=0.7, label="Recta verdadera (y=2+3x)")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Ajuste de los modelos")
plt.legend()

plt.tight_layout()
plt.show()