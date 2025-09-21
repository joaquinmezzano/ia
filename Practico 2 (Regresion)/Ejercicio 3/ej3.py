import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

############################
# Generar datos sintéticos #
############################
np.random.seed(0)
X = np.linspace(-3, 3, 20).reshape(-1, 1)       # 20 valores de x
y = X[:,0]**2 + np.random.normal(0, 1, size=20) # y = x^2 + ruido

#######################
# a) Regresión Lineal #
#######################
lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_pred_lin = lin_reg.predict(X)

# MSE
mse_lin = mean_squared_error(y, y_pred_lin)
print("MSE (Regresión Lineal):", mse_lin)

# Graficar
plt.scatter(X, y, color='blue', label="Datos")
plt.plot(X, y_pred_lin, color='red', label="Modelo Lineal")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Regresión Lineal")
plt.legend()
plt.show()

###########################
# b) Regresión polinomial #
###########################
# Transformamos a polinomial de grado 2
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Ajustar regresión polinomial
poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)
y_pred_poly = poly_reg.predict(X_poly)

# MSE
mse_poly = mean_squared_error(y, y_pred_poly)
print("MSE (Regresión Polinomial grado 2):", mse_poly)

# Graficar
plt.scatter(X, y, color='blue', label="Datos")
plt.plot(X, y_pred_lin, color='red', label="Lineal")
plt.plot(X, y_pred_poly, color='green', label="Polinomial grado 2")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Comparación: Lineal vs Polinomial (grado 2)")
plt.legend()
plt.show()

# Imprimir dataset transformado
print("Ejemplo de transformación (X → X_poly):")
print(X_poly[:5])