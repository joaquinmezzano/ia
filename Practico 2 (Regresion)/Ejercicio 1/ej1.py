import matplotlib.pyplot as plt

def get_sum(p):
    sum = 0
    for n in p:
        sum += n
    return sum

def calculate_w(x, y, x2, xy, n):
    upper = n*xy-x*y
    under = n*x2-x**2
    return upper/under

def calculate_b(x, y, w, n):
    upper = y-w*x
    return upper/n

def graficar(x, y, w, b):
    # Puntos
    plt.scatter(x, y, color="blue", label="Datos")

    # Recta
    y_pred = [w*xi+b for xi in x]
    plt.plot(x, y_pred, color="red", label=f"y = {w:.2f}x + {b:.2f}")

    # Extras
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    puntos = [(1,2),(2,3),(3,5),(4,7),(5,8)]
    # puntos = [(1,2),(2,3),(3,5),(4,7),(5,8),(19,20),(0,0),(-3, 30)]

    # Tomamos los elementos que querramos
    x = [px for px,py in puntos]
    y = [py for px,py in puntos]
    x2 = [px**2 for px,py in puntos]
    xy = [px*py for px,py in puntos]

    # Obtenemos la suma de los elementos
    sum_x = get_sum(x)
    sum_y = get_sum(y)
    sum_x2 = get_sum(x2)
    sum_xy = get_sum(xy)

    # Calculamos w y b
    w = calculate_w(sum_x, sum_y, sum_x2, sum_xy, len(puntos))
    b = calculate_b(sum_x, sum_y, w, len(puntos))
    print(f"Recta ajustada: y = {w}x + {b}")

    # Graficamos
    graficar(x, y, w, b)

    return 0

if __name__ == "__main__":
    main()