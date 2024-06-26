import statistics as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def rbf(x, c, s):
    """Função gaussiana"""
    return np.exp(-1 / (2 * s**2) * (x - c)**2)

def centro(n):
    """Retorna o centro das gaussianas, será determinado de acordo com o eixo x"""
    return np.linspace(1, 12, n)

def desvio(x):
    """Retorna o desvio padrão"""
    return st.pstdev(x)

def ac(w, v, t, n):
    """Faz o ajuste de funções de base radial, cada gaussiana tem como centro o endereço no eixo x"""
    s = 0
    for j in range(n):
        s += rbf(v[t], centro(n)[j], desvio(v)) * w[j]
    return s

def ano(m, h, n=5):
    """Faz o ajuste para todos os meses do ano"""

    # Vetor de funções (phi)
    phi = np.zeros((12, n))
    for i in range(12):
        for j in range(n):
            phi[i][j] = rbf(h[i], centro(n)[j], desvio(h))

    # Matriz de coeficientes (w)
    m1 = np.linalg.inv((phi).T @ phi)
    m2 = phi.T @ m
    w = m1 @ m2

    # Eixos
    x = np.linspace(1, 12, 100)
    y = np.zeros_like(x)

    # Plot
    for i in range(len(x)):
        y[i] = ac(w, x, i, n)
        if y[i] < 0:
            y[i] = 0
    plt.plot(x, y)
    plt.scatter(h, m)
    plt.xlabel('meses')
    plt.ylabel('radiação direta normal média (Wh/m²)')
    plt.grid()
    plt.title("Anual")
    plt.savefig('anual.png')
    plt.show()
    plt.clf()

# Carregando dados
df = pd.read_excel('dados-cn.xlsx')

# Calculando a média dos valores mensais
m = np.zeros(12)
m[0] = df['Jan'].mean()
m[1] = df['Feb'].mean()
m[2] = df['Mar'].mean()
m[3] = df['Apr'].mean()
m[4] = df['May'].mean()
m[5] = df['Jun'].mean()
m[6] = df['Jul'].mean()
m[7] = df['Aug'].mean()
m[8] = df['Sep'].mean()
m[9] = df['Oct'].mean()
m[10] = df['Nov'].mean()
m[11] = df['Dec'].mean()

# Array de meses
h = np.linspace(1, 12, 12)

# Executando o ajuste anual
ano(m, h)
