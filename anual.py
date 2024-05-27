
"""Ajuste anual"""

import numpy as np
import matplotlib.pyplot as plt
import statistics as st

# Dados (sendo cada termo a média aritmética de potências de cada mês)
a = [318, 313.53333333, 326.93333333, 318, 297.06666667, 265.8, 272.4, 337.73333333, 375.93333333, 413.13333333, 419.13333333, 356.53333333]

# Criando as matrizes
h = np.linspace(1, 12, 12)
n = 6
phi = np.zeros((12, n))
f = np.zeros((12, 1))
w = np.zeros((n, 1))

# Função gaussiana
def rbf(x, c, s):
    return np.exp(-1 / (2 * s**2) * (x - c)**2)

# Cálculo dos centros
def centro(n):
    return np.linspace(1,12,n)

# Cálculo do desvio padrão
def desvio(x):
    return st.pstdev(x)

# Preenchendo o vetor de funções (phi)
for i in range(12):
    for j in range(n):
        phi[i][j] = rbf(h[i], centro(n)[j], desvio(h))

# Ajuste de curvas RBF
def ac(w, phi, v, t):
    s = 0
    for j in range(n):
        s += rbf(v[t], centro(n)[j], desvio(v)) * w[j]
    return s
def mes(m):
  #for i in range(12):
    #f[i] = m[i]

# Matriz de coeficientes (w)
  m1 = np.linalg.inv((phi).T @ phi)
  m2 = (phi).T @ m
  w = (m1 @ m2)

# Eixos
  x = np.linspace(1, 12, 100)
  y = np.linspace(1, 12, 100)

# Plot
  for i in range(len(x)):
    y[i] = ac(w, phi, x, i)
    if(y[i]<0):
      y[i]=0
  print('w = ', w)
  plt.plot(x, y)
  plt.scatter(h, m)
  plt.grid()
  plt.show()
plt.title("Anual")
mes(a)