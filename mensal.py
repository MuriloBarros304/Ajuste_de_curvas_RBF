import numpy as np
import matplotlib.pyplot as plt
import statistics as st
import pandas as pd

# Função gaussiana
def rbf(x, c, s):
    return np.exp(-1 / (2 * s**2) * (x - c)**2)

# Cálculo dos centros
def centro(n):
    return np.linspace(6, 21, n)

# Cálculo do desvio padrão
def desvio(x):
    return st.pstdev(x)

# Ajuste de curvas RBF
def ac(w, v, t, n):
    s = 0
    for j in range(n):
        s += rbf(v[t], centro(n)[j], desvio(v)) * w[j]
    return s

# Função para calcular e plotar o ajuste de curvas de cada mês
def mes(m, h, n=5):
    # Preenchendo o vetor de funções (phi)
    phi = np.zeros((len(h), n))  # Use len(h) em vez de 24 para maior flexibilidade
    for i in range(len(h)):
        for j in range(n):
            phi[i][j] = rbf(h[i], centro(n)[j], desvio(h))

    # Matriz de coeficientes (w)
    m1 = np.linalg.inv(phi.T @ phi)
    m2 = phi.T @ m
    w = m1 @ m2

    # Eixos
    x = np.linspace(6, 21, 100)
    y = np.zeros_like(x)

    # Plot
    for i in range(len(x)):
        y[i] = ac(w, x, i, n)
        if y[i] < 0:  # para evitar valores negativos
            y[i] = 0

    #print('w = ', w)
    plt.xlabel('horas (h)')
    plt.ylabel('radiação direta normal (Wh/m²)')
    plt.legend()
    plt.plot(x, y)
    plt.scatter(h, m)
    plt.grid()
    #plt.show()


# Carregando a planilha Excel
df = pd.read_excel('dados-cn.xlsx')

# Horas
h = df['Horas'].tolist()

# Meses
meses = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Plotando para cada mês
i = 0
for mes_nome in meses:
    plt.title(mes_nome)
    mes(df[mes_nome].tolist(), h)
    plt.savefig(meses[i])
    plt.clf()
    i+=1
