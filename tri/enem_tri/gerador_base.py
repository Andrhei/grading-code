import numpy as np
import pandas as pd

# Configuração de parâmetros de simulação
np.random.seed(123)
n_students = 50000    # Número de alunos simulados
n_items = 90         # Número de itens (por exemplo, uma área do ENEM)

# Habilidades dos alunos
thetas = np.random.normal(loc=0, scale=1, size=n_students)

# Parâmetros dos itens (valores fictícios, mas realistas)
a_params = np.random.lognormal(mean=0, sigma=0.2, size=n_items)  # discriminação
b_params = np.random.normal(loc=0, scale=1, size=n_items)        # dificuldade
c_params = np.random.beta(a=5, b=15, size=n_items)               # chance de chute

# Função 3PL
def p_3pl(theta, a, b, c):
    return c + (1 - c) / (1 + np.exp(-a * (theta - b)))

# Geração da matriz de respostas (0/1)
response_matrix = np.zeros((n_students, n_items), dtype=int)
for i in range(n_items):
    probs = p_3pl(thetas, a_params[i], b_params[i], c_params[i])
    response_matrix[:, i] = np.random.binomial(1, probs)

# DataFrame de respostas
item_cols = [f'Item_{i+1}' for i in range(n_items)]
responses_df = pd.DataFrame(response_matrix, columns=item_cols)

# Salvar em CSV para calibração posterior
responses_df.to_csv('respostas_simuladas.csv', index=False)
