import numpy as np

# Carregue seus parâmetros salvos:
params = np.load('estimates.npz')
a     = params['a']      # shape (M,)
b     = params['b']      # shape (M,)
c     = params['c']      # shape (M,)
theta = params['theta']  # shape (N,)

def compute_P(a, b, c, theta):
    """
    Retorna matriz P de probabilidades de acerto:
      P[i,j] = c[j] + (1 - c[j]) * sigmoid( a[j] * (theta[i] - b[j]) )
    Saída: array de shape (N, M)
    """
    # Expande dimensões
    theta = theta[:, None]    # (N,1)
    a     = a[None, :]        # (1,M)
    b     = b[None, :]        # (1,M)
    c     = c[None, :]        # (1,M)

    # sigmoid(a*(θ - b))
    logistic = 1.0 / (1.0 + np.exp(-a * (theta - b)))
    P = c + (1.0 - c) * logistic
    return P

# calcula P_ij para todos os alunos e itens
P = compute_P(a, b, c, theta)  # shape (N_alunos, M_itens)

# “Nota esperada” de cada aluno = soma das probabilidades de acerto
expected_scores = P.sum(axis=1)      # shape (N,)

# expected_scores: array shape (N,), soma das probabilidades por aluno
# M: número de itens (colunas)
N, M = P.shape

# Método 1: direto pela soma normalizada
score_1000 = expected_scores / M * 1000

# (Opcional) se quiser inteiro
score_1000_int = np.round(score_1000).astype(int)

# Exemplo de saída para os 5 primeiros alunos:
for i in range(5):
    print(f'Aluno {i:3d}: Escore esperado = {expected_scores[i]:.2f}, '
          f'Nota = {score_1000_int[i]:4d}')
